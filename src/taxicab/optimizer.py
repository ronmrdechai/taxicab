from __future__ import annotations

import math
import heapq
import importlib
import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.optimize import minimize
from tqdm.auto import tqdm

from .data import Holding, PricePoint, sector_targets
from .metrics import (
    FREQUENCIES,
    beta_to_benchmark,
    daily_returns,
    observations_overlap,
    period_end_points,
    simulated_realized_loss_rate,
    simulated_tax_alpha,
)


SELECTION_METHODS = {"optimized", "random-weighted", "greedy", "beam", "miqp"}
WEIGHT_METHODS = {"slsqp", "index-normalized"}
REPLACEMENT_METHODS = {"ranked", "random"}


@dataclass(frozen=True)
class Candidate:
    ticker: str
    index_weight: float
    sector: str
    beta: float
    tax_alpha: float
    simulated_tax_alpha: float = 0.0
    gross_harvestable_loss_rate: float = 0.0
    observations: int = 0
    returns: Optional[Dict[date, float]] = None
    industry: str = "Unknown"


@dataclass(frozen=True)
class TrackingModel:
    covariance_matrix: np.ndarray
    asset_benchmark_covariance: np.ndarray
    benchmark_variance: float
    observations: int
    annualization: float = 252.0


def save_tracking_model_artifact(
    model: TrackingModel,
    tickers: Sequence[str],
    path: str | Path,
) -> None:
    """Persist a tracking model in a compact NumPy-native artifact."""
    target = Path(path)
    if target.parent != Path("."):
        target.parent.mkdir(parents=True, exist_ok=True)
    if model.covariance_matrix.shape != (len(tickers), len(tickers)):
        raise ValueError("ticker count must match covariance matrix dimensions")
    if model.asset_benchmark_covariance.shape != (len(tickers),):
        raise ValueError("ticker count must match benchmark covariance vector")
    np.savez_compressed(
        target,
        schema_version=np.asarray(1, dtype=np.int64),
        tickers=np.asarray(list(tickers), dtype=np.str_),
        covariance_matrix=np.asarray(model.covariance_matrix, dtype=float),
        asset_benchmark_covariance=np.asarray(model.asset_benchmark_covariance, dtype=float),
        benchmark_variance=np.asarray(model.benchmark_variance, dtype=float),
        observations=np.asarray(model.observations, dtype=np.int64),
        annualization=np.asarray(model.annualization, dtype=float),
    )


def load_tracking_model_artifact(path: str | Path) -> Tuple[List[str], TrackingModel]:
    """Load a tracking model saved by save_tracking_model_artifact."""
    with np.load(Path(path), allow_pickle=False) as payload:
        schema_version = int(payload["schema_version"].item())
        if schema_version != 1:
            raise ValueError(f"unsupported tracking model artifact version: {schema_version}")
        tickers = [str(item) for item in payload["tickers"].tolist()]
        model = TrackingModel(
            covariance_matrix=np.asarray(payload["covariance_matrix"], dtype=float),
            asset_benchmark_covariance=np.asarray(payload["asset_benchmark_covariance"], dtype=float),
            benchmark_variance=float(payload["benchmark_variance"].item()),
            observations=int(payload["observations"].item()),
            annualization=float(payload["annualization"].item()),
        )
    if model.covariance_matrix.shape != (len(tickers), len(tickers)):
        raise ValueError("tracking model covariance matrix does not match ticker count")
    if model.asset_benchmark_covariance.shape != (len(tickers),):
        raise ValueError("tracking model benchmark covariance vector does not match ticker count")
    return tickers, model


@dataclass(frozen=True)
class ReplayConstraintConfig:
    tracking_error_limit: float
    path_tracking_error_limit: float
    sector_band: float = 0.02
    sector_abs_error_limit: float = 0.05
    beta_target: float = 1.0
    beta_band: float = 0.02
    active_share_warning: float = 0.35
    active_share_limit: float = 0.35
    cash_weight_limit: float = 0.0001
    enforce_min_names: int = 50
    max_weight_limit: Optional[float] = None


class _TrackingArrayCache:
    def __init__(
        self,
        candidates: Sequence[Candidate],
        benchmark_returns: Mapping[date, float],
    ) -> None:
        self.ticker_to_index = {candidate.ticker: idx for idx, candidate in enumerate(candidates)}
        date_set = set(benchmark_returns)
        for candidate in candidates:
            if candidate.returns:
                date_set.update(candidate.returns)
        self.dates = sorted(date_set)
        date_index = {day: idx for idx, day in enumerate(self.dates)}

        self.benchmark = np.full(len(self.dates), np.nan, dtype=float)
        for day, value in benchmark_returns.items():
            idx = date_index.get(day)
            if idx is not None:
                self.benchmark[idx] = float(value)

        self.asset_returns = np.full((len(candidates), len(self.dates)), np.nan, dtype=float)
        for row, candidate in enumerate(candidates):
            if not candidate.returns:
                continue
            for day, value in candidate.returns.items():
                idx = date_index.get(day)
                if idx is not None:
                    self.asset_returns[row, idx] = float(value)

    def active_variance(
        self,
        candidates: Sequence[Candidate],
        weights: Sequence[float],
    ) -> Optional[Tuple[float, int]]:
        if len(candidates) == 0:
            return None
        indices = []
        for candidate in candidates:
            idx = self.ticker_to_index.get(candidate.ticker)
            if idx is None:
                return None
            indices.append(idx)
        asset_rows = self.asset_returns[np.asarray(indices, dtype=int)]
        valid = np.isfinite(self.benchmark) & np.all(np.isfinite(asset_rows), axis=0)
        observations = int(np.count_nonzero(valid))
        if observations < 2:
            return None
        weight_array = np.asarray(weights, dtype=float)
        portfolio_returns = weight_array @ asset_rows[:, valid]
        active_returns = portfolio_returns - self.benchmark[valid]
        return float(np.var(active_returns, ddof=1)), observations

    def tracking_error(
        self,
        candidates: Sequence[Candidate],
        weights: Sequence[float],
        annualization: float = 252.0,
    ) -> Optional[Tuple[float, int]]:
        result = self.active_variance(candidates, weights)
        if result is None:
            return None
        active_variance, observations = result
        return math.sqrt(max(active_variance, 0.0) * annualization), observations


class _ReturnCorrelationCache:
    def __init__(self, candidates: Sequence[Candidate]) -> None:
        self.ticker_to_index = {candidate.ticker: idx for idx, candidate in enumerate(candidates)}
        date_set: Set[date] = set()
        for candidate in candidates:
            if candidate.returns:
                date_set.update(candidate.returns)
        self.dates = sorted(date_set)
        date_index = {day: idx for idx, day in enumerate(self.dates)}
        self.asset_returns = np.full((len(candidates), len(self.dates)), np.nan, dtype=float)
        for row, candidate in enumerate(candidates):
            if not candidate.returns:
                continue
            for day, value in candidate.returns.items():
                idx = date_index.get(day)
                if idx is not None:
                    self.asset_returns[row, idx] = float(value)
        self.cache: Dict[Tuple[str, str], float] = {}

    def correlation(self, left: Candidate, right: Candidate) -> float:
        left_ticker, right_ticker = sorted((left.ticker, right.ticker))
        key = (left_ticker, right_ticker)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        left_idx = self.ticker_to_index.get(left.ticker)
        right_idx = self.ticker_to_index.get(right.ticker)
        if left_idx is None or right_idx is None:
            value = _return_correlation(left.returns, right.returns)
            self.cache[key] = value
            return value
        left_values = self.asset_returns[left_idx]
        right_values = self.asset_returns[right_idx]
        valid = np.isfinite(left_values) & np.isfinite(right_values)
        if int(np.count_nonzero(valid)) < 2:
            self.cache[key] = 0.0
            return 0.0
        left_common = left_values[valid]
        right_common = right_values[valid]
        left_centered = left_common - float(np.mean(left_common))
        right_centered = right_common - float(np.mean(right_common))
        left_variance = float(left_centered @ left_centered)
        right_variance = float(right_centered @ right_centered)
        denominator = math.sqrt(left_variance * right_variance)
        if denominator <= 0:
            self.cache[key] = 0.0
            return 0.0
        value = max(min(float(left_centered @ right_centered) / denominator, 1.0), -1.0)
        self.cache[key] = value
        return value


def benchmark_proxy_returns(candidates: Sequence[Candidate]) -> Dict[date, float]:
    weighted_candidates = [
        (candidate, max(0.0, candidate.index_weight))
        for candidate in candidates
        if candidate.returns and candidate.index_weight > 0
    ]
    total_weight = sum(weight for _, weight in weighted_candidates)
    if total_weight <= 0:
        return {}

    common_dates: Optional[Set[date]] = None
    for candidate, _ in weighted_candidates:
        assert candidate.returns is not None
        candidate_dates = set(candidate.returns)
        if common_dates is None:
            common_dates = candidate_dates
        else:
            common_dates.intersection_update(candidate_dates)
    if common_dates is None or len(common_dates) < 2:
        return {}

    proxy: Dict[date, float] = {}
    for day in sorted(common_dates):
        proxy[day] = sum(
            weight / total_weight * candidate.returns[day]
            for candidate, weight in weighted_candidates
            if candidate.returns is not None
        )
    return proxy


@dataclass(frozen=True)
class SimulatedHarvestLot:
    ticker: str
    sector: str
    industry: str
    shares: float
    basis: float
    purchase_day: date


def build_candidates(
    holdings: Sequence[Holding],
    prices: Mapping[str, Sequence[PricePoint]],
    benchmark_ticker: str,
    rebalance_frequency: str,
    min_observations: int = 252,
    tax_metric: str = "simulated",
    tax_rate: float = 0.30,
    harvest_threshold_pct: float = 0.05,
    transaction_cost_bps: float = 5.0,
    replacement_cost_bps: float = 10.0,
    harvest_frequency: Optional[str] = None,
) -> List[Candidate]:
    if tax_metric not in {"simulated", "gross"}:
        raise ValueError("tax_metric must be simulated or gross")
    if benchmark_ticker not in prices:
        raise ValueError(f"benchmark prices missing for {benchmark_ticker}")
    benchmark_returns = daily_returns(prices[benchmark_ticker])
    tax_frequency = harvest_frequency or rebalance_frequency

    candidates: List[Candidate] = []
    for holding in holdings:
        if holding.ticker not in prices:
            continue
        asset_returns = daily_returns(prices[holding.ticker])
        overlap = observations_overlap(asset_returns, benchmark_returns)
        if overlap < min_observations:
            continue
        gross = simulated_realized_loss_rate(
            prices[holding.ticker],
            tax_frequency,
            harvest_threshold_pct=harvest_threshold_pct,
        )
        simulated = simulated_tax_alpha(
            prices[holding.ticker],
            tax_frequency,
            tax_rate=tax_rate,
            harvest_threshold_pct=harvest_threshold_pct,
            transaction_cost_bps=transaction_cost_bps,
            replacement_cost_bps=replacement_cost_bps,
        )
        candidates.append(
            Candidate(
                ticker=holding.ticker,
                index_weight=holding.weight,
                sector=holding.sector or "Unknown",
                industry=holding.industry or "Unknown",
                beta=beta_to_benchmark(asset_returns, benchmark_returns),
                tax_alpha=simulated if tax_metric == "simulated" else gross,
                simulated_tax_alpha=simulated,
                gross_harvestable_loss_rate=gross,
                observations=overlap,
                returns=asset_returns,
            )
        )
    return candidates


def portfolio_metrics(
    candidates: Sequence[Candidate],
    weights: Sequence[float],
    tracking_model: Optional[TrackingModel] = None,
) -> Dict[str, object]:
    beta = sum(weight * candidate.beta for weight, candidate in zip(weights, candidates))
    tax_alpha = sum(weight * candidate.tax_alpha for weight, candidate in zip(weights, candidates))
    simulated = sum(weight * candidate.simulated_tax_alpha for weight, candidate in zip(weights, candidates))
    gross = sum(weight * candidate.gross_harvestable_loss_rate for weight, candidate in zip(weights, candidates))
    sectors: Dict[str, float] = {}
    for weight, candidate in zip(weights, candidates):
        sectors[candidate.sector] = sectors.get(candidate.sector, 0.0) + weight
    metrics: Dict[str, object] = {
        "beta": beta,
        "tax_alpha": tax_alpha,
        "simulated_tax_alpha": simulated,
        "gross_harvestable_loss_rate": gross,
        "estimated_tax_loss_alpha": tax_alpha,
        "sectors": sectors,
        "max_weight": max(weights) if weights else 0.0,
        "effective_number_of_names": 1.0 / sum(weight * weight for weight in weights) if weights else 0.0,
    }
    if tracking_model:
        tracking_error_value = tracking_error(weights, tracking_model)
        metrics["tracking_error"] = tracking_error_value
        metrics["error_percentage"] = tracking_error_value * 100.0
        metrics["tracking_error_observations"] = tracking_model.observations
    return metrics


def project_to_simplex(values: Sequence[float]) -> List[float]:
    if len(values) == 0:
        return []
    array = np.asarray(values, dtype=float)
    ordered = np.sort(array)[::-1]
    cumulative = np.cumsum(ordered)
    indices = np.arange(1, len(ordered) + 1, dtype=float)
    mask = ordered - (cumulative - 1.0) / indices > 0
    rho = int(np.nonzero(mask)[0][-1] + 1) if np.any(mask) else 0
    theta = float((cumulative[rho - 1] - 1.0) / rho) if rho else 0.0
    projected = np.maximum(array - theta, 0.0)
    total = float(np.sum(projected))
    if total <= 0:
        return [1.0 / len(values)] * len(values)
    return (projected / total).tolist()


def project_to_simplex_with_floor(values: Sequence[float], min_weight: float = 0.0) -> List[float]:
    return project_to_bounded_simplex(values, min_weight=min_weight)


def project_to_bounded_simplex(
    values: Sequence[float],
    min_weight: float = 0.0,
    max_weight: Optional[float] = None,
) -> List[float]:
    if len(values) == 0:
        return []
    if min_weight < 0:
        raise ValueError("min_weight must be nonnegative")
    if max_weight is None:
        if min_weight <= 0:
            return project_to_simplex(values)
        max_weight = 1.0
    if max_weight <= 0:
        raise ValueError("max_weight must be positive")
    if min_weight > max_weight:
        raise ValueError("min_weight cannot exceed max_weight")
    if len(values) * min_weight > 1.0 + 1e-12:
        raise ValueError("min_weight is too large for the number of positions")
    if len(values) * max_weight < 1.0 - 1e-12:
        raise ValueError("max_weight is too small for the number of positions")

    lower = min_weight
    upper = max_weight
    return _project_to_bounded_simplex_array(values, lower, upper).tolist()


def _default_scalar_max_weight(count: int) -> float:
    if count <= 0:
        return 1.0
    if count < 50:
        return 1.0
    # A 2.0% cap is realistic for a broad direct-indexing sample. For smaller
    # samples, relax just enough that a fully invested portfolio is feasible.
    return min(1.0, max(0.02, 1.05 / count))


def _benchmark_aware_upper_bounds(
    candidates: Sequence[Candidate],
    min_weight: float,
    max_weight: Optional[float],
) -> np.ndarray:
    if not candidates:
        return np.asarray([], dtype=float)
    default_cap = _default_scalar_max_weight(len(candidates))
    if max_weight is None or len(candidates) < 50:
        scalar_cap = float(max_weight) if max_weight is not None else default_cap
    else:
        scalar_cap = min(float(max_weight), default_cap)
    if scalar_cap <= 0:
        raise ValueError("max_weight must be positive")
    if min_weight > scalar_cap:
        raise ValueError("min_weight cannot exceed max_weight")
    if len(candidates) * min_weight > 1.0 + 1e-12:
        raise ValueError("min_weight is too large for the number of positions")
    if len(candidates) * scalar_cap < 1.0 - 1e-12:
        raise ValueError("max_weight is too small for the number of positions")

    anchor_weights = index_normalized_weights(candidates)
    caps = np.asarray(
        [
            max(min_weight, min(scalar_cap, 2.0 * max(anchor_weight, 0.0)))
            for anchor_weight in anchor_weights
        ],
        dtype=float,
    )
    if float(np.sum(caps)) >= 1.0 - 1e-12:
        return caps

    # The 2x-benchmark rule can be infeasible after sampling because excluded
    # names' weights must be redistributed. Relax toward the scalar cap before
    # giving up on the cap entirely.
    scalar_caps = np.full(len(candidates), scalar_cap, dtype=float)
    if float(np.sum(scalar_caps)) >= 1.0 - 1e-12:
        return scalar_caps
    return np.full(len(candidates), 1.0, dtype=float)


def _project_to_bounded_simplex_array(values: Sequence[float], lower: float, upper: float) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    low = float(np.min(array) - upper)
    high = float(np.max(array) - lower)
    for _ in range(100):
        theta = (low + high) / 2.0
        projected = np.clip(array - theta, lower, upper)
        if float(np.sum(projected)) > 1.0:
            low = theta
        else:
            high = theta
    projected = np.clip(array - high, lower, upper)
    total = float(np.sum(projected))
    if total <= 0:
        return np.full(len(values), 1.0 / len(values), dtype=float)
    adjustment = 1.0 - total
    if abs(adjustment) > 1e-10:
        projected = np.asarray(
            _adjust_bounded_sum(projected.tolist(), adjustment, lower, upper),
            dtype=float,
        )
    return projected


def _project_to_variable_bounded_simplex_array(
    values: Sequence[float] | np.ndarray,
    lower: float,
    uppers: Sequence[float] | np.ndarray,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    upper_array = np.asarray(uppers, dtype=float)
    if len(array) != len(upper_array):
        raise ValueError("values and upper bounds must have the same length")
    if len(array) == 0:
        return array
    if lower < 0:
        raise ValueError("lower bound must be nonnegative")
    if np.any(upper_array < lower):
        raise ValueError("upper bounds must be greater than or equal to lower bound")
    if len(array) * lower > 1.0 + 1e-12:
        raise ValueError("lower bound is too large for the number of positions")
    if float(np.sum(upper_array)) < 1.0 - 1e-12:
        raise ValueError("upper bounds are too small for a fully invested portfolio")

    low = float(np.min(array - upper_array))
    high = float(np.max(array - lower))
    for _ in range(100):
        theta = (low + high) / 2.0
        projected = np.minimum(np.maximum(array - theta, lower), upper_array)
        if float(np.sum(projected)) > 1.0:
            low = theta
        else:
            high = theta
    projected = np.minimum(np.maximum(array - high, lower), upper_array)
    total = float(np.sum(projected))
    adjustment = 1.0 - total
    if abs(adjustment) > 1e-10:
        projected = np.asarray(
            _adjust_variable_bounded_sum(projected.tolist(), adjustment, lower, upper_array),
            dtype=float,
        )
    return projected


def _adjust_bounded_sum(values: Sequence[float], adjustment: float, lower: float, upper: float) -> List[float]:
    adjusted = list(values)
    if adjustment > 0:
        for idx, value in enumerate(adjusted):
            room = upper - value
            delta = min(room, adjustment)
            adjusted[idx] += delta
            adjustment -= delta
            if adjustment <= 1e-12:
                break
    else:
        adjustment = -adjustment
        for idx, value in enumerate(adjusted):
            room = value - lower
            delta = min(room, adjustment)
            adjusted[idx] -= delta
            adjustment -= delta
            if adjustment <= 1e-12:
                break
    return adjusted


def _adjust_variable_bounded_sum(
    values: Sequence[float],
    adjustment: float,
    lower: float,
    uppers: Sequence[float] | np.ndarray,
) -> List[float]:
    adjusted = list(values)
    upper_array = np.asarray(uppers, dtype=float)
    if adjustment > 0:
        for idx, value in enumerate(adjusted):
            room = float(upper_array[idx]) - value
            delta = min(room, adjustment)
            adjusted[idx] += delta
            adjustment -= delta
            if adjustment <= 1e-12:
                break
    else:
        adjustment = -adjustment
        for idx, value in enumerate(adjusted):
            room = value - lower
            delta = min(room, adjustment)
            adjusted[idx] -= delta
            adjustment -= delta
            if adjustment <= 1e-12:
                break
    return adjusted


def prepare_tracking_model(
    candidates: Sequence[Candidate],
    benchmark_returns: Optional[Mapping[date, float]],
    annualization: float = 252.0,
) -> Optional[TrackingModel]:
    if not benchmark_returns:
        return None
    if not candidates:
        return None
    common_dates = set(benchmark_returns)
    for candidate in candidates:
        if not candidate.returns:
            return None
        common_dates.intersection_update(candidate.returns)
    if len(common_dates) < 2:
        return None

    dates = sorted(common_dates)
    benchmark_array = np.asarray([benchmark_returns[day] for day in dates], dtype=float)
    asset_rows = []
    for candidate in candidates:
        returns = candidate.returns
        assert returns is not None
        asset_rows.append([returns[day] for day in dates])
    asset_matrix = np.asarray(asset_rows, dtype=float)

    centered_benchmark = benchmark_array - float(np.mean(benchmark_array))
    centered_assets = asset_matrix - np.mean(asset_matrix, axis=1, keepdims=True)
    denominator = len(dates) - 1
    covariance_matrix = centered_assets @ centered_assets.T / denominator
    asset_benchmark_covariance = centered_assets @ centered_benchmark / denominator
    benchmark_variance = float(centered_benchmark @ centered_benchmark / denominator)

    return TrackingModel(
        covariance_matrix=covariance_matrix,
        asset_benchmark_covariance=asset_benchmark_covariance,
        benchmark_variance=benchmark_variance,
        observations=len(dates),
        annualization=annualization,
    )


def tracking_error(weights: Sequence[float] | np.ndarray, model: TrackingModel) -> float:
    active_variance = _active_variance(weights, model)
    return math.sqrt(max(active_variance, 0.0) * model.annualization)


def _active_variance(weights: Sequence[float] | np.ndarray, model: TrackingModel) -> float:
    weight_array = np.asarray(weights, dtype=float)
    return float(
        weight_array @ model.covariance_matrix @ weight_array
        - 2.0 * (weight_array @ model.asset_benchmark_covariance)
        + model.benchmark_variance
    )


def _active_variance_gradient(weights: Sequence[float] | np.ndarray, model: TrackingModel) -> List[float]:
    weight_array = np.asarray(weights, dtype=float)
    gradient = 2.0 * (model.covariance_matrix @ weight_array - model.asset_benchmark_covariance)
    return gradient.tolist()


def _sector_vector(candidates: Sequence[Candidate], weights: Sequence[float]) -> Dict[str, float]:
    sectors: Dict[str, float] = {}
    for candidate, weight in zip(candidates, weights):
        sectors[candidate.sector] = sectors.get(candidate.sector, 0.0) + weight
    return sectors


def _sector_exposure_arrays(
    candidates: Sequence[Candidate],
    target_sectors: Optional[Mapping[str, float]],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not target_sectors:
        return None, None
    sector_names = sorted(set(target_sectors).union(candidate.sector for candidate in candidates))
    sector_index = {sector: idx for idx, sector in enumerate(sector_names)}
    matrix = np.zeros((len(sector_names), len(candidates)), dtype=float)
    for column, candidate in enumerate(candidates):
        matrix[sector_index[candidate.sector], column] = 1.0
    targets = np.asarray([target_sectors.get(sector, 0.0) for sector in sector_names], dtype=float)
    return matrix, targets


def _progress_range(total: int, label: str, show_progress: bool) -> Iterable[int]:
    values = range(total)
    if show_progress and total > 0:
        return tqdm(values, total=total, desc=label, unit="iter")
    return values


def objective_value(
    candidates: Sequence[Candidate],
    weights: Sequence[float],
    error_margin: float,
    target_tax_alpha: float,
    target_sectors: Optional[Dict[str, float]] = None,
    benchmark_returns: Optional[Dict[date, float]] = None,
    tax_alpha_mode: str = "closest",
    tracking_error_penalty: float = 4.0,
    tax_penalty: float = 0.20,
    sector_penalty: float = 4.0,
    concentration_penalty: float = 0.05,
    index_anchor_penalty: float = 0.25,
    tracking_cache: Optional[_TrackingArrayCache] = None,
) -> float:
    if tax_alpha_mode not in {"closest", "at-least"}:
        raise ValueError("tax_alpha_mode must be closest or at-least")
    if error_margin <= 0:
        raise ValueError("error_margin must be positive")
    tracking_error_value: float
    if tracking_cache is not None:
        cached_tracking = tracking_cache.tracking_error(candidates, weights)
        if cached_tracking is None:
            raise ValueError("benchmark returns and candidate returns are required to optimize error margin")
        tracking_error_value, _ = cached_tracking
        tax_alpha = sum(weight * candidate.tax_alpha for weight, candidate in zip(weights, candidates))
        sectors = _sector_vector(candidates, weights)
    else:
        tracking_model = prepare_tracking_model(candidates, benchmark_returns)
        if tracking_model is None:
            raise ValueError("benchmark returns and candidate returns are required to optimize error margin")
        tracking_error_value = tracking_error(weights, tracking_model)
        tax_alpha = sum(weight * candidate.tax_alpha for weight, candidate in zip(weights, candidates))
        sectors = _sector_vector(candidates, weights)
    tracking_error_scale = max(error_margin, 0.005)
    tax_scale = max(abs(target_tax_alpha), 0.005)
    tracking_error_residual = tracking_error_value / tracking_error_scale
    tax_delta = tax_alpha - target_tax_alpha
    if tax_alpha_mode == "at-least":
        tax_delta = min(tax_delta, 0.0)
    tax_residual = tax_delta / tax_scale
    value = tracking_error_penalty * tracking_error_residual**2 + tax_penalty * tax_residual**2
    if target_sectors:
        all_sectors = set(target_sectors).union(sectors)
        value += sector_penalty * sum(
            (float(sectors.get(sector, 0.0)) - target_sectors.get(sector, 0.0)) ** 2
            for sector in all_sectors
        )
    value += concentration_penalty * sum(weight * weight for weight in weights)
    anchor_weights = index_normalized_weights(candidates)
    value += index_anchor_penalty * sum(
        (weight - anchor_weight) ** 2 for weight, anchor_weight in zip(weights, anchor_weights)
    )
    return value


def index_normalized_weights(candidates: Sequence[Candidate]) -> List[float]:
    total = sum(max(0.0, candidate.index_weight) for candidate in candidates)
    if total <= 0:
        return [1.0 / len(candidates)] * len(candidates)
    return [max(0.0, candidate.index_weight) / total for candidate in candidates]


def index_weighted_weights(
    candidates: Sequence[Candidate],
    min_weight: float = 0.0,
    max_weight: Optional[float] = None,
) -> List[float]:
    upper_bounds = _benchmark_aware_upper_bounds(candidates, min_weight, max_weight)
    return _project_to_variable_bounded_simplex_array(
        index_normalized_weights(candidates),
        lower=min_weight,
        uppers=upper_bounds,
    ).tolist()


def candidate_score(
    candidate: Candidate,
    error_margin: float,
    target_tax_alpha: float,
    benchmark_returns: Dict[date, float],
    tax_alpha_mode: str = "closest",
    tracking_cache: Optional[_TrackingArrayCache] = None,
) -> float:
    if tax_alpha_mode not in {"closest", "at-least"}:
        raise ValueError("tax_alpha_mode must be closest or at-least")
    if error_margin <= 0:
        raise ValueError("error_margin must be positive")
    error_scale = max(error_margin, 0.005)
    if tracking_cache is not None:
        cached_tracking = tracking_cache.tracking_error([candidate], [1.0])
        error_score = (
            (cached_tracking[0] / error_scale) ** 2
            if cached_tracking is not None
            else float("inf")
        )
    else:
        tracking_model = prepare_tracking_model([candidate], benchmark_returns)
        if tracking_model is None:
            error_score = float("inf")
        else:
            error_score = (tracking_error([1.0], tracking_model) / error_scale) ** 2
    tax_scale = max(abs(target_tax_alpha), 0.005)
    tax_delta = candidate.tax_alpha - target_tax_alpha
    if tax_alpha_mode == "at-least":
        tax_delta = min(tax_delta, 0.0)
    tax_score = (tax_delta / tax_scale) ** 2
    weight_bonus = math.sqrt(max(candidate.index_weight, 0.0))
    return error_score + tax_score - 0.05 * weight_bonus


def _weighted_sample_without_replacement(
    candidates: Sequence[Candidate],
    sample_size: int,
    rng: np.random.Generator,
) -> List[Candidate]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if len(candidates) < sample_size:
        raise ValueError(f"only {len(candidates)} candidates available for sample size {sample_size}")
    weights = np.asarray([max(0.0, candidate.index_weight) for candidate in candidates], dtype=float)
    total_weight = float(np.sum(weights))
    probabilities = weights / total_weight if total_weight > 0 else None
    indices = rng.choice(len(candidates), size=sample_size, replace=False, p=probabilities)
    return [candidates[int(index)] for index in indices]


def random_weighted_selection(
    candidates: Sequence[Candidate],
    sample_size: int,
    random_seed: int = 7,
    match_sectors: bool = False,
    target_sectors: Optional[Dict[str, float]] = None,
) -> List[Candidate]:
    rng = np.random.default_rng(random_seed)
    if not match_sectors or not target_sectors:
        return _weighted_sample_without_replacement(candidates, sample_size, rng)

    by_sector: Dict[str, List[Candidate]] = {}
    for candidate in candidates:
        by_sector.setdefault(candidate.sector, []).append(candidate)

    selected: List[Candidate] = []
    selected_tickers: Set[str] = set()
    quotas = _sector_quotas(target_sectors, sample_size)
    for sector, quota in sorted(quotas.items(), key=lambda item: target_sectors.get(item[0], 0.0), reverse=True):
        if quota <= 0:
            continue
        pool = by_sector.get(sector, [])
        if not pool:
            continue
        count = min(quota, len(pool))
        for candidate in _weighted_sample_without_replacement(pool, count, rng):
            selected.append(candidate)
            selected_tickers.add(candidate.ticker)

    if len(selected) < sample_size:
        remainder = [candidate for candidate in candidates if candidate.ticker not in selected_tickers]
        selected.extend(
            _weighted_sample_without_replacement(
                remainder,
                sample_size - len(selected),
                rng,
            )
        )
    return selected


def _selection_score(
    selection: Sequence[Candidate],
    *,
    sample_size: int,
    error_margin: float,
    target_tax_alpha: float,
    benchmark_returns: Dict[date, float],
    target_sectors: Optional[Dict[str, float]],
    match_sectors: bool,
    tax_alpha_mode: str,
    universe_weight_total: float,
    tracking_cache: _TrackingArrayCache,
) -> float:
    weights = index_normalized_weights(selection)
    value = objective_value(
        selection,
        weights,
        error_margin,
        target_tax_alpha,
        target_sectors if match_sectors and len(selection) == sample_size else None,
        benchmark_returns=benchmark_returns,
        tax_alpha_mode=tax_alpha_mode,
        sector_penalty=10.0 if match_sectors else 0.0,
        tracking_cache=tracking_cache,
    )
    if universe_weight_total > 0:
        coverage = sum(max(0.0, candidate.index_weight) for candidate in selection) / universe_weight_total
        value += 25.0 * (1.0 - coverage) ** 2
    return value


def _sector_counts(selection: Sequence[Candidate]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for candidate in selection:
        counts[candidate.sector] = counts.get(candidate.sector, 0) + 1
    return counts


def beam_selection(
    candidates: Sequence[Candidate],
    sample_size: int,
    error_margin: float,
    target_tax_alpha: float,
    benchmark_returns: Dict[date, float],
    target_sectors: Optional[Dict[str, float]],
    match_sectors: bool,
    tax_alpha_mode: str = "closest",
    beam_width: int = 5,
    tracking_cache: Optional[_TrackingArrayCache] = None,
) -> List[Candidate]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if len(candidates) < sample_size:
        raise ValueError(f"only {len(candidates)} candidates available for sample size {sample_size}")
    if beam_width <= 0:
        raise ValueError("beam_width must be positive")
    if tracking_cache is None:
        tracking_cache = _TrackingArrayCache(candidates, benchmark_returns)

    sorted_candidates = sorted(
        candidates,
        key=lambda candidate: (
            candidate_score(
                candidate,
                error_margin,
                target_tax_alpha,
                benchmark_returns,
                tax_alpha_mode,
                tracking_cache=tracking_cache,
            ),
            -candidate.index_weight,
            candidate.ticker,
        ),
    )
    candidate_by_ticker = {candidate.ticker: candidate for candidate in candidates}
    universe_weight_total = sum(max(0.0, candidate.index_weight) for candidate in candidates)
    quotas = _sector_quotas(target_sectors or {}, sample_size) if match_sectors and target_sectors else {}
    branch_count = max(24, beam_width * 8)
    states: List[Tuple[str, ...]] = [()]

    for _ in range(sample_size):
        proposals: List[Tuple[float, Tuple[str, ...]]] = []
        seen: Set[Tuple[str, ...]] = set()
        for state in states:
            selected_tickers = set(state)
            selected_candidates = [candidate_by_ticker[ticker] for ticker in state]
            counts = _sector_counts(selected_candidates)
            additions: List[Candidate] = []
            for candidate in sorted_candidates:
                if candidate.ticker in selected_tickers:
                    continue
                if quotas and counts.get(candidate.sector, 0) >= quotas.get(candidate.sector, 0):
                    continue
                additions.append(candidate)
                if len(additions) >= branch_count:
                    break
            if not additions and quotas:
                for candidate in sorted_candidates:
                    if candidate.ticker not in selected_tickers:
                        additions.append(candidate)
                        if len(additions) >= branch_count:
                            break
            for candidate in additions:
                proposal = tuple(sorted((*state, candidate.ticker)))
                if proposal in seen:
                    continue
                seen.add(proposal)
                selection = [candidate_by_ticker[ticker] for ticker in proposal]
                score = _selection_score(
                    selection,
                    sample_size=sample_size,
                    error_margin=error_margin,
                    target_tax_alpha=target_tax_alpha,
                    benchmark_returns=benchmark_returns,
                    target_sectors=target_sectors,
                    match_sectors=match_sectors,
                    tax_alpha_mode=tax_alpha_mode,
                    universe_weight_total=universe_weight_total,
                    tracking_cache=tracking_cache,
                )
                proposals.append((score, proposal))
        if not proposals:
            break
        states = [
            proposal
            for _, proposal in heapq.nsmallest(
                beam_width,
                proposals,
                key=lambda item: (item[0], item[1]),
            )
        ]

    if not states or len(states[0]) < sample_size:
        fallback = initial_selection(
            candidates,
            sample_size,
            error_margin,
            target_tax_alpha,
            benchmark_returns,
            match_sectors,
            target_sectors,
            tax_alpha_mode=tax_alpha_mode,
            tracking_cache=tracking_cache,
        )
        return fallback
    return [candidate_by_ticker[ticker] for ticker in states[0]]


def _load_pyscipopt():
    try:
        return importlib.import_module("pyscipopt")
    except ImportError as exc:
        raise RuntimeError(
            "MIQP selection requires the PySCIPOpt Python package. "
            "Install project dependencies or run with PySCIPOpt available."
        ) from exc


def _miqp_upper_bounds(
    candidates: Sequence[Candidate],
    min_weight: float,
    max_weight: Optional[float],
    sample_size: int,
) -> np.ndarray:
    if not candidates:
        return np.asarray([], dtype=float)
    scalar_cap = float(max_weight) if max_weight is not None else _default_scalar_max_weight(sample_size)
    if scalar_cap <= 0:
        raise ValueError("max_weight must be positive")
    if min_weight > scalar_cap:
        raise ValueError("min_weight cannot exceed max_weight")
    if sample_size * min_weight > 1.0 + 1e-12:
        raise ValueError("min_weight is too large for the sample size")
    if sample_size * scalar_cap < 1.0 - 1e-12:
        raise ValueError("max_weight is too small for the sample size")
    return np.full(len(candidates), scalar_cap, dtype=float)


def _miqp_warm_start_selection(
    candidates: Sequence[Candidate],
    sample_size: int,
    error_margin: float,
    target_tax_alpha: float,
    benchmark_returns: Dict[date, float],
    target_sectors: Optional[Dict[str, float]],
    match_sectors: bool,
    tax_alpha_mode: str,
) -> List[Candidate]:
    return initial_selection(
        candidates,
        sample_size,
        error_margin,
        target_tax_alpha,
        benchmark_returns,
        match_sectors,
        target_sectors,
        tax_alpha_mode=tax_alpha_mode,
        index_weight_priority=True,
    )


def _add_miqp_warm_start(
    model,
    selected_vars: Sequence[object],
    weight_vars: Sequence[object],
    candidates: Sequence[Candidate],
    sample_size: int,
    min_weight: float,
    upper_bounds: np.ndarray,
    error_margin: float,
    target_tax_alpha: float,
    benchmark_returns: Dict[date, float],
    target_sectors: Optional[Dict[str, float]],
    match_sectors: bool,
    tax_alpha_mode: str,
) -> bool:
    warm_selection = _miqp_warm_start_selection(
        candidates,
        sample_size,
        error_margin,
        target_tax_alpha,
        benchmark_returns,
        target_sectors,
        match_sectors,
        tax_alpha_mode,
    )
    warm_tickers = {candidate.ticker for candidate in warm_selection}
    warm_weights = _project_to_variable_bounded_simplex_array(
        index_normalized_weights(warm_selection),
        lower=min_weight,
        uppers=np.asarray(
            [
                float(upper_bounds[idx])
                for idx, candidate in enumerate(candidates)
                if candidate.ticker in warm_tickers
            ],
            dtype=float,
        ),
    )
    warm_weight_by_ticker = {
        candidate.ticker: float(weight)
        for candidate, weight in zip(warm_selection, warm_weights)
    }
    solution = model.createSol()
    for idx, candidate in enumerate(candidates):
        selected_value = 1.0 if candidate.ticker in warm_tickers else 0.0
        weight_value = warm_weight_by_ticker.get(candidate.ticker, 0.0)
        model.setSolVal(solution, selected_vars[idx], selected_value)
        model.setSolVal(solution, weight_vars[idx], weight_value)
    return bool(model.addSol(solution))


def miqp_selection(
    candidates: Sequence[Candidate],
    sample_size: int,
    error_margin: float,
    target_tax_alpha: float,
    benchmark_returns: Dict[date, float],
    target_sectors: Optional[Dict[str, float]],
    match_sectors: bool,
    tax_alpha_mode: str = "closest",
    min_weight: float = 0.0,
    max_weight: Optional[float] = None,
    tracking_error_penalty: float = 6.0,
    tax_penalty: float = 0.20,
    sector_penalty: float = 12.0,
    concentration_penalty: float = 0.05,
    index_anchor_penalty: float = 25.0,
    beta_target: float = 1.0,
    beta_band: float = 0.02,
    beta_penalty: float = 4.0,
    miqp_time_limit: Optional[float] = 60.0,
    miqp_gap: Optional[float] = 0.01,
) -> Tuple[List[Candidate], Dict[str, object]]:
    if tax_alpha_mode not in {"closest", "at-least"}:
        raise ValueError("tax_alpha_mode must be closest or at-least")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if len(candidates) < sample_size:
        raise ValueError(f"only {len(candidates)} candidates available for sample size {sample_size}")
    if error_margin <= 0:
        raise ValueError("error_margin must be positive")

    tracking_model = prepare_tracking_model(candidates, benchmark_returns)
    if tracking_model is None:
        raise ValueError("benchmark returns and candidate returns are required to optimize error margin")

    pyscipopt = _load_pyscipopt()
    nonlinear_recipe = importlib.import_module("pyscipopt.recipes.nonlinear")
    model = pyscipopt.Model("taxicab_miqp_selection")
    model.hideOutput()
    if miqp_time_limit is not None and miqp_time_limit > 0:
        model.setParam("limits/time", float(miqp_time_limit))
    if miqp_gap is not None and miqp_gap > 0:
        model.setParam("limits/gap", float(miqp_gap))

    count = len(candidates)
    upper_bounds = _miqp_upper_bounds(candidates, min_weight, max_weight, sample_size)
    selected_vars = [
        model.addVar(vtype="B", name=f"x_{idx}_{candidate.ticker}")
        for idx, candidate in enumerate(candidates)
    ]
    weight_vars = [
        model.addVar(lb=0.0, ub=float(upper_bounds[idx]), vtype="C", name=f"w_{idx}_{candidate.ticker}")
        for idx, candidate in enumerate(candidates)
    ]
    quicksum = pyscipopt.quicksum

    model.addCons(quicksum(selected_vars) == sample_size, name="sample_size")
    model.addCons(quicksum(weight_vars) == 1.0, name="fully_invested")
    for idx in range(count):
        model.addCons(weight_vars[idx] <= float(upper_bounds[idx]) * selected_vars[idx], name=f"upper_link_{idx}")
        if min_weight > 0:
            model.addCons(weight_vars[idx] >= float(min_weight) * selected_vars[idx], name=f"lower_link_{idx}")

    tax_values = [float(candidate.tax_alpha) for candidate in candidates]
    beta_values = [float(candidate.beta) for candidate in candidates]
    anchor_weights = index_normalized_weights(candidates)
    tax_expr = quicksum(tax_values[idx] * weight_vars[idx] for idx in range(count))
    beta_expr = quicksum(beta_values[idx] * weight_vars[idx] for idx in range(count))
    tax_scale = max(abs(target_tax_alpha), 0.005)
    tracking_error_scale = max(error_margin, 0.005) ** 2

    active_variance_expr = quicksum(
        float(tracking_model.covariance_matrix[row, column]) * weight_vars[row] * weight_vars[column]
        for row in range(count)
        for column in range(count)
        if abs(float(tracking_model.covariance_matrix[row, column])) > 1e-16
    )
    active_variance_expr -= 2.0 * quicksum(
        float(tracking_model.asset_benchmark_covariance[idx]) * weight_vars[idx]
        for idx in range(count)
    )

    objective = (
        tracking_error_penalty
        * tracking_model.annualization
        * active_variance_expr
        / tracking_error_scale
    )
    if tax_alpha_mode == "at-least":
        shortfall = model.addVar(lb=0.0, vtype="C", name="tax_shortfall")
        model.addCons(shortfall >= float(target_tax_alpha) - tax_expr, name="tax_shortfall_floor")
        objective += tax_penalty * shortfall * shortfall / (tax_scale**2)
    else:
        tax_delta = tax_expr - float(target_tax_alpha)
        objective += tax_penalty * tax_delta * tax_delta / (tax_scale**2)

    if match_sectors and target_sectors:
        sector_names = sorted(set(target_sectors).union(candidate.sector for candidate in candidates))
        for sector in sector_names:
            sector_expr = quicksum(
                weight_vars[idx]
                for idx, candidate in enumerate(candidates)
                if candidate.sector == sector
            )
            sector_delta = sector_expr - float(target_sectors.get(sector, 0.0))
            objective += sector_penalty * sector_delta * sector_delta

    beta_delta = beta_expr - float(beta_target)
    objective += beta_penalty * beta_delta * beta_delta / (max(beta_band, 0.005) ** 2)
    objective += concentration_penalty * quicksum(weight_vars[idx] * weight_vars[idx] for idx in range(count))
    objective += index_anchor_penalty * quicksum(
        (weight_vars[idx] - float(anchor_weights[idx])) * (weight_vars[idx] - float(anchor_weights[idx]))
        for idx in range(count)
    )
    nonlinear_recipe.set_nonlinear_objective(model, objective, "minimize")
    warm_start_accepted = _add_miqp_warm_start(
        model,
        selected_vars,
        weight_vars,
        candidates,
        sample_size,
        min_weight,
        upper_bounds,
        error_margin,
        target_tax_alpha,
        benchmark_returns,
        target_sectors,
        match_sectors,
        tax_alpha_mode,
    )
    model.optimize()

    solution = model.getBestSol()
    if solution is None:
        raise RuntimeError(f"MIQP solver did not return a feasible solution; status={model.getStatus()}")
    selected = [
        candidate
        for idx, candidate in enumerate(candidates)
        if float(model.getSolVal(solution, selected_vars[idx])) >= 0.5
    ]
    if len(selected) != sample_size:
        raise RuntimeError(
            f"MIQP solver selected {len(selected)} names; expected {sample_size}"
        )
    diagnostics: Dict[str, object] = {
        "selection_solver": "PySCIPOpt",
        "selection_solver_status": str(model.getStatus()),
        "selection_solver_objective": float(model.getObjVal()),
        "selection_solver_gap": float(model.getGap()),
        "selection_warm_start_accepted": warm_start_accepted,
    }
    if miqp_time_limit is not None:
        diagnostics["miqp_time_limit"] = miqp_time_limit
    if miqp_gap is not None:
        diagnostics["miqp_gap"] = miqp_gap
    return selected, diagnostics


def _sector_quotas(targets: Dict[str, float], sample_size: int) -> Dict[str, int]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if not targets:
        return {}
    raw = {sector: weight * sample_size for sector, weight in targets.items()}
    quotas = {sector: int(math.floor(value)) for sector, value in raw.items()}
    remaining = sample_size - sum(quotas.values())
    for sector, _ in sorted(raw.items(), key=lambda item: item[1] - math.floor(item[1]), reverse=True):
        if remaining <= 0:
            break
        quotas[sector] += 1
        remaining -= 1
    positive = [sector for sector, weight in targets.items() if weight > 0]
    if sample_size >= len(positive):
        for sector in positive:
            if quotas.get(sector, 0) == 0:
                donor = max(quotas, key=lambda item: quotas[item])
                if quotas[donor] > 1:
                    quotas[donor] -= 1
                    quotas[sector] = 1
    return quotas


def initial_selection(
    candidates: Sequence[Candidate],
    sample_size: int,
    error_margin: float,
    target_tax_alpha: float,
    benchmark_returns: Dict[date, float],
    match_sectors: bool,
    target_sectors: Optional[Dict[str, float]],
    tax_alpha_mode: str = "closest",
    index_weight_priority: bool = False,
    tracking_cache: Optional[_TrackingArrayCache] = None,
) -> List[Candidate]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if len(candidates) < sample_size:
        raise ValueError(f"only {len(candidates)} candidates available for sample size {sample_size}")

    if index_weight_priority:
        sorted_candidates = sorted(candidates, key=lambda candidate: candidate.index_weight, reverse=True)
    else:
        sorted_candidates = sorted(
            candidates,
            key=lambda candidate: (
                candidate_score(
                    candidate,
                    error_margin,
                    target_tax_alpha,
                    benchmark_returns,
                    tax_alpha_mode,
                    tracking_cache=tracking_cache,
                ),
                -candidate.index_weight,
            ),
        )
    if not match_sectors or not target_sectors:
        return sorted_candidates[:sample_size]

    by_sector: Dict[str, List[Candidate]] = {}
    for candidate in sorted_candidates:
        by_sector.setdefault(candidate.sector, []).append(candidate)

    selected: List[Candidate] = []
    selected_tickers: Set[str] = set()
    quotas = _sector_quotas(target_sectors, sample_size)
    for sector, quota in sorted(quotas.items(), key=lambda item: target_sectors.get(item[0], 0.0), reverse=True):
        for candidate in by_sector.get(sector, [])[:quota]:
            selected.append(candidate)
            selected_tickers.add(candidate.ticker)

    for candidate in sorted_candidates:
        if len(selected) >= sample_size:
            break
        if candidate.ticker not in selected_tickers:
            selected.append(candidate)
            selected_tickers.add(candidate.ticker)
    return selected


def optimize_selection(
    candidates: Sequence[Candidate],
    selected: Sequence[Candidate],
    sample_size: int,
    error_margin: float,
    target_tax_alpha: float,
    benchmark_returns: Dict[date, float],
    target_sectors: Optional[Dict[str, float]],
    match_sectors: bool,
    tax_alpha_mode: str = "closest",
    iterations: int = 1000,
    random_seed: int = 7,
    show_progress: bool = False,
    progress_label: str = "Selection",
    tracking_cache: Optional[_TrackingArrayCache] = None,
) -> List[Candidate]:
    rng = random.Random(random_seed)
    if tracking_cache is None:
        tracking_cache = _TrackingArrayCache(candidates, benchmark_returns)
    selected_by_ticker = {candidate.ticker: candidate for candidate in selected}
    universe_by_ticker = {candidate.ticker: candidate for candidate in candidates}
    universe_weight_total = sum(max(0.0, candidate.index_weight) for candidate in candidates)
    by_sector: Dict[str, List[Candidate]] = {}
    for candidate in candidates:
        by_sector.setdefault(candidate.sector, []).append(candidate)

    def score(selection: Sequence[Candidate]) -> float:
        weights = index_normalized_weights(selection)
        value = objective_value(
            selection,
            weights,
            error_margin,
            target_tax_alpha,
            target_sectors if match_sectors else None,
            benchmark_returns=benchmark_returns,
            tax_alpha_mode=tax_alpha_mode,
            sector_penalty=10.0 if match_sectors else 0.0,
            tracking_cache=tracking_cache,
        )
        if universe_weight_total > 0:
            coverage = sum(max(0.0, candidate.index_weight) for candidate in selection) / universe_weight_total
            value += 25.0 * (1.0 - coverage) ** 2
        return value

    best_selection = list(selected_by_ticker.values())
    best_score = score(best_selection)
    current_selection = list(best_selection)
    current_score = best_score

    if iterations <= 0:
        return best_selection

    for step in _progress_range(iterations, progress_label, show_progress):
        out_candidate = rng.choice(current_selection)
        current_tickers = {candidate.ticker for candidate in current_selection}
        if match_sectors:
            replacement_pool = [
                candidate
                for candidate in by_sector.get(out_candidate.sector, [])
                if candidate.ticker not in current_tickers
            ]
        else:
            replacement_pool = [
                candidate
                for candidate in candidates
                if candidate.ticker not in current_tickers
            ]
        if not replacement_pool:
            continue
        replacement = rng.choice(replacement_pool)
        proposal = [candidate for candidate in current_selection if candidate.ticker != out_candidate.ticker]
        proposal.append(universe_by_ticker[replacement.ticker])
        proposal_score = score(proposal)
        temperature = max(0.005, 0.05 * (1.0 - step / max(iterations, 1)))
        accept = proposal_score < current_score or rng.random() < math.exp(
            min(0.0, (current_score - proposal_score) / temperature)
        )
        if accept:
            current_selection = proposal
            current_score = proposal_score
            if current_score < best_score:
                best_selection = list(current_selection)
                best_score = current_score

    return sorted(best_selection, key=lambda candidate: candidate.ticker)


def optimize_weights(
    candidates: Sequence[Candidate],
    error_margin: float,
    target_tax_alpha: float,
    target_sectors: Optional[Dict[str, float]] = None,
    tax_alpha_mode: str = "closest",
    iterations: int = 2000,
    learning_rate: float = 0.08,
    tax_penalty: float = 0.20,
    sector_penalty: float = 12.0,
    concentration_penalty: float = 0.05,
    min_weight: float = 0.0,
    max_weight: Optional[float] = None,
    benchmark_returns: Optional[Dict[date, float]] = None,
    tracking_error_penalty: float = 6.0,
    index_anchor_penalty: float = 25.0,
    sector_band: Optional[float] = 0.02,
    beta_target: float = 1.0,
    beta_band: float = 0.02,
    beta_penalty: float = 4.0,
    show_progress: bool = False,
    progress_label: str = "Weights",
) -> List[float]:
    if tax_alpha_mode not in {"closest", "at-least"}:
        raise ValueError("tax_alpha_mode must be closest or at-least")
    if error_margin <= 0:
        raise ValueError("error_margin must be positive")
    if not candidates:
        raise ValueError("no candidates supplied")
    upper_bounds = _benchmark_aware_upper_bounds(candidates, min_weight, max_weight)
    anchor_weights_array = _project_to_variable_bounded_simplex_array(
        index_normalized_weights(candidates),
        lower=min_weight,
        uppers=upper_bounds,
    )
    weights_array = anchor_weights_array.copy()
    tax_values = np.asarray([candidate.tax_alpha for candidate in candidates], dtype=float)
    beta_values = np.asarray([candidate.beta for candidate in candidates], dtype=float)
    sector_matrix, target_sector_values = _sector_exposure_arrays(candidates, target_sectors)
    tax_scale = max(abs(target_tax_alpha), 0.005)
    tracking_model = prepare_tracking_model(candidates, benchmark_returns)
    if tracking_model is None:
        raise ValueError("benchmark returns and candidate returns are required to optimize error margin")
    tracking_error_scale = max(error_margin, 0.005) ** 2
    max_iterations = max(0, iterations)

    if max_iterations <= 0:
        return weights_array.tolist()

    def objective(weights: np.ndarray) -> float:
        tax_alpha = float(weights @ tax_values)
        tax_delta = tax_alpha - target_tax_alpha
        if tax_alpha_mode == "at-least":
            tax_delta = min(tax_delta, 0.0)
        value = (
            tracking_error_penalty
            * tracking_model.annualization
            * _active_variance(weights, tracking_model)
            / tracking_error_scale
        )
        value += tax_penalty * (tax_delta / tax_scale) ** 2
        if sector_matrix is not None and target_sector_values is not None:
            sector_residual = sector_matrix @ weights - target_sector_values
            value += sector_penalty * float(sector_residual @ sector_residual)
        beta_delta = float(weights @ beta_values) - beta_target
        value += beta_penalty * (beta_delta / max(beta_band, 0.005)) ** 2
        value += concentration_penalty * float(weights @ weights)
        value += index_anchor_penalty * float((weights - anchor_weights_array) @ (weights - anchor_weights_array))
        return float(value)

    def objective_gradient(weights: np.ndarray) -> np.ndarray:
        tax_alpha = float(weights @ tax_values)
        tax_delta = tax_alpha - target_tax_alpha
        if tax_alpha_mode == "at-least":
            tax_delta = min(tax_delta, 0.0)
        active_gradient = np.asarray(_active_variance_gradient(weights, tracking_model), dtype=float)
        gradient = tracking_error_penalty * tracking_model.annualization * active_gradient / tracking_error_scale
        if tax_delta != 0.0:
            gradient += 2.0 * tax_penalty * tax_delta * tax_values / (tax_scale**2)
        if sector_matrix is not None and target_sector_values is not None:
            sector_residual = sector_matrix @ weights - target_sector_values
            gradient += 2.0 * sector_penalty * (sector_matrix.T @ sector_residual)
        beta_delta = float(weights @ beta_values) - beta_target
        gradient += 2.0 * beta_penalty * beta_delta * beta_values / (max(beta_band, 0.005) ** 2)
        gradient += 2.0 * concentration_penalty * weights
        gradient += 2.0 * index_anchor_penalty * (weights - anchor_weights_array)
        return gradient

    def tracking_constraint(weights: np.ndarray) -> float:
        return error_margin**2 - tracking_model.annualization * _active_variance(weights, tracking_model)

    def tracking_constraint_gradient(weights: np.ndarray) -> np.ndarray:
        active_gradient = np.asarray(_active_variance_gradient(weights, tracking_model), dtype=float)
        return -tracking_model.annualization * active_gradient

    constraints = [
        {
            "type": "eq",
            "fun": lambda weights: float(np.sum(weights) - 1.0),
            "jac": lambda weights: np.ones_like(weights),
        },
        {
            "type": "ineq",
            "fun": tracking_constraint,
            "jac": tracking_constraint_gradient,
        },
    ]
    if sector_matrix is not None and target_sector_values is not None and sector_band is not None:
        sector_limit = max(float(sector_band), 0.0)
        for row_index in range(sector_matrix.shape[0]):
            row = sector_matrix[row_index].copy()
            target_value = float(target_sector_values[row_index])
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda weights, row=row, target_value=target_value: sector_limit
                    - float(row @ weights - target_value),
                    "jac": lambda weights, row=row: -row,
                }
            )
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda weights, row=row, target_value=target_value: sector_limit
                    + float(row @ weights - target_value),
                    "jac": lambda weights, row=row: row,
                }
            )

    progress_bar = tqdm(total=max_iterations, desc=progress_label, unit="iter") if show_progress else None

    def update_progress(_weights: np.ndarray) -> None:
        if progress_bar is not None:
            progress_bar.update(1)

    try:
        result = minimize(
            objective,
            weights_array,
            method="SLSQP",
            jac=objective_gradient,
            bounds=[(min_weight, float(upper)) for upper in upper_bounds],
            constraints=constraints,
            callback=update_progress if progress_bar is not None else None,
            options={"maxiter": max_iterations, "ftol": 1e-12, "disp": False},
        )
    except BaseException:
        if progress_bar is not None:
            progress_bar.close()
        raise
    if progress_bar is not None:
        if result.success and progress_bar.n < max_iterations:
            progress_bar.set_postfix_str(f"converged after {progress_bar.n} iter")
            progress_bar.refresh()
        progress_bar.close()

    if result.x is not None and len(result.x) == len(candidates) and np.all(np.isfinite(result.x)):
        weights_array = np.asarray(result.x, dtype=float)

    weights_array = _project_to_variable_bounded_simplex_array(
        weights_array,
        lower=min_weight,
        uppers=upper_bounds,
    )
    if tracking_error(weights_array, tracking_model) > error_margin + 1e-8:
        repaired = _repair_tracking_error(weights_array, anchor_weights_array, tracking_model, error_margin)
        if tracking_error(repaired, tracking_model) <= error_margin + 1e-8:
            weights_array = np.asarray(repaired, dtype=float)
    return weights_array.tolist()


def _repair_tracking_error(
    weights: Sequence[float] | np.ndarray,
    anchor_weights: Sequence[float] | np.ndarray,
    tracking_model: TrackingModel,
    error_margin: float,
) -> List[float]:
    if tracking_error(weights, tracking_model) <= error_margin:
        return list(weights)
    if tracking_error(anchor_weights, tracking_model) > error_margin:
        return list(weights)
    low = 0.0
    high = 1.0
    best = list(anchor_weights)
    for _ in range(40):
        mix = (low + high) / 2.0
        blended = [
            anchor_weight + mix * (weight - anchor_weight)
            for weight, anchor_weight in zip(weights, anchor_weights)
        ]
        if tracking_error(blended, tracking_model) <= error_margin:
            best = blended
            low = mix
        else:
            high = mix
    return best


def sector_error(sectors: Dict[str, float], targets: Dict[str, float]) -> float:
    return sum(abs(sectors.get(sector, 0.0) - targets.get(sector, 0.0)) for sector in set(sectors).union(targets))


def _object_to_float(value: object) -> Optional[float]:
    if not isinstance(value, (int, float, str)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _add_percent_metric_aliases(metrics: Dict[str, object]) -> None:
    tracking_error_value = _object_to_float(metrics.get("tracking_error"))
    if tracking_error_value is not None:
        metrics["tracking_error_annualized_pct"] = tracking_error_value * 100.0
    active_share_value = _object_to_float(metrics.get("active_share"))
    if active_share_value is not None:
        metrics["active_share_pct"] = active_share_value * 100.0
    max_weight_value = _object_to_float(metrics.get("max_weight"))
    if max_weight_value is not None:
        metrics["max_weight_pct"] = max_weight_value * 100.0
    sector_abs_error_value = _object_to_float(metrics.get("sector_abs_error"))
    if sector_abs_error_value is not None:
        metrics["sector_absolute_error_pct"] = sector_abs_error_value * 100.0
    gross_loss_value = _object_to_float(metrics.get("gross_harvestable_loss_rate"))
    if gross_loss_value is not None:
        metrics["gross_harvestable_loss_rate_pct_per_year"] = gross_loss_value * 100.0
    simulated_tax_alpha_value = _object_to_float(metrics.get("simulated_tax_alpha"))
    if simulated_tax_alpha_value is not None:
        metrics["simulated_after_tax_alpha_pct_per_year"] = simulated_tax_alpha_value * 100.0


def _constraint_warnings(metrics: Mapping[str, object], sample_size: int) -> List[str]:
    warnings: List[str] = []

    def number(key: str) -> Optional[float]:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        return None

    tracking_error_value = number("tracking_error")
    if tracking_error_value is not None:
        if tracking_error_value > 0.02:
            warnings.append("tracking_error_above_2pct_hard_warning")
        elif tracking_error_value > 0.015:
            warnings.append("tracking_error_above_1_5pct_warning")
    active_share_value = number("active_share")
    if active_share_value is not None:
        if active_share_value > 0.50:
            warnings.append("active_share_above_50pct_hard_warning")
        elif active_share_value > 0.35:
            warnings.append("active_share_above_35pct_warning")
    effective_names = number("effective_number_of_names")
    if effective_names is not None and sample_size >= 250 and effective_names <= 100.0:
        warnings.append("effective_names_not_above_100")
    sector_abs_error = number("sector_abs_error")
    if sector_abs_error is not None:
        if sector_abs_error > 0.05:
            warnings.append("sector_absolute_error_above_5pct_hard_warning")
        elif sector_abs_error > 0.02:
            warnings.append("sector_absolute_error_above_2pct_warning")
    max_weight = number("max_weight")
    if max_weight is not None and sample_size >= 50 and max_weight > _default_scalar_max_weight(sample_size) + 1e-8:
        warnings.append("max_weight_above_default_direct_indexing_cap")
    beta = number("beta")
    if beta is not None and abs(beta - 1.0) > 0.02:
        warnings.append("beta_outside_0_98_to_1_02")
    return warnings


def _hard_constraint_violations(
    metrics: Mapping[str, object],
    candidates: Sequence[Candidate],
    weights: Sequence[float],
    sample_size: int,
) -> List[str]:
    if sample_size < 50:
        return []
    violations: List[str] = []

    def number(key: str) -> Optional[float]:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        return None

    beta = number("beta")
    if beta is not None and not 0.98 <= beta <= 1.02:
        violations.append(f"beta={beta:.4f} outside 0.98-1.02")
    tracking_error_value = number("tracking_error")
    if tracking_error_value is not None and tracking_error_value > 0.02:
        violations.append(f"tracking_error={tracking_error_value:.4f} above 0.0200")
    active_share_value = number("active_share")
    if active_share_value is not None and active_share_value > 0.50:
        violations.append(f"active_share={active_share_value:.4f} above 0.5000")
    effective_names = number("effective_number_of_names")
    if sample_size >= 250 and effective_names is not None and effective_names < 100.0:
        violations.append(f"effective_names={effective_names:.1f} below 100")
    sector_abs_error = number("sector_abs_error")
    if sector_abs_error is not None and sector_abs_error > 0.05:
        violations.append(f"sector_abs_error={sector_abs_error:.4f} above 0.0500")
    for candidate, weight in zip(candidates, weights):
        allowed_weight = max(0.02, max(0.0, candidate.index_weight))
        if weight > allowed_weight + 1e-8:
            violations.append(
                f"{candidate.ticker} weight={weight:.4f} above allowed {allowed_weight:.4f}"
            )
            break
    return violations


def _return_correlation(
    left: Optional[Mapping[date, float]],
    right: Optional[Mapping[date, float]],
) -> float:
    if not left or not right:
        return 0.0
    dates = sorted(set(left).intersection(right))
    if len(dates) < 2:
        return 0.0
    left_values = [float(left[day]) for day in dates]
    right_values = [float(right[day]) for day in dates]
    left_mean = sum(left_values) / len(left_values)
    right_mean = sum(right_values) / len(right_values)
    covariance_value = sum((x - left_mean) * (y - right_mean) for x, y in zip(left_values, right_values))
    left_variance = sum((x - left_mean) ** 2 for x in left_values)
    right_variance = sum((y - right_mean) ** 2 for y in right_values)
    denominator = math.sqrt(left_variance * right_variance)
    if denominator <= 0:
        return 0.0
    return max(min(covariance_value / denominator, 1.0), -1.0)


def _replacement_similarity(
    source: Candidate,
    replacement: Candidate,
    correlation_cache: Optional[_ReturnCorrelationCache] = None,
) -> Dict[str, float]:
    same_industry = (
        source.industry != "Unknown"
        and replacement.industry != "Unknown"
        and source.industry == replacement.industry
    )
    correlation = (
        correlation_cache.correlation(source, replacement)
        if correlation_cache is not None
        else _return_correlation(source.returns, replacement.returns)
    )
    beta_delta = abs(source.beta - replacement.beta)
    industry_penalty = 0.0 if same_industry else 1.0
    score = 3.0 * industry_penalty + beta_delta + (1.0 - correlation)
    return {
        "score": score,
        "correlation": correlation,
        "beta_delta": beta_delta,
        "industry_match": 1.0 if same_industry else 0.0,
    }


def replacement_candidates(
    universe: Sequence[Candidate],
    selected: Sequence[Candidate],
    error_margin: float,
    target_tax_alpha: float,
    benchmark_returns: Dict[date, float],
    tax_alpha_mode: str = "closest",
    limit: int = 5,
) -> Dict[str, List[Dict[str, object]]]:
    selected_tickers = {candidate.ticker for candidate in selected}
    replacements: Dict[str, List[Dict[str, object]]] = {}
    tracking_cache = _TrackingArrayCache(universe, benchmark_returns)
    score_cache: Dict[str, float] = {}
    replacement_scores: Dict[Tuple[str, str], Dict[str, float]] = {}
    correlation_cache = _ReturnCorrelationCache(universe)

    def score_candidate(candidate: Candidate) -> float:
        if candidate.ticker not in score_cache:
            score_cache[candidate.ticker] = candidate_score(
                candidate,
                error_margin,
                target_tax_alpha,
                benchmark_returns,
                tax_alpha_mode,
                tracking_cache=tracking_cache,
            )
        return score_cache[candidate.ticker]

    candidate_scores = {candidate.ticker: score_candidate(candidate) for candidate in universe}
    ranked_replacements = _ranked_same_sector_replacements(
        universe,
        candidate_scores,
        replacement_scores,
        correlation_cache,
    )

    for candidate in selected:
        pool = [
            item
            for item in ranked_replacements.get(candidate.ticker, [])
            if item.ticker not in selected_tickers
        ]
        replacements[candidate.ticker] = [
            _replacement_row(candidate, item, correlation_cache)
            for item in pool[:limit]
        ]
    return replacements


def _replacement_row(
    source: Candidate,
    replacement: Candidate,
    correlation_cache: Optional[_ReturnCorrelationCache] = None,
) -> Dict[str, object]:
    similarity = _replacement_similarity(source, replacement, correlation_cache)
    return {
        "ticker": replacement.ticker,
        "sector": replacement.sector,
        "industry": replacement.industry,
        "index_weight": replacement.index_weight,
        "beta": replacement.beta,
        "beta_delta": similarity["beta_delta"],
        "return_correlation": similarity["correlation"],
        "industry_match": bool(similarity["industry_match"]),
        "replacement_score": similarity["score"],
        "tax_alpha": replacement.tax_alpha,
        "simulated_tax_alpha": replacement.simulated_tax_alpha,
        "gross_harvestable_loss_rate": replacement.gross_harvestable_loss_rate,
        "estimated_tax_loss_alpha": replacement.tax_alpha,
    }


def _empty_harvest_simulation(
    reason: str,
    rebalance_frequency: str,
    harvest_frequency: Optional[str] = None,
) -> Dict[str, object]:
    tax_frequency = harvest_frequency or rebalance_frequency
    return {
        "status": "skipped",
        "reason": reason,
        "frequency": tax_frequency,
        "rebalance_frequency": rebalance_frequency,
        "harvest_frequency": tax_frequency,
        "start": None,
        "end": None,
        "years": 0.0,
        "average_portfolio_value": 0.0,
        "ending_portfolio_value": 0.0,
        "total_realized_loss": 0.0,
        "annual_realized_loss": 0.0,
        "portfolio_realized_loss_rate": 0.0,
        "total_tax_benefit": 0.0,
        "total_transaction_cost": 0.0,
        "total_replacement_cost": 0.0,
        "total_net_tax_benefit": 0.0,
        "full_liquidation_after_tax_benefit": 0.0,
        "portfolio_simulated_tax_alpha": 0.0,
        "portfolio_harvest_annualized_return": 0.0,
        "benchmark_annualized_return": 0.0,
        "portfolio_harvest_active_return": 0.0,
        "portfolio_harvest_active_return_pct_per_year": 0.0,
        "portfolio_harvest_tracking_error": 0.0,
        "portfolio_harvest_tracking_error_annualized_pct": 0.0,
        "portfolio_harvest_beta": 0.0,
        "portfolio_harvest_correlation": 0.0,
        "portfolio_harvest_observations": 0,
        "harvest_count": 0,
        "rebalance_count": 0,
        "rebalance_dates": [],
        "skipped_no_replacement": 0,
        "skipped_nonpositive_net_benefit": 0,
        "skipped_constraint_violation": 0,
        "skipped_harvests_by_reason": {},
        "harvest_diagnostics": [],
        "skipped_harvest_events": [],
        "cash_drag": 0.0,
        "immediate_tax_savings_pct_per_year": 0.0,
        "immediate_net_tax_savings_pct_per_year": 0.0,
        "realized_loss_rate_pct_per_year": 0.0,
        "simulated_after_tax_alpha_pct_per_year": 0.0,
        "full_liquidation_after_tax_alpha_pct_per_year": 0.0,
        "terminal_after_tax_wealth_difference_pct": 0.0,
        "period_realized_losses": [],
        "sample_events": [],
    }


def _series_returns(values: Sequence[float]) -> List[float]:
    returns: List[float] = []
    for previous, current in zip(values, values[1:]):
        if previous > 0 and current > 0:
            returns.append(current / previous - 1.0)
    return returns


def _sample_variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    average = sum(values) / len(values)
    return sum((value - average) ** 2 for value in values) / (len(values) - 1)


def _sample_covariance(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("series lengths must match")
    if len(left) < 2:
        return 0.0
    left_average = sum(left) / len(left)
    right_average = sum(right) / len(right)
    return sum(
        (left_value - left_average) * (right_value - right_average)
        for left_value, right_value in zip(left, right)
    ) / (len(left) - 1)


def _annualized_return(values: Sequence[float], years: float) -> float:
    if len(values) < 2 or values[0] <= 0 or values[-1] <= 0 or years <= 0:
        return 0.0
    return (values[-1] / values[0]) ** (1.0 / years) - 1.0


def _benchmark_weight_map(universe: Sequence[Candidate]) -> Dict[str, float]:
    total = sum(max(0.0, candidate.index_weight) for candidate in universe)
    if total <= 0:
        return {}
    return {
        candidate.ticker: max(0.0, candidate.index_weight) / total
        for candidate in universe
    }


def _sector_targets_from_candidates(universe: Sequence[Candidate]) -> Dict[str, float]:
    total = sum(max(0.0, candidate.index_weight) for candidate in universe)
    if total <= 0:
        return {}
    targets: Dict[str, float] = {}
    for candidate in universe:
        targets[candidate.sector] = targets.get(candidate.sector, 0.0) + max(0.0, candidate.index_weight) / total
    return targets


def _active_share_from_weight_maps(
    portfolio_weights: Mapping[str, float],
    benchmark_weights: Mapping[str, float],
) -> float:
    tickers = set(portfolio_weights).union(benchmark_weights)
    return 0.5 * sum(abs(portfolio_weights.get(ticker, 0.0) - benchmark_weights.get(ticker, 0.0)) for ticker in tickers)


def _values_from_lots(
    lots: Sequence[SimulatedHarvestLot],
    day_prices: Mapping[str, float],
) -> Tuple[Dict[str, float], float]:
    values: Dict[str, float] = {}
    total_value = 0.0
    for lot in lots:
        price = day_prices.get(lot.ticker, 0.0)
        if price <= 0:
            continue
        value = lot.shares * price
        if value <= 0:
            continue
        values[lot.ticker] = values.get(lot.ticker, 0.0) + value
        total_value += value
    if total_value <= 0:
        return {}, 0.0
    return values, total_value


def _weights_from_values(
    values: Mapping[str, float],
    total_value: float,
) -> Dict[str, float]:
    if total_value <= 0:
        return {}
    return {
        ticker: value / total_value
        for ticker, value in values.items()
        if value > 0
    }


def _replay_snapshot_from_values(
    values_by_ticker: Mapping[str, float],
    total_value: float,
    candidate_by_ticker: Mapping[str, Candidate],
    benchmark_weights: Mapping[str, float],
    target_sectors: Mapping[str, float],
    tracking_cache: _TrackingArrayCache,
    *,
    include_tracking: bool = True,
) -> Dict[str, object]:
    weight_by_ticker = _weights_from_values(values_by_ticker, total_value)
    candidates: List[Candidate] = []
    weights: List[float] = []
    missing_weight = 0.0
    for ticker, weight in sorted(weight_by_ticker.items()):
        candidate = candidate_by_ticker.get(ticker)
        if candidate is None:
            missing_weight += weight
            continue
        candidates.append(candidate)
        weights.append(weight)

    sectors = _sector_vector(candidates, weights) if candidates else {}
    cached_tracking = (
        tracking_cache.tracking_error(candidates, weights)
        if candidates and include_tracking
        else None
    )
    tracking_error_value = cached_tracking[0] if cached_tracking is not None else 0.0
    beta = sum(weight * candidate.beta for candidate, weight in zip(candidates, weights))
    active_share_value = (
        _active_share_from_weight_maps(weight_by_ticker, benchmark_weights)
        if benchmark_weights
        else 0.0
    )
    effective_names = 1.0 / sum(weight * weight for weight in weight_by_ticker.values()) if weight_by_ticker else 0.0
    max_weight = max(weight_by_ticker.values()) if weight_by_ticker else 0.0
    sector_abs_error = sector_error(dict(sectors), dict(target_sectors)) if target_sectors else 0.0
    return {
        "portfolio_value": total_value,
        "cash_weight": 0.0,
        "missing_candidate_weight": missing_weight,
        "tracking_error": tracking_error_value,
        "tracking_error_annualized_pct": tracking_error_value * 100.0,
        "beta": beta,
        "active_share": active_share_value,
        "active_share_pct": active_share_value * 100.0,
        "max_weight": max_weight,
        "max_weight_pct": max_weight * 100.0,
        "effective_names": effective_names,
        "sector_abs_error": sector_abs_error,
        "sector_absolute_error_pct": sector_abs_error * 100.0,
        "position_count": len(weight_by_ticker),
        "weights": weight_by_ticker,
        "sectors": sectors,
    }


def _replay_snapshot(
    lots: Sequence[SimulatedHarvestLot],
    day_prices: Mapping[str, float],
    candidate_by_ticker: Mapping[str, Candidate],
    benchmark_weights: Mapping[str, float],
    target_sectors: Mapping[str, float],
    tracking_cache: _TrackingArrayCache,
) -> Dict[str, object]:
    values_by_ticker, total_value = _values_from_lots(lots, day_prices)
    return _replay_snapshot_from_values(
        values_by_ticker,
        total_value,
        candidate_by_ticker,
        benchmark_weights,
        target_sectors,
        tracking_cache,
    )


def _snapshot_number(snapshot: Mapping[str, object], key: str) -> float:
    value = snapshot.get(key)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _replay_violation_score(snapshot: Mapping[str, object], constraints: ReplayConstraintConfig) -> float:
    score = 0.0
    tracking_error_value = _snapshot_number(snapshot, "tracking_error")
    score += max(0.0, tracking_error_value - constraints.path_tracking_error_limit) / max(
        constraints.path_tracking_error_limit,
        0.005,
    )
    beta = _snapshot_number(snapshot, "beta")
    score += max(0.0, abs(beta - constraints.beta_target) - constraints.beta_band) / max(
        constraints.beta_band,
        0.005,
    )
    sector_abs_error = _snapshot_number(snapshot, "sector_abs_error")
    score += max(0.0, sector_abs_error - constraints.sector_abs_error_limit) / max(
        constraints.sector_abs_error_limit,
        0.005,
    )
    active_share_value = _snapshot_number(snapshot, "active_share")
    score += max(0.0, active_share_value - constraints.active_share_limit) / max(
        constraints.active_share_limit,
        0.01,
    )
    max_weight = _snapshot_number(snapshot, "max_weight")
    if constraints.max_weight_limit is not None:
        score += max(0.0, max_weight - constraints.max_weight_limit) / max(
            constraints.max_weight_limit,
            0.005,
        )
    cash_weight = _snapshot_number(snapshot, "cash_weight")
    score += max(0.0, cash_weight - constraints.cash_weight_limit) / max(
        constraints.cash_weight_limit,
        1e-6,
    )
    return score


def _replay_violations(snapshot: Mapping[str, object], constraints: ReplayConstraintConfig) -> List[str]:
    violations: List[str] = []
    if _snapshot_number(snapshot, "tracking_error") > constraints.path_tracking_error_limit + 1e-12:
        violations.append("tracking_error")
    if abs(_snapshot_number(snapshot, "beta") - constraints.beta_target) > constraints.beta_band + 1e-12:
        violations.append("beta")
    if _snapshot_number(snapshot, "sector_abs_error") > constraints.sector_abs_error_limit + 1e-12:
        violations.append("sector_abs_error")
    if _snapshot_number(snapshot, "active_share") > constraints.active_share_limit + 1e-12:
        violations.append("active_share")
    if (
        constraints.max_weight_limit is not None
        and _snapshot_number(snapshot, "max_weight") > constraints.max_weight_limit + 1e-12
    ):
        violations.append("max_weight")
    if _snapshot_number(snapshot, "cash_weight") > constraints.cash_weight_limit + 1e-12:
        violations.append("cash")
    return violations


def _replay_constraints_allow(
    before: Mapping[str, object],
    after: Mapping[str, object],
    constraints: ReplayConstraintConfig,
    selected_count: int,
) -> Tuple[bool, List[str]]:
    if selected_count < constraints.enforce_min_names:
        return True, []
    after_violations = _replay_violations(after, constraints)
    before_tracking_error = _snapshot_number(before, "tracking_error")
    after_tracking_error = _snapshot_number(after, "tracking_error")
    if after_tracking_error > before_tracking_error + 1e-6:
        after_violations.append("tracking_error_increase")
    if not after_violations:
        return True, []
    return False, after_violations


def _replacement_plans(pool: Sequence[Candidate], replacement_count: int) -> List[List[Candidate]]:
    if not pool or replacement_count <= 0:
        return []
    count = min(replacement_count, len(pool))
    if count == 1:
        return [[candidate] for candidate in pool[:8]]
    if count == 2:
        limited = list(pool[:6])
        plans: List[List[Candidate]] = []
        if len(limited) >= 2:
            plans.append([limited[0], limited[1]])
        for candidate in limited[2:]:
            plans.append([limited[0], candidate])
        if len(limited) >= 4:
            plans.append([limited[1], limited[2]])
        return plans or [limited[:count]]
    return [list(pool[:count])]


def _choose_replacement_plan(
    pool: Sequence[Candidate],
    replacement_count: int,
    market_value: float,
    day_prices: Mapping[str, float],
    base_values_by_ticker: Mapping[str, float],
    base_value: float,
    candidate_by_ticker: Mapping[str, Candidate],
    benchmark_weights: Mapping[str, float],
    target_sectors: Mapping[str, float],
    tracking_cache: _TrackingArrayCache,
    replacement_scores: Dict[Tuple[str, str], Dict[str, float]],
    correlation_cache: _ReturnCorrelationCache,
    source_candidate: Candidate,
    before_snapshot: Mapping[str, object],
    constraints: ReplayConstraintConfig,
    selected_count: int,
    replacement_method: str = "ranked",
) -> Tuple[List[Tuple[Candidate, float, float]], Dict[str, object], List[str]]:
    if replacement_method not in REPLACEMENT_METHODS:
        raise ValueError("replacement_method must be ranked or random")
    best_plan: List[Tuple[Candidate, float, float]] = []
    best_snapshot: Dict[str, object] = {}
    best_violations: List[str] = []
    best_score = float("inf")
    best_allowed_score = float("inf")
    cheap_candidates: List[
        Tuple[float, List[Tuple[Candidate, float, float]], Dict[str, float], float, float]
    ] = []

    for plan in _replacement_plans(pool, replacement_count):
        allocation_plan: List[Tuple[Candidate, float, float]] = []
        for replacement, allocation in _replacement_allocations(plan, market_value):
            replacement_price = day_prices.get(replacement.ticker)
            if replacement_price and replacement_price > 0 and allocation > 0:
                allocation_plan.append((replacement, allocation, replacement_price))
        if not allocation_plan:
            continue

        proposed_values = dict(base_values_by_ticker)
        for replacement, allocation, _ in allocation_plan:
            proposed_values[replacement.ticker] = proposed_values.get(replacement.ticker, 0.0) + allocation
        proposed_total = base_value + sum(allocation for _, allocation, _ in allocation_plan)
        cheap_snapshot = _replay_snapshot_from_values(
            proposed_values,
            proposed_total,
            candidate_by_ticker,
            benchmark_weights,
            target_sectors,
            tracking_cache,
            include_tracking=False,
        )
        similarity_score = sum(
            _cached_replacement_similarity(
                source_candidate,
                replacement,
                replacement_scores,
                correlation_cache,
            )["score"]
            for replacement, _, _ in allocation_plan
        ) / len(allocation_plan)
        beta_delta = abs(_snapshot_number(cheap_snapshot, "beta") - constraints.beta_target)
        sector_abs_error = _snapshot_number(cheap_snapshot, "sector_abs_error")
        active_share_value = _snapshot_number(cheap_snapshot, "active_share")
        max_weight = _snapshot_number(cheap_snapshot, "max_weight")
        cheap_score = (
            4.0 * (beta_delta / max(constraints.beta_band, 0.005)) ** 2
            + 3.0 * (sector_abs_error / max(constraints.sector_band, 0.005)) ** 2
            + 1.5 * active_share_value**2
            + 0.5 * max_weight**2
            + 0.1 * similarity_score
        )
        cheap_candidates.append(
            (cheap_score, allocation_plan, proposed_values, proposed_total, similarity_score)
        )

    if replacement_method == "random":
        fallback_plan: List[Tuple[Candidate, float, float]] = []
        fallback_snapshot: Dict[str, object] = {}
        fallback_violations: List[str] = []
        for _, allocation_plan, proposed_values, proposed_total, _ in cheap_candidates:
            snapshot = _replay_snapshot_from_values(
                proposed_values,
                proposed_total,
                candidate_by_ticker,
                benchmark_weights,
                target_sectors,
                tracking_cache,
            )
            allowed, violations = _replay_constraints_allow(before_snapshot, snapshot, constraints, selected_count)
            if not fallback_plan:
                fallback_plan = allocation_plan
                fallback_snapshot = snapshot
                fallback_violations = violations
            if allowed:
                return allocation_plan, snapshot, violations
        return fallback_plan, fallback_snapshot, fallback_violations

    for cheap_score, allocation_plan, proposed_values, proposed_total, _ in sorted(
        cheap_candidates,
        key=lambda item: item[0],
    )[:3]:
        snapshot = _replay_snapshot_from_values(
            proposed_values,
            proposed_total,
            candidate_by_ticker,
            benchmark_weights,
            target_sectors,
            tracking_cache,
        )
        allowed, violations = _replay_constraints_allow(before_snapshot, snapshot, constraints, selected_count)
        tracking_error_value = _snapshot_number(snapshot, "tracking_error")
        score = (
            8.0 * (tracking_error_value / max(constraints.tracking_error_limit, 0.005)) ** 2
            + cheap_score
        )
        if allowed and score < best_allowed_score:
            best_allowed_score = score
            best_score = score
            best_plan = allocation_plan
            best_snapshot = snapshot
            best_violations = violations
        elif not best_plan and score < best_score:
            best_score = score
            best_plan = allocation_plan
            best_snapshot = snapshot
            best_violations = violations

    return best_plan, best_snapshot, best_violations


def _harvest_path_metrics(
    portfolio_values: Sequence[float],
    benchmark_values: Sequence[float],
    years: float,
    harvest_frequency: str,
) -> Dict[str, object]:
    portfolio_returns = _series_returns(portfolio_values)
    benchmark_returns = _series_returns(benchmark_values)
    observation_count = min(len(portfolio_returns), len(benchmark_returns))
    portfolio_returns = portfolio_returns[:observation_count]
    benchmark_returns = benchmark_returns[:observation_count]
    active_returns = [
        portfolio_return - benchmark_return
        for portfolio_return, benchmark_return in zip(portfolio_returns, benchmark_returns)
    ]
    periods_per_year = FREQUENCIES[harvest_frequency]
    active_mean = sum(active_returns) / len(active_returns) if active_returns else 0.0
    active_variance = _sample_variance(active_returns)
    benchmark_variance = _sample_variance(benchmark_returns)
    portfolio_variance = _sample_variance(portfolio_returns)
    covariance = _sample_covariance(portfolio_returns, benchmark_returns) if observation_count >= 2 else 0.0
    correlation_denominator = math.sqrt(portfolio_variance * benchmark_variance)

    tracking_error_value = math.sqrt(max(active_variance, 0.0) * periods_per_year)
    active_return_value = active_mean * periods_per_year
    return {
        "portfolio_harvest_annualized_return": _annualized_return(portfolio_values, years),
        "benchmark_annualized_return": _annualized_return(benchmark_values, years),
        "portfolio_harvest_active_return": active_return_value,
        "portfolio_harvest_active_return_pct_per_year": active_return_value * 100.0,
        "portfolio_harvest_tracking_error": tracking_error_value,
        "portfolio_harvest_tracking_error_annualized_pct": tracking_error_value * 100.0,
        "portfolio_harvest_beta": covariance / benchmark_variance if benchmark_variance > 0 else 0.0,
        "portfolio_harvest_correlation": covariance / correlation_denominator if correlation_denominator > 0 else 0.0,
        "portfolio_harvest_observations": observation_count,
    }


def _price_lookup_for_dates(points: Sequence[PricePoint], dates: Sequence[date]) -> Dict[date, float]:
    ordered_points = sorted(points, key=lambda point: point.day)
    ordered_dates = sorted(dates)
    lookup: Dict[date, float] = {}
    point_index = 0
    latest: Optional[float] = None
    for day in ordered_dates:
        while point_index < len(ordered_points) and ordered_points[point_index].day <= day:
            price = ordered_points[point_index].adj_close
            if price > 0:
                latest = price
            point_index += 1
        if latest is not None:
            lookup[day] = latest
    return lookup


def _price_table(
    tickers: Iterable[str],
    prices: Mapping[str, Sequence[PricePoint]],
    dates: Sequence[date],
) -> Dict[str, Dict[date, float]]:
    table: Dict[str, Dict[date, float]] = {}
    for ticker in tickers:
        lookup = _price_lookup_for_dates(prices.get(ticker, []), dates)
        if lookup:
            table[ticker] = lookup
    return table


def _cached_replacement_similarity(
    source: Candidate,
    replacement: Candidate,
    cache: Dict[Tuple[str, str], Dict[str, float]],
    correlation_cache: Optional[_ReturnCorrelationCache] = None,
) -> Dict[str, float]:
    key = (source.ticker, replacement.ticker)
    if key not in cache:
        cache[key] = _replacement_similarity(source, replacement, correlation_cache)
    return cache[key]


def _ranked_same_sector_replacements(
    universe: Sequence[Candidate],
    candidate_scores: Mapping[str, float],
    replacement_scores: Dict[Tuple[str, str], Dict[str, float]],
    correlation_cache: Optional[_ReturnCorrelationCache] = None,
) -> Dict[str, List[Candidate]]:
    by_sector: Dict[str, List[Candidate]] = {}
    for candidate in universe:
        by_sector.setdefault(candidate.sector, []).append(candidate)

    rankings: Dict[str, List[Candidate]] = {}
    for source in universe:
        pool = [candidate for candidate in by_sector.get(source.sector, []) if candidate.ticker != source.ticker]
        pool.sort(
            key=lambda candidate: (
                _cached_replacement_similarity(source, candidate, replacement_scores, correlation_cache)["score"],
                candidate_scores.get(candidate.ticker, float("inf")),
                -candidate.index_weight,
                candidate.ticker,
            )
        )
        rankings[source.ticker] = pool
    return rankings


def _active_historical_holdings(
    historical_holdings: Optional[Mapping[date, Sequence[Holding]]],
    day: date,
) -> Optional[Sequence[Holding]]:
    if not historical_holdings:
        return None
    active_day = None
    for snapshot_day in sorted(historical_holdings):
        if snapshot_day <= day:
            active_day = snapshot_day
        else:
            break
    if active_day is None:
        return []
    return historical_holdings[active_day]


def _active_historical_tickers(
    historical_holdings: Optional[Mapping[date, Sequence[Holding]]],
    day: date,
) -> Optional[Set[str]]:
    active = _active_historical_holdings(historical_holdings, day)
    if active is None:
        return None
    return {holding.ticker for holding in active}


def _same_sector_replacements(
    universe: Sequence[Candidate],
    source: Candidate,
    unavailable: Set[str],
    price_table: Mapping[str, Mapping[date, float]],
    day: date,
    candidate_scores: Mapping[str, float],
    replacement_scores: Dict[Tuple[str, str], Dict[str, float]],
    replacement_count: int,
    active_tickers: Optional[Set[str]] = None,
    ranked_replacements: Optional[Mapping[str, Sequence[Candidate]]] = None,
    correlation_cache: Optional[_ReturnCorrelationCache] = None,
    relaxed_unavailable: Optional[Set[str]] = None,
    day_prices: Optional[Mapping[str, float]] = None,
    replacement_method: str = "ranked",
    rng: Optional[random.Random] = None,
) -> List[Candidate]:
    if replacement_count <= 0:
        return []
    if replacement_method not in REPLACEMENT_METHODS:
        raise ValueError("replacement_method must be ranked or random")
    unavailable_tickers = unavailable
    relaxed_unavailable_tickers = relaxed_unavailable
    if replacement_method == "random":
        replacement_rng = rng or random.Random()

        def has_price(candidate: Candidate) -> bool:
            price = (
                day_prices.get(candidate.ticker, 0.0)
                if day_prices is not None
                else price_table.get(candidate.ticker, {}).get(day, 0.0)
            )
            return price > 0

        pool = [
            candidate
            for candidate in universe
            if candidate.sector == source.sector
            and candidate.ticker not in unavailable_tickers
            and (active_tickers is None or candidate.ticker in active_tickers)
            and has_price(candidate)
        ]
        if not pool and relaxed_unavailable_tickers is not None:
            pool = [
                candidate
                for candidate in universe
                if candidate.sector == source.sector
                and candidate.ticker not in relaxed_unavailable_tickers
                and (active_tickers is None or candidate.ticker in active_tickers)
                and has_price(candidate)
            ]
        replacement_rng.shuffle(pool)
        return pool[:replacement_count]

    if ranked_replacements is not None:
        replacements: List[Candidate] = []
        relaxed_replacements: List[Candidate] = []
        for candidate in ranked_replacements.get(source.ticker, []):
            price = (
                day_prices.get(candidate.ticker, 0.0)
                if day_prices is not None
                else price_table.get(candidate.ticker, {}).get(day, 0.0)
            )
            if (
                (active_tickers is not None and candidate.ticker not in active_tickers)
                or price <= 0
            ):
                continue
            if candidate.ticker not in unavailable_tickers:
                replacements.append(candidate)
                if len(replacements) >= replacement_count:
                    break
            if (
                relaxed_unavailable_tickers is not None
                and candidate.ticker not in relaxed_unavailable_tickers
                and len(relaxed_replacements) < replacement_count
            ):
                relaxed_replacements.append(candidate)
        if replacements:
            return replacements
        if relaxed_unavailable_tickers is not None:
            return relaxed_replacements
        return replacements

    pool = [
        candidate
        for candidate in universe
        if candidate.sector == source.sector
        and candidate.ticker not in unavailable_tickers
        and (active_tickers is None or candidate.ticker in active_tickers)
        and (
            day_prices.get(candidate.ticker, 0.0)
            if day_prices is not None
            else price_table.get(candidate.ticker, {}).get(day, 0.0)
        )
        > 0
    ]
    if not pool and relaxed_unavailable_tickers is not None:
        pool = [
            candidate
            for candidate in universe
            if candidate.sector == source.sector
            and candidate.ticker not in relaxed_unavailable_tickers
            and (active_tickers is None or candidate.ticker in active_tickers)
            and (
                day_prices.get(candidate.ticker, 0.0)
                if day_prices is not None
                else price_table.get(candidate.ticker, {}).get(day, 0.0)
            )
            > 0
        ]
    pool.sort(
        key=lambda candidate: (
            _cached_replacement_similarity(source, candidate, replacement_scores, correlation_cache)["score"],
            candidate_scores.get(candidate.ticker, float("inf")),
            -candidate.index_weight,
            candidate.ticker,
        )
    )
    return pool[:replacement_count]


def _replacement_allocations(replacements: Sequence[Candidate], value: float) -> List[Tuple[Candidate, float]]:
    if not replacements or value <= 0:
        return []
    index_weight_total = sum(max(0.0, candidate.index_weight) for candidate in replacements)
    if index_weight_total > 0:
        return [
            (candidate, value * max(0.0, candidate.index_weight) / index_weight_total)
            for candidate in replacements
        ]
    equal_value = value / len(replacements)
    return [(candidate, equal_value) for candidate in replacements]


def _rebalance_lots_to_targets(
    lots: Sequence[SimulatedHarvestLot],
    selected_pairs: Sequence[Tuple[Candidate, float]],
    price_table: Mapping[str, Mapping[date, float]],
    day: date,
    banned_until: Mapping[str, date],
    active_tickers: Optional[Set[str]],
) -> Tuple[List[SimulatedHarvestLot], bool]:
    if not lots or not selected_pairs:
        return list(lots), False

    lot_values: List[Tuple[SimulatedHarvestLot, float, float]] = []
    portfolio_value = 0.0
    for lot in lots:
        price = price_table.get(lot.ticker, {}).get(day)
        if price is None or price <= 0:
            return list(lots), False
        market_value = lot.shares * price
        lot_values.append((lot, price, market_value))
        portfolio_value += market_value
    if portfolio_value <= 0:
        return list(lots), False

    available_targets: List[Tuple[Candidate, float]] = []
    for candidate, weight in selected_pairs:
        is_banned = banned_until.get(candidate.ticker, date.min) >= day
        is_active = active_tickers is None or candidate.ticker in active_tickers
        has_price = price_table.get(candidate.ticker, {}).get(day, 0.0) > 0
        if is_banned or not is_active or not has_price:
            continue
        available_targets.append((candidate, weight))

    target_weight_total = sum(weight for _, weight in available_targets)
    if target_weight_total <= 0:
        return list(lots), False

    target_values = {
        candidate.ticker: portfolio_value * weight / target_weight_total
        for candidate, weight in available_targets
    }
    current_values: Dict[str, float] = {}
    for lot, _, market_value in lot_values:
        current_values[lot.ticker] = current_values.get(lot.ticker, 0.0) + market_value

    tolerance = portfolio_value * 1e-9
    changed = False
    rebalanced: List[SimulatedHarvestLot] = []
    for lot, _, market_value in lot_values:
        target_value = target_values.get(lot.ticker, 0.0)
        current_value = current_values.get(lot.ticker, 0.0)
        if current_value <= 0 or target_value <= 0:
            changed = changed or market_value > tolerance
            continue
        keep_ratio = min(target_value / current_value, 1.0)
        if keep_ratio < 1.0 - 1e-9:
            changed = True
        kept_shares = lot.shares * keep_ratio
        kept_basis = lot.basis * keep_ratio
        if kept_shares > 0 and kept_basis > 0:
            rebalanced.append(
                SimulatedHarvestLot(
                    ticker=lot.ticker,
                    sector=lot.sector,
                    industry=lot.industry,
                    shares=kept_shares,
                    basis=kept_basis,
                    purchase_day=lot.purchase_day,
                )
            )

    candidate_by_ticker = {candidate.ticker: candidate for candidate, _ in available_targets}
    for ticker, target_value in target_values.items():
        current_value = min(current_values.get(ticker, 0.0), target_value)
        buy_value = target_value - current_value
        if buy_value <= tolerance:
            continue
        candidate = candidate_by_ticker[ticker]
        price = price_table[candidate.ticker][day]
        rebalanced.append(
            SimulatedHarvestLot(
                ticker=candidate.ticker,
                sector=candidate.sector,
                industry=candidate.industry,
                shares=buy_value / price,
                basis=buy_value,
                purchase_day=day,
            )
        )
        changed = True

    return rebalanced, changed


def simulate_portfolio_harvests(
    selected: Sequence[Candidate],
    weights: Sequence[float],
    universe: Sequence[Candidate],
    prices: Mapping[str, Sequence[PricePoint]],
    benchmark_ticker: str,
    benchmark_returns: Dict[date, float],
    rebalance_frequency: str,
    error_margin: float,
    target_tax_alpha: float,
    tax_alpha_mode: str = "closest",
    tax_rate: float = 0.30,
    harvest_threshold_pct: float = 0.05,
    transaction_cost_bps: float = 5.0,
    replacement_cost_bps: float = 10.0,
    replacement_count: int = 2,
    wash_sale_days: int = 31,
    sample_event_limit: int = 20,
    historical_holdings: Optional[Mapping[date, Sequence[Holding]]] = None,
    harvest_frequency: Optional[str] = None,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
    max_weight_limit: Optional[float] = None,
    replacement_method: str = "ranked",
    random_seed: int = 7,
) -> Dict[str, object]:
    tax_frequency = harvest_frequency or rebalance_frequency
    if replacement_method not in REPLACEMENT_METHODS:
        raise ValueError("replacement_method must be ranked or random")
    benchmark_points = prices.get(benchmark_ticker)
    if not benchmark_points:
        return _empty_harvest_simulation("benchmark prices are missing", rebalance_frequency, tax_frequency)
    if rebalance_frequency not in FREQUENCIES:
        return _empty_harvest_simulation("unsupported rebalance frequency", rebalance_frequency, tax_frequency)
    if tax_frequency not in FREQUENCIES:
        return _empty_harvest_simulation("unsupported harvest frequency", rebalance_frequency, tax_frequency)
    if len(selected) != len(weights):
        raise ValueError("selected candidates and weights must have the same length")
    if tax_rate < 0:
        raise ValueError("tax_rate must be nonnegative")
    if harvest_threshold_pct < 0:
        raise ValueError("harvest_threshold_pct must be nonnegative")
    if replacement_count < 1:
        raise ValueError("replacement_count must be positive")
    if wash_sale_days < 0:
        raise ValueError("wash_sale_days must be nonnegative")

    harvest_days = {point.day for point in period_end_points(benchmark_points, tax_frequency)}
    rebalance_days = {point.day for point in period_end_points(benchmark_points, rebalance_frequency)}
    schedule = sorted(harvest_days.union(rebalance_days))
    if len(schedule) < 2:
        return _empty_harvest_simulation("not enough harvest dates", rebalance_frequency, tax_frequency)

    tickers = {candidate.ticker for candidate in universe}.union(candidate.ticker for candidate in selected)
    table = _price_table(tickers, prices, schedule)
    tracking_benchmark_returns = benchmark_proxy_returns(universe) or benchmark_returns
    tracking_cache = _TrackingArrayCache(universe, tracking_benchmark_returns)
    benchmark_weights = _benchmark_weight_map(universe)
    target_sectors = _sector_targets_from_candidates(universe)
    construction_tracking_limit = min(error_margin, 0.02)
    constraints = ReplayConstraintConfig(
        tracking_error_limit=construction_tracking_limit,
        path_tracking_error_limit=construction_tracking_limit,
        max_weight_limit=max_weight_limit
        if max_weight_limit is not None
        else (_default_scalar_max_weight(len(selected)) if len(selected) >= 50 else None),
    )
    candidate_scores = {
        candidate.ticker: candidate_score(
            candidate,
            construction_tracking_limit,
            target_tax_alpha,
            tracking_benchmark_returns,
            tax_alpha_mode,
            tracking_cache=tracking_cache,
        )
        for candidate in universe
    }
    candidate_by_ticker = {candidate.ticker: candidate for candidate in universe}
    replacement_scores: Dict[Tuple[str, str], Dict[str, float]] = {}
    correlation_cache = _ReturnCorrelationCache(universe)
    replacement_rng = random.Random(random_seed)
    ranked_replacements = _ranked_same_sector_replacements(
        universe,
        candidate_scores,
        replacement_scores,
        correlation_cache,
    )
    selected_pairs = [
        (candidate, max(0.0, weight))
        for candidate, weight in zip(selected, weights)
        if weight > 0
    ]
    if not selected_pairs:
        return _empty_harvest_simulation("portfolio has no positive weights", rebalance_frequency, tax_frequency)

    start_day = None
    for day in schedule:
        active_tickers = _active_historical_tickers(historical_holdings, day)
        selected_are_active = active_tickers is None or all(
            candidate.ticker in active_tickers
            for candidate, _ in selected_pairs
        )
        selected_have_prices = all(
            table.get(candidate.ticker, {}).get(day, 0.0) > 0
            for candidate, _ in selected_pairs
        )
        if selected_are_active and selected_have_prices:
            start_day = day
            break
    if start_day is None:
        return _empty_harvest_simulation(
            "selected positions do not share a priced start date",
            rebalance_frequency,
            tax_frequency,
        )

    total_weight = sum(weight for _, weight in selected_pairs)
    lots: List[SimulatedHarvestLot] = []
    for candidate, weight in selected_pairs:
        price = table[candidate.ticker][start_day]
        allocation = weight / total_weight
        lots.append(
            SimulatedHarvestLot(
                ticker=candidate.ticker,
                sector=candidate.sector,
                industry=candidate.industry,
                shares=allocation / price,
                basis=allocation,
                purchase_day=start_day,
            )
        )

    simulation_dates = [day for day in schedule if day >= start_day]
    cost_rate = max(transaction_cost_bps, 0.0) / 10000.0
    replacement_cost_rate = max(replacement_cost_bps, 0.0) / 10000.0
    total_realized_loss = 0.0
    total_tax_benefit = 0.0
    total_transaction_cost = 0.0
    total_replacement_cost = 0.0
    harvest_count = 0
    rebalance_count = 0
    skipped_no_replacement = 0
    skipped_nonpositive_net_benefit = 0
    skipped_constraint_violation = 0
    skipped_harvests_by_reason: Dict[str, int] = {}
    banned_until: Dict[str, date] = {}
    portfolio_values: List[float] = []
    period_rows: Dict[date, Dict[str, float]] = {}
    period_replacement_names: Dict[date, List[str]] = {}
    sample_events: List[Dict[str, object]] = []
    harvest_diagnostics: List[Dict[str, object]] = []
    skipped_harvest_events: List[Dict[str, object]] = []
    rebalance_dates: List[str] = []

    def record_skip(reason: str, event: Optional[Dict[str, object]] = None) -> None:
        skipped_harvests_by_reason[reason] = skipped_harvests_by_reason.get(reason, 0) + 1
        if event is not None and len(skipped_harvest_events) < sample_event_limit:
            skipped_harvest_events.append(event)

    def report_progress(count: int) -> None:
        if on_progress is None:
            return
        on_progress(
            count,
            len(simulation_dates),
            f"lots: {len(lots)}, harvests: {harvest_count}, rebalances: {rebalance_count}",
        )

    for day_index, day in enumerate(simulation_dates):
        day_prices = {
            ticker: price
            for ticker, lookup in table.items()
            if (price := lookup.get(day, 0.0)) > 0
        }
        portfolio_value = 0.0
        for lot in lots:
            price = day_prices.get(lot.ticker)
            if price is not None:
                portfolio_value += lot.shares * price
        portfolio_values.append(portfolio_value)

        if day_index == 0:
            report_progress(day_index + 1)
            continue

        pit_active_tickers = _active_historical_tickers(historical_holdings, day)
        if day in harvest_days:
            next_lots: List[SimulatedHarvestLot] = []
            next_lot_tickers: Set[str] = set()
            lot_tickers = {lot.ticker for lot in lots}
            banned_tickers = {
                ticker
                for ticker, banned_day in banned_until.items()
                if banned_day >= day
            }
            sold_this_period: Set[str] = set()
            for lot_index, lot in enumerate(lots):
                price = day_prices.get(lot.ticker)
                if price is None or price <= 0 or lot.basis <= 0:
                    next_lots.append(lot)
                    next_lot_tickers.add(lot.ticker)
                    continue
                market_value = lot.shares * price
                unrealized_return = market_value / lot.basis - 1.0
                realized_loss = lot.basis - market_value
                if unrealized_return > -harvest_threshold_pct or realized_loss <= 0:
                    next_lots.append(lot)
                    next_lot_tickers.add(lot.ticker)
                    continue

                tax_benefit = realized_loss * tax_rate
                transaction_cost = 2.0 * cost_rate * market_value
                replacement_cost = replacement_cost_rate * market_value
                net_tax_benefit = tax_benefit - transaction_cost - replacement_cost
                if net_tax_benefit <= 0:
                    skipped_nonpositive_net_benefit += 1
                    record_skip(
                        "nonpositive_net_benefit",
                        {
                            "date": day.isoformat(),
                            "ticker": lot.ticker,
                            "realized_loss": realized_loss,
                            "net_tax_benefit": net_tax_benefit,
                        },
                    )
                    next_lots.append(lot)
                    next_lot_tickers.add(lot.ticker)
                    continue

                unavailable = lot_tickers.union(next_lot_tickers, banned_tickers, sold_this_period, {lot.ticker})
                source_candidate = candidate_by_ticker.get(lot.ticker)
                if source_candidate is None:
                    next_lots.append(lot)
                    next_lot_tickers.add(lot.ticker)
                    continue
                relaxed_unavailable = banned_tickers.union(sold_this_period, {lot.ticker})
                replacement_pool = _same_sector_replacements(
                    universe,
                    source_candidate,
                    unavailable,
                    table,
                    day,
                    candidate_scores,
                    replacement_scores,
                    max(replacement_count * 8, 14),
                    active_tickers=pit_active_tickers,
                    ranked_replacements=ranked_replacements,
                    correlation_cache=correlation_cache,
                    relaxed_unavailable=relaxed_unavailable,
                    day_prices=day_prices,
                    replacement_method=replacement_method,
                    rng=replacement_rng,
                )
                before_snapshot = _replay_snapshot(
                    lots,
                    day_prices,
                    candidate_by_ticker,
                    benchmark_weights,
                    target_sectors,
                    tracking_cache,
                )
                remaining_lots = list(lots[lot_index + 1 :])
                lots_after_replacement_base = next_lots + remaining_lots
                base_values_by_ticker, base_value = _values_from_lots(
                    lots_after_replacement_base,
                    day_prices,
                )
                allocation_plan, after_snapshot, replacement_violations = _choose_replacement_plan(
                    replacement_pool,
                    replacement_count,
                    market_value,
                    day_prices,
                    base_values_by_ticker,
                    base_value,
                    candidate_by_ticker,
                    benchmark_weights,
                    target_sectors,
                    tracking_cache,
                    replacement_scores,
                    correlation_cache,
                    source_candidate,
                    before_snapshot,
                    constraints,
                    len(selected_pairs),
                    replacement_method=replacement_method,
                )
                if not allocation_plan:
                    skipped_no_replacement += 1
                    record_skip(
                        "no_replacement",
                        {
                            "date": day.isoformat(),
                            "ticker": lot.ticker,
                            "market_value": market_value,
                            "cash_before": _snapshot_number(before_snapshot, "cash_weight"),
                            "cash_after": _snapshot_number(before_snapshot, "cash_weight"),
                        },
                    )
                    next_lots.append(lot)
                    next_lot_tickers.add(lot.ticker)
                    continue
                allowed, violations = _replay_constraints_allow(
                    before_snapshot,
                    after_snapshot,
                    constraints,
                    len(selected_pairs),
                )
                if not allowed:
                    skipped_constraint_violation += 1
                    reason = "constraint_violation"
                    record_skip(
                        reason,
                        {
                            "date": day.isoformat(),
                            "ticker": lot.ticker,
                            "violations": violations,
                            "replacement_candidates": [replacement.ticker for replacement, _, _ in allocation_plan],
                            "cash_before": _snapshot_number(before_snapshot, "cash_weight"),
                            "cash_after": _snapshot_number(after_snapshot, "cash_weight"),
                            "tracking_error_before": _snapshot_number(before_snapshot, "tracking_error"),
                            "tracking_error_after": _snapshot_number(after_snapshot, "tracking_error"),
                            "beta_before": _snapshot_number(before_snapshot, "beta"),
                            "beta_after": _snapshot_number(after_snapshot, "beta"),
                            "active_share_before": _snapshot_number(before_snapshot, "active_share"),
                            "active_share_after": _snapshot_number(after_snapshot, "active_share"),
                            "sector_abs_error_before": _snapshot_number(before_snapshot, "sector_abs_error"),
                            "sector_abs_error_after": _snapshot_number(after_snapshot, "sector_abs_error"),
                        },
                    )
                    next_lots.append(lot)
                    next_lot_tickers.add(lot.ticker)
                    continue

                total_realized_loss += realized_loss
                total_tax_benefit += tax_benefit
                total_transaction_cost += transaction_cost
                total_replacement_cost += replacement_cost
                harvest_count += 1
                sold_this_period.add(lot.ticker)
                banned_until[lot.ticker] = day + timedelta(days=wash_sale_days)
                banned_tickers.add(lot.ticker)
                period = period_rows.setdefault(
                    day,
                    {
                        "realized_loss": 0.0,
                        "tax_benefit": 0.0,
                        "transaction_cost": 0.0,
                        "replacement_cost": 0.0,
                        "net_tax_benefit": 0.0,
                        "harvest_count": 0.0,
                    },
                )
                period["realized_loss"] += realized_loss
                period["tax_benefit"] += tax_benefit
                period["transaction_cost"] += transaction_cost
                period["replacement_cost"] += replacement_cost
                period["net_tax_benefit"] += net_tax_benefit
                period["harvest_count"] += 1.0

                replacement_event_rows: List[Dict[str, object]] = []
                for replacement, allocation, replacement_price in allocation_plan:
                    similarity = _cached_replacement_similarity(
                        source_candidate,
                        replacement,
                        replacement_scores,
                        correlation_cache,
                    )
                    next_lots.append(
                        SimulatedHarvestLot(
                            ticker=replacement.ticker,
                            sector=replacement.sector,
                            industry=replacement.industry,
                            shares=allocation / replacement_price,
                            basis=allocation,
                            purchase_day=day,
                        )
                    )
                    next_lot_tickers.add(replacement.ticker)
                    replacement_event_rows.append(
                        {
                            "ticker": replacement.ticker,
                            "industry": replacement.industry,
                            "value": allocation,
                            "price": replacement_price,
                            "beta_delta": similarity["beta_delta"],
                            "return_correlation": similarity["correlation"],
                            "industry_match": bool(similarity["industry_match"]),
                        }
                    )
                period_replacement_names.setdefault(day, []).extend(
                    str(row["ticker"]) for row in replacement_event_rows
                )
                if len(sample_events) < sample_event_limit:
                    sample_events.append(
                        {
                            "date": day.isoformat(),
                            "sold": lot.ticker,
                            "sector": lot.sector,
                            "industry": lot.industry,
                            "basis": lot.basis,
                            "market_value": market_value,
                            "realized_loss": realized_loss,
                            "unrealized_return": unrealized_return,
                            "replacements": replacement_event_rows,
                            "diagnostics_before": {
                                "cash_weight": _snapshot_number(before_snapshot, "cash_weight"),
                                "tracking_error": _snapshot_number(before_snapshot, "tracking_error"),
                                "beta": _snapshot_number(before_snapshot, "beta"),
                                "active_share": _snapshot_number(before_snapshot, "active_share"),
                                "sector_abs_error": _snapshot_number(before_snapshot, "sector_abs_error"),
                                "max_weight": _snapshot_number(before_snapshot, "max_weight"),
                                "effective_names": _snapshot_number(before_snapshot, "effective_names"),
                            },
                            "diagnostics_after": {
                                "cash_weight": _snapshot_number(after_snapshot, "cash_weight"),
                                "tracking_error": _snapshot_number(after_snapshot, "tracking_error"),
                                "beta": _snapshot_number(after_snapshot, "beta"),
                                "active_share": _snapshot_number(after_snapshot, "active_share"),
                                "sector_abs_error": _snapshot_number(after_snapshot, "sector_abs_error"),
                                "max_weight": _snapshot_number(after_snapshot, "max_weight"),
                                "effective_names": _snapshot_number(after_snapshot, "effective_names"),
                            },
                            "constraint_violations_after": replacement_violations,
                        }
                    )
                if len(harvest_diagnostics) < sample_event_limit:
                    harvest_diagnostics.append(
                        {
                            "date": day.isoformat(),
                            "sold": lot.ticker,
                            "replacement_names": [replacement.ticker for replacement, _, _ in allocation_plan],
                            "realized_loss": realized_loss,
                            "cash_before": _snapshot_number(before_snapshot, "cash_weight"),
                            "cash_after": _snapshot_number(after_snapshot, "cash_weight"),
                            "tracking_error_before": _snapshot_number(before_snapshot, "tracking_error"),
                            "tracking_error_after": _snapshot_number(after_snapshot, "tracking_error"),
                            "beta_before": _snapshot_number(before_snapshot, "beta"),
                            "beta_after": _snapshot_number(after_snapshot, "beta"),
                            "active_share_before": _snapshot_number(before_snapshot, "active_share"),
                            "active_share_after": _snapshot_number(after_snapshot, "active_share"),
                            "sector_abs_error_before": _snapshot_number(before_snapshot, "sector_abs_error"),
                            "sector_abs_error_after": _snapshot_number(after_snapshot, "sector_abs_error"),
                            "max_weight_before": _snapshot_number(before_snapshot, "max_weight"),
                            "max_weight_after": _snapshot_number(after_snapshot, "max_weight"),
                            "effective_names_before": _snapshot_number(before_snapshot, "effective_names"),
                            "effective_names_after": _snapshot_number(after_snapshot, "effective_names"),
                            "constraint_violations_after": replacement_violations,
                        }
                    )
            lots = next_lots

        if day in rebalance_days:
            lots, rebalanced = _rebalance_lots_to_targets(
                lots,
                selected_pairs,
                table,
                day,
                banned_until,
                pit_active_tickers,
            )
            if rebalanced:
                rebalance_count += 1
                rebalance_dates.append(day.isoformat())

        report_progress(day_index + 1)

    end_day = simulation_dates[-1]
    years = max((end_day - start_day).days / 365.25, 1.0 / FREQUENCIES[tax_frequency])
    average_portfolio_value = sum(portfolio_values) / len(portfolio_values) if portfolio_values else 0.0
    ending_portfolio_value = portfolio_values[-1] if portfolio_values else 0.0
    total_net_tax_benefit = total_tax_benefit - total_transaction_cost - total_replacement_cost
    full_liquidation_after_tax_benefit = total_net_tax_benefit - total_tax_benefit
    if average_portfolio_value > 0:
        portfolio_realized_loss_rate = total_realized_loss / average_portfolio_value / years
        immediate_tax_savings_rate = total_tax_benefit / average_portfolio_value / years
        immediate_net_tax_savings_rate = total_net_tax_benefit / average_portfolio_value / years
        portfolio_simulated_tax_alpha = full_liquidation_after_tax_benefit / average_portfolio_value / years
    else:
        portfolio_realized_loss_rate = 0.0
        portfolio_simulated_tax_alpha = 0.0
        immediate_tax_savings_rate = 0.0
        immediate_net_tax_savings_rate = 0.0
    aligned_portfolio_values = []
    aligned_benchmark_values = []
    if portfolio_values and portfolio_values[0] > 0:
        aligned_portfolio_values.append(portfolio_values[0])
        aligned_benchmark_values.append(portfolio_values[0])
        benchmark_value = portfolio_values[0]
        for day, portfolio_value in zip(simulation_dates[1:], portfolio_values[1:]):
            benchmark_return = tracking_benchmark_returns.get(day)
            if benchmark_return is None or portfolio_value <= 0:
                continue
            benchmark_value *= 1.0 + benchmark_return
            if benchmark_value > 0:
                aligned_portfolio_values.append(portfolio_value)
                aligned_benchmark_values.append(benchmark_value)
    path_metrics = _harvest_path_metrics(
        aligned_portfolio_values,
        aligned_benchmark_values,
        years,
        tax_frequency,
    )
    terminal_after_tax_wealth_difference = (
        full_liquidation_after_tax_benefit / ending_portfolio_value
        if ending_portfolio_value > 0
        else 0.0
    )

    return {
        "status": "ok",
        "frequency": tax_frequency,
        "rebalance_frequency": rebalance_frequency,
        "harvest_frequency": tax_frequency,
        "start": start_day.isoformat(),
        "end": end_day.isoformat(),
        "years": years,
        "average_portfolio_value": average_portfolio_value,
        "ending_portfolio_value": ending_portfolio_value,
        "total_realized_loss": total_realized_loss,
        "annual_realized_loss": total_realized_loss / years,
        "portfolio_realized_loss_rate": portfolio_realized_loss_rate,
        "total_tax_benefit": total_tax_benefit,
        "total_transaction_cost": total_transaction_cost,
        "total_replacement_cost": total_replacement_cost,
        "total_net_tax_benefit": total_net_tax_benefit,
        "full_liquidation_after_tax_benefit": full_liquidation_after_tax_benefit,
        "portfolio_simulated_tax_alpha": portfolio_simulated_tax_alpha,
        "realized_loss_rate_pct_per_year": portfolio_realized_loss_rate * 100.0,
        "immediate_tax_savings_pct_per_year": immediate_tax_savings_rate * 100.0,
        "immediate_net_tax_savings_pct_per_year": immediate_net_tax_savings_rate * 100.0,
        "simulated_after_tax_alpha_pct_per_year": portfolio_simulated_tax_alpha * 100.0,
        "full_liquidation_after_tax_alpha_pct_per_year": portfolio_simulated_tax_alpha * 100.0,
        "terminal_after_tax_wealth_difference_pct": terminal_after_tax_wealth_difference * 100.0,
        "cash_drag": 0.0,
        **path_metrics,
        "harvest_count": harvest_count,
        "rebalance_count": rebalance_count,
        "rebalance_dates": rebalance_dates,
        "skipped_no_replacement": skipped_no_replacement,
        "skipped_nonpositive_net_benefit": skipped_nonpositive_net_benefit,
        "skipped_constraint_violation": skipped_constraint_violation,
        "skipped_harvests_by_reason": skipped_harvests_by_reason,
        "tax_rate": tax_rate,
        "harvest_threshold_pct": harvest_threshold_pct,
        "transaction_cost_bps": transaction_cost_bps,
        "replacement_cost_bps": replacement_cost_bps,
        "replacement_count": replacement_count,
        "replacement_method": replacement_method,
        "wash_sale_days": wash_sale_days,
        "random_seed": random_seed,
        "replay_constraints": {
            "tracking_error_limit": constraints.tracking_error_limit,
            "path_tracking_error_limit": constraints.path_tracking_error_limit,
            "sector_band": constraints.sector_band,
            "sector_abs_error_limit": constraints.sector_abs_error_limit,
            "beta_target": constraints.beta_target,
            "beta_band": constraints.beta_band,
            "active_share_limit": constraints.active_share_limit,
            "cash_weight_limit": constraints.cash_weight_limit,
            "max_weight_limit": constraints.max_weight_limit,
            "enforce_min_names": constraints.enforce_min_names,
        },
        "point_in_time_constituents": bool(historical_holdings),
        "harvest_diagnostics": harvest_diagnostics,
        "skipped_harvest_events": skipped_harvest_events,
        "period_realized_losses": [
            {
                "date": day.isoformat(),
                "realized_loss": row["realized_loss"],
                "tax_benefit": row["tax_benefit"],
                "transaction_cost": row["transaction_cost"],
                "replacement_cost": row["replacement_cost"],
                "net_tax_benefit": row["net_tax_benefit"],
                "harvest_count": int(row["harvest_count"]),
                "replacement_names": period_replacement_names.get(day, []),
            }
            for day, row in sorted(period_rows.items())
        ],
        "sample_events": sample_events,
    }


def active_share(
    candidates: Sequence[Candidate],
    weights: Sequence[float],
    holdings: Sequence[Holding],
) -> float:
    portfolio_weights = {candidate.ticker: weight for candidate, weight in zip(candidates, weights)}
    index_weights = {holding.ticker: holding.weight for holding in holdings}
    tickers = set(portfolio_weights).union(index_weights)
    return 0.5 * sum(abs(portfolio_weights.get(ticker, 0.0) - index_weights.get(ticker, 0.0)) for ticker in tickers)


def construct_portfolio(
    candidates: Sequence[Candidate],
    holdings: Sequence[Holding],
    sample_size: int,
    error_margin: float,
    target_tax_alpha: float,
    rebalance_frequency: str,
    match_sectors: bool = False,
    harvest_frequency: Optional[str] = None,
    tax_alpha_mode: str = "closest",
    min_weight: float = 0.0,
    max_weight: Optional[float] = None,
    benchmark_returns: Optional[Dict[date, float]] = None,
    tracking_error_penalty: float = 1.0,
    index_anchor_penalty: float = 0.0,
    tax_metric: str = "simulated",
    tax_assumptions: Optional[Dict[str, float]] = None,
    selection_iterations: int = 1000,
    weight_iterations: int = 2000,
    random_seed: int = 7,
    selection_method: str = "optimized",
    weight_method: str = "slsqp",
    beam_width: int = 5,
    miqp_time_limit: Optional[float] = 60.0,
    miqp_gap: Optional[float] = 0.01,
    replacement_method: str = "ranked",
    allow_constraint_violations: bool = False,
    show_progress: bool = False,
    progress_label: str = "Optimization",
    replacement_universe: Optional[Sequence[Candidate]] = None,
) -> Dict[str, object]:
    if benchmark_returns is None:
        raise ValueError("benchmark_returns are required to optimize error margin")
    if selection_method not in SELECTION_METHODS:
        raise ValueError("selection_method must be optimized, random-weighted, greedy, beam, or miqp")
    if weight_method not in WEIGHT_METHODS:
        raise ValueError("weight_method must be slsqp or index-normalized")
    if replacement_method not in REPLACEMENT_METHODS:
        raise ValueError("replacement_method must be ranked or random")
    tax_frequency = harvest_frequency or rebalance_frequency
    construction_error_margin = min(error_margin, 0.02)
    construction_benchmark_returns = benchmark_proxy_returns(candidates) or benchmark_returns
    targets = sector_targets(holdings) if match_sectors else None
    tracking_cache = _TrackingArrayCache(candidates, construction_benchmark_returns)
    selection_diagnostics: Dict[str, object] = {}
    if selection_method == "optimized":
        initial = initial_selection(
            candidates,
            sample_size,
            construction_error_margin,
            target_tax_alpha,
            construction_benchmark_returns,
            match_sectors,
            targets,
            tax_alpha_mode=tax_alpha_mode,
            index_weight_priority=True,
            tracking_cache=tracking_cache,
        )
        selected = optimize_selection(
            candidates,
            initial,
            sample_size,
            construction_error_margin,
            target_tax_alpha,
            construction_benchmark_returns,
            targets,
            match_sectors,
            tax_alpha_mode=tax_alpha_mode,
            iterations=selection_iterations,
            random_seed=random_seed,
            show_progress=show_progress,
            progress_label=f"{progress_label} selection",
            tracking_cache=tracking_cache,
        )
    elif selection_method == "random-weighted":
        selected = random_weighted_selection(
            candidates,
            sample_size,
            random_seed=random_seed,
            match_sectors=match_sectors,
            target_sectors=targets,
        )
    elif selection_method == "miqp":
        selected, selection_diagnostics = miqp_selection(
            candidates,
            sample_size,
            construction_error_margin,
            target_tax_alpha,
            construction_benchmark_returns,
            targets,
            match_sectors,
            tax_alpha_mode=tax_alpha_mode,
            min_weight=min_weight,
            max_weight=max_weight,
            tracking_error_penalty=tracking_error_penalty,
            index_anchor_penalty=index_anchor_penalty,
            miqp_time_limit=miqp_time_limit,
            miqp_gap=miqp_gap,
        )
    else:
        selected = beam_selection(
            candidates,
            sample_size,
            construction_error_margin,
            target_tax_alpha,
            construction_benchmark_returns,
            targets,
            match_sectors,
            tax_alpha_mode=tax_alpha_mode,
            beam_width=1 if selection_method == "greedy" else beam_width,
            tracking_cache=tracking_cache,
        )
    if weight_method == "slsqp":
        weights = optimize_weights(
            selected,
            construction_error_margin,
            target_tax_alpha,
            targets if match_sectors else None,
            tax_alpha_mode=tax_alpha_mode,
            iterations=weight_iterations,
            min_weight=min_weight,
            max_weight=max_weight,
            benchmark_returns=construction_benchmark_returns,
            tracking_error_penalty=tracking_error_penalty,
            index_anchor_penalty=index_anchor_penalty,
            sector_band=0.02 if match_sectors else None,
            show_progress=show_progress,
            progress_label=f"{progress_label} weights",
        )
    else:
        weights = index_weighted_weights(selected, min_weight=min_weight, max_weight=max_weight)
    tracking_model = prepare_tracking_model(selected, construction_benchmark_returns)
    metrics = portfolio_metrics(selected, weights, tracking_model=tracking_model)
    metrics.update(selection_diagnostics)
    price_tracking_model = prepare_tracking_model(selected, benchmark_returns)
    if price_tracking_model is not None:
        metrics["price_benchmark_tracking_error"] = tracking_error(weights, price_tracking_model)
        metrics["price_benchmark_tracking_error_annualized_pct"] = (
            float(metrics["price_benchmark_tracking_error"]) * 100.0
        )
    metrics["active_share"] = active_share(selected, weights, holdings)
    sector_targets_out = targets or {}
    if sector_targets_out:
        metrics["sector_abs_error"] = sector_error(_sector_vector(selected, weights), sector_targets_out)
    _add_percent_metric_aliases(metrics)
    metrics["constraint_warnings"] = _constraint_warnings(metrics, sample_size)
    hard_violations = _hard_constraint_violations(metrics, selected, weights, sample_size)
    metrics["constraint_violations"] = hard_violations
    if hard_violations and not allow_constraint_violations:
        raise ValueError(
            "constructed portfolio violates hard benchmark-fidelity constraints: "
            + "; ".join(hard_violations)
        )

    positions = []
    for candidate, weight in sorted(zip(selected, weights), key=lambda item: item[1], reverse=True):
        positions.append(
            {
                "ticker": candidate.ticker,
                "weight": weight,
                "sector": candidate.sector,
                "industry": candidate.industry,
                "index_weight": candidate.index_weight,
                "beta": candidate.beta,
                "tax_alpha": candidate.tax_alpha,
                "simulated_tax_alpha": candidate.simulated_tax_alpha,
                "gross_harvestable_loss_rate": candidate.gross_harvestable_loss_rate,
                "estimated_tax_loss_alpha": candidate.tax_alpha,
                "observations": candidate.observations,
            }
        )

    return {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "targets": {
            "sample_size": sample_size,
            "error_margin": error_margin,
            "construction_tracking_error_limit": construction_error_margin,
            "tracking_model": "current_constituent_proxy",
            "estimated_tax_loss_alpha": target_tax_alpha,
            "tax_alpha_mode": tax_alpha_mode,
            "min_weight": min_weight,
            "max_weight": max_weight,
            "tracking_error_penalty": tracking_error_penalty,
            "index_anchor_penalty": index_anchor_penalty,
            "selection_method": selection_method,
            "weight_method": weight_method,
            "beam_width": beam_width,
            "miqp_time_limit": miqp_time_limit,
            "miqp_gap": miqp_gap,
            "replacement_method": replacement_method,
            "allow_constraint_violations": allow_constraint_violations,
            "random_seed": random_seed,
            "tax_metric": tax_metric,
            "tax_assumptions": tax_assumptions or {},
            "rebalance_frequency": rebalance_frequency,
            "harvest_frequency": tax_frequency,
            "sector_match": match_sectors,
        },
        "metrics": metrics,
        "sector_targets": sector_targets_out,
        "positions": positions,
        "replacement_candidates": replacement_candidates(
            replacement_universe or candidates,
            selected,
            error_margin,
            target_tax_alpha,
            benchmark_proxy_returns(replacement_universe or candidates) or construction_benchmark_returns,
            tax_alpha_mode=tax_alpha_mode,
        ),
    }
