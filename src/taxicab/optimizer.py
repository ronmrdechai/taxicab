from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - dependency fallback
    tqdm = None

from .data import Holding, PricePoint, sector_targets
from .metrics import (
    FREQUENCIES,
    beta_to_benchmark,
    daily_returns,
    estimated_tax_loss_alpha,
    observations_overlap,
    period_end_points,
    simulated_tax_alpha,
)


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


@dataclass(frozen=True)
class TrackingModel:
    covariance_matrix: np.ndarray
    asset_benchmark_covariance: np.ndarray
    benchmark_variance: float
    observations: int
    annualization: float = 252.0


@dataclass(frozen=True)
class SimulatedHarvestLot:
    ticker: str
    sector: str
    shares: float
    basis: float
    purchase_day: date


def build_candidates(
    holdings: Sequence[Holding],
    prices: Dict[str, Sequence[PricePoint]],
    benchmark_ticker: str,
    rebalance_frequency: str,
    min_observations: int = 252,
    tax_metric: str = "simulated",
    tax_rate: float = 0.30,
    harvest_threshold_pct: float = 0.05,
    transaction_cost_bps: float = 5.0,
    replacement_cost_bps: float = 10.0,
) -> List[Candidate]:
    if tax_metric not in {"simulated", "gross"}:
        raise ValueError("tax_metric must be simulated or gross")
    if benchmark_ticker not in prices:
        raise ValueError(f"benchmark prices missing for {benchmark_ticker}")
    benchmark_returns = daily_returns(prices[benchmark_ticker])

    candidates: List[Candidate] = []
    for holding in holdings:
        if holding.ticker not in prices:
            continue
        asset_returns = daily_returns(prices[holding.ticker])
        overlap = observations_overlap(asset_returns, benchmark_returns)
        if overlap < min_observations:
            continue
        gross = estimated_tax_loss_alpha(prices[holding.ticker], rebalance_frequency)
        simulated = simulated_tax_alpha(
            prices[holding.ticker],
            rebalance_frequency,
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


def prepare_tracking_model(
    candidates: Sequence[Candidate],
    benchmark_returns: Optional[Dict[date, float]],
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


def tracking_error(weights: Sequence[float], model: TrackingModel) -> float:
    active_variance = _active_variance(weights, model)
    return math.sqrt(max(active_variance, 0.0) * model.annualization)


def _active_variance(weights: Sequence[float], model: TrackingModel) -> float:
    weight_array = np.asarray(weights, dtype=float)
    return float(
        weight_array @ model.covariance_matrix @ weight_array
        - 2.0 * (weight_array @ model.asset_benchmark_covariance)
        + model.benchmark_variance
    )


def _active_variance_gradient(weights: Sequence[float], model: TrackingModel) -> List[float]:
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
    matrix = np.zeros((len(sector_names), len(candidates)), dtype=float)
    for column, candidate in enumerate(candidates):
        matrix[sector_names.index(candidate.sector), column] = 1.0
    targets = np.asarray([target_sectors.get(sector, 0.0) for sector in sector_names], dtype=float)
    return matrix, targets


def _progress_range(total: int, label: str, show_progress: bool) -> Iterable[int]:
    values = range(total)
    if show_progress and tqdm is not None and total > 0:
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
    tracking_error_penalty: float = 1.0,
    tax_penalty: float = 1.0,
    sector_penalty: float = 1.0,
    concentration_penalty: float = 0.01,
) -> float:
    if tax_alpha_mode not in {"closest", "at-least"}:
        raise ValueError("tax_alpha_mode must be closest or at-least")
    if error_margin <= 0:
        raise ValueError("error_margin must be positive")
    tracking_model = prepare_tracking_model(candidates, benchmark_returns)
    if tracking_model is None:
        raise ValueError("benchmark returns and candidate returns are required to optimize error margin")
    metrics = portfolio_metrics(candidates, weights, tracking_model=tracking_model)
    tracking_error_scale = max(error_margin, 0.005)
    tax_scale = max(abs(target_tax_alpha), 0.005)
    tracking_error_residual = float(metrics["tracking_error"]) / tracking_error_scale
    tax_delta = float(metrics["estimated_tax_loss_alpha"]) - target_tax_alpha
    if tax_alpha_mode == "at-least":
        tax_delta = min(tax_delta, 0.0)
    tax_residual = tax_delta / tax_scale
    value = tracking_error_penalty * tracking_error_residual**2 + tax_penalty * tax_residual**2
    if target_sectors:
        sectors = metrics["sectors"]
        all_sectors = set(target_sectors).union(sectors)
        value += sector_penalty * sum(
            (float(sectors.get(sector, 0.0)) - target_sectors.get(sector, 0.0)) ** 2
            for sector in all_sectors
        )
    value += concentration_penalty * sum(weight * weight for weight in weights)
    return value


def index_normalized_weights(candidates: Sequence[Candidate]) -> List[float]:
    total = sum(max(0.0, candidate.index_weight) for candidate in candidates)
    if total <= 0:
        return [1.0 / len(candidates)] * len(candidates)
    return [max(0.0, candidate.index_weight) / total for candidate in candidates]


def candidate_score(
    candidate: Candidate,
    error_margin: float,
    target_tax_alpha: float,
    benchmark_returns: Dict[date, float],
    tax_alpha_mode: str = "closest",
) -> float:
    if tax_alpha_mode not in {"closest", "at-least"}:
        raise ValueError("tax_alpha_mode must be closest or at-least")
    if error_margin <= 0:
        raise ValueError("error_margin must be positive")
    tracking_model = prepare_tracking_model([candidate], benchmark_returns)
    if tracking_model is None:
        error_score = float("inf")
    else:
        error_scale = max(error_margin, 0.005)
        error_score = (tracking_error([1.0], tracking_model) / error_scale) ** 2
    tax_scale = max(abs(target_tax_alpha), 0.005)
    tax_delta = candidate.tax_alpha - target_tax_alpha
    if tax_alpha_mode == "at-least":
        tax_delta = min(tax_delta, 0.0)
    tax_score = (tax_delta / tax_scale) ** 2
    weight_bonus = math.sqrt(max(candidate.index_weight, 0.0))
    return error_score + tax_score - 0.05 * weight_bonus


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
                candidate_score(candidate, error_margin, target_tax_alpha, benchmark_returns, tax_alpha_mode),
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
) -> List[Candidate]:
    rng = random.Random(random_seed)
    selected_by_ticker = {candidate.ticker: candidate for candidate in selected}
    universe_by_ticker = {candidate.ticker: candidate for candidate in candidates}
    by_sector: Dict[str, List[Candidate]] = {}
    for candidate in candidates:
        by_sector.setdefault(candidate.sector, []).append(candidate)

    def score(selection: Sequence[Candidate]) -> float:
        weights = index_normalized_weights(selection)
        return objective_value(
            selection,
            weights,
            error_margin,
            target_tax_alpha,
            target_sectors if match_sectors else None,
            benchmark_returns=benchmark_returns,
            tax_alpha_mode=tax_alpha_mode,
            sector_penalty=2.0 if match_sectors else 0.0,
        )

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
    tax_penalty: float = 1.0,
    sector_penalty: float = 2.0,
    concentration_penalty: float = 0.005,
    min_weight: float = 0.0,
    max_weight: Optional[float] = None,
    benchmark_returns: Optional[Dict[date, float]] = None,
    tracking_error_penalty: float = 1.0,
    index_anchor_penalty: float = 0.0,
    show_progress: bool = False,
    progress_label: str = "Weights",
) -> List[float]:
    if tax_alpha_mode not in {"closest", "at-least"}:
        raise ValueError("tax_alpha_mode must be closest or at-least")
    if error_margin <= 0:
        raise ValueError("error_margin must be positive")
    if not candidates:
        raise ValueError("no candidates supplied")
    max_weight_for_projection = max_weight if max_weight is not None else 1.0
    anchor_weights_array = _project_to_bounded_simplex_array(
        index_normalized_weights(candidates),
        lower=min_weight,
        upper=max_weight_for_projection,
    )
    weights_array = anchor_weights_array.copy()
    tax_values = np.asarray([candidate.tax_alpha for candidate in candidates], dtype=float)
    sector_matrix, target_sector_values = _sector_exposure_arrays(candidates, target_sectors)
    tax_scale = max(abs(target_tax_alpha), 0.005)
    tracking_model = prepare_tracking_model(candidates, benchmark_returns)
    if tracking_model is None:
        raise ValueError("benchmark returns and candidate returns are required to optimize error margin")
    tracking_error_scale = max(error_margin, 0.005) ** 2

    for step_index in _progress_range(iterations, progress_label, show_progress):
        tax_alpha = float(weights_array @ tax_values)
        tax_delta = tax_alpha - target_tax_alpha
        if tax_alpha_mode == "at-least":
            tax_delta = min(tax_delta, 0.0)
        tax_residual = tax_delta / (tax_scale**2)
        active_gradient = np.asarray(_active_variance_gradient(weights_array, tracking_model), dtype=float)
        gradient = tracking_error_penalty * tracking_model.annualization * active_gradient / tracking_error_scale
        gradient += 2.0 * tax_penalty * tax_residual * tax_values
        if sector_matrix is not None and target_sector_values is not None:
            sector_residual = sector_matrix @ weights_array - target_sector_values
            gradient += 2.0 * sector_penalty * (sector_matrix.T @ sector_residual)
        gradient += 2.0 * concentration_penalty * weights_array
        gradient += 2.0 * index_anchor_penalty * (weights_array - anchor_weights_array)
        step = learning_rate / math.sqrt(1.0 + step_index / 25.0)
        weights_array = _project_to_bounded_simplex_array(
            weights_array - step * gradient,
            lower=min_weight,
            upper=max_weight_for_projection,
        )
        if tracking_error(weights_array, tracking_model) > error_margin:
            weights_array = np.asarray(
                _repair_tracking_error(weights_array, anchor_weights_array, tracking_model, error_margin),
                dtype=float,
            )
    return weights_array.tolist()


def _repair_tracking_error(
    weights: Sequence[float],
    anchor_weights: Sequence[float],
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
    by_sector: Dict[str, List[Candidate]] = {}
    for candidate in universe:
        if candidate.ticker not in selected_tickers:
            by_sector.setdefault(candidate.sector, []).append(candidate)
    replacements: Dict[str, List[Dict[str, object]]] = {}
    score_cache: Dict[str, float] = {}

    def score_candidate(candidate: Candidate) -> float:
        if candidate.ticker not in score_cache:
            score_cache[candidate.ticker] = candidate_score(
                candidate,
                error_margin,
                target_tax_alpha,
                benchmark_returns,
                tax_alpha_mode,
            )
        return score_cache[candidate.ticker]

    for candidate in selected:
        pool = sorted(
            by_sector.get(candidate.sector, []),
            key=lambda item: (
                score_candidate(item),
                -item.index_weight,
            ),
        )
        replacements[candidate.ticker] = [
            {
                "ticker": item.ticker,
                "sector": item.sector,
                "index_weight": item.index_weight,
                "beta": item.beta,
                "tax_alpha": item.tax_alpha,
                "simulated_tax_alpha": item.simulated_tax_alpha,
                "gross_harvestable_loss_rate": item.gross_harvestable_loss_rate,
                "estimated_tax_loss_alpha": item.tax_alpha,
            }
            for item in pool[:limit]
        ]
    return replacements


def _empty_harvest_simulation(reason: str, frequency: str) -> Dict[str, object]:
    return {
        "status": "skipped",
        "reason": reason,
        "frequency": frequency,
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
        "portfolio_simulated_tax_alpha": 0.0,
        "harvest_count": 0,
        "skipped_no_replacement": 0,
        "skipped_nonpositive_net_benefit": 0,
        "period_realized_losses": [],
        "sample_events": [],
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


def _same_sector_replacements(
    universe: Sequence[Candidate],
    sector: str,
    unavailable: Set[str],
    price_table: Mapping[str, Mapping[date, float]],
    day: date,
    candidate_scores: Mapping[str, float],
    replacement_count: int,
) -> List[Candidate]:
    if replacement_count <= 0:
        return []
    unavailable_upper = {ticker.upper() for ticker in unavailable}
    pool = [
        candidate
        for candidate in universe
        if candidate.sector == sector
        and candidate.ticker not in unavailable_upper
        and price_table.get(candidate.ticker, {}).get(day, 0.0) > 0
    ]
    pool.sort(
        key=lambda candidate: (
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
) -> Dict[str, object]:
    benchmark_points = prices.get(benchmark_ticker)
    if not benchmark_points:
        return _empty_harvest_simulation("benchmark prices are missing", rebalance_frequency)
    if rebalance_frequency not in FREQUENCIES:
        return _empty_harvest_simulation("unsupported rebalance frequency", rebalance_frequency)
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

    schedule = [point.day for point in period_end_points(benchmark_points, rebalance_frequency)]
    if len(schedule) < 2:
        return _empty_harvest_simulation("not enough rebalance dates", rebalance_frequency)

    tickers = {candidate.ticker for candidate in universe}.union(candidate.ticker for candidate in selected)
    table = _price_table(tickers, prices, schedule)
    candidate_scores = {
        candidate.ticker: candidate_score(
            candidate,
            error_margin,
            target_tax_alpha,
            benchmark_returns,
            tax_alpha_mode,
        )
        for candidate in universe
    }
    selected_pairs = [
        (candidate, max(0.0, weight))
        for candidate, weight in zip(selected, weights)
        if weight > 0
    ]
    if not selected_pairs:
        return _empty_harvest_simulation("portfolio has no positive weights", rebalance_frequency)

    start_day = None
    for day in schedule:
        if all(table.get(candidate.ticker, {}).get(day, 0.0) > 0 for candidate, _ in selected_pairs):
            start_day = day
            break
    if start_day is None:
        return _empty_harvest_simulation("selected positions do not share a priced start date", rebalance_frequency)

    total_weight = sum(weight for _, weight in selected_pairs)
    lots: List[SimulatedHarvestLot] = []
    for candidate, weight in selected_pairs:
        price = table[candidate.ticker][start_day]
        allocation = weight / total_weight
        lots.append(
            SimulatedHarvestLot(
                ticker=candidate.ticker,
                sector=candidate.sector,
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
    skipped_no_replacement = 0
    skipped_nonpositive_net_benefit = 0
    banned_until: Dict[str, date] = {}
    portfolio_values: List[float] = []
    period_rows: Dict[date, Dict[str, float]] = {}
    sample_events: List[Dict[str, object]] = []

    for day_index, day in enumerate(simulation_dates):
        portfolio_value = 0.0
        for lot in lots:
            price = table.get(lot.ticker, {}).get(day)
            if price is not None:
                portfolio_value += lot.shares * price
        portfolio_values.append(portfolio_value)

        if day_index == 0:
            continue

        next_lots: List[SimulatedHarvestLot] = []
        sold_this_period: Set[str] = set()
        for lot_index, lot in enumerate(lots):
            price = table.get(lot.ticker, {}).get(day)
            if price is None or price <= 0 or lot.basis <= 0:
                next_lots.append(lot)
                continue
            market_value = lot.shares * price
            unrealized_return = market_value / lot.basis - 1.0
            realized_loss = lot.basis - market_value
            if unrealized_return > -harvest_threshold_pct or realized_loss <= 0:
                next_lots.append(lot)
                continue

            tax_benefit = realized_loss * tax_rate
            transaction_cost = 2.0 * cost_rate * market_value
            replacement_cost = replacement_cost_rate * market_value
            net_tax_benefit = tax_benefit - transaction_cost - replacement_cost
            if net_tax_benefit <= 0:
                skipped_nonpositive_net_benefit += 1
                next_lots.append(lot)
                continue

            active_tickers = {other.ticker for idx, other in enumerate(lots) if idx != lot_index}
            active_tickers.update(other.ticker for other in next_lots)
            banned_tickers = {
                ticker
                for ticker, banned_day in banned_until.items()
                if banned_day >= day
            }
            unavailable = active_tickers.union(banned_tickers, sold_this_period, {lot.ticker})
            replacements = _same_sector_replacements(
                universe,
                lot.sector,
                unavailable,
                table,
                day,
                candidate_scores,
                replacement_count,
            )
            if not replacements:
                relaxed_unavailable = banned_tickers.union(sold_this_period, {lot.ticker})
                replacements = _same_sector_replacements(
                    universe,
                    lot.sector,
                    relaxed_unavailable,
                    table,
                    day,
                    candidate_scores,
                    replacement_count,
                )
            allocation_plan: List[Tuple[Candidate, float, float]] = []
            for replacement, allocation in _replacement_allocations(replacements, market_value):
                replacement_price = table.get(replacement.ticker, {}).get(day)
                if replacement_price and replacement_price > 0 and allocation > 0:
                    allocation_plan.append((replacement, allocation, replacement_price))
            if not allocation_plan:
                skipped_no_replacement += 1
                next_lots.append(lot)
                continue

            total_realized_loss += realized_loss
            total_tax_benefit += tax_benefit
            total_transaction_cost += transaction_cost
            total_replacement_cost += replacement_cost
            harvest_count += 1
            sold_this_period.add(lot.ticker)
            banned_until[lot.ticker] = day + timedelta(days=wash_sale_days)
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
                next_lots.append(
                    SimulatedHarvestLot(
                        ticker=replacement.ticker,
                        sector=replacement.sector,
                        shares=allocation / replacement_price,
                        basis=allocation,
                        purchase_day=day,
                    )
                )
                replacement_event_rows.append(
                    {
                        "ticker": replacement.ticker,
                        "value": allocation,
                        "price": replacement_price,
                    }
                )
            if len(sample_events) < sample_event_limit:
                sample_events.append(
                    {
                        "date": day.isoformat(),
                        "sold": lot.ticker,
                        "sector": lot.sector,
                        "basis": lot.basis,
                        "market_value": market_value,
                        "realized_loss": realized_loss,
                        "unrealized_return": unrealized_return,
                        "replacements": replacement_event_rows,
                    }
                )
        lots = next_lots

    end_day = simulation_dates[-1]
    years = max((end_day - start_day).days / 365.25, 1.0 / FREQUENCIES[rebalance_frequency])
    average_portfolio_value = sum(portfolio_values) / len(portfolio_values) if portfolio_values else 0.0
    ending_portfolio_value = portfolio_values[-1] if portfolio_values else 0.0
    total_net_tax_benefit = total_tax_benefit - total_transaction_cost - total_replacement_cost
    if average_portfolio_value > 0:
        portfolio_realized_loss_rate = total_realized_loss / average_portfolio_value / years
        portfolio_simulated_tax_alpha = total_net_tax_benefit / average_portfolio_value / years
    else:
        portfolio_realized_loss_rate = 0.0
        portfolio_simulated_tax_alpha = 0.0

    return {
        "status": "ok",
        "frequency": rebalance_frequency,
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
        "portfolio_simulated_tax_alpha": portfolio_simulated_tax_alpha,
        "harvest_count": harvest_count,
        "skipped_no_replacement": skipped_no_replacement,
        "skipped_nonpositive_net_benefit": skipped_nonpositive_net_benefit,
        "tax_rate": tax_rate,
        "harvest_threshold_pct": harvest_threshold_pct,
        "transaction_cost_bps": transaction_cost_bps,
        "replacement_cost_bps": replacement_cost_bps,
        "replacement_count": replacement_count,
        "wash_sale_days": wash_sale_days,
        "period_realized_losses": [
            {
                "date": day.isoformat(),
                "realized_loss": row["realized_loss"],
                "tax_benefit": row["tax_benefit"],
                "transaction_cost": row["transaction_cost"],
                "replacement_cost": row["replacement_cost"],
                "net_tax_benefit": row["net_tax_benefit"],
                "harvest_count": int(row["harvest_count"]),
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
    show_progress: bool = False,
    progress_label: str = "Optimization",
) -> Dict[str, object]:
    if benchmark_returns is None:
        raise ValueError("benchmark_returns are required to optimize error margin")
    targets = sector_targets(holdings) if match_sectors else None
    initial = initial_selection(
        candidates,
        sample_size,
        error_margin,
        target_tax_alpha,
        benchmark_returns,
        match_sectors,
        targets,
        tax_alpha_mode=tax_alpha_mode,
        index_weight_priority=False,
    )
    selected = optimize_selection(
        candidates,
        initial,
        sample_size,
        error_margin,
        target_tax_alpha,
        benchmark_returns,
        targets,
        match_sectors,
        tax_alpha_mode=tax_alpha_mode,
        iterations=selection_iterations,
        random_seed=random_seed,
        show_progress=show_progress,
        progress_label=f"{progress_label} selection",
    )
    weights = optimize_weights(
        selected,
        error_margin,
        target_tax_alpha,
        targets if match_sectors else None,
        tax_alpha_mode=tax_alpha_mode,
        iterations=weight_iterations,
        min_weight=min_weight,
        max_weight=max_weight,
        benchmark_returns=benchmark_returns,
        tracking_error_penalty=tracking_error_penalty,
        index_anchor_penalty=index_anchor_penalty,
        show_progress=show_progress,
        progress_label=f"{progress_label} weights",
    )
    tracking_model = prepare_tracking_model(selected, benchmark_returns)
    metrics = portfolio_metrics(selected, weights, tracking_model=tracking_model)
    metrics["active_share"] = active_share(selected, weights, holdings)
    sector_targets_out = targets or {}
    if sector_targets_out:
        metrics["sector_abs_error"] = sector_error(metrics["sectors"], sector_targets_out)

    positions = []
    for candidate, weight in sorted(zip(selected, weights), key=lambda item: item[1], reverse=True):
        positions.append(
            {
                "ticker": candidate.ticker,
                "weight": weight,
                "sector": candidate.sector,
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
            "estimated_tax_loss_alpha": target_tax_alpha,
            "tax_alpha_mode": tax_alpha_mode,
            "min_weight": min_weight,
            "max_weight": max_weight,
            "tracking_error_penalty": tracking_error_penalty,
            "index_anchor_penalty": index_anchor_penalty,
            "tax_metric": tax_metric,
            "tax_assumptions": tax_assumptions or {},
            "rebalance_frequency": rebalance_frequency,
            "sector_match": match_sectors,
        },
        "metrics": metrics,
        "sector_targets": sector_targets_out,
        "positions": positions,
        "replacement_candidates": replacement_candidates(
            candidates,
            selected,
            error_margin,
            target_tax_alpha,
            benchmark_returns,
            tax_alpha_mode=tax_alpha_mode,
        ),
    }
