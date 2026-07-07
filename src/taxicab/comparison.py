from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple, cast

from .data import Holding, PricePoint, sector_targets
from .metrics import cumulative_return, daily_returns
from .metric_registry import (
    OUTPUT_SCHEMA_VERSION,
    build_harvest_replay_delta_metrics,
    build_harvest_replay_summary,
    build_objective_metrics,
    build_pairwise_metrics,
    build_portfolio_comparison_metrics,
    metric_group,
    require_schema_version,
)


ANNUALIZATION = 252.0


@dataclass(frozen=True)
class ReturnSeries:
    values: Dict[date, float]
    missing_tickers: List[str]
    covered_weight: float


def compare_portfolios(
    portfolios: Mapping[str, Mapping[str, object]],
    holdings: Sequence[Holding],
    prices: Mapping[str, Sequence[PricePoint]],
    benchmark_ticker: str,
    harvest_replays: Mapping[str, Mapping[str, object]] | None = None,
) -> Dict[str, object]:
    for label, portfolio in portfolios.items():
        try:
            require_schema_version(portfolio, "portfolio")
        except ValueError as exc:
            raise ValueError(f"{label}: {exc}") from exc

    benchmark = benchmark_ticker.upper()
    if benchmark not in prices:
        raise ValueError(f"benchmark prices missing for {benchmark}")

    index_weights = {holding.ticker: holding.weight for holding in holdings}
    target_sectors = sector_targets(holdings)
    benchmark_returns = daily_returns(prices[benchmark])
    portfolio_returns = {
        label: portfolio_return_series(portfolio, prices)
        for label, portfolio in portfolios.items()
    }

    summaries = {}
    for label, portfolio in portfolios.items():
        returns = portfolio_returns[label]
        summaries[label] = portfolio_summary(
            portfolio,
            returns,
            benchmark_returns,
            index_weights,
            target_sectors,
        )
        if harvest_replays and label in harvest_replays:
            summaries[label]["harvest_replay"] = build_harvest_replay_summary(harvest_replays[label])

    labels = list(portfolios)
    pairs = []
    for idx, left in enumerate(labels):
        for right in labels[idx + 1 :]:
            pair_summary = pairwise_summary(
                left,
                right,
                portfolios[left],
                portfolios[right],
                portfolio_returns[left].values,
                portfolio_returns[right].values,
            )
            if harvest_replays and left in harvest_replays and right in harvest_replays:
                pair_summary["harvest_replay_delta"] = {
                    "left_status": harvest_replays[left].get("status"),
                    "right_status": harvest_replays[right].get("status"),
                    "metrics": build_harvest_replay_delta_metrics(harvest_replays[left], harvest_replays[right]),
                }
            pairs.append(pair_summary)

    return {
        "version": OUTPUT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": benchmark,
        "index_position_count": len(holdings),
        "index_sector_targets": target_sectors,
        "portfolios": summaries,
        "pairwise": pairs,
    }


def harvest_replay_summary(replay: Mapping[str, object]) -> Dict[str, object]:
    return build_harvest_replay_summary(replay)


def pairwise_harvest_replay_summary(
    left_label: str,
    right_label: str,
    left: Mapping[str, object],
    right: Mapping[str, object],
) -> Dict[str, object]:
    return {
        "left": left_label,
        "right": right_label,
        "left_status": left.get("status"),
        "right_status": right.get("status"),
        "metrics": build_harvest_replay_delta_metrics(left, right),
    }


def portfolio_return_series(
    portfolio: Mapping[str, object],
    prices: Mapping[str, Sequence[PricePoint]],
) -> ReturnSeries:
    weights = position_weights(portfolio)
    returns_by_ticker = {}
    missing = []
    for ticker in weights:
        if ticker not in prices:
            missing.append(ticker)
            continue
        returns = daily_returns(prices[ticker])
        if returns:
            returns_by_ticker[ticker] = returns
        else:
            missing.append(ticker)

    covered_weight = sum(weights[ticker] for ticker in returns_by_ticker)
    if not returns_by_ticker:
        return ReturnSeries({}, sorted(missing), covered_weight)

    common_dates = set.intersection(*(set(returns) for returns in returns_by_ticker.values()))
    values = {
        day: sum(weights[ticker] * returns_by_ticker[ticker][day] for ticker in returns_by_ticker)
        for day in sorted(common_dates)
    }
    return ReturnSeries(values, sorted(missing), covered_weight)


def portfolio_summary(
    portfolio: Mapping[str, object],
    returns: ReturnSeries,
    benchmark_returns: Mapping[date, float],
    index_weights: Mapping[str, float],
    target_sectors: Mapping[str, float],
) -> Dict[str, object]:
    weights = position_weights(portfolio)
    sectors = portfolio_sector_weights(portfolio)
    common_dates = sorted(set(returns.values).intersection(benchmark_returns))
    portfolio_values = [returns.values[day] for day in common_dates]
    benchmark_values = [benchmark_returns[day] for day in common_dates]
    return_metrics = return_summary(common_dates, portfolio_values, benchmark_values)

    feature_summary = portfolio_feature_summary(portfolio, weights, sectors, target_sectors, return_metrics)
    construction_metrics = metric_group(portfolio, "construction")
    harvest_metrics = metric_group(portfolio, "harvest_replay")
    comparison_context = {
        "portfolio": portfolio,
        "returns": return_metrics,
        "position_count": len(weights),
        "covered_price_weight": returns.covered_weight,
        "missing_price_tickers": returns.missing_tickers,
        "weighted_overlap_with_index": weighted_overlap(weights, index_weights),
        "active_share_to_index": active_share(weights, index_weights),
        "sector_active_share_to_index": 0.5 * l1_distance(sectors, target_sectors),
        "sector_similarity_to_index": cosine_similarity(sectors, target_sectors),
        "sector_overlap_to_index": weighted_overlap(sectors, target_sectors),
        "construction_tracking_error": _float_or_none(construction_metrics.get("tracking_error")),
        "construction_beta": _float_or_none(construction_metrics.get("beta")),
        "benchmark_tracking_error": _float_or_none(return_metrics.get("benchmark_tracking_error")),
        "benchmark_beta": _float_or_none(return_metrics.get("benchmark_beta")),
        "harvest_replay_metrics": harvest_metrics,
        "effective_names": effective_count(weights.values()),
        "max_weight": max(weights.values()) if weights else 0.0,
        "sector_abs_error": l1_distance(sectors, target_sectors),
        "error_margin": _float_or_none(_mapping(portfolio.get("targets")).get("error_margin")),
    }
    return {
        "metrics": build_portfolio_comparison_metrics(comparison_context),
        "ticker_overlap_with_index": len(set(weights).intersection(index_weights)),
        "sector_weights": sectors,
        "sector_abs_error_to_index": l1_distance(sectors, target_sectors),
        "effective_sector_count": effective_count(sectors.values()),
        "returns": return_metrics,
        "feature_vector": feature_summary,
        "objective_decomposition": {"metrics": objective_decomposition(portfolio, feature_summary)},
    }


def pairwise_summary(
    left_label: str,
    right_label: str,
    left: Mapping[str, object],
    right: Mapping[str, object],
    left_returns: Mapping[date, float],
    right_returns: Mapping[date, float],
) -> Dict[str, object]:
    left_weights = position_weights(left)
    right_weights = position_weights(right)
    left_sectors = portfolio_sector_weights(left)
    right_sectors = portfolio_sector_weights(right)
    common_dates = sorted(set(left_returns).intersection(right_returns))
    left_values = [left_returns[day] for day in common_dates]
    right_values = [right_returns[day] for day in common_dates]

    left_tickers = set(left_weights)
    right_tickers = set(right_weights)
    union = left_tickers.union(right_tickers)
    jaccard = len(left_tickers.intersection(right_tickers)) / len(union) if union else 1.0

    left_factors = factor_exposures(left, left_values)
    right_factors = factor_exposures(right, right_values)
    pair_context = {
        "left": left_label,
        "right": right_label,
        "ticker_overlap_count": len(left_tickers.intersection(right_tickers)),
        "weight_cosine_similarity": cosine_similarity(left_weights, right_weights),
        "ticker_jaccard": jaccard,
        "weighted_overlap": weighted_overlap(left_weights, right_weights),
        "active_share": active_share(left_weights, right_weights),
        "sector_abs_distance": l1_distance(left_sectors, right_sectors),
        "sector_active_share": 0.5 * l1_distance(left_sectors, right_sectors),
        "sector_similarity": cosine_similarity(left_sectors, right_sectors),
        "sector_overlap": weighted_overlap(left_sectors, right_sectors),
        "factor_abs_distance": l1_distance(left_factors, right_factors),
        "tax_lot_action_overlap": tax_lot_action_overlap(left, right),
        "returns": pair_return_summary(common_dates, left_values, right_values),
    }
    return {
        "left": left_label,
        "right": right_label,
        "metrics": build_pairwise_metrics(pair_context),
        "returns": pair_context["returns"],
    }



def portfolio_feature_summary(
    portfolio: Mapping[str, object],
    weights: Mapping[str, float],
    sectors: Mapping[str, float],
    target_sectors: Mapping[str, float],
    return_metrics: Mapping[str, object],
) -> Dict[str, object]:
    construction = metric_group(portfolio, "construction")
    harvest = metric_group(portfolio, "harvest_replay")
    targets = _mapping(portfolio.get("targets"))
    features: Dict[str, object] = {
        "tracking_error": _float_or_none(construction.get("tracking_error"))
        or _float_or_none(return_metrics.get("benchmark_tracking_error"))
        or 0.0,
        "beta": _float_or_none(construction.get("beta")) or _float_or_none(return_metrics.get("benchmark_beta")) or 0.0,
        "tax_alpha": _float_or_none(construction.get("tax_alpha")) or 0.0,
        "simulated_tax_alpha": _float_or_none(construction.get("simulated_tax_alpha")) or 0.0,
        "realized_loss_rate": _float_or_none(harvest.get("portfolio_realized_loss_rate")) or 0.0,
        "turnover": _float_or_none(harvest.get("total_transaction_cost")) or 0.0,
        "effective_names": effective_count(weights.values()),
        "max_weight": max(weights.values()) if weights else 0.0,
        "active_share_to_index": _float_or_none(construction.get("active_share")) or 0.0,
        "sector_abs_error": l1_distance(sectors, target_sectors),
        "tracking_constraint_slack": max(
            (_float_or_none(targets.get("error_margin")) or 0.0)
            - (
                _float_or_none(construction.get("tracking_error"))
                or _float_or_none(return_metrics.get("benchmark_tracking_error"))
                or 0.0
            ),
            0.0,
        ),
    }
    for sector, target in target_sectors.items():
        features[f"sector_drift:{sector}"] = sectors.get(sector, 0.0) - target
    features.update({f"factor:{key}": value for key, value in factor_exposures(portfolio, []).items()})
    return features


def factor_exposures(portfolio: Mapping[str, object], returns: Sequence[float]) -> Dict[str, float]:
    construction = metric_group(portfolio, "construction")
    harvest = metric_group(portfolio, "harvest_replay")
    explicit = _mapping(construction.get("factor_exposures")) or _mapping(portfolio.get("factor_exposures"))
    if explicit:
        return {str(key): value for key, raw in explicit.items() if (value := _float_or_none(raw)) is not None}
    exposures = {}
    for source, source_key, factor_key in (
        (construction, "beta", "beta"),
        (harvest, "portfolio_harvest_beta", "harvest_beta"),
        (construction, "tracking_error", "tracking_error"),
        (harvest, "portfolio_harvest_tracking_error", "harvest_tracking_error"),
        (construction, "effective_number_of_names", "effective_names"),
    ):
        value = _float_or_none(source.get(source_key))
        if value is not None:
            exposures[factor_key] = value
    if returns:
        exposures["volatility"] = annualized_std(returns)
    return exposures


def tax_lot_action_overlap(left: Mapping[str, object], right: Mapping[str, object]) -> float | None:
    left_actions = tax_lot_action_tickers(left)
    right_actions = tax_lot_action_tickers(right)
    if not left_actions and not right_actions:
        return None
    union = left_actions.union(right_actions)
    return len(left_actions.intersection(right_actions)) / len(union) if union else None


def tax_lot_action_tickers(portfolio: Mapping[str, object]) -> set[str]:
    replay = _mapping(portfolio.get("portfolio_harvest_simulation"))
    tickers: set[str] = set()
    for key in ("harvests", "actions", "tax_lot_actions"):
        rows = replay.get(key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            for field in ("ticker", "sold_ticker", "harvested_ticker", "replacement_ticker"):
                raw = cast(Mapping[str, object], row).get(field)
                if raw:
                    tickers.add(str(raw).upper())
    return tickers


def objective_decomposition(portfolio: Mapping[str, object], features: Mapping[str, object]) -> Dict[str, float]:
    harvest = metric_group(portfolio, "harvest_replay")
    targets = _mapping(portfolio.get("targets"))
    context = {
        "target_tax_alpha": _float_or_none(targets.get("estimated_tax_loss_alpha"))
        or _float_or_none(targets.get("target_tax_alpha"))
        or 0.0,
        "tax_alpha_mode": targets.get("tax_alpha_mode"),
        "error_margin": _float_or_none(targets.get("error_margin")) or 0.005,
        "tracking_error": _float_or_none(features.get("tracking_error")) or 0.0,
        "sector_abs_error": _float_or_none(features.get("sector_abs_error")) or 0.0,
        "beta": _float_or_none(features.get("beta")) or 1.0,
        "max_weight": _float_or_none(features.get("max_weight")) or 0.0,
        "tax_alpha": _float_or_none(features.get("tax_alpha")) or 0.0,
        "total_transaction_cost": _float_or_none(harvest.get("total_transaction_cost")) or 0.0,
        "skipped_constraint_violation": _float_or_none(harvest.get("skipped_constraint_violation")) or 0.0,
        "skipped_no_replacement": _float_or_none(harvest.get("skipped_no_replacement")) or 0.0,
        "cash_penalty": abs(1.0 - sum(position_weights(portfolio).values())),
    }
    return {key: float(value) for key, value in build_objective_metrics(context).items() if isinstance(value, (int, float))}


def _mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return cast(Mapping[str, object], value)
    return {}


def position_weights(portfolio: Mapping[str, object]) -> Dict[str, float]:
    positions = portfolio.get("positions", [])
    if not isinstance(positions, list):
        raise ValueError("portfolio positions must be a list")
    weights: Dict[str, float] = {}
    for item in positions:
        if not isinstance(item, dict):
            continue
        position = cast(Mapping[str, object], item)
        ticker = str(position.get("ticker", "")).upper()
        if not ticker:
            continue
        weight = _float_value(position.get("weight", 0.0))
        weights[ticker] = weights.get(ticker, 0.0) + weight
    return weights


def portfolio_sector_weights(portfolio: Mapping[str, object]) -> Dict[str, float]:
    positions = portfolio.get("positions", [])
    if not isinstance(positions, list):
        raise ValueError("portfolio positions must be a list")
    sectors: Dict[str, float] = {}
    for item in positions:
        if not isinstance(item, dict):
            continue
        position = cast(Mapping[str, object], item)
        sector = str(position.get("sector") or "Unknown")
        weight = _float_value(position.get("weight", 0.0))
        sectors[sector] = sectors.get(sector, 0.0) + weight
    return sectors


def return_summary(
    dates: Sequence[date],
    values: Sequence[float],
    benchmark_values: Sequence[float],
) -> Dict[str, object]:
    summary = standalone_return_summary(dates, values)
    if len(values) != len(benchmark_values):
        raise ValueError("portfolio and benchmark returns must have the same length")
    active = [value - benchmark for value, benchmark in zip(values, benchmark_values)]
    summary.update(
        {
            "benchmark_observations": len(benchmark_values),
            "benchmark_correlation": correlation(values, benchmark_values),
            "benchmark_beta": beta(values, benchmark_values),
            "benchmark_tracking_error": annualized_std(active),
            "benchmark_annualized_active_return": mean(active) * ANNUALIZATION if active else 0.0,
            "benchmark_cumulative_return": cumulative_return(benchmark_values),
        }
    )
    return summary


def pair_return_summary(
    dates: Sequence[date],
    left_values: Sequence[float],
    right_values: Sequence[float],
) -> Dict[str, object]:
    left_cumulative_return = cumulative_return(left_values)
    right_cumulative_return = cumulative_return(right_values)
    active = [left - right for left, right in zip(left_values, right_values)]
    return {
        "observations": len(left_values),
        "start": dates[0].isoformat() if dates else None,
        "end": dates[-1].isoformat() if dates else None,
        "correlation": correlation(left_values, right_values),
        "left_beta_to_right": beta(left_values, right_values),
        "tracking_error": annualized_std(active),
        "annualized_active_return": mean(active) * ANNUALIZATION if active else 0.0,
        "left_cumulative_return": left_cumulative_return,
        "right_cumulative_return": right_cumulative_return,
        "cumulative_return_difference": left_cumulative_return - right_cumulative_return,
    }


def standalone_return_summary(dates: Sequence[date], values: Sequence[float]) -> Dict[str, object]:
    total = cumulative_return(values)
    years = year_span(dates, values)
    annualized = (1.0 + total) ** (1.0 / years) - 1.0 if total > -1.0 and years > 0 else -1.0
    return {
        "observations": len(values),
        "start": dates[0].isoformat() if dates else None,
        "end": dates[-1].isoformat() if dates else None,
        "years": years,
        "cumulative_return": total,
        "annualized_return": annualized,
        "annualized_volatility": annualized_std(values),
        "max_drawdown": max_drawdown(values),
    }


def year_span(dates: Sequence[date], values: Sequence[float]) -> float:
    if len(dates) >= 2:
        return max((dates[-1] - dates[0]).days / 365.25, 1.0 / ANNUALIZATION)
    if values:
        return max(len(values) / ANNUALIZATION, 1.0 / ANNUALIZATION)
    return 0.0


def max_drawdown(values: Sequence[float]) -> float:
    wealth = 1.0
    peak = 1.0
    worst = 0.0
    for value in values:
        wealth *= 1.0 + value
        peak = max(peak, wealth)
        if peak > 0:
            worst = min(worst, wealth / peak - 1.0)
    return worst


def weighted_overlap(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    return sum(min(left.get(key, 0.0), right.get(key, 0.0)) for key in set(left).union(right))


def active_share(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    return 0.5 * l1_distance(left, right)


def l1_distance(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    return sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in set(left).union(right))


def cosine_similarity(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    keys = set(left).union(right)
    left_norm = math.sqrt(sum(left.get(key, 0.0) ** 2 for key in keys))
    right_norm = math.sqrt(sum(right.get(key, 0.0) ** 2 for key in keys))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    dot = sum(left.get(key, 0.0) * right.get(key, 0.0) for key in keys)
    return dot / (left_norm * right_norm)


def effective_count(values: Iterable[float]) -> float:
    squared = sum(value * value for value in values)
    return 1.0 / squared if squared > 0 else 0.0


def annualized_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return math.sqrt(sample_variance(values) * ANNUALIZATION)


def correlation(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("return series lengths must match")
    if len(left) < 2:
        return 0.0
    left_var = sample_variance(left)
    right_var = sample_variance(right)
    if left_var <= 0 or right_var <= 0:
        return 0.0
    return sample_covariance(left, right) / math.sqrt(left_var * right_var)


def beta(asset: Sequence[float], benchmark: Sequence[float]) -> float:
    if len(asset) != len(benchmark):
        raise ValueError("return series lengths must match")
    var = sample_variance(benchmark)
    if var <= 0:
        return 0.0
    return sample_covariance(asset, benchmark) / var


def sample_covariance(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("series lengths must match")
    if len(left) < 2:
        return 0.0
    left_mean = mean(left)
    right_mean = mean(right)
    return sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right)) / (len(left) - 1)


def sample_variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    average = mean(values)
    return sum((value - average) ** 2 for value in values) / (len(values) - 1)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _float_value(value: object) -> float:
    if isinstance(value, (int, float, str)):
        return float(value)
    raise TypeError(f"expected a numeric value, got {type(value).__name__}")


def _float_or_none(value: object) -> float | None:
    if not isinstance(value, (int, float, str)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_labeled_path(value: str) -> Tuple[str, Path]:
    if "=" in value:
        label, raw_path = value.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"portfolio label is empty: {value}")
        return label, Path(raw_path)
    path = Path(value)
    return path.stem, path
