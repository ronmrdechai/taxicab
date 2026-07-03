from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple, cast

from .data import Holding, PricePoint, sector_targets
from .metrics import cumulative_return, daily_returns


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
            summaries[label]["harvest_replay"] = harvest_replay_summary(harvest_replays[label])

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
                pair_summary["harvest_replay_deltas"] = pairwise_harvest_replay_summary(
                    left,
                    right,
                    harvest_replays[left],
                    harvest_replays[right],
                )
            pairs.append(pair_summary)

    return {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": benchmark,
        "index_position_count": len(holdings),
        "index_sector_targets": target_sectors,
        "portfolios": summaries,
        "pairwise": pairs,
    }


def harvest_replay_summary(replay: Mapping[str, object]) -> Dict[str, object]:
    summary_keys = [
        "status",
        "reason",
        "start",
        "end",
        "years",
        "rebalance_frequency",
        "harvest_frequency",
        "portfolio_harvest_annualized_return",
        "benchmark_annualized_return",
        "portfolio_harvest_active_return",
        "portfolio_harvest_tracking_error",
        "portfolio_harvest_tracking_error_annualized_pct",
        "portfolio_harvest_beta",
        "portfolio_harvest_correlation",
        "portfolio_harvest_observations",
        "portfolio_simulated_tax_alpha",
        "portfolio_realized_loss_rate",
        "annual_realized_loss",
        "total_realized_loss",
        "total_tax_benefit",
        "total_transaction_cost",
        "total_replacement_cost",
        "total_net_tax_benefit",
        "realized_loss_rate_pct_per_year",
        "immediate_tax_savings_pct_per_year",
        "immediate_net_tax_savings_pct_per_year",
        "simulated_after_tax_alpha_pct_per_year",
        "terminal_after_tax_wealth_difference_pct",
        "harvest_count",
        "rebalance_count",
        "skipped_no_replacement",
        "skipped_nonpositive_net_benefit",
        "skipped_constraint_violation",
        "skipped_harvests_by_reason",
        "selected_position_count",
        "missing_position_tickers",
    ]
    summary = {key: replay[key] for key in summary_keys if key in replay}
    terminal_pct = _float_or_none(replay.get("terminal_after_tax_wealth_difference_pct"))
    if terminal_pct is not None:
        summary["terminal_after_tax_wealth_difference"] = terminal_pct / 100.0
    return summary


def pairwise_harvest_replay_summary(
    left_label: str,
    right_label: str,
    left: Mapping[str, object],
    right: Mapping[str, object],
) -> Dict[str, object]:
    delta_keys = [
        "portfolio_harvest_annualized_return",
        "benchmark_annualized_return",
        "portfolio_harvest_active_return",
        "portfolio_harvest_tracking_error",
        "portfolio_harvest_beta",
        "portfolio_harvest_correlation",
        "portfolio_simulated_tax_alpha",
        "portfolio_realized_loss_rate",
        "annual_realized_loss",
        "total_realized_loss",
        "total_tax_benefit",
        "total_transaction_cost",
        "total_replacement_cost",
        "total_net_tax_benefit",
        "harvest_count",
        "rebalance_count",
    ]
    deltas = {}
    for key in delta_keys:
        left_value = _float_or_none(left.get(key))
        right_value = _float_or_none(right.get(key))
        if left_value is not None and right_value is not None:
            deltas[key] = left_value - right_value
    left_terminal = _float_or_none(left.get("terminal_after_tax_wealth_difference_pct"))
    right_terminal = _float_or_none(right.get("terminal_after_tax_wealth_difference_pct"))
    if left_terminal is not None and right_terminal is not None:
        deltas["terminal_after_tax_wealth_difference"] = (left_terminal - right_terminal) / 100.0
        deltas["terminal_after_tax_wealth_difference_pct"] = left_terminal - right_terminal
    return {
        "left": left_label,
        "right": right_label,
        "left_status": left.get("status"),
        "right_status": right.get("status"),
        "left_minus_right": deltas,
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

    return {
        "position_count": len(weights),
        "covered_price_weight": returns.covered_weight,
        "missing_price_tickers": returns.missing_tickers,
        "ticker_overlap_with_index": len(set(weights).intersection(index_weights)),
        "weighted_overlap_with_index": weighted_overlap(weights, index_weights),
        "active_share_to_index": active_share(weights, index_weights),
        "sector_weights": sectors,
        "sector_abs_error_to_index": l1_distance(sectors, target_sectors),
        "sector_active_share_to_index": 0.5 * l1_distance(sectors, target_sectors),
        "sector_similarity_to_index": cosine_similarity(sectors, target_sectors),
        "sector_overlap_to_index": weighted_overlap(sectors, target_sectors),
        "effective_sector_count": effective_count(sectors.values()),
        "returns": return_metrics,
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

    return {
        "left": left_label,
        "right": right_label,
        "ticker_overlap_count": len(left_tickers.intersection(right_tickers)),
        "ticker_jaccard": jaccard,
        "weighted_overlap": weighted_overlap(left_weights, right_weights),
        "active_share": active_share(left_weights, right_weights),
        "sector_abs_distance": l1_distance(left_sectors, right_sectors),
        "sector_active_share": 0.5 * l1_distance(left_sectors, right_sectors),
        "sector_similarity": cosine_similarity(left_sectors, right_sectors),
        "sector_overlap": weighted_overlap(left_sectors, right_sectors),
        "returns": pair_return_summary(common_dates, left_values, right_values),
    }


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
