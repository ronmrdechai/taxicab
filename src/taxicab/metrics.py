from __future__ import annotations

from datetime import date
from statistics import mean
from typing import Dict, Iterable, List, Sequence, Tuple

from .data import PricePoint


FREQUENCIES = {
    "monthly": 12,
    "quarterly": 4,
    "half-yearly": 2,
    "halfly": 2,
    "annually": 1,
    "yearly": 1,
}


def daily_returns(points: Sequence[PricePoint]) -> Dict[date, float]:
    ordered = sorted(points, key=lambda p: p.day)
    returns: Dict[date, float] = {}
    previous = None
    for point in ordered:
        if previous and previous.adj_close > 0:
            returns[point.day] = point.adj_close / previous.adj_close - 1.0
        previous = point
    return returns


def align_returns(
    asset_returns: Dict[date, float],
    benchmark_returns: Dict[date, float],
) -> Tuple[List[float], List[float]]:
    dates = sorted(set(asset_returns).intersection(benchmark_returns))
    return [asset_returns[day] for day in dates], [benchmark_returns[day] for day in dates]


def covariance(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError("series lengths must match")
    if len(xs) < 2:
        return 0.0
    x_bar = mean(xs)
    y_bar = mean(ys)
    return sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys)) / (len(xs) - 1)


def variance(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    x_bar = mean(xs)
    return sum((x - x_bar) ** 2 for x in xs) / (len(xs) - 1)


def beta_to_benchmark(asset_returns: Dict[date, float], benchmark_returns: Dict[date, float]) -> float:
    xs, ys = align_returns(asset_returns, benchmark_returns)
    var = variance(ys)
    if var == 0:
        return 0.0
    return covariance(xs, ys) / var


def period_key(day: date, frequency: str) -> Tuple[int, int]:
    if frequency == "monthly":
        return day.year, day.month
    if frequency == "quarterly":
        return day.year, (day.month - 1) // 3
    if frequency in {"half-yearly", "halfly"}:
        return day.year, (day.month - 1) // 6
    if frequency in {"annually", "yearly"}:
        return day.year, 0
    raise ValueError(f"unsupported frequency: {frequency}")


def period_end_points(points: Sequence[PricePoint], frequency: str) -> List[PricePoint]:
    if frequency not in FREQUENCIES:
        raise ValueError(f"unsupported frequency: {frequency}")
    by_period: Dict[Tuple[int, int], PricePoint] = {}
    for point in sorted(points, key=lambda p: p.day):
        by_period[period_key(point.day, frequency)] = point
    return [by_period[key] for key in sorted(by_period)]


def period_returns(points: Sequence[PricePoint], frequency: str) -> List[float]:
    endpoints = period_end_points(points, frequency)
    returns: List[float] = []
    for previous, current in zip(endpoints, endpoints[1:]):
        if previous.adj_close > 0:
            returns.append(current.adj_close / previous.adj_close - 1.0)
    return returns


def estimated_tax_loss_alpha(points: Sequence[PricePoint], frequency: str) -> float:
    returns = period_returns(points, frequency)
    if not returns:
        return 0.0
    losses = [max(0.0, -period_return) for period_return in returns]
    return mean(losses) * FREQUENCIES[frequency]


def simulated_tax_alpha(
    points: Sequence[PricePoint],
    frequency: str,
    tax_rate: float = 0.30,
    harvest_threshold_pct: float = 0.05,
    transaction_cost_bps: float = 5.0,
    replacement_cost_bps: float = 10.0,
) -> float:
    endpoints = period_end_points(points, frequency)
    if len(endpoints) < 2:
        return 0.0
    if tax_rate < 0:
        raise ValueError("tax_rate must be nonnegative")
    if harvest_threshold_pct < 0:
        raise ValueError("harvest_threshold_pct must be nonnegative")

    basis = endpoints[0].adj_close
    total_net_benefit = 0.0
    cost_rate = max(transaction_cost_bps, 0.0) / 10000.0
    replacement_cost_rate = max(replacement_cost_bps, 0.0) / 10000.0

    for point in endpoints[1:]:
        value = point.adj_close
        if basis <= 0 or value <= 0:
            continue
        unrealized_return = value / basis - 1.0
        if unrealized_return > -harvest_threshold_pct:
            continue
        realized_loss = basis - value
        tax_benefit = realized_loss * tax_rate
        round_trip_cost = 2.0 * cost_rate * value
        replacement_cost = replacement_cost_rate * value
        net_benefit = tax_benefit - round_trip_cost - replacement_cost
        if net_benefit <= 0:
            continue
        total_net_benefit += net_benefit
        basis = value

    years = max((endpoints[-1].day - endpoints[0].day).days / 365.25, 1.0 / FREQUENCIES[frequency])
    average_capital = mean(point.adj_close for point in endpoints)
    if average_capital <= 0:
        return 0.0
    return total_net_benefit / average_capital / years


def observations_overlap(
    asset_returns: Dict[date, float],
    benchmark_returns: Dict[date, float],
) -> int:
    return len(set(asset_returns).intersection(benchmark_returns))


def cumulative_return(returns: Iterable[float]) -> float:
    value = 1.0
    for item in returns:
        value *= 1.0 + item
    return value - 1.0
