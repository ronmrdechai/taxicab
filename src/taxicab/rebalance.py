from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, cast


@dataclass(frozen=True)
class CurrentPosition:
    ticker: str
    market_value: float
    shares: Optional[float] = None
    price: Optional[float] = None
    cost_basis: Optional[float] = None

    @property
    def unrealized_gain(self) -> Optional[float]:
        if self.cost_basis is None:
            return None
        return self.market_value - self.cost_basis

    @property
    def unrealized_return(self) -> Optional[float]:
        if self.cost_basis is None or self.cost_basis <= 0:
            return None
        return self.market_value / self.cost_basis - 1.0


@dataclass(frozen=True)
class Operation:
    action: str
    ticker: str
    value: float
    shares: Optional[float]
    price: Optional[float]
    reason: str
    replacement_for: str = ""


def _float_or_none(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    number = float(text)
    if not math.isfinite(number):
        return None
    return number


def _column_lookup(fieldnames: Sequence[str]) -> Dict[str, str]:
    return {name.strip().lower(): name for name in fieldnames}


def _first_column(columns: Dict[str, str], names: Sequence[str]) -> Optional[str]:
    for name in names:
        if name in columns:
            return columns[name]
    return None


def read_current_positions(path: str) -> List[CurrentPosition]:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no header row")
        columns = _column_lookup(reader.fieldnames)
        ticker_col = _first_column(columns, ["ticker", "symbol"])
        shares_col = _first_column(columns, ["shares", "quantity", "qty"])
        price_col = _first_column(columns, ["price", "last_price", "current_price"])
        value_col = _first_column(columns, ["market_value", "value", "current_value"])
        cost_col = _first_column(columns, ["cost_basis", "total_cost_basis", "cost basis"])
        cost_per_share_col = _first_column(
            columns,
            ["cost_basis_per_share", "cost per share", "average_cost", "avg_cost"],
        )
        if not ticker_col:
            raise ValueError(f"{path} needs a ticker column")

        positions: List[CurrentPosition] = []
        for row in reader:
            ticker = str(row.get(ticker_col, "")).strip().upper()
            if not ticker:
                continue
            shares = _float_or_none(row.get(shares_col)) if shares_col else None
            price = _float_or_none(row.get(price_col)) if price_col else None
            market_value = _float_or_none(row.get(value_col)) if value_col else None
            if market_value is None and shares is not None and price is not None:
                market_value = shares * price
            if price is None and market_value is not None and shares not in (None, 0):
                price = market_value / shares
            if market_value is None:
                raise ValueError(f"position {ticker} needs market_value or shares and price")

            cost_basis = _float_or_none(row.get(cost_col)) if cost_col else None
            cost_per_share = _float_or_none(row.get(cost_per_share_col)) if cost_per_share_col else None
            if cost_basis is None and cost_per_share is not None and shares is not None:
                cost_basis = cost_per_share * shares

            positions.append(
                CurrentPosition(
                    ticker=ticker,
                    market_value=market_value,
                    shares=shares,
                    price=price,
                    cost_basis=cost_basis,
                )
            )
    return positions


def _operation(
    action: str,
    ticker: str,
    value: float,
    price: Optional[float],
    reason: str,
    replacement_for: str = "",
) -> Operation:
    shares = abs(value) / price if price and price > 0 else None
    return Operation(
        action=action,
        ticker=ticker,
        value=abs(value),
        shares=shares,
        price=price,
        reason=reason,
        replacement_for=replacement_for,
    )


def _target_weights(state: Mapping[str, object]) -> Dict[str, float]:
    positions = state.get("positions")
    if not isinstance(positions, list):
        raise ValueError("state file is missing positions")
    weights: Dict[str, float] = {}
    for item in positions:
        if not isinstance(item, dict):
            continue
        position = cast(Mapping[str, object], item)
        ticker = str(position.get("ticker", "")).upper()
        if ticker:
            weights[ticker] = _float_value(position.get("weight", 0.0))
    return weights


def _target_prices(state: Mapping[str, object]) -> Dict[str, float]:
    positions = state.get("positions")
    if not isinstance(positions, list):
        return {}
    prices: Dict[str, float] = {}
    for item in positions:
        if not isinstance(item, dict):
            continue
        position = cast(Mapping[str, object], item)
        ticker = str(position.get("ticker", "")).upper()
        price = _float_or_none(position.get("last_price"))
        if ticker and price and price > 0:
            prices[ticker] = price
    return prices


def _replacement_for(
    state: Mapping[str, object],
    ticker: str,
    unavailable: Iterable[str],
) -> Optional[str]:
    unavailable_set = {item.upper() for item in unavailable}
    replacements = state.get("replacement_candidates", {})
    if not isinstance(replacements, dict):
        return None
    replacement_map = cast(Mapping[str, object], replacements)
    candidates = replacement_map.get(ticker.upper()) or []
    if not isinstance(candidates, list):
        return None
    for item in candidates:
        if not isinstance(item, dict):
            continue
        candidate = cast(Mapping[str, object], item)
        replacement = str(candidate.get("ticker", "")).upper()
        if replacement and replacement not in unavailable_set:
            return replacement
    return None


def _float_value(value: object) -> float:
    if isinstance(value, (int, float, str)):
        return float(value)
    raise TypeError(f"expected a numeric value, got {type(value).__name__}")


def plan_rebalance(
    state: Mapping[str, object],
    current_positions: Sequence[CurrentPosition],
    portfolio_value: Optional[float] = None,
    drift_threshold_pct: float = 0.005,
    harvest_loss_threshold_pct: float = 0.03,
    harvest_loss_threshold_amount: float = 0.0,
) -> List[Operation]:
    target_weights = _target_weights(state)
    target_prices = _target_prices(state)
    total_value = portfolio_value if portfolio_value and portfolio_value > 0 else sum(
        position.market_value for position in current_positions
    )
    if total_value <= 0:
        raise ValueError("portfolio value must be positive")
    threshold_value = total_value * drift_threshold_pct
    current = {position.ticker: position for position in current_positions}
    operations: List[Operation] = []
    harvested: Dict[str, CurrentPosition] = {}

    for position in current_positions:
        unrealized_gain = position.unrealized_gain
        unrealized_return = position.unrealized_return
        if unrealized_gain is None or unrealized_return is None:
            continue
        loss_amount = -unrealized_gain
        if unrealized_return <= -harvest_loss_threshold_pct and loss_amount >= harvest_loss_threshold_amount:
            harvested[position.ticker] = position
            operations.append(
                _operation(
                    "SELL",
                    position.ticker,
                    position.market_value,
                    position.price,
                    "tax-loss harvest",
                )
            )

    unavailable = set(harvested)
    unavailable.update(current)
    for ticker, position in harvested.items():
        replacement = _replacement_for(state, ticker, unavailable)
        if replacement:
            operations.append(
                _operation(
                    "BUY",
                    replacement,
                    position.market_value,
                    None,
                    "same-sector replacement for harvested position",
                    replacement_for=ticker,
                )
            )
            unavailable.add(replacement)

    for ticker, position in current.items():
        if ticker in harvested:
            continue
        if ticker not in target_weights and position.market_value >= threshold_value:
            operations.append(
                _operation(
                    "SELL",
                    ticker,
                    position.market_value,
                    position.price,
                    "not in target direct index",
                )
            )

    for ticker, target_weight in target_weights.items():
        if ticker in harvested:
            continue
        target_value = target_weight * total_value
        position = current.get(ticker)
        current_value = position.market_value if position else 0.0
        delta = target_value - current_value
        if abs(delta) < threshold_value:
            continue
        price = position.price if position else None
        if price is None:
            price = target_prices.get(ticker)
        action = "BUY" if delta > 0 else "SELL"
        reason = "rebalance to target weight"
        operations.append(_operation(action, ticker, delta, price, reason))

    action_order = {"SELL": 0, "BUY": 1}
    operations.sort(key=lambda op: (action_order.get(op.action, 9), op.ticker, op.reason))
    return operations


def write_operations_csv(operations: Sequence[Operation], path: Optional[str]) -> None:
    fieldnames = ["action", "ticker", "value", "shares", "price", "reason", "replacement_for"]
    handle = open(path, "w", newline="", encoding="utf-8") if path else None
    try:
        output = handle if handle is not None else __import__("sys").stdout
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for operation in operations:
            writer.writerow(
                {
                    "action": operation.action,
                    "ticker": operation.ticker,
                    "value": f"{operation.value:.2f}",
                    "shares": "" if operation.shares is None else f"{operation.shares:.6f}",
                    "price": "" if operation.price is None else f"{operation.price:.4f}",
                    "reason": operation.reason,
                    "replacement_for": operation.replacement_for,
                }
            )
    finally:
        if handle is not None:
            handle.close()
