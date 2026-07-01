from __future__ import annotations

import csv
import json
import math
import os
import io
import time
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET


USER_AGENT = "taxicab/0.1"
XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


@dataclass(frozen=True)
class Holding:
    ticker: str
    weight: float
    sector: str = "Unknown"


@dataclass(frozen=True)
class PricePoint:
    day: date
    adj_close: float


def parse_date(value: str) -> date:
    return datetime.strptime(value.strip(), "%Y-%m-%d").date()


def today_utc() -> date:
    return datetime.now(timezone.utc).date()


def normalize_ticker(value: str) -> str:
    return value.strip().upper()


def normalize_sector(value: str) -> str:
    sector = (value or "").strip()
    if not sector or sector.upper() in {"-", "N/A", "NA", "NONE"}:
        return "Unknown"
    return sector


def yahoo_symbol(ticker: str) -> str:
    return normalize_ticker(ticker).replace(".", "-")


def normalize_weight(value: str) -> float:
    text = (value or "").strip().replace("%", "")
    if not text:
        return 0.0
    text = text.replace(",", "")
    number = float(text)
    if number > 1.0:
        return number / 100.0
    return number


def _column_lookup(fieldnames: Sequence[str]) -> Dict[str, str]:
    return {name.strip().lower(): name for name in fieldnames}


def _first_column(columns: Dict[str, str], names: Sequence[str]) -> Optional[str]:
    for name in names:
        if name in columns:
            return columns[name]
    return None


def read_holdings_csv(path: os.PathLike[str] | str) -> List[Holding]:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return parse_holdings_csv(handle.read())


def parse_holdings_csv(text: str) -> List[Holding]:
    sample = _holdings_csv_body(text)
    if not sample:
        raise ValueError("holdings CSV is empty")

    reader = csv.DictReader(sample.splitlines())
    if not reader.fieldnames:
        raise ValueError("holdings CSV has no header row")

    columns = _column_lookup(reader.fieldnames)
    ticker_col = _first_column(
        columns,
        [
            "ticker",
            "symbol",
            "holding_ticker",
            "holding ticker",
            "ticker symbol",
        ],
    )
    if not ticker_col:
        raise ValueError("holdings CSV needs a ticker column")

    weight_col = _first_column(
        columns,
        [
            "weight",
            "weight_pct",
            "weight (%)",
            "% weight",
            "portfolio_weight",
            "portfolio weight",
            "index weight",
        ],
    )
    sector_col = _first_column(
        columns,
        [
            "sector",
            "gics_sector",
            "gics sector",
            "morningstar_sector",
            "morningstar sector",
        ],
    )

    holdings: List[Holding] = []
    for row in reader:
        ticker = normalize_ticker(row.get(ticker_col, ""))
        if not ticker or ticker in {"-", "N/A", "CASH", "USD"}:
            continue
        raw_weight = row.get(weight_col, "") if weight_col else ""
        weight = normalize_weight(raw_weight) if weight_col else 0.0
        sector = normalize_sector(row.get(sector_col, "") if sector_col else "")
        holdings.append(Holding(ticker=ticker, weight=weight, sector=sector))

    if not holdings:
        raise ValueError("holdings CSV did not contain any usable tickers")

    if all(h.weight <= 0 for h in holdings):
        equal_weight = 1.0 / len(holdings)
        return [Holding(h.ticker, equal_weight, h.sector) for h in holdings]

    total = sum(max(0.0, h.weight) for h in holdings)
    if total <= 0:
        raise ValueError("holdings CSV weights sum to zero")
    return [Holding(h.ticker, max(0.0, h.weight) / total, h.sector) for h in holdings]


def _holdings_csv_body(text: str) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    ticker_names = ["ticker", "symbol", "holding_ticker", "holding ticker", "ticker symbol"]
    for idx, line in enumerate(lines[:100]):
        try:
            fields = next(csv.reader([line]))
        except csv.Error:
            continue
        columns = _column_lookup(fields)
        if _first_column(columns, ticker_names):
            return "\n".join(lines[idx:])
    return text.lstrip()


def write_holdings_csv(holdings: Sequence[Holding], path: os.PathLike[str] | str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ticker", "weight", "sector"])
        writer.writeheader()
        for holding in holdings:
            writer.writerow(
                {
                    "ticker": holding.ticker,
                    "weight": f"{holding.weight:.12f}",
                    "sector": holding.sector,
                }
            )


def read_holdings_xlsx(path: os.PathLike[str] | str) -> List[Holding]:
    with open(path, "rb") as handle:
        return parse_holdings_xlsx(handle.read())


def parse_holdings_xlsx(data: bytes) -> List[Holding]:
    rows = _xlsx_rows(data)
    output = io.StringIO()
    writer = csv.writer(output)
    for row in rows:
        writer.writerow(row)
    return parse_holdings_csv(output.getvalue())


def _xlsx_rows(data: bytes) -> List[List[str]]:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = set(archive.namelist())
        shared_strings = _xlsx_shared_strings(archive) if "xl/sharedStrings.xml" in names else []
        sheet_name = "xl/worksheets/sheet1.xml"
        if sheet_name not in names:
            sheet_name = sorted(name for name in names if name.startswith("xl/worksheets/sheet"))[0]
        root = ET.fromstring(archive.read(sheet_name))

    rows: List[List[str]] = []
    for row_el in root.iter(f"{XLSX_NS}row"):
        values: List[str] = []
        for cell in row_el.findall(f"{XLSX_NS}c"):
            index = _xlsx_column_index(cell.attrib.get("r", ""))
            while len(values) <= index:
                values.append("")
            values[index] = _xlsx_cell_value(cell, shared_strings)
        if any(value for value in values):
            rows.append(values)
    return rows


def _xlsx_shared_strings(archive: zipfile.ZipFile) -> List[str]:
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    strings: List[str] = []
    for item in root.findall(f"{XLSX_NS}si"):
        strings.append("".join(text.text or "" for text in item.iter(f"{XLSX_NS}t")))
    return strings


def _xlsx_column_index(reference: str) -> int:
    letters = "".join(char for char in reference if char.isalpha())
    if not letters:
        return 0
    value = 0
    for char in letters:
        value = value * 26 + ord(char.upper()) - 64
    return value - 1


def _xlsx_cell_value(cell: ET.Element, shared_strings: Sequence[str]) -> str:
    cell_type = cell.attrib.get("t")
    value = cell.find(f"{XLSX_NS}v")
    if cell_type == "s" and value is not None:
        return shared_strings[int(value.text or "0")]
    if cell_type == "inlineStr":
        inline = cell.find(f"{XLSX_NS}is")
        if inline is not None:
            return "".join(text.text or "" for text in inline.iter(f"{XLSX_NS}t"))
    if value is not None:
        return value.text or ""
    return ""


def download_text(url: str, timeout: int = 30) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(content_type, errors="replace")


def download_bytes(url: str, timeout: int = 30) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def load_holdings_source(
    csv_path: Optional[str],
    csv_url: Optional[str],
    xlsx_path: Optional[str] = None,
    xlsx_url: Optional[str] = None,
) -> List[Holding]:
    sources = [csv_path, csv_url, xlsx_path, xlsx_url]
    if sum(1 for source in sources if source) != 1:
        raise ValueError("provide exactly one holdings source")
    if csv_path:
        return read_holdings_csv(csv_path)
    if csv_url:
        return parse_holdings_csv(download_text(csv_url))
    if xlsx_path:
        return read_holdings_xlsx(xlsx_path)
    if xlsx_url:
        return parse_holdings_xlsx(download_bytes(xlsx_url))
    raise ValueError("provide a holdings source")


def read_price_series_csv(path: os.PathLike[str] | str) -> List[PricePoint]:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no header row")
        columns = _column_lookup(reader.fieldnames)
        date_col = _first_column(columns, ["date", "day"])
        price_col = _first_column(columns, ["adj_close", "adjusted close", "close", "price"])
        if not date_col or not price_col:
            raise ValueError(f"{path} needs date and adj_close columns")
        points = []
        for row in reader:
            raw_price = (row.get(price_col) or "").strip().replace(",", "")
            if not raw_price:
                continue
            price = float(raw_price)
            if price > 0 and math.isfinite(price):
                points.append(PricePoint(parse_date(row[date_col]), price))
        return sorted(points, key=lambda p: p.day)


def download_yahoo_prices(
    ticker: str,
    start: date,
    end: date,
    price_field: str = "close",
    timeout: int = 30,
) -> List[PricePoint]:
    if price_field not in {"close", "adj_close"}:
        raise ValueError("price_field must be close or adj_close")
    start_dt = datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc)
    end_dt = datetime.combine(end + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
    query = urllib.parse.urlencode(
        {
            "period1": int(start_dt.timestamp()),
            "period2": int(end_dt.timestamp()),
            "interval": "1d",
            "events": "history",
            "includeAdjustedClose": "true",
        }
    )
    symbol = urllib.parse.quote(yahoo_symbol(ticker))
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))

    chart = payload.get("chart", {})
    error = chart.get("error")
    if error:
        raise RuntimeError(f"Yahoo price fetch failed for {ticker}: {error}")
    results = chart.get("result") or []
    if not results:
        raise RuntimeError(f"Yahoo price fetch returned no data for {ticker}")

    result = results[0]
    timestamps = result.get("timestamp") or []
    indicators = result.get("indicators", {})
    close = ((indicators.get("quote") or [{}])[0]).get("close") or []
    adj = ((indicators.get("adjclose") or [{}])[0]).get("adjclose") or []

    points: List[PricePoint] = []
    for idx, timestamp in enumerate(timestamps):
        primary = close if price_field == "close" else adj
        fallback = adj if price_field == "close" else close
        price = primary[idx] if idx < len(primary) else None
        if price is None and idx < len(fallback):
            price = fallback[idx]
        if price is None:
            continue
        price = float(price)
        if price <= 0 or not math.isfinite(price):
            continue
        day = datetime.fromtimestamp(timestamp, tz=timezone.utc).date()
        points.append(PricePoint(day=day, adj_close=price))
    return points


def download_yahoo_sector(ticker: str, timeout: int = 20) -> Optional[str]:
    symbol = urllib.parse.quote(yahoo_symbol(ticker))
    url = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{symbol}?modules=assetProfile"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))

    result = (payload.get("quoteSummary", {}).get("result") or [{}])[0]
    profile = result.get("assetProfile") or {}
    sector = profile.get("sector")
    if isinstance(sector, str) and sector.strip():
        return sector.strip()
    return None


def enrich_sectors(
    holdings: Sequence[Holding],
    source: str = "yahoo",
    pause_seconds: float = 0.05,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> Tuple[List[Holding], Dict[str, str]]:
    if source == "none":
        if on_progress:
            on_progress(len(holdings), len(holdings))
        return list(holdings), {}
    if source != "yahoo":
        raise ValueError(f"unsupported sector source: {source}")

    enriched: List[Holding] = []
    failures: Dict[str, str] = {}
    total = len(holdings)
    for idx, holding in enumerate(holdings, start=1):
        if holding.sector and holding.sector != "Unknown":
            enriched.append(holding)
            if on_progress:
                on_progress(idx, total)
            continue
        try:
            sector = download_yahoo_sector(holding.ticker)
        except Exception as exc:  # pragma: no cover - network dependent
            sector = None
            failures[holding.ticker] = str(exc)
        enriched.append(
            Holding(
                ticker=holding.ticker,
                weight=holding.weight,
                sector=sector or holding.sector or "Unknown",
            )
        )
        if pause_seconds > 0:
            time.sleep(pause_seconds)
        if on_progress:
            on_progress(idx, total)
    return enriched, failures


def write_prices_csv(
    prices_by_ticker: Dict[str, Sequence[PricePoint]],
    path: os.PathLike[str] | str,
) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "ticker", "adj_close"])
        writer.writeheader()
        for ticker in sorted(prices_by_ticker):
            for point in sorted(prices_by_ticker[ticker], key=lambda p: p.day):
                writer.writerow(
                    {
                        "date": point.day.isoformat(),
                        "ticker": ticker,
                        "adj_close": f"{point.adj_close:.8f}",
                    }
                )


def read_prices_csv(path: os.PathLike[str] | str) -> Dict[str, List[PricePoint]]:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no header row")
        columns = _column_lookup(reader.fieldnames)
        date_col = _first_column(columns, ["date", "day"])
        ticker_col = _first_column(columns, ["ticker", "symbol"])
        price_col = _first_column(columns, ["adj_close", "adjusted close", "close", "price"])
        if not date_col or not ticker_col or not price_col:
            raise ValueError(f"{path} needs date, ticker, and adj_close columns")

        prices: Dict[str, List[PricePoint]] = {}
        for row in reader:
            ticker = normalize_ticker(row.get(ticker_col, ""))
            if not ticker:
                continue
            raw_price = (row.get(price_col) or "").strip().replace(",", "")
            if not raw_price:
                continue
            price = float(raw_price)
            if price <= 0 or not math.isfinite(price):
                continue
            prices.setdefault(ticker, []).append(PricePoint(parse_date(row[date_col]), price))

    for ticker, points in prices.items():
        prices[ticker] = sorted(points, key=lambda p: p.day)
    return prices


def write_json(data: object, path: os.PathLike[str] | str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_json(path: os.PathLike[str] | str) -> object:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def cache_paths(data_dir: os.PathLike[str] | str) -> Dict[str, Path]:
    root = Path(data_dir)
    return {
        "root": root,
        "holdings": root / "holdings.csv",
        "prices": root / "prices.csv",
        "metadata": root / "metadata.json",
    }


def write_cache(
    data_dir: os.PathLike[str] | str,
    holdings: Sequence[Holding],
    prices_by_ticker: Dict[str, Sequence[PricePoint]],
    metadata: Dict[str, object],
) -> None:
    paths = cache_paths(data_dir)
    paths["root"].mkdir(parents=True, exist_ok=True)
    write_holdings_csv(holdings, paths["holdings"])
    write_prices_csv(prices_by_ticker, paths["prices"])
    write_json(metadata, paths["metadata"])


def read_cache(
    data_dir: os.PathLike[str] | str,
) -> Tuple[List[Holding], Dict[str, List[PricePoint]], Dict[str, object]]:
    paths = cache_paths(data_dir)
    holdings = read_holdings_csv(paths["holdings"])
    prices = read_prices_csv(paths["prices"])
    metadata = read_json(paths["metadata"])
    if not isinstance(metadata, dict):
        raise ValueError("metadata.json must contain an object")
    return holdings, prices, metadata


def date_range_for_years(years: int, end: Optional[date] = None) -> Tuple[date, date]:
    end_day = end or today_utc()
    start_day = end_day - timedelta(days=int(years * 365.25))
    return start_day, end_day


def sector_targets(holdings: Iterable[Holding]) -> Dict[str, float]:
    targets: Dict[str, float] = {}
    for holding in holdings:
        sector = holding.sector or "Unknown"
        targets[sector] = targets.get(sector, 0.0) + holding.weight
    total = sum(targets.values())
    if total <= 0:
        return {}
    return {sector: weight / total for sector, weight in targets.items()}
