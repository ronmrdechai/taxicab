from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Dict, List, Mapping, Optional, Sequence, cast

from tqdm.auto import tqdm

from .comparison import compare_portfolios, parse_labeled_path
from .data import (
    cache_paths,
    date_range_for_years,
    download_yahoo_prices,
    enrich_sectors,
    holdings_universe,
    latest_historical_holdings,
    load_holdings_source,
    parse_date,
    PricePoint,
    read_historical_holdings_cache,
    read_historical_holdings_csv,
    read_cache,
    read_json,
    read_price_series_csv,
    read_prices_csv,
    write_cache,
    write_json,
)
from .metrics import FREQUENCIES, daily_returns
from .optimizer import build_candidates, construct_portfolio, simulate_portfolio_harvests
from .rebalance import plan_rebalance, read_current_positions, write_operations_csv


REBALANCE_FREQUENCIES = {name for name in FREQUENCIES if name != "daily"}


class Progress:
    def __init__(self, label: str, total: int, every: int = 25) -> None:
        self.total = max(total, 1)
        self.last_count = 0
        self.bar = tqdm(total=self.total, desc=label, unit="item")

    def update(self, count: int, suffix: str = "") -> None:
        delta = max(count - self.last_count, 0)
        if suffix:
            self.bar.set_postfix_str(suffix)
        if delta:
            self.bar.update(delta)
        self.last_count = count

    def done(self, suffix: str = "") -> None:
        if suffix:
            self.bar.set_postfix_str(suffix)
        if self.last_count < self.total:
            self.bar.update(self.total - self.last_count)
        self.bar.close()


def add_construct_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_sector_match: bool = True,
    include_output: bool = True,
) -> None:
    parser.add_argument("--data-dir", required=True, help="Cache directory from the download command.")
    parser.add_argument("--sample-size", type=int, required=True, help="Number of stocks to hold.")
    parser.add_argument(
        "--error-margin",
        type=float,
        required=True,
        help="Annualized tracking-error margin to optimize, for example 0.05 for 5%%.",
    )
    parser.add_argument(
        "--target-tax-alpha",
        type=float,
        required=True,
        help="Target estimated annual tax-loss alpha, for example 0.03 for 3%%.",
    )
    parser.add_argument(
        "--tax-alpha-mode",
        choices=["closest", "at-least"],
        default="closest",
        help="Treat tax alpha as an exact target or a minimum threshold. Default: closest.",
    )
    parser.add_argument(
        "--tax-metric",
        choices=["simulated", "gross"],
        default="simulated",
        help="Tax metric to optimize. Default: simulated.",
    )
    parser.add_argument(
        "--tax-rate",
        type=float,
        default=0.30,
        help="Tax rate applied to harvested losses in simulated mode. Default: 0.30.",
    )
    parser.add_argument(
        "--harvest-threshold-pct",
        type=float,
        default=0.05,
        help="Minimum unrealized loss fraction before harvesting in simulated mode. Default: 0.05.",
    )
    parser.add_argument(
        "--transaction-cost-bps",
        type=float,
        default=5.0,
        help="One-way transaction cost in basis points per harvest leg. Default: 5.",
    )
    parser.add_argument(
        "--replacement-cost-bps",
        type=float,
        default=10.0,
        help="Estimated replacement/tracking cost in basis points per harvest. Default: 10.",
    )
    parser.add_argument(
        "--replacement-count",
        type=int,
        default=2,
        help="Same-sector replacements to buy for each harvested position in the portfolio simulation. Default: 2.",
    )
    parser.add_argument(
        "--wash-sale-days",
        type=int,
        default=31,
        help="Days to avoid repurchasing a harvested ticker in the portfolio simulation. Default: 31.",
    )
    parser.add_argument(
        "--rebalance-frequency",
        choices=sorted(REBALANCE_FREQUENCIES),
        default="quarterly",
        help="Target-weight rebalance cadence. Default: quarterly.",
    )
    parser.add_argument(
        "--harvest-frequency",
        choices=sorted(FREQUENCIES),
        default="daily",
        help="Tax-loss harvesting check cadence used for tax-alpha estimates and simulations. Default: daily.",
    )
    if include_sector_match:
        parser.add_argument("--sector-match", action="store_true", help="Match sector weights to the index.")
    parser.add_argument("--min-observations", type=int, default=252, help="Minimum daily overlap with benchmark.")
    parser.add_argument("--selection-iterations", type=int, default=1000, help="Random local-search iterations.")
    parser.add_argument("--weight-iterations", type=int, default=2000, help="Projected-gradient iterations.")
    parser.add_argument(
        "--min-weight",
        type=float,
        default=0.0001,
        help="Minimum nonzero weight for each selected name. Default: 0.0001.",
    )
    parser.add_argument(
        "--max-weight",
        type=float,
        help="Maximum weight for any selected name, for example 0.02 for 2%%.",
    )
    parser.add_argument(
        "--tracking-error-penalty",
        type=float,
        default=6.0,
        help="Penalty strength for the tracking-error objective. Default: 6.",
    )
    parser.add_argument(
        "--index-anchor-penalty",
        type=float,
        default=25.0,
        help="Penalty for moving away from selected index weights. Default: 25.",
    )
    progress = parser.add_mutually_exclusive_group()
    progress.add_argument(
        "--progress",
        dest="progress",
        action="store_true",
        default=None,
        help="Show tqdm optimization progress bars. Default: auto when stderr is interactive.",
    )
    progress.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="Disable optimization progress bars.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed for repeatable sampling.")
    if include_output:
        parser.add_argument("--output", required=True, help="Portfolio state JSON to write.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="taxicab",
        description="Build and maintain a sampled direct index targeting tracking error and estimated tax-loss alpha.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Download all data into a local cache.")
    download.add_argument("--index", required=True, help="Benchmark/index price symbol, for example SPY.")
    holdings = download.add_mutually_exclusive_group(required=True)
    holdings.add_argument("--holdings-csv", help="Local holdings CSV for the ETF or abstract index.")
    holdings.add_argument("--holdings-url", help="Remote holdings CSV URL for the ETF or abstract index.")
    holdings.add_argument("--holdings-xlsx", help="Local holdings XLSX for the ETF or abstract index.")
    holdings.add_argument("--holdings-xlsx-url", help="Remote holdings XLSX URL for the ETF or abstract index.")
    holdings.add_argument(
        "--historical-holdings-csv",
        help="Point-in-time holdings snapshots CSV. Uses the latest snapshot as current holdings.",
    )
    download.add_argument("--index-prices-csv", help="Optional local benchmark price CSV for abstract indexes.")
    download.add_argument(
        "--prices-csv",
        help="Optional local ticker/date/price CSV for constituents, including delisted names.",
    )
    download.add_argument("--data-dir", required=True, help="Cache directory to create or overwrite.")
    download.add_argument("--years", type=int, default=30, help="Historical years to download. Default: 30.")
    download.add_argument("--start", help="Start date YYYY-MM-DD. Overrides --years when supplied.")
    download.add_argument("--end", help="End date YYYY-MM-DD. Default: today UTC.")
    download.add_argument(
        "--price-field",
        choices=["close", "adj_close"],
        default="close",
        help="Yahoo price field to cache. Use close for price return. Default: close.",
    )
    download.add_argument(
        "--sector-source",
        choices=["yahoo", "none"],
        default="yahoo",
        help="Fill missing sectors from this source. Default: yahoo.",
    )
    download.add_argument(
        "--fail-on-missing-prices",
        action="store_true",
        help="Fail if any constituent price history cannot be downloaded.",
    )

    construct = subparsers.add_parser("construct", help="Construct an optimized direct index portfolio.")
    add_construct_arguments(construct)

    sector_study = subparsers.add_parser(
        "sector-study",
        help="Run matched construct jobs with and without sector matching, then compare them.",
    )
    add_construct_arguments(sector_study, include_sector_match=False, include_output=False)
    sector_study.add_argument(
        "--output-prefix",
        required=True,
        help="Output prefix. Writes *_no_sector.json, *_sector.json, and *_comparison.json.",
    )

    compare = subparsers.add_parser("compare", help="Compare portfolio states against each other and the index.")
    compare.add_argument("--data-dir", required=True, help="Cache directory used to evaluate historical returns.")
    compare.add_argument(
        "--portfolio",
        action="append",
        required=True,
        help="Portfolio JSON path, or label=path. Repeat to compare multiple portfolios.",
    )
    compare.add_argument("--output", help="Comparison JSON to write. Defaults to stdout summary only.")

    rebalance = subparsers.add_parser("rebalance", help="Emit buy/sell suggestions from current holdings.")
    rebalance.add_argument("--state", required=True, help="Portfolio JSON from construct.")
    rebalance.add_argument("--current-csv", required=True, help="Current positions CSV.")
    rebalance.add_argument("--output", help="CSV output path. Defaults to stdout.")
    rebalance.add_argument("--portfolio-value", type=float, help="Override total portfolio value.")
    rebalance.add_argument(
        "--drift-threshold-pct",
        type=float,
        default=0.005,
        help="Ignore rebalance trades below this fraction of portfolio value. Default: 0.005.",
    )
    rebalance.add_argument(
        "--harvest-loss-threshold-pct",
        type=float,
        default=0.03,
        help="Harvest lots below this unrealized return. Default: 0.03.",
    )
    rebalance.add_argument(
        "--harvest-loss-threshold-amount",
        type=float,
        default=0.0,
        help="Minimum dollar loss to harvest. Default: 0.",
    )
    return parser


def command_download(args: argparse.Namespace) -> int:
    requested_end = parse_date(args.end) if args.end else None
    if args.start:
        start = parse_date(args.start)
        end_day = requested_end or date_range_for_years(args.years)[1]
    else:
        start, end_day = date_range_for_years(args.years, requested_end)
    historical_holdings = {}
    if args.historical_holdings_csv:
        historical_holdings = read_historical_holdings_csv(args.historical_holdings_csv)
        holdings = latest_historical_holdings(historical_holdings)
        print(
            "Loaded {} point-in-time snapshots; latest has {} holdings".format(
                len(historical_holdings),
                len(holdings),
            ),
            flush=True,
        )
    else:
        holdings = load_holdings_source(
            args.holdings_csv,
            args.holdings_url,
            args.holdings_xlsx,
            args.holdings_xlsx_url,
        )
    print(f"Loaded {len(holdings)} holdings", flush=True)
    sector_progress = Progress("Sectors", len(holdings))
    holdings, sector_failures = enrich_sectors(
        holdings,
        source=args.sector_source,
        on_progress=lambda count, total: sector_progress.update(count),
    )
    sector_progress.done()
    if args.sector_source != "none":
        print(f"Sector enrichment complete; failures: {len(sector_failures)}", flush=True)

    prices: Dict[str, List[PricePoint]] = read_prices_csv(args.prices_csv) if args.prices_csv else {}
    failures: Dict[str, str] = {}
    benchmark = args.index.upper()
    print(f"Downloading benchmark prices for {benchmark}", flush=True)
    if args.index_prices_csv:
        prices[benchmark] = read_price_series_csv(args.index_prices_csv)
    elif benchmark not in prices:
        try:
            prices[benchmark] = download_yahoo_prices(benchmark, start, end_day, price_field=args.price_field)
        except Exception as exc:
            raise RuntimeError(f"failed to download benchmark prices for {benchmark}: {exc}") from exc

    price_holdings = holdings_universe(holdings, historical_holdings)
    price_progress = Progress("Prices", len(price_holdings))
    for idx, holding in enumerate(price_holdings, start=1):
        if holding.ticker in prices:
            price_progress.update(idx, suffix=f"failures: {len(failures)}")
            continue
        try:
            points = download_yahoo_prices(holding.ticker, start, end_day, price_field=args.price_field)
            if points:
                prices[holding.ticker] = points
            else:
                failures[holding.ticker] = "empty price series"
        except Exception as exc:  # pragma: no cover - network dependent
            failures[holding.ticker] = str(exc)
            if args.fail_on_missing_prices:
                raise
        price_progress.update(idx, suffix=f"failures: {len(failures)}")
    price_progress.done(suffix=f"failures: {len(failures)}")

    metadata = {
        "index": benchmark,
        "start": start.isoformat(),
        "end": end_day.isoformat(),
        "years": args.years,
        "price_field": args.price_field,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sector_source": args.sector_source,
        "sector_failures": sector_failures,
        "price_failures": failures,
        "historical_holdings": bool(historical_holdings),
        "historical_holdings_snapshots": len(historical_holdings),
        "historical_holdings_start": min(historical_holdings).isoformat() if historical_holdings else None,
        "historical_holdings_end": max(historical_holdings).isoformat() if historical_holdings else None,
    }
    write_cache(args.data_dir, holdings, prices, metadata, historical_holdings=historical_holdings)
    paths = cache_paths(args.data_dir)
    print(f"Wrote cache to {paths['root']}")
    print(f"Holdings: {len(holdings)}; price series: {len(prices)}; missing prices: {len(failures)}")
    return 0


def construct_context(args: argparse.Namespace):
    holdings, prices, metadata = read_cache(args.data_dir)
    historical_holdings = read_historical_holdings_cache(args.data_dir)
    benchmark = str(metadata.get("index", "")).upper()
    if not benchmark:
        raise ValueError("metadata.json is missing index")
    universe_holdings = holdings_universe(holdings, historical_holdings)
    all_candidates = build_candidates(
        universe_holdings,
        prices,
        benchmark,
        args.rebalance_frequency,
        min_observations=args.min_observations,
        tax_metric=args.tax_metric,
        tax_rate=args.tax_rate,
        harvest_threshold_pct=args.harvest_threshold_pct,
        transaction_cost_bps=args.transaction_cost_bps,
        replacement_cost_bps=args.replacement_cost_bps,
        harvest_frequency=args.harvest_frequency,
    )
    current_tickers = {holding.ticker for holding in holdings}
    candidates = [candidate for candidate in all_candidates if candidate.ticker in current_tickers]
    if len(candidates) < args.sample_size:
        raise ValueError(
            f"Only {len(candidates)} candidates have enough price history; "
            f"sample size is {args.sample_size}."
        )
    return holdings, prices, metadata, benchmark, candidates, all_candidates, historical_holdings


def construct_from_args(
    args: argparse.Namespace,
    holdings,
    prices,
    metadata,
    benchmark: str,
    candidates,
    replacement_universe,
    historical_holdings,
    *,
    match_sectors: bool,
    progress_label: str = "Optimization",
):
    show_progress = args.progress if args.progress is not None else sys.stderr.isatty()
    portfolio = construct_portfolio(
        candidates,
        holdings,
        args.sample_size,
        args.error_margin,
        args.target_tax_alpha,
        args.rebalance_frequency,
        match_sectors=match_sectors,
        harvest_frequency=args.harvest_frequency,
        tax_alpha_mode=args.tax_alpha_mode,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
        benchmark_returns=daily_returns(prices[benchmark]),
        tracking_error_penalty=args.tracking_error_penalty,
        index_anchor_penalty=args.index_anchor_penalty,
        tax_metric=args.tax_metric,
        tax_assumptions={
            "tax_rate": args.tax_rate,
            "harvest_threshold_pct": args.harvest_threshold_pct,
            "transaction_cost_bps": args.transaction_cost_bps,
            "replacement_cost_bps": args.replacement_cost_bps,
            "replacement_count": args.replacement_count,
            "wash_sale_days": args.wash_sale_days,
        },
        selection_iterations=args.selection_iterations,
        weight_iterations=args.weight_iterations,
        random_seed=args.seed,
        show_progress=show_progress,
        progress_label=progress_label,
        replacement_universe=replacement_universe,
    )
    attach_portfolio_context(
        portfolio,
        args.data_dir,
        prices,
        metadata,
        replacement_universe,
        historical_holdings,
        show_progress=show_progress,
    )
    return portfolio


def attach_portfolio_context(
    portfolio,
    data_dir: str,
    prices,
    metadata,
    candidates,
    historical_holdings=None,
    *,
    show_progress: bool = False,
) -> None:
    for position in portfolio["positions"]:
        ticker = str(position.get("ticker", "")).upper()
        points = prices.get(ticker, [])
        if points:
            position["last_price"] = points[-1].adj_close
    targets = portfolio.get("targets", {})
    tax_assumptions = targets.get("tax_assumptions", {}) if isinstance(targets, dict) else {}
    benchmark = str(metadata.get("index", "")).upper()
    candidate_by_ticker = {candidate.ticker: candidate for candidate in candidates}
    selected = []
    weights = []
    for position in portfolio.get("positions", []):
        ticker = str(position.get("ticker", "")).upper()
        candidate = candidate_by_ticker.get(ticker)
        if candidate is None:
            continue
        selected.append(candidate)
        weights.append(float(position.get("weight", 0.0)))
    if benchmark and benchmark in prices and selected:
        harvest_progress = None

        def update_harvest_progress(count: int, total: int, suffix: str) -> None:
            nonlocal harvest_progress
            if not show_progress:
                return
            if harvest_progress is None:
                harvest_progress = tqdm(total=total, desc="Harvest replay", unit="day")
            if suffix:
                harvest_progress.set_postfix_str(suffix)
            harvest_progress.update(max(count - harvest_progress.n, 0))

        try:
            simulation = simulate_portfolio_harvests(
                selected,
                weights,
                candidates,
                prices,
                benchmark,
                daily_returns(prices[benchmark]),
                str(targets.get("rebalance_frequency", "quarterly")),
                float(targets.get("error_margin", 0.05)),
                float(targets.get("estimated_tax_loss_alpha", 0.0)),
                tax_alpha_mode=str(targets.get("tax_alpha_mode", "closest")),
                tax_rate=float(tax_assumptions.get("tax_rate", 0.30)),
                harvest_threshold_pct=float(tax_assumptions.get("harvest_threshold_pct", 0.05)),
                transaction_cost_bps=float(tax_assumptions.get("transaction_cost_bps", 5.0)),
                replacement_cost_bps=float(tax_assumptions.get("replacement_cost_bps", 10.0)),
                replacement_count=int(tax_assumptions.get("replacement_count", 2)),
                wash_sale_days=int(tax_assumptions.get("wash_sale_days", 31)),
                historical_holdings=historical_holdings or None,
                harvest_frequency=str(
                    targets.get(
                        "harvest_frequency",
                        targets.get("rebalance_frequency", "quarterly"),
                    )
                ),
                on_progress=update_harvest_progress if show_progress else None,
                max_weight_limit=(
                    float(targets["max_weight"])
                    if isinstance(targets.get("max_weight"), (int, float))
                    and float(targets["max_weight"]) > 0
                    else None
                ),
            )
        finally:
            if harvest_progress is not None:
                harvest_progress.close()
        sample_size = int(targets.get("sample_size", 0)) if isinstance(targets, dict) else 0
        if sample_size >= 50 and simulation.get("status") == "ok":
            replay_violations = []
            path_tracking_error = _float_or_none(simulation.get("portfolio_harvest_tracking_error")) or 0.0
            path_beta = _float_or_none(simulation.get("portfolio_harvest_beta")) or 0.0
            if path_tracking_error > 0.025:
                replay_violations.append(
                    f"harvest_path_tracking_error={path_tracking_error:.4f} above 0.0250"
                )
            if not 0.98 <= path_beta <= 1.02:
                replay_violations.append(f"harvest_path_beta={path_beta:.4f} outside 0.98-1.02")
            if replay_violations:
                raise ValueError(
                    "harvest replay violates hard benchmark-fidelity constraints: "
                    + "; ".join(replay_violations)
                )
        portfolio["portfolio_harvest_simulation"] = simulation
        metrics = portfolio.setdefault("metrics", {})
        metrics["portfolio_simulated_tax_alpha"] = simulation["portfolio_simulated_tax_alpha"]
        metrics["portfolio_realized_loss_rate"] = simulation["portfolio_realized_loss_rate"]
        metrics["portfolio_annual_realized_loss"] = simulation["annual_realized_loss"]
        metrics["portfolio_total_realized_loss"] = simulation["total_realized_loss"]
        metrics["portfolio_harvest_count"] = simulation["harvest_count"]
        metrics["portfolio_rebalance_count"] = simulation["rebalance_count"]
        metrics["portfolio_harvest_annualized_return"] = simulation["portfolio_harvest_annualized_return"]
        metrics["benchmark_annualized_return"] = simulation["benchmark_annualized_return"]
        metrics["portfolio_harvest_active_return"] = simulation["portfolio_harvest_active_return"]
        metrics["portfolio_harvest_tracking_error"] = simulation["portfolio_harvest_tracking_error"]
        metrics["portfolio_harvest_tracking_error_annualized_pct"] = simulation[
            "portfolio_harvest_tracking_error_annualized_pct"
        ]
        metrics["portfolio_harvest_beta"] = simulation["portfolio_harvest_beta"]
        metrics["portfolio_harvest_correlation"] = simulation["portfolio_harvest_correlation"]
        metrics["portfolio_harvest_observations"] = simulation["portfolio_harvest_observations"]
        metrics["realized_loss_rate_pct_per_year"] = simulation["realized_loss_rate_pct_per_year"]
        metrics["immediate_tax_savings_pct_per_year"] = simulation["immediate_tax_savings_pct_per_year"]
        metrics["immediate_net_tax_savings_pct_per_year"] = simulation["immediate_net_tax_savings_pct_per_year"]
        metrics["simulated_after_tax_alpha_pct_per_year"] = simulation["simulated_after_tax_alpha_pct_per_year"]
        metrics["full_liquidation_after_tax_alpha_pct_per_year"] = simulation[
            "full_liquidation_after_tax_alpha_pct_per_year"
        ]
        metrics["terminal_after_tax_wealth_difference_pct"] = simulation["terminal_after_tax_wealth_difference_pct"]
    portfolio["source_cache"] = {
        "data_dir": str(Path(data_dir).resolve()),
        "metadata": metadata,
        "candidate_count": len(candidates),
        "point_in_time_constituents": bool(historical_holdings),
    }


def write_json_with_parents(data: object, path: str | Path) -> None:
    target = Path(path)
    if target.parent != Path("."):
        target.parent.mkdir(parents=True, exist_ok=True)
    write_json(data, target)


def print_construct_summary(portfolio, *, sector_match: bool) -> None:
    metrics = portfolio["metrics"]
    tracking_error_pct = float(metrics.get("tracking_error_annualized_pct", metrics.get("error_percentage", 0.0)))
    simulated_alpha_pct = float(
        metrics.get(
            "simulated_after_tax_alpha_pct_per_year",
            float(metrics["simulated_tax_alpha"]) * 100.0,
        )
    )
    gross_loss_pct = float(
        metrics.get(
            "gross_harvestable_loss_rate_pct_per_year",
            float(metrics["gross_harvestable_loss_rate"]) * 100.0,
        )
    )
    print(
        (
            "Portfolio beta={:.4f}, tracking_error_annualized_pct={:.2f}%, "
            "tax_alpha_pct_per_year={:.2f}%, "
            "simulated_after_tax_alpha_pct_per_year={:.2f}%, "
            "gross_harvestable_loss_rate_pct_per_year={:.2f}%"
        ).format(
            float(metrics["beta"]),
            tracking_error_pct,
            float(metrics["tax_alpha"]) * 100.0,
            simulated_alpha_pct,
            gross_loss_pct,
        )
    )
    if "tracking_error" in metrics:
        active_share_pct = float(metrics.get("active_share_pct", float(metrics.get("active_share", 0.0)) * 100.0))
        max_weight_pct = float(metrics.get("max_weight_pct", float(metrics.get("max_weight", 0.0)) * 100.0))
        print(
            (
                "tracking_error_annualized_pct={:.2f}%, active_share_pct={:.2f}%, "
                "max_weight_pct={:.2f}%, effective_names={:.1f}"
            ).format(
                float(metrics.get("tracking_error_annualized_pct", float(metrics["tracking_error"]) * 100.0)),
                active_share_pct,
                max_weight_pct,
                float(metrics.get("effective_number_of_names", 0.0)),
            )
        )
    if "portfolio_simulated_tax_alpha" in metrics:
        replay_alpha_pct = float(
            metrics.get(
                "simulated_after_tax_alpha_pct_per_year",
                float(metrics["portfolio_simulated_tax_alpha"]) * 100.0,
            )
        )
        realized_loss_pct = float(
            metrics.get(
                "realized_loss_rate_pct_per_year",
                float(metrics["portfolio_realized_loss_rate"]) * 100.0,
            )
        )
        print(
            (
                "simulated_after_tax_alpha_pct_per_year={:.2f}%, "
                "realized_loss_rate_pct_per_year={:.2f}%, "
                "immediate_tax_savings_pct_per_year={:.2f}%, harvests={}, rebalances={}"
            ).format(
                replay_alpha_pct,
                realized_loss_pct,
                float(metrics.get("immediate_tax_savings_pct_per_year", 0.0)),
                int(metrics["portfolio_harvest_count"]),
                int(metrics.get("portfolio_rebalance_count", 0)),
            )
        )
    if "portfolio_harvest_tracking_error" in metrics:
        harvest_tracking_error_pct = float(
            metrics.get(
                "portfolio_harvest_tracking_error_annualized_pct",
                float(metrics["portfolio_harvest_tracking_error"]) * 100.0,
            )
        )
        print(
            (
                "harvest_path_tracking_error_annualized_pct={:.2f}%, "
                "harvest_path_active_return_pct_per_year={:.2f}%, beta={:.4f}, "
                "correlation={:.4f}"
            ).format(
                harvest_tracking_error_pct,
                float(metrics["portfolio_harvest_active_return"]) * 100.0,
                float(metrics["portfolio_harvest_beta"]),
                float(metrics["portfolio_harvest_correlation"]),
            )
        )
    if sector_match:
        sector_error_pct = float(
            metrics.get(
                "sector_absolute_error_pct",
                float(metrics.get("sector_abs_error", 0.0)) * 100.0,
            )
        )
        print(
            "sector_absolute_error_pct={:.2f}%".format(sector_error_pct)
        )
    warnings = metrics.get("constraint_warnings")
    if isinstance(warnings, list) and warnings:
        print("Constraint warnings: {}".format(", ".join(str(item) for item in warnings)))


def command_construct(args: argparse.Namespace) -> int:
    (
        holdings,
        prices,
        metadata,
        benchmark,
        candidates,
        replacement_universe,
        historical_holdings,
    ) = construct_context(args)
    portfolio = construct_from_args(
        args,
        holdings,
        prices,
        metadata,
        benchmark,
        candidates,
        replacement_universe,
        historical_holdings,
        match_sectors=args.sector_match,
        progress_label="Construct",
    )
    write_json_with_parents(portfolio, args.output)
    print(f"Wrote portfolio to {args.output}")
    print_construct_summary(portfolio, sector_match=args.sector_match)
    return 0


def command_sector_study(args: argparse.Namespace) -> int:
    (
        holdings,
        prices,
        metadata,
        benchmark,
        candidates,
        replacement_universe,
        historical_holdings,
    ) = construct_context(args)
    prefix = Path(args.output_prefix)
    no_sector_path = Path(f"{prefix}_no_sector.json")
    sector_path = Path(f"{prefix}_sector.json")
    comparison_path = Path(f"{prefix}_comparison.json")

    no_sector = construct_from_args(
        args,
        holdings,
        prices,
        metadata,
        benchmark,
        candidates,
        replacement_universe,
        historical_holdings,
        match_sectors=False,
        progress_label="No-sector",
    )
    sector = construct_from_args(
        args,
        holdings,
        prices,
        metadata,
        benchmark,
        candidates,
        replacement_universe,
        historical_holdings,
        match_sectors=True,
        progress_label="Sector-matched",
    )
    comparison = compare_portfolios(
        {"no_sector": no_sector, "sector": sector},
        holdings,
        prices,
        benchmark,
    )
    comparison["sources"] = {
        "no_sector": str(no_sector_path),
        "sector": str(sector_path),
    }

    write_json_with_parents(no_sector, no_sector_path)
    write_json_with_parents(sector, sector_path)
    write_json_with_parents(comparison, comparison_path)

    print(f"Wrote no-sector portfolio to {no_sector_path}")
    print_construct_summary(no_sector, sector_match=False)
    print(f"Wrote sector-matched portfolio to {sector_path}")
    print_construct_summary(sector, sector_match=True)
    print(f"Wrote comparison to {comparison_path}")
    print_comparison_summary(comparison)
    return 0


def command_compare(args: argparse.Namespace) -> int:
    holdings, prices, metadata = read_cache(args.data_dir)
    benchmark = str(metadata.get("index", "")).upper()
    if not benchmark:
        raise ValueError("metadata.json is missing index")

    portfolios: Dict[str, Mapping[str, object]] = {}
    sources: Dict[str, str] = {}
    for raw in args.portfolio:
        label, path = parse_labeled_path(raw)
        if label in portfolios:
            raise ValueError(f"duplicate portfolio label: {label}")
        state = read_json(path)
        if not isinstance(state, dict):
            raise ValueError(f"{path} must contain a portfolio JSON object")
        portfolios[label] = cast(Dict[str, object], state)
        sources[label] = str(path)

    comparison = compare_portfolios(portfolios, holdings, prices, benchmark)
    comparison["sources"] = sources
    if args.output:
        write_json_with_parents(comparison, args.output)
        print(f"Wrote comparison to {args.output}")
    print_comparison_summary(comparison)
    return 0


def print_comparison_summary(comparison) -> None:
    benchmark = comparison.get("benchmark", "benchmark")
    portfolios = comparison.get("portfolios", {})
    print(f"Compared {len(portfolios)} portfolio(s) against {benchmark}")
    if isinstance(portfolios, dict):
        for label, summary in portfolios.items():
            if not isinstance(summary, dict):
                continue
            returns = summary.get("returns", {})
            if not isinstance(returns, dict):
                returns = {}
            print(
                (
                    "{}: annualized return={}, tracking error={}, beta={}, "
                    "sector active share={}, sector similarity={}, active share={}"
                ).format(
                    label,
                    format_pct(returns.get("annualized_return")),
                    format_pct(returns.get("benchmark_tracking_error")),
                    format_number(returns.get("benchmark_beta")),
                    format_pct(summary.get("sector_active_share_to_index")),
                    format_number(summary.get("sector_similarity_to_index")),
                    format_pct(summary.get("active_share_to_index")),
                )
            )
    pairs = comparison.get("pairwise", [])
    if isinstance(pairs, list):
        for pair in pairs:
            if not isinstance(pair, dict):
                continue
            returns = pair.get("returns", {})
            if not isinstance(returns, dict):
                returns = {}
            print(
                (
                    "{} vs {}: ticker overlap={}, weighted overlap={}, "
                    "sector distance={}, return correlation={}, pair tracking error={}"
                ).format(
                    pair.get("left"),
                    pair.get("right"),
                    pair.get("ticker_overlap_count"),
                    format_pct(pair.get("weighted_overlap")),
                    format_pct(pair.get("sector_abs_distance")),
                    format_number(returns.get("correlation")),
                    format_pct(returns.get("tracking_error")),
                )
            )


def format_pct(value: object) -> str:
    number = _float_or_none(value)
    if number is None:
        return "n/a"
    return "{:.2f}%".format(number * 100.0)


def format_number(value: object) -> str:
    number = _float_or_none(value)
    if number is None:
        return "n/a"
    return "{:.4f}".format(number)


def _float_or_none(value: object) -> Optional[float]:
    if not isinstance(value, (int, float, str)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def command_rebalance(args: argparse.Namespace) -> int:
    state = read_json(args.state)
    if not isinstance(state, dict):
        raise ValueError("state JSON must contain an object")
    state = cast(Dict[str, object], state)
    current = read_current_positions(args.current_csv)
    operations = plan_rebalance(
        state,
        current,
        portfolio_value=args.portfolio_value,
        drift_threshold_pct=args.drift_threshold_pct,
        harvest_loss_threshold_pct=args.harvest_loss_threshold_pct,
        harvest_loss_threshold_amount=args.harvest_loss_threshold_amount,
    )
    write_operations_csv(operations, args.output)
    if args.output:
        print(f"Wrote {len(operations)} suggested operations to {args.output}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "download":
        return command_download(args)
    if args.command == "construct":
        return command_construct(args)
    if args.command == "sector-study":
        return command_sector_study(args)
    if args.command == "compare":
        return command_compare(args)
    if args.command == "rebalance":
        return command_rebalance(args)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
