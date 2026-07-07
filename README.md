# Taxicab

This is a Python command-line optimizer for building and maintaining a sampled direct index. It has two phases:

1. **Download phase:** materialize all index holdings, sector tags, benchmark prices, and constituent prices into a local cache.
2. **Offline optimization phase:** construct an `N` stock portfolio and later emit rebalance suggestions from a CSV snapshot of your current holdings.

The optimizer does not connect to a brokerage and does not place trades.

## Install

```bash
python3 -m pip install -e .
```

You can also run it through `uv` without installing:

```bash
UV_CACHE_DIR=.uv-cache PYTHONPATH=src uv run taxicab -- --help
```

NumPy, SciPy, and PySCIPOpt are required dependencies. NumPy is used for tracking-error covariance math, SciPy is used for constrained continuous weight optimization, and PySCIPOpt is used for optional MIQP selection.

## Quick Start

The commands below run Taxicab from a fresh checkout without installing it into your Python environment. They use `uv` to create an isolated runtime and keep dependency caches inside the repository working tree.

1. Confirm the CLI starts:

   ```bash
   UV_CACHE_DIR=.uv-cache PYTHONPATH=src uv run taxicab -- --help
   ```

2. Download an offline data cache from a holdings export. Replace `./spy_holdings.csv` with your own issuer-provided CSV/XLSX holdings file if you do not want to use SPY data:

   ```bash
   taxicab download \
     --index SPY \
     --holdings-csv ./spy_holdings.csv \
     --data-dir ./cache/spy \
     --years 5 \
     --price-field close
   ```

3. Construct a sample portfolio from that cache:

   ```bash
   taxicab construct \
     --data-dir ./cache/spy \
     --sample-size 50 \
     --error-margin 0.05 \
     --target-tax-alpha 0.02 \
     --tax-metric simulated \
     --output ./portfolio.json
   ```

Generated caches, portfolio files, run outputs, and order CSVs are local user data and are ignored by default. Review any generated output before acting on it; Taxicab does not provide tax or investment advice and does not place trades.

## Data Model

The cache is built from:

- a benchmark index symbol, such as `SPY`, `IVV`, `VOO`, or an abstract symbol you use internally
- a current holdings CSV/XLSX file or URL, or a point-in-time historical holdings CSV
- historical close prices downloaded during the explicit `download` command, plus optional local constituent prices
- sector and industry tags from the holdings CSV and, when missing, Yahoo Finance sector metadata for current holdings

Holdings CSV columns are intentionally flexible. The loader recognizes common names:

- ticker: `ticker`, `symbol`, `holding_ticker`
- weight: `weight`, `weight_pct`, `% weight`, `portfolio_weight`
- sector: `sector`, `gics_sector`, `morningstar_sector`
- industry: `industry`, `gics_industry`, `gics_sub_industry`

Weights can be percentages (`6.2`) or decimals (`0.062`).

Construction writes the human-readable portfolio/run state as JSON. It also writes a sibling `*.tracking-model.npz` artifact for the selected portfolio's rehydratable tracking model. The JSON records the artifact filename and ticker order, while the binary NumPy artifact stores the covariance matrix and benchmark covariance vector without expanding large numeric arrays into JSON.

For ETFs, use the issuer's holdings export as `--holdings-csv`, `--holdings-url`, `--holdings-xlsx`, or `--holdings-xlsx-url`. For abstract indexes, provide your own holdings file. This keeps the optimizer deterministic and avoids relying on one brittle ETF holdings endpoint.

For survivorship-bias-aware backtests, provide point-in-time snapshots with `--historical-holdings-csv`. The CSV needs a snapshot date column (`date`, `as_of`, or `snapshot_date`) plus ticker/weight/sector/industry columns. Taxicab uses the latest snapshot as the current construction universe, uses all historical snapshot tickers as the replacement universe, and restricts harvest replacements to names active in the latest snapshot on or before each simulated date. To include delisted names, also provide their historical prices in a local `--prices-csv` with `date`, `ticker`, and `adj_close`/`close` columns; public quote APIs often cannot recover delisted histories reliably.

## Example

Download a 30-year cache:

```bash
taxicab download \
  --index SPY \
  --holdings-csv ./spy_holdings.csv \
  --data-dir ./cache/spy \
  --years 30 \
  --price-field close
```

Download from point-in-time historical constituents and local constituent prices:

```bash
taxicab download \
  --index SPY \
  --historical-holdings-csv ./sp500_constituents_pit.csv \
  --prices-csv ./sp500_constituent_prices.csv \
  --data-dir ./cache/sp500_pit \
  --years 30 \
  --price-field close \
  --sector-source none
```

Construct a 75-stock sample targeting a 5% annualized error margin and 3% estimated tax-loss alpha:

```bash
taxicab construct \
  --data-dir ./cache/spy \
  --sample-size 75 \
  --error-margin 0.05 \
  --target-tax-alpha 0.03 \
  --tax-metric simulated \
  --tax-alpha-mode at-least \
  --tax-rate 0.30 \
  --harvest-threshold-pct 0.05 \
  --transaction-cost-bps 5 \
  --replacement-cost-bps 10 \
  --replacement-count 2 \
  --wash-sale-days 31 \
  --rebalance-frequency quarterly \
  --harvest-frequency daily \
  --min-weight 0.0001 \
  --max-weight 0.08 \
  --progress \
  --sector-match \
  --output ./portfolio.json
```

Compare existing portfolio states, including separate construction runs with and without `--sector-match`, against each other and the index baseline:

```bash
taxicab compare \
  --data-dir ./cache/spy \
  --portfolio no_sector=./runs/spy_sector_study_no_sector.json \
  --portfolio sector=./runs/spy_sector_study_sector.json \
  --output ./runs/spy_sector_study_comparison.json \
  --html-output ./runs/spy_sector_study_comparison.html
```

Produce rebalance suggestions from current positions:

```bash
taxicab rebalance \
  --state ./portfolio.json \
  --current-csv ./current_positions.csv \
  --output ./orders.csv
```

Current positions CSV columns:

- `ticker`
- `shares` and `price`, or `market_value`
- `cost_basis` total or `cost_basis_per_share`

## Objective

For each candidate stock, the optimizer estimates:

- **price beta:** covariance of stock returns to benchmark returns divided by benchmark variance, emitted as a diagnostic
- **tracking error:** annualized standard deviation of portfolio active returns versus the benchmark
- **simulated tax alpha:** annualized after-cost tax benefit from a simple one-stock loss-harvesting lot simulation
- **gross harvestable loss rate:** annualized pre-tax loss opportunity across historical harvest windows
- **sector:** issuer/online sector tag, used to match the index sector mix when requested

The selected portfolio minimizes tracking error against the requested error margin and tax-loss alpha, optionally matching sector weights at the same aggregate level as the index. It then solves long-only weights on the selected names with a constrained SciPy optimizer by default.

Construction can also run alternate backends for benchmarking:

- `--selection-method optimized` keeps the default random local-search selection.
- `--selection-method random-weighted` samples stocks without replacement using index weights as sampling probabilities. This is a legacy comparison baseline: paired with `--weight-method index-normalized`, it overweights selected large-cap names because the omitted tail is redistributed across the selected names.
- `--selection-method random-unbiased` uses probability-proportional-to-size sampling with certainty names and inverse-inclusion-probability weights, so the sampled portfolio equals the index in expectation. With `--sector-match`, it applies the same design inside each sector quota. This method uses its own `random-unbiased` weights rather than a separate weight optimizer.
- `--selection-method greedy` uses deterministic beam search with width 1.
- `--selection-method beam --beam-width N` keeps the best deterministic candidate portfolios at each selection step.
- `--selection-method miqp` uses PySCIPOpt to solve a mixed-integer quadratic selection model with binary inclusion variables and continuous weights.
- `--weight-method slsqp` keeps the default SciPy SLSQP constrained weight solver.
- `--weight-method index-normalized` skips continuous weight optimization and normalizes selected index weights.
- `--replacement-method ranked` keeps the default same-sector, similarity-ranked harvest replacement search.
- `--replacement-method random` chooses random eligible same-sector replacements during harvest replay.

Random, deterministic baseline, and MIQP methods can violate benchmark-fidelity constraints more easily than the default optimizer. Use `--allow-constraint-violations` to write those runs for comparison while keeping violations in the output metrics. To compare methods, construct one portfolio per method and pass them to `taxicab compare --html-output`.

The default optimizer tax-alpha model buys one lot for each candidate stock, checks it at each `--harvest-frequency` date, harvests only when the unrealized loss is beyond `--harvest-threshold-pct`, applies `--tax-rate` to the realized loss, subtracts round-trip transaction cost and replacement/tracking cost, then resets basis. The older gross metric is still emitted for diagnostics because it explains volatility-driven harvest opportunity, but it is not an after-tax return estimate.

After construction, Taxicab also runs a portfolio-level historical harvest simulation. It starts with the selected portfolio, checks positions on the requested `--harvest-frequency` (`daily`, `monthly`, `quarterly`, `half-yearly`/`halfly`, or `yearly`/`annually`), realizes loss lots whose after-tax benefit is positive after costs, and replaces each harvested stock with up to `--replacement-count` alternatives while avoiding recently harvested tickers for `--wash-sale-days`. Separately, it rebalances target weights on `--rebalance-frequency`, which defaults to quarterly. Replacements are filtered to the same sector, prefer the same industry, and are ranked by beta distance and return correlation to the harvested stock. When point-in-time historical constituents are cached, replacements are also restricted to names active in the index at that simulated date. The output JSON includes `portfolio_harvest_simulation` plus summary metrics such as `portfolio_realized_loss_rate`, `portfolio_annual_realized_loss`, `portfolio_simulated_tax_alpha`, `portfolio_harvest_tracking_error`, `portfolio_harvest_beta`, `portfolio_harvest_correlation`, and `portfolio_harvest_active_return`.

Tracking-aware construction can also use `--max-weight`. The constrained weight solver enforces `--error-margin` when the selected names can feasibly meet it, which can give up tax opportunity when the tax objective conflicts with index-like behavior.

Construction uses `tqdm` progress bars for the selection and weight optimization passes in interactive terminals. Use `--progress` to force them in redirected/non-interactive runs, or `--no-progress` to silence them.

Important caveat: simulated tax alpha is still a model, not a tax opinion or a guaranteed realized after-tax return. Real outcomes depend on your tax rate, holding periods, wash-sale management, replacement securities, transaction costs, contribution timing, and actual tax lots.

## Checks and Tests

Run linting, type checking, and the unit test suite before committing changes:

```bash
./scripts/check
```

The script runs `ruff check .`, `ty check .`, and the unit tests through `uv`.

## License

Taxicab is licensed under the Apache License, Version 2.0. See [`LICENSE`](LICENSE) for the full license text.

Per-file source headers are not required for this project because the repository-level `LICENSE` file and package metadata identify the license for the work. Add a header only when a file needs distinct copyright, attribution, or third-party license notices.
