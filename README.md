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
UV_CACHE_DIR=.uv-cache PYTHONPATH=src uv run --no-project python -m taxicab.cli --help
```

NumPy is a required dependency and is used for the optimizer's tracking-error covariance and projected-gradient math.

## Data Model

The cache is built from:

- a benchmark index symbol, such as `SPY`, `IVV`, `VOO`, or an abstract symbol you use internally
- a holdings CSV/XLSX file or URL
- historical close prices downloaded during the explicit `download` command
- sector tags from the holdings CSV and, when missing, Yahoo Finance sector metadata

Holdings CSV columns are intentionally flexible. The loader recognizes common names:

- ticker: `ticker`, `symbol`, `holding_ticker`
- weight: `weight`, `weight_pct`, `% weight`, `portfolio_weight`
- sector: `sector`, `gics_sector`, `morningstar_sector`

Weights can be percentages (`6.2`) or decimals (`0.062`).

For ETFs, use the issuer's holdings export as `--holdings-csv`, `--holdings-url`, `--holdings-xlsx`, or `--holdings-xlsx-url`. For abstract indexes, provide your own holdings file. This keeps the optimizer deterministic and avoids relying on one brittle ETF holdings endpoint.

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
  --min-weight 0.0001 \
  --max-weight 0.08 \
  --progress \
  --sector-match \
  --output ./portfolio.json
```

Run the same construction both with and without sector matching, then compare the runs against each other and the index baseline:

```bash
taxicab sector-study \
  --data-dir ./cache/spy \
  --sample-size 75 \
  --error-margin 0.05 \
  --target-tax-alpha 0.03 \
  --tax-metric simulated \
  --tax-alpha-mode at-least \
  --rebalance-frequency quarterly \
  --max-weight 0.08 \
  --progress \
  --output-prefix ./runs/spy_sector_study
```

Compare any existing portfolio states:

```bash
taxicab compare \
  --data-dir ./cache/spy \
  --portfolio no_sector=./runs/spy_sector_study_no_sector.json \
  --portfolio sector=./runs/spy_sector_study_sector.json \
  --output ./runs/spy_sector_study_comparison.json
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
- **gross harvestable loss rate:** annualized pre-tax loss opportunity across historical rebalance windows
- **sector:** issuer/online sector tag, used to match the index sector mix when requested

The selected portfolio minimizes tracking error against the requested error margin and tax-loss alpha, optionally matching sector weights at the same aggregate level as the index. It then solves long-only weights on the selected names.

The default optimizer tax-alpha model buys one lot for each candidate stock, checks it at each rebalance date, harvests only when the unrealized loss is beyond `--harvest-threshold-pct`, applies `--tax-rate` to the realized loss, subtracts round-trip transaction cost and replacement/tracking cost, then resets basis. The older gross metric is still emitted for diagnostics because it explains volatility-driven harvest opportunity, but it is not an after-tax return estimate.

After construction, Taxicab also runs a portfolio-level historical harvest simulation. It starts with the selected portfolio, checks positions on the requested `--rebalance-frequency` (`monthly`, `quarterly`, `half-yearly`/`halfly`, or `yearly`/`annually`), realizes loss lots whose after-tax benefit is positive after costs, and replaces each harvested stock with up to `--replacement-count` same-sector alternatives while avoiding recently harvested tickers for `--wash-sale-days`. The output JSON includes `portfolio_harvest_simulation` plus summary metrics such as `portfolio_realized_loss_rate`, `portfolio_annual_realized_loss`, and `portfolio_simulated_tax_alpha`.

Tracking-aware construction can also use `--max-weight`. When weight moves breach `--error-margin`, construction repairs tax-seeking moves back toward the selected index weights. This deliberately gives up tax opportunity when the tax objective conflicts with index-like behavior.

Construction uses `tqdm` progress bars for the selection and weight optimization passes in interactive terminals. Use `--progress` to force them in redirected/non-interactive runs, or `--no-progress` to silence them.

Important caveat: simulated tax alpha is still a model, not a tax opinion or a guaranteed realized after-tax return. Real outcomes depend on your tax rate, holding periods, wash-sale management, replacement securities, transaction costs, contribution timing, and actual tax lots.

## Tests

```bash
UV_CACHE_DIR=.uv-cache PYTHONPATH=src uv run --no-project python -m unittest discover -s tests
```
