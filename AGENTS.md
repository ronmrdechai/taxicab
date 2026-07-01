# Agent Instructions for Taxicab

## Clarification First

Taxicab is a Python command-line optimizer for sampled direct indexing. It works with index holdings, historical prices, sector tags, tracking-error targets, and simulated tax-loss-harvesting assumptions. When the user asks for anything moderately complex, pause before implementing and ask clarification questions.

A request is moderately complex when it has any of these traits:

- It affects multiple files, CLI commands, cache formats, output schemas, or optimizer workflows.
- It changes data ingestion, holdings parsing, price history handling, sector/industry tags, or point-in-time constituent behavior.
- It changes tracking error, beta, covariance, replacement selection, rebalancing, wash-sale handling, or simulated tax-alpha calculations.
- It involves financial, tax, data-quality, reproducibility, or user-facing behavior.
- It requires choosing between plausible implementation approaches or model assumptions.
- It is ambiguous about benchmark symbols, holdings sources, price fields, date ranges, tax rates, thresholds, transaction costs, or expected outputs.
- It could overwrite local cache data, generated run artifacts, user spreadsheets, or existing portfolio state.

Before executing a moderately complex request:

1. Explore the relevant project context.
2. Summarize the problem as understood.
3. Ask the smallest useful set of clarification questions, especially about data sources, financial assumptions, and expected CLI/output behavior.
4. State any assumptions that are safe to make, including whether the change is about model behavior, data plumbing, CLI ergonomics, or documentation.
5. Execute once the answers or assumptions make the path clear.

For simple, low-risk requests, proceed directly while still noting any important assumption.

Useful Taxicab clarification topics include:

- Which command is affected: `download`, `construct`, `sector-study`, `compare`, or `rebalance`.
- Whether inputs are current ETF holdings, point-in-time historical constituents, local price CSVs, XLSX files, or cached data.
- Whether the intended behavior prioritizes tracking accuracy, tax-loss opportunity, sector matching, turnover control, or deterministic reproducibility.
- Which financial assumptions should apply: tax rate, harvest threshold, transaction costs, replacement count, wash-sale window, rebalance frequency, sample size, and max weight.
- Whether outputs must remain backward compatible with existing JSON, CSV, and cache files.

## Execution Standards

- Prefer existing project patterns and tooling.
- Keep changes scoped to the user's request.
- Do not overwrite unrelated user work.
- Verify changes with the most relevant available checks.
- Report what changed and any verification performed.

For Taxicab specifically:

- Do not present simulated tax alpha, harvested losses, tracking error, or beta as tax advice, investment advice, or guaranteed outcomes.
- Preserve the project boundary that Taxicab does not connect to brokerages or place trades.
- Treat local cache directories, spreadsheet inputs, and `runs` outputs as user data unless the user explicitly asks to regenerate or modify them.
- Favor deterministic behavior for offline optimization and tests.
- Use the README test command when a change touches runtime behavior:

```bash
UV_CACHE_DIR=.uv-cache PYTHONPATH=src uv run --no-project python -m unittest discover -s tests
```
