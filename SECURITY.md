# Security Policy

## Sensitive data handling

Taxicab is designed to run locally and does not connect to brokerages or place trades. Do not commit local portfolio snapshots, generated orders, run outputs, quote caches, credentials, API tokens, or other private financial data.

The repository `.gitignore` excludes common local artifacts such as virtual environments, caches, generated runs, `portfolio.json`, and `orders.csv`.

## Reporting a vulnerability

If you find a vulnerability or accidentally committed sensitive data, open a private security advisory or contact the maintainers privately before filing a public issue.
