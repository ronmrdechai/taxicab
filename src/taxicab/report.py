from __future__ import annotations

import json
import math
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, cast


@dataclass(frozen=True)
class ComparisonMetric:
    key: str
    label: str
    path: Sequence[str]
    group: str
    value_format: str = "number"
    better: str = "neutral"
    target: Optional[float] = None
    full_intensity_at: Optional[float] = None
    description: str = ""


@dataclass(frozen=True)
class ObjectiveComponent:
    key: str
    label: str
    unit: str
    value_format: str = "number"
    description: str = ""


PORTFOLIO_METRICS: Sequence[ComparisonMetric] = (
    ComparisonMetric(
        "annualized_return",
        "Annualized return",
        ("returns", "annualized_return"),
        "Performance",
        "pct",
        better="higher",
        full_intensity_at=0.05,
    ),
    ComparisonMetric(
        "benchmark_annualized_active_return",
        "Active return vs benchmark",
        ("returns", "benchmark_annualized_active_return"),
        "Performance",
        "pct",
        better="higher",
        full_intensity_at=0.03,
    ),
    ComparisonMetric(
        "cumulative_return",
        "Cumulative return",
        ("returns", "cumulative_return"),
        "Performance",
        "pct",
        better="higher",
        full_intensity_at=0.20,
    ),
    ComparisonMetric(
        "max_drawdown",
        "Max drawdown",
        ("returns", "max_drawdown"),
        "Performance",
        "pct",
        better="target",
        target=0.0,
        full_intensity_at=0.10,
        description="Closer to zero is better.",
    ),
    ComparisonMetric(
        "benchmark_tracking_error",
        "Tracking error vs benchmark",
        ("returns", "benchmark_tracking_error"),
        "Benchmark fit",
        "pct",
        better="lower",
        full_intensity_at=0.03,
    ),
    ComparisonMetric(
        "benchmark_beta",
        "Beta vs benchmark",
        ("returns", "benchmark_beta"),
        "Benchmark fit",
        "number",
        better="target",
        target=1.0,
        full_intensity_at=0.10,
        description="Closer to 1.0 is better.",
    ),
    ComparisonMetric(
        "active_share_to_index",
        "Active share to index",
        ("active_share_to_index",),
        "Benchmark fit",
        "pct",
        better="lower",
        full_intensity_at=0.15,
    ),
    ComparisonMetric(
        "weighted_overlap_with_index",
        "Weighted overlap with index",
        ("weighted_overlap_with_index",),
        "Benchmark fit",
        "pct",
        better="higher",
        full_intensity_at=0.15,
    ),
    ComparisonMetric(
        "sector_active_share_to_index",
        "Sector active share",
        ("sector_active_share_to_index",),
        "Sector fit",
        "pct",
        better="lower",
        full_intensity_at=0.05,
    ),
    ComparisonMetric(
        "sector_similarity_to_index",
        "Sector similarity",
        ("sector_similarity_to_index",),
        "Sector fit",
        "number",
        better="higher",
        full_intensity_at=0.05,
    ),
    ComparisonMetric(
        "sector_overlap_to_index",
        "Sector overlap with index",
        ("sector_overlap_to_index",),
        "Sector fit",
        "pct",
        better="higher",
        full_intensity_at=0.05,
    ),
    ComparisonMetric(
        "covered_price_weight",
        "Covered price weight",
        ("covered_price_weight",),
        "Data quality",
        "pct",
        better="higher",
        full_intensity_at=0.05,
    ),
    ComparisonMetric(
        "missing_price_tickers",
        "Missing price tickers",
        ("missing_price_tickers",),
        "Data quality",
        "list",
        better="lower_length",
        full_intensity_at=5.0,
    ),
    ComparisonMetric(
        "position_count",
        "Position count",
        ("position_count",),
        "Data quality",
        "integer",
    ),
)


HARVEST_REPLAY_METRICS: Sequence[ComparisonMetric] = (
    ComparisonMetric(
        "portfolio_simulated_tax_alpha",
        "Simulated tax alpha",
        ("harvest_replay", "portfolio_simulated_tax_alpha"),
        "Harvest replay",
        "pct",
        better="higher",
        full_intensity_at=0.02,
    ),
    ComparisonMetric(
        "portfolio_realized_loss_rate",
        "Realized loss rate",
        ("harvest_replay", "portfolio_realized_loss_rate"),
        "Harvest replay",
        "pct",
        better="higher",
        full_intensity_at=0.02,
    ),
    ComparisonMetric(
        "portfolio_harvest_active_return",
        "Harvest active return",
        ("harvest_replay", "portfolio_harvest_active_return"),
        "Harvest replay",
        "pct",
        better="higher",
        full_intensity_at=0.03,
    ),
    ComparisonMetric(
        "portfolio_harvest_tracking_error",
        "Harvest tracking error",
        ("harvest_replay", "portfolio_harvest_tracking_error"),
        "Harvest replay",
        "pct",
        better="lower",
        full_intensity_at=0.03,
    ),
    ComparisonMetric(
        "portfolio_harvest_beta",
        "Harvest beta",
        ("harvest_replay", "portfolio_harvest_beta"),
        "Harvest replay",
        "number",
        better="target",
        target=1.0,
        full_intensity_at=0.10,
    ),
    ComparisonMetric(
        "total_net_tax_benefit",
        "Total net tax benefit",
        ("harvest_replay", "total_net_tax_benefit"),
        "Harvest replay",
        "number",
        better="higher",
        full_intensity_at=0.05,
    ),
    ComparisonMetric(
        "terminal_after_tax_wealth_difference",
        "Terminal after-tax wealth difference",
        ("harvest_replay", "terminal_after_tax_wealth_difference"),
        "Harvest replay",
        "pct",
        better="higher",
        full_intensity_at=0.05,
    ),
    ComparisonMetric(
        "total_transaction_cost",
        "Total transaction cost",
        ("harvest_replay", "total_transaction_cost"),
        "Harvest replay",
        "number",
        better="lower",
        full_intensity_at=0.02,
    ),
    ComparisonMetric(
        "total_replacement_cost",
        "Total replacement cost",
        ("harvest_replay", "total_replacement_cost"),
        "Harvest replay",
        "number",
        better="lower",
        full_intensity_at=0.02,
    ),
    ComparisonMetric(
        "harvest_count",
        "Harvest count",
        ("harvest_replay", "harvest_count"),
        "Harvest replay",
        "integer",
    ),
    ComparisonMetric(
        "rebalance_count",
        "Rebalance count",
        ("harvest_replay", "rebalance_count"),
        "Harvest replay",
        "integer",
    ),
)


PAIRWISE_METRICS: Sequence[ComparisonMetric] = (
    ComparisonMetric(
        "ticker_overlap_count",
        "Ticker overlap",
        ("ticker_overlap_count",),
        "Pairwise",
        "integer",
    ),
    ComparisonMetric(
        "weighted_overlap",
        "Weighted overlap",
        ("weighted_overlap",),
        "Pairwise",
        "pct",
        better="higher",
        full_intensity_at=0.15,
    ),
    ComparisonMetric(
        "active_share",
        "Pair active share",
        ("active_share",),
        "Pairwise",
        "pct",
        better="lower",
        full_intensity_at=0.15,
    ),
    ComparisonMetric(
        "sector_abs_distance",
        "Sector distance",
        ("sector_abs_distance",),
        "Pairwise",
        "pct",
        better="lower",
        full_intensity_at=0.05,
    ),
    ComparisonMetric(
        "sector_similarity",
        "Sector similarity",
        ("sector_similarity",),
        "Pairwise",
        "number",
        better="higher",
        full_intensity_at=0.05,
    ),
    ComparisonMetric(
        "correlation",
        "Return correlation",
        ("returns", "correlation"),
        "Pairwise",
        "number",
        better="higher",
        full_intensity_at=0.10,
    ),
    ComparisonMetric(
        "tracking_error",
        "Pair tracking error",
        ("returns", "tracking_error"),
        "Pairwise",
        "pct",
        better="lower",
        full_intensity_at=0.03,
    ),
)

PAIRWISE_HEATMAP_METRICS: Sequence[ComparisonMetric] = (
    ComparisonMetric("weight_cosine_similarity", "Cosine similarity of weights", ("weight_cosine_similarity",), "Heatmap", "number"),
    ComparisonMetric("active_share", "Active share distance", ("active_share",), "Heatmap", "pct"),
    ComparisonMetric("tracking_error", "Tracking-error distance", ("returns", "tracking_error"), "Heatmap", "pct"),
    ComparisonMetric("sector_abs_distance", "Sector exposure distance", ("sector_abs_distance",), "Heatmap", "pct"),
    ComparisonMetric("factor_abs_distance", "Factor exposure distance", ("factor_abs_distance",), "Heatmap", "number"),
    ComparisonMetric("tax_lot_action_overlap", "Tax-lot action overlap", ("tax_lot_action_overlap",), "Heatmap", "number"),
)

OBJECTIVE_COMPONENTS: Sequence[ObjectiveComponent] = (
    ObjectiveComponent(
        "tracking_error_penalty",
        "Tracking error / target",
        "multiple",
        "multiple",
        "Tracking error divided by the target error margin.",
    ),
    ObjectiveComponent(
        "sector_penalty",
        "Sector distance",
        "portfolio weight",
        "pct",
        "Absolute sector-weight distance from the index.",
    ),
    ObjectiveComponent(
        "factor_penalty",
        "Beta distance",
        "beta points",
        "number",
        "Absolute distance from beta 1.0.",
    ),
    ObjectiveComponent(
        "concentration_penalty",
        "Max position weight",
        "portfolio weight",
        "pct",
        "Largest single-position weight.",
    ),
    ObjectiveComponent(
        "transaction_cost",
        "Transaction cost",
        "portfolio value",
        "pct",
        "Replay transaction cost as a share of portfolio value.",
    ),
    ObjectiveComponent(
        "tax_benefit",
        "Tax-alpha shortfall benefit",
        "annualized return",
        "pct",
        "Negative values reduce the objective.",
    ),
    ObjectiveComponent(
        "wash_sale_penalty",
        "Skipped harvest count",
        "count",
        "integer",
        "Skipped harvest attempts from replacement or constraint failures.",
    ),
    ObjectiveComponent(
        "cash_penalty",
        "Cash drift",
        "portfolio weight",
        "pct",
        "Absolute distance between total position weight and 100%.",
    ),
)


def write_comparison_html_report(comparison: Mapping[str, object], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_comparison_html_report(comparison), encoding="utf-8")


def render_comparison_html_report(comparison: Mapping[str, object]) -> str:
    benchmark = str(comparison.get("benchmark", "benchmark"))
    created_at = str(comparison.get("created_at", ""))
    portfolios = _mapping(comparison.get("portfolios"))
    labels = list(portfolios)
    metrics = list(PORTFOLIO_METRICS)
    if _has_harvest_replay(portfolios):
        metrics.extend(HARVEST_REPLAY_METRICS)

    sections = [
        _render_summary_cards(comparison, labels),
        _render_sources(comparison),
        _render_metric_table("Portfolio Metrics", metrics, labels, portfolios),
        _render_sector_table(comparison, labels, portfolios),
        _render_pairwise_heatmap(comparison, labels),
        _render_pca_embedding(portfolios, labels),
        _render_frontier_chart(portfolios, labels),
        _render_objective_waterfalls(portfolios, labels),
        _render_pairwise_table(comparison, labels),
    ]
    body = "\n".join(section for section in sections if section)
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Taxicab Comparison Report</title>
<style>
:root {{
  color-scheme: light;
  --bg: #f6f7f9;
  --panel: #ffffff;
  --text: #17202a;
  --muted: #617083;
  --border: #d8dee6;
  --head: #eef2f6;
  --green: #238636;
  --red: #da3633;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-size: 14px;
  line-height: 1.45;
}}
header {{
  background: #18202a;
  color: #ffffff;
  padding: 24px clamp(18px, 4vw, 48px);
}}
header h1 {{
  margin: 0;
  font-size: clamp(24px, 3vw, 34px);
  font-weight: 700;
  letter-spacing: 0;
}}
header p {{
  margin: 6px 0 0;
  color: #d7dee8;
}}
main {{
  padding: 22px clamp(18px, 4vw, 48px) 42px;
}}
section {{
  margin: 0 0 22px;
}}
h2 {{
  margin: 0 0 10px;
  font-size: 18px;
}}
.cards {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 10px;
}}
.card {{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px 14px;
}}
.card .label {{
  color: var(--muted);
  font-size: 12px;
  text-transform: uppercase;
}}
.card .value {{
  margin-top: 3px;
  font-size: 18px;
  font-weight: 700;
}}
.note {{
  color: var(--muted);
  margin-top: 10px;
}}
.table-wrap {{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow-x: auto;
}}
table {{
  width: 100%;
  border-collapse: collapse;
  min-width: 720px;
}}
th, td {{
  border-bottom: 1px solid var(--border);
  padding: 9px 10px;
  text-align: right;
  vertical-align: middle;
  white-space: nowrap;
}}
th:first-child, td:first-child {{
  text-align: left;
  position: sticky;
  left: 0;
  background: inherit;
}}
thead th {{
  background: var(--head);
  font-weight: 700;
}}
tbody tr.group th {{
  background: #f7f9fb;
  color: #344255;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0;
}}
tbody tr:last-child td, tbody tr:last-child th {{
  border-bottom: 0;
}}
.muted {{
  color: var(--muted);
}}
.metric-label {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
}}
.metric-label[title] {{
  cursor: help;
}}
.legend {{
  display: flex;
  gap: 12px;
  align-items: center;
  flex-wrap: wrap;
  color: var(--muted);
  margin: 0 0 10px;
}}
.swatch {{
  width: 12px;
  height: 12px;
  border-radius: 3px;
  display: inline-block;
  vertical-align: -1px;
  margin-right: 4px;
}}
.good {{ background: rgba(35, 134, 54, 0.35); }}
.bad {{ background: rgba(218, 54, 51, 0.35); }}
.viz-panel {{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 14px;
  overflow-x: auto;
}}
.viz-controls {{ margin: 0 0 12px; }}
.viz-controls select {{ padding: 6px 8px; }}
svg.taxicab-viz {{ width: 100%; min-width: 640px; height: 380px; }}
.axis {{ stroke: #617083; stroke-width: 1; }}
.heat-cell {{ stroke: #ffffff; stroke-width: 1; }}
.viz-label {{ fill: #17202a; font-size: 12px; }}
.viz-muted {{ fill: #617083; font-size: 11px; }}
.objective-cell {{
  display: grid;
  grid-template-columns: minmax(74px, auto) minmax(72px, 1fr);
  gap: 8px;
  align-items: center;
}}
.objective-value {{
  font-variant-numeric: tabular-nums;
}}
.objective-bar-track {{
  height: 9px;
  background: #eef2f6;
  border-radius: 999px;
  overflow: hidden;
}}
.objective-bar {{
  display: block;
  height: 100%;
  border-radius: 999px;
}}
.objective-bar.cost {{ background: rgba(218, 54, 51, 0.45); }}
.objective-bar.benefit {{ background: rgba(35, 134, 54, 0.45); }}
.objective-table td:nth-child(3) {{
  text-align: left;
  white-space: normal;
  min-width: 220px;
  max-width: 420px;
}}
.pairwise-picker-table table {{
  min-width: 520px;
}}
.pairwise-picker-table th,
.pairwise-picker-table td {{
  text-align: center;
}}
.pairwise-picker-table tbody th {{
  text-align: left;
}}
.pairwise-picker-table td {{
  font-variant-numeric: tabular-nums;
}}
.pairwise-picker-table td.pairwise-self {{
  color: var(--muted);
  background: #f7f9fb;
}}
</style>
</head>
<body>
<header>
  <h1>Taxicab Comparison Report</h1>
  <p>Benchmark: {benchmark}{created}</p>
</header>
<main>
  <div class="legend">
    <span><span class="swatch good"></span>Better relative result</span>
    <span><span class="swatch bad"></span>Worse relative result</span>
    <span>Color intensity scales with metric gap.</span>
  </div>
  {body}
</main>
</body>
</html>
""".format(
        benchmark=escape(benchmark),
        created=f" | Created: {escape(created_at)}" if created_at else "",
        body=body,
    )


def _render_summary_cards(comparison: Mapping[str, object], labels: Sequence[str]) -> str:
    benchmark = str(comparison.get("benchmark", "benchmark"))
    index_count = comparison.get("index_position_count")
    pair_count = len(_list(comparison.get("pairwise")))
    harvest = _mapping(comparison.get("harvest_replay"))
    cards = [
        ("Benchmark", benchmark),
        ("Portfolios", str(len(labels))),
        ("Pairwise Comparisons", str(pair_count)),
        ("Index Positions", _format_scalar(index_count)),
    ]
    if harvest.get("enabled") is True:
        cards.append(("Harvest Replay", "Enabled"))
    card_html = "\n".join(
        '<div class="card"><div class="label">{}</div><div class="value">{}</div></div>'.format(
            escape(label),
            escape(value),
        )
        for label, value in cards
    )
    return f"""<section>
<div class="cards">
{card_html}
</div>
<p class="note">Simulated tax and harvest figures are model diagnostics based on the input assumptions. They are not tax advice, investment advice, or guaranteed outcomes.</p>
</section>"""


def _render_sources(comparison: Mapping[str, object]) -> str:
    sources = _mapping(comparison.get("sources"))
    if not sources:
        return ""
    rows = "\n".join(
        "<tr><th>{}</th><td>{}</td></tr>".format(escape(str(label)), escape(str(path)))
        for label, path in sources.items()
    )
    return f"""<section>
<h2>Sources</h2>
<div class="table-wrap">
<table>
<thead><tr><th>Portfolio</th><th>Path</th></tr></thead>
<tbody>
{rows}
</tbody>
</table>
</div>
</section>"""


def _render_metric_table(
    title: str,
    metrics: Sequence[ComparisonMetric],
    labels: Sequence[str],
    portfolios: Mapping[str, object],
) -> str:
    if not labels:
        return ""
    grouped: Dict[str, List[ComparisonMetric]] = {}
    for metric in metrics:
        values = [_value_at(_mapping(portfolios.get(label)), metric.path) for label in labels]
        if any(value is not None for value in values):
            grouped.setdefault(metric.group, []).append(metric)
    if not grouped:
        return ""

    header = "".join(f"<th>{escape(label)}</th>" for label in labels)
    rows: List[str] = []
    for group, group_metrics in grouped.items():
        rows.append(f'<tr class="group"><th colspan="{len(labels) + 1}">{escape(group)}</th></tr>')
        for metric in group_metrics:
            values = [_value_at(_mapping(portfolios.get(label)), metric.path) for label in labels]
            cells = []
            for value in values:
                style = _comparison_style(metric, value, values)
                cells.append(f"<td{style}>{escape(_format_value(value, metric))}</td>")
            label = escape(metric.label)
            if metric.description:
                label = f'<span class="metric-label" title="{escape(metric.description)}">{label}</span>'
            rows.append(f"<tr><th>{label}</th>{''.join(cells)}</tr>")
    return f"""<section>
<h2>{escape(title)}</h2>
<div class="table-wrap">
<table>
<thead><tr><th>Metric</th>{header}</tr></thead>
<tbody>
{''.join(rows)}
</tbody>
</table>
</div>
</section>"""


def _render_sector_table(
    comparison: Mapping[str, object],
    labels: Sequence[str],
    portfolios: Mapping[str, object],
) -> str:
    targets = _mapping(comparison.get("index_sector_targets"))
    sectors = set(str(sector) for sector in targets)
    for label in labels:
        sectors.update(str(sector) for sector in _mapping(_mapping(portfolios.get(label)).get("sector_weights")))
    if not sectors:
        return ""

    ordered = sorted(sectors, key=lambda sector: (-_numeric_or_zero(targets.get(sector)), sector))
    header = "".join(f"<th>{escape(label)}</th>" for label in labels)
    rows = []
    for sector in ordered:
        target = _numeric_or_zero(targets.get(sector))
        values = [
            _mapping(_mapping(portfolios.get(label)).get("sector_weights")).get(sector, 0.0)
            for label in labels
        ]
        cells = [f"<td>{escape(_format_percent(target))}</td>"]
        for value in values:
            style = _target_style(_numeric(value), values, target, full_intensity_at=0.02)
            cells.append(f"<td{style}>{escape(_format_percent(_numeric_or_zero(value)))}</td>")
        rows.append(f"<tr><th>{escape(sector)}</th>{''.join(cells)}</tr>")
    return f"""<section>
<h2>Sector Weights</h2>
<div class="table-wrap">
<table>
<thead><tr><th>Sector</th><th>Index</th>{header}</tr></thead>
<tbody>
{''.join(rows)}
</tbody>
</table>
</div>
</section>"""


def _render_pairwise_heatmap(comparison: Mapping[str, object], labels: Sequence[str]) -> str:
    pairs = [_mapping(pair) for pair in _list(comparison.get("pairwise"))]
    if len(labels) < 2 or not pairs:
        return ""
    metrics = {}
    for metric in PAIRWISE_HEATMAP_METRICS:
        matrix = [[1.0 if row == col and "similarity" in metric.key else 0.0 for col in labels] for row in labels]
        for pair in pairs:
            left = str(pair.get("left", ""))
            right = str(pair.get("right", ""))
            if left not in labels or right not in labels:
                continue
            value = _numeric(_value_at(pair, metric.path))
            if value is None:
                continue
            row = labels.index(left)
            col = labels.index(right)
            matrix[row][col] = value
            matrix[col][row] = value
        metrics[metric.key] = {"label": metric.label, "format": metric.value_format, "values": matrix}
    payload = _json_script_payload({"labels": list(labels), "metrics": metrics})
    return f"""<section>
<h2>Pairwise Similarity Heatmap</h2>
<div class="viz-panel">
<div class="viz-controls"><label for="heatmap-metric">Metric </label><select id="heatmap-metric"></select></div>
<svg id="pairwise-heatmap" class="taxicab-viz" role="img" aria-label="Pairwise similarity heatmap"></svg>
</div>
<script type="application/json" id="pairwise-heatmap-data">{payload}</script>
<script>{_heatmap_script()}</script>
</section>"""


def _render_pca_embedding(portfolios: Mapping[str, object], labels: Sequence[str]) -> str:
    points = _pca_points(portfolios, labels)
    if not points:
        return ""
    payload = _json_script_payload(points)
    return f"""<section>
<h2>Feature-Space PCA Embedding</h2>
<div class="viz-panel">
<p class="note">PCA is computed from run features such as objective terms, sector/factor drifts, tax metrics, turnover, diversification, and constraint slack rather than raw stock weights.</p>
<svg id="pca-embedding" class="taxicab-viz" role="img" aria-label="Feature-space PCA embedding"></svg>
</div>
<script type="application/json" id="pca-embedding-data">{payload}</script>
<script>{_scatter_script("pca-embedding", "pca-embedding-data", "PC1 score", "PC2 score", "standardized units", "standardized units")}</script>
</section>"""


def _render_frontier_chart(portfolios: Mapping[str, object], labels: Sequence[str]) -> str:
    points = []
    for label in labels:
        summary = _mapping(portfolios.get(label))
        features = _mapping(summary.get("features"))
        replay = _mapping(summary.get("harvest_replay"))
        x_value = _numeric(features.get("tracking_error"))
        y_value = (
            _numeric(replay.get("portfolio_simulated_tax_alpha"))
            or _numeric(features.get("realized_loss_rate"))
            or _numeric(features.get("simulated_tax_alpha"))
        )
        if x_value is None or y_value is None:
            continue
        points.append(
            {
                "label": label,
                "x": x_value,
                "y": y_value,
                "color": _numeric(features.get("turnover")) or 0.0,
                "size": _numeric(features.get("effective_names")) or _numeric(summary.get("active_share_to_index")) or 1.0,
            }
        )
    if not points:
        return ""
    payload = _json_script_payload(points)
    return f"""<section>
<h2>Efficient-Frontier Style View</h2>
<div class="viz-panel">
<p class="note">x = tracking error, y = simulated tax alpha or realized loss harvest, color = turnover proxy, size = effective names.</p>
<svg id="frontier-chart" class="taxicab-viz" role="img" aria-label="Efficient frontier chart"></svg>
</div>
<script type="application/json" id="frontier-data">{payload}</script>
<script>{_scatter_script("frontier-chart", "frontier-data", "Tracking error", "Tax alpha / realized loss", "annualized %", "annualized %", "pct", "pct")}</script>
</section>"""


def _render_objective_waterfalls(portfolios: Mapping[str, object], labels: Sequence[str]) -> str:
    if not labels:
        return ""
    portfolio_values: Dict[str, Dict[str, float]] = {}
    for label in labels:
        decomposition = _mapping(_mapping(portfolios.get(label)).get("objective_decomposition"))
        if not decomposition:
            continue
        portfolio_values[label] = {
            component.key: _numeric(decomposition.get(component.key)) or 0.0
            for component in OBJECTIVE_COMPONENTS
        }
    if not portfolio_values:
        return ""

    active_labels = [label for label in labels if label in portfolio_values]
    header = "".join(f"<th>{escape(label)}</th>" for label in active_labels)
    rows = []
    for component in OBJECTIVE_COMPONENTS:
        values = [portfolio_values[label].get(component.key, 0.0) for label in active_labels]
        max_abs = max([abs(value) for value in values] + [1e-12])
        cells = []
        for value in values:
            width = min(abs(value) / max_abs * 100.0, 100.0) if max_abs > 0 else 0.0
            tone = "benefit" if value < 0 else "cost"
            cells.append(
                '<td><div class="objective-cell">'
                f'<span class="objective-value">{escape(_format_objective_value(value, component))}</span>'
                '<span class="objective-bar-track">'
                f'<span class="objective-bar {tone}" style="width: {width:.1f}%;"></span>'
                "</span></div></td>"
            )
        rows.append(
            "<tr>"
            f"<th>{escape(component.label)}</th>"
            f"<td>{escape(component.unit)}</td>"
            f"<td class=\"muted\">{escape(component.description)}</td>"
            f"{''.join(cells)}"
            "</tr>"
        )
    return f"""<section>
<h2>Objective Decomposition</h2>
<div class="table-wrap">
<table class="objective-table">
<thead><tr><th>Component</th><th>Unit</th><th>Description</th>{header}</tr></thead>
<tbody>
{''.join(rows)}
</tbody>
</table>
</div>
<p class="note">Components are shown with their own units because they are diagnostics from mixed sources, not a single additive waterfall scale. Green bars are objective benefits; red bars are costs or penalties.</p>
</section>"""


def _render_pairwise_table(comparison: Mapping[str, object], labels: Sequence[str]) -> str:
    pairs = _list(comparison.get("pairwise"))
    if len(labels) < 2 or not pairs:
        return ""
    metrics = {}
    pair_maps = [_mapping(pair) for pair in pairs]
    pair_lookup = {}
    for pair in pair_maps:
        left = str(pair.get("left", ""))
        right = str(pair.get("right", ""))
        if not left or not right:
            continue
        pair_lookup[(left, right)] = pair
        pair_lookup[(right, left)] = pair
    for metric in PAIRWISE_METRICS:
        values = [_value_at(pair, metric.path) for pair in pair_maps]
        rows = []
        for row_label in labels:
            cells = []
            for col_label in labels:
                if row_label == col_label:
                    cells.append({"display": "self", "background": "", "self": True})
                    continue
                pair = pair_lookup.get((row_label, col_label))
                value = _value_at(pair, metric.path) if pair else None
                cells.append(
                    {
                        "display": _format_value(value, metric),
                        "background": _comparison_background(metric, value, values),
                        "self": False,
                    }
                )
            rows.append({"label": row_label, "cells": cells})
        metrics[metric.key] = {
            "label": metric.label,
            "description": metric.description,
            "rows": rows,
        }
    payload = _json_script_payload({"labels": list(labels), "metrics": metrics})
    return f"""<section>
<h2>Pairwise Comparisons</h2>
<div class="viz-panel">
<div class="viz-controls"><label for="pairwise-metric">Metric </label><select id="pairwise-metric"></select></div>
<div class="table-wrap pairwise-picker-table">
<table>
<thead id="pairwise-metric-head"></thead>
<tbody id="pairwise-metric-body"></tbody>
</table>
</div>
<p class="note" id="pairwise-metric-note"></p>
</div>
<script type="application/json" id="pairwise-table-data">{payload}</script>
<script>{_pairwise_table_script()}</script>
</section>"""


def _comparison_style(
    metric: ComparisonMetric,
    value: object,
    row_values: Sequence[object],
) -> str:
    background = _comparison_background(metric, value, row_values)
    return f' style="background-color: {background};"' if background else ""


def _comparison_background(
    metric: ComparisonMetric,
    value: object,
    row_values: Sequence[object],
) -> str:
    current = _score(metric, value)
    scores = [_score(metric, item) for item in row_values]
    scores = [score for score in scores if score is not None]
    if current is None or len(scores) < 2 or max(scores) == min(scores):
        return ""
    average = sum(scores) / len(scores)
    return _score_background(current - average, metric.full_intensity_at)


def _target_style(
    value: Optional[float],
    row_values: Sequence[object],
    target: float,
    full_intensity_at: float,
) -> str:
    if value is None:
        return ""
    scores = []
    for item in row_values:
        number = _numeric(item)
        if number is not None:
            scores.append(-abs(number - target))
    if len(scores) < 2 or max(scores) == min(scores):
        return ""
    current = -abs(value - target)
    average = sum(scores) / len(scores)
    return _score_style(current - average, full_intensity_at)


def _score_style(delta: float, full_intensity_at: Optional[float]) -> str:
    background = _score_background(delta, full_intensity_at)
    return f' style="background-color: {background};"' if background else ""


def _score_background(delta: float, full_intensity_at: Optional[float]) -> str:
    if abs(delta) <= 1e-12:
        return ""
    scale = full_intensity_at if full_intensity_at and full_intensity_at > 0 else abs(delta)
    strength = min(abs(delta) / scale, 1.0)
    if strength < 0.01:
        return ""
    alpha = 0.08 + 0.44 * strength
    rgb = "35, 134, 54" if delta > 0 else "218, 54, 51"
    return f"rgba({rgb}, {alpha:.3f})"


def _score(metric: ComparisonMetric, value: object) -> Optional[float]:
    if metric.better == "neutral":
        return None
    if metric.better == "lower_length":
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return -float(len(value))
        return None
    number = _numeric(value)
    if number is None:
        return None
    if metric.better == "higher":
        return number
    if metric.better == "lower":
        return -number
    if metric.better == "target":
        target = metric.target if metric.target is not None else 0.0
        return -abs(number - target)
    return None


def _value_at(mapping: Mapping[str, object], path: Sequence[str]) -> object:
    current: object = mapping
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = cast(Mapping[str, object], current).get(key)
    return current


def _format_value(value: object, metric: ComparisonMetric) -> str:
    if value is None:
        return "n/a"
    if metric.value_format == "list":
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return ", ".join(str(item) for item in value) if value else "none"
        return str(value)
    number = _numeric(value)
    if number is None:
        return str(value)
    if metric.value_format == "pct":
        return _format_percent(number)
    if metric.value_format == "integer":
        return str(int(round(number)))
    if metric.value_format == "multiple":
        return f"{number:.2f}x"
    return _format_scalar(number)


def _format_objective_value(value: float, component: ObjectiveComponent) -> str:
    if component.value_format == "pct":
        return _format_percent(value)
    if component.value_format == "integer":
        return str(int(round(value)))
    if component.value_format == "multiple":
        return f"{value:.2f}x"
    return _format_scalar(value)


def _format_percent(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def _format_scalar(value: object) -> str:
    number = _numeric(value)
    if number is None:
        return "n/a" if value is None else str(value)
    if math.isclose(number, round(number), abs_tol=1e-9):
        return f"{int(round(number)):,}"
    if abs(number) >= 100:
        return f"{number:,.1f}"
    if abs(number) >= 10:
        return f"{number:,.2f}"
    return f"{number:,.4f}"


def _feature_vector(summary: Mapping[str, object]) -> Dict[str, float]:
    features = _mapping(summary.get("features"))
    return {str(key): number for key, raw in features.items() if (number := _numeric(raw)) is not None}


def _pca_points(portfolios: Mapping[str, object], labels: Sequence[str]) -> List[Dict[str, object]]:
    vectors = [_feature_vector(_mapping(portfolios.get(label))) for label in labels]
    keys = sorted({key for vector in vectors for key in vector})
    if len(labels) < 2 or not keys:
        return []
    matrix = [[vector.get(key, 0.0) for key in keys] for vector in vectors]
    columns = list(zip(*matrix))
    means = [sum(column) / len(column) for column in columns]
    stds = []
    for column, average in zip(columns, means):
        variance = sum((item - average) ** 2 for item in column) / max(len(column) - 1, 1)
        stds.append(math.sqrt(variance) or 1.0)
    centered = [[(value - means[idx]) / stds[idx] for idx, value in enumerate(row)] for row in matrix]
    try:
        import numpy as np

        array = np.asarray(centered, dtype=float)
        _, _, vt = np.linalg.svd(array, full_matrices=False)
        coordinates = array @ vt[:2].T
        if coordinates.shape[1] == 1:
            coordinates = np.column_stack([coordinates[:, 0], np.zeros(coordinates.shape[0])])
        return [
            {"label": label, "x": float(coordinates[idx, 0]), "y": float(coordinates[idx, 1])}
            for idx, label in enumerate(labels)
        ]
    except Exception:
        return [{"label": label, "x": float(idx), "y": 0.0} for idx, label in enumerate(labels)]


def _json_script_payload(value: object) -> str:
    return escape(json.dumps(value, sort_keys=True), quote=False)


def _heatmap_script() -> str:
    return """
(function(){
const data=JSON.parse(document.getElementById('pairwise-heatmap-data').textContent),select=document.getElementById('heatmap-metric'),svg=document.getElementById('pairwise-heatmap');
Object.entries(data.metrics).forEach(([k,m])=>{const o=document.createElement('option');o.value=k;o.textContent=m.label;select.appendChild(o);});
function fmt(v,f){return f==='pct'?(v*100).toFixed(2)+'%':Number(v).toFixed(4);}
function draw(){const m=data.metrics[select.value],labels=data.labels,n=labels.length;svg.innerHTML='';const w=760,h=360,left=145,top=45,size=Math.min((w-left-25)/n,(h-top-45)/n);svg.setAttribute('viewBox',`0 0 ${w} ${h}`);const vals=m.values.flat().filter(Number.isFinite),min=Math.min(...vals,0),max=Math.max(...vals,1),span=max-min||1;labels.forEach((label,i)=>{let t=document.createElementNS('http://www.w3.org/2000/svg','text');t.setAttribute('x',left+i*size+size/2);t.setAttribute('y',28);t.setAttribute('text-anchor','middle');t.setAttribute('class','viz-muted');t.textContent=label;svg.appendChild(t);t=document.createElementNS('http://www.w3.org/2000/svg','text');t.setAttribute('x',left-8);t.setAttribute('y',top+i*size+size/2+4);t.setAttribute('text-anchor','end');t.setAttribute('class','viz-muted');t.textContent=label;svg.appendChild(t);});for(let i=0;i<n;i++){for(let j=0;j<n;j++){const v=m.values[i][j],q=Math.max(0,Math.min(1,(v-min)/span)),r=Math.round(242-180*q),g=Math.round(246-82*q),b=Math.round(252-12*q),rect=document.createElementNS('http://www.w3.org/2000/svg','rect');rect.setAttribute('x',left+j*size);rect.setAttribute('y',top+i*size);rect.setAttribute('width',size);rect.setAttribute('height',size);rect.setAttribute('class','heat-cell');rect.setAttribute('fill',`rgb(${r},${g},${b})`);const title=document.createElementNS('http://www.w3.org/2000/svg','title');title.textContent=`${labels[i]} vs ${labels[j]}: ${fmt(v,m.format)}`;rect.appendChild(title);svg.appendChild(rect);const text=document.createElementNS('http://www.w3.org/2000/svg','text');text.setAttribute('x',left+j*size+size/2);text.setAttribute('y',top+i*size+size/2+4);text.setAttribute('text-anchor','middle');text.setAttribute('class','viz-label');text.textContent=fmt(v,m.format);svg.appendChild(text);}}}
select.addEventListener('change',draw);draw();
})();"""


def _scatter_script(
    svg_id: str,
    data_id: str,
    x_label: str,
    y_label: str,
    x_unit: str = "",
    y_unit: str = "",
    x_format: str = "number",
    y_format: str = "number",
) -> str:
    x_axis_label = f"{x_label} ({x_unit})" if x_unit else x_label
    y_axis_label = f"{y_label} ({y_unit})" if y_unit else y_label
    return f"""
(function(){{
const pts=JSON.parse(document.getElementById({json.dumps(data_id)}).textContent),svg=document.getElementById({json.dumps(svg_id)}),w=760,h=390,l=88,r=38,t=30,b=76;
const xAxisLabel={json.dumps(x_axis_label)},yAxisLabel={json.dumps(y_axis_label)},xTooltipLabel={json.dumps(x_label)},yTooltipLabel={json.dumps(y_label)},xFormat={json.dumps(x_format)},yFormat={json.dumps(y_format)};
svg.setAttribute('viewBox',`0 0 ${{w}} ${{h}}`);svg.innerHTML='';
if(!pts.length){{return;}}
const xs=pts.map(p=>Number(p.x)).filter(Number.isFinite),ys=pts.map(p=>Number(p.y)).filter(Number.isFinite),cs=pts.map(p=>Number(p.color||0)).filter(Number.isFinite),ss=pts.map(p=>Number(p.size||1)).filter(Number.isFinite);
function extent(values){{let lo=Math.min(...values),hi=Math.max(...values);if(lo===hi){{const pad=Math.abs(lo)||1;lo-=pad;hi+=pad;}}else{{const pad=(hi-lo)*0.08;lo-=pad;hi+=pad;}}return [lo,hi];}}
const [x0,x1]=extent(xs),[y0,y1]=extent(ys),sx=v=>l+(v-x0)/((x1-x0)||1)*(w-l-r),sy=v=>h-b-(v-y0)/((y1-y0)||1)*(h-t-b);
function fmt(value,format){{let v=Number(value);if(!Number.isFinite(v)){{return 'n/a';}}if(Math.abs(v)<1e-12){{v=0;}}if(format==='pct'){{return (v*100).toFixed(2)+'%';}}if(Math.abs(v)>=100){{return v.toFixed(1);}}if(Math.abs(v)>=10){{return v.toFixed(2);}}return v.toFixed(3);}}
function add(name,attrs,text){{const e=document.createElementNS('http://www.w3.org/2000/svg',name);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));if(text!==undefined){{e.textContent=text;}}svg.appendChild(e);return e;}}
function line(x1,y1,x2,y2,klass='axis'){{add('line',{{x1,y1,x2,y2,class:klass}});}}
line(l,h-b,w-r,h-b);line(l,t,l,h-b);
for(let i=0;i<=4;i++){{const xv=x0+(x1-x0)*i/4,x=sx(xv);line(x,h-b,x,h-b+5);add('text',{{x,y:h-b+20,'text-anchor':'middle',class:'viz-muted'}},fmt(xv,xFormat));const yv=y0+(y1-y0)*i/4,y=sy(yv);line(l-5,y,l,y);add('text',{{x:l-9,y:y+4,'text-anchor':'end',class:'viz-muted'}},fmt(yv,yFormat));}}
add('text',{{x:w/2,y:h-18,'text-anchor':'middle',class:'viz-label'}},xAxisLabel);
add('text',{{x:20,y:h/2,transform:`rotate(-90 20 ${{h/2}})`,'text-anchor':'middle',class:'viz-label'}},yAxisLabel);
const cmin=Math.min(...cs,0),cmax=Math.max(...cs,0),smin=Math.min(...ss,1),smax=Math.max(...ss,1);
pts.forEach(p=>{{const px=Number(p.x),py=Number(p.y);if(!Number.isFinite(px)||!Number.isFinite(py)){{return;}}const ci=(Number(p.color||0)-cmin)/((cmax-cmin)||1),rad=6+10*(Number(p.size||1)-smin)/((smax-smin)||1),c=add('circle',{{cx:sx(px),cy:sy(py),r:rad,fill:`rgb(${{Math.round(48+180*ci)}}, ${{Math.round(112-60*ci)}}, 196)`,'fill-opacity':'0.72'}});const title=document.createElementNS('http://www.w3.org/2000/svg','title');title.textContent=`${{p.label}}: ${{xTooltipLabel}}=${{fmt(px,xFormat)}}, ${{yTooltipLabel}}=${{fmt(py,yFormat)}}`;c.appendChild(title);add('text',{{x:sx(px)+rad+3,y:sy(py)+4,class:'viz-muted'}},p.label);}});
}})();"""


def _pairwise_table_script() -> str:
    return """
(function(){
const data=JSON.parse(document.getElementById('pairwise-table-data').textContent),select=document.getElementById('pairwise-metric'),thead=document.getElementById('pairwise-metric-head'),tbody=document.getElementById('pairwise-metric-body'),note=document.getElementById('pairwise-metric-note');
Object.entries(data.metrics).forEach(([key,metric])=>{const option=document.createElement('option');option.value=key;option.textContent=metric.label;select.appendChild(option);});
function draw(){const metric=data.metrics[select.value];thead.innerHTML='';tbody.innerHTML='';note.textContent=metric.description||'';const headRow=document.createElement('tr'),corner=document.createElement('th');corner.textContent=metric.label;headRow.appendChild(corner);data.labels.forEach(label=>{const th=document.createElement('th');th.textContent=label;headRow.appendChild(th);});thead.appendChild(headRow);metric.rows.forEach(row=>{const tr=document.createElement('tr'),label=document.createElement('th');label.textContent=row.label;tr.appendChild(label);row.cells.forEach(cell=>{const td=document.createElement('td');td.textContent=cell.display;if(cell.background){td.style.backgroundColor=cell.background;}if(cell.self){td.className='pairwise-self';}tr.appendChild(td);});tbody.appendChild(tr);});}
select.addEventListener('change',draw);draw();
})();"""


def _has_harvest_replay(portfolios: Mapping[str, object]) -> bool:
    return any(isinstance(_mapping(summary).get("harvest_replay"), Mapping) for summary in portfolios.values())


def _mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return cast(Mapping[str, object], value)
    return {}


def _list(value: object) -> List[object]:
    if isinstance(value, list):
        return cast(List[object], value)
    return []


def _numeric(value: object) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return None
    else:
        return None
    return number if math.isfinite(number) else None


def _numeric_or_zero(value: object) -> float:
    number = _numeric(value)
    return number if number is not None else 0.0
