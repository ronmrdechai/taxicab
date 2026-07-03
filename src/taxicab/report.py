from __future__ import annotations

import math
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence


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
        _render_pairwise_table(comparison),
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


def _render_pairwise_table(comparison: Mapping[str, object]) -> str:
    pairs = _list(comparison.get("pairwise"))
    if not pairs:
        return ""
    metric_values = {
        metric.key: [_value_at(_mapping(pair), metric.path) for pair in pairs]
        for metric in PAIRWISE_METRICS
    }
    header = "".join(f"<th>{escape(metric.label)}</th>" for metric in PAIRWISE_METRICS)
    rows = []
    for pair in pairs:
        pair_map = _mapping(pair)
        cells = []
        for metric in PAIRWISE_METRICS:
            value = _value_at(pair_map, metric.path)
            style = _comparison_style(metric, value, metric_values[metric.key])
            cells.append(f"<td{style}>{escape(_format_value(value, metric))}</td>")
        label = "{} vs {}".format(pair_map.get("left", ""), pair_map.get("right", ""))
        rows.append(f"<tr><th>{escape(label)}</th>{''.join(cells)}</tr>")
    return f"""<section>
<h2>Pairwise Comparisons</h2>
<div class="table-wrap">
<table>
<thead><tr><th>Pair</th>{header}</tr></thead>
<tbody>
{''.join(rows)}
</tbody>
</table>
</div>
</section>"""


def _comparison_style(
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
    return _score_style(current - average, metric.full_intensity_at)


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
    if abs(delta) <= 1e-12:
        return ""
    scale = full_intensity_at if full_intensity_at and full_intensity_at > 0 else abs(delta)
    strength = min(abs(delta) / scale, 1.0)
    if strength < 0.01:
        return ""
    alpha = 0.08 + 0.44 * strength
    rgb = "35, 134, 54" if delta > 0 else "218, 54, 51"
    return f' style="background-color: rgba({rgb}, {alpha:.3f});"'


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
        current = current.get(key)
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
    return _format_scalar(number)


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


def _has_harvest_replay(portfolios: Mapping[str, object]) -> bool:
    return any(isinstance(_mapping(summary).get("harvest_replay"), Mapping) for summary in portfolios.values())


def _mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    return {}


def _list(value: object) -> List[object]:
    if isinstance(value, list):
        return value
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
