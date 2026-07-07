from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, cast


OUTPUT_SCHEMA_VERSION = 2

MetricCalculator = Callable[[Mapping[str, object]], object]


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    value_format: str
    description: str
    better: str = "neutral"
    target: Optional[float] = None
    full_intensity_at: Optional[float] = None
    group: str = field(default="", init=False)
    scope: str = field(default="", init=False)
    namespace: str = field(default="", init=False)
    usages: Tuple[str, ...] = ()
    diagonal_value: Optional[float] = None
    calculate: Optional[MetricCalculator] = None


@dataclass(frozen=True)
class ObjectiveComponent:
    key: str
    label: str
    unit: str
    value_format: str
    description: str
    usages: Tuple[str, ...] = ("objective_table",)
    calculate: Optional[MetricCalculator] = None


class MetricSetMeta(type):
    sets: List[type["MetricSet"]] = []
    by_scope: Dict[str, List[type["MetricSet"]]] = {}

    def __new__(mcls, name: str, bases: Tuple[type, ...], namespace: Dict[str, object]) -> type:
        cls = super().__new__(mcls, name, bases, namespace)
        if namespace.get("abstract", False):
            return cls
        group = str(getattr(cls, "group"))
        scope = str(getattr(cls, "scope"))
        default_namespace = str(getattr(cls, "namespace", "portfolio"))
        stamped = []
        for spec in cast(Sequence[MetricSpec], getattr(cls, "metrics", ())):
            stamped_spec = replace(spec)
            object.__setattr__(stamped_spec, "group", group)
            object.__setattr__(stamped_spec, "scope", scope)
            object.__setattr__(stamped_spec, "namespace", default_namespace)
            stamped.append(stamped_spec)
        setattr(cls, "metrics", tuple(stamped))
        mcls.sets.append(cast(type["MetricSet"], cls))
        mcls.by_scope.setdefault(scope, []).append(cast(type["MetricSet"], cls))
        return cls


class MetricSet(metaclass=MetricSetMeta):
    abstract = True
    scope = ""
    group = ""
    namespace = "portfolio"
    metrics: Tuple[MetricSpec, ...] = ()


def require_schema_version(mapping: Mapping[str, object], artifact: str) -> None:
    version = mapping.get("version")
    if version != OUTPUT_SCHEMA_VERSION:
        raise ValueError(f"unsupported {artifact} schema version {version}; expected version {OUTPUT_SCHEMA_VERSION}")


def metric_group(portfolio: Mapping[str, object], group: str) -> Mapping[str, object]:
    metrics = _mapping(portfolio.get("metrics"))
    return _mapping(metrics.get(group))


def metric_specs(scope: Optional[str] = None) -> Tuple[MetricSpec, ...]:
    if scope is None:
        sets = MetricSetMeta.sets
    else:
        sets = MetricSetMeta.by_scope.get(scope, [])
    return tuple(spec for metric_set in sets for spec in metric_set.metrics)


def construction_metric_specs() -> Tuple[MetricSpec, ...]:
    return metric_specs("construction")


def harvest_replay_metric_specs() -> Tuple[MetricSpec, ...]:
    return metric_specs("harvest_replay")


def portfolio_comparison_metric_specs() -> Tuple[MetricSpec, ...]:
    return metric_specs("portfolio_comparison")


def pairwise_metric_specs() -> Tuple[MetricSpec, ...]:
    return metric_specs("pairwise")


def build_construction_metrics(context: Mapping[str, object]) -> Dict[str, object]:
    return build_metric_map(construction_metric_specs(), context)


def build_selection_metrics(diagnostics: Mapping[str, object]) -> Dict[str, object]:
    return dict(diagnostics)


def build_constraint_metrics(warnings: Sequence[str], violations: Sequence[str]) -> Dict[str, object]:
    return {"constraint_warnings": list(warnings), "constraint_violations": list(violations)}


def build_harvest_replay_metrics(replay: Mapping[str, object]) -> Dict[str, object]:
    return build_metric_map(harvest_replay_metric_specs(), replay)


def build_harvest_replay_summary(replay: Mapping[str, object]) -> Dict[str, object]:
    summary = {key: replay[key] for key in HARVEST_REPLAY_METADATA_KEYS if key in replay}
    summary["metrics"] = build_harvest_replay_metrics(replay)
    return summary


def build_harvest_replay_delta_metrics(left: Mapping[str, object], right: Mapping[str, object]) -> Dict[str, object]:
    deltas: Dict[str, object] = {}
    for spec in harvest_replay_metric_specs():
        left_value = _numeric(_raw_metric_value(left, spec.key))
        right_value = _numeric(_raw_metric_value(right, spec.key))
        if left_value is not None and right_value is not None:
            deltas[spec.key] = left_value - right_value
    return deltas


def build_portfolio_comparison_metrics(context: Mapping[str, object]) -> Dict[str, object]:
    return build_metric_map(portfolio_comparison_metric_specs(), context)


def build_pairwise_metrics(context: Mapping[str, object]) -> Dict[str, object]:
    return build_metric_map(pairwise_metric_specs(), context)


def build_objective_metrics(context: Mapping[str, object]) -> Dict[str, object]:
    values: Dict[str, object] = {}
    for component in OBJECTIVE_COMPONENTS:
        value = component.calculate(context) if component.calculate is not None else context.get(component.key)
        if value is not None:
            values[component.key] = value
    return values


def build_metric_map(specs: Iterable[MetricSpec], context: Mapping[str, object]) -> Dict[str, object]:
    values: Dict[str, object] = {}
    for spec in specs:
        value = spec.calculate(context) if spec.calculate is not None else context.get(spec.key)
        if value is not None:
            values[spec.key] = value
    return values


def specs_for_usage(specs: Iterable[MetricSpec], usage: str) -> List[MetricSpec]:
    return [spec for spec in specs if usage in spec.usages]


def portfolio_report_metrics(include_harvest_replay: bool) -> List[MetricSpec]:
    metrics = specs_for_usage(portfolio_comparison_metric_specs(), "portfolio_table")
    if include_harvest_replay:
        metrics.extend(specs_for_usage(harvest_replay_metric_specs(), "portfolio_table"))
    return metrics


def pairwise_table_metrics() -> List[MetricSpec]:
    return specs_for_usage(pairwise_metric_specs(), "pairwise_table")


def pairwise_heatmap_metrics() -> List[MetricSpec]:
    return specs_for_usage(pairwise_metric_specs(), "pairwise_heatmap")


def construct_cli_metrics(usage: str) -> List[MetricSpec]:
    return specs_for_usage(construction_metric_specs(), usage) + specs_for_usage(harvest_replay_metric_specs(), usage)


def comparison_cli_metrics(usage: str) -> List[MetricSpec]:
    return (
        specs_for_usage(portfolio_comparison_metric_specs(), usage)
        + specs_for_usage(harvest_replay_metric_specs(), usage)
        + specs_for_usage(pairwise_metric_specs(), usage)
    )


def all_metric_specs() -> List[MetricSpec]:
    return list(metric_specs())


def metric_value(container: Mapping[str, object], spec: MetricSpec) -> object:
    if spec.namespace == "harvest_replay":
        return _mapping(_mapping(container.get("harvest_replay")).get("metrics")).get(spec.key)
    if spec.namespace == "pairwise_harvest_replay_delta":
        return _mapping(_mapping(container.get("harvest_replay_delta")).get("metrics")).get(spec.key)
    if spec.namespace == "objective":
        return _mapping(_mapping(container.get("objective_decomposition")).get("metrics")).get(spec.key)
    return _mapping(container.get("metrics")).get(spec.key)


def format_metric_value(value: object, spec: MetricSpec) -> str:
    if value is None:
        return "n/a"
    if spec.value_format == "list":
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return ", ".join(str(item) for item in value) if value else "none"
        return str(value)
    if spec.value_format == "mapping":
        if isinstance(value, Mapping):
            return ", ".join(f"{key}: {format_scalar(raw)}" for key, raw in sorted(value.items())) or "none"
        return str(value)
    number = _numeric(value)
    if number is None:
        return str(value)
    if spec.value_format == "pct":
        return format_percent(number)
    if spec.value_format == "integer":
        return str(int(round(number)))
    if spec.value_format == "multiple":
        return f"{number:.2f}x"
    return format_scalar(number)


def format_percent(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def format_scalar(value: object) -> str:
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


def comparison_background(spec: MetricSpec, value: object, row_values: Sequence[object]) -> str:
    current = metric_score(spec, value)
    scores = [metric_score(spec, item) for item in row_values]
    scores = [score for score in scores if score is not None]
    if current is None or len(scores) < 2 or max(scores) == min(scores):
        return ""
    average = sum(scores) / len(scores)
    return score_background(current - average, spec.full_intensity_at)


def metric_score(spec: MetricSpec, value: object) -> Optional[float]:
    if spec.better == "neutral":
        return None
    if spec.better == "lower_length":
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return -float(len(value))
        return None
    number = _numeric(value)
    if number is None:
        return None
    if spec.better == "higher":
        return number
    if spec.better == "lower":
        return -number
    if spec.better == "target":
        target = spec.target if spec.target is not None else 0.0
        return -abs(number - target)
    return None


def score_background(delta: float, full_intensity_at: Optional[float]) -> str:
    if abs(delta) <= 1e-12:
        return ""
    scale = full_intensity_at if full_intensity_at and full_intensity_at > 0 else abs(delta)
    strength = min(abs(delta) / scale, 1.0)
    if strength < 0.01:
        return ""
    alpha = 0.08 + 0.44 * strength
    rgb = "35, 134, 54" if delta > 0 else "218, 54, 51"
    return f"rgba({rgb}, {alpha:.3f})"


def numeric(value: object) -> Optional[float]:
    return _numeric(value)


def _mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return cast(Mapping[str, object], value)
    return {}


def _raw_metric_value(source: Mapping[str, object], key: str) -> object:
    metrics = _mapping(source.get("metrics"))
    if key in metrics:
        return metrics[key]
    return source.get(key)


def _metric_group_value(source: Mapping[str, object], group: str, key: str) -> object:
    return _mapping(_mapping(source.get("metrics")).get(group)).get(key)


def _ctx(key: str) -> MetricCalculator:
    return lambda context: context.get(key)


def _ctx_path(*path: str) -> MetricCalculator:
    def calculate(context: Mapping[str, object]) -> object:
        current: object = context
        for key in path:
            if not isinstance(current, Mapping):
                return None
            current = cast(Mapping[str, object], current).get(key)
        return current

    return calculate


def _ctx_first_number(*keys: str) -> MetricCalculator:
    def calculate(context: Mapping[str, object]) -> object:
        for key in keys:
            number = _numeric(context.get(key))
            if number is not None:
                return number
        return None

    return calculate


def _construction_metric(key: str) -> MetricCalculator:
    return lambda context: _metric_group_value(_mapping(context.get("portfolio")), "construction", key)


def _harvest_metric(key: str) -> MetricCalculator:
    return lambda context: _mapping(context.get("harvest_replay_metrics")).get(key)


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


def _sector_penalty(context: Mapping[str, object]) -> object:
    return _numeric(context.get("sector_abs_error")) or 0.0


def _tracking_constraint_slack(context: Mapping[str, object]) -> object:
    target = _numeric(context.get("error_margin")) or 0.0
    tracking = _numeric(context.get("tracking_error")) or 0.0
    return max(target - tracking, 0.0)


def _objective_tax_benefit(context: Mapping[str, object]) -> object:
    target_tax = _numeric(context.get("target_tax_alpha")) or 0.0
    tax_alpha = _numeric(context.get("tax_alpha")) or 0.0
    tax_delta = tax_alpha - target_tax
    if context.get("tax_alpha_mode") == "at-least":
        tax_delta = min(tax_delta, 0.0)
    return -abs(tax_delta)


def _objective_tracking_error_penalty(context: Mapping[str, object]) -> object:
    error_margin = max(_numeric(context.get("error_margin")) or 0.005, 0.005)
    tracking = _numeric(context.get("tracking_error")) or 0.0
    return tracking / error_margin if error_margin else 0.0


def _objective_factor_penalty(context: Mapping[str, object]) -> object:
    return abs((_numeric(context.get("beta")) or 1.0) - 1.0)


def _objective_wash_sale_penalty(context: Mapping[str, object]) -> object:
    return float(_numeric(context.get("skipped_constraint_violation")) or 0.0) + float(
        _numeric(context.get("skipped_no_replacement")) or 0.0
    )


class ConstructionMetricSet(MetricSet):
    scope = "construction"
    group = "Construction"
    metrics = (
        MetricSpec(key="beta", label="Beta", value_format="number", description="Weighted construction-time beta estimate versus the benchmark.", better="target", target=1.0, full_intensity_at=0.10, usages=("construct_cli_primary",)),
        MetricSpec(key="tracking_error", label="Tracking error", value_format="pct", description="Annualized construction-time tracking error versus the benchmark proxy.", better="lower", full_intensity_at=0.03, usages=("construct_cli_primary", "construct_cli_fit")),
        MetricSpec(key="tax_alpha", label="Tax alpha", value_format="pct", description="Weighted tax-alpha objective value used by construction.", better="higher", full_intensity_at=0.02, usages=("construct_cli_primary",)),
        MetricSpec(key="simulated_tax_alpha", label="Simulated tax alpha", value_format="pct", description="Construction-time simulated tax-alpha diagnostic from optimizer inputs; not tax or investment advice.", better="higher", full_intensity_at=0.02, usages=("construct_cli_primary",)),
        MetricSpec(key="gross_harvestable_loss_rate", label="Gross harvestable loss rate", value_format="pct", description="Annualized pre-tax loss opportunity diagnostic across historical harvest windows.", better="higher", full_intensity_at=0.02, usages=("construct_cli_primary",)),
        MetricSpec(key="estimated_tax_loss_alpha", label="Estimated tax loss alpha", value_format="pct", description="Construction target-aligned tax-loss alpha estimate."),
        MetricSpec(key="sectors", label="Sectors", value_format="mapping", description="Portfolio sector weights from construction."),
        MetricSpec(key="max_weight", label="Max weight", value_format="pct", description="Largest constructed position weight.", better="lower", usages=("construct_cli_fit",)),
        MetricSpec(key="effective_number_of_names", label="Effective names", value_format="number", description="Inverse Herfindahl effective number of constructed names.", better="higher", usages=("construct_cli_fit",)),
        MetricSpec(key="tracking_error_observations", label="Tracking observations", value_format="integer", description="Return observations used by the construction tracking model."),
        MetricSpec(key="price_benchmark_tracking_error", label="Price benchmark tracking error", value_format="pct", description="Tracking error measured against the price benchmark window."),
        MetricSpec(key="active_share", label="Active share", value_format="pct", description="Weighted difference between constructed holdings and the benchmark holdings.", better="lower", usages=("construct_cli_fit",)),
        MetricSpec(key="sector_abs_error", label="Sector absolute error", value_format="pct", description="Total absolute sector-weight distance from the benchmark sectors.", better="lower", usages=("construct_cli_sector",)),
    )


class HarvestReplayMetricSet(MetricSet):
    scope = "harvest_replay"
    group = "Harvest replay"
    namespace = "harvest_replay"
    metrics = (
        MetricSpec(key="portfolio_simulated_tax_alpha", label="Simulated tax alpha", value_format="pct", description="Harvest replay simulated annual tax-alpha diagnostic from the input assumptions; not tax or investment advice.", better="higher", full_intensity_at=0.02, usages=("portfolio_table", "construct_cli_harvest", "comparison_cli_harvest", "comparison_cli_delta", "frontier_y")),
        MetricSpec(key="portfolio_realized_loss_rate", label="Realized loss rate", value_format="pct", description="Annualized realized loss rate produced by the harvest replay assumptions.", better="higher", full_intensity_at=0.02, usages=("portfolio_table", "construct_cli_harvest", "comparison_cli_harvest", "comparison_cli_delta", "frontier_y")),
        MetricSpec(key="portfolio_harvest_active_return", label="Harvest active return", value_format="pct", description="Annualized active return of the harvest replay portfolio versus the benchmark.", better="higher", full_intensity_at=0.03, usages=("portfolio_table", "construct_cli_path", "comparison_cli_harvest", "comparison_cli_delta")),
        MetricSpec(key="portfolio_harvest_tracking_error", label="Harvest tracking error", value_format="pct", description="Annualized tracking error from the harvest replay path.", better="lower", full_intensity_at=0.03, usages=("portfolio_table", "construct_cli_path", "comparison_cli_harvest")),
        MetricSpec(key="portfolio_harvest_beta", label="Harvest beta", value_format="number", description="Harvest replay beta versus the benchmark.", better="target", target=1.0, full_intensity_at=0.10, usages=("portfolio_table", "construct_cli_path")),
        MetricSpec(key="portfolio_harvest_correlation", label="Harvest correlation", value_format="number", description="Harvest replay return correlation versus the benchmark.", usages=("construct_cli_path",)),
        MetricSpec(key="portfolio_harvest_annualized_return", label="Harvest annualized return", value_format="pct", description="Annualized return of the replayed portfolio path."),
        MetricSpec(key="benchmark_annualized_return", label="Benchmark annualized return", value_format="pct", description="Annualized benchmark return over the harvest replay window."),
        MetricSpec(key="portfolio_harvest_observations", label="Harvest observations", value_format="integer", description="Return observations in the harvest replay path."),
        MetricSpec(key="annual_realized_loss", label="Annual realized loss", value_format="number", description="Annualized realized loss amount in replay units."),
        MetricSpec(key="total_realized_loss", label="Total realized loss", value_format="number", description="Total realized loss amount in replay units."),
        MetricSpec(key="total_tax_benefit", label="Total tax benefit", value_format="number", description="Total simulated tax benefit before replay costs."),
        MetricSpec(key="total_net_tax_benefit", label="Total net tax benefit", value_format="number", description="Net simulated tax benefit after replay transaction and replacement costs; not tax advice.", better="higher", full_intensity_at=0.05, usages=("portfolio_table",)),
        MetricSpec(key="terminal_after_tax_wealth_difference", label="Terminal after-tax wealth difference", value_format="pct", description="Replay terminal after-tax wealth difference versus the comparison baseline.", better="higher", full_intensity_at=0.05, usages=("portfolio_table", "comparison_cli_delta")),
        MetricSpec(key="total_transaction_cost", label="Total transaction cost", value_format="number", description="Total replay transaction cost under the configured assumptions.", better="lower", full_intensity_at=0.02, usages=("portfolio_table", "frontier_color")),
        MetricSpec(key="total_replacement_cost", label="Total replacement cost", value_format="number", description="Total replay replacement-trade cost under the configured assumptions.", better="lower", full_intensity_at=0.02, usages=("portfolio_table",)),
        MetricSpec(key="immediate_tax_savings_rate", label="Immediate tax savings rate", value_format="pct", description="Annualized immediate tax savings rate before terminal liquidation effects.", usages=("construct_cli_harvest",)),
        MetricSpec(key="immediate_net_tax_savings_rate", label="Immediate net tax savings rate", value_format="pct", description="Annualized immediate tax savings rate after replay costs."),
        MetricSpec(key="full_liquidation_after_tax_alpha", label="Full liquidation after-tax alpha", value_format="pct", description="Annualized after-tax alpha after applying the full liquidation assumption."),
        MetricSpec(key="harvest_count", label="Harvest count", value_format="integer", description="Number of harvest trades executed by the replay model.", usages=("portfolio_table", "construct_cli_harvest", "comparison_cli_harvest")),
        MetricSpec(key="rebalance_count", label="Rebalance count", value_format="integer", description="Number of rebalances executed by the replay model.", usages=("portfolio_table", "construct_cli_harvest", "comparison_cli_harvest")),
        MetricSpec(key="skipped_no_replacement", label="Skipped no replacement", value_format="integer", description="Skipped harvest attempts where no replacement was available."),
        MetricSpec(key="skipped_nonpositive_net_benefit", label="Skipped nonpositive net benefit", value_format="integer", description="Skipped harvest attempts whose estimated net benefit was not positive."),
        MetricSpec(key="skipped_constraint_violation", label="Skipped constraint violation", value_format="integer", description="Skipped harvest attempts that would violate replay constraints."),
    )


class PortfolioPerformanceMetricSet(MetricSet):
    scope = "portfolio_comparison"
    group = "Performance"
    metrics = (
        MetricSpec(key="annualized_return", label="Annualized return", value_format="pct", description="Annualized portfolio return over the comparison price window.", better="higher", full_intensity_at=0.05, usages=("portfolio_table", "comparison_cli_portfolio"), calculate=_ctx_path("returns", "annualized_return")),
        MetricSpec(key="active_return", label="Active return vs benchmark", value_format="pct", description="Portfolio annualized return minus benchmark annualized return over the comparison window.", better="higher", full_intensity_at=0.03, usages=("portfolio_table",), calculate=_ctx_path("returns", "benchmark_annualized_active_return")),
        MetricSpec(key="cumulative_return", label="Cumulative return", value_format="pct", description="Total compounded portfolio return over the comparison price window.", better="higher", full_intensity_at=0.20, usages=("portfolio_table",), calculate=_ctx_path("returns", "cumulative_return")),
        MetricSpec(key="max_drawdown", label="Max drawdown", value_format="pct", description="Largest peak-to-trough decline over the comparison window; values closer to zero rank better.", better="target", target=0.0, full_intensity_at=0.10, usages=("portfolio_table",), calculate=_ctx_path("returns", "max_drawdown")),
    )


class PortfolioBenchmarkMetricSet(MetricSet):
    scope = "portfolio_comparison"
    group = "Benchmark fit"
    metrics = (
        MetricSpec(key="tracking_error", label="Tracking error vs benchmark", value_format="pct", description="Annualized volatility of active returns versus the benchmark.", better="lower", full_intensity_at=0.03, usages=("portfolio_table", "comparison_cli_portfolio", "frontier_x"), calculate=_ctx_first_number("construction_tracking_error", "benchmark_tracking_error")),
        MetricSpec(key="beta", label="Beta vs benchmark", value_format="number", description="Regression beta of portfolio returns against benchmark returns; values closer to 1.0 rank better.", better="target", target=1.0, full_intensity_at=0.10, usages=("portfolio_table", "comparison_cli_portfolio"), calculate=_ctx_first_number("construction_beta", "benchmark_beta")),
        MetricSpec(key="active_share_to_index", label="Active share to index", value_format="pct", description="Weighted difference between portfolio holdings and index holdings.", better="lower", full_intensity_at=0.15, usages=("portfolio_table", "comparison_cli_portfolio"), calculate=_ctx("active_share_to_index")),
        MetricSpec(key="weighted_overlap_with_index", label="Weighted overlap with index", value_format="pct", description="Share of portfolio weight overlapping index holdings after ticker matching.", better="higher", full_intensity_at=0.15, usages=("portfolio_table",), calculate=_ctx("weighted_overlap_with_index")),
    )


class PortfolioTaxMetricSet(MetricSet):
    scope = "portfolio_comparison"
    group = "Tax metrics"
    metrics = (
        MetricSpec(key="estimated_tax_alpha", label="Estimated tax alpha", value_format="pct", description="Construction-time simulated tax-alpha estimate from optimizer inputs, distinct from harvest replay results; not tax or investment advice.", better="higher", full_intensity_at=0.02, usages=("portfolio_table", "frontier_y"), calculate=_construction_metric("simulated_tax_alpha")),
    )


class PortfolioSectorMetricSet(MetricSet):
    scope = "portfolio_comparison"
    group = "Sector fit"
    metrics = (
        MetricSpec(key="sector_active_share_to_index", label="Sector active share", value_format="pct", description="Total absolute difference between portfolio sector weights and index sector weights.", better="lower", full_intensity_at=0.05, usages=("portfolio_table", "comparison_cli_portfolio"), calculate=_ctx("sector_active_share_to_index")),
        MetricSpec(key="sector_similarity_to_index", label="Sector similarity", value_format="number", description="Similarity score between portfolio sector weights and index sector weights.", better="higher", full_intensity_at=0.05, usages=("portfolio_table", "comparison_cli_portfolio"), calculate=_ctx("sector_similarity_to_index")),
        MetricSpec(key="sector_overlap_to_index", label="Sector overlap with index", value_format="pct", description="Sector-weight overlap between the portfolio and index.", better="higher", full_intensity_at=0.05, usages=("portfolio_table",), calculate=_ctx("sector_overlap_to_index")),
    )


class PortfolioDataMetricSet(MetricSet):
    scope = "portfolio_comparison"
    group = "Data quality"
    metrics = (
        MetricSpec(key="covered_price_weight", label="Covered price weight", value_format="pct", description="Share of portfolio weight with usable price history in the comparison window.", better="higher", full_intensity_at=0.05, usages=("portfolio_table",), calculate=_ctx("covered_price_weight")),
        MetricSpec(key="missing_price_tickers", label="Missing price tickers", value_format="list", description="Tickers missing usable price history for this comparison.", better="lower_length", full_intensity_at=5.0, usages=("portfolio_table",), calculate=_ctx("missing_price_tickers")),
        MetricSpec(key="position_count", label="Position count", value_format="integer", description="Number of portfolio positions included in the comparison summary.", usages=("portfolio_table",), calculate=_ctx("position_count")),
    )


class PortfolioFeatureMetricSet(MetricSet):
    scope = "portfolio_comparison"
    group = "Feature"
    metrics = (
        MetricSpec(key="realized_loss_rate", label="Realized loss rate", value_format="pct", description="Harvest replay realized loss rate used as a feature.", calculate=_harvest_metric("portfolio_realized_loss_rate")),
        MetricSpec(key="turnover", label="Turnover", value_format="pct", description="Replay transaction-cost proxy used as chart color.", usages=("frontier_color",), calculate=_harvest_metric("total_transaction_cost")),
        MetricSpec(key="effective_names", label="Effective names", value_format="number", description="Effective number of portfolio names used as chart size.", usages=("frontier_size",), calculate=_ctx("effective_names")),
        MetricSpec(key="max_weight", label="Max weight", value_format="pct", description="Largest portfolio position weight used as a feature.", calculate=_ctx("max_weight")),
        MetricSpec(key="sector_abs_error", label="Sector absolute error", value_format="pct", description="Total absolute sector distance used as a feature.", calculate=_ctx("sector_abs_error")),
        MetricSpec(key="tracking_constraint_slack", label="Tracking constraint slack", value_format="pct", description="Remaining tracking-error slack versus the target.", calculate=_tracking_constraint_slack),
    )


class PairwiseMetricSet(MetricSet):
    scope = "pairwise"
    group = "Pairwise"
    namespace = "pairwise"
    metrics = (
        MetricSpec(key="ticker_overlap_count", label="Ticker overlap", value_format="integer", description="Number of tickers shared by both portfolios in the pair.", usages=("pairwise_table", "comparison_cli_pairwise"), calculate=_ctx("ticker_overlap_count")),
        MetricSpec(key="weight_cosine_similarity", label="Cosine similarity of weights", value_format="number", description="Cosine similarity between pair portfolio weights.", usages=("pairwise_heatmap",), diagonal_value=1.0, calculate=_ctx("weight_cosine_similarity")),
        MetricSpec(key="ticker_jaccard", label="Ticker Jaccard", value_format="number", description="Ticker-set Jaccard similarity between the pair.", calculate=_ctx("ticker_jaccard")),
        MetricSpec(key="weighted_overlap", label="Weighted overlap", value_format="pct", description="Sum of overlapping weights between the row and column portfolios.", better="higher", full_intensity_at=0.15, usages=("pairwise_table", "comparison_cli_pairwise"), calculate=_ctx("weighted_overlap")),
        MetricSpec(key="active_share", label="Pair active share", value_format="pct", description="Weighted holding difference between the row and column portfolios.", better="lower", full_intensity_at=0.15, usages=("pairwise_table", "pairwise_heatmap"), diagonal_value=0.0, calculate=_ctx("active_share")),
        MetricSpec(key="sector_abs_distance", label="Sector distance", value_format="pct", description="Total absolute difference between the pair's sector weights.", better="lower", full_intensity_at=0.05, usages=("pairwise_table", "pairwise_heatmap", "comparison_cli_pairwise"), diagonal_value=0.0, calculate=_ctx("sector_abs_distance")),
        MetricSpec(key="sector_active_share", label="Sector active share", value_format="pct", description="Half the absolute sector-weight distance between the pair.", calculate=_ctx("sector_active_share")),
        MetricSpec(key="sector_similarity", label="Sector similarity", value_format="number", description="Similarity score between the pair's sector weights.", better="higher", full_intensity_at=0.05, usages=("pairwise_table",), diagonal_value=1.0, calculate=_ctx("sector_similarity")),
        MetricSpec(key="sector_overlap", label="Sector overlap", value_format="pct", description="Sector-weight overlap between the pair.", calculate=_ctx("sector_overlap")),
        MetricSpec(key="factor_abs_distance", label="Factor exposure distance", value_format="number", description="Total absolute distance between fallback factor exposures.", better="lower", usages=("pairwise_heatmap",), diagonal_value=0.0, calculate=_ctx("factor_abs_distance")),
        MetricSpec(key="tax_lot_action_overlap", label="Tax-lot action overlap", value_format="number", description="Jaccard overlap of tax-lot action tickers between the pair.", usages=("pairwise_heatmap",), diagonal_value=0.0, calculate=_ctx("tax_lot_action_overlap")),
        MetricSpec(key="correlation", label="Return correlation", value_format="number", description="Correlation between the pair's return series over the comparison window.", better="higher", full_intensity_at=0.10, usages=("pairwise_table", "comparison_cli_pairwise"), calculate=_ctx_path("returns", "correlation")),
        MetricSpec(key="tracking_error", label="Pair tracking error", value_format="pct", description="Annualized tracking-error distance between the pair's returns.", better="lower", full_intensity_at=0.03, usages=("pairwise_table", "pairwise_heatmap", "comparison_cli_pairwise"), diagonal_value=0.0, calculate=_ctx_path("returns", "tracking_error")),
        MetricSpec(key="annualized_active_return", label="Pair active return", value_format="pct", description="Annualized active return of the left portfolio versus the right portfolio.", calculate=_ctx_path("returns", "annualized_active_return")),
        MetricSpec(key="cumulative_return_difference", label="Pair cumulative return difference", value_format="pct", description="Cumulative return difference of the left portfolio versus the right portfolio.", calculate=_ctx_path("returns", "cumulative_return_difference")),
    )


OBJECTIVE_COMPONENTS: Tuple[ObjectiveComponent, ...] = (
    ObjectiveComponent(key="tracking_error_penalty", label="Tracking error / target", unit="multiple", value_format="multiple", description="Tracking error divided by the target error margin.", calculate=_objective_tracking_error_penalty),
    ObjectiveComponent(key="sector_penalty", label="Sector distance", unit="portfolio weight", value_format="pct", description="Absolute sector-weight distance from the index.", calculate=_sector_penalty),
    ObjectiveComponent(key="factor_penalty", label="Beta distance", unit="beta points", value_format="number", description="Absolute distance from beta 1.0.", calculate=_objective_factor_penalty),
    ObjectiveComponent(key="concentration_penalty", label="Max position weight", unit="portfolio weight", value_format="pct", description="Largest single-position weight.", calculate=_ctx("max_weight")),
    ObjectiveComponent(key="transaction_cost", label="Transaction cost", unit="portfolio value", value_format="pct", description="Replay transaction cost as a share of portfolio value.", calculate=_ctx("total_transaction_cost")),
    ObjectiveComponent(key="tax_benefit", label="Tax-alpha shortfall benefit", unit="annualized return", value_format="pct", description="Negative values reduce the objective.", calculate=_objective_tax_benefit),
    ObjectiveComponent(key="wash_sale_penalty", label="Skipped harvest count", unit="count", value_format="integer", description="Skipped harvest attempts from replacement or constraint failures.", calculate=_objective_wash_sale_penalty),
    ObjectiveComponent(key="cash_penalty", label="Cash drift", unit="portfolio weight", value_format="pct", description="Absolute distance between total position weight and 100%.", calculate=_ctx("cash_penalty")),
)


HARVEST_REPLAY_METADATA_KEYS = (
    "status",
    "reason",
    "start",
    "end",
    "years",
    "frequency",
    "rebalance_frequency",
    "harvest_frequency",
    "tax_rate",
    "harvest_threshold_pct",
    "transaction_cost_bps",
    "replacement_cost_bps",
    "replacement_count",
    "replacement_method",
    "wash_sale_days",
    "random_seed",
    "selected_position_count",
    "missing_position_tickers",
    "skipped_harvests_by_reason",
)
