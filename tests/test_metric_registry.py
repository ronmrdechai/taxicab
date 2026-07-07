import inspect
import unittest

from taxicab.metric_registry import (
    MetricSpec,
    MetricSetMeta,
    OBJECTIVE_COMPONENTS,
    all_metric_specs,
    pairwise_heatmap_metrics,
    pairwise_table_metrics,
    portfolio_report_metrics,
)


class MetricRegistryTests(unittest.TestCase):
    def test_metric_specs_are_stamped_by_metric_set_metaclass(self):
        self.assertGreater(len(MetricSetMeta.sets), 0)
        self.assertGreater(len(MetricSetMeta.by_scope), 0)
        for spec in all_metric_specs():
            self.assertTrue(spec.scope)
            self.assertTrue(spec.group)
            self.assertTrue(spec.namespace)

    def test_metric_specs_do_not_declare_placement_metadata(self):
        parameters = inspect.signature(MetricSpec).parameters
        self.assertNotIn("group", parameters)
        self.assertNotIn("scope", parameters)
        self.assertNotIn("namespace", parameters)
        for metric_set in MetricSetMeta.sets:
            for spec in metric_set.metrics:
                self.assertEqual(spec.group, metric_set.group)
                self.assertEqual(spec.scope, metric_set.scope)
                self.assertEqual(spec.namespace, metric_set.namespace)

    def test_metric_specs_have_single_required_description(self):
        for spec in all_metric_specs():
            self.assertTrue(spec.description.strip())
            self.assertFalse(hasattr(spec, "tooltip"))
        for component in OBJECTIVE_COMPONENTS:
            self.assertTrue(component.description.strip())
            self.assertFalse(hasattr(component, "tooltip"))

    def test_metric_keys_are_unique_within_scope(self):
        seen = set()
        for spec in all_metric_specs():
            identity = (spec.scope, spec.key)
            self.assertNotIn(identity, seen)
            seen.add(identity)

    def test_metric_formats_and_usages_are_known(self):
        valid_formats = {"number", "pct", "integer", "multiple", "list", "mapping"}
        valid_usages = {
            "construct_cli_primary",
            "construct_cli_fit",
            "construct_cli_sector",
            "construct_cli_harvest",
            "construct_cli_path",
            "comparison_cli_portfolio",
            "comparison_cli_harvest",
            "comparison_cli_delta",
            "comparison_cli_pairwise",
            "portfolio_table",
            "pairwise_table",
            "pairwise_heatmap",
            "frontier_x",
            "frontier_y",
            "frontier_color",
            "frontier_size",
        }
        for spec in all_metric_specs():
            self.assertIn(spec.value_format, valid_formats)
            for usage in spec.usages:
                self.assertIn(usage, valid_usages)

    def test_report_metric_views_are_resolvable(self):
        self.assertTrue(portfolio_report_metrics(include_harvest_replay=False))
        self.assertTrue(portfolio_report_metrics(include_harvest_replay=True))
        self.assertTrue(pairwise_table_metrics())
        self.assertTrue(pairwise_heatmap_metrics())


if __name__ == "__main__":
    unittest.main()
