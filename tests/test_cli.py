import contextlib
import io
from datetime import date, timedelta
import json
import tempfile
import unittest
from pathlib import Path

from taxicab.cli import main
from taxicab.data import Holding, PricePoint, write_cache
from taxicab.optimizer import load_tracking_model_artifact


def price_series(start, returns):
    points = [PricePoint(date(2021, 1, 1), start)]
    price = start
    for idx, item in enumerate(returns, start=1):
        price *= 1.0 + item
        points.append(PricePoint(date(2021, 1, 1) + timedelta(days=idx), price))
    return points


class CliTests(unittest.TestCase):
    def test_construct_command_writes_portfolio_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "cache"
            output = Path(tmp) / "portfolio.json"
            holdings = [
                Holding("AAA", 0.5, "Tech"),
                Holding("BBB", 0.5, "Health"),
            ]
            benchmark_returns = [0.01, -0.01, 0.02, -0.02, 0.01, -0.01]
            prices = {
                "IDX": price_series(100.0, benchmark_returns),
                "AAA": price_series(50.0, benchmark_returns),
                "BBB": price_series(80.0, [item * 1.1 for item in benchmark_returns]),
            }
            write_cache(data_dir, holdings, prices, {"index": "IDX"})

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = main(
                    [
                        "construct",
                        "--data-dir",
                        str(data_dir),
                        "--sample-size",
                        "2",
                        "--error-margin",
                        "0.05",
                        "--target-tax-alpha",
                        "0.03",
                        "--rebalance-frequency",
                        "monthly",
                        "--sector-match",
                        "--min-observations",
                        "5",
                        "--selection-iterations",
                        "0",
                        "--weight-iterations",
                        "20",
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(status, 0)
            state = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(len(state["positions"]), 2)
            self.assertTrue(all("last_price" in position for position in state["positions"]))
            self.assertEqual(state["targets"]["error_margin"], 0.05)
            self.assertEqual(state["targets"]["rebalance_frequency"], "monthly")
            self.assertEqual(state["targets"]["harvest_frequency"], "daily")
            self.assertEqual(state["portfolio_harvest_simulation"]["harvest_frequency"], "daily")
            self.assertIn("beta=", stdout.getvalue())
            self.assertIn("tracking_error_annualized_pct=", stdout.getvalue())
            self.assertIn("error_percentage", state["metrics"])
            self.assertIn("tracking_error_annualized_pct", state["metrics"])
            artifact_info = state["tracking_model_artifact"]
            artifact_path = output.parent / artifact_info["path"]
            self.assertTrue(artifact_path.exists())
            tickers, model = load_tracking_model_artifact(artifact_path)
            self.assertEqual(tickers, artifact_info["ticker_order"])
            self.assertEqual(model.observations, artifact_info["observations"])

    def test_construct_command_accepts_baseline_strategy_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "cache"
            output = Path(tmp) / "portfolio.json"
            benchmark_returns = [0.01, -0.01, 0.02, -0.02, 0.01, -0.01]
            holdings = [
                Holding("AAA", 0.50, "Tech"),
                Holding("BBB", 0.30, "Tech"),
                Holding("CCC", 0.20, "Health"),
            ]
            prices = {
                "IDX": price_series(100.0, benchmark_returns),
                "AAA": price_series(50.0, benchmark_returns),
                "BBB": price_series(80.0, [item * 1.1 for item in benchmark_returns]),
                "CCC": price_series(70.0, [item * 0.9 for item in benchmark_returns]),
            }
            write_cache(data_dir, holdings, prices, {"index": "IDX"})

            with contextlib.redirect_stdout(io.StringIO()):
                status = main(
                    [
                        "construct",
                        "--data-dir",
                        str(data_dir),
                        "--sample-size",
                        "2",
                        "--error-margin",
                        "0.05",
                        "--target-tax-alpha",
                        "0.03",
                        "--rebalance-frequency",
                        "monthly",
                        "--min-observations",
                        "5",
                        "--selection-method",
                        "random-weighted",
                        "--weight-method",
                        "index-normalized",
                        "--replacement-method",
                        "random",
                        "--allow-constraint-violations",
                        "--seed",
                        "19",
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(status, 0)
            state = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(state["targets"]["selection_method"], "random-weighted")
            self.assertEqual(state["targets"]["weight_method"], "index-normalized")
            self.assertEqual(state["targets"]["replacement_method"], "random")
            self.assertTrue(state["targets"]["allow_constraint_violations"])
            self.assertEqual(state["portfolio_harvest_simulation"]["replacement_method"], "random")

    def test_construct_command_accepts_random_unbiased_strategy(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "cache"
            output = Path(tmp) / "portfolio.json"
            benchmark_returns = [0.01, -0.01, 0.02, -0.02, 0.01, -0.01]
            holdings = [
                Holding("AAA", 0.60, "Tech"),
                Holding("BBB", 0.20, "Tech"),
                Holding("CCC", 0.10, "Health"),
                Holding("DDD", 0.05, "Health"),
                Holding("EEE", 0.05, "Health"),
            ]
            prices = {
                "IDX": price_series(100.0, benchmark_returns),
                "AAA": price_series(50.0, benchmark_returns),
                "BBB": price_series(80.0, [item * 1.1 for item in benchmark_returns]),
                "CCC": price_series(70.0, [item * 0.9 for item in benchmark_returns]),
                "DDD": price_series(60.0, [item * 1.05 for item in benchmark_returns]),
                "EEE": price_series(40.0, [item * 0.95 for item in benchmark_returns]),
            }
            write_cache(data_dir, holdings, prices, {"index": "IDX"})

            with contextlib.redirect_stdout(io.StringIO()):
                status = main(
                    [
                        "construct",
                        "--data-dir",
                        str(data_dir),
                        "--sample-size",
                        "3",
                        "--error-margin",
                        "0.05",
                        "--target-tax-alpha",
                        "0.03",
                        "--rebalance-frequency",
                        "monthly",
                        "--min-observations",
                        "5",
                        "--selection-method",
                        "random-unbiased",
                        "--replacement-method",
                        "random",
                        "--allow-constraint-violations",
                        "--seed",
                        "19",
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(status, 0)
            state = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(state["targets"]["selection_method"], "random-unbiased")
            self.assertEqual(state["targets"]["weight_method"], "random-unbiased")
            self.assertEqual(state["targets"]["requested_weight_method"], "slsqp")
            self.assertEqual(state["portfolio_harvest_simulation"]["replacement_method"], "random")

    def test_compare_command_writes_similarity_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "cache"
            left = Path(tmp) / "left.json"
            right = Path(tmp) / "right.json"
            output = Path(tmp) / "comparison.json"
            benchmark_returns = [0.01, -0.01, 0.02, -0.02, 0.01, -0.01]
            holdings = [
                Holding("AAA", 0.5, "Tech"),
                Holding("BBB", 0.5, "Health"),
            ]
            prices = {
                "IDX": price_series(100.0, benchmark_returns),
                "AAA": price_series(50.0, benchmark_returns),
                "BBB": price_series(80.0, [item * 1.1 for item in benchmark_returns]),
            }
            write_cache(data_dir, holdings, prices, {"index": "IDX"})
            left.write_text(
                json.dumps({"positions": [{"ticker": "AAA", "weight": 1.0, "sector": "Tech"}]}),
                encoding="utf-8",
            )
            right.write_text(
                json.dumps({"positions": [{"ticker": "BBB", "weight": 1.0, "sector": "Health"}]}),
                encoding="utf-8",
            )

            with contextlib.redirect_stdout(io.StringIO()):
                status = main(
                    [
                        "compare",
                        "--data-dir",
                        str(data_dir),
                        "--portfolio",
                        f"left={left}",
                        "--portfolio",
                        f"right={right}",
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(status, 0)
            comparison = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(comparison["pairwise"][0]["ticker_overlap_count"], 0)
            self.assertAlmostEqual(comparison["pairwise"][0]["active_share"], 1.0)
            self.assertAlmostEqual(
                comparison["portfolios"]["left"]["sector_active_share_to_index"],
                0.5,
            )

    def test_compare_command_writes_html_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "cache"
            left = Path(tmp) / "left.json"
            right = Path(tmp) / "right.json"
            html_output = Path(tmp) / "comparison.html"
            benchmark_returns = [0.01, -0.01, 0.02, -0.02, 0.01, -0.01]
            holdings = [
                Holding("AAA", 0.5, "Tech"),
                Holding("BBB", 0.5, "Health"),
            ]
            prices = {
                "IDX": price_series(100.0, benchmark_returns),
                "AAA": price_series(50.0, benchmark_returns),
                "BBB": price_series(80.0, [item * 1.1 for item in benchmark_returns]),
            }
            write_cache(data_dir, holdings, prices, {"index": "IDX"})
            left.write_text(
                json.dumps(
                    {
                        "metrics": {"simulated_tax_alpha": 0.0123},
                        "positions": [{"ticker": "AAA", "weight": 1.0, "sector": "Tech"}],
                    }
                ),
                encoding="utf-8",
            )
            right.write_text(
                json.dumps({"positions": [{"ticker": "BBB", "weight": 1.0, "sector": "Health"}]}),
                encoding="utf-8",
            )

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = main(
                    [
                        "compare",
                        "--data-dir",
                        str(data_dir),
                        "--portfolio",
                        f"left={left}",
                        "--portfolio",
                        f"right={right}",
                        "--html-output",
                        str(html_output),
                    ]
                )

            self.assertEqual(status, 0)
            report = html_output.read_text(encoding="utf-8")
            self.assertIn("<title>Taxicab Comparison Report</title>", report)
            self.assertIn("Portfolio Metrics", report)
            self.assertIn("Estimated tax alpha", report)
            self.assertIn("1.23%", report)
            self.assertIn("Sector Weights", report)
            self.assertIn("Pairwise Similarity Heatmap", report)
            self.assertIn("Feature-Space PCA Embedding", report)
            self.assertIn("Efficient-Frontier Style View", report)
            self.assertIn("PC1 score (standardized units)", report)
            self.assertIn("Tracking error (annualized %)", report)
            self.assertIn("Objective Decomposition", report)
            self.assertIn("Skipped harvest count", report)
            self.assertIn("Pairwise Comparisons", report)
            self.assertIn('id="pairwise-metric"', report)
            self.assertIn('id="pairwise-metric-head"', report)
            self.assertIn('id="pairwise-table-data"', report)
            self.assertIn('"self": true', report)
            self.assertIn("not tax advice", report)
            self.assertIn("background-color: rgba(35, 134, 54", report)
            self.assertIn("background-color: rgba(218, 54, 51", report)
            self.assertIn("Wrote HTML comparison report", stdout.getvalue())

    def test_compare_command_can_replay_harvest_performance(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "cache"
            left = Path(tmp) / "left.json"
            right = Path(tmp) / "right.json"
            output = Path(tmp) / "comparison.json"
            prices = {
                "IDX": price_series(100.0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                "AAA": price_series(100.0, [-0.10, 0.00, 0.02, 0.00, 0.00, 0.00]),
                "BBB": price_series(100.0, [0.00, 0.00, 0.00, 0.00, 0.00, 0.00]),
                "CCC": price_series(100.0, [0.00, 0.00, 0.00, 0.00, 0.00, 0.00]),
            }
            holdings = [
                Holding("AAA", 1.0 / 3.0, "Tech"),
                Holding("BBB", 1.0 / 3.0, "Tech"),
                Holding("CCC", 1.0 / 3.0, "Tech"),
            ]
            write_cache(data_dir, holdings, prices, {"index": "IDX"})
            targets = {
                "error_margin": 0.05,
                "estimated_tax_loss_alpha": 0.01,
                "tax_alpha_mode": "at-least",
                "tax_metric": "simulated",
                "rebalance_frequency": "monthly",
                "harvest_frequency": "daily",
                "tax_assumptions": {
                    "tax_rate": 0.30,
                    "harvest_threshold_pct": 0.05,
                    "transaction_cost_bps": 0.0,
                    "replacement_cost_bps": 0.0,
                    "replacement_count": 1,
                    "wash_sale_days": 31,
                },
            }
            left.write_text(
                json.dumps(
                    {
                        "targets": targets,
                        "positions": [{"ticker": "AAA", "weight": 1.0, "sector": "Tech"}],
                    }
                ),
                encoding="utf-8",
            )
            right.write_text(
                json.dumps(
                    {
                        "targets": targets,
                        "positions": [{"ticker": "BBB", "weight": 1.0, "sector": "Tech"}],
                    }
                ),
                encoding="utf-8",
            )

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = main(
                    [
                        "compare",
                        "--data-dir",
                        str(data_dir),
                        "--portfolio",
                        f"left={left}",
                        "--portfolio",
                        f"right={right}",
                        "--replay-harvests",
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(status, 0)
            comparison = json.loads(output.read_text(encoding="utf-8"))
            left_replay = comparison["portfolios"]["left"]["harvest_replay"]
            right_replay = comparison["portfolios"]["right"]["harvest_replay"]
            replay_deltas = comparison["pairwise"][0]["harvest_replay_deltas"]["left_minus_right"]
            self.assertTrue(comparison["harvest_replay"]["enabled"])
            self.assertEqual(left_replay["status"], "ok")
            self.assertEqual(left_replay["selected_position_count"], 1)
            self.assertEqual(left_replay["missing_position_tickers"], [])
            self.assertGreater(left_replay["harvest_count"], right_replay["harvest_count"])
            self.assertGreater(replay_deltas["harvest_count"], 0)
            self.assertIn("left harvest replay:", stdout.getvalue())
            self.assertIn("left vs right harvest replay delta:", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
