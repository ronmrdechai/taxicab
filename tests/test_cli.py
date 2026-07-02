import contextlib
import io
from datetime import date, timedelta
import json
import tempfile
import unittest
from pathlib import Path

from taxicab.cli import main
from taxicab.data import Holding, PricePoint, write_cache


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

    def test_sector_study_command_writes_both_runs_and_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "cache"
            prefix = Path(tmp) / "sector_study"
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

            with contextlib.redirect_stdout(io.StringIO()):
                status = main(
                    [
                        "sector-study",
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
                        "--selection-iterations",
                        "0",
                        "--weight-iterations",
                        "5",
                        "--output-prefix",
                        str(prefix),
                    ]
                )

            self.assertEqual(status, 0)
            self.assertTrue(Path(f"{prefix}_no_sector.json").exists())
            self.assertTrue(Path(f"{prefix}_sector.json").exists())
            comparison = json.loads(Path(f"{prefix}_comparison.json").read_text(encoding="utf-8"))
            self.assertEqual(len(comparison["pairwise"]), 1)


if __name__ == "__main__":
    unittest.main()
