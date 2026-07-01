import unittest

from datetime import date, timedelta

import numpy as np

from taxicab.data import Holding, PricePoint
from taxicab.optimizer import (
    Candidate,
    construct_portfolio,
    optimize_weights,
    prepare_tracking_model,
    project_to_bounded_simplex,
    project_to_simplex,
    project_to_simplex_with_floor,
    simulate_portfolio_harvests,
    tracking_error,
)


def returns(values):
    return {
        date(2021, 1, 1) + timedelta(days=idx): value
        for idx, value in enumerate(values)
    }


class OptimizerTests(unittest.TestCase):
    def test_project_to_simplex_is_long_only_and_sums_to_one(self):
        projected = project_to_simplex([0.8, -0.2, 0.6])

        self.assertAlmostEqual(sum(projected), 1.0, places=9)
        self.assertTrue(all(value >= 0.0 for value in projected))

    def test_project_to_simplex_with_floor_keeps_every_position_nonzero(self):
        projected = project_to_simplex_with_floor([0.8, -0.2, 0.6], min_weight=0.05)

        self.assertAlmostEqual(sum(projected), 1.0, places=9)
        self.assertTrue(all(value >= 0.05 for value in projected))

    def test_project_to_bounded_simplex_respects_cap(self):
        projected = project_to_bounded_simplex([0.9, 0.1, 0.0], min_weight=0.1, max_weight=0.6)

        self.assertAlmostEqual(sum(projected), 1.0, places=9)
        self.assertTrue(all(value >= 0.1 for value in projected))
        self.assertTrue(all(value <= 0.6 for value in projected))

    def test_tracking_error_is_zero_for_benchmark_like_returns(self):
        candidates = [
            Candidate(
                "A",
                1.0,
                "Tech",
                beta=1.0,
                tax_alpha=0.0,
                returns={1: 0.01, 2: -0.02, 3: 0.03},  # type: ignore[dict-item]
            )
        ]
        benchmark_returns = {1: 0.01, 2: -0.02, 3: 0.03}  # type: ignore[var-annotated]

        model = prepare_tracking_model(candidates, benchmark_returns)

        self.assertIsNotNone(model)
        assert model is not None
        self.assertIsInstance(model.covariance_matrix, np.ndarray)
        self.assertAlmostEqual(tracking_error([1.0], model), 0.0, places=9)

    def test_optimize_weights_moves_toward_error_margin_and_tax_targets(self):
        benchmark_returns = returns([0.01, -0.01, 0.02, -0.02, 0.015, -0.015] * 8)
        candidates = [
            Candidate("INDEX", 0.5, "Tech", beta=1.0, tax_alpha=0.01, returns=benchmark_returns),
            Candidate(
                "ACTIVE",
                0.5,
                "Tech",
                beta=2.0,
                tax_alpha=0.05,
                returns={day: value * 2.0 for day, value in benchmark_returns.items()},
            ),
        ]

        weights = optimize_weights(
            candidates,
            error_margin=0.02,
            target_tax_alpha=0.03,
            benchmark_returns=benchmark_returns,
            tracking_error_penalty=5.0,
            iterations=400,
        )

        model = prepare_tracking_model(candidates, benchmark_returns)
        assert model is not None
        self.assertLess(tracking_error(weights, model), tracking_error([0.5, 0.5], model))
        self.assertGreater(weights[0], weights[1])

    def test_construct_portfolio_can_match_sector_mix(self):
        benchmark_returns = returns([0.01, -0.01, 0.02, -0.02, 0.015, -0.015] * 8)
        holdings = [
            Holding("A", 0.25, "Tech"),
            Holding("B", 0.25, "Tech"),
            Holding("C", 0.25, "Health"),
            Holding("D", 0.25, "Health"),
        ]
        candidates = [
            Candidate("A", 0.25, "Tech", beta=1.0, tax_alpha=0.03, observations=300, returns=benchmark_returns),
            Candidate(
                "B",
                0.25,
                "Tech",
                beta=0.7,
                tax_alpha=0.08,
                observations=300,
                returns={day: value * 0.7 for day, value in benchmark_returns.items()},
            ),
            Candidate(
                "C",
                0.25,
                "Health",
                beta=1.1,
                tax_alpha=0.025,
                observations=300,
                returns={day: value * 1.1 for day, value in benchmark_returns.items()},
            ),
            Candidate(
                "D",
                0.25,
                "Health",
                beta=0.8,
                tax_alpha=0.05,
                observations=300,
                returns={day: value * 0.8 for day, value in benchmark_returns.items()},
            ),
        ]

        portfolio = construct_portfolio(
            candidates,
            holdings,
            sample_size=2,
            error_margin=0.05,
            target_tax_alpha=0.03,
            rebalance_frequency="quarterly",
            match_sectors=True,
            benchmark_returns=benchmark_returns,
            selection_iterations=0,
            weight_iterations=200,
        )

        self.assertEqual(len(portfolio["positions"]), 2)
        sectors = {position["sector"] for position in portfolio["positions"]}
        self.assertEqual(sectors, {"Tech", "Health"})
        self.assertLessEqual(portfolio["metrics"]["sector_abs_error"], 0.15)

    def test_portfolio_harvest_simulation_realizes_losses_into_same_sector_replacements(self):
        dates = [date(2020, 1, 31), date(2020, 2, 29), date(2020, 3, 31)]
        benchmark_returns = returns([0.0, 0.0, 0.0])
        candidates = [
            Candidate("NVDA", 0.40, "Tech", beta=1.0, tax_alpha=0.05, observations=300, returns=benchmark_returns),
            Candidate("AMD", 0.35, "Tech", beta=1.0, tax_alpha=0.05, observations=300, returns=benchmark_returns),
            Candidate("INTC", 0.25, "Tech", beta=1.0, tax_alpha=0.05, observations=300, returns=benchmark_returns),
        ]
        prices = {
            "SPY": [PricePoint(day, 100.0) for day in dates],
            "NVDA": [
                PricePoint(dates[0], 100.0),
                PricePoint(dates[1], 80.0),
                PricePoint(dates[2], 90.0),
            ],
            "AMD": [PricePoint(day, 50.0) for day in dates],
            "INTC": [PricePoint(day, 25.0) for day in dates],
        }

        simulation = simulate_portfolio_harvests(
            [candidates[0]],
            [1.0],
            candidates,
            prices,
            "SPY",
            benchmark_returns,
            "monthly",
            error_margin=0.05,
            target_tax_alpha=0.03,
            tax_rate=0.30,
            harvest_threshold_pct=0.05,
            transaction_cost_bps=0.0,
            replacement_cost_bps=0.0,
            replacement_count=2,
        )

        self.assertEqual(simulation["status"], "ok")
        self.assertEqual(simulation["harvest_count"], 1)
        self.assertAlmostEqual(simulation["total_realized_loss"], 0.2, places=9)
        self.assertGreater(simulation["portfolio_simulated_tax_alpha"], 0.0)
        event = simulation["sample_events"][0]
        self.assertEqual(event["sold"], "NVDA")
        self.assertEqual(
            {replacement["ticker"] for replacement in event["replacements"]},
            {"AMD", "INTC"},
        )


if __name__ == "__main__":
    unittest.main()
