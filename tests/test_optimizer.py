import importlib.util
import unittest

from datetime import date, timedelta
import tempfile
from pathlib import Path
from typing import Dict, List, cast

import numpy as np

from taxicab.data import Holding, PricePoint
from taxicab.optimizer import (
    Candidate,
    benchmark_proxy_returns,
    beam_selection,
    construct_portfolio,
    index_weighted_weights,
    miqp_selection,
    optimize_weights,
    load_tracking_model_artifact,
    prepare_tracking_model,
    project_to_bounded_simplex,
    project_to_simplex,
    project_to_simplex_with_floor,
    random_unbiased_selection,
    replacement_candidates,
    simulate_portfolio_harvests,
    save_tracking_model_artifact,
    tracking_error,
)


JsonObject = Dict[str, object]
HAS_PYSCIPOPT = importlib.util.find_spec("pyscipopt") is not None


def returns(values: List[float]) -> Dict[date, float]:
    return {
        date(2021, 1, 1) + timedelta(days=idx): value
        for idx, value in enumerate(values)
    }


def object_list(value: object) -> List[JsonObject]:
    assert isinstance(value, list)
    return cast(List[JsonObject], value)


def object_map(value: object) -> JsonObject:
    assert isinstance(value, dict)
    return cast(JsonObject, value)


def number(value: object) -> float:
    assert isinstance(value, (int, float))
    return float(value)


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
        benchmark_returns = returns([0.01, -0.02, 0.03])
        candidates = [
            Candidate(
                "A",
                1.0,
                "Tech",
                beta=1.0,
                tax_alpha=0.0,
                returns=benchmark_returns,
            )
        ]

        model = prepare_tracking_model(candidates, benchmark_returns)

        self.assertIsNotNone(model)
        assert model is not None
        self.assertIsInstance(model.covariance_matrix, np.ndarray)
        self.assertAlmostEqual(tracking_error([1.0], model), 0.0, places=9)

    def test_tracking_model_artifact_round_trips_numpy_arrays(self):
        benchmark_returns = returns([0.01, -0.02, 0.03])
        candidates = [
            Candidate("A", 1.0, "Tech", beta=1.0, tax_alpha=0.0, returns=benchmark_returns)
        ]
        model = prepare_tracking_model(candidates, benchmark_returns)
        assert model is not None

        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "tracking-model.npz"
            save_tracking_model_artifact(model, ["A"], artifact)
            tickers, loaded = load_tracking_model_artifact(artifact)

        self.assertEqual(tickers, ["A"])
        np.testing.assert_allclose(loaded.covariance_matrix, model.covariance_matrix)
        np.testing.assert_allclose(loaded.asset_benchmark_covariance, model.asset_benchmark_covariance)
        self.assertEqual(loaded.observations, model.observations)
        self.assertAlmostEqual(tracking_error([1.0], loaded), 0.0, places=9)

    def test_benchmark_proxy_returns_use_index_weights(self):
        left_returns = returns([0.01, -0.02, 0.03])
        right_returns = returns([0.03, -0.01, 0.01])
        candidates = [
            Candidate("LEFT", 0.75, "Tech", beta=1.0, tax_alpha=0.0, returns=left_returns),
            Candidate("RIGHT", 0.25, "Health", beta=1.0, tax_alpha=0.0, returns=right_returns),
        ]

        proxy_returns = benchmark_proxy_returns(candidates)
        model = prepare_tracking_model(candidates, proxy_returns)

        assert model is not None
        for day in left_returns:
            self.assertAlmostEqual(
                proxy_returns[day],
                0.75 * left_returns[day] + 0.25 * right_returns[day],
                places=12,
            )
        self.assertLess(tracking_error([0.75, 0.25], model), 1e-8)

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
        self.assertLessEqual(tracking_error(weights, model), 0.02000001)
        self.assertLess(tracking_error(weights, model), tracking_error([0.5, 0.5], model))
        self.assertGreater(weights[0], weights[1])

    def test_index_weighted_weights_normalize_selected_index_weights(self):
        candidates = [
            Candidate("A", 0.60, "Tech", beta=1.0, tax_alpha=0.0),
            Candidate("B", 0.30, "Tech", beta=1.0, tax_alpha=0.0),
            Candidate("C", 0.10, "Tech", beta=1.0, tax_alpha=0.0),
        ]

        weights = index_weighted_weights(candidates)

        self.assertAlmostEqual(sum(weights), 1.0, places=9)
        self.assertAlmostEqual(weights[0], 0.60, places=9)
        self.assertAlmostEqual(weights[1], 0.30, places=9)
        self.assertAlmostEqual(weights[2], 0.10, places=9)

    def test_random_unbiased_selection_uses_inclusion_probability_weights(self):
        candidates = [
            Candidate("MEGA", 0.60, "Tech", beta=1.0, tax_alpha=0.0),
            Candidate("LARGE", 0.20, "Tech", beta=1.0, tax_alpha=0.0),
            Candidate("MID", 0.10, "Tech", beta=1.0, tax_alpha=0.0),
            Candidate("SMALL1", 0.05, "Tech", beta=1.0, tax_alpha=0.0),
            Candidate("SMALL2", 0.05, "Tech", beta=1.0, tax_alpha=0.0),
        ]

        selection = random_unbiased_selection(candidates, sample_size=3, random_seed=5)
        selected_weights = {
            candidate.ticker: weight
            for candidate, weight in zip(selection.selected, selection.weights)
        }

        self.assertEqual(len(selection.selected), 3)
        self.assertAlmostEqual(sum(selection.weights), 1.0, places=9)
        self.assertAlmostEqual(selected_weights["MEGA"], 0.60, places=9)
        self.assertAlmostEqual(selected_weights["LARGE"], 0.20, places=9)
        self.assertAlmostEqual(
            selected_weights[[ticker for ticker in selected_weights if ticker not in {"MEGA", "LARGE"}][0]],
            0.20,
            places=9,
        )
        self.assertEqual(selection.inclusion_probabilities["MEGA"], 1.0)
        self.assertEqual(selection.inclusion_probabilities["LARGE"], 1.0)
        self.assertEqual(selection.diagnostics["selection_weighting"], "random_unbiased_pps")

    def test_beam_selection_is_deterministic_and_can_match_sector_quotas(self):
        benchmark_returns = returns([0.01, -0.01, 0.02, -0.02, 0.015, -0.015] * 4)
        candidates = [
            Candidate("A", 0.40, "Tech", beta=1.0, tax_alpha=0.03, returns=benchmark_returns),
            Candidate("B", 0.20, "Tech", beta=1.1, tax_alpha=0.04, returns=benchmark_returns),
            Candidate("C", 0.30, "Health", beta=1.0, tax_alpha=0.03, returns=benchmark_returns),
            Candidate("D", 0.10, "Health", beta=0.9, tax_alpha=0.04, returns=benchmark_returns),
        ]

        first = beam_selection(
            candidates,
            sample_size=2,
            error_margin=0.05,
            target_tax_alpha=0.03,
            benchmark_returns=benchmark_returns,
            target_sectors={"Tech": 0.5, "Health": 0.5},
            match_sectors=True,
            beam_width=2,
        )
        second = beam_selection(
            candidates,
            sample_size=2,
            error_margin=0.05,
            target_tax_alpha=0.03,
            benchmark_returns=benchmark_returns,
            target_sectors={"Tech": 0.5, "Health": 0.5},
            match_sectors=True,
            beam_width=2,
        )

        self.assertEqual([candidate.ticker for candidate in first], [candidate.ticker for candidate in second])
        self.assertEqual({candidate.sector for candidate in first}, {"Tech", "Health"})

    @unittest.skipUnless(HAS_PYSCIPOPT, "PySCIPOpt is not installed")
    def test_miqp_selection_uses_binary_cardinality_model(self):
        benchmark_returns = returns([0.01, -0.01, 0.02, -0.02, 0.015, -0.015] * 4)
        candidates = [
            Candidate("A", 0.40, "Tech", beta=1.0, tax_alpha=0.03, returns=benchmark_returns),
            Candidate("B", 0.20, "Tech", beta=1.2, tax_alpha=0.06, returns=benchmark_returns),
            Candidate("C", 0.30, "Health", beta=1.0, tax_alpha=0.03, returns=benchmark_returns),
            Candidate("D", 0.10, "Health", beta=0.8, tax_alpha=0.06, returns=benchmark_returns),
        ]

        selected, diagnostics = miqp_selection(
            candidates,
            sample_size=2,
            error_margin=0.05,
            target_tax_alpha=0.03,
            benchmark_returns=benchmark_returns,
            target_sectors={"Tech": 0.5, "Health": 0.5},
            match_sectors=True,
            miqp_time_limit=5.0,
            miqp_gap=0.10,
        )

        self.assertEqual(len(selected), 2)
        self.assertEqual(diagnostics["selection_solver"], "PySCIPOpt")
        self.assertIn("selection_solver_status", diagnostics)

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

        positions = object_list(portfolio["positions"])
        metrics = object_map(portfolio["metrics"])
        construction = object_map(metrics["construction"])
        self.assertEqual(len(positions), 2)
        sectors = {position["sector"] for position in positions}
        self.assertEqual(sectors, {"Tech", "Health"})
        self.assertEqual(portfolio["version"], 2)
        self.assertLessEqual(number(construction["sector_abs_error"]), 0.15)

    def test_construct_portfolio_supports_random_weighted_baseline(self):
        benchmark_returns = returns([0.01, -0.01, 0.02, -0.02, 0.015, -0.015] * 4)
        holdings = [
            Holding("A", 0.50, "Tech"),
            Holding("B", 0.30, "Tech"),
            Holding("C", 0.15, "Health"),
            Holding("D", 0.05, "Health"),
        ]
        candidates = [
            Candidate(holding.ticker, holding.weight, holding.sector, beta=1.0, tax_alpha=0.03, returns=benchmark_returns)
            for holding in holdings
        ]

        first = construct_portfolio(
            candidates,
            holdings,
            sample_size=2,
            error_margin=0.05,
            target_tax_alpha=0.03,
            rebalance_frequency="quarterly",
            benchmark_returns=benchmark_returns,
            selection_method="random-weighted",
            weight_method="index-normalized",
            random_seed=11,
            allow_constraint_violations=True,
        )
        second = construct_portfolio(
            candidates,
            holdings,
            sample_size=2,
            error_margin=0.05,
            target_tax_alpha=0.03,
            rebalance_frequency="quarterly",
            benchmark_returns=benchmark_returns,
            selection_method="random-weighted",
            weight_method="index-normalized",
            random_seed=11,
            allow_constraint_violations=True,
        )

        first_targets = object_map(first["targets"])
        self.assertEqual(first_targets["selection_method"], "random-weighted")
        self.assertEqual(first_targets["weight_method"], "index-normalized")
        self.assertEqual(
            [position["ticker"] for position in object_list(first["positions"])],
            [position["ticker"] for position in object_list(second["positions"])],
        )
        self.assertAlmostEqual(
            sum(number(position["weight"]) for position in object_list(first["positions"])),
            1.0,
            places=9,
        )

    def test_construct_portfolio_supports_random_unbiased_baseline(self):
        benchmark_returns = returns([0.01, -0.01, 0.02, -0.02, 0.015, -0.015] * 4)
        holdings = [
            Holding("A", 0.60, "Tech"),
            Holding("B", 0.20, "Tech"),
            Holding("C", 0.10, "Health"),
            Holding("D", 0.05, "Health"),
            Holding("E", 0.05, "Health"),
        ]
        candidates = [
            Candidate(holding.ticker, holding.weight, holding.sector, beta=1.0, tax_alpha=0.03, returns=benchmark_returns)
            for holding in holdings
        ]

        portfolio = construct_portfolio(
            candidates,
            holdings,
            sample_size=3,
            error_margin=0.05,
            target_tax_alpha=0.03,
            rebalance_frequency="quarterly",
            benchmark_returns=benchmark_returns,
            selection_method="random-unbiased",
            random_seed=13,
            allow_constraint_violations=True,
        )

        targets = object_map(portfolio["targets"])
        metrics = object_map(portfolio["metrics"])
        selection = object_map(metrics["selection"])
        positions = object_list(portfolio["positions"])
        self.assertEqual(targets["selection_method"], "random-unbiased")
        self.assertEqual(targets["weight_method"], "random-unbiased")
        self.assertEqual(targets["requested_weight_method"], "slsqp")
        self.assertEqual(selection["selection_weighting"], "random_unbiased_pps")
        self.assertAlmostEqual(sum(number(position["weight"]) for position in positions), 1.0, places=9)
        self.assertTrue(all("sample_inclusion_probability" in position for position in positions))

    @unittest.skipUnless(HAS_PYSCIPOPT, "PySCIPOpt is not installed")
    def test_construct_portfolio_supports_miqp_selection_method(self):
        benchmark_returns = returns([0.01, -0.01, 0.02, -0.02, 0.015, -0.015] * 4)
        holdings = [
            Holding("A", 0.40, "Tech"),
            Holding("B", 0.20, "Tech"),
            Holding("C", 0.30, "Health"),
            Holding("D", 0.10, "Health"),
        ]
        candidates = [
            Candidate(holding.ticker, holding.weight, holding.sector, beta=1.0, tax_alpha=0.03, returns=benchmark_returns)
            for holding in holdings
        ]

        portfolio = construct_portfolio(
            candidates,
            holdings,
            sample_size=2,
            error_margin=0.05,
            target_tax_alpha=0.03,
            rebalance_frequency="quarterly",
            benchmark_returns=benchmark_returns,
            selection_method="miqp",
            miqp_time_limit=5.0,
            miqp_gap=0.10,
            allow_constraint_violations=True,
            weight_iterations=20,
        )

        metrics = object_map(portfolio["metrics"])
        selection = object_map(metrics["selection"])
        targets = object_map(portfolio["targets"])
        self.assertEqual(targets["selection_method"], "miqp")
        self.assertEqual(selection["selection_solver"], "PySCIPOpt")
        self.assertEqual(len(object_list(portfolio["positions"])), 2)

    def test_construct_250_stock_portfolio_satisfies_direct_indexing_sanity_constraints(self):
        benchmark_returns = returns([0.01, -0.008, 0.006, -0.004, 0.003] * 12)
        sectors = ["Tech", "Health", "Financials", "Industrials", "Staples"]
        holdings = [
            Holding(f"TICK{idx:03d}", 1.0 / 250.0, sectors[idx % len(sectors)])
            for idx in range(250)
        ]
        candidates = [
            Candidate(
                holding.ticker,
                holding.weight,
                holding.sector,
                beta=1.0,
                tax_alpha=0.02,
                simulated_tax_alpha=0.02,
                gross_harvestable_loss_rate=0.04,
                observations=300,
                returns=benchmark_returns,
            )
            for holding in holdings
        ]

        portfolio = construct_portfolio(
            candidates,
            holdings,
            sample_size=250,
            error_margin=0.05,
            target_tax_alpha=0.02,
            rebalance_frequency="quarterly",
            match_sectors=True,
            benchmark_returns=benchmark_returns,
            selection_iterations=0,
            weight_iterations=20,
        )

        positions = object_list(portfolio["positions"])
        metrics = object_map(portfolio["metrics"])
        construction = object_map(metrics["construction"])
        weights = [number(position["weight"]) for position in positions]
        self.assertAlmostEqual(sum(weights), 1.0, places=9)
        self.assertLessEqual(max(weights), 0.0080001)
        self.assertGreater(number(construction["effective_number_of_names"]), 100.0)
        self.assertLessEqual(number(construction["sector_abs_error"]), 0.02)
        self.assertLessEqual(number(construction["tracking_error"]), 0.02)
        self.assertLessEqual(number(construction["active_share"]), 0.35)
        self.assertNotIn("tracking_error_annualized_pct", construction)
        self.assertNotIn("max_weight_pct", construction)

    def test_construct_250_stock_portfolio_rejects_hard_fidelity_violations(self):
        benchmark_returns = returns([0.01, -0.008, 0.006, -0.004, 0.003] * 12)
        holdings = [
            Holding(f"ACTIVE{idx:03d}", 1.0 / 250.0, "Tech")
            for idx in range(250)
        ]
        candidates = [
            Candidate(
                holding.ticker,
                holding.weight,
                holding.sector,
                beta=1.6,
                tax_alpha=0.05,
                observations=300,
                returns={day: value * 1.6 for day, value in benchmark_returns.items()},
            )
            for holding in holdings
        ]

        with self.assertRaisesRegex(ValueError, "hard benchmark-fidelity constraints"):
            construct_portfolio(
                candidates,
                holdings,
                sample_size=250,
                error_margin=0.05,
                target_tax_alpha=0.03,
                rebalance_frequency="quarterly",
                match_sectors=True,
                benchmark_returns=benchmark_returns,
                selection_iterations=0,
                weight_iterations=20,
            )

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
        self.assertAlmostEqual(number(simulation["total_realized_loss"]), 0.2, places=9)
        self.assertGreater(number(simulation["immediate_tax_savings_rate"]), 0.0)
        self.assertLess(
            number(simulation["full_liquidation_after_tax_alpha"]),
            number(simulation["immediate_tax_savings_rate"]),
        )
        event = object_list(simulation["sample_events"])[0]
        self.assertEqual(event["sold"], "NVDA")
        self.assertEqual(
            {replacement["ticker"] for replacement in object_list(event["replacements"])},
            {"AMD", "INTC"},
        )
        after = object_map(event["diagnostics_after"])
        self.assertLessEqual(number(after["cash_weight"]), 0.0001)
        diagnostics = object_list(simulation["harvest_diagnostics"])[0]
        self.assertEqual(diagnostics["replacement_names"], ["AMD", "INTC"])

    def test_portfolio_harvest_simulation_can_use_random_replacements(self):
        dates = [date(2020, 1, 31), date(2020, 2, 29), date(2020, 3, 31)]
        benchmark_returns = returns([0.0, 0.0, 0.0])
        candidates = [
            Candidate("LOSS", 0.40, "Tech", beta=1.0, tax_alpha=0.05, observations=300, returns=benchmark_returns),
            Candidate("ALT1", 0.30, "Tech", beta=1.0, tax_alpha=0.05, observations=300, returns=benchmark_returns),
            Candidate("ALT2", 0.20, "Tech", beta=1.0, tax_alpha=0.05, observations=300, returns=benchmark_returns),
            Candidate("ALT3", 0.10, "Tech", beta=1.0, tax_alpha=0.05, observations=300, returns=benchmark_returns),
        ]
        prices = {
            "SPY": [PricePoint(day, 100.0) for day in dates],
            "LOSS": [
                PricePoint(dates[0], 100.0),
                PricePoint(dates[1], 80.0),
                PricePoint(dates[2], 90.0),
            ],
            "ALT1": [PricePoint(day, 100.0) for day in dates],
            "ALT2": [PricePoint(day, 100.0) for day in dates],
            "ALT3": [PricePoint(day, 100.0) for day in dates],
        }

        first = simulate_portfolio_harvests(
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
            replacement_count=1,
            replacement_method="random",
            random_seed=23,
        )
        second = simulate_portfolio_harvests(
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
            replacement_count=1,
            replacement_method="random",
            random_seed=23,
        )

        first_event = object_list(first["sample_events"])[0]
        second_event = object_list(second["sample_events"])[0]
        self.assertEqual(first["replacement_method"], "random")
        self.assertEqual(first_event["replacements"], second_event["replacements"])

    def test_portfolio_harvest_simulation_can_harvest_daily_between_quarterly_rebalances(self):
        dates = [
            date(2020, 1, 1),
            date(2020, 1, 2),
            date(2020, 3, 31),
            date(2020, 4, 1),
        ]
        benchmark_returns = returns([0.0, 0.0, 0.0, 0.0])
        candidates = [
            Candidate("LOSS", 0.60, "Tech", beta=1.0, tax_alpha=0.05, observations=300, returns=benchmark_returns),
            Candidate("ALT", 0.40, "Tech", beta=1.0, tax_alpha=0.05, observations=300, returns=benchmark_returns),
        ]
        prices = {
            "SPY": [PricePoint(day, 100.0) for day in dates],
            "LOSS": [
                PricePoint(dates[0], 100.0),
                PricePoint(dates[1], 80.0),
                PricePoint(dates[2], 100.0),
                PricePoint(dates[3], 100.0),
            ],
            "ALT": [PricePoint(day, 100.0) for day in dates],
        }

        daily = simulate_portfolio_harvests(
            [candidates[0]],
            [1.0],
            candidates,
            prices,
            "SPY",
            benchmark_returns,
            "quarterly",
            error_margin=0.05,
            target_tax_alpha=0.03,
            tax_rate=0.30,
            harvest_threshold_pct=0.05,
            transaction_cost_bps=0.0,
            replacement_cost_bps=0.0,
            replacement_count=1,
            harvest_frequency="daily",
        )
        quarterly = simulate_portfolio_harvests(
            [candidates[0]],
            [1.0],
            candidates,
            prices,
            "SPY",
            benchmark_returns,
            "quarterly",
            error_margin=0.05,
            target_tax_alpha=0.03,
            tax_rate=0.30,
            harvest_threshold_pct=0.05,
            transaction_cost_bps=0.0,
            replacement_cost_bps=0.0,
            replacement_count=1,
            harvest_frequency="quarterly",
        )

        self.assertEqual(daily["rebalance_frequency"], "quarterly")
        self.assertEqual(daily["harvest_frequency"], "daily")
        self.assertEqual(daily["harvest_count"], 1)
        self.assertEqual(daily["rebalance_count"], 1)
        self.assertEqual(object_list(daily["sample_events"])[0]["date"], "2020-01-02")
        self.assertEqual(quarterly["harvest_count"], 0)

    def test_portfolio_harvest_replay_keeps_benchmark_like_path_after_replacement(self):
        dates = [
            date(2020, 1, 31),
            date(2020, 2, 29),
            date(2020, 3, 31),
            date(2020, 4, 30),
        ]
        benchmark_prices = [100.0, 80.0, 90.0, 100.0]
        benchmark_returns = {
            dates[1]: benchmark_prices[1] / benchmark_prices[0] - 1.0,
            dates[2]: benchmark_prices[2] / benchmark_prices[1] - 1.0,
            dates[3]: benchmark_prices[3] / benchmark_prices[2] - 1.0,
        }
        candidates = [
            Candidate("LOSS", 0.50, "Tech", beta=1.0, tax_alpha=0.05, observations=300, returns=benchmark_returns),
            Candidate("ALT", 0.50, "Tech", beta=1.0, tax_alpha=0.05, observations=300, returns=benchmark_returns),
        ]
        prices = {
            "SPY": [PricePoint(day, price) for day, price in zip(dates, benchmark_prices)],
            "LOSS": [PricePoint(day, price) for day, price in zip(dates, benchmark_prices)],
            "ALT": [PricePoint(day, price) for day, price in zip(dates, benchmark_prices)],
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
            replacement_count=1,
        )

        self.assertEqual(simulation["harvest_count"], 1)
        self.assertAlmostEqual(number(simulation["portfolio_harvest_beta"]), 1.0, places=12)
        self.assertAlmostEqual(number(simulation["portfolio_harvest_tracking_error"]), 0.0, places=12)
        self.assertLessEqual(number(simulation["cash_drag"]), 0.0001)

    def test_wash_sale_blocked_security_is_not_rebought_within_window(self):
        dates = [
            date(2020, 1, 1),
            date(2020, 1, 2),
            date(2020, 1, 3),
            date(2020, 2, 10),
        ]
        benchmark_returns = returns([0.0, 0.0, 0.0, 0.0])
        candidates = [
            Candidate("LOSS1", 0.34, "Tech", beta=1.0, tax_alpha=0.05, observations=300, returns=benchmark_returns),
            Candidate("LOSS2", 0.33, "Tech", beta=1.0, tax_alpha=0.05, observations=300, returns=benchmark_returns),
            Candidate("ALT", 0.20, "Tech", beta=1.0, tax_alpha=0.05, observations=300, returns=benchmark_returns),
            Candidate("SAFE", 0.13, "Tech", beta=1.0, tax_alpha=0.05, observations=300, returns=benchmark_returns),
        ]
        prices = {
            "SPY": [PricePoint(day, 100.0) for day in dates],
            "LOSS1": [
                PricePoint(dates[0], 100.0),
                PricePoint(dates[1], 80.0),
                PricePoint(dates[2], 80.0),
                PricePoint(dates[3], 100.0),
            ],
            "LOSS2": [
                PricePoint(dates[0], 100.0),
                PricePoint(dates[1], 100.0),
                PricePoint(dates[2], 80.0),
                PricePoint(dates[3], 100.0),
            ],
            "ALT": [PricePoint(day, 100.0) for day in dates],
            "SAFE": [PricePoint(day, 100.0) for day in dates],
        }

        simulation = simulate_portfolio_harvests(
            candidates[:2],
            [0.5, 0.5],
            candidates,
            prices,
            "SPY",
            benchmark_returns,
            "quarterly",
            error_margin=0.05,
            target_tax_alpha=0.03,
            tax_rate=0.30,
            harvest_threshold_pct=0.05,
            transaction_cost_bps=0.0,
            replacement_cost_bps=0.0,
            replacement_count=1,
            wash_sale_days=31,
            harvest_frequency="daily",
        )

        events = object_list(simulation["sample_events"])
        self.assertEqual([event["sold"] for event in events[:2]], ["LOSS1", "LOSS2"])
        second_replacements = object_list(events[1]["replacements"])
        self.assertNotEqual(second_replacements[0]["ticker"], "LOSS1")

    def test_portfolio_harvest_rebalances_available_targets_when_one_target_is_banned(self):
        dates = [
            date(2020, 1, 1),
            date(2020, 3, 30),
            date(2020, 3, 31),
            date(2020, 4, 1),
        ]
        benchmark_returns = returns([0.0, 0.0, 0.0, 0.0])
        candidates = [
            Candidate("LOSS", 0.50, "Tech", beta=1.0, tax_alpha=0.05, observations=300, returns=benchmark_returns),
            Candidate("KEEP", 0.50, "Tech", beta=1.0, tax_alpha=0.05, observations=300, returns=benchmark_returns),
            Candidate("ALT", 0.10, "Tech", beta=1.0, tax_alpha=0.05, observations=300, returns=benchmark_returns),
        ]
        prices = {
            "SPY": [PricePoint(day, 100.0) for day in dates],
            "LOSS": [
                PricePoint(dates[0], 100.0),
                PricePoint(dates[1], 80.0),
                PricePoint(dates[2], 80.0),
                PricePoint(dates[3], 80.0),
            ],
            "KEEP": [PricePoint(day, 100.0) for day in dates],
            "ALT": [PricePoint(day, 100.0) for day in dates],
        }

        simulation = simulate_portfolio_harvests(
            candidates[:2],
            [0.5, 0.5],
            candidates,
            prices,
            "SPY",
            benchmark_returns,
            "quarterly",
            error_margin=0.05,
            target_tax_alpha=0.03,
            tax_rate=0.30,
            harvest_threshold_pct=0.05,
            transaction_cost_bps=0.0,
            replacement_cost_bps=0.0,
            replacement_count=1,
            harvest_frequency="daily",
        )

        self.assertEqual(simulation["harvest_count"], 1)
        self.assertEqual(simulation["rebalance_count"], 1)

    def test_portfolio_harvest_path_metrics_track_realized_path(self):
        dates = [
            date(2020, 1, 31),
            date(2020, 2, 29),
            date(2020, 3, 31),
            date(2020, 4, 30),
        ]
        benchmark_prices = [100.0, 110.0, 99.0, 120.0]
        benchmark_returns = {
            dates[1]: benchmark_prices[1] / benchmark_prices[0] - 1.0,
            dates[2]: benchmark_prices[2] / benchmark_prices[1] - 1.0,
            dates[3]: benchmark_prices[3] / benchmark_prices[2] - 1.0,
        }
        candidate = Candidate(
            "MIRROR",
            1.0,
            "Tech",
            beta=1.0,
            tax_alpha=0.0,
            observations=300,
            returns=benchmark_returns,
        )
        prices = {
            "SPY": [PricePoint(day, price) for day, price in zip(dates, benchmark_prices)],
            "MIRROR": [PricePoint(day, price) for day, price in zip(dates, benchmark_prices)],
        }

        simulation = simulate_portfolio_harvests(
            [candidate],
            [1.0],
            [candidate],
            prices,
            "SPY",
            benchmark_returns,
            "monthly",
            error_margin=0.05,
            target_tax_alpha=0.03,
            tax_rate=0.30,
            harvest_threshold_pct=0.50,
            transaction_cost_bps=0.0,
            replacement_cost_bps=0.0,
            replacement_count=1,
        )

        self.assertEqual(simulation["status"], "ok")
        self.assertEqual(simulation["harvest_count"], 0)
        self.assertEqual(simulation["portfolio_harvest_observations"], 3)
        self.assertAlmostEqual(number(simulation["portfolio_harvest_tracking_error"]), 0.0, places=12)
        self.assertAlmostEqual(number(simulation["portfolio_harvest_beta"]), 1.0, places=12)
        self.assertAlmostEqual(number(simulation["portfolio_harvest_correlation"]), 1.0, places=12)
        self.assertAlmostEqual(number(simulation["portfolio_harvest_active_return"]), 0.0, places=12)
        self.assertAlmostEqual(
            number(simulation["portfolio_harvest_annualized_return"]),
            number(simulation["benchmark_annualized_return"]),
            places=12,
        )

    def test_replacement_candidates_prefer_industry_and_return_similarity(self):
        benchmark_returns = returns([0.01, -0.01, 0.02, -0.02, 0.015, -0.015] * 8)
        selected = [
            Candidate(
                "SRC",
                0.50,
                "Tech",
                beta=1.0,
                tax_alpha=0.03,
                returns=benchmark_returns,
                industry="Software",
            )
        ]
        universe = selected + [
            Candidate(
                "SAME",
                0.20,
                "Tech",
                beta=1.02,
                tax_alpha=0.03,
                returns={day: value * 1.02 for day, value in benchmark_returns.items()},
                industry="Software",
            ),
            Candidate(
                "DIFF",
                0.30,
                "Tech",
                beta=1.0,
                tax_alpha=0.03,
                returns={day: value * -1.0 for day, value in benchmark_returns.items()},
                industry="Hardware",
            ),
        ]

        replacements = replacement_candidates(
            universe,
            selected,
            error_margin=0.05,
            target_tax_alpha=0.03,
            benchmark_returns=benchmark_returns,
        )

        self.assertEqual(replacements["SRC"][0]["ticker"], "SAME")
        self.assertTrue(replacements["SRC"][0]["industry_match"])
        self.assertGreater(number(replacements["SRC"][0]["return_correlation"]), 0.9)

    def test_portfolio_harvest_simulation_uses_point_in_time_active_replacements(self):
        dates = [date(2020, 1, 31), date(2020, 2, 29), date(2020, 3, 31)]
        benchmark_returns = returns([0.0, 0.0, 0.0])
        candidates = [
            Candidate(
                "NVDA",
                0.40,
                "Tech",
                beta=1.0,
                tax_alpha=0.05,
                observations=300,
                returns=benchmark_returns,
                industry="Semiconductors",
            ),
            Candidate(
                "AMD",
                0.35,
                "Tech",
                beta=1.0,
                tax_alpha=0.05,
                observations=300,
                returns=benchmark_returns,
                industry="Semiconductors",
            ),
            Candidate(
                "INTC",
                0.25,
                "Tech",
                beta=1.0,
                tax_alpha=0.05,
                observations=300,
                returns=benchmark_returns,
                industry="Semiconductors",
            ),
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
        historical_holdings = {
            dates[0]: [
                Holding("NVDA", 0.5, "Tech", "Semiconductors"),
                Holding("AMD", 0.5, "Tech", "Semiconductors"),
            ],
            dates[1]: [
                Holding("NVDA", 0.5, "Tech", "Semiconductors"),
                Holding("AMD", 0.5, "Tech", "Semiconductors"),
            ],
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
            historical_holdings=historical_holdings,
        )

        event = object_list(simulation["sample_events"])[0]
        replacements = object_list(event["replacements"])
        self.assertEqual([replacement["ticker"] for replacement in replacements], ["AMD"])
        self.assertTrue(simulation["point_in_time_constituents"])


if __name__ == "__main__":
    unittest.main()
