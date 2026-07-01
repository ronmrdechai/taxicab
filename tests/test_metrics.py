from datetime import date, timedelta
import unittest

from taxicab.data import PricePoint
from taxicab.metrics import beta_to_benchmark, daily_returns, estimated_tax_loss_alpha, simulated_tax_alpha


def series_from_returns(start_price, returns):
    points = [PricePoint(date(2020, 1, 1), start_price)]
    price = start_price
    for idx, item in enumerate(returns, start=1):
        price *= 1.0 + item
        points.append(PricePoint(date(2020, 1, 1) + timedelta(days=idx), price))
    return points


class MetricsTests(unittest.TestCase):
    def test_beta_to_benchmark(self):
        benchmark = series_from_returns(100.0, [0.01, 0.02, -0.01, 0.03, -0.02])
        asset = series_from_returns(100.0, [0.02, 0.04, -0.02, 0.06, -0.04])

        beta = beta_to_benchmark(daily_returns(asset), daily_returns(benchmark))

        self.assertAlmostEqual(beta, 2.0, places=6)

    def test_estimated_tax_loss_alpha_annualizes_rebalance_losses(self):
        points = [
            PricePoint(date(2020, 1, 31), 100.0),
            PricePoint(date(2020, 2, 29), 90.0),
            PricePoint(date(2020, 3, 31), 99.0),
            PricePoint(date(2020, 4, 30), 89.1),
        ]

        alpha = estimated_tax_loss_alpha(points, "monthly")

        self.assertAlmostEqual(alpha, 0.8, places=6)

    def test_frequency_aliases_are_supported(self):
        points = [
            PricePoint(date(2020, 1, 31), 100.0),
            PricePoint(date(2020, 6, 30), 90.0),
            PricePoint(date(2020, 12, 31), 81.0),
        ]

        self.assertAlmostEqual(
            estimated_tax_loss_alpha(points, "half-yearly"),
            estimated_tax_loss_alpha(points, "halfly"),
            places=9,
        )
        self.assertAlmostEqual(
            estimated_tax_loss_alpha(points, "yearly"),
            estimated_tax_loss_alpha(points, "annually"),
            places=9,
        )

    def test_simulated_tax_alpha_applies_tax_rate_and_resets_basis(self):
        points = [
            PricePoint(date(2020, 1, 1), 100.0),
            PricePoint(date(2021, 1, 1), 90.0),
            PricePoint(date(2022, 1, 1), 81.0),
        ]

        alpha = simulated_tax_alpha(
            points,
            "annually",
            tax_rate=0.30,
            harvest_threshold_pct=0.0,
            transaction_cost_bps=0.0,
            replacement_cost_bps=0.0,
        )

        expected = 5.7 / ((100.0 + 90.0 + 81.0) / 3.0) / (731.0 / 365.25)
        self.assertAlmostEqual(alpha, expected, places=6)


if __name__ == "__main__":
    unittest.main()
