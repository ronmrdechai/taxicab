import unittest

from taxicab.rebalance import CurrentPosition, plan_rebalance


class RebalanceTests(unittest.TestCase):
    def test_plan_rebalance_harvests_losses_and_uses_replacement(self):
        state = {
            "positions": [
                {"ticker": "A", "weight": 0.5},
                {"ticker": "B", "weight": 0.5},
            ],
            "replacement_candidates": {
                "A": [{"ticker": "C", "sector": "Tech"}],
            },
        }
        current = [
            CurrentPosition("A", market_value=5000.0, shares=100.0, price=50.0, cost_basis=6000.0),
            CurrentPosition("B", market_value=4500.0, shares=45.0, price=100.0, cost_basis=4000.0),
            CurrentPosition("D", market_value=500.0, shares=10.0, price=50.0, cost_basis=450.0),
        ]

        operations = plan_rebalance(
            state,
            current,
            drift_threshold_pct=0.001,
            harvest_loss_threshold_pct=0.03,
        )
        simplified = {(op.action, op.ticker, op.reason, op.replacement_for) for op in operations}

        self.assertIn(("SELL", "A", "tax-loss harvest", ""), simplified)
        self.assertIn(("BUY", "C", "same-sector replacement for harvested position", "A"), simplified)
        self.assertIn(("BUY", "B", "rebalance to target weight", ""), simplified)
        self.assertIn(("SELL", "D", "not in target direct index", ""), simplified)


if __name__ == "__main__":
    unittest.main()
