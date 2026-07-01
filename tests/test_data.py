import unittest

from taxicab.data import parse_holdings_csv


class DataTests(unittest.TestCase):
    def test_holdings_parser_skips_issuer_preamble(self):
        text = """Fund Holdings as of 2026-06-30
Not investment advice
Ticker,Weight (%),Sector
AAPL,6.5,Information Technology
MSFT,5.0,Information Technology
"""

        holdings = parse_holdings_csv(text)

        self.assertEqual([holding.ticker for holding in holdings], ["AAPL", "MSFT"])
        self.assertAlmostEqual(sum(holding.weight for holding in holdings), 1.0)
        self.assertEqual(holdings[0].sector, "Information Technology")
