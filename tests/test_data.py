import unittest

from datetime import date

from taxicab.data import parse_historical_holdings_csv, parse_holdings_csv


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

    def test_holdings_parser_reads_industry(self):
        text = """Ticker,Weight (%),Sector,Industry
AAPL,6.5,Information Technology,Technology Hardware
MSFT,5.0,Information Technology,Software
"""

        holdings = parse_holdings_csv(text)

        self.assertEqual(holdings[0].industry, "Technology Hardware")
        self.assertEqual(holdings[1].industry, "Software")

    def test_historical_holdings_parser_groups_and_normalizes_snapshots(self):
        text = """date,ticker,weight,sector,industry
2020-01-31,A,60,Tech,Software
2020-01-31,B,40,Tech,Semiconductors
2020-02-29,A,30,Tech,Software
2020-02-29,C,70,Health,Pharma
"""

        snapshots = parse_historical_holdings_csv(text)

        self.assertEqual(set(snapshots), {date(2020, 1, 31), date(2020, 2, 29)})
        self.assertAlmostEqual(sum(holding.weight for holding in snapshots[date(2020, 1, 31)]), 1.0)
        self.assertEqual([holding.ticker for holding in snapshots[date(2020, 2, 29)]], ["A", "C"])
        self.assertEqual(snapshots[date(2020, 2, 29)][1].industry, "Pharma")
