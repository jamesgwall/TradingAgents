"""Tests for ETF fundamentals handling across yfinance and Alpha Vantage."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from tradingagents.dataflows import alpha_vantage_fundamentals, y_finance


@pytest.mark.unit
class TestYFinanceEtfFundamentals:
    def test_get_fundamentals_etf_returns_overview_without_error(self):
        fake_ticker = MagicMock()
        fake_ticker.info = {
            "longName": "SPDR S&P 500 ETF Trust",
            "category": "Large Blend",
            "fundFamily": "SPDR State Street Global Advisors",
            "totalAssets": 500000000000,
            "navPrice": 520.5,
            "fiftyTwoWeekHigh": 550.0,
            "fiftyTwoWeekLow": 400.0,
        }

        with patch.object(y_finance.yf, "Ticker", return_value=fake_ticker):
            out = y_finance.get_fundamentals("SPY")

        assert "# ETF Fundamentals for SPY" in out
        assert "SPDR S&P 500 ETF Trust" in out
        assert "Large Blend" in out
        assert "Corporate financial statements" in out

    def test_financial_statements_return_not_applicable_for_etfs(self):
        bs = y_finance.get_balance_sheet("SPY")
        cf = y_finance.get_cashflow("QQQ")
        inc = y_finance.get_income_statement("IWM")

        assert "do not apply to Exchange Traded Funds" in bs
        assert "do not apply to Exchange Traded Funds" in cf
        assert "do not apply to Exchange Traded Funds" in inc


@pytest.mark.unit
class TestAlphaVantageEtfFundamentals:
    def test_alpha_vantage_etf_skips_corporate_statements(self):
        f = alpha_vantage_fundamentals.get_fundamentals("SPY")
        f_data = json.loads(f)
        assert f_data["AssetType"] == "ETF"
        assert "SPY" in f_data["Symbol"]

        bs = alpha_vantage_fundamentals.get_balance_sheet("QQQ")
        bs_data = json.loads(bs)
        assert bs_data["annualReports"] == []
        assert "not apply" in bs_data["note"]
