"""
Unit tests for the Congressional Trades Analyst node.

No database or LLM required — the congress_trades store fetch and the LLM are
mocked.

Run from the TradingAgents repo root:
    python -m pytest tests/test_congressional_trades_analyst.py -v
"""

import sys
from datetime import date
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from tradingagents.agents.analysts.congressional_trades_analyst import (
    CONGRESS_STUB_MARKER,
    create_congressional_trades_analyst,
)
from tradingagents.dataflows.congress_trades_store import CongressTradesError

# Patch the fetch where it is defined so the local import inside the node picks
# up the mock.
_FETCH_PATH = "tradingagents.dataflows.congress_trades_store.fetch_congressional_trades"


def _llm(response: str = "### Summary\nSynthesized.") -> MagicMock:
    llm = MagicMock()
    resp = MagicMock()
    resp.content = response
    llm.invoke.return_value = resp
    return llm


def _state(ticker: str = "AAPL", trade_date: str = "2026-05-05") -> dict:
    return {"company_of_interest": ticker, "trade_date": trade_date}


def _row(**over) -> dict:
    base = {
        "representative": "Jane Smith",
        "party": "D",
        "chamber": "Senate",
        "committee": "Banking",
        "trader": "Self",
        "transaction_type": "Buy",
        "amount_range": "$15K-$50K",
        "trade_date": date(2026, 3, 1),
        "disclosure_date": date(2026, 4, 10),
        "lag_days": 40,
    }
    base.update(over)
    return base


# ─── Store returns a seeded row ───────────────────────────────────────────────


@pytest.mark.unit
class TestSeededRow:
    def test_report_renders_the_trade(self):
        with patch(_FETCH_PATH, return_value=[_row()]):
            result = create_congressional_trades_analyst(_llm())(_state())
        report = result["congressional_trades_report"]
        assert report != CONGRESS_STUB_MARKER
        assert "Jane Smith" in report          # row rendered in the table
        assert "Notable Trades" in report
        assert result["messages"][0].content == report

    def test_llm_synthesis_invoked_once(self):
        llm = _llm()
        with patch(_FETCH_PATH, return_value=[_row()]):
            create_congressional_trades_analyst(llm)(_state())
        assert llm.invoke.call_count == 1


# ─── Empty store: fail open ───────────────────────────────────────────────────


@pytest.mark.unit
class TestEmptyStore:
    def test_no_data_report_not_stub(self):
        with patch(_FETCH_PATH, return_value=[]):
            result = create_congressional_trades_analyst(_llm())(_state())
        report = result["congressional_trades_report"]
        assert "No congressional trade disclosures" in report
        assert result["messages"][0].content == report

    def test_llm_not_called_when_empty(self):
        llm = _llm()
        with patch(_FETCH_PATH, return_value=[]):
            create_congressional_trades_analyst(llm)(_state())
        assert llm.invoke.call_count == 0


# ─── Store unreachable / import error: stub, never raise ──────────────────────


@pytest.mark.unit
class TestFailOpen:
    def test_store_error_produces_stub(self):
        with patch(_FETCH_PATH, side_effect=CongressTradesError("down")):
            result = create_congressional_trades_analyst(_llm())(_state())
        assert result["congressional_trades_report"] == CONGRESS_STUB_MARKER
        assert isinstance(result["messages"][0], AIMessage)

    def test_import_error_produces_stub(self):
        with patch.dict(sys.modules, {"tradingagents.dataflows.congress_trades_store": None}):
            result = create_congressional_trades_analyst(_llm())(_state())
        assert result["congressional_trades_report"] == CONGRESS_STUB_MARKER
