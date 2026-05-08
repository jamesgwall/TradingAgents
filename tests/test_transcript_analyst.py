"""
Unit tests for the Transcript Analyst node.

No database or Ollama instance required — TranscriptStore and LLM are mocked.

Run from the TradingAgents repo root:
    python -m pytest tests/test_transcript_analyst.py -v
"""

import datetime
import sys
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from tradingagents.agents.analysts.transcript_analyst import (
    TRANSCRIPT_STUB_MARKER,
    create_transcript_analyst,
)

# Path to patch — we patch TranscriptStore where it is defined so the `from
# ... import TranscriptStore` inside the analyst node picks up the mock.
_STORE_PATH = "tradingagents.dataflows.transcript_store.TranscriptStore"


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _chunks(n: int = 3) -> list[dict]:
    pub = datetime.datetime(2026, 5, 1, tzinfo=datetime.timezone.utc)
    return [
        {
            "video_id": f"vid{i}",
            "channel_name": "Test Channel",
            "video_title": "Test Video",
            "published_at": pub,
            "content_type": "earnings_call",
            "chunk_index": i,
            "chunk_text": f"Chunk {i}: earnings and revenue guidance for the quarter.",
            "ticker_tags": ["AAPL"],
            "similarity": round(0.9 - i * 0.05, 2),
        }
        for i in range(n)
    ]


def _llm(response: str = "Synthesized research note.") -> MagicMock:
    llm = MagicMock()
    resp = MagicMock()
    resp.content = response
    llm.invoke.return_value = resp
    return llm


def _state(ticker: str = "AAPL", trade_date: str = "2026-05-05") -> dict:
    return {"company_of_interest": ticker, "trade_date": trade_date}


def _wired_store(MockStore, ticker_results, macro_results, merged=None):
    inst = MockStore.return_value
    inst.ticker_query.return_value = ticker_results
    inst.macro_query.return_value = macro_results
    inst.merge_and_deduplicate.return_value = (
        merged if merged is not None else ticker_results + macro_results
    )
    return inst


# ─── Store returns chunks ─────────────────────────────────────────────────────


@pytest.mark.unit
class TestTranscriptAnalystWithChunks:
    def test_report_is_not_stub(self):
        chunks = _chunks(3)
        with patch(_STORE_PATH) as MS:
            _wired_store(MS, chunks, [], chunks)
            result = create_transcript_analyst(_llm())(_state())
        assert result["transcript_report"] != TRANSCRIPT_STUB_MARKER
        assert result["transcript_report"].strip()

    def test_ticker_query_and_macro_query_each_called_once(self):
        chunks = _chunks(2)
        with patch(_STORE_PATH) as MS:
            inst = _wired_store(MS, chunks, [], chunks)
            create_transcript_analyst(_llm())(_state())
        assert inst.ticker_query.call_count == 1
        assert inst.macro_query.call_count == 1

    def test_ticker_passed_to_ticker_query(self):
        chunks = _chunks(1)
        with patch(_STORE_PATH) as MS:
            inst = _wired_store(MS, chunks, [], chunks)
            create_transcript_analyst(_llm())(_state("TSLA"))
        assert inst.ticker_query.call_args[0][0] == "TSLA"

    def test_llm_called_twice_macro_then_synthesis(self):
        chunks = _chunks(2)
        llm = _llm()
        with patch(_STORE_PATH) as MS:
            _wired_store(MS, chunks, [], chunks)
            create_transcript_analyst(llm)(_state())
        assert llm.invoke.call_count == 2

    def test_messages_contains_ai_message_with_report(self):
        chunks = _chunks(1)
        with patch(_STORE_PATH) as MS:
            _wired_store(MS, chunks, [], chunks)
            result = create_transcript_analyst(_llm("Report text."))(_state())
        assert isinstance(result["messages"][0], AIMessage)
        assert result["messages"][0].content == "Report text."

    def test_store_closed_after_success(self):
        chunks = _chunks(1)
        with patch(_STORE_PATH) as MS:
            inst = _wired_store(MS, chunks, [], chunks)
            create_transcript_analyst(_llm())(_state())
        inst.close.assert_called_once()

    def test_merge_called_with_ticker_and_macro_results(self):
        ticker_chunks = _chunks(2)
        macro_chunks = _chunks(1)
        with patch(_STORE_PATH) as MS:
            inst = _wired_store(MS, ticker_chunks, macro_chunks, ticker_chunks + macro_chunks)
            create_transcript_analyst(_llm())(_state())
        inst.merge_and_deduplicate.assert_called_once_with(ticker_chunks, macro_chunks)


# ─── Store returns no chunks ──────────────────────────────────────────────────


@pytest.mark.unit
class TestTranscriptAnalystNoData:
    def test_stub_marker_in_report(self):
        with patch(_STORE_PATH) as MS:
            _wired_store(MS, [], [], [])
            result = create_transcript_analyst(_llm())(_state())
        assert result["transcript_report"] == TRANSCRIPT_STUB_MARKER

    def test_stub_marker_in_messages(self):
        with patch(_STORE_PATH) as MS:
            _wired_store(MS, [], [], [])
            result = create_transcript_analyst(_llm())(_state())
        assert result["messages"][0].content == TRANSCRIPT_STUB_MARKER

    def test_store_closed_even_when_no_chunks(self):
        with patch(_STORE_PATH) as MS:
            inst = _wired_store(MS, [], [], [])
            create_transcript_analyst(_llm())(_state())
        inst.close.assert_called_once()

    def test_synthesis_llm_not_called_when_no_chunks(self):
        llm = _llm()
        with patch(_STORE_PATH) as MS:
            _wired_store(MS, [], [], [])
            create_transcript_analyst(llm)(_state())
        # Only the macro-query-generation call should fire; synthesis is skipped.
        # (macro query gen = 1 call; synthesis = 0 because chunks list is empty)
        assert llm.invoke.call_count == 1


# ─── psycopg2 / pgvector unavailable ─────────────────────────────────────────


@pytest.mark.unit
class TestTranscriptAnalystImportError:
    def test_import_error_produces_stub(self):
        # Simulate psycopg2/pgvector not installed by hiding the store module.
        with patch.dict(sys.modules, {"tradingagents.dataflows.transcript_store": None}):
            result = create_transcript_analyst(_llm())(_state())
        assert result["transcript_report"] == TRANSCRIPT_STUB_MARKER

    def test_import_error_does_not_raise(self):
        with patch.dict(sys.modules, {"tradingagents.dataflows.transcript_store": None}):
            node = create_transcript_analyst(_llm())
            result = node(_state())
        assert "transcript_report" in result
        assert "messages" in result
