"""
Tests for the House e-filed PTR source.

PDF text extraction is asserted against captured fixture PDFs (one e-filed, one
scanned). The structurer is stubbed so prompt assembly and output normalization
are exercised without a live LLM. The bulk-index parser is asserted against an
in-memory ZIP, and the ingest pipeline is driven with fake client/store doubles.
A live smoke test hits the real House Clerk index and auto-skips when offline.

Run from the TradingAgents repo root:
    python -m pytest tests/test_congress_house_efd.py -v
"""

from __future__ import annotations

import io
import zipfile
from datetime import date
from pathlib import Path

import pytest

from tradingagents.dataflows import congress_house_efd as house
from tradingagents.dataflows.congress_house_efd import (
    PTR_FILING_TYPE,
    HouseEFDClient,
    HouseEFDError,
    ScannedFilingError,
    build_structurer_prompt,
    extract_pdf_text,
    ingest_house_ptrs,
    is_scanned,
    normalize_transactions,
    parse_house_ptr,
)

_FIXTURES = Path(__file__).parent / "fixtures"
_EFILED_PDF = _FIXTURES / "house_ptr_efiled.pdf"
_SCANNED_PDF = _FIXTURES / "house_ptr_scanned.pdf"


def _filing(**over) -> dict:
    base = {
        "filing_id": "20260001",
        "member_name": "Jane Smith",
        "docid": "20260001",
        "year": 2026,
        "filed_date": date(2026, 3, 15),
        "is_paper": False,
        "report_url": "https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/2026/20260001.pdf",
    }
    base.update(over)
    return base


# ─── PDF extraction + scan detection ──────────────────────────────────────────


@pytest.mark.unit
class TestExtraction:
    def test_efiled_pdf_yields_text(self):
        text = extract_pdf_text(_EFILED_PDF.read_bytes())
        assert "AAPL" in text
        assert "NVIDIA Corporation (NVDA)" in text
        assert not is_scanned(text)

    def test_scanned_pdf_is_detected(self):
        text = extract_pdf_text(_SCANNED_PDF.read_bytes())
        assert is_scanned(text)


# ─── Prompt assembly + normalization (structurer stubbed) ─────────────────────


@pytest.mark.unit
class TestParseAndNormalize:
    def _structured(self):
        # What a correct structurer returns for the e-filed fixture. The AAPL
        # ticker is wrapped in parens to prove paren-stripping; the Treasury row
        # has no ticker and must be dropped.
        return [
            {
                "owner": "SP",
                "ticker": "(AAPL)",
                "asset_name": "Apple Inc. (AAPL)",
                "type": "Purchase",
                "date": "03/01/2026",
                "amount": "$15,001 - $50,000",
            },
            {
                "owner": "JT",
                "ticker": "NVDA",
                "asset_name": "NVIDIA Corporation",
                "type": "Sale (partial)",
                "date": "03/04/2026",
                "amount": "$1,001 - $15,000",
            },
            {
                "owner": "SP",
                "ticker": "",
                "asset_name": "US Treasury Note 2.5% 2030",
                "type": "Purchase",
                "date": "03/06/2026",
                "amount": "$50,001 - $100,000",
            },
        ]

    def test_prompt_contains_instructions_and_extracted_text(self):
        text = extract_pdf_text(_EFILED_PDF.read_bytes())
        prompt = build_structurer_prompt(text)
        assert "JSON" in prompt
        assert "NO parentheses" in prompt
        assert "Apple Inc. (AAPL)" in prompt  # the raw report text is embedded

    def test_parse_feeds_extracted_text_to_structurer(self):
        captured = {}

        def stub(prompt):
            captured["prompt"] = prompt
            return self._structured()

        rows = parse_house_ptr(_EFILED_PDF.read_bytes(), filing=_filing(), structurer=stub)
        # The structurer received the assembled prompt for the extracted text.
        expected = build_structurer_prompt(extract_pdf_text(_EFILED_PDF.read_bytes()))
        assert captured["prompt"] == expected
        assert [r["ticker"] for r in rows] == ["AAPL", "NVDA"]

    def test_normalization_shape_and_paren_strip(self):
        rows = normalize_transactions(self._structured(), filing=_filing())
        assert [r["ticker"] for r in rows] == ["AAPL", "NVDA"]  # bond dropped
        aapl = rows[0]
        assert aapl == {
            "source_filing_id": "20260001",
            "row_index": 1,
            "chamber": "House",
            "member_name": "Jane Smith",
            "party": "",
            "owner_type": "SP",
            "committee": None,
            "ticker": "AAPL",  # parens stripped
            "asset_name": "Apple Inc. (AAPL)",
            "transaction_type": "Purchase",
            "amount_range": "$15,001 - $50,000",
            "transaction_date": date(2026, 3, 1),
            "disclosure_date": date(2026, 3, 15),
            "source_url": _filing()["report_url"],
        }
        # row_index is reassigned over surviving rows (bond skipped, not gap=3).
        assert [r["row_index"] for r in rows] == [1, 2]

    def test_scanned_pdf_raises_before_structurer(self):
        def stub(prompt):  # pragma: no cover - must never be called
            raise AssertionError("structurer called on a scan")

        with pytest.raises(ScannedFilingError):
            parse_house_ptr(_SCANNED_PDF.read_bytes(), filing=_filing(), structurer=stub)


# ─── Bulk index parsing + e-filed/paper classification ────────────────────────

_HEADER = "Prefix\tLast\tFirst\tSuffix\tFilingType\tStateDst\tYear\tFilingDate\tDocID"


def _index_zip(rows: list[str], *, name="2026FD.txt") -> bytes:
    body = "\n".join([_HEADER, *rows])
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, body)
    return buf.getvalue()


@pytest.mark.unit
class TestIndex:
    def test_filters_to_ptr_and_classifies_by_docid_prefix(self):
        rows = [
            "Hon.\tSmith\tJane\t\tP\tCA01\t2026\t03/15/2026\t20260001",  # e-filed PTR
            "Hon.\tDoe\tJohn\t\tP\tNY02\t2026\t03/10/2026\t10260002",  # paper PTR (prefix 1)
            "Hon.\tRoe\tRich\t\tO\tTX03\t2026\t05/15/2026\t20260003",  # not a PTR (FilingType O)
        ]
        filings = HouseEFDClient.parse_index(_index_zip(rows), 2026)
        assert len(filings) == 2  # only the two PTRs; the FD original is dropped

        efiled, paper = filings
        assert efiled["docid"] == "20260001"
        assert efiled["is_paper"] is False
        assert efiled["member_name"] == "Jane Smith"
        assert efiled["filed_date"] == date(2026, 3, 15)
        assert efiled["report_url"].endswith("/ptr-pdfs/2026/20260001.pdf")

        assert paper["is_paper"] is True
        assert paper["report_url"] == ""  # no e-file PDF for a paper filing

    def test_bad_zip_raises(self):
        with pytest.raises(HouseEFDError):
            HouseEFDClient.parse_index(b"not a zip", 2026)

    def test_fetch_filters_to_date_window(self):
        rows = [
            "Hon.\tA\tA\t\tP\tCA01\t2026\t03/15/2026\t20260001",  # in window
            "Hon.\tB\tB\t\tP\tNY02\t2026\t01/02/2026\t20260002",  # before window
        ]

        class _FakeResp:
            content = _index_zip(rows)

            def raise_for_status(self):
                pass

        class _FakeSession:
            headers = {}

            def get(self, url, timeout=None):
                return _FakeResp()

        client = HouseEFDClient(session=_FakeSession())
        filings = client.fetch_ptr_filings(
            2026, start_date=date(2026, 3, 1), end_date=date(2026, 3, 31)
        )
        assert [f["docid"] for f in filings] == ["20260001"]

    def test_filing_type_constant(self):
        assert PTR_FILING_TYPE == "P"


# ─── Ingest pipeline (fake client + store + structurer) ───────────────────────


class _FakeStore:
    def __init__(self):
        self.upserted = []
        self.closed = False

    def upsert(self, rows):
        self.upserted.extend(rows)
        return len(rows)

    def close(self):
        self.closed = True


class _FakeClient:
    def __init__(self, filings, pdf_for):
        self._filings = filings
        self._pdf_for = pdf_for  # dict: report_url -> pdf bytes
        self.fetched_urls = []

    def fetch_ptr_filings(self, year, *, start_date=None, end_date=None):
        return self._filings

    def fetch_ptr_pdf(self, report_url):
        self.fetched_urls.append(report_url)
        return self._pdf_for[report_url]


def _good_structurer(prompt):
    return [
        {
            "owner": "SP",
            "ticker": "(AAPL)",
            "asset_name": "Apple Inc. (AAPL)",
            "type": "Purchase",
            "date": "03/01/2026",
            "amount": "$15,001 - $50,000",
        },
        {
            "owner": "JT",
            "ticker": "NVDA",
            "asset_name": "NVIDIA Corporation",
            "type": "Sale",
            "date": "03/04/2026",
            "amount": "$1,001 - $15,000",
        },
        {
            "owner": "SP",
            "ticker": "",
            "asset_name": "US Treasury Note",
            "type": "Purchase",
            "date": "03/06/2026",
            "amount": "$50,001 - $100,000",
        },
    ]


@pytest.mark.unit
class TestIngest:
    def _filings(self):
        return [
            _filing(
                filing_id="20260001",
                docid="20260001",
                is_paper=False,
                report_url="https://x/ptr/efiled.pdf",
            ),
            _filing(
                filing_id="10260002",
                docid="10260002",
                is_paper=True,
                member_name="Paper Filer",
                report_url="",
            ),
        ]

    def test_parses_efiled_skips_paper_and_upserts(self):
        store = _FakeStore()
        client = _FakeClient(
            self._filings(), {"https://x/ptr/efiled.pdf": _EFILED_PDF.read_bytes()}
        )
        summary = ingest_house_ptrs(
            client=client,
            store=store,
            structurer=_good_structurer,
            year=2026,
            as_of="2026-03-31",
        )
        assert summary["filings_total"] == 2
        assert summary["filings_parsed"] == 1
        assert summary["filings_skipped_paper"] == 1
        assert summary["rows_parsed"] == 2  # AAPL + NVDA (bond dropped)
        assert summary["rows_upserted"] == 2
        # Paper filing's (empty) URL was never fetched.
        assert client.fetched_urls == ["https://x/ptr/efiled.pdf"]
        assert {r["ticker"] for r in store.upserted} == {"AAPL", "NVDA"}
        # Caller-owned store is not closed by the pipeline.
        assert store.closed is False

    def test_scanned_efile_is_skipped_not_parsed(self):
        filings = [_filing(report_url="https://x/ptr/scan.pdf")]
        client = _FakeClient(filings, {"https://x/ptr/scan.pdf": _SCANNED_PDF.read_bytes()})
        store = _FakeStore()
        summary = ingest_house_ptrs(
            client=client,
            store=store,
            structurer=_good_structurer,
            year=2026,
            as_of="2026-03-31",
        )
        assert summary["filings_skipped_scanned"] == 1
        assert summary["rows_upserted"] == 0
        assert store.upserted == []

    def test_dry_run_does_not_upsert(self):
        store = _FakeStore()
        client = _FakeClient(
            [_filing(report_url="https://x/ptr/efiled.pdf")],
            {"https://x/ptr/efiled.pdf": _EFILED_PDF.read_bytes()},
        )
        summary = ingest_house_ptrs(
            client=client,
            store=store,
            structurer=_good_structurer,
            year=2026,
            as_of="2026-03-31",
            dry_run=True,
        )
        assert summary["rows_parsed"] == 2
        assert summary["rows_upserted"] == 0
        assert store.upserted == []

    def test_per_filing_structurer_failure_is_isolated(self):
        def _boom(prompt):
            raise HouseEFDError("ollama down")

        client = _FakeClient(
            [_filing(report_url="https://x/ptr/efiled.pdf")],
            {"https://x/ptr/efiled.pdf": _EFILED_PDF.read_bytes()},
        )
        store = _FakeStore()
        summary = ingest_house_ptrs(
            client=client,
            store=store,
            structurer=_boom,
            year=2026,
            as_of="2026-03-31",
        )
        assert summary["filings_failed"] == 1
        assert summary["rows_upserted"] == 0


# ─── Configurable structurer model ────────────────────────────────────────────


@pytest.mark.unit
class TestStructurerConfig:
    def test_model_defaults_then_env_then_arg(self, monkeypatch):
        monkeypatch.delenv("CONGRESS_STRUCTURER_MODEL", raising=False)
        assert house.OllamaStructurer().model == house.DEFAULT_STRUCTURER_MODEL

        monkeypatch.setenv("CONGRESS_STRUCTURER_MODEL", "qwen3:8b")
        assert house.OllamaStructurer().model == "qwen3:8b"

        # Explicit constructor arg wins over the env var.
        assert house.OllamaStructurer(model="gemma3:12b").model == "gemma3:12b"


# ─── Live smoke (auto-skips offline) ──────────────────────────────────────────


@pytest.mark.integration
class TestLiveSmoke:
    def test_real_index_download_lists_ptrs(self):
        client = HouseEFDClient(timeout=20)
        try:
            filings = client.fetch_ptr_filings(date.today().year)
        except HouseEFDError:
            pytest.skip("House Clerk index not reachable")
        assert isinstance(filings, list)
