"""
Tests for the Senate e-filed PTR source.

The HTML parser is asserted against a captured fixture page; the fetcher is
exercised against a fake ``requests.Session`` (handshake + PTR filtering +
pagination). The ingest pipeline is driven with fake client/store doubles. A
live smoke test hits the real EFD portal and auto-skips when offline.

Run from the TradingAgents repo root:
    python -m pytest tests/test_congress_senate_efd.py -v
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
import requests

from tradingagents.dataflows import congress_senate_efd as efd
from tradingagents.dataflows.congress_senate_efd import (
    PTR_REPORT_TYPE,
    REPORT_DATA_PATH,
    SenateEFDClient,
    SenateEFDError,
    ingest_senate_ptrs,
    parse_ptr_html,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "senate_ptr_efiled.html"


# ─── HTML parser (exact rows from a captured page) ────────────────────────────


@pytest.mark.unit
class TestParser:
    def _filing(self):
        return {
            "filing_id": "abc-123",
            "member_name": "Jane Smith",
            "report_url": "https://efdsearch.senate.gov/search/view/ptr/abc-123/",
            "filed_date": date(2026, 4, 10),
        }

    def test_parses_exact_equity_rows(self):
        html = _FIXTURE.read_text()
        rows = parse_ptr_html(html, filing=self._filing())

        # The corporate-bond row (ticker "--") must be dropped.
        assert [r["ticker"] for r in rows] == ["AAPL", "MSFT"]

        aapl, msft = rows
        assert aapl == {
            "source_filing_id": "abc-123",
            "row_index": 1,
            "chamber": "Senate",
            "member_name": "Jane Smith",
            "party": "",
            "owner_type": "Spouse",
            "committee": None,
            "ticker": "AAPL",
            "asset_name": "Apple Inc. - Common Stock",
            "transaction_type": "Purchase",
            "amount_range": "$15,001 - $50,000",
            "transaction_date": date(2026, 3, 1),
            "disclosure_date": date(2026, 4, 10),
            "source_url": "https://efdsearch.senate.gov/search/view/ptr/abc-123/",
        }
        assert msft["ticker"] == "MSFT"
        assert msft["transaction_type"] == "Sale (Full)"
        assert msft["amount_range"] == "$1,001 - $15,000"
        assert msft["transaction_date"] == date(2026, 3, 4)
        assert msft["owner_type"] == "Self"

    def test_dropped_non_equity_row(self):
        rows = parse_ptr_html(_FIXTURE.read_text(), filing=self._filing())
        assert all(r["asset_name"] != "US Treasury Note 2.5% 2030" for r in rows)


# ─── Fake HTTP plumbing for the fetcher ───────────────────────────────────────


class _FakeResponse:
    def __init__(self, *, text="", json_data=None, status=200):
        self.text = text
        self._json = json_data
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")

    def json(self):
        if self._json is None:
            raise json.JSONDecodeError("no json", "", 0)
        return self._json


class _FakeSession:
    """Records calls; serves a home page, agreement, and report-data pages."""

    def __init__(self, *, home_html, report_pages, report_html="", set_cookie=True):
        self.headers = {}
        self.cookies = {}  # dict.get(key, default) matches requests' cookie jar
        self._home_html = home_html
        self._report_pages = list(report_pages)
        self._report_html = report_html
        self._set_cookie = set_cookie
        self.gets = []
        self.posts = []

    def get(self, url, timeout=None):
        self.gets.append(url)
        if url.endswith(efd.HOME_PATH):
            if self._set_cookie:
                self.cookies["csrftoken"] = "COOKIETOKEN"
            return _FakeResponse(text=self._home_html)
        return _FakeResponse(text=self._report_html)

    def post(self, url, data=None, headers=None, timeout=None):
        self.posts.append({"url": url, "data": data, "headers": headers})
        if url.endswith(REPORT_DATA_PATH):
            page = (
                self._report_pages.pop(0)
                if self._report_pages
                else {"data": [], "recordsFiltered": 0}
            )
            return _FakeResponse(json_data=page)
        return _FakeResponse(text="agreement accepted")


_HOME_HTML = '<form><input type="hidden" name="csrfmiddlewaretoken" value="FORMTOKEN"></form>'


def _ptr_row(uuid="u1", first="Jane", last="Smith", date_str="04/10/2026"):
    href = f"/search/view/ptr/{uuid}/"
    return [first, last, "Senator", f'<a href="{href}">View</a>', date_str]


def _paper_row(uuid="p1"):
    href = f"/search/view/paper/{uuid}/"
    return ["Paper", "Filer", "Senator", f'<a href="{href}">View</a>', "04/02/2026"]


@pytest.mark.unit
class TestFetcher:
    def test_handshake_then_ptr_query(self):
        session = _FakeSession(
            home_html=_HOME_HTML,
            report_pages=[{"data": [_ptr_row(), _paper_row()], "recordsFiltered": 2}],
        )
        client = SenateEFDClient(session=session)
        filings = client.fetch_ptr_filings(date(2026, 4, 1), date(2026, 4, 30))

        # Handshake: GET home, then POST agreement, then POST report-data.
        assert session.gets[0].endswith(efd.HOME_PATH)
        assert session.posts[0]["url"].endswith(efd.HOME_PATH)
        assert session.posts[0]["data"]["prohibition_agreement"] == "1"
        report_post = session.posts[1]
        assert report_post["url"].endswith(REPORT_DATA_PATH)

        # PTR filter + window were sent on the report-data query.
        assert report_post["data"]["report_types[]"] == [PTR_REPORT_TYPE]
        assert report_post["data"]["submitted_start_date"] == "04/01/2026"
        assert report_post["data"]["submitted_end_date"] == "04/30/2026"
        # CSRF token (refreshed cookie) accompanies the AJAX POST.
        assert report_post["headers"]["X-CSRFToken"] == "COOKIETOKEN"

        # Both filings returned; paper one flagged for the caller to skip.
        assert len(filings) == 2
        ptr, paper = filings
        assert ptr["is_paper"] is False
        assert ptr["filing_id"] == "u1"
        assert ptr["member_name"] == "Jane Smith"
        assert ptr["filed_date"] == date(2026, 4, 10)
        assert paper["is_paper"] is True

    def test_pagination_walks_all_pages(self):
        # recordsFiltered=150 over PAGE_LENGTH=100 -> two report-data POSTs.
        page1 = {"data": [_ptr_row(uuid=f"a{i}") for i in range(100)], "recordsFiltered": 150}
        page2 = {"data": [_ptr_row(uuid=f"b{i}") for i in range(50)], "recordsFiltered": 150}
        session = _FakeSession(home_html=_HOME_HTML, report_pages=[page1, page2])
        client = SenateEFDClient(session=session)
        filings = client.fetch_ptr_filings(date(2026, 4, 1), date(2026, 4, 30))
        assert len(filings) == 150
        report_posts = [p for p in session.posts if p["url"].endswith(REPORT_DATA_PATH)]
        assert len(report_posts) == 2
        assert report_posts[1]["data"]["start"] == "100"

    def test_handshake_without_token_raises(self):
        session = _FakeSession(
            home_html="<form>no token here</form>", report_pages=[], set_cookie=False
        )
        client = SenateEFDClient(session=session)
        with pytest.raises(SenateEFDError):
            client.fetch_ptr_filings(date(2026, 4, 1), date(2026, 4, 30))

    def test_network_error_becomes_senate_error(self):
        class _Boom(_FakeSession):
            def get(self, url, timeout=None):
                raise requests.ConnectionError("refused")

        client = SenateEFDClient(session=_Boom(home_html="", report_pages=[]))
        with pytest.raises(SenateEFDError):
            client.fetch_ptr_filings(date(2026, 4, 1), date(2026, 4, 30))


# ─── Ingest pipeline (fake client + store) ────────────────────────────────────


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
    def __init__(self, filings, report_html):
        self._filings = filings
        self._report_html = report_html
        self.fetched_urls = []

    def fetch_ptr_filings(self, start, end):
        return self._filings

    def fetch_report_html(self, url):
        self.fetched_urls.append(url)
        return self._report_html


@pytest.mark.unit
class TestIngest:
    def _filings(self):
        return [
            {
                "filing_id": "u1",
                "member_name": "Jane Smith",
                "report_url": "https://efdsearch.senate.gov/search/view/ptr/u1/",
                "filed_date": date(2026, 4, 10),
                "is_paper": False,
            },
            {
                "filing_id": "p1",
                "member_name": "Paper Filer",
                "report_url": "https://efdsearch.senate.gov/search/view/paper/p1/",
                "filed_date": date(2026, 4, 2),
                "is_paper": True,
            },
        ]

    def test_parses_efiled_skips_paper_and_upserts(self):
        store = _FakeStore()
        client = _FakeClient(self._filings(), _FIXTURE.read_text())
        summary = ingest_senate_ptrs(client=client, store=store, days=30, as_of="2026-04-30")

        assert summary["filings_total"] == 2
        assert summary["filings_parsed"] == 1
        assert summary["filings_skipped_paper"] == 1
        assert summary["rows_parsed"] == 2  # AAPL + MSFT (bond dropped)
        assert summary["rows_upserted"] == 2
        # Paper filing was never fetched.
        assert client.fetched_urls == ["https://efdsearch.senate.gov/search/view/ptr/u1/"]
        assert {r["ticker"] for r in store.upserted} == {"AAPL", "MSFT"}
        # Caller-owned store is not closed by the pipeline.
        assert store.closed is False

    def test_dry_run_does_not_upsert(self):
        store = _FakeStore()
        client = _FakeClient(self._filings(), _FIXTURE.read_text())
        summary = ingest_senate_ptrs(client=client, store=store, dry_run=True)
        assert summary["rows_parsed"] == 2
        assert summary["rows_upserted"] == 0
        assert store.upserted == []

    def test_per_filing_failure_is_isolated(self):
        class _FlakyClient(_FakeClient):
            def fetch_report_html(self, url):
                raise SenateEFDError("report 500")

        store = _FakeStore()
        client = _FlakyClient(self._filings(), "")
        summary = ingest_senate_ptrs(client=client, store=store, as_of="2026-04-30")
        assert summary["filings_failed"] == 1
        assert summary["rows_upserted"] == 0

    def test_committee_enrichment_populates_rows(self):
        """Slice 4: an injected resolver populates the committee column."""

        class _FakeResolver:
            def committee_for(self, member_name, chamber):
                return "Banking" if member_name == "Jane Smith" else None

        store = _FakeStore()
        client = _FakeClient(self._filings(), _FIXTURE.read_text())
        ingest_senate_ptrs(
            client=client,
            store=store,
            days=30,
            as_of="2026-04-30",
            committee_resolver=_FakeResolver(),
        )
        assert {r["committee"] for r in store.upserted} == {"Banking"}

    def test_no_resolver_leaves_committee_null(self):
        store = _FakeStore()
        client = _FakeClient(self._filings(), _FIXTURE.read_text())
        ingest_senate_ptrs(client=client, store=store, days=30, as_of="2026-04-30")
        assert all(r["committee"] is None for r in store.upserted)


# ─── Live smoke (auto-skips offline) ──────────────────────────────────────────


@pytest.mark.integration
class TestLiveSmoke:
    def test_real_handshake_returns_filings(self):
        client = SenateEFDClient(timeout=20)
        try:
            filings = client.fetch_ptr_filings(date.today().replace(day=1), date.today())
        except SenateEFDError:
            pytest.skip("EFD portal not reachable")
        # The portal always has recent PTRs; just prove the round trip works.
        assert isinstance(filings, list)
