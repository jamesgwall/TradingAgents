"""
Senate e-filed PTR source → congress_trades store.

Fetches Periodic Transaction Reports (PTRs) from the U.S. Senate's Electronic
Financial Disclosure (EFD) portal (https://efdsearch.senate.gov) and lands the
parsed transactions in the local ``congress_trades`` store behind the
``fetch_congressional_trades`` seam.

Flow (the handshake was verified during design prototyping):

1. GET the search/home page to obtain the CSRF cookie + the hidden form token.
2. POST the prohibition-agreement acceptance to establish the session.
3. POST the report-data query (filtered to the PTR report type and a submitted-
   date window) to retrieve the JSON filing list (DataTables server-side).
4. For each filing, fetch its per-report view page. **E-filed** reports are clean
   HTML tables with a dedicated ticker column and are parsed deterministically
   (no LLM). **Paper** filings (distinguishable by their view-URL pattern) are
   scanned images — they are skipped with a logged warning, never guessed at.
5. Normalize to the store row shape and upsert (idempotent on
   ``(source_filing_id, row_index)``).

Run on demand (before the nightly pre-step is wired):

    python -m tradingagents.dataflows.congress_senate_efd --days 30 -v
    python -m tradingagents.dataflows.congress_senate_efd --days 7 --dry-run

Standard library + ``requests`` + ``parsel`` (both already TradingAgents deps).
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime, timedelta

import requests
from parsel import Selector

from tradingagents.dataflows.congress_committees import (
    CommitteeResolver,
    enrich_committees,
)
from tradingagents.dataflows.congress_trades_store import CongressTradesStore

log = logging.getLogger(__name__)

# ─── Portal constants ─────────────────────────────────────────────────────────

BASE_URL = "https://efdsearch.senate.gov"
HOME_PATH = "/search/home/"
SEARCH_PATH = "/search/"
REPORT_DATA_PATH = "/search/report/data/"

# Report-type and filer-type codes used by the EFD report-data form.
PTR_REPORT_TYPE = "11"                 # Periodic Transaction Report
SENATOR_FILER_TYPES = ("1", "4", "5")  # Senator, Candidate, Former Senator

PAGE_LENGTH = 100        # DataTables page size for the report-data query
REQUEST_TIMEOUT = 30     # seconds, consistent with the other vendors
DEFAULT_LOOKBACK_DAYS = 30

_UA = "tradingagents/0.2 (+https://github.com/TauricResearch/TradingAgents)"

# View-URL discriminators: e-filed PTRs carry a uuid under /view/ptr/, paper
# filings under /view/paper/ (scanned images — not machine-readable).
_EFILED_PTR_RE = re.compile(r"/search/view/ptr/([^/]+)/?")
_PAPER_RE = re.compile(r"/search/view/paper/")
_HREF_RE = re.compile(r'href=["\']([^"\']+)["\']', re.I)

# Ticker-column sentinels meaning "no exchange symbol" (non-equity assets).
_NO_TICKER = {"", "--", "—", "n/a", "na", "none"}


class SenateEFDError(RuntimeError):
    """Raised when the EFD portal handshake or a report query fails."""


# ─── Small parsing helpers ────────────────────────────────────────────────────

def _clean(value) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _fmt_mdy(d: date) -> str:
    return d.strftime("%m/%d/%Y")


def _parse_mdy(value) -> date | None:
    """Parse the portal's MM/DD/YYYY dates; tolerate trailing time/garbage."""
    text = _clean(value)
    if not text:
        return None
    text = text.split(" ")[0]
    try:
        return datetime.strptime(text, "%m/%d/%Y").date()
    except ValueError:
        return None


def _coerce_date(value) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = _clean(value)
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _to_int(value, default: int) -> int:
    try:
        return int(re.sub(r"\D", "", str(value)))
    except (ValueError, TypeError):
        return default


def _href_from_anchor(html_fragment: str) -> str:
    """Extract the first href from a DataTables cell containing an <a> tag."""
    match = _HREF_RE.search(html_fragment or "")
    return match.group(1) if match else ""


def _filing_id_from_href(href: str) -> str:
    """Stable filing id = the report uuid from the view URL (fallback: the href)."""
    m = _EFILED_PTR_RE.search(href or "")
    return m.group(1) if m else _clean(href)


def _filing_from_row(row: list) -> dict:
    """Map one DataTables ``data`` row onto a filing descriptor.

    Row shape (server-side DataTables): ``[first_name, last_name, office,
    report_link_html, submitted_date]``.
    """
    first = _clean(row[0]) if len(row) > 0 else ""
    last = _clean(row[1]) if len(row) > 1 else ""
    link_html = row[3] if len(row) > 3 else ""
    href = _href_from_anchor(link_html)
    report_url = (BASE_URL + href) if href.startswith("/") else href
    return {
        "member_name": _clean(f"{first} {last}"),
        "report_url": report_url,
        "filed_date": _parse_mdy(row[4]) if len(row) > 4 else None,
        "is_paper": bool(href and _PAPER_RE.search(href)),
        "filing_id": _filing_id_from_href(href),
    }


# ─── HTML report parser (deterministic, no LLM) ───────────────────────────────

def parse_ptr_html(html: str, *, filing: dict) -> list[dict]:
    """Parse an e-filed Senate PTR view page into ``congress_trades`` rows.

    The e-filed table columns are: #, transaction date, owner, ticker, asset
    name, asset type, type, amount, comment. Rows without an exchange ticker
    (non-equity assets) are dropped — the analyst is keyed on ticker.

    Returns store-shaped row dicts (keyed by the table's column names) ready for
    ``CongressTradesStore.upsert``.
    """
    sel = Selector(text=html)
    rows: list[dict] = []
    for tr in sel.css("table tbody tr"):
        cells = [_clean(" ".join(td.css("::text").getall())) for td in tr.css("td")]
        if len(cells) < 8:
            continue
        num, txn_date, owner, ticker, asset_name, _asset_type, txn_type, amount = cells[:8]
        if ticker.lower() in _NO_TICKER:
            continue
        rows.append({
            "source_filing_id": filing["filing_id"],
            "row_index": _to_int(num, default=len(rows)),
            "chamber": "Senate",
            "member_name": filing["member_name"],
            "party": "",
            "owner_type": owner,
            "committee": None,
            "ticker": ticker.upper(),
            "asset_name": asset_name,
            "transaction_type": txn_type,
            "amount_range": amount,
            "transaction_date": _parse_mdy(txn_date),
            "disclosure_date": filing["filed_date"],
            "source_url": filing["report_url"],
        })
    return rows


# ─── EFD portal client ────────────────────────────────────────────────────────

class SenateEFDClient:
    """Thin session wrapper over the EFD portal handshake + report queries."""

    def __init__(self, *, timeout: int = REQUEST_TIMEOUT, session: requests.Session | None = None):
        self._timeout = timeout
        self._session = session or requests.Session()
        self._session.headers.setdefault("User-Agent", _UA)
        self._csrf = ""
        self._established = False

    @staticmethod
    def _csrf_from_html(html: str) -> str:
        sel = Selector(text=html)
        return sel.css('input[name="csrfmiddlewaretoken"]::attr(value)').get() or ""

    def _establish(self) -> None:
        """Complete the CSRF + prohibition-agreement handshake (once)."""
        if self._established:
            return
        try:
            home = self._session.get(BASE_URL + HOME_PATH, timeout=self._timeout)
            home.raise_for_status()
            token = self._csrf_from_html(home.text) or self._session.cookies.get("csrftoken", "")
            if not token:
                raise SenateEFDError("EFD home page did not yield a CSRF token")
            agree = self._session.post(
                BASE_URL + HOME_PATH,
                data={"prohibition_agreement": "1", "csrfmiddlewaretoken": token},
                headers={"Referer": BASE_URL + HOME_PATH},
                timeout=self._timeout,
            )
            agree.raise_for_status()
        except requests.RequestException as err:
            raise SenateEFDError(f"EFD handshake failed: {err}") from err
        # The agreement POST refreshes the csrftoken cookie; prefer it.
        self._csrf = self._session.cookies.get("csrftoken", "") or token
        self._established = True

    def fetch_ptr_filings(self, start_date: date, end_date: date) -> list[dict]:
        """Return PTR filing descriptors submitted in [start_date, end_date]."""
        self._establish()
        filings: list[dict] = []
        start = 0
        while True:
            payload = {
                "draw": "1",
                "start": str(start),
                "length": str(PAGE_LENGTH),
                "report_types[]": [PTR_REPORT_TYPE],
                "filer_types[]": list(SENATOR_FILER_TYPES),
                "submitted_start_date": _fmt_mdy(start_date),
                "submitted_end_date": _fmt_mdy(end_date),
                "first_name": "",
                "last_name": "",
                "senator_state": "",
                "office_id": "",
                "candidate_state": "",
                "csrfmiddlewaretoken": self._csrf,
            }
            try:
                resp = self._session.post(
                    BASE_URL + REPORT_DATA_PATH,
                    data=payload,
                    headers={
                        "Referer": BASE_URL + SEARCH_PATH,
                        "X-Requested-With": "XMLHttpRequest",
                        "X-CSRFToken": self._csrf,
                    },
                    timeout=self._timeout,
                )
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as err:
                raise SenateEFDError(f"EFD report-data query failed: {err}") from err
            except json.JSONDecodeError as err:
                raise SenateEFDError(f"EFD report-data returned non-JSON: {err}") from err

            page = data.get("data", []) if isinstance(data, dict) else []
            filings.extend(_filing_from_row(row) for row in page)
            total = data.get("recordsFiltered", len(filings)) if isinstance(data, dict) else len(filings)
            start += PAGE_LENGTH
            if not page or start >= total:
                break
        return filings

    def fetch_report_html(self, report_url: str) -> str:
        self._establish()
        try:
            resp = self._session.get(report_url, timeout=self._timeout)
            resp.raise_for_status()
        except requests.RequestException as err:
            raise SenateEFDError(f"EFD report fetch failed for {report_url}: {err}") from err
        return resp.text


# ─── Orchestration ────────────────────────────────────────────────────────────

def ingest_senate_ptrs(
    *,
    days: int = DEFAULT_LOOKBACK_DAYS,
    as_of: str | date | None = None,
    client: SenateEFDClient | None = None,
    store: CongressTradesStore | None = None,
    committee_resolver: CommitteeResolver | None = None,
    dry_run: bool = False,
) -> dict:
    """Fetch, parse, and upsert Senate e-filed PTRs in a submitted-date window.

    Per-filing fetch/parse errors are logged and skipped so one bad report can't
    abort the run. A failed handshake or report-data query raises
    ``SenateEFDError`` (the caller decides how to fail open). Returns a summary
    dict of counts.

    ``committee_resolver`` is best-effort committee enrichment (slice 4): when
    supplied, each parsed row's ``committee`` column is populated from the
    member's standing-committee assignments. None (the default) leaves
    ``committee`` null — so existing callers and tests are unaffected.
    """
    reference = _coerce_date(as_of) or date.today()
    start = reference - timedelta(days=days)
    client = client or SenateEFDClient()

    filings = client.fetch_ptr_filings(start, reference)
    parsed_rows: list[dict] = []
    parsed_filings = 0
    skipped_paper = 0
    failed_filings = 0

    for filing in filings:
        if filing["is_paper"]:
            skipped_paper += 1
            log.warning(
                "Skipping Senate paper filing (scanned, not machine-readable): %s",
                filing["report_url"],
            )
            continue
        try:
            html = client.fetch_report_html(filing["report_url"])
            rows = parse_ptr_html(html, filing=filing)
        except (SenateEFDError, ValueError) as err:
            failed_filings += 1
            log.warning("Failed to fetch/parse Senate PTR %s: %s", filing["report_url"], err)
            continue
        parsed_rows.extend(rows)
        parsed_filings += 1

    # Best-effort committee enrichment (no-op when no resolver is supplied).
    enrich_committees(parsed_rows, committee_resolver)

    summary = {
        "filings_total": len(filings),
        "filings_parsed": parsed_filings,
        "filings_skipped_paper": skipped_paper,
        "filings_failed": failed_filings,
        "rows_parsed": len(parsed_rows),
        "rows_upserted": 0,
    }

    if dry_run or not parsed_rows:
        return summary

    own_store = store is None
    store = store or CongressTradesStore()
    try:
        summary["rows_upserted"] = store.upsert(parsed_rows)
    finally:
        if own_store:
            store.close()
    return summary


def _main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest Senate e-filed PTRs from the EFD portal into the congress_trades store."
    )
    parser.add_argument("--days", type=int, default=DEFAULT_LOOKBACK_DAYS,
                        help=f"submitted-date lookback window (default {DEFAULT_LOOKBACK_DAYS})")
    parser.add_argument("--as-of", default=None,
                        help="window end date (YYYY-MM-DD); defaults to today")
    parser.add_argument("--dry-run", action="store_true",
                        help="fetch and parse but do not write to the store")
    parser.add_argument("-v", "--verbose", action="store_true", help="log INFO progress")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        summary = ingest_senate_ptrs(
            days=args.days,
            as_of=args.as_of,
            committee_resolver=CommitteeResolver.load(),
            dry_run=args.dry_run,
        )
    except SenateEFDError as err:
        log.error("Senate EFD ingestion failed: %s", err)
        return 1
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
