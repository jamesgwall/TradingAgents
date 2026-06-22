"""
House e-filed PTR source → congress_trades store.

Fetches Periodic Transaction Reports (PTRs) from the U.S. House Clerk's
authoritative financial-disclosure bulk index
(https://disclosures-clerk.house.gov) and lands the parsed transactions in the
local ``congress_trades`` store behind the ``fetch_congressional_trades`` seam.

Flow:

1. Download the Clerk's annual financial-disclosure bulk ZIP, which contains a
   tab-delimited index of every filing for the year. Filter to PTR filings
   (``FilingType == "P"``).
2. Classify each filing as **e-filed** or **paper** by its document-id prefix.
   E-filed PTRs are machine-generated PDFs; paper filings are scanned images.
3. For each e-filed filing, fetch its PTR PDF from the Clerk's public
   ``ptr-pdfs`` path by document id.
4. Extract the PDF text deterministically with PyMuPDF, then structure the
   transactions via an **injected local-LLM structurer** (Ollama gemma
   ``e4b-it-qat`` by default, configurable via ``CONGRESS_STRUCTURER_MODEL``).
   The ticker often appears in parentheses inside the asset name — surrounding
   parentheses are stripped during normalization.
5. **Paper filings** (and any e-filed PDF that extracts near-zero text — i.e. a
   scan) are skipped with a logged warning, never guessed at. Local vision
   models proved unreliable on these forms, so there is no fallback.
6. Normalize to the store row shape and upsert (idempotent on
   ``(source_filing_id, row_index)``).

Run on demand (before the nightly pre-step is wired):

    python -m tradingagents.dataflows.congress_house_efd --year 2026 -v
    python -m tradingagents.dataflows.congress_house_efd --days 30 --dry-run

Standard library + ``requests`` + ``PyMuPDF`` (``fitz``), all TradingAgents deps.
The structurer talks to a local Ollama over the same HTTP convention the
transcript pipeline uses (``OLLAMA_BASE_URL``).
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import urllib.request
import zipfile
from collections.abc import Callable
from datetime import date, datetime, timedelta

import requests

from tradingagents.dataflows.congress_trades_store import CongressTradesStore

log = logging.getLogger(__name__)

# ─── Portal constants ─────────────────────────────────────────────────────────

BASE_URL = "https://disclosures-clerk.house.gov"
# Annual bulk index: a ZIP holding ``{YEAR}FD.txt`` (tab-delimited) + an XML.
INDEX_PATH = "/public_disc/financial-pdfs/{year}FD.zip"
# E-filed PTR PDFs are served by document id under this path.
PTR_PDF_PATH = "/public_disc/ptr-pdfs/{year}/{docid}.pdf"

PTR_FILING_TYPE = "P"     # FilingType code for a Periodic Transaction Report
# E-filed disclosure document ids start with this digit; paper-filing ids do
# not (they are older/lower ranges). Classification by prefix per the Clerk's
# observed numbering — the single knob to adjust if the convention shifts.
_EFILED_DOCID_PREFIX = "2"

REQUEST_TIMEOUT = 30      # seconds, consistent with the other vendors
DEFAULT_LOOKBACK_DAYS = 30
# Below this many non-whitespace characters an extracted PDF is treated as a
# scan (paper filing or image-only e-file) rather than machine-readable text.
SCANNED_TEXT_THRESHOLD = 40

_UA = "tradingagents/0.2 (+https://github.com/TauricResearch/TradingAgents)"

# Local-LLM structurer model. Gemma ``e4b-it-qat`` structured a real complex
# e-filed PTR at 18/18 transactions correct in prototyping, beating larger
# variants. Override without code edits via CONGRESS_STRUCTURER_MODEL.
DEFAULT_STRUCTURER_MODEL = "gemma3n:e4b-it-qat"

# Ticker-column sentinels meaning "no exchange symbol" (non-equity assets).
_NO_TICKER = {"", "--", "—", "n/a", "na", "none", "n/a."}

# A signature any callable structurer must satisfy: prompt -> list of raw
# transaction dicts (owner/ticker/asset_name/type/date/amount keys).
Structurer = Callable[[str], list[dict]]


class HouseEFDError(RuntimeError):
    """Raised when the House bulk index or a PTR PDF fetch/parse fails."""


class ScannedFilingError(HouseEFDError):
    """Raised when a PDF extracts near-zero text (a scan, not machine-readable)."""


# ─── Small parsing helpers ────────────────────────────────────────────────────

def _clean(value) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _parse_mdy(value) -> date | None:
    """Parse the Clerk's MM/DD/YYYY dates; tolerate trailing time/garbage."""
    text = _clean(value)
    if not text:
        return None
    text = text.split(" ")[0]
    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
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


def _is_paper_docid(docid: str) -> bool:
    """E-filed document ids are numeric and start with the e-filed prefix."""
    docid = _clean(docid)
    return not (docid.isdigit() and docid.startswith(_EFILED_DOCID_PREFIX))


def _strip_ticker(raw_ticker) -> str:
    """Normalize a ticker: drop surrounding parentheses/brackets, uppercase.

    The structurer often echoes the symbol exactly as it appears in the asset
    name (e.g. ``Apple Inc. (AAPL)`` -> ``(AAPL)``); strip the wrappers.
    """
    text = _clean(raw_ticker)
    text = re.sub(r"^[\(\[\{]+|[\)\]\}]+$", "", text).strip()
    return text.upper()


def _filing_from_index_row(fields: list[str], year: int) -> dict:
    """Map one tab-delimited index row onto a filing descriptor.

    Index columns: Prefix, Last, First, Suffix, FilingType, StateDst, Year,
    FilingDate, DocID.
    """
    def get(i: int) -> str:
        return _clean(fields[i]) if i < len(fields) else ""

    last, first = get(1), get(2)
    docid = get(8)
    is_paper = _is_paper_docid(docid)
    report_url = "" if is_paper else BASE_URL + PTR_PDF_PATH.format(year=year, docid=docid)
    return {
        "filing_id": docid,
        "member_name": _clean(f"{first} {last}"),
        "docid": docid,
        "year": year,
        "filed_date": _parse_mdy(get(7)),
        "is_paper": is_paper,
        "report_url": report_url,
    }


# ─── PDF text extraction + scan detection ─────────────────────────────────────

def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract all text from a PDF deterministically with PyMuPDF."""
    try:
        import fitz  # PyMuPDF
    except ImportError as err:  # pragma: no cover - dependency guard
        raise HouseEFDError(
            "PyMuPDF is required for House PTR extraction. Run: pip install pymupdf"
        ) from err
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            return "\n".join(page.get_text() for page in doc)
    except Exception as err:  # corrupt/unreadable PDF
        raise HouseEFDError(f"Could not read PTR PDF: {err}") from err


def is_scanned(text: str) -> bool:
    """A PDF is treated as a scan when it yields near-zero machine-readable text."""
    return len(re.sub(r"\s+", "", text or "")) < SCANNED_TEXT_THRESHOLD


# ─── Structurer (prompt assembly + injected LLM) ──────────────────────────────

def build_structurer_prompt(text: str) -> str:
    """Assemble the local-LLM prompt that turns raw PTR text into JSON rows."""
    return (
        "You are a precise data-extraction tool for U.S. House Periodic "
        "Transaction Reports (PTRs). Extract every transaction from the report "
        "text below.\n\n"
        "Return ONLY a JSON object of the form:\n"
        '{"transactions": [{"owner": "", "ticker": "", "asset_name": "", '
        '"type": "", "date": "MM/DD/YYYY", "amount": ""}]}\n\n'
        "Rules:\n"
        "- owner: the owner code/label as written (e.g. SP, DC, JT, Self).\n"
        "- ticker: the stock exchange symbol only, with NO parentheses. If the "
        "asset has no ticker (bonds, funds without a symbol, real estate), use "
        "an empty string.\n"
        "- asset_name: the full asset description.\n"
        "- type: the transaction type as written (Purchase, Sale, etc.).\n"
        "- date: the transaction date in MM/DD/YYYY.\n"
        "- amount: the disclosed amount range as written.\n"
        "- Do not invent rows. Do not add commentary.\n\n"
        "REPORT TEXT:\n"
        f"{text}\n"
    )


def _coerce_structurer_output(raw) -> list[dict]:
    """Accept either a bare list or a ``{'transactions': [...]}`` envelope."""
    if isinstance(raw, dict):
        raw = raw.get("transactions", [])
    if not isinstance(raw, list):
        return []
    return [r for r in raw if isinstance(r, dict)]


class OllamaStructurer:
    """Default structurer: a local Ollama model emitting JSON transactions.

    Callable as ``structurer(prompt) -> list[dict]``. The model is configurable
    (constructor arg > ``CONGRESS_STRUCTURER_MODEL`` env > built-in default) so
    the structurer can be swapped without code edits.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        ollama_url: str | None = None,
        timeout: int = 120,
    ):
        self.model = model or os.environ.get("CONGRESS_STRUCTURER_MODEL", DEFAULT_STRUCTURER_MODEL)
        self._base = (
            ollama_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        ).rstrip("/")
        self._timeout = timeout

    def __call__(self, prompt: str) -> list[dict]:
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0},
        }).encode()
        req = urllib.request.Request(
            f"{self._base}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read())
        except Exception as err:
            raise HouseEFDError(f"Ollama structurer call failed: {err}") from err
        body = data.get("response", "")
        try:
            parsed = json.loads(body)
        except (json.JSONDecodeError, TypeError) as err:
            raise HouseEFDError(f"Ollama structurer returned non-JSON: {err}") from err
        return _coerce_structurer_output(parsed)


def normalize_transactions(raw_transactions: list[dict], *, filing: dict) -> list[dict]:
    """Map structured transactions onto ``congress_trades`` rows.

    Rows without an exchange ticker (after paren-stripping) are dropped — the
    analyst is keyed on ticker. ``row_index`` is assigned by surviving-row order
    so the dedup key is stable across re-runs of the same filing.
    """
    rows: list[dict] = []
    for raw in raw_transactions:
        ticker = _strip_ticker(raw.get("ticker"))
        if ticker.lower() in _NO_TICKER:
            continue
        rows.append({
            "source_filing_id": filing["filing_id"],
            "row_index": len(rows) + 1,
            "chamber": "House",
            "member_name": filing["member_name"],
            "party": "",
            "owner_type": _clean(raw.get("owner")),
            "committee": None,
            "ticker": ticker,
            "asset_name": _clean(raw.get("asset_name")),
            "transaction_type": _clean(raw.get("type")),
            "amount_range": _clean(raw.get("amount")),
            "transaction_date": _parse_mdy(raw.get("date")),
            "disclosure_date": filing["filed_date"],
            "source_url": filing["report_url"],
        })
    return rows


def parse_house_ptr(pdf_bytes: bytes, *, filing: dict, structurer: Structurer) -> list[dict]:
    """Extract → (scan-guard) → structure → normalize one House PTR PDF.

    Raises ``ScannedFilingError`` for image-only PDFs (the caller skips them) and
    ``HouseEFDError`` if the structurer fails.
    """
    text = extract_pdf_text(pdf_bytes)
    if is_scanned(text):
        raise ScannedFilingError(
            f"PTR PDF {filing.get('report_url') or filing.get('filing_id')} "
            "is a scan (near-zero extractable text)"
        )
    prompt = build_structurer_prompt(text)
    raw = structurer(prompt)
    return normalize_transactions(raw, filing=filing)


# ─── House Clerk client ───────────────────────────────────────────────────────

class HouseEFDClient:
    """Downloads the bulk index and fetches PTR PDFs from the House Clerk."""

    def __init__(self, *, timeout: int = REQUEST_TIMEOUT, session: requests.Session | None = None):
        self._timeout = timeout
        self._session = session or requests.Session()
        self._session.headers.setdefault("User-Agent", _UA)

    def _download_index_zip(self, year: int) -> bytes:
        url = BASE_URL + INDEX_PATH.format(year=year)
        try:
            resp = self._session.get(url, timeout=self._timeout)
            resp.raise_for_status()
        except requests.RequestException as err:
            raise HouseEFDError(f"House bulk index download failed for {year}: {err}") from err
        return resp.content

    @staticmethod
    def parse_index(zip_bytes: bytes, year: int) -> list[dict]:
        """Parse the bulk ZIP, returning PTR filing descriptors only."""
        try:
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                txt_names = [n for n in zf.namelist() if n.lower().endswith(".txt")]
                if not txt_names:
                    raise HouseEFDError("House bulk index ZIP contained no .txt index")
                raw = zf.read(txt_names[0]).decode("utf-8", errors="replace")
        except zipfile.BadZipFile as err:
            raise HouseEFDError(f"House bulk index is not a valid ZIP: {err}") from err

        filings: list[dict] = []
        lines = raw.splitlines()
        for line in lines[1:]:  # row 0 is the header
            if not line.strip():
                continue
            fields = line.split("\t")
            filing_type = _clean(fields[4]) if len(fields) > 4 else ""
            if filing_type != PTR_FILING_TYPE:
                continue
            filings.append(_filing_from_index_row(fields, year))
        return filings

    def fetch_ptr_filings(
        self,
        year: int,
        *,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[dict]:
        """Return PTR filing descriptors for ``year``, optionally date-windowed.

        The window is applied to the filing (disclosure) date when both bounds
        are given; filings without a parseable date are kept.
        """
        filings = self.parse_index(self._download_index_zip(year), year)
        if start_date is None or end_date is None:
            return filings
        return [
            f for f in filings
            if f["filed_date"] is None or start_date <= f["filed_date"] <= end_date
        ]

    def fetch_ptr_pdf(self, report_url: str) -> bytes:
        try:
            resp = self._session.get(report_url, timeout=self._timeout)
            resp.raise_for_status()
        except requests.RequestException as err:
            raise HouseEFDError(f"PTR PDF fetch failed for {report_url}: {err}") from err
        return resp.content


# ─── Orchestration ────────────────────────────────────────────────────────────

def ingest_house_ptrs(
    *,
    year: int | None = None,
    days: int = DEFAULT_LOOKBACK_DAYS,
    as_of: str | date | None = None,
    client: HouseEFDClient | None = None,
    store: CongressTradesStore | None = None,
    structurer: Structurer | None = None,
    dry_run: bool = False,
) -> dict:
    """Fetch, parse, and upsert House e-filed PTRs in a filing-date window.

    Paper filings and scans are skipped with a logged warning. Per-filing
    fetch/parse/structure errors are logged and skipped so one bad report can't
    abort the run. A failed index download raises ``HouseEFDError`` (the caller
    decides how to fail open). Returns a summary dict of counts.
    """
    reference = _coerce_date(as_of) or date.today()
    start = reference - timedelta(days=days)
    year = year or reference.year
    client = client or HouseEFDClient()
    structurer = structurer or OllamaStructurer()

    filings = client.fetch_ptr_filings(year, start_date=start, end_date=reference)
    parsed_rows: list[dict] = []
    parsed_filings = 0
    skipped_paper = 0
    skipped_scanned = 0
    failed_filings = 0

    for filing in filings:
        if filing["is_paper"]:
            skipped_paper += 1
            log.warning(
                "Skipping House paper filing (scanned, not machine-readable): docid=%s",
                filing["docid"],
            )
            continue
        try:
            pdf = client.fetch_ptr_pdf(filing["report_url"])
            rows = parse_house_ptr(pdf, filing=filing, structurer=structurer)
        except ScannedFilingError as err:
            skipped_scanned += 1
            log.warning("Skipping House scanned e-file: %s", err)
            continue
        except (HouseEFDError, ValueError) as err:
            failed_filings += 1
            log.warning("Failed to fetch/parse House PTR %s: %s", filing["report_url"], err)
            continue
        parsed_rows.extend(rows)
        parsed_filings += 1

    summary = {
        "year": year,
        "filings_total": len(filings),
        "filings_parsed": parsed_filings,
        "filings_skipped_paper": skipped_paper,
        "filings_skipped_scanned": skipped_scanned,
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
        description="Ingest House e-filed PTRs from the Clerk bulk index into the congress_trades store."
    )
    parser.add_argument("--year", type=int, default=None,
                        help="disclosure year to ingest (default: year of --as-of/today)")
    parser.add_argument("--days", type=int, default=DEFAULT_LOOKBACK_DAYS,
                        help=f"filing-date lookback window (default {DEFAULT_LOOKBACK_DAYS})")
    parser.add_argument("--as-of", default=None,
                        help="window end date (YYYY-MM-DD); defaults to today")
    parser.add_argument("--model", default=None,
                        help="Ollama structurer model (overrides CONGRESS_STRUCTURER_MODEL)")
    parser.add_argument("--dry-run", action="store_true",
                        help="fetch and parse but do not write to the store")
    parser.add_argument("-v", "--verbose", action="store_true", help="log INFO progress")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    structurer = OllamaStructurer(model=args.model) if args.model else None
    try:
        summary = ingest_house_ptrs(
            year=args.year, days=args.days, as_of=args.as_of,
            structurer=structurer, dry_run=args.dry_run,
        )
    except HouseEFDError as err:
        log.error("House EFD ingestion failed: %s", err)
        return 1
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
