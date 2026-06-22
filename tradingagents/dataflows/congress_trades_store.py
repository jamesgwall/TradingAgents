"""
Congress trades store: relational read/write interface over Postgres.

The local store behind the ``fetch_congressional_trades`` seam (which previously
called a remote disclosure API). The ``congress_trades`` table lives in the same
Postgres instance as the transcript store (DB ``transcripts``)
but is a PLAIN indexed table — no pgvector. Rows are written by the nightly
congress ingestor (later slices) and read here by the
CongressionalTradesAnalyst through the unchanged ``fetch_congressional_trades``
seam.

The canonical schema lives in the orchestrator repo at
``data/pgvector/init/init.sql`` (runs on a fresh Docker volume). ``_ensure_schema``
mirrors it idempotently so the table also appears on an already-running volume
without a destructive wipe — keep the two definitions in sync.

Standard library + psycopg2 only (no pgvector needed for this plain table).

Env vars (all optional — defaults shown), shared with the transcript store:
    PGVECTOR_HOST     localhost
    PGVECTOR_PORT     5432
    PGVECTOR_DB       transcripts
    PGVECTOR_USER     postgres
    PGVECTOR_PASSWORD postgres
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime

log = logging.getLogger(__name__)

DEFAULT_WINDOW_DAYS = 180

# Columns the ingestor supplies per row. ``id`` and ``ingested_at`` are filled by
# the database defaults.
_UPSERT_COLUMNS = (
    "source_filing_id",
    "row_index",
    "chamber",
    "member_name",
    "party",
    "owner_type",
    "committee",
    "ticker",
    "asset_name",
    "transaction_type",
    "amount_range",
    "transaction_date",
    "disclosure_date",
    "source_url",
)

# Mirror of data/pgvector/init/init.sql — applied idempotently so the table
# exists on an already-running volume. Keep in sync with that file.
_SCHEMA_DDL = """
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS congress_trades (
    id                uuid        PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_filing_id  text        NOT NULL,
    row_index         int         NOT NULL,
    chamber           text        NOT NULL DEFAULT '',
    member_name       text        NOT NULL,
    party             text        NOT NULL DEFAULT '',
    owner_type        text        NOT NULL DEFAULT '',
    committee         text,
    ticker            text        NOT NULL,
    asset_name        text        NOT NULL DEFAULT '',
    transaction_type  text        NOT NULL DEFAULT '',
    amount_range      text        NOT NULL DEFAULT '',
    transaction_date  date,
    disclosure_date   date        NOT NULL,
    source_url        text        NOT NULL DEFAULT '',
    ingested_at       timestamptz NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS uix_congress_filing_row
    ON congress_trades (source_filing_id, row_index);

CREATE INDEX IF NOT EXISTS idx_congress_ticker_disclosure
    ON congress_trades (ticker, disclosure_date DESC);
"""


class CongressTradesError(RuntimeError):
    """Raised when the congress_trades store is unreachable or misconfigured."""


# ─── Normalization helpers (shared by the analyst) ────────────────────────────

def _parse_date(value) -> date | None:
    """Best-effort parse of a date string/object into a ``date``.

    Returns None for missing/unparseable values so callers can skip a record
    rather than crash the whole nightly run on one malformed row.
    """
    if not value:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("Z", "").split("T")[0].split(" ")[0]
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        return None


def _normalize_transaction(raw: str | None) -> str:
    """Map a transaction string onto Buy / Sell / the raw value."""
    if not raw:
        return "Unknown"
    text = str(raw).strip().lower()
    if text.startswith("purchase") or text.startswith("buy"):
        return "Buy"
    if text.startswith("sale") or text.startswith("sell"):
        return "Sell"
    return str(raw).strip()


def _normalize_chamber(raw: str | None) -> str:
    """Map a raw chamber string onto a human chamber label."""
    if not raw:
        return ""
    text = str(raw).strip().lower()
    if text in ("representatives", "house", "rep"):
        return "House"
    if text in ("senate", "sen"):
        return "Senate"
    return str(raw).strip()


def _normalize_trader(raw: str | None) -> str:
    """Render the owner type as a display label (e.g. ``self`` -> ``Self``)."""
    text = str(raw or "").strip()
    return text.title() if text else "Self"


def _row_to_normalized(row: dict) -> dict:
    """Map a raw ``congress_trades`` row onto the analyst's stable dict shape."""
    trade = _parse_date(row.get("transaction_date"))
    disclosure = _parse_date(row.get("disclosure_date"))
    lag = (disclosure - trade).days if (disclosure and trade) else None
    return {
        "representative":   str(row.get("member_name") or "").strip(),
        "party":            str(row.get("party") or "").strip(),
        "chamber":          _normalize_chamber(row.get("chamber")),
        "committee":        str(row.get("committee") or "").strip(),
        "trader":           _normalize_trader(row.get("owner_type")),
        "transaction_type": _normalize_transaction(row.get("transaction_type")),
        "amount_range":     str(row.get("amount_range") or "").strip(),
        "trade_date":       trade,
        "disclosure_date":  disclosure,
        "lag_days":         lag,
    }


# ─── Connection ───────────────────────────────────────────────────────────────

def _open_conn():
    try:
        import psycopg2
    except ImportError as err:
        raise ImportError(
            "psycopg2-binary is required for the congress trades store. "
            "Run: pip install psycopg2-binary"
        ) from err
    return psycopg2.connect(
        host=os.environ.get("PGVECTOR_HOST", "localhost"),
        port=int(os.environ.get("PGVECTOR_PORT", "5432")),
        dbname=os.environ.get("PGVECTOR_DB", "transcripts"),
        user=os.environ.get("PGVECTOR_USER", "postgres"),
        password=os.environ.get("PGVECTOR_PASSWORD", "postgres"),
    )


class CongressTradesStore:
    """
    Read/write interface to the relational ``congress_trades`` table.

    Instantiate once per use; call ``close()`` when done. The connection is
    opened lazily and the schema ensured on first use.
    """

    def __init__(self, ensure_schema: bool = True):
        self._conn = None
        self._ensure_schema = ensure_schema

    def _ensure_conn(self):
        if self._conn is not None and not self._conn.closed:
            return self._conn
        try:
            self._conn = _open_conn()
        except Exception as err:  # connection refused, bad creds, etc.
            raise CongressTradesError(
                f"Could not connect to the congress_trades store: {err}"
            ) from err
        if self._ensure_schema:
            self._apply_schema()
        return self._conn

    def _apply_schema(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(_SCHEMA_DDL)
        self._conn.commit()

    def close(self) -> None:
        if self._conn and not self._conn.closed:
            self._conn.close()

    def upsert(self, rows: list[dict]) -> int:
        """
        Idempotently insert/update congress trade rows.

        Dedup key is ``(source_filing_id, row_index)`` so inserting the same
        batch twice yields no duplicates. ``rows`` are dicts keyed by the
        ``congress_trades`` column names (``id``/``ingested_at`` are set by the
        database). Returns the number of rows processed.
        """
        if not rows:
            return 0
        conn = self._ensure_conn()
        placeholders = ", ".join(["%s"] * len(_UPSERT_COLUMNS))
        updatable = [c for c in _UPSERT_COLUMNS if c not in ("source_filing_id", "row_index")]
        set_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in updatable)
        sql = (
            f"INSERT INTO congress_trades ({', '.join(_UPSERT_COLUMNS)}) "
            f"VALUES ({placeholders}) "
            f"ON CONFLICT (source_filing_id, row_index) DO UPDATE SET {set_clause}"
        )
        with conn.cursor() as cur:
            for row in rows:
                cur.execute(sql, tuple(row.get(c) for c in _UPSERT_COLUMNS))
        conn.commit()
        return len(rows)

    def latest_disclosure_date(self) -> date | None:
        """Most recent ``disclosure_date`` in the table, or None when empty.

        Used by the nightly pre-step (slice 5) to fetch only filings newer than
        what is already stored. Returns None on an empty table so the caller
        falls back to a default lookback window.
        """
        conn = self._ensure_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT MAX(disclosure_date) FROM congress_trades")
            row = cur.fetchone()
        return _parse_date(row[0]) if row and row[0] is not None else None

    def query(
        self,
        ticker: str,
        as_of: str | date | None = None,
        days: int = DEFAULT_WINDOW_DAYS,
    ) -> list[dict]:
        """
        Return normalized in-window disclosures for ``ticker``.

        The window is anchored on the *disclosure date*: rows whose disclosure
        date falls within ``days`` of ``as_of`` (inclusive), newest disclosure
        first. Output matches the legacy fetch shape exactly so the analyst,
        graph wiring, and bull/bear/neutral integration are unchanged.
        """
        conn = self._ensure_conn()
        reference = _parse_date(as_of) or date.today()
        earliest = date.fromordinal(reference.toordinal() - days)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT source_filing_id, row_index, chamber, member_name, party,
                       owner_type, committee, ticker, asset_name, transaction_type,
                       amount_range, transaction_date, disclosure_date, source_url
                FROM congress_trades
                WHERE ticker = %s
                  AND disclosure_date BETWEEN %s AND %s
                ORDER BY disclosure_date DESC
                """,
                (ticker.upper(), earliest, reference),
            )
            cols = [d[0] for d in cur.description]
            raw = [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]
        return [_row_to_normalized(r) for r in raw]


def fetch_congressional_trades(
    ticker: str,
    as_of: str | date | None = None,
    days: int = DEFAULT_WINDOW_DAYS,
) -> list[dict]:
    """Fetch normalized congressional trade disclosures for ``ticker``.

    Identical signature and normalized return shape to the legacy remote fetch,
    now sourced from the local ``congress_trades`` store. Each dict is shaped::

        {
            "representative":  "Jane Smith",
            "party":           "D",
            "chamber":         "Senate",
            "committee":       "Banking",     # "" when absent
            "trader":          "Self",        # Self / Spouse / ...
            "transaction_type":"Buy",
            "amount_range":    "$15,001 - $50,000",
            "trade_date":      date(2026, 3, 1),
            "disclosure_date": date(2026, 4, 10),
            "lag_days":        40,            # disclosure - trade, None if unknown
        }

    Records are filtered to those whose disclosure date falls within ``days`` of
    ``as_of`` (inclusive) and sorted by disclosure date, newest first.

    Raises:
        CongressTradesError: if the store is unreachable.
    """
    store = CongressTradesStore()
    try:
        rows = store.query(ticker, as_of=as_of, days=days)
    finally:
        store.close()
    log.info(
        "congress_trades store: %d trade(s) for %s within %dd window",
        len(rows), ticker, days,
    )
    return rows
