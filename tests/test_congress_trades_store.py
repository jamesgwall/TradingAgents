"""
Unit tests for the congress_trades store.

Pure-function and mocked-connection tests run with no database. A real
round-trip / idempotency test is included but auto-skips when Postgres is not
reachable.

Run from the TradingAgents repo root:
    python -m pytest tests/test_congress_trades_store.py -v
"""

from datetime import date

import pytest

from tradingagents.dataflows.congress_trades_store import (
    CongressTradesError,
    CongressTradesStore,
    _normalize_chamber,
    _normalize_transaction,
    _row_to_normalized,
    fetch_congressional_trades,
)

# ─── Fakes ────────────────────────────────────────────────────────────────────


class _FakeCursor:
    def __init__(self, rows=None, description=None):
        self._rows = rows or []
        self.description = description
        self.executed = []  # list of (sql, params)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchall(self):
        return self._rows


class _FakeConn:
    def __init__(self, cursor):
        self._cursor = cursor
        self.closed = False
        self.commits = 0

    def cursor(self):
        return self._cursor

    def commit(self):
        self.commits += 1

    def close(self):
        self.closed = True


def _store_with(conn):
    store = CongressTradesStore(ensure_schema=False)
    store._conn = conn
    return store


# ─── Normalization (the stable return shape) ──────────────────────────────────


@pytest.mark.unit
class TestNormalization:
    def test_row_maps_to_legacy_shape(self):
        raw = {
            "member_name": "Jane Smith",
            "party": "D",
            "chamber": "Senate",
            "owner_type": "spouse",
            "committee": "Banking",
            "ticker": "AAPL",
            "asset_name": "Apple Inc",
            "transaction_type": "Purchase",
            "amount_range": "$15,001 - $50,000",
            "transaction_date": date(2026, 3, 1),
            "disclosure_date": date(2026, 4, 10),
        }
        out = _row_to_normalized(raw)
        assert out == {
            "representative": "Jane Smith",
            "party": "D",
            "chamber": "Senate",
            "committee": "Banking",
            "trader": "Spouse",
            "transaction_type": "Buy",
            "amount_range": "$15,001 - $50,000",
            "trade_date": date(2026, 3, 1),
            "disclosure_date": date(2026, 4, 10),
            "lag_days": 40,
        }

    def test_lag_is_none_without_trade_date(self):
        out = _row_to_normalized(
            {"member_name": "X", "disclosure_date": date(2026, 4, 10), "transaction_date": None}
        )
        assert out["lag_days"] is None
        assert out["trade_date"] is None

    def test_blank_committee_becomes_empty_string(self):
        out = _row_to_normalized(
            {"member_name": "X", "committee": None, "disclosure_date": date(2026, 4, 10)}
        )
        assert out["committee"] == ""

    def test_string_dates_are_parsed(self):
        out = _row_to_normalized(
            {"member_name": "X", "transaction_date": "2026-03-01", "disclosure_date": "2026-04-10"}
        )
        assert out["trade_date"] == date(2026, 3, 1)
        assert out["lag_days"] == 40

    def test_owner_type_defaults_to_self(self):
        assert (
            _row_to_normalized({"member_name": "X", "disclosure_date": date(2026, 4, 10)})["trader"]
            == "Self"
        )

    def test_transaction_and_chamber_normalizers(self):
        assert _normalize_transaction("Sale (Full)") == "Sell"
        assert _normalize_transaction("buy") == "Buy"
        assert _normalize_chamber("representatives") == "House"
        assert _normalize_chamber("Senate") == "Senate"


# ─── query() ──────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestQuery:
    def _cursor(self):
        description = [
            ("source_filing_id",),
            ("row_index",),
            ("chamber",),
            ("member_name",),
            ("party",),
            ("owner_type",),
            ("committee",),
            ("ticker",),
            ("asset_name",),
            ("transaction_type",),
            ("amount_range",),
            ("transaction_date",),
            ("disclosure_date",),
            ("source_url",),
        ]
        rows = [
            (
                "F1",
                0,
                "Senate",
                "Jane Smith",
                "D",
                "self",
                "Banking",
                "AAPL",
                "Apple Inc",
                "Purchase",
                "$15K-$50K",
                date(2026, 3, 1),
                date(2026, 4, 10),
                "http://x",
            ),
        ]
        return _FakeCursor(rows=rows, description=description)

    def test_query_uppercases_ticker_and_bounds_window(self):
        cur = self._cursor()
        store = _store_with(_FakeConn(cur))
        store.query("aapl", as_of="2026-04-15", days=180)
        _, params = cur.executed[-1]
        # params: (ticker, earliest, reference)
        assert params[0] == "AAPL"
        assert params[2] == date(2026, 4, 15)  # reference = as_of
        assert params[1] == date.fromordinal(date(2026, 4, 15).toordinal() - 180)  # earliest

    def test_query_returns_normalized_dicts(self):
        store = _store_with(_FakeConn(self._cursor()))
        rows = store.query("AAPL", as_of="2026-04-15")
        assert len(rows) == 1
        assert rows[0]["representative"] == "Jane Smith"
        assert rows[0]["transaction_type"] == "Buy"
        assert rows[0]["lag_days"] == 40


# ─── latest_disclosure_date() (slice 5 incremental window) ────────────────────


class _OneValueCursor(_FakeCursor):
    """Fake cursor whose fetchone() returns a single (value,) row."""

    def __init__(self, value):
        super().__init__()
        self._value = value

    def fetchone(self):
        return (self._value,)


@pytest.mark.unit
class TestLatestDisclosureDate:
    def test_returns_max_date_when_present(self):
        store = _store_with(_FakeConn(_OneValueCursor(date(2026, 4, 10))))
        assert store.latest_disclosure_date() == date(2026, 4, 10)

    def test_parses_string_date(self):
        store = _store_with(_FakeConn(_OneValueCursor("2026-04-10")))
        assert store.latest_disclosure_date() == date(2026, 4, 10)

    def test_empty_table_returns_none(self):
        store = _store_with(_FakeConn(_OneValueCursor(None)))
        assert store.latest_disclosure_date() is None


# ─── upsert() ─────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestUpsert:
    def test_empty_rows_is_noop(self):
        cur = _FakeCursor()
        store = _store_with(_FakeConn(cur))
        assert store.upsert([]) == 0
        assert cur.executed == []

    def test_upsert_uses_on_conflict_dedup_key(self):
        cur = _FakeCursor()
        conn = _FakeConn(cur)
        store = _store_with(conn)
        n = store.upsert(
            [
                {
                    "source_filing_id": "F1",
                    "row_index": 0,
                    "member_name": "X",
                    "ticker": "AAPL",
                    "disclosure_date": date(2026, 4, 10),
                },
                {
                    "source_filing_id": "F1",
                    "row_index": 1,
                    "member_name": "Y",
                    "ticker": "AAPL",
                    "disclosure_date": date(2026, 4, 10),
                },
            ]
        )
        assert n == 2
        assert len(cur.executed) == 2
        sql = cur.executed[0][0]
        assert "ON CONFLICT (source_filing_id, row_index) DO UPDATE" in sql
        # The dedup-key columns must not appear in the SET clause.
        set_part = sql.split("DO UPDATE SET", 1)[1]
        assert "source_filing_id = EXCLUDED" not in set_part
        assert "row_index = EXCLUDED" not in set_part
        assert conn.commits == 1


# ─── fetch_congressional_trades wrapper ───────────────────────────────────────


@pytest.mark.unit
class TestFetchWrapper:
    def test_connection_failure_raises_congress_error(self, monkeypatch):
        def boom():
            raise OSError("connection refused")

        monkeypatch.setattr("tradingagents.dataflows.congress_trades_store._open_conn", boom)
        with pytest.raises(CongressTradesError):
            fetch_congressional_trades("AAPL", as_of="2026-04-15")


# ─── Real round-trip / idempotency (skips without Postgres) ───────────────────


@pytest.mark.integration
class TestRealRoundTrip:
    @pytest.fixture
    def store(self):
        s = CongressTradesStore()
        try:
            s._ensure_conn()
        except CongressTradesError:
            pytest.skip("Postgres congress_trades store not reachable")
        yield s
        # Clean up fixtures for this test's filing id.
        with s._conn.cursor() as cur:
            cur.execute("DELETE FROM congress_trades WHERE source_filing_id = %s", ("TEST_RT",))
        s._conn.commit()
        s.close()

    def _fixture_rows(self):
        return [
            {
                "source_filing_id": "TEST_RT",
                "row_index": 0,
                "chamber": "Senate",
                "member_name": "Test Member",
                "party": "D",
                "owner_type": "self",
                "committee": "Banking",
                "ticker": "ZZZT",
                "asset_name": "Test Co",
                "transaction_type": "Purchase",
                "amount_range": "$1K-$15K",
                "transaction_date": date(2026, 3, 1),
                "disclosure_date": date(2026, 4, 10),
                "source_url": "http://example.test",
            }
        ]

    def test_upsert_is_idempotent(self, store):
        store.upsert(self._fixture_rows())
        store.upsert(self._fixture_rows())  # second time must not duplicate
        rows = store.query("ZZZT", as_of="2026-04-15", days=180)
        assert len(rows) == 1
        assert rows[0]["representative"] == "Test Member"
        assert rows[0]["lag_days"] == 40

    def test_out_of_window_rows_excluded(self, store):
        store.upsert(self._fixture_rows())
        # Window ends well before the disclosure date -> no rows.
        rows = store.query("ZZZT", as_of="2026-01-01", days=30)
        assert rows == []
