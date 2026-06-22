"""
Transcript store: read-only query interface wrapping pgvector.

Used exclusively by the TranscriptAnalyst. Encapsulates connection management,
cosine similarity search, relevance window filtering, and result deduplication.

Requires psycopg2-binary and pgvector packages:
    pip install psycopg2-binary pgvector

Env vars (all optional — defaults shown):
    PGVECTOR_HOST     localhost
    PGVECTOR_PORT     5432
    PGVECTOR_DB       transcripts
    PGVECTOR_USER     postgres
    PGVECTOR_PASSWORD postgres
    OLLAMA_BASE_URL   http://localhost:11434
"""

from __future__ import annotations

import json
import os
import urllib.request

EMBED_MODEL = "nomic-embed-text"
MACRO_WINDOW_DAYS = 30
DEFAULT_TOP_K = 8


def _embed_text(text: str, ollama_url: str) -> list[float]:
    base = ollama_url.rstrip("/")
    payload = json.dumps({"model": EMBED_MODEL, "input": [text]}).encode()
    req = urllib.request.Request(
        f"{base}/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    if "embeddings" not in data:
        raise RuntimeError(f"Ollama embed response missing 'embeddings': {data}")
    return data["embeddings"][0]


def _open_conn():
    try:
        import psycopg2
    except ImportError as err:
        raise ImportError(
            "psycopg2-binary is required for the transcript store. Run: pip install psycopg2-binary"
        ) from err
    return psycopg2.connect(
        host=os.environ.get("PGVECTOR_HOST", "localhost"),
        port=int(os.environ.get("PGVECTOR_PORT", "5432")),
        dbname=os.environ.get("PGVECTOR_DB", "transcripts"),
        user=os.environ.get("PGVECTOR_USER", "postgres"),
        password=os.environ.get("PGVECTOR_PASSWORD", "postgres"),
    )


class TranscriptStore:
    """
    Query interface to the pgvector transcript database.

    Instantiate once per analyst invocation; call close() when done.
    The connection is opened lazily on first query.
    """

    def __init__(self, ollama_url: str | None = None):
        self._ollama_url: str = ollama_url or os.environ.get(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        self._conn = None

    def _ensure_conn(self) -> None:
        if self._conn is not None and not self._conn.closed:
            return
        self._conn = _open_conn()
        try:
            from pgvector.psycopg2 import register_vector

            register_vector(self._conn)
        except ImportError as err:
            raise ImportError(
                "pgvector Python package is required. Run: pip install pgvector"
            ) from err

    def close(self) -> None:
        if self._conn and not self._conn.closed:
            self._conn.close()

    def ticker_query(
        self,
        ticker: str,
        query_text: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> list[dict]:
        """
        Semantic search filtered by ticker_tags and per-chunk relevance window.

        Returns up to top_k chunks ranked by cosine similarity where:
          - ticker is in the chunk's ticker_tags array
          - published_at is within the chunk's own relevance_days window
        """
        self._ensure_conn()
        embedding = _embed_text(query_text, self._ollama_url)
        import numpy as np

        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT
                video_id, channel_name, video_title, published_at,
                content_type, chunk_index, chunk_text, ticker_tags,
                1 - (embedding <=> %s::vector) AS similarity
            FROM transcript_chunks
            WHERE
                %s = ANY(ticker_tags)
                AND published_at > NOW() - (relevance_days * INTERVAL '1 day')
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (np.array(embedding), ticker, np.array(embedding), top_k),
        )
        return self._to_dicts(cur.fetchall())

    def macro_query(
        self,
        query_text: str,
        top_k: int = DEFAULT_TOP_K,
        window_days: int = MACRO_WINDOW_DAYS,
    ) -> list[dict]:
        """
        Broad semantic search with no ticker filter over a short recent window.

        Surfaces macro commentary (Fed policy, rates, inflation) that may not
        mention any specific ticker symbol.
        """
        self._ensure_conn()
        embedding = _embed_text(query_text, self._ollama_url)
        import numpy as np

        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT
                video_id, channel_name, video_title, published_at,
                content_type, chunk_index, chunk_text, ticker_tags,
                1 - (embedding <=> %s::vector) AS similarity
            FROM transcript_chunks
            WHERE published_at > NOW() - (%s * INTERVAL '1 day')
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (np.array(embedding), window_days, np.array(embedding), top_k),
        )
        return self._to_dicts(cur.fetchall())

    def merge_and_deduplicate(
        self,
        ticker_results: list[dict],
        macro_results: list[dict],
        top_k: int = DEFAULT_TOP_K,
    ) -> list[dict]:
        """
        Merge ticker-specific and macro results, deduplicate by (video_id, chunk_index),
        and return up to top_k chunks sorted by similarity descending.
        """
        seen: set[tuple] = set()
        merged: list[dict] = []
        for chunk in ticker_results + macro_results:
            key = (chunk["video_id"], chunk["chunk_index"])
            if key not in seen:
                seen.add(key)
                merged.append(chunk)
        return sorted(merged, key=lambda c: c["similarity"], reverse=True)[:top_k]

    @staticmethod
    def _to_dicts(rows) -> list[dict]:
        keys = (
            "video_id",
            "channel_name",
            "video_title",
            "published_at",
            "content_type",
            "chunk_index",
            "chunk_text",
            "ticker_tags",
            "similarity",
        )
        return [dict(zip(keys, row, strict=True)) for row in rows]
