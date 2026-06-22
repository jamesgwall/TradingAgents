"""
Congressional Trades Analyst — Stage 1 analyst that synthesizes STOCK Act
congressional trade-disclosure signal from the local congress_trades store.

Like the transcript analyst, this node does not use LangGraph tool calls. It
fetches disclosures directly, computes deterministic stats and a markdown table
in Python (so the numbers are never hallucinated), and makes a single quick_llm
call to write the prose sections (summary, pattern analysis, caveats). The
report is returned in one pass.

The should_continue_congress method in ConditionalLogic always routes to
"Msg Clear Congress" because this node never emits tool calls.
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import date, datetime

from langchain_core.messages import AIMessage

log = logging.getLogger(__name__)

CONGRESS_STUB_MARKER = "NO_CONGRESS_DATA"
SUMMARY_WINDOW_DAYS = 30


def _fmt_date(value) -> str:
    if isinstance(value, (date, datetime)):
        return value.strftime("%Y-%m-%d")
    return str(value) if value else ""


def _build_table(rows: list[dict]) -> str:
    """Render the deterministic notable-trades table from normalized rows."""
    header = (
        "| Politician | Party | Chamber | Committee | Trader | Type | Amount "
        "| Trade Date | Disclosure Date | Lag (days) |\n"
        "|---|---|---|---|---|---|---|---|---|---|"
    )
    lines = [header]
    for r in rows:
        lag = "" if r["lag_days"] is None else str(r["lag_days"])
        lines.append(
            f"| {r['representative']} | {r['party']} | {r['chamber']} "
            f"| {r['committee']} | {r['trader']} | {r['transaction_type']} "
            f"| {r['amount_range']} | {_fmt_date(r['trade_date'])} "
            f"| {_fmt_date(r['disclosure_date'])} | {lag} |"
        )
    return "\n".join(lines)


def _build_stats(rows: list[dict], reference: date) -> dict:
    """Compute deterministic counts for the full window and the last 30 days."""
    cutoff = reference.toordinal() - SUMMARY_WINDOW_DAYS

    def summarize(subset: list[dict]) -> dict:
        types = Counter(r["transaction_type"] for r in subset)
        parties = Counter(r["party"] for r in subset if r["party"])
        return {
            "total": len(subset),
            "buys": types.get("Buy", 0),
            "sells": types.get("Sell", 0),
            "members": len({r["representative"] for r in subset if r["representative"]}),
            "party_breakdown": dict(parties),
        }

    recent = [r for r in rows if r["disclosure_date"].toordinal() >= cutoff]
    committees = sorted({r["committee"] for r in rows if r["committee"]})
    return {
        "window": summarize(rows),
        "recent": summarize(recent),
        "committees": committees,
    }


def create_congressional_trades_analyst(llm, window_days: int = 180):
    """Factory for the congressional trades analyst node.

    Args:
        llm: the quick_thinking_llm instance (used for prose synthesis).
        window_days: disclosure-date lookback window (default 180).
    """

    def congressional_trades_analyst_node(state: dict) -> dict:
        ticker     = state["company_of_interest"]
        trade_date = state["trade_date"]

        try:
            from tradingagents.dataflows.congress_trades_store import (
                CongressTradesError,
                fetch_congressional_trades,
            )
        except ImportError as e:
            log.warning(f"Congress trades store unavailable: {e}")
            return {
                "messages": [AIMessage(content=CONGRESS_STUB_MARKER)],
                "congressional_trades_report": CONGRESS_STUB_MARKER,
            }

        try:
            rows = fetch_congressional_trades(ticker, as_of=trade_date, days=window_days)
        except CongressTradesError as e:
            # Fail open: an unreachable store must not abort the nightly run for
            # this ticker — the debate proceeds without it.
            log.warning(f"Congressional trades unavailable for {ticker}: {e}")
            return {
                "messages": [AIMessage(content=CONGRESS_STUB_MARKER)],
                "congressional_trades_report": CONGRESS_STUB_MARKER,
            }

        if not rows:
            report = (
                f"## Congressional Trades — {ticker}\n\n"
                f"No congressional trade disclosures for {ticker} in the last "
                f"{window_days} days (as of {trade_date})."
            )
            return {
                "messages": [AIMessage(content=report)],
                "congressional_trades_report": report,
            }

        from tradingagents.dataflows.congress_trades_store import _parse_date  # local: date helper
        reference = _parse_date(trade_date) or date.today()
        stats = _build_stats(rows, reference)
        table = _build_table(rows)

        # Compact, factual context for the LLM. It writes prose only; all
        # figures it cites come from these precomputed stats.
        recent, window = stats["recent"], stats["window"]
        committees = ", ".join(stats["committees"]) if stats["committees"] else "none reported"
        facts = (
            f"Ticker: {ticker}\n"
            f"As of: {trade_date}\n"
            f"Last {SUMMARY_WINDOW_DAYS} days — disclosures: {recent['total']} "
            f"({recent['buys']} buys, {recent['sells']} sells) by {recent['members']} member(s); "
            f"party breakdown: {recent['party_breakdown'] or 'n/a'}\n"
            f"Full {window_days}-day window — disclosures: {window['total']} "
            f"({window['buys']} buys, {window['sells']} sells) by {window['members']} member(s); "
            f"party breakdown: {window['party_breakdown'] or 'n/a'}\n"
            f"Committees represented among traders: {committees}\n\n"
            f"Trade detail (newest disclosure first):\n{table}"
        )

        synthesis_prompt = (
            f"You are a financial research analyst reviewing disclosed congressional "
            f"(STOCK Act) trades in {ticker} as of {trade_date}. Using ONLY the data "
            f"below, write three short sections in markdown. Do not invent numbers — "
            f"every figure must come from the data provided.\n\n"
            f"{facts}\n\n"
            f"Write exactly these sections:\n"
            f"### Summary\n"
            f"One paragraph emphasizing the last {SUMMARY_WINDOW_DAYS} days, then the "
            f"full {window_days}-day picture and any relevant committee context.\n"
            f"### Pattern Analysis\n"
            f"Note any clustering by party, direction, timing, or committee relevance "
            f"(e.g. a committee member trading a stock their committee oversees). If no "
            f"clear pattern exists, say so plainly.\n"
            f"### Caveats\n"
            f"Remind the reader that disclosure lag (often 30-180 days) means the market "
            f"may have already reacted, and that correlation is not evidence of inside "
            f"knowledge or illegal trading."
        )

        try:
            resp = llm.invoke(synthesis_prompt)
            prose = resp.content if hasattr(resp, "content") else str(resp)
        except Exception as e:
            log.warning(f"Congressional trades synthesis failed for {ticker}: {e}", exc_info=True)
            prose = (
                "### Summary\n"
                f"{window['total']} congressional disclosure(s) for {ticker} in the last "
                f"{window_days} days ({window['buys']} buys, {window['sells']} sells). "
                "LLM synthesis unavailable; see the table below.\n"
                "### Caveats\nDisclosure lag means the market may have already reacted."
            )

        report = (
            f"## Congressional Trades — {ticker}\n\n"
            f"{prose}\n\n"
            f"### Notable Trades ({window_days}-day window)\n\n"
            f"{table}"
        )
        return {
            "messages": [AIMessage(content=report)],
            "congressional_trades_report": report,
        }

    return congressional_trades_analyst_node
