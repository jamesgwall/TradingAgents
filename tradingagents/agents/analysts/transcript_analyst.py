"""
Transcript Analyst — Stage 1 analyst that synthesizes YouTube transcript signal.

Unlike the other analysts, this node does not use LangGraph tool calls.
It queries the TranscriptStore directly, makes two quick_llm calls (macro query
generation + synthesis), and returns a completed report in a single pass.

The should_continue_transcript method in ConditionalLogic always routes to
"Msg Clear Transcript" because this node never emits tool calls.
"""

from __future__ import annotations

import contextlib
import logging

from langchain_core.messages import AIMessage

log = logging.getLogger(__name__)

TRANSCRIPT_STUB_MARKER = "NO_TRANSCRIPT_DATA"


def create_transcript_analyst(llm):
    """Factory for the transcript analyst node.

    Args:
        llm: The quick_thinking_llm instance (used for both query generation and synthesis).
    """

    def transcript_analyst_node(state: dict) -> dict:
        ticker     = state["company_of_interest"]
        trade_date = state["trade_date"]

        try:
            from tradingagents.dataflows.transcript_store import TranscriptStore
        except ImportError as e:
            log.warning(f"TranscriptStore unavailable (missing psycopg2/pgvector?): {e}")
            stub = TRANSCRIPT_STUB_MARKER
            return {"messages": [AIMessage(content=stub)], "transcript_report": stub}

        store = None
        try:
            store = TranscriptStore()
            # Step 1: use quick_llm to generate a targeted macro query
            macro_prompt = (
                f"You are a financial analyst. Write a single concise semantic search query "
                f"that would surface macro-economic commentary (interest rates, inflation, "
                f"Fed policy, sector conditions) most relevant to evaluating {ticker} on "
                f"{trade_date}. Output only the query sentence — nothing else."
            )
            macro_resp = llm.invoke(macro_prompt)
            macro_query = (
                macro_resp.content.strip()
                if hasattr(macro_resp, "content")
                else str(macro_resp).strip()
            )

            # Step 2: dual semantic search
            ticker_query = f"{ticker} earnings revenue guidance outlook valuation"
            ticker_results = store.ticker_query(ticker, ticker_query)
            macro_results  = store.macro_query(macro_query)

            # Step 3: merge and deduplicate
            chunks = store.merge_and_deduplicate(ticker_results, macro_results)

            if not chunks:
                stub = TRANSCRIPT_STUB_MARKER
                return {"messages": [AIMessage(content=stub)], "transcript_report": stub}

            # Step 4: synthesize top-k chunks
            sections: list[str] = []
            for chunk in chunks:
                pub = chunk["published_at"]
                pub_str = pub.strftime("%Y-%m-%d") if hasattr(pub, "strftime") else str(pub)[:10]
                sim_pct = f"{chunk['similarity'] * 100:.0f}%"
                sections.append(
                    f"[{chunk['channel_name']} — {pub_str} — "
                    f"{chunk['content_type']} — sim {sim_pct}]\n{chunk['chunk_text']}"
                )
            context = "\n\n---\n\n".join(sections)

            synthesis_prompt = (
                f"You are a financial research analyst synthesizing transcript excerpts "
                f"from financial YouTube channels for {ticker} as of {trade_date}.\n\n"
                f"{context}\n\n"
                f"Write a concise research note covering:\n"
                f"1. Key themes and sentiment about {ticker} or its sector\n"
                f"2. Relevant macro factors that could impact {ticker}\n"
                f"3. Any specific forward guidance, earnings commentary, or analyst views\n"
                f"4. Overall qualitative signal (bullish / neutral / bearish) with rationale\n\n"
                f"Cite the channel name and date for each observation. "
                f"If the transcripts contain no relevant signal for {ticker}, state that explicitly."
            )
            synth_resp = llm.invoke(synthesis_prompt)
            report = (
                synth_resp.content if hasattr(synth_resp, "content") else str(synth_resp)
            )

        except Exception as e:
            log.warning(f"Transcript analyst unavailable for {ticker} — DB or Ollama unreachable: {e}", exc_info=True)
            report = TRANSCRIPT_STUB_MARKER
        finally:
            if store is not None:
                with contextlib.suppress(Exception):
                    store.close()

        return {"messages": [AIMessage(content=report)], "transcript_report": report}

    return transcript_analyst_node
