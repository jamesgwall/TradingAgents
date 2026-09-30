from collections.abc import Iterable
from dataclasses import dataclass

from tradingagents.agents.analysts import fundamentals_analyst, market_analyst, news_analyst


@dataclass(frozen=True)
class AnalystNodeSpec:
    key: str
    agent_node: str
    report_key: str
    tools: tuple = ()


@dataclass(frozen=True)
class AnalystExecutionPlan:
    specs: list[AnalystNodeSpec]
    concurrency_limit: int


ANALYST_NODE_SPECS: dict[str, AnalystNodeSpec] = {
    "market": AnalystNodeSpec(
        key="market",
        agent_node="Market Analyst",
        report_key="market_report",
        tools=market_analyst.TOOLS,
    ),
    "social": AnalystNodeSpec(
        # Saved configs select this analyst as "social". It fetches its
        # sources before calling the model, so it has no tools.
        key="social",
        agent_node="Sentiment Analyst",
        report_key="sentiment_report",
    ),
    "news": AnalystNodeSpec(
        key="news",
        agent_node="News Analyst",
        report_key="news_report",
        tools=news_analyst.TOOLS,
    ),
    "fundamentals": AnalystNodeSpec(
        key="fundamentals",
        agent_node="Fundamentals Analyst",
        report_key="fundamentals_report",
        tools=fundamentals_analyst.TOOLS,
    ),
    "transcript": AnalystNodeSpec(
        key="transcript",
        agent_node="Transcript Analyst",
        report_key="transcript_report",
    ),
    "congress": AnalystNodeSpec(
        key="congress",
        agent_node="Congressional Trades Analyst",
        report_key="congressional_trades_report",
    ),
}


def build_analyst_execution_plan(
    selected_analysts: Iterable[str],
    concurrency_limit: int = 1,
) -> AnalystExecutionPlan:
    if concurrency_limit < 1:
        raise ValueError("analyst concurrency limit must be >= 1")

    specs: list[AnalystNodeSpec] = []
    for analyst_key in selected_analysts:
        spec = ANALYST_NODE_SPECS.get(analyst_key)
        if spec is None:
            raise ValueError(f"unknown analyst key: {analyst_key}")
        specs.append(spec)

    if not specs:
        raise ValueError("at least one analyst must be selected")

    return AnalystExecutionPlan(specs=specs, concurrency_limit=concurrency_limit)
