"""Tests for the optional B2 ``reasoning`` tier (debate-node offload).

The reasoning tier is opt-in: the bull/bear researchers and the three risk
debators build from it, but when it is unset every reasoning node falls back to
the quick tier so the graph is unchanged (regression-safe). These tests cover
the config env overrides, the defaults, and the GraphSetup node wiring.
"""

from __future__ import annotations

import importlib

from langgraph.prebuilt import ToolNode

import tradingagents.default_config as default_config_module
from tradingagents.graph import setup as graph_setup_mod
from tradingagents.graph.conditional_logic import ConditionalLogic
from tradingagents.graph.setup import GraphSetup


# --- config: defaults + env overrides ---------------------------------------


def _reload_with_env(monkeypatch, **overrides):
    for key in list(default_config_module._ENV_OVERRIDES):
        monkeypatch.delenv(key, raising=False)
    for key, val in overrides.items():
        monkeypatch.setenv(key, val)
    return importlib.reload(default_config_module)


def test_reasoning_tier_defaults_are_unset(monkeypatch):
    dc = _reload_with_env(monkeypatch)
    # Unset by default → callers fall back to the quick tier.
    assert dc.DEFAULT_CONFIG["reasoning_llm_provider"] is None
    assert dc.DEFAULT_CONFIG["reasoning_think_llm"] is None
    assert dc.DEFAULT_CONFIG["reasoning_backend_url"] is None
    assert dc.DEFAULT_CONFIG["reasoning_provider_kwargs"] == {}


def test_reasoning_tier_env_overrides(monkeypatch):
    dc = _reload_with_env(
        monkeypatch,
        TRADINGAGENTS_REASONING_LLM_PROVIDER="openai_compatible",
        TRADINGAGENTS_REASONING_THINK_LLM="gemini-3.5-flash",
        TRADINGAGENTS_REASONING_LLM_BACKEND_URL="http://localhost:11500/v1",
    )
    assert dc.DEFAULT_CONFIG["reasoning_llm_provider"] == "openai_compatible"
    assert dc.DEFAULT_CONFIG["reasoning_think_llm"] == "gemini-3.5-flash"
    assert dc.DEFAULT_CONFIG["reasoning_backend_url"] == "http://localhost:11500/v1"
    # Reasoning env vars are per-tier: they must not touch quick/deep.
    assert dc.DEFAULT_CONFIG["quick_llm_provider"] == "openai"
    assert dc.DEFAULT_CONFIG["deep_llm_provider"] == "openai"


def test_reasoning_env_empty_is_passthrough(monkeypatch):
    dc = _reload_with_env(monkeypatch, TRADINGAGENTS_REASONING_LLM_PROVIDER="")
    assert dc.DEFAULT_CONFIG["reasoning_llm_provider"] is None


# --- GraphSetup wiring ------------------------------------------------------


def _capture_setup(monkeypatch):
    """Patch the agent factories in graph_setup_mod to record the LLM they get.

    Returns a dict mapping each node label to the LLM object passed to its
    factory, populated once ``setup_graph`` runs.
    """
    seen: dict[str, object] = {}

    def _factory(label):
        def _make(llm, *args, **kwargs):
            seen[label] = llm

            def _node(state):  # a valid no-op graph node
                return state

            return _node

        return _make

    factories = {
        "create_market_analyst": "market",
        "create_sentiment_analyst": "social",
        "create_news_analyst": "news",
        "create_fundamentals_analyst": "fundamentals",
        "create_transcript_analyst": "transcript",
        "create_congressional_trades_analyst": "congress",
        "create_bull_researcher": "Bull Researcher",
        "create_bear_researcher": "Bear Researcher",
        "create_research_manager": "Research Manager",
        "create_trader": "Trader",
        "create_aggressive_debator": "Aggressive Analyst",
        "create_neutral_debator": "Neutral Analyst",
        "create_conservative_debator": "Conservative Analyst",
        "create_portfolio_manager": "Portfolio Manager",
    }
    for name, label in factories.items():
        monkeypatch.setattr(graph_setup_mod, name, _factory(label))
    # msg-delete node takes no llm; keep it a valid no-op node.
    monkeypatch.setattr(graph_setup_mod, "create_msg_delete", lambda: (lambda state: state))
    return seen


def _build(monkeypatch, *, reasoning_thinking_llm):
    seen = _capture_setup(monkeypatch)
    quick, deep = object(), object()
    tool_nodes = {"market": ToolNode([])}
    gs = GraphSetup(
        quick_thinking_llm=quick,
        deep_thinking_llm=deep,
        tool_nodes=tool_nodes,
        conditional_logic=ConditionalLogic(),
        reasoning_thinking_llm=reasoning_thinking_llm,
    )
    gs.setup_graph(selected_analysts=("market",))
    return seen, quick, deep


def test_debate_nodes_use_reasoning_tier_when_set(monkeypatch):
    reasoning = object()
    seen, quick, deep = _build(monkeypatch, reasoning_thinking_llm=reasoning)

    # Bull/bear researchers + the three risk debators run on the reasoning tier.
    for label in (
        "Bull Researcher",
        "Bear Researcher",
        "Aggressive Analyst",
        "Neutral Analyst",
        "Conservative Analyst",
    ):
        assert seen[label] is reasoning, label

    # Managers stay deep; trader + analysts stay quick (additive, not a move).
    assert seen["Research Manager"] is deep
    assert seen["Portfolio Manager"] is deep
    assert seen["Trader"] is quick
    assert seen["market"] is quick


def test_debate_nodes_fall_back_to_quick_when_unset(monkeypatch):
    seen, quick, deep = _build(monkeypatch, reasoning_thinking_llm=None)

    # With the reasoning tier unset, every debate node is the quick LLM —
    # identical to the pre-B2 graph.
    for label in (
        "Bull Researcher",
        "Bear Researcher",
        "Aggressive Analyst",
        "Neutral Analyst",
        "Conservative Analyst",
        "Trader",
        "market",
    ):
        assert seen[label] is quick, label
    assert seen["Research Manager"] is deep
    assert seen["Portfolio Manager"] is deep
