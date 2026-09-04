# Fork Customizations Reference

This document catalogs the personalized modifications in `jamesgwall/TradingAgents` that must be preserved during merges from upstream `TauricResearch/TradingAgents`.

---

## 1. LLM Configuration & Multi-Tier Routing

### Configuration (`tradingagents/default_config.py`)
- `quick_llm_provider`, `quick_think_llm`, `quick_backend_url`, `quick_provider_kwargs`
- `deep_llm_provider`, `deep_think_llm`, `deep_backend_url`, `deep_provider_kwargs`
- `reasoning_llm_provider`, `reasoning_think_llm`, `reasoning_backend_url`, `reasoning_provider_kwargs`
- `analyst_concurrency_limit`
- Environment mapping via `_ENV_OVERRIDES` for `TRADINGAGENTS_LLM_PROVIDER` -> `("quick_llm_provider", "deep_llm_provider")`, etc.

### Graph Routing & Execution (`tradingagents/graph/trading_graph.py`, `setup.py`)
- Independent LLM instantiation for quick, deep, and reasoning tiers.
- `_get_provider_kwargs(self, tier: str = "quick")` helper.
- `_build_reasoning_llm()` creating debate-node LLM instance fallback to quick LLM when unset.
- Debate nodes (`bull_researcher`, `bear_researcher`, `risk_debator` nodes) wired to `reasoning_thinking_llm`.

---

## 2. Custom Analysts & Tools

### Transcript Analyst
- **Agent**: `tradingagents/agents/analysts/transcript_analyst.py`
- **Dataflow / Store**: `tradingagents/dataflows/transcript_store.py` (PostgreSQL / `pgvector` store)
- **Node Wiring**: Included in `ANALYST_NODE_SPECS` and `_create_tool_nodes()` in `trading_graph.py`.

### Congressional Trades Analyst
- **Agent**: `tradingagents/agents/analysts/congressional_trades_analyst.py`
- **Dataflows**:
  - `tradingagents/dataflows/congress_house_efd.py` (House PTR e-filed reports)
  - `tradingagents/dataflows/congress_senate_efd.py` (Senate PTR e-filed reports)
  - `tradingagents/dataflows/congress_committees.py` (Legislator committee enrichments)
  - `tradingagents/dataflows/congress_trades_store.py` (Local store / caching)
- **Node Wiring**: Included in `ANALYST_NODE_SPECS` and `_create_tool_nodes()` in `trading_graph.py`.

---

## 3. Model Catalog & Capabilities

### Capabilities Table (`tradingagents/llm_clients/capabilities.py`)
- Text-only wrapper models:
  ```python
  "gemini-3.8-flash": _WRAPPER_TEXT_ONLY,
  "gemini-3.8-flash-high": _WRAPPER_TEXT_ONLY,
  "gemini-3.7-flash": _WRAPPER_TEXT_ONLY,
  "gemini-3.7-flash-high": _WRAPPER_TEXT_ONLY,
  "gemini-3.6-flash": _WRAPPER_TEXT_ONLY,
  "gemini-3.6-flash-high": _WRAPPER_TEXT_ONLY,
  "gemini-3.5-flash": _WRAPPER_TEXT_ONLY,
  "gemini-3.5-flash-high": _WRAPPER_TEXT_ONLY,
  ```

### Model Catalog (`tradingagents/llm_clients/model_catalog.py`)
- Google options including Gemini 3.8 Flash, Gemini 3.7 Flash, and Gemini 3.6 Flash across quick and deep tiers.

---

## 4. Dataflow Hardening & Resilience

- **Senate EFD**: Retry on 503 Service Unavailable with backoff.
- **Reddit**: 429 rate limit cooldown cache.
- **House EFD**: Request timeout and page limit guards.
- **ETF Fundamentals**: ETF detection (`is_etf`) to bypass stock-only fundamental ratios without error.
- **FRED**: Catch exceptions in macro series requests and return informative error strings rather than aborting runs.
