---
name: sync-upstream
description: Pulls latest upstream changes from TauricResearch/TradingAgents into personal fork JamesgWall/TradingAgents, resolves merge conflicts while preserving all fork customizations, validates with pytest/ruff, and pushes to origin/main. Use when updating or syncing this fork with upstream TradingAgents.
compatibility: Requires git, uv, and configured remotes (origin -> jamesgwall/TradingAgents, upstream -> TauricResearch/TradingAgents).
metadata:
  author: jamesgwall
  version: "1.0"
---

# Sync Upstream Skill

This skill guides the automated synchronization of the upstream [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents) repository into the personal fork [jamesgwall/TradingAgents](https://github.com/jamesgwall/TradingAgents), resolving any merge conflicts while strictly preserving all custom fork functionality.

---

## 1. Remote Structure & Safety Rules

- **`origin`**: `https://github.com/jamesgwall/TradingAgents.git` (Personal fork — **write / push target**)
- **`upstream`**: `https://github.com/TauricResearch/TradingAgents.git` (Upstream repository — **read / fetch only, NEVER push**)

> [!CRITICAL]
> Never push branches, commits, or tags to `upstream`. All pushes and PRs must target `origin`.

---

## 2. Fork Customizations to Preserve

The personal fork contains several key customizations that must **always** be retained during merge conflict resolutions:

1. **Per-Tier LLM Configuration & Multi-Tier Routing**:
   - `quick_llm_provider`, `quick_think_llm`, `quick_backend_url`, `quick_provider_kwargs`
   - `deep_llm_provider`, `deep_think_llm`, `deep_backend_url`, `deep_provider_kwargs`
   - `reasoning_llm_provider`, `reasoning_think_llm`, `reasoning_backend_url`, `reasoning_provider_kwargs` (debate node offload)
   - `analyst_concurrency_limit` configuration.

2. **Custom Analysts & Nodes**:
   - **Transcript Analyst**: PostgreSQL / pgvector transcript store & retrieval node (`tradingagents/agents/analysts/transcript_analyst.py`, `tradingagents/dataflows/transcript_store.py`).
   - **Congressional Trades Analyst**: House and Senate e-filed PTR disclosures, committee enrichments, and store (`tradingagents/agents/analysts/congressional_trades_analyst.py`, `tradingagents/dataflows/congress_*`).

3. **Google LLM Capabilities & Model Catalog**:
   - Gemini 3.8 Flash, Gemini 3.7 Flash, Gemini 3.6 Flash, Gemini 3.5 Flash options in catalog and capabilities table.
   - Text-only wrapper capability row (`gemini-3.8-flash`, `gemini-3.8-flash-high`, `gemini-3.7-flash`, `gemini-3.7-flash-high`, `gemini-3.6-flash`, etc. with `preferred_structured_method="none"`).

4. **Dataflow Resilience & Hardening**:
   - Senate 503 retry logic.
   - Reddit 429 cache & backoff cooldown.
   - House PDF/HTML timeout & cap.
   - ETF fundamentals detection & handling.
   - FRED macro dataflow exception catching and graceful degradation.

5. **Tooling & Environment**:
   - Python 3.14 support & `uv.lock` compatibility.

See [Fork Customizations Reference](references/fork-customizations.md) for detailed code locations.

---

## 3. Step-by-Step Sync Workflow

### Step 1: Pre-flight Verification
Ensure the working tree is clean and remotes are properly configured:
```bash
git status
git remote -v
```
If `upstream` remote is missing, add it:
```bash
git remote add upstream https://github.com/TauricResearch/TradingAgents.git
```

### Step 2: Fetch Remotes & Inspect Changes
```bash
git fetch origin
git fetch upstream
git log --oneline main..upstream/main
```

### Step 3: Create a Dedicated Sync Branch
Create a sync branch named after the upstream release or date:
```bash
git checkout -b sync/upstream-<version-or-date> main
```
*Example:* `git checkout -b sync/upstream-v0.4.0 main`

### Step 4: Merge Upstream Main
```bash
git merge upstream/main
```

### Step 5: Resolve Merge Conflicts
When conflicts occur, inspect each unmerged file with `git status` and resolve them following these guidelines:

- **`tradingagents/default_config.py` & `tests/test_env_overrides.py`**:
  - Keep fork's per-tier `quick_*`, `deep_*`, `reasoning_*` settings in `_ENV_OVERRIDES` and `DEFAULT_CONFIG`.
  - Incorporate upstream additions (e.g. `max_tokens`, `google_thinking_level`, `openai_reasoning_effort`, `anthropic_effort`).
  - Update default model names to upstream's latest recommendations (e.g. `gpt-5.6` / `gpt-5.6-luna`).

- **`tradingagents/graph/trading_graph.py`**:
  - Keep `quick_thinking_llm`, `deep_thinking_llm`, `reasoning_thinking_llm`, `analyst_concurrency_limit`.
  - Keep tool node entries for `"transcript"` and `"congress"`.
  - Incorporate upstream fixes (e.g. `resolution_date` tracking, checkpoint resumption fixes, `save_reports`, report tree export).
  - Ensure `_get_provider_kwargs(self, tier: str = "quick")` has a default tier parameter.

- **`tradingagents/dataflows/fred.py`**:
  - Keep `try ... except FredNotConfiguredError: raise / except Exception: return ...` error handling.
  - Incorporate upstream vintage point-in-time pinning (`realtime_start` / `realtime_end`).

- **`tradingagents/llm_clients/capabilities.py` & `tests/test_capabilities.py`**:
  - Keep `_WRAPPER_TEXT_ONLY` capability entries for `gemini-3.8-flash`, `gemini-3.7-flash`, `gemini-3.6-flash`, `gemini-3.5-flash`.
  - Keep `TestWrapperTextOnly` tests.
  - Incorporate upstream namespace handling (e.g. OpenRouter `deepseek/` stripping) and `TestOpenRouterDeepSeekNamespace`.

- **`tradingagents/llm_clients/model_catalog.py` & `google_client.py` & `openai_client.py`**:
  - Keep Gemini 3.8 / 3.7 / 3.6 catalog additions and capabilities.
  - Incorporate upstream `max_tokens` / `max_output_tokens` passthrough and updated model listings.

- **`tradingagents/agents/schemas.py`**:
  - Ensure upstream nullish float coercion validators (`@field_validator(..., mode="before")`) are intact.

### Step 6: Validate & Format
Run the linting, formatting, and full test suite to guarantee 100% pass rate:
```bash
uv run --extra dev ruff check .
uv run --extra dev ruff format .
uv run --extra dev pytest
```

### Step 7: Commit the Merge
```bash
git add -u
git commit -m "Merge remote-tracking branch 'upstream/main' into sync/upstream-<version-or-date>

Sync upstream <version> into fork, keeping all personalized customizations:
- Per-tier quick/deep/reasoning LLM provider configurations
- Google model catalog & capabilities (Gemini 3.8 Flash, Gemini 3.7 Flash, Gemini 3.6 Flash)
- Congressional trades & transcript analyst dataflows and nodes
- Dataflow hardening (Senate retry, Reddit cooldown, House timeout, ETF fundamentals, FRED error handling)
- Python 3.14 / uv tooling support"
```

### Step 8: Update Main, Push to Fork & Clean Up
Fast-forward or merge the sync branch into local `main`, push `main` to `origin`, and delete the temporary local sync branch:
```bash
git checkout main
git merge sync/upstream-<version-or-date>
git push origin main
git branch -d sync/upstream-<version-or-date>
```
Verify `git status` is clean on `main`.

