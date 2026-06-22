"""Tests for env-driven CLI behavior (#897, #873).

The config-layer override (TRADINGAGENTS_* -> DEFAULT_CONFIG) is covered by
test_env_overrides.py. These tests cover the CLI helper layer.

NOTE: This fork replaced the single-provider CLI flow with a per-tier
selection (``select_thinking_tier`` for quick + deep independently), which
dropped upstream's "skip interactive LLM prompts when configured via env"
shortcut (the old ``TestCliSkipsPromptsFromEnv``). The interactive CLI now
always prompts per tier; env vars still configure the programmatic
``DEFAULT_CONFIG`` used by the Python API (see test_env_overrides.py). The
prompt-skip test was removed rather than adapted because the behaviour it
asserted no longer exists.
"""

import os
import unittest
from unittest import mock

import pytest


@pytest.mark.unit
class TestProviderDefaultUrl(unittest.TestCase):
    def test_known_providers_resolve(self):
        from cli.utils import provider_default_url

        self.assertEqual(provider_default_url("openai"), "https://api.openai.com/v1")
        self.assertEqual(provider_default_url("DeepSeek"), "https://api.deepseek.com")
        self.assertIsNone(provider_default_url("google"))  # uses SDK default

    def test_unknown_provider_returns_none(self):
        from cli.utils import provider_default_url

        self.assertIsNone(provider_default_url("not-a-provider"))

    def test_ollama_honors_base_url_env(self):
        from cli.utils import provider_default_url

        with mock.patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://host:1234/v1"}):
            self.assertEqual(provider_default_url("ollama"), "http://host:1234/v1")


if __name__ == "__main__":
    unittest.main()
