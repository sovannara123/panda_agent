import logging

import pytest

from agent import Agent
from config import Config
from logger import log
from storage import MemoryStorage
from tools import validate_tool_call, plan_tool_call, execute_tool


@pytest.fixture 
def agent(tmp_path):
    """Fresh agent backed by temp storage for each test."""
    return Agent(storage=MemoryStorage(
        str(tmp_path / "memory.json"),
        str(tmp_path / "meta.json")
    ))


class TestToolCalling:
    def test_valid_price_call(self):
        call = plan_tool_call("What is the price of laptop?")
        assert call is not None
        assert validate_tool_call(call)

    def test_invalid_missing_arg(self):
        assert not validate_tool_call({"tool": "get_product_price", "arguments": {}})

    def test_unknown_tool_rejected(self):
        assert not validate_tool_call({"tool": "nonexistent", "arguments": {}})

    def test_case_insensitive_order(self):
        call = plan_tool_call("track order a100")
        assert call["arguments"]["order_id"] == "A100"


class TestAgentResponses:
    def test_price_response_format(self, agent):
        response = agent.respond("price of phone")
        assert "$699" in response

    def test_usage_limit_blocks(self, agent):
        agent.metadata["messages_used"] = Config.MESSAGE_LIMIT_FREE
        response = agent.respond("hello")
        assert "limit" in response.lower()
        assert agent.metadata["messages_used"] == Config.MESSAGE_LIMIT_FREE


class TestUsageLimit:
    def test_at_limit_blocks(self, agent):
        agent.metadata["messages_used"] = Config.MESSAGE_LIMIT_FREE
        response = agent.respond("hello")
        assert "limit" in response.lower()
        assert agent.metadata["messages_used"] == Config.MESSAGE_LIMIT_FREE

    def test_above_limit_blocks(self, agent):
        agent.metadata["messages_used"] = Config.MESSAGE_LIMIT_FREE + 1
        response = agent.respond("hello")
        assert "limit" in response.lower()
        assert agent.metadata["messages_used"] == Config.MESSAGE_LIMIT_FREE + 1

    def test_below_limit_allowed_and_increments(self, agent):
        agent.metadata["messages_used"] = Config.MESSAGE_LIMIT_FREE - 1
        response = agent.respond("hello")
        assert "limit" not in response.lower()
        assert agent.metadata["messages_used"] == Config.MESSAGE_LIMIT_FREE

    def test_custom_limit_from_config(self, agent, monkeypatch):
        monkeypatch.setattr(Config, "MESSAGE_LIMIT_FREE", 3)
        agent.metadata["messages_used"] = 2
        response = agent.respond("hello")
        assert "limit" not in response.lower()
        assert agent.metadata["messages_used"] == 3

        response = agent.respond("hello")
        assert "limit" in response.lower()
        assert agent.metadata["messages_used"] == 3

    def test_blocked_turn_not_stored(self, agent):
        agent.metadata["messages_used"] = Config.MESSAGE_LIMIT_FREE
        agent.respond("hello")
        assert agent.storage.load_messages() == []

    def test_premium_unlimited(self, agent):
        agent.metadata["user_plan"] = "premium"
        agent.metadata["messages_used"] = 10**5
        response = agent.respond("hello")
        assert "limit" not in response.lower()
        assert agent.metadata["messages_used"] == 10**5 + 1

    def test_unknown_plan_defaults_to_free(self, agent):
        agent.metadata["user_plan"] = "enterprise"
        agent.metadata["messages_used"] = Config.MESSAGE_LIMIT_FREE
        response = agent.respond("hello")
        assert "limit" in response.lower()


class TestMemory:
    def test_memory_persists(self, agent):
        agent.respond("test message")
        reloaded = agent.storage.load_messages()
        assert any(m["content"] == "test message" for m in reloaded)


class TestFallbackLogging:
    def _fallback_messages(self, caplog):
        return list(dict.fromkeys(
            r.getMessage()
            for r in caplog.records
            if '"event": "fallback"' in r.getMessage()
        ))

    def test_fallback_logged_on_usage_limit(self, agent, caplog, monkeypatch):
        monkeypatch.setattr(log, "propagate", True)
        caplog.set_level(logging.INFO, logger="agent")
        agent.metadata["messages_used"] = Config.MESSAGE_LIMIT_FREE
        agent.respond("hello")
        messages = self._fallback_messages(caplog)
        assert len(messages) == 1
        assert "usage_limit" in messages[0]

    def test_fallback_logged_on_llm_error(self, agent, caplog, monkeypatch):
        monkeypatch.setattr(log, "propagate", True)
        caplog.set_level(logging.INFO, logger="agent")
        response = agent.respond("crash")
        assert "unavailable" in response.lower()
        messages = self._fallback_messages(caplog)
        assert len(messages) == 1
        assert "llm_error" in messages[0]

    def test_fallback_logged_on_tool_failure(self, agent, caplog, monkeypatch):
        monkeypatch.setattr(log, "propagate", True)
        caplog.set_level(logging.INFO, logger="agent")
        response = agent.respond("simulate failure")
        assert "action failed" in response.lower()
        messages = self._fallback_messages(caplog)
        assert len(messages) == 1
        assert "tool_failed" in messages[0]


class TestFailureTool:
    def test_plan_simulate_failure(self):
        call = plan_tool_call("simulate failure now")
        assert call is not None
        assert call["tool"] == "test_failure"
        assert validate_tool_call(call)

    def test_plan_test_failure_keyword(self):
        call = plan_tool_call("test failure please")
        assert call is not None
        assert call["tool"] == "test_failure"

    def test_test_failure_tool_always_fails(self):
        call = plan_tool_call("test failure")
        result = execute_tool(call)
        assert result["status"] == "error"
        assert "degradation" in result["message"]

    def test_agent_graceful_fallback(self, agent):
        response = agent.respond("simulate failure")
        assert "action failed" in response.lower()
        assert "degradation" in response.lower()