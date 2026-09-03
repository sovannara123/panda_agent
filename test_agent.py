import json
import logging

import pytest

from agent import Agent
from config import get_config
from context import RequestContext
from llm_adapters import LLMError, MockLLMClient, FlakyMockLLMClient
from logger import log
from retry import retry_with_backoff, RetryError
from storage import MemoryStorage
from tools import validate_tool_call, plan_tool_call, execute_tool


@pytest.fixture 
def agent(tmp_path):
    """Fresh agent backed by temp storage for each test."""
    agent = Agent(storage=MemoryStorage(
        str(tmp_path / "memory.json"),
        str(tmp_path / "meta.json")
    ))
    agent.llm = MockLLMClient()
    agent.metadata["user_plan"] = "free"  # Default to free for tests
    return agent


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
        assert call is not None
        assert call["arguments"]["order_id"] == "A100"


class TestAgentResponses:
    def test_price_response_format(self, agent):
        response = agent.respond("price of phone")
        assert "$699" in response

    def test_usage_limit_blocks(self, agent):
        agent.metadata["messages_used"] = get_config().MESSAGE_LIMIT_FREE
        response = agent.respond("hello")
        assert "limit" in response.lower()
        assert agent.metadata["messages_used"] == get_config().MESSAGE_LIMIT_FREE


class TestUsageLimit:
    def test_at_limit_blocks(self, agent):
        agent.metadata["messages_used"] = get_config().MESSAGE_LIMIT_FREE
        response = agent.respond("hello")
        assert "limit" in response.lower()
        assert agent.metadata["messages_used"] == get_config().MESSAGE_LIMIT_FREE

    def test_above_limit_blocks(self, agent):
        agent.metadata["messages_used"] = get_config().MESSAGE_LIMIT_FREE + 1
        response = agent.respond("hello")
        assert "limit" in response.lower()
        assert agent.metadata["messages_used"] == get_config().MESSAGE_LIMIT_FREE + 1

    def test_below_limit_allowed_and_increments(self, agent):
        agent.metadata["messages_used"] = get_config().MESSAGE_LIMIT_FREE - 1
        response = agent.respond("hello")
        assert "limit" not in response.lower()
        assert agent.metadata["messages_used"] == get_config().MESSAGE_LIMIT_FREE

    def test_custom_limit_from_config(self, agent, monkeypatch):
        from config import reload_config
        monkeypatch.setenv("MESSAGE_LIMIT_FREE", "3")
        reload_config()
        agent.metadata["messages_used"] = 2
        response = agent.respond("hello")
        assert "limit" not in response.lower()
        assert agent.metadata["messages_used"] == 3

        response = agent.respond("hello")
        assert "limit" in response.lower()
        assert agent.metadata["messages_used"] == 3

    def test_blocked_turn_not_counted(self, agent):
        agent.metadata["messages_used"] = get_config().MESSAGE_LIMIT_FREE
        response = agent.respond("hello")
        assert "limit" in response.lower()
        assert agent.metadata["messages_used"] == get_config().MESSAGE_LIMIT_FREE
        reloaded = agent.storage.load_messages()
        assert len(reloaded) == 1
        assert reloaded[0]["role"] == "assistant"

    def test_premium_unlimited(self, agent):
        agent.metadata["user_plan"] = "premium"
        agent.metadata["messages_used"] = 10**5
        response = agent.respond("hello")
        assert "limit" not in response.lower()
        assert agent.metadata["messages_used"] == 10**5 + 1

    def test_unknown_plan_defaults_to_free(self, agent):
        agent.metadata["user_plan"] = "enterprise"
        agent.metadata["messages_used"] = get_config().MESSAGE_LIMIT_FREE
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
        agent.metadata["messages_used"] = get_config().MESSAGE_LIMIT_FREE
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
        assert "llm_unavailable" in messages[0]

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


class TestContext:
    def test_request_context_creation(self):
        ctx = RequestContext.new()
        assert ctx.session_id
        assert ctx.request_id
        assert ctx.session_id != ctx.request_id

    def test_request_context_with_session_id(self):
        session = "test-session-123"
        ctx = RequestContext.new(session_id=session)
        assert ctx.session_id == session
        assert ctx.request_id

    def test_agent_accepts_session_id(self, agent):
        response = agent.respond("hello", session_id="session-abc")
        assert "hello" not in response.lower() or "received" in response.lower()

    def test_multiple_requests_different_request_ids(self, agent, caplog, monkeypatch):
        monkeypatch.setattr(log, "propagate", True)
        with caplog.at_level(logging.INFO):
            agent.respond("first message", session_id="session-1")
            agent.respond("second message", session_id="session-1")

        logs = [record.getMessage() for record in caplog.records]

        request_ids = []
        for log_line in logs:
            if '"request_id":' in log_line:
                try:
                    data = json.loads(log_line)
                    if "request_id" in data:
                        request_ids.append(data["request_id"])
                except Exception:
                    pass

        assert len(set(request_ids)) >= 2


class TestRetry:
    def test_recovers_after_transient_failures(self):
        calls = {"count": 0}

        def flaky():
            calls["count"] += 1
            if calls["count"] < 3:
                raise LLMError("transient failure")
            return "ok"

        result = retry_with_backoff(flaky, max_attempts=3, delay_seconds=0)
        assert result == "ok"
        assert calls["count"] == 3

    def test_raises_retry_error_after_max_attempts(self):
        def always_fails():
            raise LLMError("boom")

        with pytest.raises(RetryError):
            retry_with_backoff(always_fails, max_attempts=3, delay_seconds=0, exceptions=(LLMError,))

    def test_only_retries_matching_exceptions(self):
        calls = {"count": 0}

        def raises_value_error():
            calls["count"] += 1
            raise ValueError("not retried")

        with pytest.raises(ValueError):
            retry_with_backoff(
                raises_value_error,
                max_attempts=3,
                delay_seconds=0,
                exceptions=(LLMError,),
            )
        assert calls["count"] == 1

    def test_flaky_mock_fails_initial_attempts_then_succeeds(self):
        client = FlakyMockLLMClient(fail_times=2)
        result = retry_with_backoff(
            lambda: client.generate("hello"),
            max_attempts=5,
            delay_seconds=0,
            exceptions=(LLMError,),
        )
        assert result["reply"]
        assert client.attempts == 3

    def test_agent_retries_flaky_llm_and_recovers(self, agent):
        agent.llm = FlakyMockLLMClient(fail_times=2)
        response = agent.respond("hello")
        assert "received" in response.lower()
        assert agent.llm.attempts == 3

    def test_agent_falls_back_when_llm_keeps_failing(self, agent):
        agent.llm = FlakyMockLLMClient(fail_times=99)
        response = agent.respond("hello")
        assert "unavailable" in response.lower()
        assert agent.llm.attempts == 3