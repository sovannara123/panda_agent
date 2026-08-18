import pytest

from agent import Agent
from storage import MemoryStorage
from tools import validate_tool_call, plan_tool_call


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
        agent.metadata["messages_used"] = 10
        response = agent.respond("hello")
        assert "limit" in response.lower()
        assert agent.metadata["messages_used"] == 10


class TestMemory:
    def test_memory_persists(self, agent):
        agent.respond("test message")
        reloaded = agent.storage.load_messages()
        assert any(m["content"] == "test message" for m in reloaded)