import pytest
from fastapi.testclient import TestClient

from panda_agent.api.app import app
from panda_agent.core.config import get_config

client = TestClient(app)

def test_metrics_endpoint_unauthorized():
    """Test that metrics endpoint requires authentication."""
    config = get_config()
    orig = config.REQUIRE_API_KEY
    orig_key = config.API_KEY
    try:
        config.REQUIRE_API_KEY = True
        config.API_KEY = "secret_key_to_force_auth"
        response = client.get("/metrics")
        assert response.status_code == 401
    finally:
        config.REQUIRE_API_KEY = orig
        config.API_KEY = orig_key

def test_metrics_endpoint_authorized():
    """Test that metrics endpoint returns Prometheus data."""
    config = get_config()
    api_key = config.API_KEY
    if not api_key:
        api_key = "test-api-key"
    
    orig = config.REQUIRE_API_KEY
    orig_key = config.API_KEY
    try:
        config.REQUIRE_API_KEY = True
        config.API_KEY = api_key
        headers = {"Authorization": f"Bearer {api_key}"}
        response = client.get("/metrics", headers=headers)
        
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        
        # Check for basic Python prometheus metrics (process, python, or our custom ones)
        content = response.text
        assert "panda_active_sessions" in content
        assert "panda_llm_latency_seconds" in content
        assert "panda_tool_latency_seconds" in content
        assert "panda_rag_latency_seconds" in content
        assert "panda_tokens_total" in content
        assert "panda_tool_errors_total" in content
    finally:
        config.REQUIRE_API_KEY = orig
        config.API_KEY = orig_key
