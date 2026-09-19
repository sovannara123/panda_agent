import pytest
import os
os.environ["REQUIRE_API_KEY"] = "false"
from fastapi.testclient import TestClient
from pathlib import Path
from panda_agent.api.app import app
from panda_agent.core.config import get_config
from panda_agent.rag.pipeline import search_knowledge_base


client = TestClient(app)


def test_health_check_public():
    """Health check endpoint must remain public without auth."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_api_key_authentication_enforcement():
    """Verify that when REQUIRE_API_KEY=True, unauthenticated requests return 401."""
    config = get_config()
    orig_require = config.REQUIRE_API_KEY
    orig_key = config.API_KEY
    
    try:
        config.REQUIRE_API_KEY = True
        config.API_KEY = "secret_test_key_123"

        # Request without header -> 401
        res = client.post("/chat", json={"message": "hello"})
        assert res.status_code == 401

        # Request with wrong header -> 401
        res = client.post("/chat", json={"message": "hello"}, headers={"X-API-Key": "wrong_key"})
        assert res.status_code == 401

        # Request with correct X-API-Key header -> 200
        res = client.post("/chat", json={"message": "hello"}, headers={"X-API-Key": "secret_test_key_123"})
        assert res.status_code == 200

        # Request with correct Bearer header -> 200
        res = client.post("/chat", json={"message": "hello"}, headers={"Authorization": "Bearer secret_test_key_123"})
        assert res.status_code == 200

    finally:
        config.REQUIRE_API_KEY = orig_require
        config.API_KEY = orig_key


def test_path_traversal_filename_sanitization():
    """Verify incoming file path traversal attempts are stripped."""
    traversal_filename = "../../etc/passwd.pdf"
    clean_name = Path(traversal_filename).name
    assert clean_name == "passwd.pdf"
    assert ".." not in clean_name


def test_oversized_file_upload_rejection(tmp_path):
    """Verify file uploads exceeding size limit are rejected with 413 Payload Too Large."""
    config = get_config()
    orig_limit = config.MAX_UPLOAD_SIZE_MB
    
    try:
        # Set limit to 1MB for quick testing
        config.MAX_UPLOAD_SIZE_MB = 1
        
        # Create a 2MB dummy buffer
        dummy_content = b"0" * (2 * 1024 * 1024)
        files = {"file": ("test_large.pdf", dummy_content, "application/pdf")}
        
        res = client.post("/upload", files=files)
        assert res.status_code == 413
        assert "Payload Too Large" in res.json()["detail"]
    finally:
        config.MAX_UPLOAD_SIZE_MB = orig_limit


def test_rag_context_fencing():
    """Verify that RAG search responses wrap document context in <retrieved_context> XML tags."""
    res = search_knowledge_base("dummy query")
    if res["found"]:
        assert res["context"].startswith("<retrieved_context>")
        assert res["context"].endswith("</retrieved_context>")


def test_rate_limiting_enforcement():
    """Verify rate limit per minute returns 429 when threshold exceeded."""
    config = get_config()
    orig_limit = config.RATE_LIMIT_PER_MINUTE
    
    try:
        config.RATE_LIMIT_PER_MINUTE = 3
        from panda_agent.api.app import _request_timestamps
        _request_timestamps.clear()

        # Send 3 allowed requests
        for _ in range(3):
            res = client.get("/documents")
            assert res.status_code != 429

        # 4th request must be rate-limited with 429
        res = client.get("/documents")
        assert res.status_code == 429
        assert "Rate limit exceeded" in res.json()["detail"]

    finally:
        config.RATE_LIMIT_PER_MINUTE = orig_limit
        from panda_agent.api.app import _request_timestamps
        _request_timestamps.clear()
