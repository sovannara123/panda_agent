import pytest
from fastapi.testclient import TestClient
from panda_agent.api.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


class TestAPIEndpoints:
    """Test suite for FastAPI endpoints."""

    def test_health_endpoint(self, client):
        """Test GET /health returns 200 and expected schema."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["agent_ready"] is True
        assert data["version"] == "1.0.0"

    def test_root_frontend_endpoint(self, client):
        """Test GET / serves static index.html with HTML content."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        assert len(response.content) > 0

    def test_documents_list_endpoint(self, client):
        """Test GET /documents returns list of ingested documents."""
        response = client.get("/documents")
        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
        assert isinstance(data["documents"], list)

    def test_chat_endpoint_valid_tool_call(self, client):
        """Test POST /chat executes tool call and returns ChatResponse."""
        response = client.post("/chat", json={"message": "What is the price of laptop?"})
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "999" in data["response"]
        assert "session_id" in data

    def test_chat_endpoint_empty_message_rejected(self, client):
        """Test POST /chat rejects empty message according to schema."""
        response = client.post("/chat", json={"message": "   "})
        assert response.status_code == 422  # Pydantic validation error

    def test_chat_stream_endpoint(self, client):
        """Test POST /chat/stream streams content successfully."""
        response = client.post("/chat/stream", json={"message": "Hello"})
        assert response.status_code == 200
        assert len(response.content) > 0
