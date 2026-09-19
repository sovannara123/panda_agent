import uuid
from locust import HttpUser, task, between


class PandaAgentLoadUser(HttpUser):
    """Locust load test user simulating high-concurrency panda agent requests."""

    wait_time = between(1, 3)

    def on_start(self):
        """Generate unique session ID for each virtual user."""
        self.session_id = str(uuid.uuid4())
        self.headers = {
            "Content-Type": "application/json",
            # Optional: Add authorization if REQUIRE_API_KEY is enabled
            # "Authorization": "Bearer test-api-key"
        }

    @task(4)
    def test_sync_chat(self):
        """Simulate synchronous chat queries."""
        payload = {
            "message": "What is the price of laptop?",
            "session_id": self.session_id
        }
        with self.client.post("/chat", json=payload, headers=self.headers, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Chat failed with status {response.status_code}: {response.text}")

    @task(3)
    def test_stream_chat(self):
        """Simulate high-concurrency SSE streaming chat requests."""
        payload = {
            "message": "Give me a 1 sentence tip on Python performance.",
            "session_id": self.session_id
        }
        with self.client.post("/chat/stream", json=payload, headers=self.headers, stream=True, catch_response=True) as response:
            if response.status_code == 200:
                chunks = 0
                for chunk in response.iter_content(chunk_size=512):
                    if chunk:
                        chunks += 1
                if chunks > 0:
                    response.success()
                else:
                    response.failure("Streaming response returned no chunks")
            else:
                response.failure(f"Stream chat failed with status {response.status_code}")

    @task(2)
    def test_get_documents(self):
        """Simulate fetching document knowledge base list."""
        self.client.get("/documents", headers=self.headers)

    @task(1)
    def test_health_and_metrics(self):
        """Simulate monitoring / load balancer polling."""
        self.client.get("/health")
        self.client.get("/metrics", headers=self.headers)
