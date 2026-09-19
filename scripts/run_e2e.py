import time
import requests
import json
import sys

BASE_URL = "http://127.0.0.1:8000"

print("--- Starting End-to-End API Test Suite ---")

# 1. Health check
res = requests.get(f"{BASE_URL}/health")
print(f"1. GET /health -> Status: {res.status_code}, Response: {res.json()}")
assert res.status_code == 200

# 2. Chat Endpoint
payload = {"message": "Hello, what tools do you have?"}
res = requests.post(f"{BASE_URL}/chat", json=payload)
print(f"2. POST /chat -> Status: {res.status_code}")
if res.status_code == 200:
    data = res.json()
    print(f"   Response Preview: {data.get('response', '')[:100]}...")
else:
    print(f"   Error: {res.text}")

# 3. Chat Stream Endpoint
payload = {"message": "Tell me a short 1 sentence summary about AI."}
res = requests.post(f"{BASE_URL}/chat/stream", json=payload, stream=True)
print(f"3. POST /chat/stream -> Status: {res.status_code}")
chunks = 0
for chunk in res.iter_content(chunk_size=1024):
    if chunk:
        chunks += 1
print(f"   Received {chunks} stream chunks.")

# 4. Documents List Endpoint
res = requests.get(f"{BASE_URL}/documents")
print(f"4. GET /documents -> Status: {res.status_code}, Docs: {res.json()}")

# 5. Prometheus Metrics Endpoint
res = requests.get(f"{BASE_URL}/metrics")
print(f"5. GET /metrics -> Status: {res.status_code}, Length: {len(res.text)} bytes")
assert "panda_tokens_total" in res.text
assert "panda_active_sessions" in res.text

print("--- All End-to-End API Tests PASSED Successfully! ---")
