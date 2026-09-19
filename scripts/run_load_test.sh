#!/usr/bin/env bash

set -e

HOST=${1:-"http://127.0.0.1:8000"}
USERS=${2:-100}
SPAWN_RATE=${3:-20}
RUN_TIME=${4:-"30s"}

echo "--------------------------------------------------------"
echo "🔥 Starting High-Concurrency Locust Load Test"
echo "Target Host: ${HOST}"
echo "Concurrent Users: ${USERS}"
echo "Spawn Rate: ${SPAWN_RATE} users/sec"
echo "Duration: ${RUN_TIME}"
echo "--------------------------------------------------------"

locust -f tests/locustfile.py \
  --headless \
  --users "${USERS}" \
  --spawn-rate "${SPAWN_RATE}" \
  --run-time "${RUN_TIME}" \
  --host "${HOST}" \
  --only-summary

echo "✅ Load test execution complete."
