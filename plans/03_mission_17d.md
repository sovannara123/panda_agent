# Mission 17D — Retry System Review

When the user returns and says **hello**, remind them to review this mission.

## Goal

- Retry system
- Flaky mock LLM
- Agent recovery from API failure
- Retry tests

## What's implemented

- `retry.py` — `retry_with_backoff` with exponential backoff + `RetryError`
- `llm.py:48` — `FlakyMockLLMClient` (fails N times, then succeeds)
- `agent.py:81` — `generate_llm_response` wraps LLM calls with retry; `respond` falls back on `RetryError`
- `test_agent.py` — `TestRetry` (6 tests) covering recovery, RetryError, exception filtering, agent recovery, agent fallback

All 31 tests pass.
