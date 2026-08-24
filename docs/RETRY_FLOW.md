# Retry System Flow Diagram

## Overview
The retry system handles transient LLM failures with exponential backoff and graceful fallback.

## Flow

```
User Input → agent.respond()
    │
    ├──     Tool call? → Execute tool → Success/Fallback
    │
    └── LLM call needed?
            │
            ▼
    generate_llm_response(prompt)
            │
            ▼
    retry_with_backoff(llm.generate, max=3, delay=0.1, exceptions=LLMError)
            │
            ├── Attempt 1: Success? → Return result
            │
            ├── Attempt 1: LLMError? → Log → Sleep 0.1s → Attempt 2
            │
            ├── Attempt 2: LLMError? → Log → Sleep 0.2s → Attempt 3
            │
            ├── Attempt 3: LLMError? → Raise RetryError
            │
            └── Other exception? → Bubble up immediately
            │
            ▼
    Catch RetryError → Fallback "llm_unavailable"
    Catch LLMError → Fallback "llm_unavailable"
```

## Components

### 1. Core Retry Logic (`retry.py:10-38`)
- `retry_with_backoff(func, max_attempts=3, delay_seconds=0.1, exceptions=(Exception,))`
- Exponential backoff: `delay * 2^(attempt-1)`
- Logs each retry attempt
- Raises `RetryError` after max attempts

### 2. Flaky Mock LLM (`llm.py:48-62`)
- `FlakyMockLLMClient(fail_times=N)` - fails N times, then succeeds
- Used for testing retry behavior

### 3. Agent Integration (`agent.py:81-87`)
```python
def generate_llm_response(self, prompt):
    return retry_with_backoff(
        lambda: self.llm.generate(prompt),
        max_attempts=Config.RETRY_ATTEMPTS,
        delay_seconds=Config.RETRY_DELAY,
        exceptions=LLMError,
    )
```

### 4. Agent Fallback (`agent.py:161-169`)
- `RetryError` → all retries exhausted → fallback
- `LLMError` (not caught by retry) → immediate fallback

## Test Coverage (`test_agent.py:214-272`)
| Test | Verifies |
|------|----------|
| `test_recovers_after_transient_failures` | Retry succeeds after 2 failures |
| `test_raises_retry_error_after_max_attempts` | Raises `RetryError` after max attempts |
| `test_only_retries_matching_exceptions` | Non-matching exceptions bubble up |
| `test_flaky_mock_fails_initial_attempts_then_succeeds` | FlakyMock works with retry |
| `test_agent_retries_flaky_llm_and_recovers` | Agent recovers from flaky LLM |
| `test_agent_falls_back_when_llm_keeps_failing` | Agent falls back after all retries fail |

## Configuration
- `Config.RETRY_ATTEMPTS = 3`
- `Config.RETRY_DELAY = 0.1`