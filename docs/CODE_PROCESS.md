# Panda Agent - Detailed Code Process Flow

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        PANDA AGENT                              │
├─────────────────────────────────────────────────────────────────┤
│  main.py → Agent → LLM Adapters → Tools → Storage → Logging    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Main Request Flow (`main.py` → `agent.py`)

### 1. Entry Point (`main.py:38-39`)

```python
if __name__ == "__main__":
    main()  # Creates Agent(), loops on user input
```

### 2. Agent Initialization (`agent.py:43-63`)

```python
class Agent:
    def __init__(self, storage=None, llm_client=None, session_id=None):
        self.name = get_config().AGENT_NAME # this get the again name ffrom the configuration 
        self.session_id = session_id # which conversation to load 
        self.storage = storage or SQLiteStorage()  # or MemoryStorage
        self.llm = llm_client or create_llm_client()  # Mock/OpenAI/Groq/Ollama #which Ai / llm to use 
        self.memory = self.storage.load_messages(session_id, limit=MAX_HISTORY)  # load the memory 
        self.metadata = self.storage.load_metadata()
        self.system_prompt = build_system_prompt(self.name, TOOL_SCHEMAS)
```

---

## Response Generation Flow (`agent.py:223-363`)

### Two Response Modes

| Mode | Condition | Method |
|------|-----------|--------|
| **Function Calling** | LLM supports `generate_with_tools` & not Mock | `respond_with_function_calling()` |
| **Legacy (Rule-based)** | Fallback | `_respond_legacy()` |

---

### Legacy Flow (`_respond_legacy`, lines 133-221)

```
User Input
    │
    ▼
validate_user_input() ──► reject if empty/long/invalid
    │
    ▼
check_usage_limit() ──► block if free plan & limit reached
    │
    ▼
plan_tool_call() ──► keyword-based tool detection
    │
    ├─► Tool Found → validate_tool_call() → execute_tool() → create_tool_response()
    │
    └─► No Tool → build_prompt() → generate_llm_response() (with retry)
    │
    ▼
add_message() → storage.save()
    │
    ▼
Return response
```
---

### Function Calling Flow (`respond_with_function_calling`, lines 231-299)

```
User Input
    │
    ▼
validate_user_input()
    │
    ▼
check_usage_limit()
    │
    ▼
Build messages: [system] + history[-10:] + [user]
    │
    ▼
llm.generate_with_tools(messages, OPENAI_TOOLS)
    │
    ├─► Tool Calls → _handle_tool_calls() → execute tools → llm.generate_with_tools() again
    │
    └─► Direct Reply → return
    │
    ▼
add_message() → storage.save()
    │
    ▼
Return response
```

---

## Tool System (`tools.py`)

### Tool Planning (`plan_tool_call`, lines 58-105)

- **Keyword-based** detection (not LLM-based)
- Matches: `price/cost/rate` → `get_product_price`
- Matches: `order/track/status` + `A###` → `check_order_status`
- Matches: `weather` + city → `get_weather`
- Matches: `test failure/simulate failure` → `test_failure`

### Tool Validation (`validate_tool_call`, lines 108-125)

- Checks tool exists in `TOOL_SCHEMAS`
- Validates required arguments against JSON schema

### Tool Execution (`execute_tool`, lines 156-202)

```python
tool_registry = {
    "get_product_price": get_product_price,    # Looks up PRODUCTS dict
    "check_order_status": check_order_status,  # Looks up ORDERS dict  
    "get_weather": get_weather,                # Looks up WEATHER dict
    "test_failure": test_failure               # Always returns error
}
```

---

## Storage Layer (`storage.py`)

### Two Implementations

| Storage | Use Case | Features |
|---------|----------|----------|
| `MemoryStorage` | Testing/Dev | JSON files, legacy migration |
| `SQLiteStorage` | Production | Thread-safe, session-aware, atomic writes, indexing |

### SQLite Schema

```sql
messages (id, session_id, role, content, created_at)
metadata (key, value)  -- user_plan, messages_used
```

---

## LLM Adapters (`llm_adapters.py`)

### Provider Factory (`create_llm_client`, line 1209)

```python
provider = get_config().LLM_PROVIDER  # "mock" | "openai" | "groq" | "ollama"
```

### Interface (`LLMClient` abstract base, lines 22-69)

```python
# Sync
generate(prompt) → dict
generate_with_tools(messages, tools) → dict
generate_stream(prompt) → Generator[dict]
generate_with_tools_stream(messages, tools) → Generator[dict]

# Async (all implemented)
generate_async()
generate_with_tools_async()
generate_stream_async() → AsyncGenerator[dict]
generate_with_tools_stream_async() → AsyncGenerator[dict]
```

### Streaming Format

```python
# Content chunk
{"type": "content", "content": "token"}

# Tool calls detected
{"type": "tool_calls", "tool_calls": [...]}

# End of stream
{"type": "done", "full_content": "...", "usage": {...}}
```

---

## Retry Logic (`retry.py`)

```python
retry_with_backoff(func, max_attempts=3, delay_seconds=0.2, exceptions=(Exception,))
```

- Exponential backoff: `delay * 2^(attempt-1)`
- Logs each retry attempt
- Raises `RetryError` after max attempts

---

## Logging System (`logger.py`)

### Structured JSON Logging with Correlation IDs

```json
{
  "timestamp": "14:30:05",
  "level": "INFO",
  "logger": "agent",
  "message": "{\"event\": \"user_input\", \"message\": \"hello\", \"session_id\": \"abc\", \"request_id\": \"xyz\"}",
  "session_id": "abc",
  "request_id": "xyz"
}
```

### Event Types

- `user_input`, `usage_check`, `usage_blocked`
- `tool_planned`, `tool_result`, `tool_failed`
- `llm_call`, `response_sent`, `fallback`, `retry`

---

## Fallback Responses (`fallback.py`)

| Reason | Response |
|--------|----------|
| `tool_failed` | "Sorry, that action failed. You can try again..." |
| `llm_unavailable` | "Sorry, the AI service is unavailable right now..." |
| `invalid_tool` | "I could not process that tool request. Try: 'price of laptop'..." |
| `usage_limit` | "You have reached the free message limit. Upgrade to continue." |
| `unknown` | "I am not sure how to help with that yet." |

---

## Configuration (`config.py`)

```python
LLM_PROVIDER = "mock"           # mock|openai|groq|ollama|gemini
MODEL_NAME = "gpt-4o-mini"
MESSAGE_LIMIT_FREE = 10         # Free tier limit
USER_PLAN = "free"              # free|premium
MAX_HISTORY = 20                # Conversation window
RETRY_ATTEMPTS = 3
RETRY_DELAY = 0.2
LOG_LEVEL = "INFO"
```

---

## Async Agent (`async_agent.py`)

Extends `Agent` with concurrent tool execution:

- `respond_async()` - Full async response
- `respond_stream_async()` - Async streaming
- `_handle_tool_calls_async()` - Uses `asyncio.gather()` for parallel tool execution

---

## Testing (`test_agent.py`)

**31 tests covering:**

- Tool calling (validation, planning)
- Agent responses (price format, usage limits)
- Usage limits (at/above/below limit, premium unlimited)
- Memory persistence
- Fallback logging (usage, LLM error, tool failure)
- Failure tool (graceful degradation)
- Request context (session/request IDs)
- Retry logic (transient failures, max attempts, exception filtering, flaky mock)

---

## Data Flow Summary




```
┌──────────────┐
│   User Input │
└──────┬───────┘
       ▼
┌──────────────┐     ┌─────────────────┐
│  validate()  │────►│ check_usage()   │
└──────────────┘     └────────┬────────┘
                              ▼
                    ┌─────────────────┐
                    │ plan_tool_call()│
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
      ┌───────────────┐             ┌───────────────┐
      │ Tool Detected │             │  No Tool      │
      └───────┬───────┘             └───────┬───────┘
              ▼                             ▼
      ┌───────────────┐             ┌───────────────┐
      │ validate()    │             │ build_prompt()│
      └───────┬───────┘             └───────┬───────┘
              ▼                             ▼
      ┌───────────────┐             ┌───────────────┐
      │ execute_tool()│             │ generate_llm()│
      └───────┬───────┘             └───────┬───────┘
              ▼                             ▼
      ┌───────────────┐             ┌───────────────┐
      │ create_resp() │             │   (retry)     │
      └───────┬───────┘             └───────┬───────┘
              │                             │
              └──────────────┬──────────────┘
                             ▼
                    ┌─────────────────┐
                    │ add_message()   │
                    │ storage.save()  │
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │ Return Response │
                    └─────────────────┘
```





---

## Key Design Patterns

1. **Strategy Pattern** - LLM adapters (swap providers)
2. **Template Method** - `_respond_legacy` vs `respond_with_function_calling`
3. **Decorator** - `retry_with_backoff` wraps LLM calls
4. **Observer** - Structured logging with correlation IDs
5. **Fallback** - Graceful degradation at every failure point