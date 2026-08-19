# Request Flow: From User Input to Response

## Flow (file by file)

```
main.py
  └─ agent.respond(user_input) ────────────────────────────► agent.py:63
       ├─ check_usage_limit()                                agent.py:53 (reads metadata)
       ├─ log_usage_check() ────────────────────────────────► logger.py:34
       ├─ add_message("user", ...) ─────────────────────────► storage.py:54 (memory.json)
       │
       ├─ plan_tool_call(user_input) ───────────────────────► tools.py:56
       │    └─ reads TOOL_SCHEMAS ──────────────────────────► tool_schemas.py
       │
       ├─ [tool path]
       │    ├─ validate_tool_call() ────────────────────────► tools.py:97 (vs tool_schemas.py)
       │    ├─ execute_tool() ──────────────────────────────► tools.py:124
       │    ├─ log_tool_call() ─────────────────────────────► logger.py:19
       │    └─ create_tool_response()                        agent.py:36
       │
       └─ [LLM path]
            ├─ build_prompt() ──────────────────────────────► prompts.py:35
            ├─ build_user_prompt()                           prompts.py:35
            ├─ llm.generate() ──────────────────────────────► llm.py:18 (mock/openai/gemini)
            └─ log_llm_call() ──────────────────────────────► logger.py:25
                     
       └─ add_message("assistant", ...) ────────────────────► storage.py:61
```

**Summary:** `main.py` → `agent.py` → `tools.py` (+ `tool_schemas.py`) or
`llm.py` → `prompts.py` → `storage.py`/`logger.py` → back to `main.py`.

## Phases

1. **Guardrail check** — `check_usage_limit()` (agent.py:53) reads plan +
   messages used; `log_usage_check()` (agent.py:67) records the outcome. If
   blocked, the turn is neither counted nor stored.
2. **Intent detection** — `plan_tool_call()` (tools.py:56) runs regex/keyword
   matching against product/order/weather patterns.
3. **Dispatch** — a valid tool call is validated against `TOOL_SCHEMAS` and
   executed via `tool_registry`; otherwise the LLM fallback builds a prompt
   from `prompts.py` and calls `llm.py`.
4. **Persist & return** — assistant reply saved to `memory.json`, returned to
   `main.py`, printed.

## File responsibilities

| File               | Responsibility                                     | Key entry points             |
| ------------------ | -------------------------------------------------- | ---------------------------- |
| `main.py`          | CLI loop, reads user input, prints response        | `main()`                     |
| `agent.py`         | Orchestration: guardrails, intent, dispatch        | `respond()`, `check_usage_limit()` |
| `tools.py`         | Intent planning, schema validation, tool execution | `plan_tool_call()`, `execute_tool()` |
| `tool_schemas.py`  | JSON schemas (name/description/parameters)         | `TOOL_SCHEMAS`               |
| `prompts.py`       | Prompt templates (system + user)                   | `build_system_prompt()`, `build_user_prompt()` |
| `llm.py`           | LLM providers (mock/openai/gemini)                 | `generate()`, `create_llm_client()` |
| `storage.py`       | Persistence (memory.json, user_meta.json)          | `load_messages()`, `save_metadata()` |
| `logger.py`        | Structured JSON event logging                      | `log_tool_call()`, `log_llm_call()`, `log_usage_check()` |

## Observability hooks

| Step                       | Event            | Where fired             |
| -------------------------- | ---------------- | ----------------------- |
| Every turn (input)         | `user_input`     | agent.py:64            |
| Every turn (guardrail)     | `usage_check`    | agent.py:68             |
| Turn blocked by limit      | `usage_blocked`  | agent.py:74             |
| Every tool planning        | `tool_planned`   | agent.py:85             |
| Every tool execution       | `tool_call`      | agent.py:96             |
| Tool failure               | `tool_failed`    | agent.py:102            |
| Every LLM attempt          | `llm_call`       | agent.py:109 (success), agent.py:113 (failure) |
| Fallback response used     | `fallback`       | agent.py:39/78/93/122 |
| Every reply                | `response_sent`  | agent.py:75/89/119      |

## Example walkthrough — `"price of phone"`

1. `main.py:28` calls `agent.respond("price of phone")`.
2. Guardrail passes; `messages_used` incremented (agent.py:76).
3. `plan_tool_call()` matches `price` keyword + whole-word `phone` →
   `{"tool": "get_product_price", "arguments": {"product_name": "phone"}}`.
4. `validate_tool_call()` confirms `product_name` is required and present.
5. `execute_tool()` → `get_product_price("phone")` → `{"status": "success",
   "result": {"product": "phone", "price": 699}}`.
6. `create_tool_response()` formats → `"The price of phone is $699."`
7. Reply saved to `memory.json` (storage.py:61) and returned to `main.py`.