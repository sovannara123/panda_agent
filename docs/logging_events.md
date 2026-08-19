# Logging Events

Structured JSON events emitted by `logger.py` and fired from `agent.py`.

## Structured events (logger.py → agent.py)

| Event | Fired when | Location |
| ----- | ---------- | -------- |
| `user_input` | user message received | `respond()` entry |
| `usage_check` | every turn, guardrail check | `respond()` |
| `usage_blocked` | free-tier limit reached, turn blocked | `respond()` blocked branch |
| `tool_planned` | `plan_tool_call()` matched a tool | `respond()` |
| `tool_call` | tool executed (success + duration) | `respond()` |
| `tool_failed` | tool returned error status | `respond()` |
| `llm_call` | LLM attempt, success or error | `respond()` try/except |
| `response_sent` | reply returned (all paths) | `respond()` |

## Other logged events (standard logging)

| Module | Event | Level |
| ------ | ----- | ----- |
| `tools.py` | Unknown tool | WARNING |
| `tools.py` | Invalid tool arguments | WARNING |
| `tools.py` | Tool failed | ERROR (exception) |
| `storage.py` | Failed to load memory file | WARNING |
| `llm.py` | Empty prompt rejected | WARNING |
| `llm.py` | Simulated LLM crash | WARNING |
| `llm.py` | LLM response generated | INFO |