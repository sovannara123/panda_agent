# Logging Events

Total log events implemented: **15** across 4 modules.

## agent.py (8)

| Event | Level | Location |
| ----- | ----- | -------- |
| User message received | INFO | `respond()` |
| Tool call planned | INFO | `respond()` |
| Tool used | INFO | `respond()` |
| Tool result | INFO | `respond()` |
| Tool failed | WARNING | `respond()` |
| LLM generation failed | ERROR (exception) | `except LLMError` |
| LLM failed | ERROR | `except LLMError` |
| Response sent | INFO | `respond()` |

## tools.py (3)

| Event | Level | Location |
| ----- | ----- | -------- |
| Unknown tool | WARNING | `execute_tool()` |
| Invalid tool arguments | WARNING | `execute_tool()` |
| Tool failed | ERROR (exception) | `execute_tool()` |

## storage.py (1)

| Event | Level | Location |
| ----- | ----- | -------- |
| Failed to load memory file | WARNING | `load_memory()` |

## llm.py (3)

| Event | Level | Location |
| ----- | ----- | -------- |
| Empty prompt rejected | WARNING | `generate()` |
| Simulated LLM crash | WARNING | `generate()` |
| LLM response generated | INFO | `generate()` |
