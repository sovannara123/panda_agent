# Panda Agent Developer Rules

These rules configure how the AI assistant should behave and automatically route requests to the appropriate expert skills based on the context of the user's prompt or the files being edited.

## Skill Routing Rules

Whenever I ask for help, analyze the request and the files involved. If the context matches one of the criteria below, YOU MUST read and apply the corresponding skill from your global skills directory (`~/.gemini/config/skills/`) before proceeding with the task.

- **RAG & Embeddings**: If the request involves Retrieval-Augmented Generation, vector databases, embeddings, document parsing, or touches files in the `panda_agent/rag/` directory, apply the `agency-rag-pipeline-engineer` skill.
- **Agent Orchestration & Tools**: If the request involves agent logic, execution flow, tool calling, or touches files in `panda_agent/agent/` or `panda_agent/tools/`, apply the `agency-agents-orchestrator` skill.
- **Backend & APIs**: If the request involves the web server, endpoints, or touches files in the `panda_agent/api/` directory, apply the `agency-backend-architect` skill.
- **LLM & Prompts**: If the request involves prompt tuning, LLM integrations, or touches files in the `panda_agent/llm/` directory, apply the `agency-prompt-engineer` skill.
- **Testing**: If the request involves unit tests, integration tests, or touches files in the `tests/` directory, apply the `agency-test-automation-engineer` skill.
- **Architecture**: If the request is about system design, refactoring, or project structure, apply the `agency-software-architect` skill.

## Post-Execution Documentation Rule

Whenever any plan or major task is executed and completed:
- Update the relevant plan file in `plans/` (mark tasks as completed `- [x]`, update Status to `Completed`, and note any execution outcomes).
- Update technical documentation in `docs/` and `README.md` if public contracts, flow diagrams, API interfaces, architecture, or environment configurations changed.
- Ensure all docstrings, comments, and reference files accurately reflect the updated implementation.

## General Project Guidelines
- Always write type-hinted, clean, and maintainable Python code.
- Prefer explicit over implicit implementations.
- Maintain and update docstrings when making significant changes to functions or classes.

