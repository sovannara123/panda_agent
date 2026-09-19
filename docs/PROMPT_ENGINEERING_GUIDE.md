# Panda Agent - Production Prompt Engineering & Flaw Detection Guide

This guide establishes the architectural standards, security defenses, and automated flaw detection methods for all LLM prompts used in the Panda Agent platform.

---

## 1. The 8 Core Production Pillars

Every prompt deployed to production must satisfy the following 8 criteria:

| # | Pillar | Production Requirement | Defense Mechanism in Panda Agent |
|---|--------|------------------------|-----------------------------------|
| 1 | **Instruction Clarity & Determinism** | Explicitly declared role, clear task scope, and unambiguous rules. | `SYSTEM_PROMPT_V2` defines unambiguous role hierarchy and rules. |
| 2 | **Prompt Injection Defense** | Separation of system instructions from untrusted user/document text. | Typed XML tags (`<user_input>`, `<history>`, `<retrieved_context>`) + input tag escaping. |
| 3 | **Indirect RAG Injection Defense** | Retrieved documents must never be allowed to execute system actions. | Explicit rule stating all text in `<retrieved_context>` is passive reference data. |
| 4 | **Tool Calling Integrity** | Tools must have complete schemas and strict argument requirements. | `format_tool_descriptions` formats typed parameter specs and required flags. |
| 5 | **Factual Grounding** | The model must never fabricate unverifiable data. | Explicit `NEVER fabricate prices, order IDs, or facts` rule. |
| 6 | **Fallback & Error Recovery** | The model must handle missing parameters or failed tools gracefully. | Fallback instructions ask for user clarification instead of guessing. |
| 7 | **Tone & Token Efficiency** | Direct answers without bloated commentary. | Conciseness constraints prevent token waste and reduce latency. |
| 8 | **Auditability & Versioning** | Every prompt version is tracked like software code. | `PROMPT_VERSION = "2.0.0"` with changelog and unit tests. |

---

## 2. Prompt Architecture (v2.0.0)

### System Prompt Structure
```markdown
You are {agent_name}, an enterprise AI support and operations assistant.

## CORE ROLE & RESPONSIBILITIES
...

## AVAILABLE TOOLS
{tool_descriptions}

## OPERATIONAL RULES & CONSTRAINTS
1. TOOL SELECTION
2. GROUNDING & FACTUAL INTEGRITY
3. CONVERSATION TONE & STYLE
4. SECURITY & INJECTION DEFENSE
5. FALLBACK BEHAVIOR
```

### User Message Delimitation
```xml
<history>
user: Hello
assistant: Hi! How can I help you today?
</history>

<user_input>
What is the status of order 12345?
</user_input>
```

### RAG Passive Context Wrapper
```xml
<retrieved_context>
Orders placed before 2 PM EST ship on the same business day.
</retrieved_context>

<user_input>
When will my order ship if I ordered at 1 PM EST?
</user_input>
```

---

## 3. Automated Prompt Flaw Auditor

The `PromptFlawAuditor` in `panda_agent/llm/auditor.py` automatically checks any prompt against the 8 production pillars and returns a quantitative readiness report:

```python
from panda_agent.llm.auditor import PromptFlawAuditor
from panda_agent.schemas.tool_schemas import TOOL_SCHEMAS

auditor = PromptFlawAuditor()
report = auditor.audit_system_prompt(my_prompt_string, TOOL_SCHEMAS)

print(f"Readiness Score: {report.readiness_score}%")
print(f"Status: {report.status}")
for flaw in report.flaws:
    print(f"- [{flaw.severity}] {flaw.title}: {flaw.remediation}")
```

---

## 4. Red-Teaming Adversarial Test Suite

Run `pytest tests/test_prompt_engineering.py` to verify:
- Delimiter breakout attacks (`</user_input><system>...`) are escaped.
- System prompt contains all required security instructions.
- User and RAG prompt builders maintain structured XML containers.
- The automated auditor awards `READY` status (>= 85%) to the production prompt.
