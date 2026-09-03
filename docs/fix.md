My brutal recommendation

Don't add more fancy AI features yet.

Before adding RAG, multiple agents, long-term memory, voice, etc., fix these first:

🔴 Fix streaming response/memory bug
🔴 Fix shared mutable self.memory concurrency
🔴 Add tool-call validation
🔴 Add tool authorization
🟠 Add tool timeout
🟠 Add retry policy
🟠 Add maximum tool-call iterations
🟠 Improve exception handling
🟡 Replace run_in_executor() with asyncio.to_thread() where appropriate
🟡 Improve observability with latency/token metrics