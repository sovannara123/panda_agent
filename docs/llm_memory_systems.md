# How ChatGPT and Claude handle memory (and what Panda does today)

## 1. They don't remember — they recompute

Both are stateless. The "memory" is the **context window**: every request
re-sends the conversation history (or a compressed version) to the model.
`memory.json` + `build_prompt()` in `agent.py` is the same idea — replay
history as text.

## 2. Sliding window (Panda's `MAX_HISTORY = 5`)

Both keep only the most recent N messages. ChatGPT uses roughly a
30–128K token window; Panda uses a fixed message count. Same concept,
different units (tokens vs messages).

## 3. Summarization / compaction (the big difference)

When the window overflows, they don't just drop old messages — they
**summarize** them into a compressed "so far we discussed X, user prefers Y"
blob and keep that instead. Panda hard-drops old messages. This is the
upgrade that matters most.

## 4. Long-term memory via retrieval (RAG)

For context beyond the window:

- Facts are embedded into vectors and stored in a **vector DB** (e.g.,
  pgvector, Chroma, Pinecone).
- On each turn, the agent does a **semantic search** and injects only the
  *relevant* past facts into the prompt. Panda filters by recency
  (`self.memory[-5:]`); retrieval filters by relevance instead.

## 5. Persistent structured facts

ChatGPT's "Memory" stores explicit facts (name, location, preferences) in a
**structured store keyed by user_id** — not in the chat log. Panda's
`memory.json` `user_plan` / `message_used` fields are a primitive version of
this (per-user metadata).

## 6. Server-side storage + sessions

In production, `memory.json` (a single local file) becomes a **database table
keyed by session_id/user_id** — one row per message, one per user.

## 7. Caching to cut cost

Both cache prompts/KV so repeated prefixes aren't re-sent to the model each
time — purely an economics optimization, not memory.

---

## Summary for Panda

Panda already has the sliding window (concept 2). The realistic professional
upgrades, in order of value:

1. Summarize old messages instead of dropping (concept 3)
2. SQLite storage keyed by session_id (concept 6)
3. Enforce the free/premium limit (uses the existing `user_plan` field)