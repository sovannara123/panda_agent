# Panda — AI Learning Assistant

A lightweight conversational AI agent with memory, intent detection, and
optional OpenAI integration. Built as a practical exercise in building AI
agents in Python.

## Features

- Rule-based intent detection (`tools.py`) with local fallback responses
- Optional real LLM responses via the OpenAI API (`llm.py`)
- Persistent conversation memory stored in `memory.json` and user metadata in
  `user_meta.json` (`storage.py`)
- Free / premium access tiers with a message limit (`agent.py`)

## Requirements

- Python 3.10+
- An optional [OpenAI API key](https://platform.openai.com/api-keys)

## Installation

```bash
git clone https://github.com/<your-username>/panda-agent.git
cd panda-agent
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration

Copy the example environment file and add your API key:

```bash
cp .env.example .env
```

| Variable            | Default         | Description                         |
| ------------------- | --------------- | ----------------------------------- |
| `OPENAI_API_KEY`    | *(empty)*       | API key for real LLM responses      |
| `MODEL_NAME`        | `gpt-4o-mini`   | OpenAI model to use                 |
| `AGENT_NAME`        | `Panda`          | Agent display name                  |
| `AGENT_ROLE`        | `AI learning assistant` | Agent role                     |
| `MESSAGE_LIMIT_FREE`| `10`            | Messages allowed on the free tier   |

Without an API key the agent falls back to local rule-based responses, so it
always runs out of the box.

## Usage

```bash
python main.py
```

Start chatting. Type `quit` or `exit` to end the session. Conversation history
is saved to `memory.json` and user metadata (plan, message usage) to
`user_meta.json` automatically. Free-tier users are limited to
`MESSAGE_LIMIT_FREE` messages; hitting the limit returns an "upgrade" message
and the blocked turn is neither counted nor stored.

### Resetting usage / testing the rate limit

Both files are runtime state and gitignored. To reset for testing:

```bash
rm user_meta.json memory.json   # next run starts fresh
# or just reset the counter
echo '{"user_plan": "free", "messages_used": 0}' > user_meta.json
```

To see the limit fire, set `MESSAGE_LIMIT_FREE` low in `.env` (e.g. `3`) or
send more messages than the default 10.

## Project structure

```
panda-agent/
├── main.py        # Entry point
├── agent.py       # Agent class, access control, conversation loop
├── config.py      # Settings loaded from environment variables
├── tools.py       # Intent detection + local fallback responses
├── llm.py         # Optional OpenAI chat completions
├── storage.py     # Persist messages + user metadata in separate stores
└── .env.example   # Example environment configuration
```

## License

