import json
from copy import deepcopy
from pathlib import Path

MEMORY_FILE = Path(__file__).with_name("memory.json")

DEFAULT_MEMORY = {
    "conversation": [],
    "user_plan": "free",
    "message_used": 0,
}


def load_memory():
    if not MEMORY_FILE.exists():
        return deepcopy(DEFAULT_MEMORY)
    try:
        data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return deepcopy(DEFAULT_MEMORY)
    memory = deepcopy(DEFAULT_MEMORY)
    memory.update(data)
    return memory


def save_memory(memory):
    MEMORY_FILE.write_text(
        json.dumps(memory, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )