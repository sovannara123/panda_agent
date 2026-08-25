import json
import logging
from pathlib import Path
from config import Config

logger = logging.getLogger(__name__)

DEFAULT_METADATA = {
    "user_plan": Config.USER_PLAN,
    "messages_used": 0,
}


class MemoryStorage:
    def __init__(self, messages_path="memory.json", meta_path="user_meta.json"):
        self.messages_path = Path(messages_path)
        self.meta_path = Path(meta_path)

        self._migrate_if_needed()

        if not self.messages_path.exists():
            self.messages_path.write_text("[]", encoding="utf-8")

        if not self.meta_path.exists():
            self.meta_path.write_text(
                json.dumps(DEFAULT_METADATA, indent=2),
                encoding="utf-8",
            )

    def _migrate_if_needed(self):
        if not self.messages_path.exists():
            return
        try:
            data = json.loads(self.messages_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(data, dict):
            return

        messages = data.get("conversation", [])
        metadata = {
            "user_plan": data.get("user_plan", DEFAULT_METADATA["user_plan"]),
            "messages_used": data.get("message_used", DEFAULT_METADATA["messages_used"]),
        }
        self.messages_path.write_text(
            json.dumps(messages, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self.meta_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Migrated legacy memory.json into separate stores")

    def load_messages(self):
        try:
            data = json.loads(self.messages_path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def save_messages(self, messages):
        self.messages_path.write_text(
            json.dumps(messages, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def load_metadata(self):
        try:
            data = json.loads(self.meta_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else dict(DEFAULT_METADATA)
        except (FileNotFoundError, json.JSONDecodeError):
            return dict(DEFAULT_METADATA)

    def save_metadata(self, metadata):
        self.meta_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def increment_message_used(self):
        metadata = self.load_metadata()
        metadata["messages_used"] = metadata.get("messages_used", 0) + 1
        self.save_metadata(metadata)
        return metadata
