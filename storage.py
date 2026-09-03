import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from config import get_config
from logger import get_logger

logger = get_logger(__name__)

DEFAULT_METADATA = {
    "user_plan": get_config().USER_PLAN,
    "messages_used": 0,
}


class MemoryStorage:
    """JSON file-based storage (legacy, not thread-safe)."""

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


class SQLiteStorage:
    """SQLite-based storage with atomic writes and thread safety."""

    def __init__(self, db_path: str = "panda_agent.db"):
        self.db_path = Path(db_path)
        self._init_db()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_session 
                ON messages(session_id)
            """)

            # Initialize default metadata if not present
            cursor = conn.execute("SELECT value FROM metadata WHERE key = 'user_plan'")
            if not cursor.fetchone():
                conn.execute(
                    "INSERT INTO metadata (key, value) VALUES (?, ?)",
                    ("user_plan", get_config().USER_PLAN)
                )
            cursor = conn.execute("SELECT value FROM metadata WHERE key = 'messages_used'")
            if not cursor.fetchone():
                conn.execute(
                    "INSERT INTO metadata (key, value) VALUES (?, ?)",
                    ("messages_used", "0")
                )

    def load_messages(self, session_id: str | None = None, limit: int | None = None) -> list[dict]:
        with self._get_connection() as conn:
            if session_id:
                query = """
                    SELECT role, content FROM messages 
                    WHERE session_id = ? 
                    ORDER BY created_at ASC
                """
                params = [session_id]
            else:
                query = """
                    SELECT role, content FROM messages 
                    ORDER BY created_at ASC
                """
                params = []

            if limit:
                query += f" LIMIT {limit}"

            cursor = conn.execute(query, params)
            return [{"role": row["role"], "content": row["content"]} for row in cursor.fetchall()]

    def save_messages(self, messages: list[dict], session_id: str | None = None):
        """Replace all messages for a session (used by legacy code)."""
        # Use default session if not provided
        sid = session_id or "default"
        with self._get_connection() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (sid,))
            for msg in messages:
                conn.execute(
                    "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                    (sid, msg["role"], msg["content"])
                )

    def append_message(self, session_id: str, role: str, content: str):
        """Append a single message atomically."""
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, role, content)
            )

    def load_metadata(self) -> dict[str, Any]:
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT key, value FROM metadata")
            result = {}
            for row in cursor.fetchall():
                # Try to parse as int
                try:
                    result[row["key"]] = int(row["value"])
                except ValueError:
                    result[row["key"]] = row["value"]
            return result

    def save_metadata(self, metadata: dict[str, Any]):
        with self._get_connection() as conn:
            for key, value in metadata.items():
                conn.execute(
                    "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                    (key, str(value))
                )

    def increment_message_used(self) -> dict[str, Any]:
        """Atomically increment message counter."""
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE metadata SET value = CAST(value AS INTEGER) + 1 WHERE key = 'messages_used'"
            )
            cursor = conn.execute("SELECT key, value FROM metadata")
            result = {}
            for row in cursor.fetchall():
                try:
                    result[row["key"]] = int(row["value"])
                except ValueError:
                    result[row["key"]] = row["value"]
            return result

    def get_message_count(self, session_id: str) -> int:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) as cnt FROM messages WHERE session_id = ?",
                (session_id,)
            )
            return cursor.fetchone()["cnt"]

    def cleanup_old_sessions(self, max_sessions: int = 1000, max_messages_per_session: int = 100):
        """Clean up old sessions to prevent unbounded growth."""
        with self._get_connection() as conn:
            # Keep only latest max_sessions
            conn.execute("""
                DELETE FROM messages 
                WHERE session_id NOT IN (
                    SELECT DISTINCT session_id FROM messages 
                    ORDER BY MAX(created_at) DESC 
                    LIMIT ?
                )
            """, (max_sessions,))

            # Trim messages per session
            conn.execute("""
                DELETE FROM messages 
                WHERE id NOT IN (
                    SELECT id FROM messages 
                    WHERE session_id = messages.session_id
                    ORDER BY created_at DESC 
                    LIMIT ?
                )
            """, (max_messages_per_session,))