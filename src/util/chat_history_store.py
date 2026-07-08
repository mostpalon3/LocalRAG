import sqlite3
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import PROJECT_ROOT


CHAT_HISTORY_DB_PATH = Path(PROJECT_ROOT) / "data" / "chat_history.sqlite3"
DEFAULT_CHAT_TITLE = "New chat"


def _connect() -> sqlite3.Connection:
    CHAT_HISTORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(CHAT_HISTORY_DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_chat_history_store() -> None:
    with _connect() as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(chat_id) REFERENCES chats(id) ON DELETE CASCADE
            )
            """
        )
        if _needs_legacy_migration(connection):
            _migrate_legacy_chat_history(connection)
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_chats_updated_at ON chats(updated_at DESC, created_at DESC)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_messages_chat_id_id ON chat_messages(chat_id, id)"
        )


def _needs_legacy_migration(connection: sqlite3.Connection) -> bool:
    columns = {row["name"] for row in connection.execute("PRAGMA table_info(chat_messages)").fetchall()}
    return "session_key" in columns and "chat_id" not in columns


def _migrate_legacy_chat_history(connection: sqlite3.Connection) -> None:
    connection.execute("ALTER TABLE chat_messages RENAME TO chat_messages_legacy")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(chat_id) REFERENCES chats(id) ON DELETE CASCADE
        )
        """
    )

    legacy_chat_ids = [
        row["session_key"]
        for row in connection.execute(
            "SELECT DISTINCT session_key FROM chat_messages_legacy ORDER BY session_key"
        ).fetchall()
    ]

    for chat_id in legacy_chat_ids:
        connection.execute(
            "INSERT OR IGNORE INTO chats (id, title) VALUES (?, ?)",
            (chat_id, DEFAULT_CHAT_TITLE),
        )

    connection.execute(
        """
        INSERT INTO chat_messages (chat_id, role, content, created_at)
        SELECT session_key, role, content, created_at
        FROM chat_messages_legacy
        ORDER BY id ASC
        """
    )
    connection.execute("DROP TABLE chat_messages_legacy")


def create_chat(title: str = DEFAULT_CHAT_TITLE) -> str:
    initialize_chat_history_store()
    chat_id = str(uuid.uuid4())
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO chats (id, title)
            VALUES (?, ?)
            """,
            (chat_id, title),
        )
    return chat_id


def list_chats() -> List[Dict[str, Any]]:
    initialize_chat_history_store()
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT
                c.id,
                c.title,
                c.updated_at,
                COUNT(m.id) AS message_count
            FROM chats c
            LEFT JOIN chat_messages m ON m.chat_id = c.id
            GROUP BY c.id, c.title, c.updated_at
            ORDER BY c.updated_at DESC, c.created_at DESC
            """
        ).fetchall()

    return [
        {
            "id": row["id"],
            "title": row["title"],
            "updated_at": row["updated_at"],
            "message_count": int(row["message_count"]),
        }
        for row in rows
    ]


def get_latest_chat_id() -> Optional[str]:
    chats = list_chats()
    if not chats:
        return None
    return chats[0]["id"]


def load_chat_messages(chat_id: str) -> List[Dict[str, str]]:
    initialize_chat_history_store()
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT role, content
            FROM chat_messages
            WHERE chat_id = ?
            ORDER BY id ASC
            """,
            (chat_id,),
        ).fetchall()

    return [{"role": row["role"], "content": row["content"]} for row in rows]


def append_chat_message(chat_id: str, role: str, content: str) -> None:
    initialize_chat_history_store()
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO chat_messages (chat_id, role, content)
            VALUES (?, ?, ?)
            """,
            (chat_id, role, content),
        )
        connection.execute(
            "UPDATE chats SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (chat_id,),
        )


def clear_chat_messages(chat_id: str) -> None:
    initialize_chat_history_store()
    with _connect() as connection:
        connection.execute(
            "DELETE FROM chat_messages WHERE chat_id = ?",
            (chat_id,),
        )
        connection.execute(
            "UPDATE chats SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (chat_id,),
        )


def set_chat_title(chat_id: str, title: str) -> None:
    initialize_chat_history_store()
    clean_title = title.strip()[:80] or DEFAULT_CHAT_TITLE
    with _connect() as connection:
        connection.execute(
            "UPDATE chats SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (clean_title, chat_id),
        )