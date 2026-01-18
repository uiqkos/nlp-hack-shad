import json
import sqlite3
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).parent / "chat_data.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database tables."""
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_summaries (
            chat_id INTEGER PRIMARY KEY,
            last_message_id INTEGER DEFAULT 0,
            summary TEXT DEFAULT '{}',
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def get_chat_data(chat_id: int) -> dict[str, Any]:
    """Get summary data for a chat."""
    conn = get_connection()
    row = conn.execute(
        "SELECT summary, last_message_id FROM chat_summaries WHERE chat_id = ?",
        (chat_id,),
    ).fetchone()
    conn.close()

    if row:
        return {
            "summary": json.loads(row["summary"]),
            "last_message_id": row["last_message_id"],
        }
    return {
        "summary": {"overview": "", "problems": [], "decisions": [], "key_points": []},
        "last_message_id": 0,
    }


def save_chat_data(chat_id: int, summary: dict, last_message_id: int):
    """Save or update summary data for a chat."""
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO chat_summaries (chat_id, summary, last_message_id, updated_at)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(chat_id) DO UPDATE SET
            summary = excluded.summary,
            last_message_id = excluded.last_message_id,
            updated_at = CURRENT_TIMESTAMP
    """,
        (chat_id, json.dumps(summary, ensure_ascii=False), last_message_id),
    )
    conn.commit()
    conn.close()


def clear_chat_data(chat_id: int):
    """Clear summary data for a chat."""
    conn = get_connection()
    conn.execute("DELETE FROM chat_summaries WHERE chat_id = ?", (chat_id,))
    conn.commit()
    conn.close()


# Initialize DB on module import
init_db()
