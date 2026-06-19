"""
app/db/chat_store.py
=====================
SQLite-backed chat history store.
Zero dependencies beyond Python stdlib — single .db file on disk.

Schema:
    conversations — one row per chat session (id, title, created_at, mode)
    messages      — one row per message (id, conversation_id, role, content, metadata_json)
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

DB_PATH = Path("data/chat_history.db")


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist."""
    with _conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id          TEXT PRIMARY KEY,
                title       TEXT NOT NULL DEFAULT 'New Chat',
                mode        TEXT NOT NULL DEFAULT 'chat',
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role            TEXT NOT NULL,
                content         TEXT NOT NULL,
                metadata        TEXT,
                created_at      TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            );
        """)


# ------------------------------------------------------------------ #
# Conversations
# ------------------------------------------------------------------ #

def new_conversation(mode: str = "chat") -> str:
    """Create a new conversation, return its ID."""
    cid = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    with _conn() as conn:
        conn.execute(
            "INSERT INTO conversations (id, title, mode, created_at, updated_at) VALUES (?,?,?,?,?)",
            (cid, "New Chat", mode, now, now),
        )
    return cid


def update_conversation_title(cid: str, title: str) -> None:
    """Set conversation title from first user message (truncated)."""
    short = title[:60] + "…" if len(title) > 60 else title
    now = datetime.utcnow().isoformat()
    with _conn() as conn:
        conn.execute(
            "UPDATE conversations SET title=?, updated_at=? WHERE id=?",
            (short, now, cid),
        )


def touch_conversation(cid: str) -> None:
    now = datetime.utcnow().isoformat()
    with _conn() as conn:
        conn.execute("UPDATE conversations SET updated_at=? WHERE id=?", (now, cid))


def delete_conversation(cid: str) -> None:
    with _conn() as conn:
        conn.execute("DELETE FROM messages WHERE conversation_id=?", (cid,))
        conn.execute("DELETE FROM conversations WHERE id=?", (cid,))


def list_conversations(limit: int = 40) -> list[dict]:
    """Return conversations newest-first."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT id, title, mode, created_at, updated_at FROM conversations ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


# ------------------------------------------------------------------ #
# Messages
# ------------------------------------------------------------------ #

def add_message(
    cid: str,
    role: str,
    content: str,
    metadata: Optional[dict] = None,
) -> None:
    """Append a message to a conversation."""
    now = datetime.utcnow().isoformat()
    meta_str = json.dumps(metadata) if metadata else None
    with _conn() as conn:
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, metadata, created_at) VALUES (?,?,?,?,?)",
            (cid, role, content, meta_str, now),
        )
    touch_conversation(cid)


def get_messages(cid: str) -> list[dict]:
    """Return all messages for a conversation, oldest-first."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT role, content, metadata, created_at FROM messages WHERE conversation_id=? ORDER BY id ASC",
            (cid,),
        ).fetchall()
    result = []
    for r in rows:
        msg = dict(r)
        msg["metadata"] = json.loads(msg["metadata"]) if msg["metadata"] else {}
        result.append(msg)
    return result


def group_conversations_by_date(conversations: list[dict]) -> dict[str, list[dict]]:
    """Group conversation list into Today / Yesterday / Earlier."""
    from datetime import date, timedelta
    today = date.today()
    yesterday = today - timedelta(days=1)
    week_ago = today - timedelta(days=7)

    groups: dict[str, list] = {"Today": [], "Yesterday": [], "This Week": [], "Earlier": []}
    for c in conversations:
        try:
            d = datetime.fromisoformat(c["updated_at"]).date()
        except Exception:
            d = today

        if d == today:
            groups["Today"].append(c)
        elif d == yesterday:
            groups["Yesterday"].append(c)
        elif d >= week_ago:
            groups["This Week"].append(c)
        else:
            groups["Earlier"].append(c)

    return {k: v for k, v in groups.items() if v}   # drop empty groups


# Initialise on import
init_db()
