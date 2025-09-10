"""Concrete implementations for persistence managers."""

import json
import logging
import os
import sqlite3
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

from .models import Conversation

logger = logging.getLogger(__name__)


class Store(ABC):
    """Interface for saving and loading conversation data."""

    @abstractmethod
    def load_conversation(self, user_id: str, convo_id: str) -> Optional[Conversation]:
        """Loads a single conversation from the persistence layer."""
        pass

    @abstractmethod
    def save_conversation(self, user_id: str, conversation: Conversation):
        """Saves a single conversation to the persistence layer."""
        pass

    @abstractmethod
    def list_conversations(self, user_id: str) -> List[str]:
        """Lists all conversation IDs for a given user."""
        pass

    @abstractmethod
    def get_next_conversation_id(self, user_id: str) -> str:
        """Generates a new, unique conversation ID for a user."""
        pass


class InMemory(Store):
    """In-memory conversation storage with proper user isolation.

    Stores conversations in memory using user_id/convo_id composite keys to ensure
    complete isolation between users. Each user's conversations are stored separately.

    Features:
    - Full user isolation (users cannot see each other's conversations)
    - Per-user conversation ID generation
    - Raises KeyError if user_id doesn't exist when accessing conversations
    - Suitable for development, testing, and single-process applications

    For persistent storage, use File or SQLite store implementations.
    """

    def __init__(self):
        self._store: Dict[str, Dict[str, Conversation]] = {}

    def load_conversation(self, user_id: str, convo_id: str) -> Optional[Conversation]:
        """Load a conversation. Returns None if user or conversation doesn't exist."""
        return self._store.get(user_id, {}).get(convo_id)

    def save_conversation(self, user_id: str, conversation: Conversation):
        self._store[user_id][conversation.id] = conversation.copy(deep=True)

    def list_conversations(self, user_id: str) -> List[str]:
        """Lists all conversation IDs for a given user. Returns empty list if user doesn't exist."""
        user_conversations = self._store.get(user_id, {})
        return sorted(
            user_conversations.keys(),
            key=lambda x: int(x) if x.isdigit() else 0,
            reverse=True,
        )

    def get_next_conversation_id(self, user_id: str) -> str:
        """Generates a new, unique conversation ID for a user."""
        # Create user namespace if it doesn't exist (for new users)
        if user_id not in self._store:
            self._store[user_id] = {}
        user_conversations = self._store[user_id]
        return f"{len(user_conversations) + 1:03d}"


class File(Store):
    """Saves and loads conversations from the local file system as JSON."""

    def __init__(self, base_dir: str):
        """
        Initialize with mandatory base directory.

        Args:
            base_dir: Directory where user conversations will be stored.
                     No default to prevent unexpected file creation.
        """
        self.base_dir = Path(base_dir)
        self._write_locks: Dict[str, Lock] = {}  # Per-conversation write locks
        self._list_lock = Lock()  # For directory operations

        # Create base directory if it doesn't exist
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_user_dir(self, user_id: str) -> Path:
        """Get user directory path, create if needed."""
        user_dir = self.base_dir / user_id
        user_dir.mkdir(exist_ok=True)
        return user_dir

    def _get_conversation_dir(self, user_id: str, convo_id: str) -> Path:
        """
        Gets the conversation directory path.
        """
        return self._get_user_dir(user_id) / convo_id

    def _get_write_lock(self, user_id: str, convo_id: str) -> Lock:
        """Get or create a write lock for a specific conversation."""
        lock_key = f"{user_id}/{convo_id}"
        if lock_key not in self._write_locks:
            self._write_locks[lock_key] = Lock()
        return self._write_locks[lock_key]

    def _atomic_write_json(self, file_path: Path, data: dict):
        """Write JSON data atomically using temp file + move."""
        # Write to temporary file first
        with tempfile.NamedTemporaryFile(
            mode="w", dir=file_path.parent, delete=False, suffix=".tmp"
        ) as tmp_file:
            json.dump(data, tmp_file, indent=2)
            tmp_name = tmp_file.name

        # Atomic move
        os.replace(tmp_name, file_path)

    def _append_jsonl(self, file_path: Path, data: dict):
        """Append single JSON line to JSONL file."""
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")

    def load_conversation(self, user_id: str, convo_id: str) -> Optional[Conversation]:
        """
        Load conversation from messages.json file.
        """
        try:
            messages_file = (
                self._get_conversation_dir(user_id, convo_id) / "messages.json"
            )

            if not messages_file.exists():
                return None

            with open(messages_file, "r", encoding="utf-8") as f:
                messages_data = json.load(f)

            return Conversation(id=convo_id, messages=messages_data)

        except (json.JSONDecodeError, FileNotFoundError, PermissionError):
            # Log error in production, return None for now
            return None

    def save_conversation(self, user_id: str, conversation: Conversation):
        """
        Save conversation to messages.json with atomic write.
        """
        lock = self._get_write_lock(user_id, conversation.id)

        with lock:
            try:
                convo_dir = self._get_conversation_dir(user_id, conversation.id)
                convo_dir.mkdir(exist_ok=True)
                messages_file = convo_dir / "messages.json"

                messages_data = [msg.model_dump() for msg in conversation.messages]

                self._atomic_write_json(messages_file, messages_data)

            except (PermissionError, OSError) as e:
                raise RuntimeError(
                    f"Failed to save conversation {conversation.id}: {e}"
                )

    def save_raw_api_response(self, user_id: str, convo_id: str, raw_response: dict):
        """Append raw API response to JSONL file."""
        lock = self._get_write_lock(user_id, convo_id)

        with lock:
            try:
                convo_dir = self._get_conversation_dir(user_id, convo_id)
                convo_dir.mkdir(exist_ok=True)
                raw_file = convo_dir / "raw_api_responses.jsonl"

                self._append_jsonl(raw_file, raw_response)

            except (PermissionError, OSError) as e:
                # Log the error - raw API response saving is critical for debugging
                logger.error(f"Failed to save raw API response for conversation {convo_id}: {e}")

    def list_conversations(self, user_id: str) -> List[str]:
        """List all conversation IDs for user by scanning directories."""
        with self._list_lock:  # Prevent concurrent directory reads
            try:
                user_dir = self.base_dir / user_id

                if not user_dir.exists():
                    return []

                conversations = []
                for item in user_dir.iterdir():
                    if item.is_dir() and (item / "messages.json").exists():
                        conversations.append(item.name)

                return sorted(
                    conversations,
                    key=lambda x: (user_dir / x).stat().st_mtime,
                    reverse=True,
                )

            except (PermissionError, OSError):
                return []

    def get_next_conversation_id(self, user_id: str) -> str:
        """Generate next conversation ID by finding highest existing + 1."""
        with self._list_lock:
            user_dir = self._get_user_dir(user_id)  # Creates directory if needed

            conversations = []
            for item in user_dir.iterdir():
                if item.is_dir() and (item / "messages.json").exists():
                    conversations.append(item.name)

            if not conversations:
                return "001"

            # Find highest numeric ID
            highest = 0
            for conv_id in conversations:
                if conv_id.isdigit():
                    highest = max(highest, int(conv_id))

            return f"{highest + 1:03d}"  # Zero-padded 3 digits


class SQLite(Store):
    """Saves and loads conversations using SQLite database."""

    def __init__(self, db_path: str):
        """
        Initialize with mandatory database file path.

        Args:
            db_path: Path to SQLite database file.
                    No default to prevent unexpected file creation.
        """
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Initialize database and create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Enable foreign keys
            cursor.execute("PRAGMA foreign_keys = ON")

            # Create tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    user_id TEXT,
                    conversation_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, conversation_id),
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    user_id TEXT,
                    conversation_id TEXT, 
                    message_index INTEGER,
                    role TEXT,
                    content TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, conversation_id, message_index),
                    FOREIGN KEY (user_id, conversation_id) 
                        REFERENCES conversations(user_id, conversation_id)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS raw_api_responses (
                    user_id TEXT,
                    conversation_id TEXT,
                    response_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id, conversation_id) 
                        REFERENCES conversations(user_id, conversation_id)
                )
            """)

            conn.commit()

    def _ensure_user_exists(self, user_id: str):
        """Ensure user exists in database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR IGNORE INTO users (user_id) VALUES (?)", (user_id,)
            )
            conn.commit()

    def load_conversation(self, user_id: str, convo_id: str) -> Optional[Conversation]:
        """Load conversation from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Get messages for this conversation
                cursor.execute(
                    """
                    SELECT role, content 
                    FROM messages 
                    WHERE user_id = ? AND conversation_id = ?
                    ORDER BY message_index
                """,
                    (user_id, convo_id),
                )

                rows = cursor.fetchall()
                if not rows:
                    return None

                # Convert to ChatMessage objects
                from .models import ChatMessage

                messages = [ChatMessage(role=row[0], content=row[1]) for row in rows]

                return Conversation(id=convo_id, messages=messages)

        except sqlite3.Error:
            return None

    def save_conversation(self, user_id: str, conversation: Conversation):
        """Save conversation to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Ensure user exists
                self._ensure_user_exists(user_id)

                # Insert or update conversation record with millisecond precision timestamps
                cursor.execute(
                    """
                    INSERT INTO conversations (user_id, conversation_id, created_at, updated_at)
                    VALUES (?, ?, datetime('now', 'subsec'), datetime('now', 'subsec'))
                    ON CONFLICT(user_id, conversation_id)
                    DO UPDATE SET updated_at = datetime('now', 'subsec')
                    """,
                    (user_id, conversation.id),
                )

                # Clear existing messages for this conversation
                cursor.execute(
                    """
                    DELETE FROM messages 
                    WHERE user_id = ? AND conversation_id = ?
                """,
                    (user_id, conversation.id),
                )

                # Insert all messages
                for i, message in enumerate(conversation.messages):
                    cursor.execute(
                        """
                        INSERT INTO messages 
                        (user_id, conversation_id, message_index, role, content)
                        VALUES (?, ?, ?, ?, ?)
                    """,
                        (user_id, conversation.id, i, message.role, message.content),
                    )

                conn.commit()

        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to save conversation {conversation.id}: {e}")

    def save_raw_api_response(self, user_id: str, convo_id: str, raw_response: dict):
        """Save raw API response to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT INTO raw_api_responses 
                    (user_id, conversation_id, response_data)
                    VALUES (?, ?, ?)
                """,
                    (user_id, convo_id, json.dumps(raw_response)),
                )

                conn.commit()

        except sqlite3.Error:
            # Non-critical - don't fail the main operation
            pass

    def list_conversations(self, user_id: str) -> List[str]:
        """List all conversation IDs for user from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT conversation_id 
                    FROM conversations 
                    WHERE user_id = ?
                    ORDER BY updated_at DESC
                """,
                    (user_id,),
                )

                return [row[0] for row in cursor.fetchall()]

        except sqlite3.Error:
            return []

    def get_next_conversation_id(self, user_id: str) -> str:
        """Generate next conversation ID from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT conversation_id 
                FROM conversations 
                WHERE user_id = ? AND conversation_id GLOB '[0-9][0-9][0-9]'
                ORDER BY conversation_id DESC 
                LIMIT 1
            """,
                (user_id,),
            )

            row = cursor.fetchone()
            if not row:
                return "001"

            highest = int(row[0])
            return f"{highest + 1:03d}"
