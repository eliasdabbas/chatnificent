"""Concrete implementations for persistence managers."""

import json
import logging
import os
import sqlite3
import tempfile
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

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

    def save_file(
        self,
        user_id: str,
        convo_id: str,
        filename: str,
        data: bytes,
        **kwargs: Any,
    ) -> None:
        """Save a conversation-scoped file.

        Parameters
        ----------
        user_id : str
            User namespace.
        convo_id : str
            Conversation namespace.
        filename : str
            Logical filename within the conversation scope.
        data : bytes
            File content to persist.
        **kwargs : Any
            Backend-specific options such as ``append=True``.
        """
        return None

    def load_file(
        self, user_id: str, convo_id: str, filename: str, **kwargs: Any
    ) -> Optional[bytes]:
        """Load a conversation-scoped file."""
        return None

    def list_files(self, user_id: str, convo_id: str) -> List[str]:
        """List conversation-scoped filenames."""
        return []

    def save_raw_api_request(
        self, user_id: str, convo_id: str, raw_request: Dict[str, Any]
    ) -> None:
        """Persist a raw request payload in JSONL format when supported."""
        self.save_file(
            user_id,
            convo_id,
            "raw_api_requests.jsonl",
            (json.dumps(raw_request) + "\n").encode("utf-8"),
            append=True,
        )

    def save_raw_api_response(
        self, user_id: str, convo_id: str, raw_response: Dict[str, Any] | List[Any]
    ) -> None:
        """Persist a raw response payload in JSONL format when supported."""
        self.save_file(
            user_id,
            convo_id,
            "raw_api_responses.jsonl",
            (json.dumps(raw_response) + "\n").encode("utf-8"),
            append=True,
        )

    def load_raw_api_requests(self, user_id: str, convo_id: str) -> List[Any]:
        """Load raw request payloads from the conversation scope."""
        return self._load_jsonl_file(user_id, convo_id, "raw_api_requests.jsonl")

    def load_raw_api_responses(self, user_id: str, convo_id: str) -> List[Any]:
        """Load raw response payloads from the conversation scope."""
        return self._load_jsonl_file(user_id, convo_id, "raw_api_responses.jsonl")

    def _load_jsonl_file(self, user_id: str, convo_id: str, filename: str) -> List[Any]:
        payload = self.load_file(user_id, convo_id, filename)
        if not payload:
            return []

        items = []
        for line in payload.decode("utf-8").splitlines():
            if not line.strip():
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping invalid JSONL line in %s", filename)
        return items


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
        self._files: Dict[str, Dict[str, Dict[str, bytes]]] = {}

    def load_conversation(self, user_id: str, convo_id: str) -> Optional[Conversation]:
        """Load a conversation. Returns None if user or conversation doesn't exist."""
        return self._store.get(user_id, {}).get(convo_id)

    def save_conversation(self, user_id: str, conversation: Conversation):
        self._store.setdefault(user_id, {})[conversation.id] = conversation.copy(
            deep=True
        )

    def list_conversations(self, user_id: str) -> List[str]:
        """Lists all conversation IDs for a given user. Returns empty list if user doesn't exist."""
        user_conversations = self._store.get(user_id, {})
        return list(user_conversations.keys())

    def save_file(
        self,
        user_id: str,
        convo_id: str,
        filename: str,
        data: bytes,
        **kwargs: Any,
    ) -> None:
        user_files = self._files.setdefault(user_id, {})
        convo_files = user_files.setdefault(convo_id, {})
        if kwargs.get("append"):
            convo_files[filename] = convo_files.get(filename, b"") + data
        else:
            convo_files[filename] = bytes(data)

    def load_file(
        self, user_id: str, convo_id: str, filename: str, **kwargs: Any
    ) -> Optional[bytes]:
        return self._files.get(user_id, {}).get(convo_id, {}).get(filename)

    def list_files(self, user_id: str, convo_id: str) -> List[str]:
        return sorted(self._files.get(user_id, {}).get(convo_id, {}).keys())


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

    def _validate_path_segment(self, segment: str, label: str) -> None:
        """Reject path segments that could escape base_dir."""
        if (
            not segment
            or ".." in segment
            or "/" in segment
            or "\\" in segment
            or "\x00" in segment
        ):
            raise ValueError(f"Unsafe {label} rejected (path traversal): {segment!r}")
        if segment.startswith("/"):
            raise ValueError(f"Unsafe {label} rejected (path traversal): {segment!r}")

    def _get_user_dir(self, user_id: str) -> Path:
        """Get user directory path, create if needed."""
        self._validate_path_segment(user_id, "user_id")
        user_dir = self.base_dir / user_id
        resolved = user_dir.resolve()
        if not str(resolved).startswith(str(self.base_dir.resolve())):
            raise ValueError(f"Unsafe user_id rejected (path traversal): {user_id!r}")
        user_dir.mkdir(exist_ok=True)
        return user_dir

    def _get_conversation_dir(self, user_id: str, convo_id: str) -> Path:
        """Gets the conversation directory path."""
        self._validate_path_segment(convo_id, "convo_id")
        convo_dir = self._get_user_dir(user_id) / convo_id
        resolved = convo_dir.resolve()
        if not str(resolved).startswith(str(self.base_dir.resolve())):
            raise ValueError(f"Unsafe convo_id rejected (path traversal): {convo_id!r}")
        return convo_dir

    def _get_file_path(self, user_id: str, convo_id: str, filename: str) -> Path:
        """Get a validated file path inside the conversation directory."""
        self._validate_path_segment(filename, "filename")
        return self._get_conversation_dir(user_id, convo_id) / filename

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

                self._atomic_write_json(messages_file, conversation.messages)

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
                logger.error(
                    f"Failed to save raw API response for conversation {convo_id}: {e}"
                )

    def save_raw_api_request(self, user_id: str, convo_id: str, raw_request: dict):
        """Append raw API request to JSONL file."""
        lock = self._get_write_lock(user_id, convo_id)

        with lock:
            try:
                convo_dir = self._get_conversation_dir(user_id, convo_id)
                convo_dir.mkdir(exist_ok=True)
                raw_file = convo_dir / "raw_api_requests.jsonl"

                self._append_jsonl(raw_file, raw_request)

            except (PermissionError, OSError) as e:
                logger.error(
                    f"Failed to save raw API request for conversation {convo_id}: {e}"
                )

    def list_conversations(self, user_id: str) -> List[str]:
        """List all conversation IDs for user by scanning directories."""
        self._validate_path_segment(user_id, "user_id")
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

    def save_file(
        self,
        user_id: str,
        convo_id: str,
        filename: str,
        data: bytes,
        **kwargs: Any,
    ) -> None:
        """Save raw bytes into the conversation directory."""
        lock = self._get_write_lock(user_id, convo_id)

        with lock:
            try:
                convo_dir = self._get_conversation_dir(user_id, convo_id)
                convo_dir.mkdir(exist_ok=True)
                file_path = self._get_file_path(user_id, convo_id, filename)
                mode = "ab" if kwargs.get("append") else "wb"
                with open(file_path, mode) as f:
                    f.write(data)
            except (PermissionError, OSError) as e:
                raise RuntimeError(f"Failed to save file {filename} for {convo_id}: {e}")

    def load_file(
        self, user_id: str, convo_id: str, filename: str, **kwargs: Any
    ) -> Optional[bytes]:
        """Load raw bytes from the conversation directory."""
        try:
            file_path = self._get_file_path(user_id, convo_id, filename)
            if not file_path.exists():
                return None
            return file_path.read_bytes()
        except (FileNotFoundError, PermissionError):
            return None

    def list_files(self, user_id: str, convo_id: str) -> List[str]:
        """List auxiliary files stored alongside the canonical conversation."""
        try:
            convo_dir = self._get_conversation_dir(user_id, convo_id)
            if not convo_dir.exists():
                return []
            return sorted(
                item.name
                for item in convo_dir.iterdir()
                if item.is_file() and item.name != "messages.json"
            )
        except (PermissionError, OSError):
            return []


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

    @contextmanager
    def _connect(self):
        """Context manager that commits/rolls back AND closes the connection."""
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_database(self):
        """Initialize database and create tables if they don't exist."""
        with self._connect() as conn:
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
                    message_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, conversation_id, message_index),
                    FOREIGN KEY (user_id, conversation_id) 
                        REFERENCES conversations(user_id, conversation_id)
                )
            """)

            # Migrate: add message_data column to existing databases
            cursor.execute("PRAGMA table_info(messages)")
            columns = {row[1] for row in cursor.fetchall()}
            if "message_data" not in columns:
                cursor.execute("ALTER TABLE messages ADD COLUMN message_data TEXT")

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

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS raw_api_requests (
                    user_id TEXT,
                    conversation_id TEXT,
                    request_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id, conversation_id) 
                        REFERENCES conversations(user_id, conversation_id)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS files (
                    user_id TEXT,
                    conversation_id TEXT,
                    filename TEXT,
                    data BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, conversation_id, filename)
                )
            """)

            conn.commit()

    def _ensure_user_exists(self, user_id: str):
        """Ensure user exists in database."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR IGNORE INTO users (user_id) VALUES (?)", (user_id,)
            )
            conn.commit()

    def load_conversation(self, user_id: str, convo_id: str) -> Optional[Conversation]:
        """Load conversation from database.

        Reads from ``message_data`` (full JSON blob) when available,
        falling back to ``role``/``content`` columns for rows written
        before the migration.
        """
        try:
            with self._connect() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT message_data, role, content
                    FROM messages 
                    WHERE user_id = ? AND conversation_id = ?
                    ORDER BY message_index
                    """,
                    (user_id, convo_id),
                )

                rows = cursor.fetchall()
                if not rows:
                    return None

                messages = []
                for message_data, role, content in rows:
                    if message_data:
                        messages.append(json.loads(message_data))
                    else:
                        messages.append({"role": role, "content": content})

                return Conversation(id=convo_id, messages=messages)

        except sqlite3.Error:
            return None

    def save_conversation(self, user_id: str, conversation: Conversation):
        """Save conversation to database."""
        try:
            with self._connect() as conn:
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

                # Insert all messages with full JSON blob
                for i, message in enumerate(conversation.messages):
                    content = message.get("content", "")
                    if not isinstance(content, str):
                        content = json.dumps(content)
                    cursor.execute(
                        """
                        INSERT INTO messages 
                        (user_id, conversation_id, message_index, role, content, message_data)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            user_id,
                            conversation.id,
                            i,
                            message.get("role", ""),
                            content,
                            json.dumps(message),
                        ),
                    )

                conn.commit()

        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to save conversation {conversation.id}: {e}")

    def save_raw_api_response(self, user_id: str, convo_id: str, raw_response: dict):
        """Save raw API response to database."""
        try:
            with self._connect() as conn:
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

    def save_raw_api_request(self, user_id: str, convo_id: str, raw_request: dict):
        """Save raw API request to database."""
        try:
            with self._connect() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT INTO raw_api_requests 
                    (user_id, conversation_id, request_data)
                    VALUES (?, ?, ?)
                """,
                    (user_id, convo_id, json.dumps(raw_request)),
                )

                conn.commit()

        except sqlite3.Error:
            pass

    def list_conversations(self, user_id: str) -> List[str]:
        """List all conversation IDs for user from database."""
        try:
            with self._connect() as conn:
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

    def save_file(
        self,
        user_id: str,
        convo_id: str,
        filename: str,
        data: bytes,
        **kwargs: Any,
    ) -> None:
        """Save a conversation-scoped file as a BLOB."""
        try:
            with self._connect() as conn:
                cursor = conn.cursor()

                self._ensure_user_exists(user_id)
                cursor.execute(
                    """
                    INSERT INTO conversations (user_id, conversation_id, created_at, updated_at)
                    VALUES (?, ?, datetime('now', 'subsec'), datetime('now', 'subsec'))
                    ON CONFLICT(user_id, conversation_id)
                    DO UPDATE SET updated_at = datetime('now', 'subsec')
                    """,
                    (user_id, convo_id),
                )

                blob = data
                if kwargs.get("append"):
                    cursor.execute(
                        """
                        SELECT data
                        FROM files
                        WHERE user_id = ? AND conversation_id = ? AND filename = ?
                        """,
                        (user_id, convo_id, filename),
                    )
                    row = cursor.fetchone()
                    if row and row[0]:
                        blob = row[0] + data

                cursor.execute(
                    """
                    INSERT INTO files (user_id, conversation_id, filename, data, created_at, updated_at)
                    VALUES (?, ?, ?, ?, datetime('now', 'subsec'), datetime('now', 'subsec'))
                    ON CONFLICT(user_id, conversation_id, filename)
                    DO UPDATE SET data = excluded.data, updated_at = datetime('now', 'subsec')
                    """,
                    (user_id, convo_id, filename, blob),
                )

                conn.commit()
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to save file {filename} for {convo_id}: {e}")

    def load_file(
        self, user_id: str, convo_id: str, filename: str, **kwargs: Any
    ) -> Optional[bytes]:
        """Load a conversation-scoped file from SQLite."""
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT data
                    FROM files
                    WHERE user_id = ? AND conversation_id = ? AND filename = ?
                    """,
                    (user_id, convo_id, filename),
                )
                row = cursor.fetchone()
                if not row:
                    return None
                return row[0]
        except sqlite3.Error:
            return None

    def list_files(self, user_id: str, convo_id: str) -> List[str]:
        """List conversation-scoped filenames stored in SQLite."""
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT filename
                    FROM files
                    WHERE user_id = ? AND conversation_id = ?
                    ORDER BY filename
                    """,
                    (user_id, convo_id),
                )
                return [row[0] for row in cursor.fetchall()]
        except sqlite3.Error:
            return []

    def load_raw_api_requests(self, user_id: str, convo_id: str) -> List[Any]:
        """Load raw API requests from the dedicated SQLite table."""
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT request_data
                    FROM raw_api_requests
                    WHERE user_id = ? AND conversation_id = ?
                    ORDER BY created_at
                    """,
                    (user_id, convo_id),
                )
                return [json.loads(row[0]) for row in cursor.fetchall()]
        except (sqlite3.Error, json.JSONDecodeError):
            return []

    def load_raw_api_responses(self, user_id: str, convo_id: str) -> List[Any]:
        """Load raw API responses from the dedicated SQLite table."""
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT response_data
                    FROM raw_api_responses
                    WHERE user_id = ? AND conversation_id = ?
                    ORDER BY created_at
                    """,
                    (user_id, convo_id),
                )
                return [json.loads(row[0]) for row in cursor.fetchall()]
        except (sqlite3.Error, json.JSONDecodeError):
            return []
