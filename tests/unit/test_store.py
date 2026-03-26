"""
Tests for the Store pillar implementations.

The Store pillar handles conversation persistence with three implementations:
InMemory, File, and SQLite. We test incrementally: abstract base class first,
then each implementation to catch issues early.
"""

import sqlite3
import tempfile
from pathlib import Path
from typing import Optional

import pytest
from chatnificent.models import Conversation
from chatnificent.store import File, InMemory, SQLite, Store


class TestStoreInterface:
    """Test the Store abstract base class interface."""

    def test_store_is_abstract(self):
        """Test that Store cannot be instantiated directly."""
        with pytest.raises(TypeError) as exc_info:
            Store()

        error_message = str(exc_info.value)
        assert "abstract" in error_message.lower()

    def test_store_requires_all_methods(self):
        """Test that Store subclasses must implement all required methods."""

        class IncompleteStore1(Store):
            def save_conversation(self, user_id: str, conversation: Conversation):
                pass

            def list_conversations(self, user_id: str):
                return []

        with pytest.raises(TypeError) as exc_info:
            IncompleteStore1()

        error_message = str(exc_info.value)
        assert "load_conversation" in error_message

        class IncompleteStore2(Store):
            def load_conversation(
                self, user_id: str, convo_id: str
            ) -> Optional[Conversation]:
                return None

            def list_conversations(self, user_id: str):
                return []

        with pytest.raises(TypeError) as exc_info:
            IncompleteStore2()

        error_message = str(exc_info.value)
        assert "save_conversation" in error_message

        class IncompleteStore3(Store):
            def load_conversation(
                self, user_id: str, convo_id: str
            ) -> Optional[Conversation]:
                return None

            def save_conversation(self, user_id: str, conversation: Conversation):
                pass

        with pytest.raises(TypeError) as exc_info:
            IncompleteStore3()

        error_message = str(exc_info.value)
        assert "list_conversations" in error_message

    def test_store_subclass_with_all_methods_works(self):
        """Test that complete Store subclasses work correctly."""

        class CustomStore(Store):
            def __init__(self):
                self._data = {}

            def load_conversation(
                self, user_id: str, convo_id: str
            ) -> Optional[Conversation]:
                return self._data.get(f"{user_id}:{convo_id}")

            def save_conversation(self, user_id: str, conversation: Conversation):
                self._data[f"{user_id}:{conversation.id}"] = conversation

            def list_conversations(self, user_id: str):
                return [
                    key.split(":")[1]
                    for key in self._data.keys()
                    if key.startswith(f"{user_id}:")
                ]

        store = CustomStore()
        assert isinstance(store, Store)

        conversation = Conversation(
            id="test_conv", messages=[{"role": "user", "content": "Hello"}]
        )

        store.save_conversation("user1", conversation)
        loaded = store.load_conversation("user1", "test_conv")
        assert loaded is not None
        assert loaded.id == "test_conv"
        assert len(loaded.messages) == 1
        assert loaded.messages[0]["content"] == "Hello"

        conversations = store.list_conversations("user1")
        assert "test_conv" in conversations

    def test_store_optional_file_methods_default_to_noop(self):
        """Optional file helpers should not force custom stores to implement them."""

        class CustomStore(Store):
            def load_conversation(
                self, user_id: str, convo_id: str
            ) -> Optional[Conversation]:
                return None

            def save_conversation(self, user_id: str, conversation: Conversation):
                pass

            def list_conversations(self, user_id: str):
                return []

        store = CustomStore()

        assert store.load_file("user1", "conv1", "notes.txt") is None
        assert store.list_files("user1", "conv1") == []
        assert store.load_raw_api_requests("user1", "conv1") == []
        assert store.load_raw_api_responses("user1", "conv1") == []
        assert store.save_file("user1", "conv1", "notes.txt", b"hello") is None
        assert store.save_raw_api_request("user1", "conv1", {"hello": "world"}) is None
        assert store.save_raw_api_response("user1", "conv1", {"hello": "world"}) is None


class TestFile:
    """Test the File store implementation specifically."""

    def test_file_creation_with_base_dir(self):
        """Test File store can be created with base directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = File(temp_dir)
            assert isinstance(store, Store)
            assert isinstance(store, File)
            assert store.base_dir == Path(temp_dir)

    def test_file_creates_base_directory(self):
        """Test File store creates base directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            non_existent_dir = Path(temp_dir) / "conversations" / "nested"

            store = File(str(non_existent_dir))

            assert non_existent_dir.exists()
            assert non_existent_dir.is_dir()

    def test_file_initial_state(self):
        """Test File store starts empty."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = File(temp_dir)

            # Should have no conversations initially
            conversations = store.list_conversations("any_user")
            assert conversations == []

            # Loading non-existent conversation should return None
            result = store.load_conversation("user1", "nonexistent")
            assert result is None

    def test_file_save_and_load_single_conversation(self):
        """Test saving and loading a single conversation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = File(temp_dir)

            # Create test conversation
            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
            conversation = Conversation(id="conv_001", messages=messages)

            # Save conversation
            store.save_conversation("user1", conversation)

            # Check file structure was created
            user_dir = Path(temp_dir) / "user1"
            conv_dir = user_dir / "conv_001"
            messages_file = conv_dir / "messages.json"

            assert user_dir.exists()
            assert conv_dir.exists()
            assert messages_file.exists()

            # Load it back
            loaded = store.load_conversation("user1", "conv_001")
            assert loaded is not None
            assert loaded.id == "conv_001"
            assert len(loaded.messages) == 2
            assert loaded.messages[0]["role"] == "user"
            assert loaded.messages[0]["content"] == "Hello"
            assert loaded.messages[1]["role"] == "assistant"
            assert loaded.messages[1]["content"] == "Hi there!"

    def test_file_user_isolation(self):
        """Test that different users' conversations are properly isolated."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = File(temp_dir)

            # Create conversations for different users with same conversation ID
            conv1 = Conversation(
                id="conv_001",
                messages=[{"role": "user", "content": "User1 message"}],
            )
            conv2 = Conversation(
                id="conv_001",
                messages=[{"role": "user", "content": "User2 message"}],
            )

            store.save_conversation("user1", conv1)
            store.save_conversation("user2", conv2)

            # Each user should load their own conversation
            loaded1 = store.load_conversation("user1", "conv_001")
            loaded2 = store.load_conversation("user2", "conv_001")

            assert loaded1 is not None
            assert loaded2 is not None
            assert loaded1.messages[0]["content"] == "User1 message"
            assert loaded2.messages[0]["content"] == "User2 message"

            # Each user should only see their own conversations
            convs1 = store.list_conversations("user1")
            convs2 = store.list_conversations("user2")

            assert convs1 == ["conv_001"]
            assert convs2 == ["conv_001"]

            # Check file system structure
            user1_dir = Path(temp_dir) / "user1" / "conv_001"
            user2_dir = Path(temp_dir) / "user2" / "conv_001"

            assert user1_dir.exists()
            assert user2_dir.exists()
            assert user1_dir != user2_dir  # Different directories

    def test_file_list_conversations_ordering(self):
        """Test that list_conversations returns conversations in correct order."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = File(temp_dir)

            # Create conversations with different timestamps
            convs = [
                Conversation(id="001", messages=[{"role": "user", "content": "First"}]),
                Conversation(
                    id="002", messages=[{"role": "user", "content": "Second"}]
                ),
                Conversation(id="003", messages=[{"role": "user", "content": "Third"}]),
            ]

            for conv in convs:
                store.save_conversation("user1", conv)

            conversations = store.list_conversations("user1")
            assert conversations[0] == "003"  # Most recent
            assert conversations[-1] == "001"  # Oldest

    def test_file_save_load_and_list_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = File(temp_dir)

            store.save_file("user1", "conv_001", "notes.txt", b"hello")

            assert store.load_file("user1", "conv_001", "notes.txt") == b"hello"
            assert store.list_files("user1", "conv_001") == ["notes.txt"]

    def test_file_append_file_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = File(temp_dir)

            store.save_file("user1", "conv_001", "notes.txt", b"hello")
            store.save_file(
                "user1",
                "conv_001",
                "notes.txt",
                b" world",
                append=True,
            )

            assert store.load_file("user1", "conv_001", "notes.txt") == b"hello world"

    def test_file_list_files_excludes_messages_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = File(temp_dir)
            conversation = Conversation(
                id="conv_001",
                messages=[{"role": "user", "content": "Hello"}],
            )

            store.save_conversation("user1", conversation)
            store.save_file("user1", "conv_001", "notes.txt", b"hello")

            assert store.list_files("user1", "conv_001") == ["notes.txt"]

    class TestPathTraversal:
        """File store must reject path traversal attacks in user_id and convo_id."""

        def test_reject_dotdot_in_user_id(self):
            with tempfile.TemporaryDirectory() as d:
                store = File(d)
                conv = Conversation(
                    id="c1", messages=[{"role": "user", "content": "x"}]
                )
                with pytest.raises(ValueError, match="path traversal"):
                    store.save_conversation("../etc", conv)

        def test_reject_dotdot_in_convo_id(self):
            with tempfile.TemporaryDirectory() as d:
                store = File(d)
                conv = Conversation(
                    id="../../passwd", messages=[{"role": "user", "content": "x"}]
                )
                with pytest.raises(ValueError, match="path traversal"):
                    store.save_conversation("user1", conv)

        def test_reject_slash_in_ids(self):
            with tempfile.TemporaryDirectory() as d:
                store = File(d)
                conv = Conversation(
                    id="foo/bar", messages=[{"role": "user", "content": "x"}]
                )
                with pytest.raises(ValueError, match="path traversal"):
                    store.save_conversation("user1", conv)

        def test_reject_backslash_in_ids(self):
            with tempfile.TemporaryDirectory() as d:
                store = File(d)
                conv = Conversation(
                    id="foo\\bar", messages=[{"role": "user", "content": "x"}]
                )
                with pytest.raises(ValueError, match="path traversal"):
                    store.save_conversation("user1", conv)

        def test_reject_null_bytes(self):
            with tempfile.TemporaryDirectory() as d:
                store = File(d)
                conv = Conversation(
                    id="valid", messages=[{"role": "user", "content": "x"}]
                )
                with pytest.raises(ValueError, match="path traversal"):
                    store.save_conversation("user\x00evil", conv)

        def test_reject_traversal_on_load(self):
            with tempfile.TemporaryDirectory() as d:
                store = File(d)
                with pytest.raises(ValueError, match="path traversal"):
                    store.load_conversation("../etc", "shadow")

        def test_reject_traversal_on_list(self):
            with tempfile.TemporaryDirectory() as d:
                store = File(d)
                with pytest.raises(ValueError, match="path traversal"):
                    store.list_conversations("../etc")

        def test_reject_traversal_in_filename(self):
            with tempfile.TemporaryDirectory() as d:
                store = File(d)
                with pytest.raises(ValueError, match="path traversal"):
                    store.save_file("user1", "conv1", "../notes.txt", b"x")

        def test_normal_ids_still_work(self):
            with tempfile.TemporaryDirectory() as d:
                store = File(d)
                conv = Conversation(
                    id="a1b2c3d4", messages=[{"role": "user", "content": "hello"}]
                )
                store.save_conversation("user123", conv)
                loaded = store.load_conversation("user123", "a1b2c3d4")
                assert loaded is not None
                assert loaded.id == "a1b2c3d4"

        def test_uuid_style_ids_work(self):
            with tempfile.TemporaryDirectory() as d:
                store = File(d)
                conv = Conversation(
                    id="f8c781b3", messages=[{"role": "user", "content": "hello"}]
                )
                store.save_conversation("e4a9b2c1", conv)
                loaded = store.load_conversation("e4a9b2c1", "f8c781b3")
                assert loaded is not None

        def test_reject_absolute_path_as_id(self):
            with tempfile.TemporaryDirectory() as d:
                store = File(d)
                conv = Conversation(
                    id="/etc/passwd", messages=[{"role": "user", "content": "x"}]
                )
                with pytest.raises(ValueError, match="path traversal"):
                    store.save_conversation("user1", conv)

        def test_resolved_path_must_stay_inside_base_dir(self):
            """Defense in depth: even if segment validation is bypassed,
            the resolved path must stay inside base_dir."""
            with tempfile.TemporaryDirectory() as d:
                store = File(d)
                with pytest.raises(ValueError, match="path traversal"):
                    store.save_conversation(
                        "..", Conversation(id="escape", messages=[])
                    )


class TestSQLite:
    """Test the SQLite store implementation specifically."""

    def test_sqlite_creation_with_db_path(self):
        """Test SQLite store can be created with database path."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_db:
            temp_db_path = temp_db.name

        try:
            store = SQLite(temp_db_path)
            assert isinstance(store, Store)
            assert isinstance(store, SQLite)
            assert store.db_path == temp_db_path
        finally:
            # Clean up
            Path(temp_db_path).unlink(missing_ok=True)

    def test_sqlite_creates_tables_on_init(self):
        """Test SQLite store creates required tables on initialization."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_db:
            temp_db_path = temp_db.name

        try:
            store = SQLite(temp_db_path)

            # Check that tables were created
            conn = sqlite3.connect(temp_db_path)
            try:
                cursor = conn.cursor()

                # Check users table
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='users'"
                )
                assert cursor.fetchone() is not None

                # Check conversations table
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='conversations'"
                )
                assert cursor.fetchone() is not None

                # Check messages table
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='messages'"
                )
                assert cursor.fetchone() is not None

                # Check raw_api_responses table
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='raw_api_responses'"
                )
                assert cursor.fetchone() is not None

                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='files'"
                )
                assert cursor.fetchone() is not None
            finally:
                conn.close()
        finally:
            Path(temp_db_path).unlink(missing_ok=True)

    def test_sqlite_initial_state(self):
        """Test SQLite store starts empty."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_db:
            temp_db_path = temp_db.name

        try:
            store = SQLite(temp_db_path)

            # Should have no conversations initially
            conversations = store.list_conversations("any_user")
            assert conversations == []

            # Loading non-existent conversation should return None
            result = store.load_conversation("user1", "nonexistent")
            assert result is None
        finally:
            Path(temp_db_path).unlink(missing_ok=True)

    def test_sqlite_save_and_load_single_conversation(self):
        """Test saving and loading a single conversation."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_db:
            temp_db_path = temp_db.name

        try:
            store = SQLite(temp_db_path)

            # Create test conversation
            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
            conversation = Conversation(id="conv_001", messages=messages)

            # Save conversation
            store.save_conversation("user1", conversation)

            # Load it back
            loaded = store.load_conversation("user1", "conv_001")
            assert loaded is not None
            assert loaded.id == "conv_001"
            assert len(loaded.messages) == 2
            assert loaded.messages[0]["role"] == "user"
            assert loaded.messages[0]["content"] == "Hello"
            assert loaded.messages[1]["role"] == "assistant"
            assert loaded.messages[1]["content"] == "Hi there!"
        finally:
            Path(temp_db_path).unlink(missing_ok=True)

    def test_sqlite_user_isolation(self):
        """Test that different users' conversations are properly isolated."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_db:
            temp_db_path = temp_db.name

        try:
            store = SQLite(temp_db_path)

            # Create conversations for different users with same conversation ID
            conv1 = Conversation(
                id="conv_001",
                messages=[{"role": "user", "content": "User1 message"}],
            )
            conv2 = Conversation(
                id="conv_001",
                messages=[{"role": "user", "content": "User2 message"}],
            )

            store.save_conversation("user1", conv1)
            store.save_conversation("user2", conv2)

            # Each user should load their own conversation
            loaded1 = store.load_conversation("user1", "conv_001")
            loaded2 = store.load_conversation("user2", "conv_001")

            assert loaded1 is not None
            assert loaded2 is not None
            assert loaded1.messages[0]["content"] == "User1 message"
            assert loaded2.messages[0]["content"] == "User2 message"

            # Each user should only see their own conversations
            convs1 = store.list_conversations("user1")
            convs2 = store.list_conversations("user2")

            assert convs1 == ["conv_001"]
            assert convs2 == ["conv_001"]
        finally:
            Path(temp_db_path).unlink(missing_ok=True)

    def test_sqlite_list_conversations_ordering(self):
        """Test that list_conversations returns conversations in correct order."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_db:
            temp_db_path = temp_db.name

        try:
            store = SQLite(temp_db_path)

            # Create conversations with different content
            convs = [
                Conversation(id="001", messages=[{"role": "user", "content": "First"}]),
                Conversation(
                    id="002", messages=[{"role": "user", "content": "Second"}]
                ),
                Conversation(id="003", messages=[{"role": "user", "content": "Third"}]),
            ]

            for conv in convs:
                store.save_conversation("user1", conv)

            # Should be sorted by updated_at, descending (latest first)
            conversations = store.list_conversations("user1")
            assert conversations[0] == "003"  # Most recent
            assert conversations[-1] == "001"  # Oldest
        finally:
            Path(temp_db_path).unlink(missing_ok=True)

    def test_sqlite_timestamp_debugging(self):
        """Debug test to see what's happening with SQLite timestamps."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_db:
            temp_db_path = temp_db.name

        try:
            store = SQLite(temp_db_path)

            convs = [
                Conversation(id="001", messages=[{"role": "user", "content": "First"}]),
                Conversation(
                    id="002", messages=[{"role": "user", "content": "Second"}]
                ),
                Conversation(id="003", messages=[{"role": "user", "content": "Third"}]),
            ]

            for conv in convs:
                store.save_conversation("user1", conv)

            # Check what's actually in the database
            conn = sqlite3.connect(temp_db_path)
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT conversation_id, created_at, updated_at 
                    FROM conversations 
                    WHERE user_id = 'user1' 
                    ORDER BY updated_at DESC
                """)
                rows = cursor.fetchall()
                print(f"\nDatabase contents (ordered by updated_at DESC):")
                for row in rows:
                    print(f"  {row}")
            finally:
                conn.close()

            # Get the order from list_conversations
            conversations = store.list_conversations("user1")
            print(f"list_conversations returned: {conversations}")

            assert len(conversations) == 3
            assert set(conversations) == {"001", "002", "003"}
        finally:
            Path(temp_db_path).unlink(missing_ok=True)

    def test_sqlite_save_load_and_list_files(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_db:
            temp_db_path = temp_db.name

        try:
            store = SQLite(temp_db_path)

            store.save_file("user1", "conv_001", "notes.txt", b"hello")

            assert store.load_file("user1", "conv_001", "notes.txt") == b"hello"
            assert store.list_files("user1", "conv_001") == ["notes.txt"]
        finally:
            Path(temp_db_path).unlink(missing_ok=True)

    def test_sqlite_append_file_data(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_db:
            temp_db_path = temp_db.name

        try:
            store = SQLite(temp_db_path)

            store.save_file("user1", "conv_001", "notes.txt", b"hello")
            store.save_file(
                "user1",
                "conv_001",
                "notes.txt",
                b" world",
                append=True,
            )

            assert store.load_file("user1", "conv_001", "notes.txt") == b"hello world"
        finally:
            Path(temp_db_path).unlink(missing_ok=True)

    def test_sqlite_conversation_updates(self):
        """Test updating existing conversations in SQLite."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_db:
            temp_db_path = temp_db.name

        try:
            store = SQLite(temp_db_path)
            # Create initial conversation
            original = Conversation(
                id="conv_001",
                messages=[{"role": "user", "content": "Original message"}],
            )
            store.save_conversation("user1", original)

            # Update with more messages
            updated = Conversation(
                id="conv_001",
                messages=[
                    {"role": "user", "content": "Original message"},
                    {"role": "assistant", "content": "Assistant response"},
                    {"role": "user", "content": "Follow-up message"},
                ],
            )
            store.save_conversation("user1", updated)

            # Load should return updated version
            loaded = store.load_conversation("user1", "conv_001")
            assert loaded is not None
            assert len(loaded.messages) == 3
            assert loaded.messages[-1]["content"] == "Follow-up message"

            # Should still be only one conversation in list
            conversations = store.list_conversations("user1")
            assert len(conversations) == 1
            assert conversations[0] == "conv_001"
        finally:
            Path(temp_db_path).unlink(missing_ok=True)

    def test_sqlite_full_message_roundtrip(self, tmp_path):
        """Verify that all message dict keys survive a save/load round-trip.

        Before ``message_data`` was added, only ``role`` and ``content``
        were persisted, silently dropping keys like ``tool_calls``.
        """
        store = SQLite(db_path=str(tmp_path / "roundtrip.db"))
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expr":"2+2"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_abc",
                "name": "calculator",
                "content": "4",
            },
            {"role": "assistant", "content": "2 + 2 = 4"},
        ]
        convo = Conversation(id="round", messages=messages)
        store.save_conversation("u1", convo)
        loaded = store.load_conversation("u1", "round")
        assert loaded is not None
        assert loaded.messages == messages


# InMemory tests - documents current single-user behavior (NO user isolation)
class TestInMemory:
    """Test the InMemory store implementation specifically.

    NOTE: These tests document the current InMemory behavior which does NOT
    provide user isolation. This is intentional for single-user demos/prototypes.
    The user_isolation test explicitly verifies this shared behavior.
    """

    def test_inmemory_creation(self):
        """Test InMemory store can be created and inherits from Store."""
        store = InMemory()
        assert isinstance(store, Store)
        assert isinstance(store, InMemory)

    def test_inmemory_initial_state(self):
        """Test InMemory store starts empty and handles non-existent users gracefully."""
        store = InMemory()

        # Should return empty/None for non-existent users (graceful like dict.get())
        assert store.list_conversations("nonexistent_user") == []
        assert store.load_conversation("nonexistent_user", "any_convo") is None

    def test_inmemory_save_and_load_single_conversation(self):
        """Test saving and loading a single conversation."""
        store = InMemory()
        store._store["user1"] = {}  # Create user namespace

        # Create test conversation
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        conversation = Conversation(id="conv_001", messages=messages)

        store.save_conversation("user1", conversation)

        loaded = store.load_conversation("user1", "conv_001")
        assert loaded is not None
        assert loaded.id == "conv_001"
        assert len(loaded.messages) == 2
        assert loaded.messages[0]["role"] == "user"
        assert loaded.messages[0]["content"] == "Hello"
        assert loaded.messages[1]["role"] == "assistant"
        assert loaded.messages[1]["content"] == "Hi there!"

    def test_inmemory_save_creates_deep_copy(self):
        """Test that save_conversation creates a deep copy to prevent mutations."""
        store = InMemory()
        store._store["user1"] = {}  # Create user namespace

        messages = [{"role": "user", "content": "Original content"}]
        conversation = Conversation(id="conv_001", messages=messages)

        store.save_conversation("user1", conversation)

        conversation.messages[0]["content"] = "Modified content"
        conversation.messages.append({"role": "assistant", "content": "New message"})

        loaded = store.load_conversation("user1", "conv_001")
        assert loaded is not None
        assert len(loaded.messages) == 1  # Still only original message
        assert loaded.messages[0]["content"] == "Original content"

    def test_inmemory_multiple_conversations_same_user(self):
        """Test storing multiple conversations for the same user."""
        store = InMemory()
        store._store["user1"] = {}  # Create user namespace

        conv1 = Conversation(
            id="conv_001",
            messages=[{"role": "user", "content": "First conversation"}],
        )
        conv2 = Conversation(
            id="conv_002",
            messages=[{"role": "user", "content": "Second conversation"}],
        )

        store.save_conversation("user1", conv1)
        store.save_conversation("user1", conv2)

        loaded1 = store.load_conversation("user1", "conv_001")
        loaded2 = store.load_conversation("user1", "conv_002")

        assert loaded1 is not None
        assert loaded2 is not None
        assert loaded1.messages[0]["content"] == "First conversation"
        assert loaded2.messages[0]["content"] == "Second conversation"

        conversations = store.list_conversations("user1")
        assert len(conversations) == 2
        assert "conv_001" in conversations
        assert "conv_002" in conversations

    def test_inmemory_multiple_users_isolated(self):
        """Test that different users' conversations are properly isolated."""
        store = InMemory()

        conv1 = Conversation(
            id="conv_001", messages=[{"role": "user", "content": "User1 message"}]
        )
        conv2 = Conversation(
            id="conv_001",
            messages=[{"role": "user", "content": "User2 message"}],
        )

        # Create user namespaces by saving conversations
        store._store["user1"] = {}
        store._store["user2"] = {}

        store.save_conversation("user1", conv1)
        store.save_conversation("user2", conv2)

        loaded1 = store.load_conversation("user1", "conv_001")
        loaded2 = store.load_conversation("user2", "conv_001")

        # Users should get their own conversations
        assert loaded1 is not None
        assert loaded2 is not None
        assert loaded1.messages[0]["content"] == "User1 message"
        assert loaded2.messages[0]["content"] == "User2 message"

        # List conversations should be per-user
        convs1 = store.list_conversations("user1")
        convs2 = store.list_conversations("user2")

        # Each user sees only their own conversations
        assert convs1 == ["conv_001"]
        assert convs2 == ["conv_001"]
        assert loaded1 != loaded2  # Different conversation objects

    def test_inmemory_list_conversations(self):
        """Test that list_conversations returns all conversation IDs."""
        store = InMemory()

        convs = [
            Conversation(id="abc", messages=[{"role": "user", "content": "First"}]),
            Conversation(id="def", messages=[{"role": "user", "content": "Second"}]),
            Conversation(id="ghi", messages=[{"role": "user", "content": "Third"}]),
        ]

        for conv in convs:
            store.save_conversation("user1", conv)

        conversations = store.list_conversations("user1")
        assert set(conversations) == {"abc", "def", "ghi"}

    def test_inmemory_save_load_and_list_files(self):
        store = InMemory()

        store.save_file("user1", "conv_001", "notes.txt", b"hello")

        assert store.load_file("user1", "conv_001", "notes.txt") == b"hello"
        assert store.list_files("user1", "conv_001") == ["notes.txt"]

    def test_inmemory_append_file_data(self):
        store = InMemory()

        store.save_file("user1", "conv_001", "notes.txt", b"hello")
        store.save_file("user1", "conv_001", "notes.txt", b" world", append=True)

        assert store.load_file("user1", "conv_001", "notes.txt") == b"hello world"

    def test_inmemory_update_existing_conversation(self):
        """Test updating an existing conversation."""
        store = InMemory()
        store._store["user1"] = {}  # Create user namespace

        original = Conversation(
            id="conv_001",
            messages=[{"role": "user", "content": "Original message"}],
        )
        store.save_conversation("user1", original)

        updated = Conversation(
            id="conv_001",
            messages=[
                {"role": "user", "content": "Original message"},
                {"role": "assistant", "content": "Assistant response"},
                {"role": "user", "content": "Follow-up message"},
            ],
        )
        store.save_conversation("user1", updated)

        loaded = store.load_conversation("user1", "conv_001")
        assert loaded is not None
        assert len(loaded.messages) == 3
        assert loaded.messages[-1]["content"] == "Follow-up message"

        conversations = store.list_conversations("user1")
        assert len(conversations) == 1

    def test_inmemory_save_without_prior_namespace(self):
        """save_conversation should work even if the user namespace doesn't exist yet."""
        store = InMemory()
        convo = Conversation(
            id="conv_001",
            messages=[{"role": "user", "content": "Hello"}],
        )
        store.save_conversation("new_user", convo)

        loaded = store.load_conversation("new_user", "conv_001")
        assert loaded is not None
        assert loaded.messages[0]["content"] == "Hello"
