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
from chatnificent.models import ChatMessage, Conversation
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

            def get_next_conversation_id(self, user_id: str) -> str:
                return "001"

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

            def get_next_conversation_id(self, user_id: str) -> str:
                return "001"

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

            def get_next_conversation_id(self, user_id: str) -> str:
                return "001"

        with pytest.raises(TypeError) as exc_info:
            IncompleteStore3()

        error_message = str(exc_info.value)
        assert "list_conversations" in error_message

        class IncompleteStore4(Store):
            def load_conversation(
                self, user_id: str, convo_id: str
            ) -> Optional[Conversation]:
                return None

            def save_conversation(self, user_id: str, conversation: Conversation):
                pass

            def list_conversations(self, user_id: str):
                return []

        with pytest.raises(TypeError) as exc_info:
            IncompleteStore4()

        error_message = str(exc_info.value)
        assert "get_next_conversation_id" in error_message

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

            def get_next_conversation_id(self, user_id: str) -> str:
                return "custom_001"

        store = CustomStore()
        assert isinstance(store, Store)

        conversation = Conversation(
            id="test_conv", messages=[ChatMessage(role="user", content="Hello")]
        )

        store.save_conversation("user1", conversation)
        loaded = store.load_conversation("user1", "test_conv")
        assert loaded is not None
        assert loaded.id == "test_conv"
        assert len(loaded.messages) == 1
        assert loaded.messages[0].content == "Hello"

        conversations = store.list_conversations("user1")
        assert "test_conv" in conversations

        next_id = store.get_next_conversation_id("user1")
        assert next_id == "custom_001"


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

            # Next conversation ID should start at "001"
            next_id = store.get_next_conversation_id("user1")
            assert next_id == "001"

    def test_file_save_and_load_single_conversation(self):
        """Test saving and loading a single conversation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = File(temp_dir)

            # Create test conversation
            messages = [
                ChatMessage(role="user", content="Hello"),
                ChatMessage(role="assistant", content="Hi there!"),
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
            assert loaded.messages[0].role == "user"
            assert loaded.messages[0].content == "Hello"
            assert loaded.messages[1].role == "assistant"
            assert loaded.messages[1].content == "Hi there!"

    def test_file_user_isolation(self):
        """Test that different users' conversations are properly isolated."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = File(temp_dir)

            # Create conversations for different users with same conversation ID
            conv1 = Conversation(
                id="conv_001",
                messages=[ChatMessage(role="user", content="User1 message")],
            )
            conv2 = Conversation(
                id="conv_001",
                messages=[ChatMessage(role="user", content="User2 message")],
            )

            store.save_conversation("user1", conv1)
            store.save_conversation("user2", conv2)

            # Each user should load their own conversation
            loaded1 = store.load_conversation("user1", "conv_001")
            loaded2 = store.load_conversation("user2", "conv_001")

            assert loaded1 is not None
            assert loaded2 is not None
            assert loaded1.messages[0].content == "User1 message"
            assert loaded2.messages[0].content == "User2 message"

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

    def test_file_conversation_id_generation(self):
        """Test conversation ID generation for File store."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = File(temp_dir)

            # First ID should be "001"
            id1 = store.get_next_conversation_id("user1")
            assert id1 == "001"

            # Add a conversation
            conv = Conversation(
                id="001", messages=[ChatMessage(role="user", content="test")]
            )
            store.save_conversation("user1", conv)

            # Next ID should be "002"
            id2 = store.get_next_conversation_id("user1")
            assert id2 == "002"

            # Add non-sequential conversation
            conv3 = Conversation(
                id="005", messages=[ChatMessage(role="user", content="test")]
            )
            store.save_conversation("user1", conv3)

            # Next ID should be "006" (highest + 1)
            id3 = store.get_next_conversation_id("user1")
            assert id3 == "006"

    def test_file_list_conversations_ordering(self):
        """Test that list_conversations returns conversations in correct order."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = File(temp_dir)

            # Create conversations with different timestamps
            convs = [
                Conversation(
                    id="001", messages=[ChatMessage(role="user", content="First")]
                ),
                Conversation(
                    id="002", messages=[ChatMessage(role="user", content="Second")]
                ),
                Conversation(
                    id="003", messages=[ChatMessage(role="user", content="Third")]
                ),
            ]

            for conv in convs:
                store.save_conversation("user1", conv)

            conversations = store.list_conversations("user1")
            assert conversations[0] == "003"  # Most recent
            assert conversations[-1] == "001"  # Oldest


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
            with sqlite3.connect(temp_db_path) as conn:
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

            # Next conversation ID should start at "001"
            next_id = store.get_next_conversation_id("user1")
            assert next_id == "001"
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
                ChatMessage(role="user", content="Hello"),
                ChatMessage(role="assistant", content="Hi there!"),
            ]
            conversation = Conversation(id="conv_001", messages=messages)

            # Save conversation
            store.save_conversation("user1", conversation)

            # Load it back
            loaded = store.load_conversation("user1", "conv_001")
            assert loaded is not None
            assert loaded.id == "conv_001"
            assert len(loaded.messages) == 2
            assert loaded.messages[0].role == "user"
            assert loaded.messages[0].content == "Hello"
            assert loaded.messages[1].role == "assistant"
            assert loaded.messages[1].content == "Hi there!"
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
                messages=[ChatMessage(role="user", content="User1 message")],
            )
            conv2 = Conversation(
                id="conv_001",
                messages=[ChatMessage(role="user", content="User2 message")],
            )

            store.save_conversation("user1", conv1)
            store.save_conversation("user2", conv2)

            # Each user should load their own conversation
            loaded1 = store.load_conversation("user1", "conv_001")
            loaded2 = store.load_conversation("user2", "conv_001")

            assert loaded1 is not None
            assert loaded2 is not None
            assert loaded1.messages[0].content == "User1 message"
            assert loaded2.messages[0].content == "User2 message"

            # Each user should only see their own conversations
            convs1 = store.list_conversations("user1")
            convs2 = store.list_conversations("user2")

            assert convs1 == ["conv_001"]
            assert convs2 == ["conv_001"]
        finally:
            Path(temp_db_path).unlink(missing_ok=True)

    def test_sqlite_conversation_id_generation(self):
        """Test conversation ID generation for SQLite store."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_db:
            temp_db_path = temp_db.name

        try:
            store = SQLite(temp_db_path)

            # First ID should be "001"
            id1 = store.get_next_conversation_id("user1")
            assert id1 == "001"

            # Add a conversation
            conv = Conversation(
                id="001", messages=[ChatMessage(role="user", content="test")]
            )
            store.save_conversation("user1", conv)

            # Next ID should be "002"
            id2 = store.get_next_conversation_id("user1")
            assert id2 == "002"

            # Add non-sequential conversation
            conv3 = Conversation(
                id="005", messages=[ChatMessage(role="user", content="test")]
            )
            store.save_conversation("user1", conv3)

            # Next ID should be "006" (highest + 1)
            id3 = store.get_next_conversation_id("user1")
            assert id3 == "006"
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
                Conversation(
                    id="001", messages=[ChatMessage(role="user", content="First")]
                ),
                Conversation(
                    id="002", messages=[ChatMessage(role="user", content="Second")]
                ),
                Conversation(
                    id="003", messages=[ChatMessage(role="user", content="Third")]
                ),
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
                Conversation(
                    id="001", messages=[ChatMessage(role="user", content="First")]
                ),
                Conversation(
                    id="002", messages=[ChatMessage(role="user", content="Second")]
                ),
                Conversation(
                    id="003", messages=[ChatMessage(role="user", content="Third")]
                ),
            ]

            for conv in convs:
                store.save_conversation("user1", conv)

            # Check what's actually in the database
            with sqlite3.connect(temp_db_path) as conn:
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

            # Get the order from list_conversations
            conversations = store.list_conversations("user1")
            print(f"list_conversations returned: {conversations}")

            assert len(conversations) == 3
            assert set(conversations) == {"001", "002", "003"}
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
                messages=[ChatMessage(role="user", content="Original message")],
            )
            store.save_conversation("user1", original)

            # Update with more messages
            updated = Conversation(
                id="conv_001",
                messages=[
                    ChatMessage(role="user", content="Original message"),
                    ChatMessage(role="assistant", content="Assistant response"),
                    ChatMessage(role="user", content="Follow-up message"),
                ],
            )
            store.save_conversation("user1", updated)

            # Load should return updated version
            loaded = store.load_conversation("user1", "conv_001")
            assert loaded is not None
            assert len(loaded.messages) == 3
            assert loaded.messages[-1].content == "Follow-up message"

            # Should still be only one conversation in list
            conversations = store.list_conversations("user1")
            assert len(conversations) == 1
            assert conversations[0] == "conv_001"
        finally:
            Path(temp_db_path).unlink(missing_ok=True)


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

        # get_next_conversation_id should work for new users (creates their namespace)
        next_id = store.get_next_conversation_id("new_user")
        assert next_id == "001"

        # Store should start with empty internal structure, but after get_next_conversation_id 
        # is called, it creates the user namespace
        assert store._store == {"new_user": {}}

    def test_inmemory_save_and_load_single_conversation(self):
        """Test saving and loading a single conversation."""
        store = InMemory()
        store._store["user1"] = {}  # Create user namespace

        # Create test conversation
        messages = [
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi there!"),
        ]
        conversation = Conversation(id="conv_001", messages=messages)

        store.save_conversation("user1", conversation)

        loaded = store.load_conversation("user1", "conv_001")
        assert loaded is not None
        assert loaded.id == "conv_001"
        assert len(loaded.messages) == 2
        assert loaded.messages[0].role == "user"
        assert loaded.messages[0].content == "Hello"
        assert loaded.messages[1].role == "assistant"
        assert loaded.messages[1].content == "Hi there!"

    def test_inmemory_save_creates_deep_copy(self):
        """Test that save_conversation creates a deep copy to prevent mutations."""
        store = InMemory()
        store._store["user1"] = {}  # Create user namespace

        messages = [ChatMessage(role="user", content="Original content")]
        conversation = Conversation(id="conv_001", messages=messages)

        store.save_conversation("user1", conversation)

        conversation.messages[0].content = "Modified content"
        conversation.messages.append(
            ChatMessage(role="assistant", content="New message")
        )

        loaded = store.load_conversation("user1", "conv_001")
        assert loaded is not None
        assert len(loaded.messages) == 1  # Still only original message
        assert loaded.messages[0].content == "Original content"

    def test_inmemory_multiple_conversations_same_user(self):
        """Test storing multiple conversations for the same user."""
        store = InMemory()
        store._store["user1"] = {}  # Create user namespace

        conv1 = Conversation(
            id="conv_001",
            messages=[ChatMessage(role="user", content="First conversation")],
        )
        conv2 = Conversation(
            id="conv_002",
            messages=[ChatMessage(role="user", content="Second conversation")],
        )

        store.save_conversation("user1", conv1)
        store.save_conversation("user1", conv2)

        loaded1 = store.load_conversation("user1", "conv_001")
        loaded2 = store.load_conversation("user1", "conv_002")

        assert loaded1 is not None
        assert loaded2 is not None
        assert loaded1.messages[0].content == "First conversation"
        assert loaded2.messages[0].content == "Second conversation"

        conversations = store.list_conversations("user1")
        assert len(conversations) == 2
        assert "conv_001" in conversations
        assert "conv_002" in conversations

    def test_inmemory_multiple_users_isolated(self):
        """Test that different users' conversations are properly isolated."""
        store = InMemory()

        conv1 = Conversation(
            id="conv_001", messages=[ChatMessage(role="user", content="User1 message")]
        )
        conv2 = Conversation(
            id="conv_001",
            messages=[ChatMessage(role="user", content="User2 message")],
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
        assert loaded1.messages[0].content == "User1 message"
        assert loaded2.messages[0].content == "User2 message"

        # List conversations should be per-user
        convs1 = store.list_conversations("user1")
        convs2 = store.list_conversations("user2")

        # Each user sees only their own conversations
        assert convs1 == ["conv_001"]
        assert convs2 == ["conv_001"]
        assert loaded1 != loaded2  # Different conversation objects

    def test_inmemory_conversation_id_generation(self):
        """Test conversation ID generation."""
        store = InMemory()
        store._store["user1"] = {}  # Create user namespace

        id1 = store.get_next_conversation_id("user1")
        assert id1 == "001"

        conv = Conversation(id="001", messages=[ChatMessage(role="user", content="test")])
        store.save_conversation("user1", conv)

        id2 = store.get_next_conversation_id("user1")
        assert id2 == "002"

        conv2 = Conversation(
            id="002", messages=[ChatMessage(role="user", content="test2")]
        )
        store.save_conversation("user1", conv2)

        id3 = store.get_next_conversation_id("user1")
        assert id3 == "003"

    def test_inmemory_list_conversations_ordering(self):
        """Test that list_conversations returns conversations in correct order."""
        store = InMemory()
        store._store["user1"] = {}  # Create user namespace

        convs = [
            Conversation(id="5", messages=[ChatMessage(role="user", content="Fifth")]),
            Conversation(id="1", messages=[ChatMessage(role="user", content="First")]),
            Conversation(id="10", messages=[ChatMessage(role="user", content="Tenth")]),
            Conversation(id="2", messages=[ChatMessage(role="user", content="Second")]),
        ]

        for conv in convs:
            store.save_conversation("user1", conv)

        conversations = store.list_conversations("user1")

        numeric_order = [int(conv_id) for conv_id in conversations if conv_id.isdigit()]
        assert numeric_order == sorted(numeric_order, reverse=True)

        assert conversations[0] == "10"

    def test_inmemory_update_existing_conversation(self):
        """Test updating an existing conversation."""
        store = InMemory()
        store._store["user1"] = {}  # Create user namespace

        original = Conversation(
            id="conv_001",
            messages=[ChatMessage(role="user", content="Original message")],
        )
        store.save_conversation("user1", original)

        updated = Conversation(
            id="conv_001",
            messages=[
                ChatMessage(role="user", content="Original message"),
                ChatMessage(role="assistant", content="Assistant response"),
                ChatMessage(role="user", content="Follow-up message"),
            ],
        )
        store.save_conversation("user1", updated)

        loaded = store.load_conversation("user1", "conv_001")
        assert loaded is not None
        assert len(loaded.messages) == 3
        assert loaded.messages[-1].content == "Follow-up message"

        conversations = store.list_conversations("user1")
        assert len(conversations) == 1
        assert conversations[0] == "conv_001"
