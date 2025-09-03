"""Integration tests for Engine + Store interaction."""

from pathlib import Path
from unittest.mock import Mock

import pytest
from chatnificent import Chatnificent
from chatnificent.llm import Echo
from chatnificent.models import ASSISTANT_ROLE, USER_ROLE, ChatMessage, Conversation
from chatnificent.store import File, InMemory, SQLite


class TestEngineStoreIntegration:
    """Test Engine and Store working together with different implementations."""

    @pytest.fixture
    def store_implementations(self, tmp_path):
        """All store implementations to test."""
        return [
            ("InMemory", InMemory()),
            ("File", File(str(tmp_path / "file_store"))),
            ("SQLite", SQLite(str(tmp_path / "test.db"))),
        ]

    def test_conversation_persistence_across_stores(self, store_implementations):
        """Test that conversations persist correctly with all store types."""
        for store_name, store in store_implementations:
            app = Chatnificent(llm=Echo(), store=store)

            # Create a conversation
            app.engine.handle_message(
                user_input=f"Test message for {store_name}",
                user_id="test_user",
                convo_id_from_url=None,
            )

            # Get the conversation ID
            conversations = store.list_conversations("test_user")
            assert len(conversations) > 0, f"No conversations for {store_name}"
            convo_id = conversations[0]

            # Load and verify
            loaded = store.load_conversation("test_user", convo_id)
            assert loaded is not None, f"Failed to load conversation for {store_name}"
            assert len(loaded.messages) == 2
            assert loaded.messages[0].content == f"Test message for {store_name}"

    def test_multiple_users_isolated(self, store_implementations):
        """Test that conversations are isolated between users."""
        for store_name, store in store_implementations:
            # Skip InMemory as it doesn't isolate by user (documented limitation)
            if store_name == "InMemory":
                continue

            app = Chatnificent(llm=Echo(), store=store)

            # User 1 creates conversation
            app.engine.handle_message(
                user_input="User 1 message", user_id="user1", convo_id_from_url=None
            )

            # User 2 creates conversation
            app.engine.handle_message(
                user_input="User 2 message", user_id="user2", convo_id_from_url=None
            )

            # Check isolation
            user1_convos = store.list_conversations("user1")
            user2_convos = store.list_conversations("user2")

            assert len(user1_convos) > 0
            assert len(user2_convos) > 0

            # Load and verify content
            user1_conv = store.load_conversation("user1", user1_convos[0])
            user2_conv = store.load_conversation("user2", user2_convos[0])

            assert user1_conv.messages[0].content == "User 1 message"
            assert user2_conv.messages[0].content == "User 2 message"

    def test_conversation_id_generation(self, store_implementations):
        """Test that stores generate unique conversation IDs correctly."""
        for store_name, store in store_implementations:
            app = Chatnificent(llm=Echo(), store=store)

            # Create multiple conversations
            for i in range(3):
                app.engine.handle_message(
                    user_input=f"Message {i}",
                    user_id="test_user",
                    convo_id_from_url=None,
                )

            conversations = store.list_conversations("test_user")

            # Should have 3 conversations
            assert len(conversations) >= 3, (
                f"Expected 3+ conversations for {store_name}"
            )

            # All IDs should be unique
            assert len(set(conversations)) == len(conversations)

    def test_continuing_existing_conversation(self, store_implementations):
        """Test continuing an existing conversation."""
        for _, store in store_implementations:
            app = Chatnificent(llm=Echo(), store=store)

            # Start conversation
            app.engine.handle_message(
                user_input="First message", user_id="test_user", convo_id_from_url=None
            )

            conversations = store.list_conversations("test_user")
            convo_id = conversations[0]

            # Continue conversation
            app.engine.handle_message(
                user_input="Second message",
                user_id="test_user",
                convo_id_from_url=convo_id,
            )

            # Verify both messages are there
            loaded = store.load_conversation("test_user", convo_id)
            assert len(loaded.messages) == 4  # 2 user + 2 assistant
            assert loaded.messages[0].content == "First message"
            assert loaded.messages[2].content == "Second message"

    def test_store_save_raw_api_response(self, tmp_path):
        """Test that raw API responses are saved when supported."""
        # Only File and SQLite support raw response saving
        stores_with_raw = [
            ("File", File(str(tmp_path / "file_store"))),
            ("SQLite", SQLite(str(tmp_path / "test.db"))),
        ]

        for store_name, store in stores_with_raw:
            mock_llm = Mock()
            mock_llm.generate_response.return_value = {"raw": "response"}
            mock_llm.extract_content.return_value = "Test response"
            mock_llm.parse_tool_calls.return_value = []

            app = Chatnificent(llm=mock_llm, store=store)

            app.engine.handle_message(
                user_input="Test", user_id="test_user", convo_id_from_url=None
            )

            # For File store, check the raw response file exists
            if store_name == "File":
                conversations = store.list_conversations("test_user")
                convo_dir = (
                    Path(tmp_path) / "file_store" / "test_user" / conversations[0]
                )
                raw_file = convo_dir / "raw_api_responses.jsonl"
                assert raw_file.exists()

    def test_nonexistent_conversation_handling(self, store_implementations):
        """Test handling of nonexistent conversation IDs."""
        for _, store in store_implementations:
            app = Chatnificent(llm=Echo(), store=store)

            # Try to continue a nonexistent conversation
            app.engine.handle_message(
                user_input="Test message",
                user_id="test_user",
                convo_id_from_url="nonexistent_id",
            )

            # Should create a new conversation
            conversations = store.list_conversations("test_user")
            assert len(conversations) > 0

            # The new conversation should contain the message
            loaded = store.load_conversation("test_user", conversations[0])
            assert loaded.messages[0].content == "Test message"

    def test_empty_conversation_handling(self, store_implementations):
        """Test handling when trying to continue non-existent conversations."""
        for _, store in store_implementations:
            app = Chatnificent(llm=Echo(), store=store)

            # Try to continue a conversation that doesn't exist
            app.engine.handle_message(
                user_input="Message for nonexistent convo",
                user_id="test_user",
                convo_id_from_url="does_not_exist",
            )

            # Engine creates a new conversation when requested one doesn't exist
            conversations = store.list_conversations("test_user")
            assert len(conversations) > 0

            # The new conversation should contain the message
            newest_convo = store.load_conversation("test_user", conversations[0])
            assert newest_convo is not None
            assert len(newest_convo.messages) == 2
            assert newest_convo.messages[0].content == "Message for nonexistent convo"


class TestStoreEdgeCases:
    """Test edge cases in store implementations."""

    def test_concurrent_saves(self, tmp_path):
        """Test that stores handle concurrent saves correctly."""
        store = File(str(tmp_path / "concurrent_test"))
        app = Chatnificent(llm=Echo(), store=store)

        # Create initial conversation
        app.engine.handle_message(
            user_input="Initial", user_id="test_user", convo_id_from_url=None
        )

        conversations = store.list_conversations("test_user")
        convo_id = conversations[0]

        # Simulate concurrent modifications
        conv1 = store.load_conversation("test_user", convo_id)
        conv2 = store.load_conversation("test_user", convo_id)

        # Both add messages
        conv1.messages.append(ChatMessage(role=USER_ROLE, content="From conv1"))
        conv2.messages.append(ChatMessage(role=USER_ROLE, content="From conv2"))

        # Save both (second should overwrite first)
        store.save_conversation("test_user", conv1)
        store.save_conversation("test_user", conv2)

        # Load and check - should have conv2's version
        final = store.load_conversation("test_user", convo_id)
        assert final.messages[-1].content == "From conv2"

    def test_special_characters_in_ids(self, tmp_path):
        """Test stores handle special characters in user/conversation IDs."""
        stores = [
            InMemory(),
            File(str(tmp_path / "special_chars")),
            SQLite(str(tmp_path / "special.db")),
        ]

        special_ids = [
            "user-with-dashes",
            "user_with_underscores",
            "user.with.dots",
            "user@email.com",
        ]

        for store in stores:
            for user_id in special_ids:
                app = Chatnificent(llm=Echo(), store=store)

                # Create conversation with special user ID
                app.engine.handle_message(
                    user_input=f"Test from {user_id}",
                    user_id=user_id,
                    convo_id_from_url=None,
                )

                # Should be able to list and load
                conversations = store.list_conversations(user_id)
                assert len(conversations) > 0

                loaded = store.load_conversation(user_id, conversations[0])
                assert loaded.messages[0].content == f"Test from {user_id}"

    def test_large_conversation_handling(self, tmp_path):
        """Test stores handle large conversations efficiently."""
        stores = [
            ("InMemory", InMemory()),
            ("File", File(str(tmp_path / "large"))),
            ("SQLite", SQLite(str(tmp_path / "large.db"))),
        ]

        for _, store in stores:
            # Create large conversation
            conv = Conversation(id="large", messages=[])
            for i in range(100):
                conv.messages.append(
                    ChatMessage(
                        role=USER_ROLE if i % 2 == 0 else ASSISTANT_ROLE,
                        content=f"Message {i}",
                    )
                )

            # Establish user namespace first (mimicking realistic engine flow)
            store.get_next_conversation_id("test_user")

            # Save and reload
            store.save_conversation("test_user", conv)
            loaded = store.load_conversation("test_user", "large")

            assert len(loaded.messages) == 100
            assert loaded.messages[0].content == "Message 0"
            assert loaded.messages[99].content == "Message 99"


class TestStoreListingBehavior:
    """Test conversation listing behavior across stores."""

    def test_listing_order(self, tmp_path):
        """Test that conversations are listed in expected order."""
        import time

        stores = [
            ("File", File(str(tmp_path / "order_test"))),
            ("SQLite", SQLite(str(tmp_path / "order.db"))),
        ]

        for _, store in stores:
            app = Chatnificent(llm=Echo(), store=store)

            # Create conversations with slight delays
            for i in range(3):
                app.engine.handle_message(
                    user_input=f"Message {i}",
                    user_id="test_user",
                    convo_id_from_url=None,
                )
                time.sleep(0.01)  # Small delay to ensure different timestamps

            conversations = store.list_conversations("test_user")

            # Should be in reverse chronological order (newest first)
            # Load each to verify
            for i, convo_id in enumerate(conversations):
                loaded = store.load_conversation("test_user", convo_id)
                # Newest conversation should be first in list
                expected_msg_num = 2 - i  # 2, 1, 0
                if expected_msg_num >= 0:  # Safety check
                    assert f"Message {expected_msg_num}" in loaded.messages[0].content

    def test_empty_user_listing(self, tmp_path):
        """Test listing conversations for user with no conversations."""
        stores = [
            InMemory(),
            File(str(tmp_path / "empty_user")),
            SQLite(str(tmp_path / "empty.db")),
        ]

        for store in stores:
            # List for nonexistent user
            conversations = store.list_conversations("nonexistent_user")
            assert conversations == []
