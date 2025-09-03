"""
Tests for the core Pydantic data models.

These models form the data contract between all pillars, so their validation
behavior is critical for the framework's reliability.
"""

import pytest
from chatnificent.models import (
    ASSISTANT_ROLE,
    SYSTEM_ROLE,
    TOOL_ROLE,
    USER_ROLE,
    ChatMessage,
    Conversation,
    Role,
    ToolCall,
    ToolResult,
)
from pydantic import ValidationError


class TestChatMessage:
    """Test ChatMessage model validation and behavior."""

    def test_valid_message_creation(self):
        """Test creating valid ChatMessage instances."""
        # Test user message
        user_msg = ChatMessage(role=USER_ROLE, content="Hello!")
        assert user_msg.role == "user"
        assert user_msg.content == "Hello!"

        # Test assistant message
        assistant_msg = ChatMessage(role=ASSISTANT_ROLE, content="Hi there!")
        assert assistant_msg.role == "assistant"
        assert assistant_msg.content == "Hi there!"

        # Test system message
        system_msg = ChatMessage(
            role=SYSTEM_ROLE, content="You are a helpful assistant."
        )
        assert system_msg.role == "system"

        # Test tool message
        tool_msg = ChatMessage(role=TOOL_ROLE, content="Function executed successfully")
        assert tool_msg.role == "tool"

    def test_message_immutability_after_creation(self):
        """Test that message objects behave predictably after creation."""
        msg = ChatMessage(role=USER_ROLE, content="Original")
        original_role = msg.role
        original_content = msg.content

        # Verify properties don't change unexpectedly
        assert msg.role == original_role
        assert msg.content == original_content

        # Pydantic models are mutable by default, but let's document this behavior
        msg.content = "Modified"
        assert msg.content == "Modified"  # This is expected behavior

    def test_missing_required_fields(self):
        """Test that a missing `role` raises a validation error."""
        # Missing role should raise an error
        with pytest.raises(ValidationError) as exc_info:
            ChatMessage(content="Hello")
        assert any(error["loc"] == ("role",) for error in exc_info.value.errors())

        # Missing content is now allowed (defaults to None)
        msg = ChatMessage(role=USER_ROLE)
        assert msg.content is None

        # Missing role should still be the primary error
        with pytest.raises(ValidationError) as exc_info:
            ChatMessage()
        errors = exc_info.value.errors()
        assert any(error["loc"] == ("role",) for error in errors)
        assert not any(
            error["loc"] == ("content",) for error in errors
        )  # Content has a default

    def test_none_values_for_role_rejected(self):
        """Test that `role=None` is rejected."""
        with pytest.raises(ValidationError):
            ChatMessage(role=None, content="Hello")

    def test_none_content_is_allowed(self):
        """Test that `content=None` is allowed."""
        msg = ChatMessage(role=USER_ROLE, content=None)
        assert msg.content is None

    def test_type_validation_strictness(self):
        """Test that Pydantic enforces strict typing."""
        # Numbers should NOT be auto-coerced to strings in Pydantic v2
        with pytest.raises(ValidationError):
            ChatMessage(role=USER_ROLE, content=42)

        with pytest.raises(ValidationError):
            ChatMessage(role=USER_ROLE, content=True)

        # this is now allowed
        # with pytest.raises(ValidationError):
        #     ChatMessage(role=USER_ROLE, content=[1, 2, 3])

        # Only strings are accepted for content
        msg = ChatMessage(role=USER_ROLE, content="42")
        assert msg.content == "42"
        assert isinstance(msg.content, str)

    def test_list_content_is_allowed(self):
        """Test that list[ContentBlock] is allowed for content."""
        list_content = [
            {"type": "text", "text": "Hello!"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.png"},
            },
        ]
        msg = ChatMessage(role=ASSISTANT_ROLE, content=list_content)
        assert msg.content == list_content

    def test_role_constants(self):
        """Test that role constants match expected values."""
        assert USER_ROLE == "user"
        assert ASSISTANT_ROLE == "assistant"
        assert SYSTEM_ROLE == "system"
        assert TOOL_ROLE == "tool"

    def test_invalid_role_validation(self):
        """Test that invalid roles are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ChatMessage(role="invalid_role", content="Test message")

        error = exc_info.value.errors()[0]
        assert error["type"] == "literal_error"
        assert "invalid_role" in str(error["input"])

    def test_empty_content_allowed(self):
        """Test that empty content is allowed."""
        msg = ChatMessage(role=USER_ROLE, content="")
        assert msg.content == ""

    def test_unicode_content(self):
        """Test Unicode content handling."""
        unicode_content = "Hello 世界! 🌍 مرحبا עולם"
        msg = ChatMessage(role=USER_ROLE, content=unicode_content)
        assert msg.content == unicode_content

    def test_very_long_content(self):
        """Test handling of very long content."""
        long_content = "A" * 10000  # 10K character message
        msg = ChatMessage(role=ASSISTANT_ROLE, content=long_content)
        assert len(msg.content) == 10000
        assert msg.content == long_content

    def test_message_serialization(self):
        """Test message serialization to dict."""
        msg = ChatMessage(role=USER_ROLE, content="Test message")
        serialized = msg.model_dump()

        expected = {"role": "user", "content": "Test message"}
        assert serialized == expected

    def test_message_deserialization(self):
        """Test message creation from dict."""
        data = {"role": "assistant", "content": "Response message"}
        msg = ChatMessage(**data)

        assert msg.role == "assistant"
        assert msg.content == "Response message"

    def test_model_copy(self):
        """Test creating copies of messages."""
        original = ChatMessage(role=USER_ROLE, content="Original content")
        copy = original.model_copy()

        assert copy.role == original.role
        assert copy.content == original.content
        assert copy is not original  # Different instances

    def test_model_copy_with_changes(self):
        """Test creating modified copies."""
        original = ChatMessage(role=USER_ROLE, content="Original")
        modified = original.model_copy(update={"content": "Modified"})

        assert modified.role == "user"
        assert modified.content == "Modified"
        assert original.content == "Original"  # Original unchanged


class TestConversation:
    """Test Conversation model validation and behavior."""

    def test_valid_conversation_creation(self):
        """Test creating valid conversations."""
        conv = Conversation(id="test_001")
        assert conv.id == "test_001"
        assert conv.messages == []  # Default empty list

    def test_conversation_with_messages(self, sample_messages):
        """Test conversation with predefined messages."""
        conv = Conversation(id="test_002", messages=sample_messages)
        assert conv.id == "test_002"
        assert len(conv.messages) == len(sample_messages)
        assert all(isinstance(msg, ChatMessage) for msg in conv.messages)

    def test_conversation_id_validation(self):
        """Test conversation ID requirements."""
        # Valid IDs
        valid_ids = ["001", "conversation_123", "user-chat-001", "conv_αβγ"]
        for conv_id in valid_ids:
            conv = Conversation(id=conv_id)
            assert conv.id == conv_id

    def test_empty_conversation_id_rejected(self):
        """Test that empty conversation IDs are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            Conversation(id="")

        error = exc_info.value.errors()[0]
        assert error["type"] == "string_too_short"
        assert error["loc"] == ("id",)

    def test_message_list_modification(self):
        """Test that message lists can be modified."""
        conv = Conversation(id="test_003")

        # Add messages
        msg1 = ChatMessage(role=USER_ROLE, content="First message")
        conv.messages.append(msg1)
        assert len(conv.messages) == 1

        msg2 = ChatMessage(role=ASSISTANT_ROLE, content="Second message")
        conv.messages.append(msg2)
        assert len(conv.messages) == 2

        # Verify content
        assert conv.messages[0].content == "First message"
        assert conv.messages[1].content == "Second message"

    def test_conversation_serialization(self, sample_messages):
        """Test conversation serialization."""
        conv = Conversation(id="test_004", messages=sample_messages)
        serialized = conv.model_dump()

        assert serialized["id"] == "test_004"
        assert len(serialized["messages"]) == len(sample_messages)
        assert all(isinstance(msg, dict) for msg in serialized["messages"])

        # Check first message structure
        first_msg = serialized["messages"][0]
        assert "role" in first_msg
        assert "content" in first_msg

    def test_conversation_deserialization(self):
        """Test conversation creation from dict."""
        data = {
            "id": "test_005",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ],
        }

        conv = Conversation(**data)
        assert conv.id == "test_005"
        assert len(conv.messages) == 2
        assert isinstance(conv.messages[0], ChatMessage)
        assert conv.messages[0].role == "user"
        assert conv.messages[1].content == "Hi!"

    def test_conversation_copy(self, sample_messages):
        """Test deep copying of conversations."""
        original = Conversation(id="original", messages=sample_messages)
        copy = original.model_copy(deep=True)

        assert copy.id == original.id
        assert len(copy.messages) == len(original.messages)
        assert copy is not original
        assert copy.messages is not original.messages  # Deep copy

        # Verify message contents are the same but different instances
        for orig_msg, copy_msg in zip(original.messages, copy.messages):
            assert orig_msg.role == copy_msg.role
            assert orig_msg.content == copy_msg.content
            assert orig_msg is not copy_msg  # Different instances

    def test_mixed_message_roles(self):
        """Test conversations with different message role patterns."""
        messages = [
            ChatMessage(role=SYSTEM_ROLE, content="System prompt"),
            ChatMessage(role=USER_ROLE, content="User question"),
            ChatMessage(role=ASSISTANT_ROLE, content="Assistant response"),
            ChatMessage(role=TOOL_ROLE, content="Tool output"),
            ChatMessage(role=USER_ROLE, content="Follow-up question"),
        ]

        conv = Conversation(id="mixed_roles", messages=messages)
        assert len(conv.messages) == 5

        roles = [msg.role for msg in conv.messages]
        expected_roles = ["system", "user", "assistant", "tool", "user"]
        assert roles == expected_roles

    def test_conversation_id_edge_cases(self):
        """Test conversation ID validation edge cases."""
        # Single character IDs should work
        conv = Conversation(id="1")
        assert conv.id == "1"

        conv = Conversation(id="a")
        assert conv.id == "a"

        # Special characters should work
        special_ids = ["test-123", "conv_001", "user.chat.001", "αβγ", "测试", "🔥chat"]
        for conv_id in special_ids:
            conv = Conversation(id=conv_id)
            assert conv.id == conv_id

        # Whitespace-only IDs are allowed by Pydantic (min_length counts whitespace)
        # This documents current behavior
        whitespace_conv = Conversation(id="   ")
        assert whitespace_conv.id == "   "
        assert len(whitespace_conv.id) >= 1  # Meets min_length requirement

    def test_conversation_messages_list_behavior(self):
        """Test message list edge cases and behavior."""
        conv = Conversation(id="test")

        # Empty list behavior
        assert conv.messages == []
        assert len(conv.messages) == 0
        assert list(conv.messages) == []

        # List operations
        msg1 = ChatMessage(role=USER_ROLE, content="First")
        conv.messages.append(msg1)
        assert len(conv.messages) == 1

        # Insert at beginning
        msg0 = ChatMessage(role=SYSTEM_ROLE, content="System")
        conv.messages.insert(0, msg0)
        assert len(conv.messages) == 2
        assert conv.messages[0].role == "system"
        assert conv.messages[1].role == "user"

        # Remove messages
        conv.messages.pop(0)
        assert len(conv.messages) == 1
        assert conv.messages[0].role == "user"

        # Clear all messages
        conv.messages.clear()
        assert len(conv.messages) == 0

    def test_conversation_message_type_handling(self):
        """Test how Pydantic handles different message types."""
        # Strings should be rejected
        with pytest.raises(ValidationError):
            Conversation(id="test", messages=["not a message"])

        # Dicts should be auto-converted to ChatMessage objects
        conv = Conversation(
            id="test",
            messages=[{"role": "user", "content": "dict auto-converted"}],
        )
        assert isinstance(conv.messages[0], ChatMessage)
        assert conv.messages[0].role == "user"
        assert conv.messages[0].content == "dict auto-converted"

        # Mixed valid dict and invalid string should fail
        with pytest.raises(ValidationError):
            Conversation(
                id="test",
                messages=[{"role": "user", "content": "valid dict"}, "invalid string"],
            )

    def test_large_conversation_performance(self):
        """Test behavior with large conversations."""
        # Create conversation with many messages
        messages = []
        for i in range(1000):
            role = USER_ROLE if i % 2 == 0 else ASSISTANT_ROLE
            content = f"Message number {i}"
            messages.append(ChatMessage(role=role, content=content))

        conv = Conversation(id="large_conv", messages=messages)
        assert len(conv.messages) == 1000

        # Test serialization performance doesn't crash
        serialized = conv.model_dump()
        assert len(serialized["messages"]) == 1000

        # Test deep copy performance
        copied = conv.model_copy(deep=True)
        assert len(copied.messages) == 1000
        assert copied.messages is not conv.messages  # Different lists

    def test_conversation_id_type_validation(self):
        """Test conversation ID type validation strictness."""
        # Numbers should NOT be auto-coerced to strings
        with pytest.raises(ValidationError):
            Conversation(id=123)

        # None should be rejected
        with pytest.raises(ValidationError):
            Conversation(id=None)

        # Only strings are accepted
        conv = Conversation(id="123")
        assert conv.id == "123"
        assert isinstance(conv.id, str)

    def test_conversation_field_access_patterns(self):
        """Test different ways of accessing and modifying conversation fields."""
        conv = Conversation(id="access_test")

        # Test attribute access
        assert hasattr(conv, "id")
        assert hasattr(conv, "messages")

        # Test dict-like access via model_dump
        data = conv.model_dump()
        assert data["id"] == "access_test"
        assert data["messages"] == []

        # Test field assignment
        conv.id = "new_id"
        assert conv.id == "new_id"

        # Test message list assignment
        new_messages = [ChatMessage(role=USER_ROLE, content="New message")]
        conv.messages = new_messages
        assert len(conv.messages) == 1
        assert conv.messages[0].content == "New message"


class TestRoleTypeAlias:
    """Test the Role type alias behavior."""

    def test_role_type_annotation(self):
        """Test that Role type alias works for type checking."""
        # This test mainly documents the expected behavior
        # Static type checkers would catch issues here

        def process_role(role: Role) -> str:
            return f"Processing {role}"

        # These should all be valid Role values
        valid_roles = [USER_ROLE, ASSISTANT_ROLE, SYSTEM_ROLE, TOOL_ROLE]
        for role in valid_roles:
            result = process_role(role)
            assert role in result


class TestModelIntegration:
    """Integration tests between models."""

    def test_conversation_workflow(self):
        """Test a complete conversation building workflow."""
        # Start with empty conversation
        conv = Conversation(id="workflow_test")
        assert len(conv.messages) == 0

        # Add system prompt
        system_msg = ChatMessage(
            role=SYSTEM_ROLE, content="You are a helpful assistant."
        )
        conv.messages.append(system_msg)

        # Simulate conversation flow
        user_msg = ChatMessage(role=USER_ROLE, content="What is Python?")
        conv.messages.append(user_msg)

        assistant_msg = ChatMessage(
            role=ASSISTANT_ROLE,
            content="Python is a high-level programming language...",
        )
        conv.messages.append(assistant_msg)

        # Verify final state
        assert len(conv.messages) == 3
        assert conv.messages[0].role == "system"
        assert conv.messages[1].role == "user"
        assert conv.messages[2].role == "assistant"

        # Verify serialization works end-to-end
        serialized = conv.model_dump()
        reconstructed = Conversation(**serialized)
        assert reconstructed.id == conv.id
        assert len(reconstructed.messages) == len(conv.messages)

    def test_conversation_message_validation_error_handling(self):
        """Test that invalid messages in conversation data are handled."""
        invalid_data = {
            "id": "error_test",
            "messages": [
                {"role": "user", "content": "Valid message"},
                {
                    "role": "invalid_role",
                    "content": "Invalid message",
                },  # This should fail
            ],
        }

        with pytest.raises(ValidationError) as exc_info:
            Conversation(**invalid_data)

        # Verify the error relates to the message validation
        errors = exc_info.value.errors()
        assert any("literal_error" in error["type"] for error in errors)


class TestModelStressTests:
    """Additional stress tests for model robustness."""

    def test_extremely_long_content(self):
        """Test handling of very long message content."""
        # 100KB of content - realistic for large code snippets
        very_long_content = "A" * (100 * 1024)
        msg = ChatMessage(role=USER_ROLE, content=very_long_content)
        assert len(msg.content) == 100 * 1024

        # Test in conversation
        conv = Conversation(id="stress_test", messages=[msg])
        assert len(conv.messages[0].content) == 100 * 1024

    def test_unicode_comprehensive(self):
        """Test comprehensive Unicode support."""
        unicode_tests = [
            "Émojis: 😀🎉🔥👍",
            "Math: ∀x∈ℝ: x²≥0",
            "Chinese: 你好世界",
            "Arabic: مرحبا بالعالم",
            "Mixed: Hello 世界 🌍",
        ]

        for i, content in enumerate(unicode_tests):
            msg = ChatMessage(role=USER_ROLE, content=content)
            conv = Conversation(id=f"unicode_{i}", messages=[msg])

            # Round-trip through serialization
            serialized = conv.model_dump()
            reconstructed = Conversation(**serialized)
            assert reconstructed.messages[0].content == content

    def test_json_round_trip_edge_cases(self):
        """Test JSON serialization with special characters."""
        problematic_content = [
            'JSON chars: {"key": "value"}',
            'Quotes: single and "double"',
            "Backslashes: \\ and \n",
        ]

        for content in problematic_content:
            msg = ChatMessage(role=USER_ROLE, content=content)
            conv = Conversation(id="json_test", messages=[msg])

            # JSON round-trip
            json_str = conv.model_dump_json()
            reconstructed = Conversation.model_validate_json(json_str)
            assert reconstructed.messages[0].content == content


class TestModelDefaults:
    """Test default values and factory behavior."""

    def test_conversation_default_messages(self):
        """Test that Conversation.messages defaults to an empty list."""
        conv = Conversation(id="test")
        assert conv.messages == []
        assert isinstance(conv.messages, list)

        # Ensure different instances get different lists
        conv1 = Conversation(id="test1")
        conv2 = Conversation(id="test2")
        conv1.messages.append(ChatMessage(role=USER_ROLE, content="msg"))
        assert len(conv2.messages) == 0  # conv2 should not be affected

    def test_chat_message_default_content(self):
        """Test that ChatMessage.content defaults to None."""
        msg = ChatMessage(role=USER_ROLE)
        assert msg.content is None

        # Test serialization with default content
        serialized = msg.model_dump()
        assert serialized["content"] is None

        # Test deserialization
        reconstructed = ChatMessage(**serialized)
        assert reconstructed.content is None

    def test_default_values_in_validation(self):
        """Test how default values interact with validation."""
        # Creating a message with only a role should work and use the default for content
        msg = ChatMessage(role=USER_ROLE)
        assert msg.role == "user"
        assert msg.content is None

        # Creating a conversation with only an ID should work
        conv = Conversation(id="test_id")
        assert conv.id == "test_id"
        assert conv.messages == []

    def test_default_values_not_shared(self):
        """Verify that default factories create unique objects."""
        # This is a critical test for mutable defaults
        conv1 = Conversation(id="c1")
        conv2 = Conversation(id="c2")

        # Modify the list on one instance
        conv1.messages.append(ChatMessage(role=USER_ROLE, content="test"))

        # The other instance should remain unaffected
        assert len(conv1.messages) == 1
        assert len(conv2.messages) == 0
        assert conv1.messages is not conv2.messages


class TestToolModels:
    """Tests for ToolCall and ToolResult models."""

    def test_tool_call_creation(self):
        """Test creating a valid ToolCall."""
        tool_call = ToolCall(
            id="call_1",
            function_name="get_weather",
            function_args='{"location": "San Francisco"}',
        )
        assert tool_call.id == "call_1"
        assert tool_call.function_name == "get_weather"
        assert tool_call.function_args == '{"location": "San Francisco"}'

    def test_tool_call_get_args_dict(self):
        """Test that get_args_dict correctly deserializes JSON arguments."""
        tool_call = ToolCall(
            id="call_1",
            function_name="get_weather",
            function_args='{"location": "San Francisco", "unit": "celsius"}',
        )
        args = tool_call.get_args_dict()
        assert args == {"location": "San Francisco", "unit": "celsius"}

    def test_tool_call_get_args_dict_invalid_json(self):
        """Test that get_args_dict returns an empty dict for malformed JSON."""
        tool_call = ToolCall(
            id="call_1",
            function_name="get_weather",
            function_args='{"location": "San Francisco",',
        )
        args = tool_call.get_args_dict()
        assert args == {}

    def test_tool_result_creation(self):
        """Test creating a valid ToolResult."""
        tool_result = ToolResult(
            tool_call_id="call_1",
            function_name="get_weather",
            content="The weather in San Francisco is 70 degrees and sunny.",
        )
        assert tool_result.tool_call_id == "call_1"
        assert tool_result.is_error is False
        assert "70 degrees" in tool_result.content

    def test_tool_result_error(self):
        """Test creating an error ToolResult."""
        tool_result = ToolResult(
            tool_call_id="call_1",
            function_name="get_weather",
            content="Error: Location not found.",
            is_error=True,
        )
        assert tool_result.is_error is True
        assert "Location not found" in tool_result.content
