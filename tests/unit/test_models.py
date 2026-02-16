"""
Tests for the core data models.

Conversation is a dataclass. Messages are plain dicts.
Role constants prevent typos across pillars.
"""

import copy
import json

import pytest
from chatnificent.models import (
    ASSISTANT_ROLE,
    MODEL_ROLE,
    SYSTEM_ROLE,
    TOOL_ROLE,
    USER_ROLE,
    Conversation,
)


class TestRoleConstants:
    """Test role constants have expected values."""

    def test_role_constants(self):
        assert USER_ROLE == "user"
        assert ASSISTANT_ROLE == "assistant"
        assert SYSTEM_ROLE == "system"
        assert TOOL_ROLE == "tool"
        assert MODEL_ROLE == "model"


class TestConversation:
    """Test Conversation dataclass."""

    def test_creation_with_defaults(self):
        conv = Conversation(id="test_001")
        assert conv.id == "test_001"
        assert conv.messages == []

    def test_creation_with_messages(self):
        messages = [
            {"role": USER_ROLE, "content": "Hello"},
            {"role": ASSISTANT_ROLE, "content": "Hi!"},
        ]
        conv = Conversation(id="test_002", messages=messages)
        assert conv.id == "test_002"
        assert len(conv.messages) == 2
        assert conv.messages[0]["role"] == "user"
        assert conv.messages[1]["content"] == "Hi!"

    def test_message_list_modification(self):
        conv = Conversation(id="test_003")
        conv.messages.append({"role": USER_ROLE, "content": "First message"})
        assert len(conv.messages) == 1
        conv.messages.append({"role": ASSISTANT_ROLE, "content": "Second message"})
        assert len(conv.messages) == 2
        assert conv.messages[0]["content"] == "First message"
        assert conv.messages[1]["content"] == "Second message"

    def test_default_messages_not_shared(self):
        """Different instances get different lists."""
        conv1 = Conversation(id="c1")
        conv2 = Conversation(id="c2")
        conv1.messages.append({"role": USER_ROLE, "content": "msg"})
        assert len(conv2.messages) == 0
        assert conv1.messages is not conv2.messages

    def test_copy_shallow(self):
        messages = [{"role": USER_ROLE, "content": "Hello"}]
        original = Conversation(id="orig", messages=messages)
        copied = original.copy()

        assert copied.id == original.id
        assert copied.messages == original.messages
        assert copied is not original
        assert copied.messages is not original.messages

    def test_copy_deep(self):
        messages = [{"role": USER_ROLE, "content": "Hello"}]
        original = Conversation(id="orig", messages=messages)
        copied = original.copy(deep=True)

        assert copied.id == original.id
        assert copied.messages == original.messages
        assert copied is not original
        assert copied.messages is not original.messages
        assert copied.messages[0] is not original.messages[0]

    def test_copy_deep_prevents_mutation(self):
        original = Conversation(
            id="orig", messages=[{"role": USER_ROLE, "content": "Original"}]
        )
        copied = original.copy(deep=True)

        copied.messages[0]["content"] = "Modified"
        assert original.messages[0]["content"] == "Original"

    def test_conversation_id_types(self):
        valid_ids = ["001", "conversation_123", "user-chat-001", "a", "1"]
        for conv_id in valid_ids:
            conv = Conversation(id=conv_id)
            assert conv.id == conv_id

    def test_mixed_message_roles(self):
        messages = [
            {"role": SYSTEM_ROLE, "content": "System prompt"},
            {"role": USER_ROLE, "content": "User question"},
            {"role": ASSISTANT_ROLE, "content": "Assistant response"},
            {"role": TOOL_ROLE, "content": "Tool output"},
            {"role": USER_ROLE, "content": "Follow-up question"},
        ]
        conv = Conversation(id="mixed_roles", messages=messages)
        assert len(conv.messages) == 5
        roles = [msg["role"] for msg in conv.messages]
        assert roles == ["system", "user", "assistant", "tool", "user"]

    def test_unicode_content(self):
        unicode_content = "Hello 世界! 🌍 مرحبا עולם"
        msg = {"role": USER_ROLE, "content": unicode_content}
        conv = Conversation(id="unicode", messages=[msg])
        assert conv.messages[0]["content"] == unicode_content

    def test_very_long_content(self):
        long_content = "A" * 100_000
        msg = {"role": ASSISTANT_ROLE, "content": long_content}
        conv = Conversation(id="long", messages=[msg])
        assert len(conv.messages[0]["content"]) == 100_000

    def test_none_content(self):
        msg = {"role": USER_ROLE, "content": None}
        conv = Conversation(id="none_content", messages=[msg])
        assert conv.messages[0]["content"] is None

    def test_list_content(self):
        list_content = [
            {"type": "text", "text": "Hello!"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
        ]
        msg = {"role": ASSISTANT_ROLE, "content": list_content}
        conv = Conversation(id="list_content", messages=[msg])
        assert conv.messages[0]["content"] == list_content

    def test_message_list_operations(self):
        conv = Conversation(id="test")
        conv.messages.append({"role": USER_ROLE, "content": "First"})
        conv.messages.insert(0, {"role": SYSTEM_ROLE, "content": "System"})
        assert len(conv.messages) == 2
        assert conv.messages[0]["role"] == "system"
        assert conv.messages[1]["role"] == "user"

        conv.messages.pop(0)
        assert len(conv.messages) == 1
        assert conv.messages[0]["role"] == "user"

        conv.messages.clear()
        assert len(conv.messages) == 0

    def test_large_conversation(self):
        messages = []
        for i in range(1000):
            role = USER_ROLE if i % 2 == 0 else ASSISTANT_ROLE
            messages.append({"role": role, "content": f"Message number {i}"})
        conv = Conversation(id="large_conv", messages=messages)
        assert len(conv.messages) == 1000

        copied = conv.copy(deep=True)
        assert len(copied.messages) == 1000
        assert copied.messages is not conv.messages

    def test_json_round_trip(self):
        messages = [
            {"role": USER_ROLE, "content": 'JSON chars: {"key": "value"}'},
            {"role": ASSISTANT_ROLE, "content": "Backslashes: \\\\ and newlines"},
        ]
        conv = Conversation(id="json_test", messages=messages)

        json_str = json.dumps({"id": conv.id, "messages": conv.messages})
        data = json.loads(json_str)
        reconstructed = Conversation(id=data["id"], messages=data["messages"])
        assert reconstructed.messages[0]["content"] == messages[0]["content"]

    def test_extra_keys_in_messages(self):
        msg = {
            "role": ASSISTANT_ROLE,
            "content": "Using tool",
            "tool_calls": [{"id": "call_1", "function": {"name": "f"}}],
        }
        conv = Conversation(id="extra_keys", messages=[msg])
        assert conv.messages[0]["tool_calls"][0]["id"] == "call_1"


class TestModelWorkflow:
    """Integration-style tests for model usage patterns."""

    def test_conversation_building_workflow(self):
        conv = Conversation(id="workflow_test")
        conv.messages.append(
            {"role": SYSTEM_ROLE, "content": "You are a helpful assistant."}
        )
        conv.messages.append({"role": USER_ROLE, "content": "What is Python?"})
        conv.messages.append(
            {
                "role": ASSISTANT_ROLE,
                "content": "Python is a high-level programming language...",
            }
        )
        assert len(conv.messages) == 3
        assert conv.messages[0]["role"] == "system"
        assert conv.messages[1]["role"] == "user"
        assert conv.messages[2]["role"] == "assistant"
