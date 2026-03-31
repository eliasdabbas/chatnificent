"""Contract tests: assistant-message persistence across providers × stores.

Verifies that provider-native message dicts produced by create_assistant_message()
round-trip faithfully through all Store implementations (InMemory, File, SQLite).
"""

import pytest
from chatnificent.models import Conversation

# =============================================================================
# Provider-native message fixtures — shapes taken from each LLM's
# create_assistant_message() output
# =============================================================================


OPENAI_ASSISTANT_MSG = {
    "role": "assistant",
    "content": "Hello from OpenAI!",
}

OPENAI_TOOL_CALL_MSG = {
    "role": "assistant",
    "content": None,
    "tool_calls": [
        {
            "id": "call_abc123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "San Francisco", "unit": "celsius"}',
            },
        }
    ],
}

ANTHROPIC_ASSISTANT_MSG = {
    "role": "assistant",
    "content": [
        {"type": "text", "text": "Here is the result."},
    ],
}

ANTHROPIC_TOOL_USE_MSG = {
    "role": "assistant",
    "content": [
        {"type": "text", "text": "Let me check that."},
        {
            "type": "tool_use",
            "id": "toolu_01A",
            "name": "get_weather",
            "input": {"location": "Paris"},
        },
    ],
}

GEMINI_ASSISTANT_MSG = {
    "role": "model",
    "parts": [{"text": "Hello from Gemini!"}],
}

GEMINI_TOOL_CALL_MSG = {
    "role": "model",
    "parts": [
        {
            "function_call": {
                "name": "get_weather",
                "args": {"location": "Tokyo"},
            },
            "thought_signature": "abc123sig",
        }
    ],
}

PROVIDER_MESSAGES = [
    pytest.param(OPENAI_ASSISTANT_MSG, id="openai-text"),
    pytest.param(OPENAI_TOOL_CALL_MSG, id="openai-tool-calls"),
    pytest.param(ANTHROPIC_ASSISTANT_MSG, id="anthropic-text"),
    pytest.param(ANTHROPIC_TOOL_USE_MSG, id="anthropic-tool-use"),
    pytest.param(GEMINI_ASSISTANT_MSG, id="gemini-text"),
    pytest.param(GEMINI_TOOL_CALL_MSG, id="gemini-tool-call"),
]


# =============================================================================
# Tests
# =============================================================================


class TestAssistantMessagePersistence:
    """create_assistant_message() output must survive save → load without data loss."""

    @pytest.fixture(params=["InMemory", "File", "SQLite"])
    def store(self, request, tmp_path):
        from chatnificent import store as store_mod

        if request.param == "InMemory":
            return store_mod.InMemory()
        elif request.param == "File":
            return store_mod.File(str(tmp_path / "file_store"))
        elif request.param == "SQLite":
            return store_mod.SQLite(str(tmp_path / "test.db"))

    @pytest.mark.parametrize("assistant_msg", PROVIDER_MESSAGES)
    def test_assistant_message_round_trips(self, store, assistant_msg):
        """Provider-native assistant message survives save → load exactly."""
        convo = Conversation(
            id="c001",
            messages=[
                {"role": "user", "content": "hello"},
                assistant_msg,
            ],
        )
        store.save_conversation("user1", convo)
        loaded = store.load_conversation("user1", "c001")

        assert loaded is not None
        assert len(loaded.messages) == 2
        assert loaded.messages[1] == assistant_msg

    @pytest.mark.parametrize("assistant_msg", PROVIDER_MESSAGES)
    def test_message_keys_preserved(self, store, assistant_msg):
        """All dict keys in the provider message must survive persistence."""
        convo = Conversation(
            id="c002",
            messages=[assistant_msg],
        )
        store.save_conversation("user1", convo)
        loaded = store.load_conversation("user1", "c002")

        stored_msg = loaded.messages[0]
        assert set(stored_msg.keys()) == set(assistant_msg.keys())

    def test_multi_turn_mixed_providers_persist(self, store):
        """A conversation with messages from different provider shapes round-trips."""
        messages = [
            {"role": "user", "content": "hello"},
            OPENAI_ASSISTANT_MSG,
            {"role": "user", "content": "now in anthropic style"},
            ANTHROPIC_ASSISTANT_MSG,
            {"role": "user", "content": "and gemini"},
            GEMINI_ASSISTANT_MSG,
        ]
        convo = Conversation(id="c003", messages=messages)
        store.save_conversation("user1", convo)
        loaded = store.load_conversation("user1", "c003")

        assert loaded is not None
        assert len(loaded.messages) == 6
        for original, stored in zip(messages, loaded.messages):
            assert stored == original

    def test_nested_json_in_tool_args_preserved(self, store):
        """Nested JSON structures within tool call arguments survive persistence."""
        msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_deep",
                    "type": "function",
                    "function": {
                        "name": "complex_tool",
                        "arguments": '{"nested": {"a": [1, 2, 3], "b": {"c": true}}}',
                    },
                }
            ],
        }
        convo = Conversation(id="c004", messages=[msg])
        store.save_conversation("user1", convo)
        loaded = store.load_conversation("user1", "c004")

        stored_args = loaded.messages[0]["tool_calls"][0]["function"]["arguments"]
        assert stored_args == msg["tool_calls"][0]["function"]["arguments"]

    def test_empty_content_tool_call_only_round_trips(self, store):
        """Messages with content=None (tool-call-only) must round-trip."""
        convo = Conversation(id="c005", messages=[OPENAI_TOOL_CALL_MSG])
        store.save_conversation("user1", convo)
        loaded = store.load_conversation("user1", "c005")

        assert loaded.messages[0]["content"] is None
        assert loaded.messages[0]["tool_calls"] == OPENAI_TOOL_CALL_MSG["tool_calls"]
