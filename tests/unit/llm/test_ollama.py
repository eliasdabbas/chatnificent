"""Tests for the Ollama LLM provider."""

import json
from unittest.mock import MagicMock, patch

import pytest
from chatnificent.models import (
    ASSISTANT_ROLE,
    TOOL_ROLE,
    USER_ROLE,
)

from .conftest import (
    make_ollama_empty_response,
    make_ollama_response,
    make_ollama_tool_response,
)

# ===== Fixtures =====


@pytest.fixture()
def ollama_llm():
    """Create an Ollama instance with mocked SDK client."""
    from chatnificent.llm import Ollama

    instance = object.__new__(Ollama)
    instance.client = MagicMock()
    instance.model = "llama3.2"
    return instance


# ===== Constructor tests =====


class TestOllamaConstructor:
    def test_default_model(self):
        from chatnificent.llm import Ollama

        with patch("ollama.Client"):
            instance = Ollama()
            assert instance.model == "llama3.2"

    def test_custom_model(self):
        from chatnificent.llm import Ollama

        with patch("ollama.Client"):
            instance = Ollama(model="mistral")
            assert instance.model == "mistral"


# ===== extract_content tests =====


class TestExtractContent:
    def test_text_response(self, ollama_llm):
        response = make_ollama_response("Hello from Ollama!")
        assert ollama_llm.extract_content(response) == "Hello from Ollama!"

    def test_empty_content_shows_done_reason(self, ollama_llm):
        response = make_ollama_empty_response(done_reason="length")
        result = ollama_llm.extract_content(response)
        assert "length" in result
        assert "Empty response" in result
        assert "llama3.2" in result

    def test_empty_content_stop(self, ollama_llm):
        response = make_ollama_empty_response(done_reason="stop")
        result = ollama_llm.extract_content(response)
        assert "stop" in result

    def test_missing_message_key(self, ollama_llm):
        """Graceful handling when 'message' key is missing."""
        response = {"done_reason": "stop"}
        result = ollama_llm.extract_content(response)
        assert "Empty response" in result


# ===== parse_tool_calls tests =====


class TestParseToolCalls:
    def test_no_tool_calls(self, ollama_llm):
        response = make_ollama_response("Just text")
        assert ollama_llm.parse_tool_calls(response) is None

    def test_single_tool_call(self, ollama_llm):
        response = make_ollama_tool_response(
            [{"name": "get_weather", "arguments": {"location": "Boston"}}]
        )
        tool_calls = ollama_llm.parse_tool_calls(response)
        assert len(tool_calls) == 1
        assert tool_calls[0]["function_name"] == "get_weather"
        assert tool_calls[0]["id"].startswith("ollama-tool-call-")

    def test_multiple_tool_calls(self, ollama_llm):
        response = make_ollama_tool_response(
            [
                {"name": "fn_a", "arguments": {}},
                {"name": "fn_b", "arguments": {"x": 1}},
            ]
        )
        tool_calls = ollama_llm.parse_tool_calls(response)
        assert len(tool_calls) == 2

    def test_returns_standardized_objects(self, ollama_llm):
        response = make_ollama_tool_response([{"name": "fn", "arguments": {"k": "v"}}])
        tc = ollama_llm.parse_tool_calls(response)[0]
        assert isinstance(tc, dict)


# ===== create_assistant_message tests =====


class TestCreateAssistantMessage:
    def test_text_response(self, ollama_llm):
        response = make_ollama_response("Hello!")
        msg = ollama_llm.create_assistant_message(response)
        assert msg["role"] == ASSISTANT_ROLE
        assert msg["content"] == "Hello!"

    def test_tool_call_response(self, ollama_llm):
        response = make_ollama_tool_response([{"name": "fn", "arguments": {"x": 1}}])
        msg = ollama_llm.create_assistant_message(response)
        assert msg["role"] == ASSISTANT_ROLE
        assert msg["tool_calls"] is not None

    def test_missing_message(self, ollama_llm):
        """Graceful handling when response has no message key."""
        response = {}
        msg = ollama_llm.create_assistant_message(response)
        assert msg["role"] == ASSISTANT_ROLE
        assert msg["content"] == ""


# ===== create_tool_result_messages tests =====


class TestCreateToolResultMessages:
    def test_single_result(self, ollama_llm):
        results = [
            {"tool_call_id": "call_1", "function_name": "fn", "content": "result"}
        ]
        from chatnificent.models import Conversation

        convo = Conversation(id="test", messages=[])
        msgs = ollama_llm.create_tool_result_messages(results, convo)
        assert len(msgs) == 1
        assert msgs[0]["role"] == TOOL_ROLE
        assert msgs[0]["content"] == "result"

    def test_multiple_results(self, ollama_llm):
        results = [
            {"tool_call_id": "call_1", "function_name": "a", "content": "r1"},
            {"tool_call_id": "call_2", "function_name": "b", "content": "r2"},
        ]
        from chatnificent.models import Conversation

        convo = Conversation(id="test", messages=[])
        msgs = ollama_llm.create_tool_result_messages(results, convo)
        assert len(msgs) == 2


# ===== generate_response tests =====


class TestGenerateResponse:
    def test_calls_client(self, ollama_llm):
        messages = [{"role": "user", "content": "Hello"}]
        ollama_llm.generate_response(messages)
        ollama_llm.client.chat.assert_called_once()

    def test_model_override(self, ollama_llm):
        messages = [{"role": "user", "content": "Hi"}]
        ollama_llm.generate_response(messages, model="mistral")
        call_kwargs = ollama_llm.client.chat.call_args.kwargs
        assert call_kwargs["model"] == "mistral"

    def test_tools_forwarded(self, ollama_llm):
        messages = [{"role": "user", "content": "Hi"}]
        tools = [{"type": "function", "function": {"name": "fn"}}]
        ollama_llm.generate_response(messages, tools=tools)
        call_kwargs = ollama_llm.client.chat.call_args.kwargs
        assert call_kwargs["tools"] == tools

    def test_no_tools_omitted(self, ollama_llm):
        messages = [{"role": "user", "content": "Hi"}]
        ollama_llm.generate_response(messages)
        call_kwargs = ollama_llm.client.chat.call_args.kwargs
        assert "tools" not in call_kwargs
