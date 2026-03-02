"""Tests for the Anthropic LLM provider."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("anthropic", reason="Anthropic tests require the anthropic package")

from chatnificent.models import (
    ASSISTANT_ROLE,
    USER_ROLE,
    Conversation,
)

from .conftest import (
    make_anthropic_empty_response,
    make_anthropic_response,
    make_anthropic_tool_response,
)

# ===== Fixtures =====


@pytest.fixture()
def anthropic_llm():
    """Create an Anthropic instance with mocked SDK client."""
    from chatnificent.llm import Anthropic

    instance = object.__new__(Anthropic)
    instance.client = MagicMock()
    instance.model = "claude-sonnet-4-5"
    instance.default_params = {"max_tokens": 4096}
    return instance


# ===== Constructor tests =====


class TestAnthropicConstructor:
    def test_raises_without_api_key(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(ValueError, match="Anthropic API key not found"),
        ):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            from chatnificent.llm import Anthropic

            Anthropic()

    def test_default_model(self):
        from chatnificent.llm import Anthropic

        with patch("anthropic.Anthropic"):
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
                instance = Anthropic()
                assert instance.model == "claude-sonnet-4-5"

    def test_default_max_tokens(self):
        from chatnificent.llm import Anthropic

        with patch("anthropic.Anthropic"):
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
                instance = Anthropic()
                assert instance.default_params["max_tokens"] == 4096

    def test_custom_params_merged(self):
        from chatnificent.llm import Anthropic

        with patch("anthropic.Anthropic"):
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
                instance = Anthropic(temperature=0.5, max_tokens=8192)
                assert instance.default_params["temperature"] == 0.5
                assert instance.default_params["max_tokens"] == 8192


# ===== extract_content tests =====


class TestExtractContent:
    def test_text_response(self, anthropic_llm):
        response = make_anthropic_response("Hello from Claude!")
        assert anthropic_llm.extract_content(response) == "Hello from Claude!"

    def test_empty_content_shows_stop_reason(self, anthropic_llm):
        response = make_anthropic_empty_response(stop_reason="max_tokens")
        result = anthropic_llm.extract_content(response)
        assert "max_tokens" in result
        assert "Empty response" in result
        assert "claude-sonnet-4-5" in result

    def test_empty_content_end_turn(self, anthropic_llm):
        response = make_anthropic_empty_response(stop_reason="end_turn")
        result = anthropic_llm.extract_content(response)
        assert "end_turn" in result

    def test_tool_use_blocks_only_returns_none(self, anthropic_llm):
        """When response only has tool_use blocks, no text is extracted."""
        response = make_anthropic_tool_response(
            [{"id": "tu_1", "name": "fn", "input": {}}]
        )
        result = anthropic_llm.extract_content(response)
        # tool_use blocks have type="tool_use", not "text", so extract_content
        # iterates all blocks without finding text → returns None
        assert result is None


# ===== parse_tool_calls tests =====


class TestParseToolCalls:
    def test_no_tool_calls(self, anthropic_llm):
        response = make_anthropic_response("Just text")
        assert anthropic_llm.parse_tool_calls(response) is None

    def test_single_tool_call(self, anthropic_llm):
        response = make_anthropic_tool_response(
            [{"id": "tu_abc", "name": "get_weather", "input": {"location": "Boston"}}]
        )
        tool_calls = anthropic_llm.parse_tool_calls(response)
        assert len(tool_calls) == 1
        assert tool_calls[0]["function_name"] == "get_weather"
        assert tool_calls[0]["id"] == "tu_abc"
        assert json.loads(tool_calls[0]["function_args"]) == {"location": "Boston"}

    def test_multiple_tool_calls(self, anthropic_llm):
        response = make_anthropic_tool_response(
            [
                {"id": "tu_1", "name": "fn_a", "input": {}},
                {"id": "tu_2", "name": "fn_b", "input": {"x": 1}},
            ]
        )
        tool_calls = anthropic_llm.parse_tool_calls(response)
        assert len(tool_calls) == 2
        assert tool_calls[0]["function_name"] == "fn_a"
        assert tool_calls[1]["function_name"] == "fn_b"

    def test_returns_standardized_objects(self, anthropic_llm):
        response = make_anthropic_tool_response(
            [{"id": "tu_1", "name": "fn", "input": {"k": "v"}}]
        )
        tc = anthropic_llm.parse_tool_calls(response)[0]
        assert isinstance(tc, dict)
        assert json.loads(tc["function_args"]) == {"k": "v"}


# ===== create_assistant_message tests =====


class TestCreateAssistantMessage:
    def test_text_response(self, anthropic_llm):
        response = make_anthropic_response("Hello!")
        msg = anthropic_llm.create_assistant_message(response)
        assert msg["role"] == ASSISTANT_ROLE
        assert msg["content"] == "Hello!"

    def test_tool_use_response_stores_raw_content(self, anthropic_llm):
        """When stop_reason is tool_use, raw content blocks are stored."""
        from types import SimpleNamespace

        block = SimpleNamespace(
            type="tool_use", id="tu_1", name="get_weather", input={"loc": "NYC"}
        )
        response = SimpleNamespace(
            content=[block],
            stop_reason="tool_use",
            model_dump=lambda: {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tu_1",
                        "name": "get_weather",
                        "input": {"loc": "NYC"},
                    }
                ]
            },
        )
        msg = anthropic_llm.create_assistant_message(response)
        assert msg["role"] == ASSISTANT_ROLE
        assert isinstance(msg["content"], list)
        assert msg["content"][0]["type"] == "tool_use"


# ===== create_tool_result_messages tests =====


class TestCreateToolResultMessages:
    def test_single_result(self, anthropic_llm):
        results = [{"tool_call_id": "tu_1", "function_name": "fn", "content": "result"}]
        convo = Conversation(id="test")
        msgs = anthropic_llm.create_tool_result_messages(results, convo)
        assert len(msgs) == 1
        assert msgs[0]["role"] == USER_ROLE  # Anthropic uses user role for tool results
        assert isinstance(msgs[0]["content"], list)
        assert msgs[0]["content"][0]["type"] == "tool_result"
        assert msgs[0]["content"][0]["tool_use_id"] == "tu_1"
        assert msgs[0]["content"][0]["content"] == "result"

    def test_multiple_results_grouped(self, anthropic_llm):
        """Multiple tool results are grouped into a single user message."""
        results = [
            {"tool_call_id": "tu_1", "function_name": "a", "content": "r1"},
            {"tool_call_id": "tu_2", "function_name": "b", "content": "r2"},
        ]
        convo = Conversation(id="test")
        msgs = anthropic_llm.create_tool_result_messages(results, convo)
        assert len(msgs) == 1
        assert len(msgs[0]["content"]) == 2


# ===== is_tool_message tests =====


class TestIsToolMessage:
    def test_user_tool_result(self, anthropic_llm):
        """Anthropic sends tool results as user messages with tool_result content."""
        msg = {
            "role": USER_ROLE,
            "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": "r"}],
        }
        assert anthropic_llm.is_tool_message(msg) is True

    def test_assistant_tool_use(self, anthropic_llm):
        msg = {
            "role": ASSISTANT_ROLE,
            "content": [{"type": "tool_use", "id": "tu_1", "name": "fn", "input": {}}],
        }
        assert anthropic_llm.is_tool_message(msg) is True

    def test_plain_user_message(self, anthropic_llm):
        msg = {"role": USER_ROLE, "content": "Hello"}
        assert anthropic_llm.is_tool_message(msg) is False

    def test_plain_assistant_message(self, anthropic_llm):
        msg = {"role": ASSISTANT_ROLE, "content": "Response"}
        assert anthropic_llm.is_tool_message(msg) is False


# ===== generate_response tests =====


class TestGenerateResponse:
    def test_calls_client(self, anthropic_llm):
        messages = [{"role": "user", "content": "Hello"}]
        anthropic_llm.generate_response(messages)
        anthropic_llm.client.messages.create.assert_called_once()

    def test_model_override(self, anthropic_llm):
        messages = [{"role": "user", "content": "Hi"}]
        anthropic_llm.generate_response(messages, model="claude-3-haiku-20240307")
        call_kwargs = anthropic_llm.client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-3-haiku-20240307"

    def test_system_prompt_extracted(self, anthropic_llm):
        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hi"},
        ]
        anthropic_llm.generate_response(messages)
        call_kwargs = anthropic_llm.client.messages.create.call_args.kwargs
        assert call_kwargs["system"] == "Be concise."
        # System message should not be in messages list
        assert all(m.get("role") != "system" for m in call_kwargs["messages"])

    def test_does_not_mutate_caller_messages(self, anthropic_llm):
        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hi"},
        ]
        original_len = len(messages)
        anthropic_llm.generate_response(messages)
        assert len(messages) == original_len
        assert messages[0]["role"] == "system"

    def test_tools_translated(self, anthropic_llm):
        messages = [{"role": "user", "content": "Hi"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"loc": {"type": "string"}},
                    },
                },
            }
        ]
        anthropic_llm.generate_response(messages, tools=tools)
        call_kwargs = anthropic_llm.client.messages.create.call_args.kwargs
        assert len(call_kwargs["tools"]) == 1
        assert call_kwargs["tools"][0]["name"] == "get_weather"
        assert "input_schema" in call_kwargs["tools"][0]

    def test_default_params_merged(self, anthropic_llm):
        messages = [{"role": "user", "content": "Hi"}]
        anthropic_llm.generate_response(messages)
        call_kwargs = anthropic_llm.client.messages.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 4096


# ===== _translate_tool_schema tests =====


class TestTranslateToolSchema:
    def test_single_function(self, anthropic_llm):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                },
            }
        ]
        result = anthropic_llm._translate_tool_schema(tools)
        assert len(result) == 1
        assert result[0]["name"] == "get_weather"
        assert result[0]["description"] == "Get weather"
        assert result[0]["input_schema"]["properties"]["location"]["type"] == "string"

    def test_non_function_ignored(self, anthropic_llm):
        tools = [{"type": "retrieval", "retrieval": {}}]
        result = anthropic_llm._translate_tool_schema(tools)
        assert result == []

    def test_missing_parameters(self, anthropic_llm):
        tools = [
            {
                "type": "function",
                "function": {"name": "fn", "description": "desc"},
            }
        ]
        result = anthropic_llm._translate_tool_schema(tools)
        assert result[0]["input_schema"] == {"type": "object", "properties": {}}
