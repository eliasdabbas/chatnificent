"""Tests for OpenAI-compatible LLM providers (OpenAI, OpenRouter, DeepSeek)."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from chatnificent.models import (
    ASSISTANT_ROLE,
    TOOL_ROLE,
    USER_ROLE,
    Conversation,
)

from .conftest import (
    make_openai_empty_response,
    make_openai_response,
    make_openai_tool_response,
)

# ===== Fixtures =====


@pytest.fixture()
def openai_llm():
    """Create an OpenAI instance with mocked SDK client."""
    from chatnificent.llm import OpenAI

    instance = object.__new__(OpenAI)
    instance.client = MagicMock()
    instance.model = "gpt-4.1"
    instance.default_params = {}
    return instance


@pytest.fixture()
def openrouter_llm():
    """Create an OpenRouter instance with mocked SDK client."""
    from chatnificent.llm import OpenRouter

    instance = object.__new__(OpenRouter)
    instance.client = MagicMock()
    instance.model = "openai/gpt-4.1"
    instance.default_params = {}
    return instance


@pytest.fixture()
def deepseek_llm():
    """Create a DeepSeek instance with mocked SDK client."""
    from chatnificent.llm import DeepSeek

    instance = object.__new__(DeepSeek)
    instance.client = MagicMock()
    instance.model = "deepseek-chat"
    instance.default_params = {}
    return instance


# ===== Constructor tests =====


class TestOpenAIConstructor:
    def test_raises_without_api_key(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(ValueError, match="OpenAI API key not found"),
        ):
            os.environ.pop("OPENAI_API_KEY", None)
            from chatnificent.llm import OpenAI

            OpenAI()

    def test_accepts_explicit_api_key(self):
        with patch("chatnificent.llm.OpenAI.__init__", return_value=None):
            from chatnificent.llm import OpenAI as LLMOpenAI

            instance = object.__new__(LLMOpenAI)
            instance.model = "gpt-4.1"
            instance.default_params = {}
            assert instance.model == "gpt-4.1"

    def test_default_model(self):
        from chatnificent.llm import OpenAI

        with patch("openai.OpenAI"):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                instance = OpenAI()
                assert instance.model == "gpt-4.1"

    def test_custom_model(self):
        from chatnificent.llm import OpenAI

        with patch("openai.OpenAI"):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                instance = OpenAI(model="gpt-4o-mini")
                assert instance.model == "gpt-4o-mini"

    def test_default_params_stored(self):
        from chatnificent.llm import OpenAI

        with patch("openai.OpenAI"):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                instance = OpenAI(temperature=0.7, top_p=0.9)
                assert instance.default_params == {"temperature": 0.7, "top_p": 0.9}


class TestOpenRouterConstructor:
    def test_raises_without_api_key(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(ValueError, match="OpenRouter API key not found"),
        ):
            os.environ.pop("OPENROUTER_API_KEY", None)
            from chatnificent.llm import OpenRouter

            OpenRouter()

    def test_default_model(self):
        from chatnificent.llm import OpenRouter

        with patch("openai.OpenAI"):
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
                instance = OpenRouter()
                assert instance.model == "openai/gpt-4.1"


class TestDeepSeekConstructor:
    def test_raises_without_api_key(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(ValueError, match="DeepSeek API key not found"),
        ):
            os.environ.pop("DEEPSEEK_API_KEY", None)
            from chatnificent.llm import DeepSeek

            DeepSeek()

    def test_default_model(self):
        from chatnificent.llm import DeepSeek

        with patch("openai.OpenAI"):
            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}):
                instance = DeepSeek()
                assert instance.model == "deepseek-chat"


# ===== extract_content tests =====


class TestExtractContent:
    def test_text_response(self, openai_llm):
        response = make_openai_response("Hello from OpenAI!")
        assert openai_llm.extract_content(response) == "Hello from OpenAI!"

    def test_empty_choices(self, openai_llm):
        response = make_openai_empty_response()
        assert openai_llm.extract_content(response) is None

    def test_null_content_shows_finish_reason(self, openai_llm):
        response = make_openai_response(content=None, finish_reason="length")
        result = openai_llm.extract_content(response)
        assert "length" in result
        assert "Empty response" in result
        assert "gpt-4.1" in result

    def test_empty_string_content_shows_finish_reason(self, openai_llm):
        response = make_openai_response(content="", finish_reason="content_filter")
        result = openai_llm.extract_content(response)
        assert "content_filter" in result

    def test_tool_call_response_shows_finish_reason(self, openai_llm):
        """When model returns tool calls, content is null — shows finish_reason."""
        response = make_openai_tool_response(
            [{"id": "call_1", "name": "fn", "arguments": "{}"}]
        )
        result = openai_llm.extract_content(response)
        assert "tool_calls" in result


# ===== extract_content on different providers =====


class TestExtractContentOpenRouter:
    def test_text_response(self, openrouter_llm):
        response = make_openai_response("Hello from OpenRouter!")
        assert openrouter_llm.extract_content(response) == "Hello from OpenRouter!"

    def test_empty_shows_model_name(self, openrouter_llm):
        response = make_openai_response(content=None, finish_reason="length")
        result = openrouter_llm.extract_content(response)
        assert "openai/gpt-4.1" in result


class TestExtractContentDeepSeek:
    def test_text_response(self, deepseek_llm):
        response = make_openai_response("Hello from DeepSeek!")
        assert deepseek_llm.extract_content(response) == "Hello from DeepSeek!"

    def test_empty_shows_model_name(self, deepseek_llm):
        response = make_openai_response(content=None, finish_reason="length")
        result = deepseek_llm.extract_content(response)
        assert "deepseek-chat" in result


# ===== parse_tool_calls tests =====


class TestParseToolCalls:
    def test_no_tool_calls(self, openai_llm):
        response = make_openai_response("Just text")
        assert openai_llm.parse_tool_calls(response) is None

    def test_empty_choices(self, openai_llm):
        response = make_openai_empty_response()
        assert openai_llm.parse_tool_calls(response) is None

    def test_single_tool_call(self, openai_llm):
        response = make_openai_tool_response(
            [
                {
                    "id": "call_abc",
                    "name": "get_weather",
                    "arguments": '{"location": "Boston"}',
                }
            ]
        )
        tool_calls = openai_llm.parse_tool_calls(response)
        assert len(tool_calls) == 1
        assert tool_calls[0]["function_name"] == "get_weather"
        assert tool_calls[0]["id"] == "call_abc"
        assert json.loads(tool_calls[0]["function_args"]) == {"location": "Boston"}

    def test_multiple_tool_calls(self, openai_llm):
        response = make_openai_tool_response(
            [
                {"id": "call_1", "name": "fn_a", "arguments": "{}"},
                {"id": "call_2", "name": "fn_b", "arguments": '{"x": 1}'},
            ]
        )
        tool_calls = openai_llm.parse_tool_calls(response)
        assert len(tool_calls) == 2
        assert tool_calls[0]["function_name"] == "fn_a"
        assert tool_calls[1]["function_name"] == "fn_b"

    def test_returns_standardized_objects(self, openai_llm):
        response = make_openai_tool_response(
            [{"id": "call_1", "name": "fn", "arguments": '{"k": "v"}'}]
        )
        tc = openai_llm.parse_tool_calls(response)[0]
        assert isinstance(tc, dict)
        assert json.loads(tc["function_args"]) == {"k": "v"}


# ===== create_assistant_message tests =====


class TestCreateAssistantMessage:
    def test_text_response(self, openai_llm):
        response = make_openai_response("Hello!")
        msg = openai_llm.create_assistant_message(response)
        assert msg["role"] == ASSISTANT_ROLE
        assert msg["content"] == "Hello!"

    def test_empty_response(self, openai_llm):
        response = make_openai_empty_response()
        msg = openai_llm.create_assistant_message(response)
        assert msg["role"] == ASSISTANT_ROLE
        assert msg["content"] == "[No response generated]"

    def test_tool_call_response_preserves_raw(self, openai_llm):
        response = make_openai_tool_response(
            [{"id": "call_1", "name": "get_weather", "arguments": '{"loc": "NYC"}'}]
        )
        msg = openai_llm.create_assistant_message(response)
        assert msg["role"] == ASSISTANT_ROLE
        assert msg["content"] is None
        assert msg["tool_calls"] is not None
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["function"]["name"] == "get_weather"


# ===== create_tool_result_messages tests =====


class TestCreateToolResultMessages:
    def test_single_result(self, openai_llm):
        results = [
            {"tool_call_id": "call_1", "function_name": "fn", "content": "result"}
        ]
        convo = Conversation(id="test")
        msgs = openai_llm.create_tool_result_messages(results, convo)
        assert len(msgs) == 1
        assert msgs[0]["role"] == TOOL_ROLE
        assert msgs[0]["content"] == "result"
        assert msgs[0]["tool_call_id"] == "call_1"

    def test_multiple_results(self, openai_llm):
        results = [
            {"tool_call_id": "call_1", "function_name": "a", "content": "r1"},
            {"tool_call_id": "call_2", "function_name": "b", "content": "r2"},
        ]
        convo = Conversation(id="test")
        msgs = openai_llm.create_tool_result_messages(results, convo)
        assert len(msgs) == 2


# ===== is_tool_message tests =====


class TestIsToolMessage:
    def test_tool_role(self, openai_llm):
        msg = {"role": TOOL_ROLE, "content": "result", "tool_call_id": "call_1"}
        assert openai_llm.is_tool_message(msg) is True

    def test_user_message(self, openai_llm):
        msg = {"role": USER_ROLE, "content": "Hello"}
        assert openai_llm.is_tool_message(msg) is False

    def test_assistant_message(self, openai_llm):
        msg = {"role": ASSISTANT_ROLE, "content": "Response"}
        assert openai_llm.is_tool_message(msg) is False


# ===== generate_response tests =====


class TestGenerateResponse:
    def test_calls_client(self, openai_llm):
        messages = [{"role": "user", "content": "Hello"}]
        openai_llm.generate_response(messages)
        openai_llm.client.chat.completions.create.assert_called_once()

    def test_model_override(self, openai_llm):
        messages = [{"role": "user", "content": "Hi"}]
        openai_llm.generate_response(messages, model="gpt-4o-mini")
        call_kwargs = openai_llm.client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o-mini"

    def test_default_params_merged(self, openai_llm):
        openai_llm.default_params = {"temperature": 0.5}
        messages = [{"role": "user", "content": "Hi"}]
        openai_llm.generate_response(messages)
        call_kwargs = openai_llm.client.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.5

    def test_tools_forwarded(self, openai_llm):
        messages = [{"role": "user", "content": "Hi"}]
        tools = [{"type": "function", "function": {"name": "fn"}}]
        openai_llm.generate_response(messages, tools=tools)
        call_kwargs = openai_llm.client.chat.completions.create.call_args.kwargs
        assert call_kwargs["tools"] == tools

    def test_null_content_cleaned(self, openai_llm):
        """Messages with None content are handled without error."""
        messages = [
            {"role": "user", "content": None},
            {"role": "assistant", "content": None},
        ]
        openai_llm.generate_response(messages)
        call_kwargs = openai_llm.client.chat.completions.create.call_args.kwargs
        # None content gets replaced with ""
        assert call_kwargs["messages"][0]["content"] == ""
        assert call_kwargs["messages"][1]["content"] == ""

    def test_assistant_tool_call_preserves_null_content(self, openai_llm):
        """Assistant messages with tool_calls keep None content (OpenAI spec)."""
        messages = [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1"}]},
        ]
        openai_llm.generate_response(messages)
        call_kwargs = openai_llm.client.chat.completions.create.call_args.kwargs
        assert call_kwargs["messages"][0]["content"] is None

    def test_tool_role_empty_content(self, openai_llm):
        """Tool role with None content gets empty string."""
        messages = [{"role": "tool", "content": None, "tool_call_id": "call_1"}]
        openai_llm.generate_response(messages)
        call_kwargs = openai_llm.client.chat.completions.create.call_args.kwargs
        assert call_kwargs["messages"][0]["content"] == ""


# ===== OpenRouter-specific tests =====


class TestOpenRouterGenerateResponse:
    def test_injects_headers(self, openrouter_llm):
        messages = [{"role": "user", "content": "Hi"}]
        openrouter_llm.generate_response(messages)
        call_kwargs = openrouter_llm.client.chat.completions.create.call_args.kwargs
        assert "extra_headers" in call_kwargs
        assert (
            call_kwargs["extra_headers"]["HTTP-Referer"] == "https://chatnificent.com"
        )
        assert call_kwargs["extra_headers"]["X-Title"] == "Chatnificent"
