"""Tests for the Echo LLM provider."""

import pytest
from chatnificent.models import ASSISTANT_ROLE, USER_ROLE

# ===== Fixtures =====


@pytest.fixture()
def echo_llm():
    """Create an Echo instance."""
    from chatnificent.llm import Echo

    return Echo()


# ===== Constructor tests =====


class TestEchoConstructor:
    def test_default_model(self):
        from chatnificent.llm import Echo

        instance = Echo()
        assert instance.model == "echo-v1"

    def test_custom_model(self):
        from chatnificent.llm import Echo

        instance = Echo(model="echo-custom")
        assert instance.model == "echo-custom"

    def test_kwargs_stored(self):
        from chatnificent.llm import Echo

        instance = Echo(temperature=0.5)
        assert instance.default_params == {"temperature": 0.5}


# ===== extract_content tests =====


class TestExtractContent:
    def test_echo_response(self, echo_llm):
        response = {
            "content": "echoed text",
            "model": "echo-v1",
            "type": "echo_response",
        }
        assert echo_llm.extract_content(response) == "echoed text"

    def test_non_echo_response(self, echo_llm):
        """Non-echo responses are stringified."""
        assert echo_llm.extract_content("raw string") == "raw string"

    def test_dict_without_type(self, echo_llm):
        """Dict without echo_response type is stringified."""
        response = {"content": "text"}
        result = echo_llm.extract_content(response)
        assert isinstance(result, str)


# ===== generate_response tests =====


class TestGenerateResponse:
    def test_echoes_user_prompt(self, echo_llm):
        messages = [{"role": "user", "content": "Hello world"}]
        response = echo_llm.generate_response(messages)
        assert response["type"] == "echo_response"
        assert "Hello world" in response["content"]

    def test_finds_last_user_message(self, echo_llm):
        messages = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Response"},
            {"role": "user", "content": "Second"},
        ]
        response = echo_llm.generate_response(messages)
        assert "Second" in response["content"]
        assert "First" not in response["content"]

    def test_no_user_message(self, echo_llm):
        messages = [{"role": "assistant", "content": "Only assistant"}]
        response = echo_llm.generate_response(messages)
        assert "No user message found" in response["content"]

    def test_tools_noted(self, echo_llm):
        messages = [{"role": "user", "content": "Hi"}]
        tools = [{"type": "function", "function": {"name": "fn"}}]
        response = echo_llm.generate_response(messages, tools=tools)
        assert "Tools were provided but ignored" in response["content"]

    def test_model_override(self, echo_llm):
        messages = [{"role": "user", "content": "Hi"}]
        response = echo_llm.generate_response(messages, model="custom-echo")
        assert response["model"] == "custom-echo"

    def test_structured_content(self, echo_llm):
        """List content produces [Structured Input] marker."""
        messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        response = echo_llm.generate_response(messages)
        assert "Structured Input" in response["content"]


# ===== Default method behavior =====


class TestDefaultMethods:
    """Echo doesn't override parse_tool_calls or create_assistant_message deeply."""

    def test_parse_tool_calls_returns_none(self, echo_llm):
        response = {"content": "text", "type": "echo_response"}
        assert echo_llm.parse_tool_calls(response) is None

    def test_create_assistant_message(self, echo_llm):
        response = {"content": "echoed", "model": "echo-v1", "type": "echo_response"}
        msg = echo_llm.create_assistant_message(response)
        assert msg["role"] == ASSISTANT_ROLE
        assert msg["content"] == "echoed"
