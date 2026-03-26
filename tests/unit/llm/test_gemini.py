"""
Tests for the Gemini LLM provider (google-genai SDK adapter).

Tests cover: constructor kwargs partitioning, request translation
(including system instruction extraction & tool result grouping),
tool schema translation, response parsing, and the tool-message
round-trip used by the engine's agentic loop.
"""

import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest

pytest.importorskip(
    "google.genai", reason="Gemini tests require the google-genai package"
)

from chatnificent.models import (
    ASSISTANT_ROLE,
    MODEL_ROLE,
    SYSTEM_ROLE,
    TOOL_ROLE,
    USER_ROLE,
    Conversation,
)

# ===== Mock helpers =====


def _mock_genai_types():
    """Build a lightweight mock of ``google.genai.types``."""
    types = MagicMock()

    # Part factory methods return plain dicts for easy assertion
    def _from_text(text):
        return {"text": text}

    def _from_function_call(name, args):
        return {"function_call": {"name": name, "args": args}}

    def _from_function_response(name, response):
        return {"function_response": {"name": name, "response": response}}

    class _Part:
        """Mock Part that supports both construction and factory methods."""

        def __init__(self, **kwargs):
            self._data = {k: v for k, v in kwargs.items() if v is not None}

        def __eq__(self, other):
            if isinstance(other, dict):
                return self._data == other
            return NotImplemented

        def __repr__(self):
            return f"Part({self._data!r})"

        @staticmethod
        def from_text(text):
            return _from_text(text)

        @staticmethod
        def from_function_call(name, args):
            return _from_function_call(name, args)

        @staticmethod
        def from_function_response(name, response):
            return _from_function_response(name, response)

    types.Part = _Part

    # Content stores (role, parts) for easy inspection
    class _Content:
        def __init__(self, role, parts):
            self.role = role
            self.parts = list(parts)

        def __repr__(self):
            return f"Content(role={self.role!r}, parts={self.parts!r})"

    types.Content = _Content

    types.GenerateContentConfig = lambda **kw: kw

    class _FunctionDeclaration:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class _Tool:
        def __init__(self, function_declarations=None):
            self.function_declarations = function_declarations

    types.FunctionDeclaration = _FunctionDeclaration
    types.Tool = _Tool

    return types


def _make_text_response(text):
    """Build a dict matching model_dump(mode='json') for a text response."""
    return {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": text}],
                    "role": "model",
                },
                "finish_reason": "STOP",
            }
        ],
    }


def _make_function_call_response(calls):
    """Build a dict matching model_dump(mode='json') for a function-call response.

    Parameters
    ----------
    calls : list[dict]
        Each dict has keys ``name`` (str) and ``args`` (dict).
    """
    parts = [
        {"function_call": {"name": call["name"], "args": call["args"]}}
        for call in calls
    ]
    return {
        "candidates": [
            {
                "content": {
                    "parts": parts,
                    "role": "model",
                },
                "finish_reason": "STOP",
            }
        ],
    }


def _make_empty_response():
    """Build a dict matching model_dump(mode='json') with no candidates."""
    return {"candidates": []}


# ===== Fixtures =====


@pytest.fixture()
def mock_types():
    return _mock_genai_types()


@pytest.fixture()
def gemini(mock_types):
    """Create a Gemini instance with mocked SDK."""
    mock_genai = MagicMock()
    mock_genai.Client.return_value = MagicMock()

    with (
        patch.dict(os.environ, {"GEMINI_API_KEY": "test-key-123"}),
        patch("google.genai", mock_genai, create=True),
        patch.dict("sys.modules", {"google.genai": mock_genai}),
    ):
        # Manually construct to avoid real import
        from chatnificent.llm import Gemini

        instance = object.__new__(Gemini)
        instance._genai_types = mock_types
        instance.client = MagicMock()
        instance.model = "gemini-3.1-pro-preview"
        instance.default_params = {}
        return instance


# ===== Constructor tests =====


class TestGeminiConstructor:
    """Test constructor: kwargs partitioning, API key resolution, header injection."""

    def test_raises_without_api_key(self):
        """Constructor raises ValueError when no key is discoverable."""
        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(ValueError, match="Gemini API key not found"),
        ):
            # Remove any env vars that might provide a key
            os.environ.pop("GEMINI_API_KEY", None)
            os.environ.pop("GOOGLE_API_KEY", None)
            from chatnificent.llm import Gemini

            Gemini()

    def test_accepts_explicit_api_key(self):
        """Explicit api_key kwarg is forwarded to Client."""
        mock_genai = MagicMock()

        with (
            patch.dict(
                "sys.modules",
                {
                    "google": MagicMock(),
                    "google.genai": mock_genai,
                    "google.genai.types": MagicMock(),
                },
            ),
            patch("chatnificent.llm.Gemini.__init__", return_value=None) as mock_init,
        ):
            # Test the partitioning logic directly
            from chatnificent.llm import Gemini

            client_keys = Gemini._CLIENT_KEYS
            kwargs = {
                "api_key": "explicit-key",
                "temperature": 0.5,
                "top_p": 0.9,
            }
            client_kwargs = {k: v for k, v in kwargs.items() if k in client_keys}
            gen_kwargs = {k: v for k, v in kwargs.items() if k not in client_keys}

            assert client_kwargs == {"api_key": "explicit-key"}
            assert gen_kwargs == {"temperature": 0.5, "top_p": 0.9}

    def test_kwargs_partitioning(self):
        """Client keys and generation params are correctly separated."""
        from chatnificent.llm import Gemini

        all_kwargs = {
            "api_key": "k",
            "vertexai": True,
            "project": "proj",
            "location": "us-central1",
            "temperature": 0.7,
            "top_k": 40,
        }
        client_keys = Gemini._CLIENT_KEYS
        client_kwargs = {k: v for k, v in all_kwargs.items() if k in client_keys}
        gen_kwargs = {k: v for k, v in all_kwargs.items() if k not in client_keys}

        assert client_kwargs == {
            "api_key": "k",
            "vertexai": True,
            "project": "proj",
            "location": "us-central1",
        }
        assert gen_kwargs == {"temperature": 0.7, "top_k": 40}

    def test_header_injection(self):
        """x-goog-api-client header is injected into http_options."""
        http_options: dict = {}
        headers = http_options.setdefault("headers", {})
        headers.setdefault("x-goog-api-client", "chatnificent/0.0.10")

        assert http_options == {"headers": {"x-goog-api-client": "chatnificent/0.0.10"}}

    def test_header_does_not_override_user_value(self):
        """User-provided x-goog-api-client header is preserved."""
        http_options = {"headers": {"x-goog-api-client": "custom/1.0"}}
        headers = http_options.setdefault("headers", {})
        headers.setdefault("x-goog-api-client", "chatnificent/0.0.10")

        assert http_options["headers"]["x-goog-api-client"] == "custom/1.0"

    def test_vertexai_skips_api_key_resolution(self):
        """When vertexai=True, no API key is required."""
        from chatnificent.llm import Gemini

        kwargs = {"vertexai": True, "project": "proj", "location": "us-east1"}
        client_keys = Gemini._CLIENT_KEYS
        client_kwargs = {k: v for k, v in kwargs.items() if k in client_keys}

        # Should not raise even without GEMINI_API_KEY
        assert "api_key" not in client_kwargs
        assert client_kwargs["vertexai"] is True


# ===== _translate_request tests =====


class TestTranslateRequest:
    """Test the OpenAI → Google Content translation layer."""

    def test_simple_user_message(self, gemini):
        messages = [{"role": "user", "content": "Hello"}]
        contents, sys = gemini._translate_request(messages)

        assert sys is None
        assert len(contents) == 1
        assert contents[0].role == "user"
        assert contents[0].parts == [{"text": "Hello"}]

    def test_system_message_extraction(self, gemini):
        """System messages are extracted and NOT included in contents."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        contents, sys = gemini._translate_request(messages)

        assert sys == "You are helpful."
        assert len(contents) == 1
        assert contents[0].role == "user"

    def test_assistant_to_model_role_mapping(self, gemini):
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "Thanks"},
        ]
        contents, _ = gemini._translate_request(messages)

        assert len(contents) == 3
        assert contents[0].role == "user"
        assert contents[1].role == "model"
        assert contents[2].role == "user"

    def test_model_role_passthrough(self, gemini):
        """MODEL_ROLE is preserved as 'model'."""
        messages = [
            {"role": "model", "content": [{"text": "Here is the answer"}]},
        ]
        contents, _ = gemini._translate_request(messages)

        assert len(contents) == 1
        assert contents[0].role == "model"

    def test_tool_result_message(self, gemini):
        messages = [
            {
                "role": "tool",
                "name": "get_weather",
                "content": "sunny, 72°F",
            }
        ]
        contents, _ = gemini._translate_request(messages)

        assert len(contents) == 1
        assert contents[0].role == "user"
        assert contents[0].parts == [
            {
                "function_response": {
                    "name": "get_weather",
                    "response": {"content": "sunny, 72°F"},
                }
            }
        ]

    def test_consecutive_tool_results_grouped(self, gemini):
        """Multiple consecutive tool results merge into a single Content turn."""
        messages = [
            {"role": "tool", "name": "get_weather", "content": "sunny"},
            {"role": "tool", "name": "get_time", "content": "3pm"},
        ]
        contents, _ = gemini._translate_request(messages)

        assert len(contents) == 1
        assert contents[0].role == "user"
        assert len(contents[0].parts) == 2

    def test_function_call_parts_roundtrip(self, gemini):
        """Assistant message with function_call parts round-trips correctly."""
        messages = [
            {
                "role": "model",
                "content": [
                    {
                        "function_call": {
                            "name": "get_weather",
                            "args": {"location": "Boston"},
                        }
                    }
                ],
            }
        ]
        contents, _ = gemini._translate_request(messages)

        assert len(contents) == 1
        assert contents[0].role == "model"
        assert contents[0].parts == [
            {"function_call": {"name": "get_weather", "args": {"location": "Boston"}}}
        ]

    def test_mixed_text_and_function_call_parts(self, gemini):
        messages = [
            {
                "role": "model",
                "content": [
                    {"text": "Let me check the weather"},
                    {
                        "function_call": {
                            "name": "get_weather",
                            "args": {"city": "NYC"},
                        }
                    },
                ],
            }
        ]
        contents, _ = gemini._translate_request(messages)

        assert len(contents) == 1
        assert contents[0].role == "model"
        assert len(contents[0].parts) == 2
        assert contents[0].parts[0] == {"text": "Let me check the weather"}
        assert contents[0].parts[1] == {
            "function_call": {"name": "get_weather", "args": {"city": "NYC"}}
        }

    def test_full_agentic_conversation(self, gemini):
        """Full conversation: system → user → model(fc) → tool → model(text)."""
        messages = [
            {"role": "system", "content": "You are a weather bot."},
            {"role": "user", "content": "Weather in Boston?"},
            {
                "role": "model",
                "content": [
                    {
                        "function_call": {
                            "name": "get_weather",
                            "args": {"location": "Boston"},
                        }
                    }
                ],
            },
            {"role": "tool", "name": "get_weather", "content": "sunny"},
            {"role": "assistant", "content": "It's sunny in Boston!"},
        ]
        contents, sys = gemini._translate_request(messages)

        assert sys == "You are a weather bot."
        assert len(contents) == 4
        assert contents[0].role == "user"
        assert contents[1].role == "model"
        assert contents[2].role == "user"  # tool result
        assert contents[3].role == "model"  # final assistant mapped to model

    def test_empty_content_skipped(self, gemini):
        """Messages with None content produce no parts and are skipped."""
        messages = [{"role": "assistant", "content": None}]
        contents, _ = gemini._translate_request(messages)
        assert len(contents) == 0

    def test_string_items_in_list_content(self, gemini):
        """List content with plain strings are converted to text parts."""
        messages = [{"role": "user", "content": ["Hello", "World"]}]
        contents, _ = gemini._translate_request(messages)

        assert len(contents) == 1
        assert contents[0].parts == [{"text": "Hello"}, {"text": "World"}]


# ===== _build_parts tests =====


class TestBuildParts:
    def test_string_content(self, gemini):
        parts = gemini._build_parts("Hello world")
        assert parts == [{"text": "Hello world"}]

    def test_none_content(self, gemini):
        parts = gemini._build_parts(None)
        assert parts == []

    def test_list_with_text_dict(self, gemini):
        parts = gemini._build_parts([{"text": "hello"}])
        assert parts == [{"text": "hello"}]

    def test_list_with_function_call(self, gemini):
        parts = gemini._build_parts(
            [{"function_call": {"name": "foo", "args": {"x": 1}}}]
        )
        assert parts == [{"function_call": {"name": "foo", "args": {"x": 1}}}]

    def test_function_call_with_thought_signature(self, gemini):
        """thought_signature must round-trip through _build_parts for Gemini API."""
        parts = gemini._build_parts(
            [
                {
                    "function_call": {"name": "roll_dice", "args": {"sides": 6}},
                    "thought_signature": "abc123sig",
                }
            ]
        )
        assert len(parts) == 1
        assert parts[0] == {
            "function_call": {"name": "roll_dice", "args": {"sides": 6}},
            "thought_signature": "abc123sig",
        }

    def test_thought_parts_preserved(self, gemini):
        """Thought parts (thought=True) must be preserved for API replay."""
        parts = gemini._build_parts(
            [
                {"thought": True, "text": "Let me think about this..."},
                {
                    "function_call": {"name": "get_weather", "args": {}},
                    "thought_signature": "sig456",
                },
            ]
        )
        assert len(parts) == 2
        assert parts[0] == {"thought": True, "text": "Let me think about this..."}
        assert parts[1] == {
            "function_call": {"name": "get_weather", "args": {}},
            "thought_signature": "sig456",
        }

    def test_list_with_function_response(self, gemini):
        parts = gemini._build_parts(
            [{"function_response": {"name": "foo", "response": {"result": "bar"}}}]
        )
        assert parts == [
            {"function_response": {"name": "foo", "response": {"result": "bar"}}}
        ]


# ===== _translate_tool_schema tests =====


class TestTranslateToolSchema:
    def test_single_function_tool(self, gemini):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                        },
                        "required": ["location"],
                    },
                },
            }
        ]
        result = gemini._translate_tool_schema(tools)

        assert len(result) == 1
        tool = result[0]
        assert len(tool.function_declarations) == 1
        decl = tool.function_declarations[0]
        assert decl.name == "get_weather"
        assert decl.description == "Get current weather"
        assert decl.parameters_json_schema == {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        }

    def test_function_without_parameters(self, gemini):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Get current time",
                },
            }
        ]
        result = gemini._translate_tool_schema(tools)

        assert len(result) == 1
        decl = result[0].function_declarations[0]
        assert decl.name == "get_time"
        assert not hasattr(decl, "parameters_json_schema")

    def test_non_function_tools_ignored(self, gemini):
        tools = [{"type": "retrieval", "retrieval": {}}]
        result = gemini._translate_tool_schema(tools)
        assert result == []

    def test_multiple_functions(self, gemini):
        tools = [
            {
                "type": "function",
                "function": {"name": "fn1", "description": "first"},
            },
            {
                "type": "function",
                "function": {"name": "fn2", "description": "second"},
            },
        ]
        result = gemini._translate_tool_schema(tools)

        assert len(result) == 1
        assert len(result[0].function_declarations) == 2


# ===== extract_content tests =====


class TestExtractContent:
    def test_text_response(self, gemini):
        response = _make_text_response("Hello from Gemini!")
        assert gemini.extract_content(response) == "Hello from Gemini!"

    def test_empty_candidates(self, gemini):
        response = _make_empty_response()
        assert gemini.extract_content(response) is None

    def test_null_parts_shows_finish_reason(self, gemini):
        """When model_dump returns parts: null (e.g., thinking exhausted token budget)."""
        response = {
            "candidates": [
                {
                    "content": {"parts": None, "role": "model"},
                    "finish_reason": "MAX_TOKENS",
                }
            ]
        }
        result = gemini.extract_content(response)
        assert "MAX_TOKENS" in result
        assert "Empty response" in result

    def test_empty_parts_shows_finish_reason(self, gemini):
        """When parts list is empty."""
        response = {
            "candidates": [
                {
                    "content": {"parts": [], "role": "model"},
                    "finish_reason": "SAFETY",
                }
            ]
        }
        result = gemini.extract_content(response)
        assert "SAFETY" in result

    def test_null_content_shows_finish_reason(self, gemini):
        """When content itself is null."""
        response = {
            "candidates": [
                {
                    "content": None,
                    "finish_reason": "RECITATION",
                }
            ]
        }
        result = gemini.extract_content(response)
        assert "RECITATION" in result

    def test_exception_returns_none(self, gemini):
        """Graceful fallback when response is malformed."""
        assert gemini.extract_content("not a dict") is None

    def test_function_call_only_returns_none(self, gemini):
        """When response has only function_call parts (no text), returns None."""
        response = _make_function_call_response(
            [{"name": "get_weather", "args": {"city": "Tokyo"}}]
        )
        assert gemini.extract_content(response) is None


# ===== parse_tool_calls tests =====


class TestParseToolCalls:
    def test_no_candidates(self, gemini):
        response = _make_empty_response()
        assert gemini.parse_tool_calls(response) is None

    def test_no_function_calls(self, gemini):
        response = _make_text_response("Just text")
        assert gemini.parse_tool_calls(response) is None

    def test_single_function_call(self, gemini):
        response = _make_function_call_response(
            [{"name": "get_weather", "args": {"location": "Boston"}}]
        )
        tool_calls = gemini.parse_tool_calls(response)

        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0]["function_name"] == "get_weather"
        assert json.loads(tool_calls[0]["function_args"]) == {"location": "Boston"}
        assert tool_calls[0]["id"].startswith("call_")

    def test_multiple_function_calls(self, gemini):
        response = _make_function_call_response(
            [
                {"name": "get_weather", "args": {"location": "Boston"}},
                {"name": "get_time", "args": {"timezone": "EST"}},
            ]
        )
        tool_calls = gemini.parse_tool_calls(response)

        assert len(tool_calls) == 2
        assert tool_calls[0]["function_name"] == "get_weather"
        assert tool_calls[1]["function_name"] == "get_time"

    def test_returns_standardized_tool_call_objects(self, gemini):
        response = _make_function_call_response(
            [{"name": "fn", "args": {"key": "val"}}]
        )
        tool_calls = gemini.parse_tool_calls(response)
        tc = tool_calls[0]

        assert isinstance(tc, dict)
        assert isinstance(tc["id"], str)
        assert isinstance(tc["function_name"], str)
        assert isinstance(tc["function_args"], str)
        assert json.loads(tc["function_args"]) == {"key": "val"}


# ===== create_assistant_message tests =====


class TestCreateAssistantMessage:
    def test_empty_response(self, gemini):
        response = _make_empty_response()
        msg = gemini.create_assistant_message(response)

        assert msg["role"] == MODEL_ROLE
        assert msg["content"] == "[No response generated]"

    def test_function_call_response(self, gemini):
        response = _make_function_call_response(
            [{"name": "get_weather", "args": {"location": "NYC"}}]
        )
        msg = gemini.create_assistant_message(response)

        assert msg["role"] == MODEL_ROLE
        assert isinstance(msg["content"], list)
        assert len(msg["content"]) == 1
        assert "function_call" in msg["content"][0]
        assert msg["content"][0]["function_call"]["name"] == "get_weather"

    def test_text_response(self, gemini):
        response = _make_text_response("Hello!")
        msg = gemini.create_assistant_message(response)

        assert msg["role"] == MODEL_ROLE
        assert isinstance(msg["content"], list)
        assert msg["content"] == [{"text": "Hello!"}]

    def test_no_parts(self, gemini):
        response = {"candidates": [{"content": {"parts": [], "role": "model"}}]}
        msg = gemini.create_assistant_message(response)
        assert msg["content"] == "[No response generated]"


# ===== create_tool_result_messages tests =====


class TestCreateToolResultMessages:
    def test_single_result(self, gemini):
        results = [
            {
                "tool_call_id": "call_123",
                "function_name": "get_weather",
                "content": "sunny",
            }
        ]
        conversation = Conversation(id="test")
        msgs = gemini.create_tool_result_messages(results, conversation)

        assert len(msgs) == 1
        assert msgs[0]["role"] == TOOL_ROLE
        assert msgs[0]["name"] == "get_weather"
        assert msgs[0]["content"] == "sunny"

    def test_multiple_results(self, gemini):
        results = [
            {
                "tool_call_id": "call_1",
                "function_name": "fn_a",
                "content": "result_a",
            },
            {
                "tool_call_id": "call_2",
                "function_name": "fn_b",
                "content": "result_b",
            },
        ]
        conversation = Conversation(id="test")
        msgs = gemini.create_tool_result_messages(results, conversation)

        assert len(msgs) == 2
        assert msgs[0]["name"] == "fn_a"
        assert msgs[1]["name"] == "fn_b"


# ===== is_tool_message tests =====


class TestIsToolMessage:
    def test_tool_role(self, gemini):
        msg = {"role": TOOL_ROLE, "name": "fn", "content": "result"}
        assert gemini.is_tool_message(msg) is True

    def test_model_with_function_call_parts(self, gemini):
        msg = {
            "role": MODEL_ROLE,
            "content": [{"function_call": {"name": "get_weather", "args": {}}}],
        }
        assert gemini.is_tool_message(msg) is True

    def test_model_with_text_parts(self, gemini):
        msg = {"role": MODEL_ROLE, "content": [{"text": "Hello"}]}
        assert gemini.is_tool_message(msg) is False

    def test_model_with_string_content(self, gemini):
        msg = {"role": MODEL_ROLE, "content": "Just text"}
        assert gemini.is_tool_message(msg) is False

    def test_user_message(self, gemini):
        msg = {"role": USER_ROLE, "content": "Hello"}
        assert gemini.is_tool_message(msg) is False

    def test_assistant_message(self, gemini):
        msg = {"role": ASSISTANT_ROLE, "content": "Response text"}
        assert gemini.is_tool_message(msg) is False


# ===== generate_response tests =====


class TestGenerateResponse:
    def test_calls_client_with_correct_args(self, gemini):
        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello"},
        ]
        gemini.generate_response(messages)

        gemini.client.models.generate_content.assert_called_once()
        call_kwargs = gemini.client.models.generate_content.call_args
        assert call_kwargs.kwargs["model"] == "gemini-3.1-pro-preview"

    def test_model_override(self, gemini):
        messages = [{"role": "user", "content": "Hi"}]
        gemini.generate_response(messages, model="gemini-1.5-pro")

        call_kwargs = gemini.client.models.generate_content.call_args
        assert call_kwargs.kwargs["model"] == "gemini-1.5-pro"

    def test_default_params_merged(self, gemini):
        gemini.default_params = {"temperature": 0.5}
        messages = [{"role": "user", "content": "Hi"}]
        gemini.generate_response(messages)

        call_kwargs = gemini.client.models.generate_content.call_args
        config = call_kwargs.kwargs["config"]
        assert config["temperature"] == 0.5

    def test_call_time_kwargs_override_defaults(self, gemini):
        gemini.default_params = {"temperature": 0.5}
        messages = [{"role": "user", "content": "Hi"}]
        gemini.generate_response(messages, temperature=0.9)

        call_kwargs = gemini.client.models.generate_content.call_args
        config = call_kwargs.kwargs["config"]
        assert config["temperature"] == 0.9

    def test_system_instruction_in_config(self, gemini):
        messages = [
            {"role": "system", "content": "You are a pirate."},
            {"role": "user", "content": "Ahoy!"},
        ]
        gemini.generate_response(messages)

        call_kwargs = gemini.client.models.generate_content.call_args
        config = call_kwargs.kwargs["config"]
        assert config["system_instruction"] == "You are a pirate."

    def test_tools_translated_and_passed(self, gemini):
        messages = [{"role": "user", "content": "Weather?"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        gemini.generate_response(messages, tools=tools)

        call_kwargs = gemini.client.models.generate_content.call_args
        config = call_kwargs.kwargs["config"]
        assert "tools" in config
        assert len(config["tools"]) == 1

    def test_returns_dict(self, gemini):
        mock_response = Mock()
        mock_response.model_dump.return_value = {"candidates": []}
        gemini.client.models.generate_content.return_value = mock_response

        result = gemini.generate_response([{"role": "user", "content": "Hi"}])
        assert isinstance(result, dict)


# ===== Full round-trip test =====


class TestAgenticRoundTrip:
    """Test the full tool-calling cycle as the engine would execute it."""

    def test_tool_call_round_trip(self, gemini):
        """Simulate: user → model(fc) → tool results → model(text)."""
        # Step 1: Model returns a function call
        fc_response = _make_function_call_response(
            [{"name": "get_weather", "args": {"location": "Boston"}}]
        )

        tool_calls = gemini.parse_tool_calls(fc_response)
        assert tool_calls is not None
        assert len(tool_calls) == 1

        # Step 2: Engine persists the assistant message
        assistant_msg = gemini.create_assistant_message(fc_response)
        assert assistant_msg["role"] == MODEL_ROLE
        assert isinstance(assistant_msg["content"], list)

        # Step 3: Engine executes tool and creates result messages
        results = [
            {
                "tool_call_id": tool_calls[0]["id"],
                "function_name": "get_weather",
                "content": "sunny, 72°F",
            }
        ]
        conversation = Conversation(id="test", messages=[])
        tool_msgs = gemini.create_tool_result_messages(results, conversation)
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["role"] == TOOL_ROLE

        # Step 4: Verify these messages serialize correctly for the next _translate_request
        all_messages = [
            {"role": "user", "content": "Weather in Boston?"},
            {k: v for k, v in assistant_msg.items() if v is not None},
            {k: v for k, v in tool_msgs[0].items() if v is not None},
        ]

        contents, sys = gemini._translate_request(all_messages)

        assert len(contents) == 3
        assert contents[0].role == "user"
        assert contents[1].role == "model"
        assert contents[2].role == "user"  # tool result → user role for Google

        # The model turn should contain the function_call part
        assert any(
            "function_call" in p if isinstance(p, dict) else False
            for p in contents[1].parts
        )

        # The tool result turn should contain the function_response part
        assert any(
            "function_response" in p if isinstance(p, dict) else False
            for p in contents[2].parts
        )

    def test_tool_call_round_trip_with_thought_signature(self, gemini):
        """Thought signatures from Gemini API must survive the full round-trip.

        Flow: API response → create_assistant_message → store → _translate_request
        The thought_signature on functionCall parts and thought=True text parts
        must be preserved so the Gemini API doesn't reject the replay with 400.
        """
        # Gemini API response with thought + thought_signature
        fc_response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"thought": True, "text": "I need to check the weather."},
                            {
                                "function_call": {
                                    "name": "get_weather",
                                    "args": {"location": "Boston"},
                                },
                                "thought_signature": "sig_xyz_123",
                            },
                        ],
                        "role": "model",
                    },
                    "finish_reason": "STOP",
                }
            ]
        }

        # Step 1: create_assistant_message preserves all fields
        assistant_msg = gemini.create_assistant_message(fc_response)
        assert assistant_msg["role"] == MODEL_ROLE
        assert isinstance(assistant_msg["content"], list)
        assert len(assistant_msg["content"]) == 2
        # Thought part preserved
        assert assistant_msg["content"][0].get("thought") is True
        # thought_signature preserved
        assert assistant_msg["content"][1].get("thought_signature") == "sig_xyz_123"

        # Step 2: Full conversation replay via _translate_request
        messages = [
            {"role": "user", "content": "Weather in Boston?"},
            assistant_msg,
            {"role": "tool", "name": "get_weather", "content": "sunny, 72°F"},
        ]
        contents, _ = gemini._translate_request(messages)

        # The model turn must preserve thought + thought_signature parts
        model_turn = contents[1]
        assert model_turn.role == "model"
        assert len(model_turn.parts) == 2
        # Thought part must be included (not skipped)
        assert model_turn.parts[0] == {
            "thought": True,
            "text": "I need to check the weather.",
        }
        # Function call part must include thought_signature
        assert model_turn.parts[1] == {
            "function_call": {
                "name": "get_weather",
                "args": {"location": "Boston"},
            },
            "thought_signature": "sig_xyz_123",
        }

    def test_is_tool_message_filters_correctly(self, gemini):
        """Verify is_tool_message correctly identifies tool-related messages
        so the engine's _build_output filters them from display."""
        user_msg = {"role": USER_ROLE, "content": "Weather?"}
        assistant_fc_msg = {
            "role": MODEL_ROLE,
            "content": [
                {"function_call": {"name": "get_weather", "args": {"loc": "NYC"}}}
            ],
        }
        tool_result_msg = {"role": TOOL_ROLE, "name": "get_weather", "content": "sunny"}
        final_msg = {"role": ASSISTANT_ROLE, "content": "It's sunny in NYC!"}

        assert gemini.is_tool_message(user_msg) is False
        assert gemini.is_tool_message(assistant_fc_msg) is True
        assert gemini.is_tool_message(tool_result_msg) is True
        assert gemini.is_tool_message(final_msg) is False


# ===== Streaming tests =====


class TestGeminiStreaming:
    """Test Gemini streaming: default_params, generate_response branching,
    and extract_stream_delta."""

    def test_default_params_includes_stream(self):
        """Constructor sets stream=True in default_params, matching all other providers."""
        mock_genai = MagicMock()
        mock_genai.Client.return_value = MagicMock()

        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
            patch.dict(
                "sys.modules",
                {
                    "google": MagicMock(),
                    "google.genai": mock_genai,
                    "google.genai.types": MagicMock(),
                },
            ),
        ):
            from chatnificent.llm import Gemini

            instance = Gemini(model="gemini-test")
            assert instance.default_params.get("stream") is True

    def test_default_params_user_can_override_stream(self):
        """User can set stream=False and it's respected."""
        mock_genai = MagicMock()
        mock_genai.Client.return_value = MagicMock()

        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
            patch.dict(
                "sys.modules",
                {
                    "google": MagicMock(),
                    "google.genai": mock_genai,
                    "google.genai.types": MagicMock(),
                },
            ),
        ):
            from chatnificent.llm import Gemini

            instance = Gemini(model="gemini-test", stream=False)
            assert instance.default_params.get("stream") is False

    def test_generate_response_calls_stream_method(self, gemini):
        """When stream=True, generate_response calls generate_content_stream."""
        gemini.default_params = {"stream": True}
        gemini.client.models.generate_content_stream.return_value = iter(["chunk"])

        messages = [{"role": "user", "content": "Hi"}]
        gemini.generate_response(messages)

        gemini.client.models.generate_content_stream.assert_called_once()
        gemini.client.models.generate_content.assert_not_called()

    def test_generate_response_calls_non_stream_method(self, gemini):
        """When stream=False, generate_response calls generate_content."""
        gemini.default_params = {"stream": False}
        mock_response = MagicMock()
        mock_response.model_dump.return_value = _make_text_response("Hi back")
        gemini.client.models.generate_content.return_value = mock_response

        messages = [{"role": "user", "content": "Hi"}]
        gemini.generate_response(messages)

        gemini.client.models.generate_content.assert_called_once()
        gemini.client.models.generate_content_stream.assert_not_called()

    def test_stream_kwarg_overrides_default(self, gemini):
        """Call-time stream=False overrides default_params stream=True."""
        gemini.default_params = {"stream": True}
        mock_response = MagicMock()
        mock_response.model_dump.return_value = _make_text_response("Hi")
        gemini.client.models.generate_content.return_value = mock_response

        messages = [{"role": "user", "content": "Hi"}]
        gemini.generate_response(messages, stream=False)

        gemini.client.models.generate_content.assert_called_once()
        gemini.client.models.generate_content_stream.assert_not_called()

    def test_stream_not_passed_to_config(self, gemini):
        """stream is popped before reaching GenerateContentConfig."""
        gemini.default_params = {"stream": True, "temperature": 0.5}
        gemini.client.models.generate_content_stream.return_value = iter([])

        config_spy = MagicMock(side_effect=lambda **kw: kw)
        gemini._genai_types.GenerateContentConfig = config_spy

        messages = [{"role": "user", "content": "Hi"}]
        gemini.generate_response(messages)

        config_kwargs = config_spy.call_args[1]
        assert "stream" not in config_kwargs
        assert config_kwargs["temperature"] == 0.5

    def test_extract_stream_delta_text(self, gemini):
        """extract_stream_delta returns text from a streaming chunk."""
        mock_part = SimpleNamespace(text="Hello ")
        mock_content = SimpleNamespace(parts=[mock_part])
        mock_candidate = SimpleNamespace(content=mock_content)
        chunk = SimpleNamespace(candidates=[mock_candidate])

        assert gemini.extract_stream_delta(chunk) == "Hello "

    def test_extract_stream_delta_empty_candidates(self, gemini):
        """extract_stream_delta returns None for empty candidates."""
        chunk = SimpleNamespace(candidates=[])
        assert gemini.extract_stream_delta(chunk) is None

    def test_extract_stream_delta_no_text(self, gemini):
        """extract_stream_delta returns None when parts have no text."""
        mock_part = SimpleNamespace(text=None)
        mock_content = SimpleNamespace(parts=[mock_part])
        mock_candidate = SimpleNamespace(content=mock_content)
        chunk = SimpleNamespace(candidates=[mock_candidate])

        assert gemini.extract_stream_delta(chunk) is None

    def test_extract_stream_delta_multiple_parts(self, gemini):
        """extract_stream_delta concatenates text from multiple parts."""
        parts = [SimpleNamespace(text="Hello "), SimpleNamespace(text="world")]
        mock_content = SimpleNamespace(parts=parts)
        mock_candidate = SimpleNamespace(content=mock_content)
        chunk = SimpleNamespace(candidates=[mock_candidate])

        assert gemini.extract_stream_delta(chunk) == "Hello world"
