"""Shared fixtures for LLM provider tests."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# ===== OpenAI-compatible response builders =====


def make_openai_response(content="Hello", finish_reason="stop", tool_calls=None):
    """Build a mock OpenAI ChatCompletion response."""
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(
        message=message,
        finish_reason=finish_reason,
    )
    return SimpleNamespace(choices=[choice])


def make_openai_empty_response():
    """Build a mock OpenAI response with no choices."""
    return SimpleNamespace(choices=[])


def make_openai_tool_response(calls):
    """Build a mock OpenAI response with tool calls.

    Parameters
    ----------
    calls : list[dict]
        Each dict has ``id``, ``name``, ``arguments`` (JSON string).
    """
    tool_calls = []
    for c in calls:
        tc = SimpleNamespace(
            id=c["id"],
            type="function",
            function=SimpleNamespace(
                name=c["name"],
                arguments=c["arguments"],
            ),
        )
        tc.model_dump = lambda _tc=tc: {
            "id": _tc.id,
            "type": "function",
            "function": {
                "name": _tc.function.name,
                "arguments": _tc.function.arguments,
            },
        }
        tool_calls.append(tc)

    message = SimpleNamespace(content=None, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason="tool_calls")
    return SimpleNamespace(choices=[choice])


# ===== Anthropic response builders =====


def make_anthropic_response(text="Hello", stop_reason="end_turn"):
    """Build a mock Anthropic messages response."""
    block = SimpleNamespace(type="text", text=text)
    return SimpleNamespace(content=[block], stop_reason=stop_reason)


def make_anthropic_empty_response(stop_reason="end_turn"):
    """Build a mock Anthropic response with empty content."""
    return SimpleNamespace(content=[], stop_reason=stop_reason)


def make_anthropic_tool_response(calls):
    """Build a mock Anthropic response with tool_use blocks.

    Parameters
    ----------
    calls : list[dict]
        Each dict has ``id``, ``name``, ``input`` (dict).
    """
    blocks = []
    for c in calls:
        blocks.append(
            SimpleNamespace(
                type="tool_use", id=c["id"], name=c["name"], input=c["input"]
            )
        )
    return SimpleNamespace(content=blocks, stop_reason="tool_use")


# ===== Ollama response builders =====


def make_ollama_response(content="Hello", done_reason="stop"):
    """Build a mock Ollama chat response dict."""
    return {
        "message": {"role": "assistant", "content": content},
        "done_reason": done_reason,
    }


def make_ollama_empty_response(done_reason="stop"):
    """Build a mock Ollama response with empty content."""
    return {
        "message": {"role": "assistant", "content": ""},
        "done_reason": done_reason,
    }


def make_ollama_tool_response(calls):
    """Build a mock Ollama response with tool calls.

    Parameters
    ----------
    calls : list[dict]
        Each dict has ``name`` and ``arguments`` (dict).
    """
    tool_calls = [
        {"function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls
    ]
    return {
        "message": {"role": "assistant", "content": "", "tool_calls": tool_calls},
        "done_reason": "stop",
    }
