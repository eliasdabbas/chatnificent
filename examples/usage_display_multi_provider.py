# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "chatnificent[openai,anthropic,gemini]",
# ]
# ///
"""
Usage Display Multi-Provider — A Portable Display Enrichment Pattern
====================================================================

This is the more robust companion to ``usage_display.py``.

The layout pattern stays the same:

1. call ``super().render_messages(...)``
2. load raw API responses from the store
3. extract usage from each saved payload
4. append a usage footer to the visible assistant message

What changes here is the payload parsing. OpenAI, Anthropic, and Gemini expose
usage metadata differently, so this example keeps a few small helpers inside
the script to normalize those shapes while leaving the framework untouched.

Prerequisites
-------------
Set the API key for the provider you want to use::

    export OPENAI_API_KEY="sk-..."
    # or
    export ANTHROPIC_API_KEY="sk-ant-..."
    # or
    export GOOGLE_API_KEY="AI..."

The active default below is OpenAI, and it requests streamed usage explicitly
with ``stream_options={"include_usage": True}``.

Running
-------
::

    uv run --script examples/usage_display_multi_provider.py

Swap providers by uncommenting a different ``llm=...`` line in the app
definition below.

What to Explore Next
--------------------
- Extend the formatter with cached tokens or reasoning token fields
- Add a per-turn cost estimate beside the token count
- Combine this with ``conversation_summary.py`` or ``display_redaction.py``
"""

from typing import Any, Optional

import chatnificent as chat

BASE_DIR = "usage_convos_multi_provider"


class UsageLayout(chat.layout.DefaultLayout):
    """Append token usage beneath assistant messages across providers."""

    def render_messages(self, messages, **kwargs):
        rendered = super().render_messages(messages, **kwargs)
        user_id = kwargs["user_id"]
        conversation = kwargs["conversation"]
        raw_responses = self.app.store.load_raw_api_responses(user_id, conversation.id)

        usage_lines = []
        for raw_response in raw_responses:
            usage_line = _usage_line(raw_response)
            if usage_line:
                usage_lines.append(usage_line)

        assistant_index = 0
        for message in rendered:
            if message.get("role") != "assistant":
                continue
            if assistant_index >= len(usage_lines):
                break
            message["content"] = (
                f"{message['content']}\n\n{usage_lines[assistant_index]}"
            )
            assistant_index += 1

        return rendered


def _usage_line(raw_response: Any) -> Optional[str]:
    """Format usage from OpenAI, Anthropic, or Gemini raw payloads."""
    usage = _find_usage(raw_response)
    if not usage:
        return None

    if {"prompt_tokens", "completion_tokens", "total_tokens"} <= usage.keys():
        prompt_tokens = usage["prompt_tokens"]
        completion_tokens = usage["completion_tokens"]
        total_tokens = usage["total_tokens"]
    elif {"input_tokens", "output_tokens"} <= usage.keys():
        prompt_tokens = usage["input_tokens"]
        completion_tokens = usage["output_tokens"]
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
    elif {
        "prompt_token_count",
        "candidates_token_count",
        "total_token_count",
    } <= usage.keys():
        prompt_tokens = usage["prompt_token_count"]
        completion_tokens = usage["candidates_token_count"]
        total_tokens = usage["total_token_count"]
    else:
        return None

    return f"Usage: ↑ {prompt_tokens} + ↓ {completion_tokens} = {total_tokens} Tokens"


def _find_usage(value: Any) -> Optional[dict]:
    """Recursively scan common raw payload shapes for usage metadata."""
    if isinstance(value, dict):
        if _looks_like_usage(value):
            return value
        for key in ("usage", "usage_metadata", "message"):
            usage = _find_usage(value.get(key))
            if usage:
                return usage
        for nested in value.values():
            usage = _find_usage(nested)
            if usage:
                return usage
        return None

    if isinstance(value, list):
        for item in reversed(value):
            usage = _find_usage(item)
            if usage:
                return usage

    return None


def _looks_like_usage(value: dict) -> bool:
    """Identify usage dicts across the supported providers."""
    return any(
        keys <= value.keys()
        for keys in (
            {"prompt_tokens", "completion_tokens", "total_tokens"},
            {"input_tokens", "output_tokens"},
            {"prompt_token_count", "candidates_token_count", "total_token_count"},
        )
    )


app = chat.Chatnificent(
    llm=chat.llm.OpenAI(stream_options={"include_usage": True}),
    # llm=chat.llm.Gemini(),
    # llm=chat.llm.Anthropic(),
    layout=UsageLayout(),
    store=chat.store.File(base_dir=BASE_DIR),
)

if __name__ == "__main__":
    app.run()
