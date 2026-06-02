# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "chatnificent[openai,anthropic,gemini]",
# ]
# ///
"""
Append Anything After Any Turn — Across Providers
==================================================

The robust companion to ``usage_display.py``. Same strategic lesson:

    **How to render anything you want after an assistant turn, without
    modifying the conversation history.**

The injection mechanism is the same two-seam pattern as the OpenAI-only
example:

1. **Streaming time — ``UsageEngine``** wraps
   ``Orchestrator.handle_message_stream`` and yields one extra ``delta``
   event right before ``done``, so the footer appears under the live reply
   without ever entering ``messages.json``.

2. **Render time — ``UsageLayout``** overrides ``render_messages`` to read
   the same ``raw_api_responses.jsonl`` sidecar and append the same footer,
   so the footer is also visible on every page refresh, sidebar click, and
   deep link.

What this example adds on top is the *portability* layer. The helpers below
detect each provider's payload by **shape** rather than by name:

- ``_looks_like_usage`` — a set-subset test against the three known
  key vocabularies (OpenAI / Anthropic / Gemini)
- ``_find_usage`` — recursive scan of arbitrarily nested raw payloads
  (lists of chunks, message wrappers, deeply nested usage_metadata blocks)
- ``_usage_line`` — single formatter that normalises all three shapes into
  one display string

The shape-detection pattern generalises beyond usage data. Anywhere you need
to read a sidecar (raw responses, raw requests, retrieval logs, tool traces)
and work across providers, this is the structure to copy.

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
"""

from typing import Any, Optional

import chatnificent as chat

BASE_DIR = "usage_convos_multi_provider"


class UsageEngine(chat.engine.Orchestrator):
    """Streaming-time seam — inject the footer as a final ``delta`` event.

    ``Orchestrator.handle_message_stream`` calls ``_save_conversation`` BEFORE
    yielding ``done``. Anything we yield between the parent generator's last
    item and the ``done`` event rides the existing SSE channel to the client
    without ever touching ``messages.json``.

    To display something else, swap ``_usage_line`` for your own formatter;
    everything else stays the same.
    """

    def handle_message_stream(self, user_input, user_id, convo_id_from_url):
        for event in super().handle_message_stream(
            user_input, user_id, convo_id_from_url
        ):
            if event.get("event") == "done":
                convo_id = event["data"]["conversation_id"]
                raw_responses = self.app.store.load_raw_api_responses(user_id, convo_id)
                if raw_responses:
                    usage_line = _usage_line(raw_responses[-1])
                    if usage_line:
                        yield {"event": "delta", "data": f"\n\n{usage_line}"}
            yield event


class UsageLayout(chat.layout.Default):
    """Render-time seam — append the footer every time a message is shown.

    Fires on page refresh, sidebar navigation, and deep links — anywhere
    ``render_messages`` is invoked. Reads the same ``raw_api_responses.jsonl``
    sidecar as ``UsageEngine`` and appends the same footer, so the visible
    transcript stays consistent across the live stream and every later view.

    Mutates the rendered copy only — ``messages.json`` is never touched.
    """

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

    return (
        f'<small style="opacity:0.6">Usage: ↑ {prompt_tokens} + '
        f"↓ {completion_tokens} = {total_tokens} Tokens</small>"
    )


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


WELCOME_MESSAGE = """## Token usage across providers

Each provider \u2014 OpenAI, Anthropic, Gemini \u2014 reports usage in its own shape, but the layout normalizes them all into the same `\u2191 prompt + \u2193 completion = total` line beneath each assistant turn.

To compare providers, send the **same prompt**, then stop the server, swap the active `llm=` line in `examples/usage_display_multi_provider.py`, and rerun. Each run writes to its own `usage_demo_multi/<provider>/` directory \u2014 so all three transcripts stay side-by-side.

<div id="suggestions">
  <button class="suggestion" data-insert-prompt="Explain quantum entanglement in 3 sentences.">
    <span class="suggestion-label">SHORT</span>
    <span class="suggestion-text">Compact answer \u2014 small token bills.</span>
  </button>
  <button class="suggestion" data-insert-prompt="Write a 200-word essay on the future of programming.">
    <span class="suggestion-label">LONG</span>
    <span class="suggestion-text">Bigger answer \u2014 watch the gap widen.</span>
  </button>
  <button class="suggestion" data-insert-prompt="List 10 prime numbers between 100 and 200, then briefly explain why each is prime.">
    <span class="suggestion-label">STRUCTURED</span>
    <span class="suggestion-text">Mixed reasoning + listing.</span>
  </button>
</div>"""


app = chat.Chatnificent(
    # llm=chat.llm.OpenAI(stream_options={"include_usage": True}),
    # llm=chat.llm.Gemini(),
    llm=chat.llm.Anthropic(),
    engine=UsageEngine(),
    layout=UsageLayout(welcome_message=WELCOME_MESSAGE),
    store=chat.store.File(base_dir=BASE_DIR),
)

if __name__ == "__main__":
    app.run()
