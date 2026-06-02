# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "chatnificent[openai]",
# ]
# ///
"""
Append Anything After Any Turn — The OpenAI Usage Display Pattern
================================================================

This example is intentionally minimal. The footer it displays — OpenAI token
usage — is just a stand-in for the real lesson:

    **How to render anything you want after an assistant turn, without
    modifying the conversation history.**

``messages.json`` stays canonical — the model reply is recorded exactly as it
was generated. The usage footer lives in ``raw_api_responses.jsonl`` (the
framework writes it for you) and is rendered into the visible transcript
from there. Two small seams, one on each side of persistence, cover both
moments the user might be looking at the conversation:

1. **Streaming time — ``UsageEngine`` (subclass of ``Orchestrator``).**
   Wraps ``handle_message_stream`` and injects one extra ``delta`` event
   right before the stream's ``done`` event. The engine has already
   persisted the conversation by then, so what we inject reaches the
   browser through the existing SSE channel and never lands in
   ``messages.json``.

2. **Render time — ``UsageLayout`` (subclass of ``Default``).**
   Overrides ``render_messages`` to read the same
   ``raw_api_responses.jsonl`` sidecar and append the same footer to each
   assistant message. This fires every time a conversation is fetched
   (page refresh, sidebar click, deep link), so the footer is visible
   forever — not just during the live stream.

The two halves share the same formatter (``_openai_usage_line``) and the
same data source. Together they give you a complete pattern for
*display-only* enrichment: live during streaming, persistent across
reloads, and zero pollution of canonical history.

Apply the same shape to display per-turn cost estimates, latency, moderation
flags, retrieval citations, tool-call traces, or anything else derivable from
the raw API response on disk. Swap ``_openai_usage_line()`` for any
formatter that returns a Markdown/HTML string (or ``None`` to skip). See
``usage_display_multi_provider.py`` for the same pattern generalised across
OpenAI, Anthropic, and Gemini payload shapes.

Prerequisites
-------------
Set your OpenAI API key before running the app::

    export OPENAI_API_KEY="sk-..."

The app opts into streamed usage data with
``stream_options={"include_usage": True}`` so the final OpenAI chunk carries
the token counts we read from ``raw_api_responses.jsonl``.

Running
-------
::

    uv run --script examples/usage_display.py
"""

import chatnificent as chat

BASE_DIR = "usage_convos"


def _openai_usage_line(raw_response):
    """Read the saved OpenAI usage payload and return the display footer."""
    chunks = raw_response if isinstance(raw_response, list) else [raw_response]
    for chunk in reversed(chunks):
        if not isinstance(chunk, dict):
            continue
        usage = chunk.get("usage")
        if usage:
            return (
                f'<small style="opacity:0.6">Usage: ↑ {usage["prompt_tokens"]} + '
                f"↓ {usage['completion_tokens']} = {usage['total_tokens']} Tokens</small>"
            )
    return None


class UsageEngine(chat.engine.Orchestrator):
    """Streaming-time seam — inject the footer as a final ``delta`` event.

    ``Orchestrator.handle_message_stream`` calls ``_save_conversation`` BEFORE
    yielding ``done``. Anything we yield between the parent generator's last
    item and the ``done`` event rides the existing SSE channel to the client
    without ever touching ``messages.json``.

    To display something else, swap ``_openai_usage_line`` for your own
    formatter; everything else stays the same.
    """

    def handle_message_stream(self, user_input, user_id, convo_id_from_url):
        for event in super().handle_message_stream(
            user_input, user_id, convo_id_from_url
        ):
            if event.get("event") == "done":
                convo_id = event["data"]["conversation_id"]
                raw_responses = self.app.store.load_raw_api_responses(user_id, convo_id)
                if raw_responses:
                    usage_line = _openai_usage_line(raw_responses[-1])
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
            usage_line = _openai_usage_line(raw_response)
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


WELCOME_MESSAGE = """## Token usage visible in transcript

Every assistant turn ends with a usage line showing input/output token counts.

<div id="suggestions">
  <button class="suggestion" data-insert-prompt="What's 2+2?">
    <span class="suggestion-label">SHORT</span>
    <span class="suggestion-text">Tiny prompt.</span>
  </button>
  <button class="suggestion" data-insert-prompt="Summarize the plot of Hamlet in one paragraph.">
    <span class="suggestion-label">MEDIUM</span>
    <span class="suggestion-text">A paragraph response.</span>
  </button>
  <button class="suggestion" data-insert-prompt="Write a 200-word essay on the future of programming.">
    <span class="suggestion-label">LONG</span>
    <span class="suggestion-text">A bigger response.</span>
  </button>
</div>"""


app = chat.Chatnificent(
    llm=chat.llm.OpenAI(stream_options={"include_usage": True}),
    engine=UsageEngine(),
    layout=UsageLayout(welcome_message=WELCOME_MESSAGE),
    store=chat.store.File(base_dir=BASE_DIR),
)

if __name__ == "__main__":
    app.run()
