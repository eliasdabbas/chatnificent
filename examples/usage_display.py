# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "chatnificent[openai]",
# ]
# ///
"""
Usage Display — The Smallest OpenAI-Only Display Enrichment Example
===================================================================

This example is intentionally minimal.

It teaches one pattern only:

1. keep the stored conversation canonical
2. read raw API responses from the store
3. enrich the visible transcript in ``Layout.render_messages()``

The assistant message itself stays untouched in ``messages.json``. Only the
rendered display gets a usage footer.

Prerequisites
-------------
Set your OpenAI API key before running the app::

    export OPENAI_API_KEY="sk-..."

This example also opts into streamed OpenAI usage data with
``stream_options={"include_usage": True}``, so the usage footer still works
with Chatnificent's default streaming UI.

Running
-------
::

    uv run --script examples/usage_display.py

Send a message and the assistant reply will render with a footer like::

    Usage: ↑ 10 + ↓ 20 = 30 Tokens

What to Explore Next
--------------------
- See ``usage_display_multi_provider.py`` for a more robust multi-provider version
- Add cost estimation beside the usage line
- Read a second sidecar file and layer another display-only enhancement
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
                f"Usage: ↑ {usage['prompt_tokens']} + "
                f"↓ {usage['completion_tokens']} = {usage['total_tokens']} Tokens"
            )
    return None


class UsageLayout(chat.layout.DefaultLayout):
    """Append OpenAI token usage beneath each assistant message."""

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


app = chat.Chatnificent(
    llm=chat.llm.OpenAI(stream_options={"include_usage": True}),
    layout=UsageLayout(),
    store=chat.store.File(base_dir=BASE_DIR),
)

if __name__ == "__main__":
    app.run()
