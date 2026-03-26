# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "chatnificent[openai,anthropic,gemini]",
# ]
# ///
"""
Conversation Summary — Keep a Running Markdown Summary Beside the Chat
======================================================================

This example uses the same pattern as ``conversation_title.py``, but with a
slightly richer sidecar:

1. keep the conversation history canonical
2. generate a new summary after each assistant turn is saved
3. append that summary to ``summaries.md``
4. let ``Layout.render_messages(...)`` prepend the latest summary to the UI

The summary is display-only metadata. It lives beside the conversation instead
of inside the message history, so the next LLM turn still sees the real chat
transcript and nothing else.

Why this pattern is useful
--------------------------
Conversation summaries are often helpful for long chats, but they should not
pollute the canonical history. By storing them in ``summaries.md`` you get a
few nice properties:

- every turn can create a new summary without rewriting old messages
- the full summary history stays available in one append-only file
- the layout can choose to show only the latest summary
- users can inspect or edit the summaries independently from the chat

This example renders the latest summary inside a collapsible block so it is
available when useful without taking over the whole transcript.

Prerequisites
-------------
Set the API key for the provider you want to use::

    export OPENAI_API_KEY="sk-..."
    # or
    export ANTHROPIC_API_KEY="sk-ant-..."
    # or
    export GOOGLE_API_KEY="AI..."

The active default below is OpenAI. To try another provider, swap the
``llm=...`` line in the app definition.

Running
-------
::

    uv run --script examples/conversation_summary.py

Send a message, then continue the conversation. The latest summary will appear
at the top of the visible transcript, while ``summaries.md`` keeps every
generated summary in append-only order.

What to Explore Next
--------------------
- Render multiple historical summaries in a timeline view
- Add tags or decision lists beside each summary block
- Combine this with ``display_redaction.py`` or ``usage_display.py``
"""

from typing import Optional

import chatnificent as chat

SUMMARY_PROMPT = """
Write a concise Markdown summary of the conversation so far, ready for display.
This must be a true synthesis, not a turn-by-turn replay.
Focus on the user's overall goal, the most important facts, answers, decisions,
and any open questions or next steps.
Combine related points and omit minor details.
Address the reader directly as "you" when natural.
Do not write things like "the user asked" or "the assistant replied".
Keep it under 300 words.
You may use short headings, bullets, or a table only if it makes the summary
clearer or more compact.
If there are genuinely useful follow-up questions, open points, or next steps,
include them briefly.
If the conversation is already complete, do not force that section.
Return only the summary, with no preamble and no code fences.
"""
SUMMARY_SEPARATOR = "\n\n<!-- chatnificent-summary -->\n\n"


def _conversation_transcript(conversation, llm) -> str:
    """Build a plain transcript for the summary prompt from visible messages."""
    lines = []
    for message in conversation.messages:
        if message.get("role") == "system":
            continue
        if llm.is_tool_message(message):
            continue

        content = message.get("content")
        if content is None:
            continue
        if not isinstance(content, str):
            continue
        if not content.strip():
            continue

        speaker = "User" if message.get("role") == "user" else "Assistant"
        lines.append(f"{speaker}: {content.strip()}")

    return "\n\n".join(lines)


def _summary_block(summary_markdown: str) -> str:
    """Wrap a summary in a collapsible block for display."""
    summary_body = summary_markdown.strip()
    return (
        "<details>\n"
        "<summary>Conversation Summary</summary>\n\n"
        f"{summary_body}\n\n"
        "</details>"
    )


def _latest_summary(markdown_text: str) -> Optional[str]:
    """Return the most recent summary block from the append-only summaries file."""
    blocks = [
        block.strip()
        for block in markdown_text.split(SUMMARY_SEPARATOR)
        if block.strip()
    ]
    return blocks[-1] if blocks else None


class ConversationSummaryEngine(chat.engine.Orchestrator):
    """Append a fresh summary block after each assistant turn is saved."""

    def _after_save(self, conversation, user_id):
        if not conversation.messages:
            return

        latest_message = conversation.messages[-1]
        if latest_message.get("role") == "user":
            return
        if latest_message.get("role") == "system":
            return
        if self.app.llm.is_tool_message(latest_message):
            return

        latest_content = latest_message.get("content")
        if not isinstance(latest_content, str) or not latest_content.strip():
            return

        transcript = _conversation_transcript(conversation, self.app.llm)
        if not transcript:
            return

        response = self.app.llm.generate_response(
            [
                {"role": "system", "content": SUMMARY_PROMPT},
                {"role": "user", "content": transcript},
            ],
            stream=False,
        )
        summary_text = (self.app.llm.extract_content(response) or "").strip()
        if not summary_text:
            return

        existing = self.app.store.load_file(user_id, conversation.id, "summaries.md")
        payload = _summary_block(summary_text)
        if existing:
            payload = SUMMARY_SEPARATOR + payload

        self.app.store.save_file(
            user_id,
            conversation.id,
            "summaries.md",
            payload.encode("utf-8"),
            append=True,
        )


class ConversationSummaryLayout(chat.layout.DefaultLayout):
    """Prepend the latest summary block to the rendered transcript."""

    def render_messages(self, messages, **kwargs):
        rendered = super().render_messages(messages, **kwargs)
        user_id = kwargs["user_id"]
        conversation = kwargs["conversation"]
        summary_bytes = self.app.store.load_file(
            user_id, conversation.id, "summaries.md"
        )
        if not summary_bytes:
            return rendered

        latest_summary = _latest_summary(summary_bytes.decode("utf-8"))
        if not latest_summary:
            return rendered

        return [{"role": "assistant", "content": latest_summary}] + rendered


app = chat.Chatnificent(
    # llm=chat.llm.OpenAI(),
    llm=chat.llm.Anthropic(),
    # llm=chat.llm.Gemini(),
    engine=ConversationSummaryEngine(),
    layout=ConversationSummaryLayout(),
    store=chat.store.File(base_dir="convo_summaries"),
)

if __name__ == "__main__":
    app.run()
