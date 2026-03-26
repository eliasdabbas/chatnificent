# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "chatnificent[openai,anthropic,gemini]",
# ]
# ///
"""
Conversation Title — Generate a Sidebar Title Without Polluting Messages
========================================================================

This example shows the cleanest version of conversation titles in Chatnificent:

1. keep the conversation history canonical
2. generate a title after the save succeeds
3. store that title in ``conversation_title.txt``
4. let ``Layout.render_conversations(...)`` decide what the sidebar shows

That keeps titles editable, replaceable, and completely separate from the
actual message history. The next LLM turn still sees only the real transcript.

Why this pattern is useful
--------------------------
Conversation titles are display metadata, not conversation content. If you
rewrite the first user message to make a better sidebar label, that label
starts leaking into the prompt on later turns.

With a sidecar file the responsibilities stay clean:

- ``messages.json`` remains the canonical transcript
- ``conversation_title.txt`` is UI metadata
- ``Engine._after_save(...)`` creates the title
- ``Layout.render_conversations(...)`` decides whether to use it

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

    uv run --script examples/conversation_title.py

Send the first message in a new conversation, then look at the sidebar. The
conversation title will come from ``conversation_title.txt`` instead of the
first 30 characters of the first user message.

What to Explore Next
--------------------
- Let users rename conversations by editing ``conversation_title.txt``
- Add tags or pinned state as additional conversation-scoped files
- Generate summaries in ``_after_save(...)`` using the same pattern
"""

import chatnificent as chat

TITLE_PROMPT = (
    "Create a concise conversation title based on this message. "
    "Return only the final title text, ready for display, with normal "
    "capitalization and no more than 40 characters. Do not use quotes, "
    "asterisks, bullets, markdown, labels, or surrounding punctuation."
)


class ConversationTitleEngine(chat.engine.Orchestrator):
    """Generate a title file after the canonical conversation is saved."""

    def _after_save(self, conversation, user_id):
        title_file = self.app.store.load_file(
            user_id, conversation.id, "conversation_title.txt"
        )
        if title_file:
            return

        first_user_message = next(
            (
                message
                for message in conversation.messages
                if message.get("role") == "user"
                and isinstance(message.get("content"), str)
                and message.get("content", "").strip()
            ),
            None,
        )
        if not first_user_message:
            return

        response = self.app.llm.generate_response(
            [
                {"role": "system", "content": TITLE_PROMPT},
                {"role": "user", "content": first_user_message["content"]},
            ],
            stream=False,
        )
        title = (
            self.app.llm.extract_content(response) or "Untitled Conversation"
        ).strip()
        title = title.splitlines()[0].strip("\"'`*_#-: ")
        if len(title) > 40:
            title = title[:37].rstrip() + "..."
        self.app.store.save_file(
            user_id,
            conversation.id,
            "conversation_title.txt",
            title.encode("utf-8"),
        )


class ConversationTitleLayout(chat.layout.DefaultLayout):
    """Render the sidebar title from ``conversation_title.txt`` when present."""

    def render_conversations(self, conversations, **kwargs):
        rendered = super().render_conversations(conversations, **kwargs)
        user_id = kwargs["user_id"]

        for conversation_item in rendered:
            title_bytes = self.app.store.load_file(
                user_id,
                conversation_item["id"],
                "conversation_title.txt",
            )
            if title_bytes:
                conversation_item["title"] = title_bytes.decode("utf-8").strip()

        return rendered


app = chat.Chatnificent(
    llm=chat.llm.OpenAI(),
    # llm=chat.llm.Anthropic(),
    # llm=chat.llm.Gemini(),
    engine=ConversationTitleEngine(),
    layout=ConversationTitleLayout(),
    store=chat.store.File(base_dir="convo_titles"),
)

if __name__ == "__main__":
    app.run()
