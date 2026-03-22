# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "chatnificent[openai]",
# ]
# ///
"""
Auto-Title — Name Conversations from the First Exchange
========================================================

By default Chatnificent shows the first 30 characters of your first message
as the sidebar title. That works, but "Can you explain the difference bet..."
isn't a great title.

This example generates a short, descriptive title after the first exchange
and updates the sidebar automatically. Instead of truncated questions, you
get titles like "EU vs US Tax Systems" or "Python Async Best Practices."

How It Works
------------
Subclass ``Orchestrator`` and override the ``_before_save`` hook. After the
first exchange (one user message + one assistant reply), make a quick
non-streaming LLM call to generate a short title from the user's question.
Then prepend it with an em dash separator to the first user message::

    "Can you explain..." → "EU vs US Tax — Can you explain..."

The sidebar truncates the first user message to 30 characters, so the
title is what appears. On subsequent turns the title is already set, so
the hook becomes a no-op.

One hook override, zero framework changes.

Running
-------
::

    export OPENAI_API_KEY="sk-..."
    uv run examples/auto_title.py

Send a message and watch the sidebar — it updates with a generated title
after the first response.

What to Explore Next
--------------------
- Use a cheaper model for title generation (e.g. ``gpt-4o-mini``)
- Combine with ``single_user.py`` so titled conversations persist across
  sessions
- Combine with ``token_counter.py`` to see the cost of the title call
"""

import chatnificent as chat

TITLE_PROMPT = (
    "Generate a concise 3-5 word title for a conversation that starts with "
    "the following user message. Return ONLY the title, no quotes, no "
    "punctuation, no explanation."
)


class AutoTitleEngine(chat.engine.Orchestrator):
    def _before_save(self, conversation):
        user_msgs = [m for m in conversation.messages if m.get("role") == "user"]
        assistant_msgs = [
            m for m in conversation.messages if m.get("role") == "assistant"
        ]

        if len(user_msgs) != 1 or len(assistant_msgs) != 1:
            return

        first_user_msg = user_msgs[0]
        content = first_user_msg.get("content", "")

        # Already titled from a previous save
        if " — " in content[:40]:
            return

        title_response = self.app.llm.generate_response(
            [
                {"role": "system", "content": TITLE_PROMPT},
                {"role": "user", "content": content},
            ],
            stream=False,
        )
        title = self.app.llm.extract_content(title_response).strip().strip("\"'")

        first_user_msg["content"] = f"{title} — {content}"


app = chat.Chatnificent(engine=AutoTitleEngine())

if __name__ == "__main__":
    app.run()
