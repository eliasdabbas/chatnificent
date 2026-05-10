# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "chatnificent[openai]",
# ]
# ///
"""
Single User — Persistent Chat with Identity
=============================================

By default Chatnificent uses ``Anonymous`` auth, where each browser session
gets a random user ID. This means conversations aren't tied to a named
identity and don't persist across sessions.

For personal apps — a research assistant, a journaling bot, a coding helper
— you want a fixed identity so your conversations survive across restarts.
Combine ``SingleUser`` auth with a persistent store and you get exactly that.

How It Works
------------
Two pillars work together:

1. **Auth**: ``SingleUser(user_id="elias")`` — every request is attributed
   to the same user. No login needed, just a fixed identifier.

2. **Store**: ``SQLite(db_path="my_chats.db")`` — conversations are saved
   to a SQLite database. Restart the app and your chat history is still
   there in the sidebar.

The combination means: one user, persistent history, zero auth complexity.

When to Use Anonymous vs SingleUser
-------------------------------------
- **Anonymous**: public-facing apps (docs chatbots, demos) where each
  visitor gets isolated, ephemeral conversations
- **SingleUser**: personal tools where you're the only user and want
  persistent history

Running
-------
::

    export OPENAI_API_KEY="sk-..."
    uv run examples/single_user.py

Chat, close the browser, restart the app — your conversations are still
there. The SQLite database (``my_chats.db``) is created in the current
directory.

What to Explore Next
--------------------
- Change the ``user_id`` to organize chats by project:
  ``SingleUser(user_id="research")`` vs ``SingleUser(user_id="coding")``
- Swap SQLite for ``File`` store to get human-readable JSON files on disk
- Combine with ``system_prompt.py`` for a personalized AI assistant
- Build a multi-user app by implementing a custom ``Auth`` subclass that
  reads from cookies, headers, or an OAuth token
"""

import chatnificent as chat

welcome_message = """## Your personal AI chat

`SingleUser` auth + SQLite means there's exactly one user — you. The URL stays the same no matter where you connect from. Open this in a second browser, your phone, or an incognito window: same conversations, same history.

<div id="suggestions">
  <button class="suggestion" data-insert-prompt="Help me draft a 3-bullet daily standup update.">
    <span class="suggestion-label">WORK</span>
    <span class="suggestion-text">Draft today's standup.</span>
  </button>
  <button class="suggestion" data-insert-prompt="Plan my week given these tasks: design review Monday, demo Wednesday, retro Friday.">
    <span class="suggestion-label">PLAN</span>
    <span class="suggestion-text">Plan the week.</span>
  </button>
  <button class="suggestion" data-insert-prompt="Open this URL in another browser — you'll see the same conversation list because there's only one user.">
    <span class="suggestion-label">VERIFY</span>
    <span class="suggestion-text">Same identity everywhere.</span>
  </button>
</div>"""

app = chat.Chatnificent(
    auth=chat.auth.SingleUser(user_id="elias"),
    store=chat.store.SQLite(db_path="my_chats.db"),
    llm=chat.llm.OpenAI(),
    layout=chat.layout.Default(welcome_message=welcome_message),
)

if __name__ == "__main__":
    app.run()
