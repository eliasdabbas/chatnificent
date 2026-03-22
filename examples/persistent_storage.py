# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "chatnificent",
# ]
# ///
"""
Persistent Storage — Keep Conversations Across Restarts
=======================================================

By default, Chatnificent uses ``InMemory`` storage — conversations live in a
Python dict and vanish when the server stops. This example shows how to persist
them to disk so they survive restarts.

Chatnificent ships with two persistent Store implementations:

- **File** — one JSON file per conversation, stored in a directory you choose
- **SQLite** — all conversations in a single SQLite database file

Both are zero-dependency (stdlib only) and work with any LLM provider.

File Store
----------
::

    store = chat.store.File(base_dir="./my_chats")

Creates a directory structure like::

    my_chats/
    └── <user_id>/
        ├── <convo_id>.json
        └── <convo_id>.json

Each conversation is a standalone JSON file. Easy to inspect, back up, or
version-control. The File store includes path traversal protection, so
malicious user/conversation IDs cannot escape the storage directory.

SQLite Store
------------
::

    store = chat.store.SQLite(db_path="chats.db")

Stores everything in a single ``chats.db`` file. Better for apps with many
conversations where filesystem overhead matters. The database is created
automatically on first use.

How Storage Works
-----------------
The Store pillar has a simple contract:

- ``save_conversation(user_id, conversation)`` — persist a conversation
- ``load_conversation(user_id, convo_id)`` — retrieve one conversation
- ``list_conversations(user_id)`` — list all conversations for a user
- ``delete_conversation(user_id, convo_id)`` — remove a conversation

The Engine calls these automatically — you don't interact with the Store
directly unless you're building a custom pillar.

Running
-------
::

    uv run examples/persistent_storage.py

Chat in the browser, stop the server (Ctrl+C), restart it, and your
conversations reappear. The ``conversations/`` directory (or ``chats.db``
file) is created next to the script.

What to Explore Next
--------------------
- Build a custom Store (e.g., Redis, PostgreSQL) by subclassing
  ``chat.store.Store`` and implementing the four abstract methods
- Combine with ``chat.auth.SingleUser(user_id="me")`` for a personal
  note-taking chat that always loads your history
"""

import chatnificent as chat

app = chat.Chatnificent(
    store=chat.store.File(base_dir="./conversations"),
    # Or use SQLite instead:
    # store=chat.store.SQLite(db_path="chats.db"),
)

if __name__ == "__main__":
    app.run()
