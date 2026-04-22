# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "chatnificent[anthropic]",
#     "google-genai",
# ]
# ///
"""
Per-User Persistent Memory — Private Storage Scoped to Each User
================================================================

``memory_tool.py`` adds LLM memory using a shared local directory. That
works for single-user local development, but when multiple users are involved
— whether concurrent or across deployments — they all read and overwrite the
same files.

This example upgrades to per-user isolation with exactly two additions over
the minimal version:

1. A ``ContextVar`` — thread-safe, per-request storage for the current user ID.
2. A ``UserAwareAuth`` subclass — captures the resolved user ID from the Auth
   pillar on every request and writes it into the ``ContextVar``.

The ``memory()`` function reads the ``ContextVar`` to know whose files to
touch. Files land at ``memory_chats/<user_id>/memories/<filename>`` via the
Store pillar's sidecar API (``save_file`` / ``load_file`` / ``list_files``).

Why ContextVar and Not a Plain Global?
--------------------------------------
A plain module-level variable would be a race condition: two concurrent
requests from different users would stomp on each other's user ID. A
``ContextVar`` gives each OS thread and asyncio Task its own independent copy
automatically — no locks, no thread-local boilerplate.

Why Subclass UserAwareAuth Instead of Using a Hook?
----------------------------------------------------
The Auth pillar is the canonical place where ``user_id`` is resolved in
Chatnificent. Subclassing it with a single 3-line override is the minimal
correct interception point — no monkey-patching, no engine overrides, no
middleware.

Running
-------
::

    export GOOGLE_API_KEY="..."   # or ANTHROPIC_API_KEY
    uv run examples/memory_tool_multi_user.py

Open two different browsers (or one normal + one incognito window). Each gets
a distinct session cookie that acts as a persistent user ID — it survives tab
closes and server restarts. Teach each user something different — they will
never see each other's memories, and each picks up exactly where they left off.

What to Explore Next
--------------------
- Swap the File store for SQLite — the sidecar API works identically across
  store backends.
- Replace ``UserAwareAuth`` with ``SingleUser`` and drop the ``ContextVar``
  if you only ever need one named user.
- For the minimal single-directory version, see ``memory_tool.py``.
"""

import contextvars

import chatnificent as chat

MEMORY_SLOT = "memories"

# ContextVar, not a plain global: each asyncio Task / thread gets its own copy,
# so concurrent requests from different users don't stomp on each other.
_current_user = contextvars.ContextVar("memory_user", default="anon")

SYSTEM_PROMPT = """
You have a `memory` tool backed by a local directory that persists across conversations. Supported commands:
  - list_files: show available memory files (aliases: list-files, listfiles).
  - read: read a file's contents.
  - create: create a file with an optional initial `content`. Safe to call if the file already exists (no-op).

At the start of every turn, silently call list_files and read anything relevant — do not narrate or mention this process.
When the user asks to add/remove/edit an item in a file (e.g. 'add milk to my todo list'), read the current file, compose
the new full contents, and create the file (it will overwrite). When the user shares a preference, fact, or decision worth
remembering, save it silently without being asked. Never mention memory files, the memory tool, or that you are
saving/reading anything — unless the user explicitly asks. Keep notes concise.
"""


class UserAwareAuth(chat.auth.Anonymous):
    """Stash the resolved user_id so tool functions can reach it."""

    def get_current_user_id(self, **kwargs) -> str:
        user_id = super().get_current_user_id(**kwargs)
        _current_user.set(user_id)
        return user_id


def memory(command: str, filename: str = "", content: str = "") -> str:
    """Persistent per-user memory across conversations.

    Parameters
    ----------
    command : str
        One of: `list_files` (aliases `list-files`, `listfiles`), `read`, `create`.
    filename : str
        Bare filename, e.g. `todo_list.txt`.
    content : str
        For `create`: the initial file contents. Omit to create an empty file.
    """
    user_id = _current_user.get()

    if command in ("list_files", "list-files", "listfiles"):
        names = sorted(app.store.list_files(user_id, MEMORY_SLOT))
        return "\n".join(f"- {n}" for n in names) if names else "(no files yet)"

    if command == "read":
        data = app.store.load_file(user_id, MEMORY_SLOT, filename)
        if data is None:
            return f"Error: {filename} does not exist."
        return data.decode("utf-8")

    if command == "create":
        existing = app.store.load_file(user_id, MEMORY_SLOT, filename)
        if existing is not None and not content:
            return f"{filename} already exists."
        app.store.save_file(user_id, MEMORY_SLOT, filename, content.encode("utf-8"))
        return f"Saved {filename}."

    return f"Error: unknown command '{command}'."


tools = chat.tools.PythonTool()
tools.register_function(memory)

app = chat.Chatnificent(
    # llm=chat.llm.Anthropic(system=SYSTEM_PROMPT),
    llm=chat.llm.Gemini(system_instruction=SYSTEM_PROMPT),
    tools=tools,
    auth=UserAwareAuth(),
    store=chat.store.File("memory_chats"),
)

if __name__ == "__main__":
    app.run()
