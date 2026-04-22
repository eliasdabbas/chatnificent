# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "chatnificent[anthropic]",
#     "google-genai",
# ]
# ///
"""
Persistent LLM Memory — Remember Facts and Lists Across Conversations
======================================================================

LLMs are stateless by design — every conversation starts from scratch. But
users expect a chat assistant to remember that they prefer metric units, that
their grocery list has oat milk and eggs, or that their project deadline is
Friday.

This example adds persistent, cross-conversation memory to any LLM in under
30 lines by registering a plain Python function as a tool. The LLM calls it
to save and retrieve notes; the notes live in a local ``./memory/`` directory
and survive restarts.

How It Works
------------
1. A ``memory(command, filename, content)`` function implements three commands:

   - ``list_files`` — list what the LLM has stored.
   - ``read`` — read a specific file.
   - ``create`` — write (or overwrite) a file.

2. Register it with ``PythonTool.register_function()`` — Chatnificent
   auto-generates the JSON schema from the function signature and docstring.
3. A system prompt instructs the LLM to check memory silently on every turn
   and save anything worth keeping, without narrating the process.

The LLM decides *when* to save and *when* to read. The three commands map to
simple ``pathlib`` operations — no database, no vendor SDK, no lock-in.

Switching LLM Providers
-----------------------
This example defaults to Gemini but the ``memory`` tool is provider-agnostic.
Uncomment the ``Anthropic`` or ``OpenAI`` line at the bottom to switch —
the tool works identically across all providers.

Running
-------
::

    export GOOGLE_API_KEY="..."   # or ANTHROPIC_API_KEY / OPENAI_API_KEY
    uv run examples/memory_tool.py

Try telling the assistant your name and a preference. Start a new conversation
and ask "what do you know about me?" — it will already know.

Limitations
-----------
All browser sessions share the same ``./memory/`` directory. For a single
developer on their own machine this is fine. For multi-user deployments or
even multiple browser tabs, see ``memory_tool_multi_user.py``.

What to Explore Next
--------------------
- Add a ``delete`` command so the LLM can clean up stale notes.
- Combine with ``auto_title.py`` or ``conversation_summary.py`` for an
  enriched sidebar alongside persistent memory.
- For per-user isolation, see ``memory_tool_multi_user.py``.
"""

from pathlib import Path

import chatnificent as chat

MEMORY_ROOT = Path("./memory").resolve()
MEMORY_ROOT.mkdir(exist_ok=True)

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


def memory(command: str, filename: str = "", content: str = "") -> str:
    """Persistent memory across conversations.

    Parameters
    ----------
    command : str
        One of: `list_files` (aliases `list-files`, `listfiles`), `read`, `create`.
    filename : str
        Bare filename, e.g. `todo_list.txt`.
    content : str
        For `create`: the initial file contents. Omit to create an empty file.
    """
    if command in ("list_files", "list-files", "listfiles"):
        names = sorted(p.name for p in MEMORY_ROOT.iterdir() if p.is_file())
        return "\n".join(f"- {n}" for n in names) if names else "(no files yet)"

    path = MEMORY_ROOT / filename

    if command == "read":
        if not path.exists():
            return f"Error: {filename} does not exist."
        return path.read_text()

    if command == "create":
        if path.exists() and not content:
            return f"{filename} already exists."
        path.write_text(content)
        return f"Saved {filename}."

    return f"Error: unknown command '{command}'."


tools = chat.tools.PythonTool()
tools.register_function(memory)

app = chat.Chatnificent(
    # llm=chat.llm.Anthropic(system=SYSTEM_PROMPT),
    llm=chat.llm.Gemini(system_instruction=SYSTEM_PROMPT),
    tools=tools,
    store=chat.store.File("memory_chats"),
)

if __name__ == "__main__":
    app.run()
