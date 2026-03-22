# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "chatnificent",
# ]
# ///
"""
Quickstart — Zero-Dependency Chat App
======================================

The simplest possible Chatnificent app. No API keys, no installs beyond
Chatnificent itself. This uses the **Echo** LLM, which mirrors back whatever
you type — perfect for exploring the UI, testing layouts, or verifying your
setup before connecting a real provider.

How It Works
------------
When no LLM provider SDK is installed (or no API key is set), Chatnificent
automatically falls back to ``Echo``. Echo's ``generate_response()`` simply
returns the user's message prefixed with "Echo: ", so you get instant feedback
without any external calls.

The default stack is:
- **Server**: ``DevServer`` — stdlib HTTP server on http://127.0.0.1:7777
- **Layout**: ``DefaultLayout`` — vanilla HTML/JS chat UI (no Dash needed)
- **Store**: ``InMemory`` — conversations live in a dict, lost on restart
- **Auth**: ``Anonymous`` — each browser session gets a random user ID
- **URL**: ``PathBased`` — URLs like ``/<user_id>/<convo_id>``

Running
-------
::

    uv run examples/quickstart.py

Then open http://127.0.0.1:7777 in your browser and start chatting.

What to Explore Next
--------------------
- Swap ``Echo`` for a real provider: see ``llm_providers.py``
- Persist conversations across restarts: see ``persistent_storage.py``
- Add function calling / tools: see ``tool_calling.py``
"""

import chatnificent as chat

app = chat.Chatnificent()

if __name__ == "__main__":
    app.run()
