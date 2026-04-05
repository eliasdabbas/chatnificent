# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "chatnificent[starlette]",
#     "openai",
# ]
# ///
"""
Starlette Multi-Mount — Multiple Chat Apps on One Website
==========================================================

Run several independent Chatnificent chat apps under one Starlette server, each
mounted at its own URL path. Each app has its own LLM configuration, conversation
store, and system prompt — completely isolated from the others.

This is useful when you want a single website to host different AI assistants,
each with a distinct personality, knowledge base, or LLM provider:

- ``/code/``   — a coding assistant
- ``/writer/`` — a creative writing helper
- ``/``        — a landing page with links to both assistants

Why Multi-Mount?
----------------
Instead of running separate servers on different ports, Starlette's ``Mount``
lets you compose multiple ASGI apps into one. Each Chatnificent instance is a
full ASGI callable (via ``Chatnificent.__call__``), so mounting is a one-liner::

    Mount("/code", app=code_app)

Each mounted app gets its own:

- **LLM pillar** — different models, providers, or system prompts
- **Store pillar** — separate conversation histories
- **Auth pillar** — independent session cookies (scoped by path)

How It Works
------------
1. Create multiple ``Chatnificent`` instances, each with
   ``server=chat.server.Starlette()`` and its own pillar configuration
2. Build a parent ``starlette.applications.Starlette`` app with ``Mount``
   routes pointing to each Chatnificent instance
3. Add a landing page route at ``/`` with links to each assistant
4. Run the parent app with ``uvicorn``

The parent app owns the top-level routing. Each Chatnificent sub-app handles
its own ``/api/chat``, ``/api/conversations``, and ``/`` endpoints relative to
its mount point. Starlette's ``Mount`` strips the prefix automatically before
passing requests to the sub-app, and Chatnificent detects the ASGI
``root_path`` to prefix all API paths, browser URLs, and session cookies
accordingly — no extra configuration needed.

.. note::

   Each mount point (``/code/``, ``/writer/``) serves a fully standalone chat
   UI. Navigate directly to ``/code/`` and everything works: the URL bar,
   conversation history, and sidebar all use the correct prefixed paths.

   The landing page provides links to each assistant. You can also navigate
   directly to any mount point.

Running
-------
::

    uv run --script examples/starlette_multi_mount.py

Then visit:

- http://127.0.0.1:7777/ — landing page with links to both assistants
- http://127.0.0.1:7777/code/ — the coding assistant (standalone, full page)
- http://127.0.0.1:7777/writer/ — the creative writing assistant (standalone)

Each assistant has its own conversation history and personality.

Alternatively, run via the uvicorn CLI::

    uvicorn starlette_multi_mount:app --port 7777

What to Explore Next
--------------------
- Start with a single Starlette app: see ``starlette_quickstart.py``
- Add middleware and custom routes: see ``starlette_server_options.py``
- Configure uvicorn options (workers, reload, SSL): see ``starlette_uvicorn_options.py``
- System prompt patterns for real LLM providers: see ``system_prompt.py``
- Give each assistant tools: see ``multi_tool_agent.py``
"""

import chatnificent as chat
from starlette.applications import Starlette
from starlette.responses import HTMLResponse
from starlette.routing import Mount, Route

# ---------------------------------------------------------------------------
# System prompts — give each assistant a distinct personality
# ---------------------------------------------------------------------------

CODE_PROMPT = """
Your name is Chatnificent Coder. You are a senior software engineer.
You write clean, idiomatic code with clear explanations. When asked a
question, provide working code examples. Use Markdown code blocks with
language tags.
"""

WRITER_PROMPT = """
Your name is Chatnificent Wordsmith. You are a creative writing
assistant. You help with brainstorming, drafting, and editing prose.
You have a warm, encouraging tone and offer constructive feedback.
You love metaphors and vivid imagery.
"""


# ---------------------------------------------------------------------------
# Custom LLM — prepends a system prompt before each call
# ---------------------------------------------------------------------------
# This is the standard pattern for system prompts with OpenAI-compatible
# providers: subclass your LLM, override generate_response(), prepend the
# system message. See system_prompt.py for details.
#
# For Anthropic, use: chat.llm.Anthropic(system="...")
# For Gemini, use:    chat.llm.Gemini(system_instruction="...")


class SystemPromptLLM(chat.llm.OpenAI):
    """OpenAI LLM that prepends a system prompt to every request."""

    def __init__(self, system_prompt, **kwargs):
        super().__init__(**kwargs)
        self._system_prompt = system_prompt

    def generate_response(self, messages, **kwargs):
        if not messages or messages[0].get("role") != "system":
            messages = [{"role": "system", "content": self._system_prompt}, *messages]
        return super().generate_response(messages, **kwargs)


# ---------------------------------------------------------------------------
# Create two independent Chatnificent apps
# ---------------------------------------------------------------------------

code_app = chat.Chatnificent(
    server=chat.server.Starlette(debug=True),
    llm=SystemPromptLLM(system_prompt=CODE_PROMPT),
    store=chat.store.File("multi_chat/code"),
)

writer_app = chat.Chatnificent(
    server=chat.server.Starlette(debug=True),
    llm=SystemPromptLLM(system_prompt=WRITER_PROMPT),
    store=chat.store.File("multi_chat/writer"),
)


# ---------------------------------------------------------------------------
# Landing page — intro and links to the two assistants
# ---------------------------------------------------------------------------


async def landing_page(request):
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Chatnificent Multi-App</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                         'Helvetica Neue', Arial, sans-serif;
            background: #f7f8fa;
            color: #1a1d21;
            display: flex;
            justify-content: center;
            padding: 4rem 1.5rem;
        }
        .container { max-width: 640px; width: 100%; }
        h1 { font-size: 1.75rem; margin-bottom: 1rem; }
        p { color: #6b7280; line-height: 1.6; margin-bottom: 1rem; }
        .cards {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            margin-top: 2rem;
        }
        .card {
            display: block;
            padding: 1.5rem;
            background: #ffffff;
            border: 1px solid #e2e5e9;
            border-radius: 16px;
            text-decoration: none;
            color: inherit;
            transition: box-shadow 0.2s, border-color 0.2s;
        }
        .card:hover {
            border-color: #2563eb;
            box-shadow: 0 2px 12px rgba(37, 99, 235, 0.12);
        }
        .card-icon { font-size: 1.75rem; margin-bottom: 0.75rem; }
        .card-title { font-weight: 600; font-size: 1.1rem; margin-bottom: 0.25rem; }
        .card-desc { color: #6b7280; font-size: 0.9rem; line-height: 1.4; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Chatnificent Multi-App</h1>
        <p>
            Welcome! This demo runs two independent AI chat assistants on a single
            website, powered by Starlette's <code>Mount</code>. Each app has its own
            LLM configuration, conversation history, and system prompt &mdash;
            completely isolated from the other.
        </p>
        <p>
            Pick an assistant below to start chatting. Your conversations are
            saved separately for each one.
        </p>
        <div class="cards">
            <a class="card" href="/code/">
                <div class="card-icon">💻</div>
                <div class="card-title">Code Assistant</div>
                <div class="card-desc">
                    A senior software engineer that writes clean, idiomatic code
                    with clear explanations.
                </div>
            </a>
            <a class="card" href="/writer/">
                <div class="card-icon">✍️</div>
                <div class="card-title">Writing Assistant</div>
                <div class="card-desc">
                    A creative writing helper that brainstorms, drafts, and
                    polishes prose with a warm tone.
                </div>
            </a>
        </div>
    </div>
</body>
</html>"""
    return HTMLResponse(html)


# ---------------------------------------------------------------------------
# Parent Starlette app — mount everything together
# ---------------------------------------------------------------------------

app = Starlette(
    routes=[
        Route("/", landing_page),
        Mount("/code", app=code_app),
        Mount("/writer", app=writer_app),
    ]
)


if __name__ == "__main__":
    import uvicorn

    print("Chatnificent Multi-App running on http://127.0.0.1:7777")
    print("  /        — Landing page (both assistants side-by-side)")
    print("  /code/   — Coding assistant (standalone)")
    print("  /writer/ — Writing assistant (standalone)")
    uvicorn.run(app, host="127.0.0.1", port=7777)
