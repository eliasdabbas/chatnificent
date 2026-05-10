# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "chatnificent[openai,anthropic,gemini]",
# ]
# ///
"""
LLM Providers — Switch Between OpenAI, Anthropic, and Gemini
=============================================================

Chatnificent supports multiple LLM providers through a single, consistent
interface. This example shows how to swap providers by changing one line —
the rest of the app stays identical.

Each provider reads its API key from the environment automatically:

- **OpenAI**: ``OPENAI_API_KEY``
- **Anthropic**: ``ANTHROPIC_API_KEY``
- **Gemini**: ``GOOGLE_API_KEY`` (or use Vertex AI with ``vertexai=True``)

Prerequisites
-------------
Set the API key for the provider you want to use::

    export OPENAI_API_KEY="sk-..."
    # or
    export ANTHROPIC_API_KEY="sk-ant-..."
    # or
    export GOOGLE_API_KEY="AI..."

How It Works
------------
The LLM pillar is one of Chatnificent's nine swappable pillars. Every
provider implements the same abstract interface (``generate_response()``,
``extract_content()``, etc.), so the Engine, Store, Layout, and all other
pillars work identically regardless of which LLM you choose.

All providers stream by default (``stream=True`` in ``default_params``),
so responses appear token-by-token in the UI immediately.

Switching Providers
-------------------
Uncomment the provider you want below. Each one is a single-line change::

    llm = chat.llm.OpenAI()  # GPT models
    llm = chat.llm.Anthropic()  # Claude models
    llm = chat.llm.Gemini()  # Gemini models

You can also customize the model::

    llm = chat.llm.OpenAI(model="gpt-4o")
    llm = chat.llm.Anthropic(model="claude-sonnet-4-20250514")
    llm = chat.llm.Gemini(model="gemini-2.5-flash")

Or pass additional generation parameters::

    llm = chat.llm.OpenAI(model="gpt-4o", temperature=0.3, max_tokens=1024)

Auto-Detection
--------------
If you don't pass an ``llm=`` argument at all, Chatnificent auto-detects:
if the ``openai`` SDK is installed and ``OPENAI_API_KEY`` is set, it uses
OpenAI. Otherwise it falls back to ``Echo`` (a zero-dep mock). This means
``pip install openai`` + setting the env var is enough — no code change
needed.

Other Providers
---------------
Beyond the three shown here, Chatnificent also supports:

- ``chat.llm.Ollama(model="llama3.2")`` — local models (see ``ollama_local.py``)
- ``chat.llm.OpenRouter()`` — many models through one API (see ``openrouter_models.py``)
- ``chat.llm.DeepSeek()`` — DeepSeek models

Running
-------
::

    export OPENAI_API_KEY="sk-..."
    uv run --script examples/llm_providers.py

Then open http://127.0.0.1:7777 in your browser.

What to Explore Next
--------------------
- Persist conversations across restarts: see ``persistent_storage.py``
- Add function calling / tools: see ``tool_calling.py``
- Customize the system prompt: see ``system_prompt.py``
"""

import chatnificent as chat

# ── Pick your provider (uncomment one) ─────────────────────────────────────

llm = chat.llm.OpenAI()
# llm = chat.llm.Anthropic()
# llm = chat.llm.Gemini()

welcome_message = """## Three providers, one chat

The LLM pillar is pluggable — swap providers by changing one line in the source. Each one supports streaming, tool calls, and the same message format. Try a prompt below to see the active provider in action.

<div id="suggestions">
  <button class="suggestion" data-insert-prompt="Say hi and tell me which model and provider you are.">
    <span class="suggestion-label">HELLO</span>
    <span class="suggestion-text">Say hi and tell me which model and provider you are.</span>
  </button>
  <button class="suggestion" data-insert-prompt="Compare OpenAI, Anthropic, and Gemini in 3 sentences each.">
    <span class="suggestion-label">COMPARE</span>
    <span class="suggestion-text">Compare OpenAI, Anthropic, and Gemini.</span>
  </button>
  <button class="suggestion" data-insert-prompt="Write a Python function that returns the nth Fibonacci number.">
    <span class="suggestion-label">CODE</span>
    <span class="suggestion-text">Write a Fibonacci function in Python.</span>
  </button>
</div>"""

# ── App ───────────────────────────────────────────────────────────────────

app = chat.Chatnificent(
    llm=llm,
    layout=chat.layout.Default(welcome_message=welcome_message),
)

if __name__ == "__main__":
    app.run()
