# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "chatnificent[openai]",
# ]
# ///
"""
OpenRouter — Access Many Models Through One API
================================================

`OpenRouter <https://openrouter.ai/>`_ is an API gateway that gives you
access to models from OpenAI, Anthropic, Google, Meta, Mistral, and many more
— all through a single API key and a unified OpenAI-compatible interface.

This is useful when you want to:
- Try different models without managing multiple API keys
- Access models that don't have a direct Python SDK
- Compare model quality across providers
- Use the cheapest model for a given task

Prerequisites
-------------
1. Get an OpenRouter API key: https://openrouter.ai/keys
2. Set it as an environment variable::

       export OPENROUTER_API_KEY="sk-or-v1-..."

How It Works
------------
Under the hood, ``chat.llm.OpenRouter()`` uses the OpenAI SDK pointed at
OpenRouter's API endpoint (``https://openrouter.ai/api/v1``). That's why the
dependency is ``chatnificent[openai]`` — no separate OpenRouter SDK needed.

Model Selection
---------------
Models use the ``provider/model`` naming convention::

    llm = chat.llm.OpenRouter(model="openai/gpt-4o")
    llm = chat.llm.OpenRouter(model="anthropic/claude-sonnet-4-20250514")
    llm = chat.llm.OpenRouter(model="google/gemini-2.5-pro-preview")
    llm = chat.llm.OpenRouter(model="meta-llama/llama-3-70b-instruct")
    llm = chat.llm.OpenRouter(model="mistralai/mistral-large")

Browse all models: https://openrouter.ai/models

Running
-------
::

    export OPENROUTER_API_KEY="sk-or-v1-..."
    uv run --script examples/openrouter_models.py

Then open http://127.0.0.1:7777 in your browser.

What to Explore Next
--------------------
- Use a provider directly for lower latency: see ``llm_providers.py``
- Run models locally instead of via API: see ``ollama_local.py``
- Add tool calling (works through OpenRouter too): see ``tool_calling.py``
"""

import chatnificent as chat

welcome_message = """## Many models, one API

OpenRouter is one API that proxies your request to one of [hundreds of models](https://openrouter.ai/models). Try changing `model=` in `chat.llm.OpenRouter(model=...)` and rerun — e.g. `openai/gpt-4o-mini`, `anthropic/claude-3.5-haiku`, `meta-llama/llama-3.3-70b-instruct`.

<div id="suggestions">
  <button class="suggestion" data-insert-prompt="Refactor this Python: def f(x): return [i*2 for i in x if i%2==0]">
    <span class="suggestion-label">CODE</span>
    <span class="suggestion-text">Refactor a small Python function.</span>
  </button>
  <button class="suggestion" data-insert-prompt="Explain BGP routing to a backend developer in 4 sentences.">
    <span class="suggestion-label">EXPLAIN</span>
    <span class="suggestion-text">BGP routing in plain English.</span>
  </button>
  <button class="suggestion" data-insert-prompt="Write a 240-character launch tweet for a chat framework called Chatnificent.">
    <span class="suggestion-label">CREATE</span>
    <span class="suggestion-text">Product-launch tweet.</span>
  </button>
</div>"""

app = chat.Chatnificent(
    llm=chat.llm.OpenRouter(model="moonshotai/kimi-k2.5"),
    layout=chat.layout.Default(welcome_message=welcome_message),
)

if __name__ == "__main__":
    app.run()
