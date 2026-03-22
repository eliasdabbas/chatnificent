# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "chatnificent[ollama]",
# ]
# ///
"""
Ollama Local — Run LLMs on Your Own Machine
============================================

This example connects Chatnificent to `Ollama <https://ollama.com/>`_, a tool
for running open-weight LLMs locally. No API key, no cloud, no data leaving
your machine.

.. important::

   Before running this example, you need to:

   1. **Install Ollama** — download from https://ollama.com/download and
      follow the installer. This gives you the ``ollama`` CLI and a local
      inference server.

   2. **Pull a model** — models are not included with Ollama. You must
      download at least one before chatting::

          ollama pull llama3.2

      This downloads ~2 GB. The model is cached locally and reused across
      runs. You only need to pull it once.

   The Ollama server starts automatically after installation. If it's not
   running, start it with ``ollama serve``.

How It Works
------------
``chatnificent[ollama]`` installs the ``ollama`` Python SDK.
``chat.llm.Ollama()`` creates a client that talks to the local Ollama server
(default: ``http://localhost:11434``). The rest of the app — streaming, UI,
conversation history — works identically to cloud providers.

Choosing a Model
----------------
Ollama supports hundreds of models. Pass any model you've pulled::

    llm = chat.llm.Ollama(model="llama3.2")  # Default — fast, capable
    llm = chat.llm.Ollama(model="mistral")  # Mistral 7B
    llm = chat.llm.Ollama(model="codellama")  # Code-focused
    llm = chat.llm.Ollama(model="gemma2")  # Google's Gemma 2

See all available models: https://ollama.com/library

Why Local?
----------
- **Privacy** — your conversations never leave your machine
- **No API costs** — run as many queries as you want
- **Offline capable** — works without internet after the initial model download
- **Experimentation** — try different models instantly with ``ollama pull``

Running
-------
::

    uv run --script examples/ollama_local.py

Then open http://127.0.0.1:7777 in your browser.

What to Explore Next
--------------------
- Compare local vs. cloud: swap ``Ollama`` for ``OpenAI`` in ``llm_providers.py``
- Add tools to a local model: see ``tool_calling.py`` (note: tool calling
  support varies by model — ``llama3.2`` supports it well)
- Persist local conversations: see ``persistent_storage.py``
"""

import chatnificent as chat

app = chat.Chatnificent(
    llm=chat.llm.Ollama(model="llama3.2"),
)

if __name__ == "__main__":
    app.run()
