# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "chatnificent[openai]",
# ]
# ///
"""
System Prompt — Customize LLM Behavior
=======================================

Every LLM provider supports "system prompts" — instructions that shape how the
model behaves. The approach differs by provider:

OpenAI-compatible providers (OpenAI, OpenRouter, DeepSeek, Ollama)
    System prompts are just messages with ``"role": "system"``. Override
    ``generate_response()`` to prepend one if it doesn't already exist.
    This example shows exactly that.

Anthropic
    The SDK has a native ``system`` parameter. Pass it through the
    constructor: ``chat.llm.Anthropic(system="You are a pirate.")``.
    It flows into ``default_params`` and gets forwarded to
    ``messages.create()``.

Gemini
    The SDK has a native ``system_instruction`` parameter:
    ``chat.llm.Gemini(system_instruction="You are a pirate.")``.
    It flows into ``GenerateContentConfig``.

Why Override ``generate_response()``?
--------------------------------------
Your first instinct for a system prompt is "configure my LLM" — and that's
exactly right. Subclass your LLM provider, override ``generate_response()``,
and prepend a system message if there isn't one already. Three lines of logic,
zero framework magic.

For dynamic prompts — per-user personas, RAG context, conditional rules — you
can also override the engine's ``_prepare_llm_payload()`` seam. See
``engine.py`` for details.

Running
-------
::

    export OPENAI_API_KEY="sk-..."
    uv run examples/system_prompt.py

Try asking questions and notice how the LLM responds as a pirate!

What to Explore Next
--------------------
- Combine system prompts with tools (``tool_calling.py``) for a persona that
  can also call functions
- Use the engine's ``_prepare_llm_payload()`` seam for dynamic prompts that
  change per-request
- Read the system prompt from an environment variable or file for easy tuning
  without code changes
"""

import chatnificent as chat

PIRATE_PROMPT = (
    "You are a friendly pirate captain. Respond to everything in "
    "pirate-speak. Use 'arr', 'matey', 'ye', and nautical metaphors. "
    "Keep answers helpful but in character."
)


class PirateAI(chat.llm.OpenAI):
    def generate_response(self, messages, **kwargs):
        if not messages or messages[0].get("role") != "system":
            messages = [{"role": "system", "content": PIRATE_PROMPT}, *messages]
        return super().generate_response(messages, **kwargs)


app = chat.Chatnificent(llm=PirateAI())
# app = chat.Chatnificent(llm=chat.llm.Anthropic(system=PIRATE_PROMPT))
# app = chat.Chatnificent(llm=chat.llm.Gemini(system_instruction=PIRATE_PROMPT))

if __name__ == "__main__":
    app.run()
