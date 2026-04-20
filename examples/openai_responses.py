# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "chatnificent[openai]",
# ]
# ///
"""
OpenAI Responses API — Battle-Test for Transparent Endpoints
=============================================================

Route Chatnificent through OpenAI's ``responses.create`` endpoint in **8 lines**
of subclass, zero framework changes. This is the smallest working implementation
we could honestly build against today's engine, and it's the baseline that the
eventual ``endpoint="responses.create"`` framework feature will need to beat.

The Two Overrides
-----------------
``generate_response``
    Route to ``client.responses.create``. Chat-completions message dicts pass
    through unchanged as Responses ``input`` items — the ``{"role", "content"}``
    shape is compatible for plain text turns.

``extract_stream_delta``
    The Responses stream yields many event types; we only propagate text deltas
    (``response.output_text.delta``). Non-text events (reasoning, tool calls,
    refusals, completion markers) become ``None`` and the engine skips them.

Why Only Two
------------
The engine's streaming no-tools path (what DevServer uses by default) calls
exactly two LLM-pillar methods: ``generate_response`` and ``extract_stream_delta``.
The finalization step builds the assistant message inline from accumulated
stream text — it never calls ``create_assistant_message`` or ``extract_content``.
And without registered tools, ``parse_tool_calls`` is never called either.
See ``engine.py:301-331`` for the exact flow.

Running
-------
::

    export OPENAI_API_KEY="sk-..."
    uv run examples/openai_responses.py

Then open http://127.0.0.1:7777.

Scope Ceiling (known limits of these 2 overrides)
-------------------------------------------------
- **No tools.** Adding tools pulls in ``parse_tool_calls`` and
  ``create_assistant_message``, both of which inherit chat-completions-shaped
  logic that crashes on a Responses ``Response`` object.
- **Streaming only.** ``handle_message`` (non-streaming) calls
  ``parse_tool_calls`` and ``extract_content`` every turn — both inherit
  broken behavior. DevServer happens to always stream, so this works out.
- **Reasoning / refusals silently dropped.** Our delta filter only forwards
  ``response.output_text.delta``. Reasoning traces and refusal events are
  discarded from the saved assistant message.
- **Raw request log has wrong shape.** ``build_request_payload`` inherits the
  chat-completions version, so ``raw_api_requests.jsonl`` persists a
  ``messages=...`` payload that doesn't match what actually went over the wire.
  Functional behavior is correct; only the audit trail lies.

Friction Log (feeds the transparent-endpoint spec)
--------------------------------------------------
1. Messages round-trip because the Responses API accepts chat-completions-shaped
   items as ``input``. The moment tools or multimodal content enter, that
   alignment breaks and we'll need real translation.
2. ``_OpenAICompatible`` hardcodes ``response.choices[0].message`` in multiple
   places. Any OpenAI-family endpoint other than chat.completions must override
   all of them or avoid the code paths that reach them. Option A: a thinner
   base class that only provides ``__init__``. Option B: make ``build_request_payload``
   the single seam and route everything through it.
3. ``build_request_payload`` is only used for raw API logging — not functionally
   required. But skipping it makes the audit log lie. Either the spec treats
   logging fidelity as optional, or every new endpoint must override it too.
"""

import chatnificent as chat


class OpenAIResponses(chat.llm.OpenAI):
    def generate_response(self, messages, **kwargs):
        return self.client.responses.create(
            model=self.model, input=messages, **{**self.default_params, **kwargs}
        )

    def extract_stream_delta(self, chunk):
        return chunk.delta if chunk.type == "response.output_text.delta" else None


app = chat.Chatnificent(llm=OpenAIResponses())

if __name__ == "__main__":
    app.run()
