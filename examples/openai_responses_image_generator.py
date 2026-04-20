# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "chatnificent[openai]",
# ]
# ///
"""
AI Image Chat — Battle-Test for Transparent Endpoints
======================================================

Every user message generates an image, inline, in the chat. Built on
OpenAI's hosted ``image_generation`` tool via the Responses API —
same pattern as the website search example, a different hosted tool.

Two Chatnificent principles on display:

**Minimally Complete**
    Enabling inline image generation is two constructor kwargs. The
    same subclass from ``openai_responses.py`` — nothing else
    changes at the framework layer::

        OpenAIResponses(
            tools=[{"type": "image_generation"}],
            tool_choice="required",
        )

    ``tool_choice="required"`` turns this into a pure image-generator
    product: every user message forces a call to the hosted
    ``image_generation`` tool. The Responses API runs the generation
    internally and emits the result in the stream. No separate
    ``images.generate`` endpoint call, no file plumbing — the model's
    response already contains the image.

**Maximally Hackable**
    Image results arrive as base64 PNG bytes on the
    ``image_generation_call`` output item. Our existing
    ``response.output_item.done`` branch in ``extract_stream_delta``
    just needs one more type check: when the completed item is an
    image generation call, format the base64 as a markdown image with
    a ``data:`` URL. Marked renders it as an ``<img>`` tag. No
    sidecar, no layout changes, no new routes — the default UI
    already supports embedded images.

    Cost of the hackability layer: a 5-line helper.

Things to Try
-------------
::

    export OPENAI_API_KEY="sk-..."
    uv run examples/openai_responses_image_generator.py

Then open http://127.0.0.1:7777 and type descriptions:

- "a tiny astronaut riding a capybara through a neon forest"
- "an art deco poster for a coffee shop on Mars"
- "a stained glass window depicting a rubber duck"

Each message produces one image. Conversation history is kept, so
you can follow up ("now in pastel", "same scene at night") and the
model has the prior context to riff on.

What This Example Does NOT Do (Yet)
------------------------------------
- **Data URLs are heavy.** A default 1024x1024 PNG base64-encoded is
  ~1–2 MB. Every image sits inline in ``messages.json`` and ships
  down the SSE stream on every replay. Production should save bytes
  as a sidecar file via the Store pillar and reference a short URL
  in the markdown — but the LLM pillar doesn't have Store access
  today. A clean fix is a framework change.
- **Only the default output format (PNG) is assumed.** Configuring
  ``output_format="webp"`` or ``"jpeg"`` on the tool would require
  updating the data URL media type.
- **No partial image rendering.** The hosted tool supports
  ``partial_images=N`` for progressive previews; we only render the
  final completed image.
- **No editing flow.** The tool supports ``action="edit"`` with an
  input image + mask, but wiring that up requires file-upload UI.

Friction Log (feeds the transparent-endpoint spec)
--------------------------------------------------
1. Binary-ish results (images, audio) would benefit from a pillar
   contract that knows about sidecar storage. Today the LLM pillar
   produces text-for-chat; artifacts ride inline as data URLs or not
   at all.
2. Hosted-tool progress events (image partials, search in-progress)
   still have no first-class status channel in
   ``extract_stream_delta``.
3. Output item types form a growing discriminated union
   (``message``, ``image_generation_call``, ``function_call``,
   ``web_search_call``, ``reasoning``, ...). A general dispatcher on
   ``item.type`` — rather than per-type branches — is probably the
   right shape for the eventual framework support.
"""

import chatnificent as chat


class OpenAIResponses(chat.llm.OpenAI):
    def generate_response(self, messages, **kwargs):
        return self.client.responses.create(
            model=self.model, input=messages, **{**self.default_params, **kwargs}
        )

    def extract_stream_delta(self, chunk):
        if chunk.type == "response.output_text.delta":
            return chunk.delta
        if chunk.type == "response.output_item.done":
            return self._format_image(chunk.item)
        return None

    def _format_image(self, item):
        if getattr(item, "type", None) != "image_generation_call":
            return None
        result = getattr(item, "result", None)
        if not result:
            return None
        return f"\n\n![Generated image](data:image/png;base64,{result})\n\n"


app = chat.Chatnificent(
    llm=OpenAIResponses(
        tools=[{"type": "image_generation"}],
        # tool_choice="required",
    ),
    store=chat.store.File(base_dir="openai_responses"),
)


if __name__ == "__main__":
    app.run()
