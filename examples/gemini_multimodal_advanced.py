# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "chatnificent[gemini]>=0.0.25",
# ]
# ///
"""
Gemini multimodal — advanced (sequential story + illustration)
=============================================================

This advanced example demonstrates how to generate a short story (text) and then a separate illustration (image) for that story, in a single assistant message — but using **two independent, stateless API calls**. There is no continuity or memory between the image and the story beyond what you pass explicitly.

**Note:** Gemini's image models do not remember previous images or story context. Each image generation is stateless and based only on the prompt you provide. There is no way to reference or edit previous images.

Why two calls and not one?
--------------------------
Gemini *does* support `response_modalities=["TEXT", "IMAGE"]` in a single call on image-capable models, but those models are tuned for the image and their text output tends to be terse. For a high-quality illustrated reply, two purpose-built calls — one chat model, one image model — produce a far better result. The trade-off (extra latency, extra cost) is honest: you get a real story plus a real illustration.

What this example teaches
-------------------------
`extract_stream_delta` can return `str`, `Artifact`, or `None`. `generate_response` can be a generator that yields chunks of any shape. This allows an assistant turn to chain two endpoints and interleave their output, with the framework persisting each piece correctly.

How this maps to the canonical recipe
-------------------------------------
Same four overrides:

* `generate_response` is a generator that chains chat → image.
* `extract_stream_delta` recognises an image sentinel and returns an `Artifact`; otherwise delegates to `super()` for text deltas.
* `parse_tool_calls` stays `None`.
* `extract_content` is irrelevant on the streaming path.

Prerequisites
-------------
::

    export GEMINI_API_KEY="..."  # or GOOGLE_API_KEY
    uv run --script artifact_examples/gemini_multimodal_advanced.py

Story and image files are saved under
`artifact_examples/_convos_gemini_multimodal_advanced/<user>/<convo>/images/`.
"""

import chatnificent as chat
from chatnificent.models import Artifact

CHAT_MODEL = "gemini-2.5-flash"
IMAGE_MODEL = "gemini-3.1-flash-image"

ILLUSTRATION_PROMPT = (
    "Create a single illustration that captures the scene below. "
    "Style: warm, storybook, painterly. No text in the image.\n\n{text}"
)


class GeminiChatPlusImage(chat.llm.Gemini):
    def generate_response(self, messages, **kwargs):
        chat_stream = super().generate_response(messages, **kwargs)
        return self._chat_then_image(chat_stream)

    def _chat_then_image(self, chat_stream):
        spoken = []
        for chunk in chat_stream:
            text = getattr(chunk, "text", None)
            if text:
                spoken.append(text)
            yield chunk

        story = "".join(spoken).strip()
        if not story:
            return

        types = self._genai_types
        response = self.client.models.generate_content(
            model=IMAGE_MODEL,
            contents=ILLUSTRATION_PROMPT.format(text=story),
            config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
        )
        for part in response.candidates[0].content.parts or []:
            inline = getattr(part, "inline_data", None)
            if inline and inline.data:
                yield ("image", inline.data)
                return

    def extract_stream_delta(self, chunk):
        if isinstance(chunk, tuple) and len(chunk) == 2 and chunk[0] == "image":
            return Artifact(data=chunk[1], ext=".png", folder="images")
        return super().extract_stream_delta(chunk)


welcome_message = """## Story + illustration (stateless, sequential)

This example generates a short story, then a separate illustration for that story — using two independent Gemini API calls. **Note:** The image model does not remember previous images or story context. Each image is generated from scratch, based only on the prompt you provide.

Two Gemini endpoints in one turn: `gemini-2.5-flash` writes the story, `gemini-3.1-flash-image` paints it.

<div id="suggestions">
  <button class="suggestion" data-insert-prompt="Tell me a 3-sentence bedtime story about a curious raccoon who discovers a lantern in the woods.">
    <span class="suggestion-label">BEDTIME</span>
    <span class="suggestion-text">A tiny story with a clear visual scene.</span>
  </button>
  <button class="suggestion" data-insert-prompt="Describe a cozy kitchen on a rainy Sunday morning in two short paragraphs.">
    <span class="suggestion-label">SCENE</span>
    <span class="suggestion-text">Atmosphere over plot — great for the illustrator.</span>
  </button>
  <button class="suggestion" data-insert-prompt="Write a 4-line poem about a tiny astronaut riding a capybara through a neon forest.">
    <span class="suggestion-label">POEM</span>
    <span class="suggestion-text">A surreal image waiting to be drawn.</span>
  </button>
  <button class="suggestion" data-insert-prompt="In 3 sentences, describe what a stained-glass window depicting a rubber duck would look like.">
    <span class="suggestion-label">DESCRIBE</span>
    <span class="suggestion-text">Ask for a visual; get it back as one.</span>
  </button>
</div>
"""

app = chat.Chatnificent(
    llm=GeminiChatPlusImage(model=CHAT_MODEL),
    store=chat.store.File(base_dir="./artifact_examples/_convos_gemini_multimodal_advanced"),
    layout=chat.layout.Default(welcome_message=welcome_message),
)

if __name__ == "__main__":
    app.run()
