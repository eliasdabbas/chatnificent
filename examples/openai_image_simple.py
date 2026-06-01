# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "chatnificent[openai]>=0.0.25",
# ]
# ///
"""
OpenAI image generation — simple
================================

The canonical recipe (see ``artifact_canonical_recipe.py``) wired to a real
provider: every user message becomes an image via OpenAI's
``images.generate`` endpoint.

What's identical to the recipe
------------------------------
* Subclass the nearest LLM provider (``chat.llm.OpenAI``).
* Override ``generate_response`` to call the SDK endpoint that returns bytes.
* Override ``extract_content`` to return ``Artifact(...)``.
* Override ``parse_tool_calls`` → ``None`` to disable the agentic loop.

What's new vs. the recipe
-------------------------
**One base64 decode.** OpenAI's images endpoint returns the PNG as a base64
**string** in ``response.data[0].b64_json`` (it's how their JSON response
shape works — the SDK does not hand us raw bytes for this endpoint). We
decode once at extraction time to recover the actual PNG bytes, then hand
those bytes to ``Artifact``. The framework writes them to
``images/0.png``, ``1.png``, … and embeds an ``<img src=".../images/0.png">``
tag in the message body. **No base64 ever appears in the conversation, on
disk, or in the served response.**

Other providers shape this differently — OpenAI's TTS returns raw bytes
via ``response.read()``, Gemini returns raw bytes on
``part.inline_data.data``. The pattern stays the same; only the one-liner
inside ``extract_content`` changes.

Prerequisites
-------------
::

    export OPENAI_API_KEY="sk-..."
    uv run --script artifact_examples/openai_image_simple.py

Open http://127.0.0.1:7777 and type any prompt — for example:

* "a tiny astronaut riding a capybara through a neon forest"
* "an art deco poster for a coffee shop on Mars"
* "a stained glass window depicting a rubber duck"

Each prompt produces one image, persisted under
``artifact_examples/_convos_openai_image_simple/<user>/<convo>/images/``.
"""

import base64

import chatnificent as chat
from chatnificent.models import USER_ROLE, Artifact


class OpenAIImage(chat.llm.OpenAI):
    def generate_response(self, messages, **kwargs):
        prompt = next(
            (
                m.get("content") or ""
                for m in reversed(messages)
                if m.get("role") == USER_ROLE
            ),
            "",
        )
        return self.client.images.generate(
            model=self.model,
            prompt=prompt,
            size="1024x1024",
            n=1,
        )

    def extract_content(self, response):
        png_bytes = base64.b64decode(response.data[0].b64_json)
        return Artifact(data=png_bytes, ext=".png", folder="images")

    def parse_tool_calls(self, response):
        return None


WELCOME_MESSAGE = """## OpenAI, draw me something

Every prompt becomes a 1024×1024 PNG via `images.generate`. Describe a scene — subject, style, lighting — and the image renders straight in the bubble.

<div id="suggestions">
  <button class="suggestion" data-insert-prompt="A tiny astronaut riding a capybara through a neon forest, cinematic lighting">
    <span class="suggestion-label">WHIMSY</span>
    <span class="suggestion-text">Unlikely hero, neon forest.</span>
  </button>
  <button class="suggestion" data-insert-prompt="An art deco travel poster for a coffee shop on Mars, bold geometric shapes, vintage palette">
    <span class="suggestion-label">POSTER</span>
    <span class="suggestion-text">Art deco, Mars edition.</span>
  </button>
  <button class="suggestion" data-insert-prompt="A stained glass window depicting a rubber duck, sunlit cathedral, intricate lead lines">
    <span class="suggestion-label">STAINED GLASS</span>
    <span class="suggestion-text">Sacred rubber duckery.</span>
  </button>
  <button class="suggestion" data-insert-prompt="Isometric pixel-art diorama of a tiny ramen shop on a rainy Tokyo street at night, glowing signs">
    <span class="suggestion-label">PIXEL ART</span>
    <span class="suggestion-text">Tiny ramen, big mood.</span>
  </button>
</div>"""


app = chat.Chatnificent(
    llm=OpenAIImage(model="gpt-image-1", stream=False),
    store=chat.store.File(base_dir="./artifact_examples/_convos_openai_image_simple"),
    layout=chat.layout.Default(welcome_message=WELCOME_MESSAGE),
)

if __name__ == "__main__":
    app.run()
