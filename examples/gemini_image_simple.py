# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "chatnificent[gemini]>=0.0.25",
# ]
# ///
"""
Gemini image generation — simple
================================

The canonical recipe (see ``artifact_canonical_recipe.py``) wired to a
different provider. Every user message becomes an image via Google's
Gemini API.

What's identical to the recipe
------------------------------
* Subclass the nearest LLM provider (``chat.llm.Gemini``).
* Override ``generate_response`` to call the SDK that returns bytes.
* Override ``extract_content`` to return ``Artifact(...)``.
* Override ``parse_tool_calls`` → ``None`` to disable the agentic loop.

What changes vs. ``openai_image_simple.py``
-------------------------------------------
**Different SDK shape, same recipe.** Three small differences from the
OpenAI version, none of which touch the Artifact contract:

1. **No dedicated images endpoint.** Gemini routes images through the
   regular ``generate_content`` call. You opt in by asking for the
   ``"IMAGE"`` modality in the ``GenerateContentConfig``.
2. **Raw bytes, not base64.** Each candidate's content is a list of
   ``Part`` objects; an image part exposes its bytes directly at
   ``part.inline_data.data`` — no ``base64.b64decode`` step needed.
3. **We don't reuse the base class's response handling.** Gemini's
   parent class returns ``response.model_dump(mode="json")`` so it can
   serialize raw bytes as base64 strings for JSON safety. For an image
   pipeline that's the wrong shape: we want to keep the raw bytes. So
   we call ``generate_content`` ourselves and let
   ``extract_content`` reach straight for ``part.inline_data.data``.

That's the whole adaptation. Different SDK, same four overrides, same
``Artifact`` contract.

Prerequisites
-------------
::

    export GEMINI_API_KEY="..."  # or GOOGLE_API_KEY
    uv run --script artifact_examples/gemini_image_simple.py

Open http://127.0.0.1:7777 and type any prompt — for example:

* "a tiny astronaut riding a capybara through a neon forest"
* "an art deco poster for a coffee shop on Mars"
* "a stained glass window depicting a rubber duck"

Each prompt produces one PNG, persisted under
``artifact_examples/_convos_gemini_image_simple/<user>/<convo>/images/``.
"""

import chatnificent as chat
from chatnificent.models import USER_ROLE, Artifact


class GeminiImage(chat.llm.Gemini):
    def generate_response(self, messages, **kwargs):
        types = self._genai_types
        prompt = next(
            (
                m.get("content") or ""
                for m in reversed(messages)
                if m.get("role") == USER_ROLE
            ),
            "",
        )
        return self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
        )

    def extract_content(self, response):
        parts = (
            (response.candidates[0].content.parts or []) if response.candidates else []
        )
        for part in parts:
            inline = getattr(part, "inline_data", None)
            if inline and inline.data:
                return Artifact(data=inline.data, ext=".png", folder="images")
        # No image returned — surface the reason rather than crashing downstream.
        finish = (
            getattr(response.candidates[0], "finish_reason", "UNKNOWN")
            if response.candidates
            else "NO_CANDIDATES"
        )
        finish_name = getattr(finish, "name", str(finish))
        if finish_name == "NO_IMAGE":
            return (
                "Gemini declined to generate an image for that prompt "
                "(finish_reason: NO_IMAGE). Try rephrasing — be more specific "
                "about subject, style, and setting, or pick a different theme."
            )
        return (
            f"No image returned (finish_reason: {finish_name}). "
            f"Is `{self.model}` actually an image-capable Gemini model? "
            "List your available models with: `for m in client.models.list(): print(m.name)`"
        )

    def parse_tool_calls(self, response):
        return None


WELCOME_MESSAGE = """## Gemini, draw me something

Every prompt becomes a PNG. Describe a scene — style, mood, subject — and Gemini renders it in the bubble below.

<div id="suggestions">
  <button class="suggestion" data-insert-prompt="A tiny astronaut riding a capybara through a neon forest, cinematic lighting">
    <span class="suggestion-label">WHIMSY</span>
    <span class="suggestion-text">Unlikely hero, neon forest.</span>
  </button>
  <button class="suggestion" data-insert-prompt="An art deco travel poster for a coffee shop on Mars, bold geometric shapes">
    <span class="suggestion-label">POSTER</span>
    <span class="suggestion-text">Art deco, Mars edition.</span>
  </button>
  <button class="suggestion" data-insert-prompt="A stained glass window depicting a rubber duck, sunlit cathedral">
    <span class="suggestion-label">STAINED GLASS</span>
    <span class="suggestion-text">Sacred rubber duckery.</span>
  </button>
  <button class="suggestion" data-insert-prompt="Isometric pixel-art diorama of a tiny ramen shop on a rainy Tokyo street at night">
    <span class="suggestion-label">PIXEL ART</span>
    <span class="suggestion-text">Tiny ramen, big mood.</span>
  </button>
</div>"""


app = chat.Chatnificent(
    llm=GeminiImage(model="gemini-3.1-flash-image", stream=False),
    store=chat.store.File(base_dir="./artifact_examples/_convos_gemini_image_simple"),
    layout=chat.layout.Default(welcome_message=WELCOME_MESSAGE),
)

if __name__ == "__main__":
    app.run()
