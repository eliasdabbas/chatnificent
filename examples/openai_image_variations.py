# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "chatnificent[openai]>=0.0.25",
# ]
# ///
"""
OpenAI image variations — one prompt, three takes
=================================================

Where ``openai_image_simple.py`` returns a single ``Artifact``, this
file returns a **list of strings and Artifacts**. The engine's
``_finalize_content`` already accepts that shape — each ``Artifact`` is
persisted via the Store and replaced inline with its HTML embed, each
``str`` passes through untouched, and the joined result becomes the
assistant message.

The display decision
--------------------
Variations exist to be **compared**, so tabs hide content and a vertical
stack scrolls forever. We render three thumbnails side-by-side; clicking
any one doubles its size in place so you can inspect detail without
losing the comparison. Pure HTML+CSS, zero JavaScript: each thumb is a
``<label>`` tied to a hidden checkbox, and an adjacent-sibling selector
swaps the width from 32% to 64% on ``:checked``.

Where the ``<style>`` block lives
---------------------------------
Not in the message. A ``<style>`` tag inside assistant HTML survives
the live SSE render but gets stripped on reload — ``marked`` wraps any
in-body ``<style>`` in a ``<p>`` and the HTML parser then ejects it.
Instead the stylesheet ships as a ``Control`` injected into the
``messages-begin`` slot, so it lives in the page chrome and bypasses
the per-message marked + DOMPurify pipeline entirely. Inline
``onclick`` was the other obvious escape hatch, but DOMPurify also
strips event handlers from message HTML.

Three teaching moments
----------------------
1. ``extract_content`` returning ``list[str | Artifact]`` works out of
   the box — strings pass through, each ``Artifact`` gets persisted and
   replaced with its rendered HTML.
2. ``Artifact.html`` per instance — the escape hatch for custom embed
   markup, no engine override needed.
3. Page-level CSS belongs in a ``Control``, not in the message stream.

Prerequisites
-------------
::

    export OPENAI_API_KEY="sk-..."
    uv run --script artifact_examples/openai_image_variations.py
"""

import base64

import chatnificent as chat
from chatnificent.layout import Control
from chatnificent.models import USER_ROLE, Artifact

# Injected once into the page chrome via a Control (see app below).
GALLERY_STYLE = """
<style>
.cn-grow-gallery { white-space: nowrap; }
.cn-grow-cb { position: absolute; left: -9999px; }
.cn-grow-label { display: inline-block; width: 32%; margin-right: 1%; vertical-align: top; cursor: zoom-in; white-space: normal; }
.cn-grow-label:last-of-type { margin-right: 0; }
.cn-grow-label img { width: 100%; height: auto; border-radius: 8px; display: block; box-shadow: 0 2px 8px rgba(0,0,0,0.15); }
.cn-grow-cb:checked + .cn-grow-label { width: 64%; cursor: zoom-out; }
</style>
"""

# {url} and {filename} are filled in by the engine when persisting each Artifact.
THUMB_HTML = """
<input class="cn-grow-cb" type="checkbox" id="grow-{filename}">
<label class="cn-grow-label" for="grow-{filename}"><img src="{url}" alt="{filename}"></label>
"""


class OpenAIImageVariations(chat.llm.OpenAI):
    def __init__(self, n: int = 3, size: str = "1024x1024", **kwargs):
        super().__init__(**kwargs)
        self.n = n
        self.size = size

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
            size=self.size,
            n=self.n,
        )

    def extract_content(self, response):
        thumbs = [
            Artifact(
                data=base64.b64decode(item.b64_json),
                ext=".png",
                folder="images",
                html=THUMB_HTML,
            )
            for item in response.data
        ]
        # Wrap the row in a block element: marked's `breaks: true` turns each `\n` between siblings into <br>, which would stack the inline-block thumbs vertically.
        return [
            '<div class="cn-grow-gallery">',
            *thumbs,
            "</div>",
        ]

    def parse_tool_calls(self, response):
        return None


welcome_message = """## One prompt, three variations

A single image rarely captures an idea on the first try. Type a prompt
and OpenAI returns three takes side-by-side — click any thumbnail to
enlarge it in place, click again to shrink it back.

<div id="suggestions">
  <button class="suggestion" data-insert-prompt="A logo for a coffee shop called 'Null Pointer' — minimalist, single color, vector style.">
    <span class="suggestion-label">LOGO</span>
    <span class="suggestion-text">Three logo directions for one brand.</span>
  </button>
  <button class="suggestion" data-insert-prompt="A character portrait of a retired space pirate, now running a botanical garden on a moon. Painterly, warm light.">
    <span class="suggestion-label">CHARACTER</span>
    <span class="suggestion-text">Three takes on the same character brief.</span>
  </button>
  <button class="suggestion" data-insert-prompt="An art deco poster for a jazz festival on Mars. Limited palette, bold geometry.">
    <span class="suggestion-label">POSTER</span>
    <span class="suggestion-text">Three poster compositions to choose from.</span>
  </button>
  <button class="suggestion" data-insert-prompt="A cozy reading nook by a rain-streaked window, photoreal, golden hour, shallow depth of field.">
    <span class="suggestion-label">SCENE</span>
    <span class="suggestion-text">Three moods for the same scene.</span>
  </button>
</div>
"""


app = chat.Chatnificent(
    llm=OpenAIImageVariations(model="gpt-image-1", stream=False),
    store=chat.store.File(
        base_dir="./artifact_examples/_convos_openai_image_variations"
    ),
    layout=chat.layout.Default(
        welcome_message=welcome_message,
        controls=[
            Control(
                id="cn-grow-gallery-style", slot="messages-begin", html=GALLERY_STYLE
            ),
        ],
    ),
)


if __name__ == "__main__":
    app.run()
