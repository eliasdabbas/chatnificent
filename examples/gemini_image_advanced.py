# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "chatnificent[gemini]>=0.0.25",
# ]
# ///
"""
Gemini image generation — advanced (Image Studio)
==================================================

Every turn is a fresh, parameterised image generation. The user picks the
*model*, *aspect ratio*, *resolution*, and *thinking level* from a sleek
"Image Studio" panel above the composer, then types a prompt. No
turn-by-turn editing — that's ``openai_image_advanced.py``'s story. This
example showcases a different axis of "advanced": **giving users
first-class access to provider knobs through Chatnificent's `Control`
pillar.**

Why no turn-by-turn editing here?
---------------------------------

Gemini-3 image models have no server-side image ID (the way OpenAI's
Responses API does with ``conversation_id``). The only way to continue
editing an image is to inline-replay its bytes *and* its original
``thought_signature`` on every turn. That works, but it's brittle (any
part-type mismatch invalidates the signature) and bloats every request
linearly with image history. We chose to lean into Gemini's strengths
instead: rich one-shot generation with all the dials exposed.

The four overrides (canonical recipe, extended)
-----------------------------------------------

1. **``generate_response``** — pops the ``_image_*`` kwargs injected by
   the ``Control`` objects out of ``**kwargs``, assembles a proper
   ``GenerateContentConfig`` (aspect ratio + image size go *nested*
   inside ``response_format.image``, not as top-level kwargs), and calls
   the ``generate_content`` SDK method.
2. **``extract_content``** — walks ``response.candidates[0].content.parts``
   for the first part with ``inline_data.data`` and returns
   ``Artifact(data=..., ext=<mime-mapped>, folder="images")``.
3. **``parse_tool_calls``** → ``None`` to disable the agentic loop.
4. *(That's it — no engine subclass, no bridge, no Store touch.)*

The ``Control`` API: four chip-groups, one panel
-------------------------------------------------

``Control`` binds one HTML element's value to one LLM kwarg. We want a
single cohesive "Image Studio" panel containing four chip groups, so we
register **five** controls on the ``composer-attachments`` slot — they
concatenate in registration order:

* **``img-studio-frame``** — owns the entire panel HTML (CSS + chip
  markup + click handlers + four hidden ``<input>`` elements).
  ``llm_param=None``, so it's visual-only.
* **``img-model``**, **``img-aspect``**, **``img-resolution``**,
  **``img-thinking``** — ``html=""``, binders only. Their ``id`` values
  match the hidden inputs inside the frame, so when a chip click calls
  ``chatInteraction(document.getElementById("img-..."))``, the value is
  POSTed to ``/api/interactions`` and merged into the next LLM call's
  kwargs as ``_image_<name>=<value>``.

Why the ``_image_`` prefix? ``Control.llm_param`` injects a single
top-level kwarg into ``generate_response(**kwargs)``. Gemini's image
params live nested inside
``GenerateContentConfig.response_format.image``. We use an internal
naming convention so the subclass knows exactly which kwargs to pop and
re-nest — and so any stray ``_image_*`` in a traceback obviously came
from a Control, not from a real Gemini API kwarg.

Run it
------

::

    export GEMINI_API_KEY="..."  # or GOOGLE_API_KEY
    uv run --script artifact_examples/gemini_image_advanced.py

Then open http://127.0.0.1:7777, expand the Image Studio panel above the
composer, pick your dials, and prompt away. Examples to try:

* Default 1:1 / 1K / Flash / Minimal → "a tiny astronaut riding a
  capybara through a neon forest"
* 16:9 / 2K / Flash / Minimal → "an art deco poster for a coffee shop
  on Mars"
* 9:16 / 1K / Flash / High → "a stained glass window depicting a rubber
  duck, intricate lead lines, sunset light"
* 1:1 / 2K / Pro / High → "minimalist logo for a bakery named
  'Flourish', single croissant motif, monochrome"

Each prompt produces one image, persisted under
``artifact_examples/_convos_gemini_image_advanced/<user>/<convo>/images/``.

Studio settings persist *per-user across conversations* (it's a tool,
not a chat property). Start a new chat → your dial settings stick.
"""

import chatnificent as chat
from chatnificent.layout import Control, Default
from chatnificent.models import USER_ROLE, Artifact

# ---- Model / dial vocabularies (single source of truth) ---------------------

MODELS = [
    ("gemini-3.1-flash-image", "Flash"),
    ("gemini-3-pro-image", "Pro"),
]
ASPECT_RATIOS = ["1:1", "16:9", "9:16", "4:3", "3:4", "21:9"]
RESOLUTIONS = ["512", "1K", "2K", "4K"]
THINKING_LEVELS = [("minimal", "Minimal"), ("high", "High")]

MIME_TO_EXT = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
}

DEFAULTS = {
    "img-model": "gemini-3.1-flash-image",
    "img-aspect": "1:1",
    "img-resolution": "1K",
    "img-thinking": "minimal",
}


# ---- Image Studio panel HTML (CSS + chips + hidden inputs + JS) -------------
# Visual idiom lifted from examples/single_app_multi_chat_mode.py — same
# rose-accented "studio" look. Self-contained: any CSS, markup, and JS the
# panel needs lives right here so the example stays one file.


def _chips(group: str, options, current: str) -> str:
    """Render the <button> chips for a single dial."""
    pieces = []
    for opt in options:
        value, label = (opt, opt) if isinstance(opt, str) else opt
        pressed = "true" if value == current else "false"
        pieces.append(
            f'<button type="button" class="img-chip" '
            f'data-group="{group}" data-value="{value}" '
            f'aria-pressed="{pressed}">{label}</button>'
        )
    return "\n      ".join(pieces)


PANEL_HTML = f"""
<style>
.img-studio {{
    --img-accent: #db2777;
    margin: 0 auto 10px;
    max-width: 640px;
    padding: 14px 18px 16px;
    border-radius: 22px;
    background: linear-gradient(135deg,
        color-mix(in srgb, var(--img-accent) 22%, white) 0%,
        color-mix(in srgb, var(--img-accent) 8%, white) 100%);
    border: 1px solid color-mix(in srgb, var(--img-accent) 35%, white);
    box-shadow:
        0 10px 32px color-mix(in srgb, var(--img-accent) 22%, transparent),
        inset 0 1px 0 rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(14px) saturate(140%);
    color: var(--text);
    text-align: left;
}}
html[data-theme="dark"] .img-studio {{
    background: linear-gradient(135deg,
        color-mix(in srgb, var(--img-accent) 18%, transparent) 0%,
        color-mix(in srgb, var(--img-accent) 4%, transparent) 100%);
    border-color: color-mix(in srgb, var(--img-accent) 30%, transparent);
    box-shadow: 0 10px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.12);
}}
.img-studio__head {{
    display: flex; align-items: center; gap: 10px;
    cursor: pointer; user-select: none;
    padding-bottom: 10px;
    border-bottom: 1px dashed color-mix(in srgb, var(--img-accent) 25%, transparent);
    margin-bottom: 12px;
}}
.img-studio__title {{
    font-size: 0.74rem; letter-spacing: 0.22em; text-transform: uppercase;
    font-weight: 700;
}}
.img-studio__led {{
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--img-accent);
    box-shadow: 0 0 8px var(--img-accent);
    animation: img-pulse 2.4s ease-in-out infinite;
    margin-left: 4px;
}}
@keyframes img-pulse {{ 0%,100% {{ opacity: .45; }} 50% {{ opacity: 1; }} }}
.img-studio__chevron {{
    margin-left: auto; width: 16px; height: 16px;
    color: var(--text-secondary);
    transition: transform 0.18s ease;
}}
.img-studio.is-collapsed {{ padding: 10px 18px; }}
.img-studio.is-collapsed .img-studio__head {{
    padding-bottom: 0; margin-bottom: 0; border-bottom: 0;
}}
.img-studio.is-collapsed .img-studio__chevron {{ transform: rotate(-90deg); }}
.img-studio.is-collapsed .img-studio__body {{ display: none; }}
.img-row {{ margin-bottom: 12px; }}
.img-row:last-child {{ margin-bottom: 0; }}
.img-row__label {{
    font-size: 0.6rem; letter-spacing: 0.22em; text-transform: uppercase;
    color: var(--text-secondary); margin-bottom: 6px; font-weight: 600;
}}
.img-chips {{ display: flex; flex-wrap: wrap; gap: 5px; }}
.img-chip {{
    appearance: none;
    background: color-mix(in srgb, var(--img-accent) 6%, var(--bg));
    border: 1px solid color-mix(in srgb, var(--img-accent) 18%, transparent);
    color: var(--text);
    padding: 5px 10px; border-radius: 999px;
    font: inherit; font-size: 0.72rem; letter-spacing: 0.04em;
    cursor: pointer;
    transition: all 0.12s ease;
}}
.img-chip:hover {{ border-color: color-mix(in srgb, var(--img-accent) 40%, transparent); }}
.img-chip[aria-pressed="true"] {{
    background: color-mix(in srgb, var(--img-accent) 22%, var(--bg));
    border-color: var(--img-accent);
    font-weight: 600;
}}
</style>

<div class="img-studio is-collapsed" id="img-studio-panel">
  <div class="img-studio__head" id="img-studio-head">
    <span class="img-studio__title">Image Studio</span>
    <span class="img-studio__led"></span>
    <svg class="img-studio__chevron" viewBox="0 0 24 24" fill="none"
         stroke="currentColor" stroke-width="2.2" stroke-linecap="round"
         stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>
  </div>
  <input type="hidden" id="img-model"      value="{DEFAULTS["img-model"]}">
  <input type="hidden" id="img-aspect"     value="{DEFAULTS["img-aspect"]}">
  <input type="hidden" id="img-resolution" value="{DEFAULTS["img-resolution"]}">
  <input type="hidden" id="img-thinking"   value="{DEFAULTS["img-thinking"]}">
  <div class="img-studio__body">
    <div class="img-row">
      <div class="img-row__label">Model</div>
      <div class="img-chips" data-group-for="img-model">
      {_chips("img-model", MODELS, DEFAULTS["img-model"])}
      </div>
    </div>
    <div class="img-row">
      <div class="img-row__label">Aspect ratio</div>
      <div class="img-chips" data-group-for="img-aspect">
      {_chips("img-aspect", ASPECT_RATIOS, DEFAULTS["img-aspect"])}
      </div>
    </div>
    <div class="img-row">
      <div class="img-row__label">Resolution</div>
      <div class="img-chips" data-group-for="img-resolution">
      {_chips("img-resolution", RESOLUTIONS, DEFAULTS["img-resolution"])}
      </div>
    </div>
    <div class="img-row">
      <div class="img-row__label">Thinking</div>
      <div class="img-chips" data-group-for="img-thinking">
      {_chips("img-thinking", THINKING_LEVELS, DEFAULTS["img-thinking"])}
      </div>
    </div>
  </div>
</div>

<script>
(function() {{
  var panel = document.getElementById("img-studio-panel");
  var head  = document.getElementById("img-studio-head");
  if (head) head.addEventListener("click", function() {{
    panel.classList.toggle("is-collapsed");
  }});
  panel.querySelectorAll(".img-chip").forEach(function(chip) {{
    chip.addEventListener("click", function() {{
      var group = chip.getAttribute("data-group");
      var value = chip.getAttribute("data-value");
      var siblings = panel.querySelectorAll(
        '.img-chip[data-group="' + group + '"]'
      );
      siblings.forEach(function(s) {{ s.setAttribute("aria-pressed", "false"); }});
      chip.setAttribute("aria-pressed", "true");
      var input = document.getElementById(group);
      if (input) {{
        input.value = value;
        chatInteraction(input);
      }}
    }});
  }});
}})();
</script>
"""


# ---- GeminiImageStudio LLM --------------------------------------------------


class GeminiImageStudio(chat.llm.Gemini):
    """One-shot Gemini image generation parameterised by Image Studio controls.

    Pops ``_image_*`` kwargs (injected by Controls) and re-nests them into
    the SDK-shaped ``GenerateContentConfig``. Returns the live response
    object so ``extract_content`` can walk straight to raw image bytes.
    """

    def generate_response(self, messages, **kwargs):
        types = self._genai_types
        # Pop Control-injected kwargs (with fallbacks for first turn before
        # any interaction has fired).
        model = kwargs.pop("_image_model", self.model)
        aspect = kwargs.pop("_image_aspect", DEFAULTS["img-aspect"])
        resolution = kwargs.pop("_image_resolution", DEFAULTS["img-resolution"])
        thinking = kwargs.pop("_image_thinking_level", DEFAULTS["img-thinking"])

        prompt = next(
            (
                m.get("content") or ""
                for m in reversed(messages)
                if m.get("role") == USER_ROLE
            ),
            "",
        )

        config = types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(aspect_ratio=aspect, image_size=resolution),
            thinking_config=types.ThinkingConfig(thinking_level=thinking),
        )
        return self.client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )

    def extract_content(self, response):
        parts = (
            (response.candidates[0].content.parts or []) if response.candidates else []
        )
        for part in parts:
            inline = getattr(part, "inline_data", None)
            if inline and inline.data:
                ext = MIME_TO_EXT.get(getattr(inline, "mime_type", "") or "", ".png")
                return Artifact(data=inline.data, ext=ext, folder="images")
        finish = (
            getattr(response.candidates[0], "finish_reason", "UNKNOWN")
            if response.candidates
            else "NO_CANDIDATES"
        )
        return f"No image returned (finish_reason: {finish})."

    def parse_tool_calls(self, response):
        return None


# ---- Controls: 1 frame + 4 binders, all on the same slot --------------------

CONTROLS = [
    Control(id="img-studio-frame", html=PANEL_HTML, slot="messages-end"),
    Control(
        id="img-model",
        html="",
        slot="messages-end",
        llm_param="_image_model",
    ),
    Control(
        id="img-aspect",
        html="",
        slot="messages-end",
        llm_param="_image_aspect",
    ),
    Control(
        id="img-resolution",
        html="",
        slot="messages-end",
        llm_param="_image_resolution",
    ),
    Control(
        id="img-thinking",
        html="",
        slot="messages-end",
        llm_param="_image_thinking_level",
    ),
]


app = chat.Chatnificent(
    llm=GeminiImageStudio(model="gemini-3.1-flash-image", stream=False),
    layout=Default(
        brand="Image Studio",
        slogan="Nano Banana \u00b7 dialled in",
        welcome_message=(
            "## Image Studio\n\n"
            "Expand the **Image Studio** panel above the composer to pick a "
            "model, aspect ratio, resolution, and thinking level — then type "
            "a prompt and hit send. Each turn is one fresh image."
        ),
        controls=CONTROLS,
    ),
    store=chat.store.File(base_dir="./artifact_examples/_convos_gemini_image_advanced"),
)

if __name__ == "__main__":
    app.run()
