# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "chatnificent>=0.0.25",
# ]
# ///
"""
File serving — the canonical Artifact recipe (zero deps, no API key)
====================================================================

This is the *start here* file for everything Chatnificent does with
files. It teaches the one pattern every file-producing example in
``examples/`` follows — with no provider SDK, no API key, and no
network call — so the pattern itself stays in the foreground.

The pattern in 4 lines
----------------------
1. **Subclass the nearest LLM provider.** Here it's ``chat.llm.Echo``
   so the example runs offline. In real examples it's
   ``chat.llm.OpenAI``, ``chat.llm.Gemini``, etc.
2. **Override ``generate_response``** to call whatever SDK endpoint
   returns your bytes (TTS, image gen, video, …).
3. **Override ``extract_content``** to return ``chat.models.Artifact(...)``
   wrapping those bytes — give it a ``folder`` (``"audio"``, ``"images"``,
   …) and an ``ext`` (``".mp3"``, ``".png"``, …).
4. **Override ``parse_tool_calls`` to return ``None``** for pure file
   endpoints — this disables the agentic tool loop so the engine treats
   the response as final and ships it straight to the UI.

That's it. The framework handles the rest:

* writes the bytes to ``./<store_dir>/<user>/<convo>/<folder>/<N>.<ext>``
* mints an absolute URL ``/<user>/<convo>/<folder>/<N>.<ext>``
* swaps the ``Artifact`` for an ``<img>`` / ``<audio>`` / ``<video>`` /
  ``<a>`` snippet (chosen by MIME family) in the persisted message
* serves the bytes back when the browser requests the URL — including
  after a page refresh, in a second tab, or for a brand-new visitor

What this demo does
-------------------
For every user message it returns a freshly generated SVG image that
labels itself with the prompt. The "model" is just Python string
formatting — but to the engine and the UI this is indistinguishable
from a real image-generation API call.

Run it
------
::

    uv run --script examples/file_serving_simple.py

Open http://127.0.0.1:7777, send any message, and a labelled circle SVG
appears inline. Look in ``./_convos_file_serving_simple/`` to see the
bytes on disk under ``<user>/<convo>/images/0.svg``, ``1.svg``, …
alongside the canonical ``messages.json``.

Where to go from here
---------------------
Ready for multiple files per turn? See ``file_serving_advanced.py``.
Ready for a real provider? Every ``examples/*_simple.py`` /
``*_advanced.py`` is this same recipe with ``Echo`` swapped for
``OpenAI`` / ``Gemini`` / ``Anthropic`` and the SVG bytes swapped for a
real endpoint's output.
"""

import html

import chatnificent as chat
from chatnificent.models import USER_ROLE, Artifact


def _render_svg(prompt: str) -> bytes:
    """Generate a tiny SVG that visibly echoes the user's prompt."""
    label = prompt[:45] + "..." if len(prompt) > 45 else (prompt or "(empty prompt)")
    safe = html.escape(label)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="320" height="200" viewBox="0 0 320 200">
            <rect width="320" height="200" fill="#fef3c7"/>
            <circle cx="160" cy="90" r="55" fill="tomato"/>
            <text x="160" y="175" text-anchor="middle" font-family="system-ui, sans-serif" font-size="14" fill="#1f2937">{safe}</text>
            </svg>""".encode("utf-8")


class SVGEcho(chat.llm.Echo):
    """Zero-dep LLM that always returns an SVG Artifact built from the prompt."""

    def generate_response(self, messages, **kwargs):
        prompt = next(
            (
                m.get("content") or ""
                for m in reversed(messages)
                if m.get("role") == USER_ROLE
            ),
            "",
        )
        # The "response" can be any object the LLM understands — the engine
        # never inspects it, it just hands it to extract_content. Here we
        # pre-render the bytes so extract_content stays trivial.
        return {"svg": _render_svg(prompt)}

    def extract_content(self, response):
        return Artifact(data=response["svg"], ext=".svg", folder="images")

    def parse_tool_calls(self, response):
        # Disable the agentic loop: this endpoint is terminal — the
        # response IS the artifact, there is nothing to "call back" with.
        return None


app = chat.Chatnificent(
    llm=SVGEcho(stream=False),
    store=chat.store.File(base_dir="./_convos_file_serving_simple"),
)

if __name__ == "__main__":
    app.run()
