# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "chatnificent>=0.0.25",
# ]
# ///
"""
File serving — multiple file types in one message (zero deps, no API key)
=========================================================================

Where ``file_serving_simple.py`` returns a single ``Artifact``, this
example returns a **list of strings and Artifacts of different file
types** in a single assistant turn. The framework:

1. **Serves files inline.** ``extract_content`` returns
   ``list[str | Artifact]``; the engine persists each ``Artifact`` and
   weaves the embeds back into the message text.
2. **Picks the embed by MIME type.** ``.svg`` → ``<img>``,
   ``.wav`` → ``<audio>``, ``.txt`` → ``<a download>``. Same
   ``Artifact`` class, different MIME family.
3. **Uses a predictable URL.** Files live at
   ``/<user>/<convo>/<folder>/<N>.<ext>`` — open the URL in any tab,
   email it to yourself, ``curl`` it. It's just a file on disk.

Try it
------
::

    uv run --script examples/file_serving_advanced.py

Send any message. You'll get an SVG card, a short WAV beep, and a
plain-text receipt — all from one turn — and you can see the actual
files at ``./_convos_file_serving_advanced/<user>/<convo>/...``.
"""

import html
import io
import math
import struct
import wave

import chatnificent as chat
from chatnificent.models import USER_ROLE, Artifact


def _svg_card(prompt: str) -> bytes:
    safe = html.escape(prompt[:55] + "..." if len(prompt) > 55 else prompt)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="320" height="160" viewBox="0 0 320 160">
  <rect width="320" height="160" fill="#0f172a"/>
  <text x="160" y="60" text-anchor="middle" font-family="system-ui" font-size="13" fill="#94a3b8">you said</text>
  <text x="160" y="95" text-anchor="middle" font-family="system-ui" font-size="16" font-weight="600" fill="#f8fafc">{safe}</text>
</svg>""".encode("utf-8")


def _wav_chime(seconds: float = 1.8) -> bytes:
    """Soft C-major arpeggio with an exponential decay envelope.

    Stdlib only — ``wave`` + ``math`` + ``struct`` produce a real 16-bit
    mono WAV. The envelope is what makes it pleasant: a hard-edged sine
    sounds like a buzzer, a decaying one sounds like a chime.
    """
    rate = 22050
    notes = [523.25, 659.25, 783.99]  # C5, E5, G5 — major triad
    total = int(seconds * rate)
    frames = bytearray()
    for i in range(total):
        t = i / rate
        # Stagger note onsets so they arpeggiate rather than play as a chord.
        sample = 0.0
        for k, f in enumerate(notes):
            onset = k * (seconds / len(notes)) * 0.5
            if t < onset:
                continue
            local = t - onset
            envelope = math.exp(-1.5 * local)
            sample += math.sin(2 * math.pi * f * local) * envelope
        sample = max(-1.0, min(1.0, sample / len(notes))) * 0.6
        frames += struct.pack("<h", int(32767 * sample))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(frames))
    return buf.getvalue()


def _receipt_txt(prompt: str) -> bytes:
    return f"echo receipt\n-----------\nprompt: {prompt}\nbytes:  {len(prompt.encode())}\n".encode()


class FilesEcho(chat.llm.Echo):
    def generate_response(self, messages, **kwargs):
        prompt = next(
            (
                m.get("content") or ""
                for m in reversed(messages)
                if m.get("role") == USER_ROLE
            ),
            "",
        )
        return {"prompt": prompt}

    def extract_content(self, response):
        prompt = response["prompt"] or "(empty)"
        return [
            "Three files, one turn:\n\n**Image** (rendered inline as `<img>`):\n\n",
            Artifact(data=_svg_card(prompt), ext=".svg", folder="cards"),
            "\n\n**Audio** (rendered inline as `<audio>`):\n\n",
            Artifact(data=_wav_chime(), ext=".wav", folder="audio"),
            "\n\n**Text** (rendered as a download link):\n\n",
            Artifact(data=_receipt_txt(prompt), ext=".txt", folder="receipts"),
            "\n\n*Type another message to see the filenames increment* — `cards/1.svg`, `cards/2.svg`, …",
        ]

    def parse_tool_calls(self, response):
        return None


WELCOME = """## Multiple files in one message

One assistant turn — three Artifacts of different MIME types. The framework picks the embed automatically: `image/*` → `<img>`, `audio/*` → `<audio>`, everything else → `<a download>`.

After you send a message, each file lives on disk under your conversation directory at:

- `cards/0.svg`
- `audio/0.wav`
- `receipts/0.txt`

Each is also served live at `/<user>/<convo>/<folder>/<N>.<ext>` — open the URLs in another tab, share them, `curl` them.

<div id="suggestions">
  <button class="suggestion" data-insert-prompt="hello world">
    <span class="suggestion-label">DEMO</span>
    <span class="suggestion-text">Send any prompt and get back three files at once.</span>
  </button>
</div>"""


app = chat.Chatnificent(
    llm=FilesEcho(stream=False),
    store=chat.store.File(base_dir="./_convos_file_serving_advanced"),
    layout=chat.layout.Default(welcome_message=WELCOME),
)

if __name__ == "__main__":
    app.run()
