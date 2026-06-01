# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "chatnificent[gemini]>=0.0.25",
# ]
# ///
"""
Gemini music generation — simple
================================

The canonical recipe (see ``artifact_canonical_recipe.py``) wired to
Gemini's Lyria 3 music modality. Every user message is turned into a
30-second MP3 clip.

What's identical to the recipe
------------------------------
* Subclass the nearest LLM provider (``chat.llm.Gemini``).
* Override ``generate_response`` to call the SDK.
* Override ``extract_content`` to return ``Artifact(...)``.
* Override ``parse_tool_calls`` → ``None`` to disable the agentic loop.

What's new vs. ``gemini_tts_simple.py``
---------------------------------------
**Two parts in one response.** Lyria returns both lyrics (text parts)
and audio (one ``inline_data`` part). ``extract_content`` reuses the
list-of-strings-and-Artifacts contract from
``openai_image_variations.py``: the audio Artifact comes first (player
on top), the joined lyrics come second (Markdown below). One MP3 per
prompt, no PCM-to-WAV dance — Lyria already gives us a playable
container.

Prerequisites
-------------
::

    export GEMINI_API_KEY="..."  # or GOOGLE_API_KEY
    uv run --script artifact_examples/gemini_music_simple.py

Open http://127.0.0.1:7777 and try:

* "A 30-second lofi hip hop beat with mellow Rhodes piano, instrumental"
* "An upbeat pop song in G major about a summer road trip"
* "A dark atmospheric trap beat at 140 BPM in D minor"

Each prompt produces one MP3 under
``artifact_examples/_convos_gemini_music_simple/<user>/<convo>/music/``.
"""

import chatnificent as chat
from chatnificent.models import USER_ROLE, Artifact

# Per-Artifact embed: Store-served URL drops into <audio controls> so
# the clip plays inline. Without this Default would render a download
# link instead.
AUDIO_HTML = '<audio controls src="{url}"></audio>'


welcome_message = """## Lyria 3 — text to music

Describe a song and Lyria 3 composes a 30-second clip with lyrics. The
more specific the prompt — genre, tempo, key, mood, instruments — the
better the result.

<div id="suggestions">
  <button class="suggestion" data-insert-prompt="A 30-second lofi hip hop beat at 85 BPM with dusty vinyl crackle, mellow Rhodes piano chords, slow boom-bap drums, and a jazzy upright bass. Instrumental only, no vocals.">
    <span class="suggestion-label">LOFI</span>
    <span class="suggestion-text">Instrumental beat to study to.</span>
  </button>
  <button class="suggestion" data-insert-prompt="An upbeat feel-good pop song in G major at 120 BPM with bright acoustic guitar strumming, claps, and warm vocal harmonies about a summer road trip with friends.">
    <span class="suggestion-label">POP</span>
    <span class="suggestion-text">Sunny pop track about a road trip.</span>
  </button>
  <button class="suggestion" data-insert-prompt="A dark atmospheric trap beat at 140 BPM in D minor with heavy 808 bass, eerie synth pads, sharp hi-hats, and a haunting vocal sample.">
    <span class="suggestion-label">TRAP</span>
    <span class="suggestion-text">Dark, cinematic, heavy 808s.</span>
  </button>
  <button class="suggestion" data-insert-prompt="A 1980s synth-pop track at 118 BPM with shimmering analog synths, a punchy drum machine, and a catchy anthemic chorus about getting ready for a Friday night party.">
    <span class="suggestion-label">SYNTH-POP</span>
    <span class="suggestion-text">Retro-futuristic 80s anthem.</span>
  </button>
</div>
"""


class GeminiMusic(chat.llm.Gemini):
    def generate_response(self, messages, **kwargs):
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
        )

    def extract_content(self, response):
        # Lyria interleaves text (lyrics / structure) and one inline_data
        # part (the MP3). Order isn't guaranteed, so collect each kind
        # separately, then compose: player first, lyrics below.
        # `response.parts` is None (not []) when the request was blocked
        # (safety filter, copyrighted-lyrics request, artist-voice request).
        # Surface the reason instead of crashing.
        if not response.parts:
            feedback = getattr(response, "prompt_feedback", None)
            reason = getattr(feedback, "block_reason", None) or "unknown reason"
            return f"_Lyria returned no audio — blocked: **{reason}**. Try rephrasing the prompt._"
        lyrics = []
        audio = None
        for part in response.parts:
            if part.text is not None:
                lyrics.append(part.text)
            elif part.inline_data is not None:
                audio = part.inline_data.data
        items = []
        if audio is not None:
            items.append(
                Artifact(data=audio, ext=".mp3", folder="music", html=AUDIO_HTML)
            )
        if lyrics:
            items.append("\n\n".join(lyrics))
        return items

    def parse_tool_calls(self, response):
        return None


app = chat.Chatnificent(
    llm=GeminiMusic(model="lyria-3-clip-preview", stream=False),
    store=chat.store.File(base_dir="./artifact_examples/_convos_gemini_music_simple"),
    layout=chat.layout.Default(welcome_message=welcome_message),
)

if __name__ == "__main__":
    app.run()
