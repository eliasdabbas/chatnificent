# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "chatnificent[gemini]>=0.0.25",
# ]
# ///
"""
Gemini text-to-speech — simple
==============================

Chat normally, then hear the answer. After Gemini finishes streaming a
text reply, the same LLM subclass kicks off a second ``generate_content``
call against an audio-capable model and yields the WAV as an
``Artifact`` into the same stream. The engine persists the file and the
default audio wrapper embeds an ``<audio controls>`` player right under
the spoken text — one bubble, text + player.

Same shape as ``openai_tts_advanced.py``: the LLM is a generator that
chains the real provider stream with one extra "thing" at the end.

What's identical to the canonical recipe
----------------------------------------
* Subclass the nearest LLM provider (``chat.llm.Gemini``).
* ``extract_stream_delta`` recognizes the trailing ``Artifact`` and
  hands it back so the engine persists + embeds it inline.

What's new compared to ``gemini_tts_simple``'s earlier "speak the prompt"
shape: we use two Gemini models — a chat model for the reply, an audio
model for the speech — and the audio model is invoked only once per turn.

Prerequisites
-------------
::

    export GEMINI_API_KEY="..."  # or GOOGLE_API_KEY
    uv run --script examples/gemini_tts_simple.py
"""

import io
import random
import wave

import chatnificent as chat
from chatnificent.models import Artifact

# Every Gemini TTS voice (see https://ai.google.dev/gemini-api/docs/speech-generation).
# Picked at random per turn so the showcase plays a different voice each time.
VOICES = (
    "Zephyr",
    "Puck",
    "Charon",
    "Kore",
    "Fenrir",
    "Leda",
    "Orus",
    "Aoede",
    "Callirrhoe",
    "Autonoe",
    "Enceladus",
    "Iapetus",
    "Umbriel",
    "Algieba",
    "Despina",
    "Erinome",
    "Algenib",
    "Rasalgethi",
    "Laomedeia",
    "Achernar",
    "Alnilam",
    "Schedar",
    "Gacrux",
    "Pulcherrima",
    "Achird",
    "Zubenelgenubi",
    "Vindemiatrix",
    "Sadachbia",
    "Sadaltager",
    "Sulafat",
)
TTS_MODEL = "gemini-3.1-flash-tts-preview"


def _pcm_to_wav(pcm: bytes, rate: int = 24_000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm)
    return buf.getvalue()


class GeminiChatPlusTTS(chat.llm.Gemini):
    """Stream a normal chat reply, then synthesize speech of that reply."""

    def generate_response(self, messages, **kwargs):
        text_stream = super().generate_response(messages, **kwargs)
        return self._chat_then_speak(text_stream)

    def _chat_then_speak(self, text_stream):
        spoken = []
        for chunk in text_stream:
            delta = super().extract_stream_delta(chunk)
            if delta:
                spoken.append(delta)
            yield chunk

        text = "".join(spoken).strip()
        if not text:
            return

        voice_name = random.choice(VOICES)
        types = self._genai_types
        voice = types.PrebuiltVoiceConfig(voice_name=voice_name)
        response = self.client.models.generate_content(
            model=TTS_MODEL,
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(prebuilt_voice_config=voice),
                ),
            ),
        )
        pcm = response.candidates[0].content.parts[0].inline_data.data
        # Tiny caption above the audio so the listener knows which of the 30
        # voices they got. Plain string — extract_stream_delta passes it
        # through unchanged.
        yield f"\n\n*Voice: **{voice_name}***\n\n"
        yield Artifact(data=_pcm_to_wav(pcm), ext=".wav", folder="audio")

    def extract_stream_delta(self, chunk):
        if isinstance(chunk, Artifact):
            return chunk
        if isinstance(chunk, str):
            return chunk
        return super().extract_stream_delta(chunk)


WELCOME_MESSAGE = """## Chat, then listen

Ask anything. Gemini streams a written answer, then synthesizes it as audio so you can read along — and play it back in the same bubble.

<div id="suggestions">
  <button class="suggestion" data-insert-prompt="Explain how server-sent events work in 3 short paragraphs.">
    <span class="suggestion-label">EXPLAIN</span>
    <span class="suggestion-text">Short, narratable answer.</span>
  </button>
  <button class="suggestion" data-insert-prompt="Tell me a 60-second bedtime story about a curious lighthouse.">
    <span class="suggestion-label">STORY</span>
    <span class="suggestion-text">Short, soothing read-aloud.</span>
  </button>
  <button class="suggestion" data-insert-prompt="Give me a friendly two-sentence welcome message for a new user of a chat app.">
    <span class="suggestion-label">VOICEOVER</span>
    <span class="suggestion-text">Tiny snippet, big warmth.</span>
  </button>
</div>"""


app = chat.Chatnificent(
    llm=GeminiChatPlusTTS(model="gemini-2.5-flash"),
    store=chat.store.File(base_dir="./artifact_examples/_convos_gemini_tts_simple"),
    layout=chat.layout.Default(welcome_message=WELCOME_MESSAGE),
)

if __name__ == "__main__":
    app.run()
