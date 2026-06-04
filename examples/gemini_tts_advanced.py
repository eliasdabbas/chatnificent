# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "chatnificent[gemini]>=0.0.25",
# ]
# ///
"""
Gemini text-to-speech — advanced (multi-speaker dialog)
=======================================================

What makes this "advanced" is **not** more code on our side — it's a
Gemini-only capability: a single ``generate_content`` call renders a
two-speaker dialog into one stitched WAV. You write a script with named
turns, the model assigns the right voice to each turn, and you get back
one audio file.

The four canonical overrides are unchanged. The only growth is inside
``speech_config``: ``multi_speaker_voice_config`` instead of
``voice_config``.

Prerequisites
-------------
::

    export GEMINI_API_KEY="..."  # or GOOGLE_API_KEY
    uv run --script artifact_examples/gemini_tts_advanced.py

Open http://127.0.0.1:7777 and click a suggestion chip — each one is a
ready-made two-speaker script. Just press Send.
"""

import io
import random
import wave

import chatnificent as chat
from chatnificent.models import USER_ROLE, Artifact

SPEAKER_A = "Alice"
SPEAKER_B = "Bob"

# Gemini's 30 TTS voices, split by perceived gender (best-effort grouping
# based on Google's AI Studio previews — see
# https://ai.google.dev/gemini-api/docs/speech-generation). We pick one
# from each pool per turn so Alice (feminine) and Bob (masculine) get
# voices that match their names while still varying every render.
FEMININE_VOICES = (
    "Aoede", "Autonoe", "Callirrhoe", "Despina", "Erinome", "Kore",
    "Laomedeia", "Leda", "Pulcherrima", "Sulafat", "Vindemiatrix", "Zephyr",
)
MASCULINE_VOICES = (
    "Achernar", "Achird", "Algenib", "Algieba", "Alnilam", "Charon",
    "Enceladus", "Fenrir", "Gacrux", "Iapetus", "Orus", "Puck",
    "Rasalgethi", "Sadaltager", "Schedar", "Umbriel", "Zubenelgenubi",
)


def _pcm_to_wav(pcm: bytes, rate: int = 24_000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm)
    return buf.getvalue()


class GeminiDialogTTS(chat.llm.Gemini):
    def generate_response(self, messages, **kwargs):
        types = self._genai_types
        script = next(
            (
                m.get("content") or ""
                for m in reversed(messages)
                if m.get("role") == USER_ROLE
            ),
            "",
        )

        def _speaker(name, voice):
            return types.SpeakerVoiceConfig(
                speaker=name,
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice),
                ),
            )

        voice_a = random.choice(FEMININE_VOICES)
        voice_b = random.choice(MASCULINE_VOICES)

        return self.client.models.generate_content(
            model=self.model,
            contents=f"TTS the following conversation between {SPEAKER_A} and {SPEAKER_B}:\n{script}",
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                        speaker_voice_configs=[
                            _speaker(SPEAKER_A, voice_a),
                            _speaker(SPEAKER_B, voice_b),
                        ],
                    ),
                ),
            ),
        )

    def extract_content(self, response):
        pcm = response.candidates[0].content.parts[0].inline_data.data
        return Artifact(data=_pcm_to_wav(pcm), ext=".wav", folder="audio")

    def parse_tool_calls(self, response):
        return None


welcome_message = """## Two-speaker dialog — Gemini's TTS superpower

One API call. Two named voices. One stitched WAV. Write a script with
**Alice:** and **Bob:** turns and Gemini renders the whole exchange.

Pick a ready-made dialog below, or write your own.

<div id="suggestions">
  <button class="suggestion" data-insert-prompt="Alice: Did you hear they finally landed the rover?&#10;Bob: I did! And it actually worked on the first try, which honestly stunned me.&#10;Alice: Same. The team must be celebrating.&#10;Bob: Champagne and tears, I'd bet.">
    <span class="suggestion-label">NEWS</span>
    <span class="suggestion-text">A short reaction to a news headline.</span>
  </button>
  <button class="suggestion" data-insert-prompt="Alice: Okay, pitch me your startup idea in one sentence.&#10;Bob: Uber, but for left socks.&#10;Alice: …I have so many questions.&#10;Bob: I have exactly zero answers, which is why I need funding.">
    <span class="suggestion-label">COMEDY</span>
    <span class="suggestion-text">A two-line joke with proper timing.</span>
  </button>
  <button class="suggestion" data-insert-prompt="Alice: Welcome back to the podcast. Today we're talking about sleep.&#10;Bob: A topic I am, ironically, very tired of.&#10;Alice: Let's start with the basics — why do we need it at all?&#10;Bob: Honestly? Nobody fully knows. But the brain definitely files paperwork while you're out.">
    <span class="suggestion-label">PODCAST</span>
    <span class="suggestion-text">Podcast intro with banter.</span>
  </button>
  <button class="suggestion" data-insert-prompt="Alice: Are you sure this is the right cave?&#10;Bob: I'm sure it's a cave. Beyond that, no promises.&#10;Alice: Fantastic. Truly inspiring leadership.&#10;Bob: Thank you. I've been working on it.">
    <span class="suggestion-label">SCENE</span>
    <span class="suggestion-text">A tiny adventure scene.</span>
  </button>
</div>
"""

app = chat.Chatnificent(
    llm=GeminiDialogTTS(model="gemini-3.1-flash-tts-preview", stream=False),
    store=chat.store.File(base_dir="./artifact_examples/_convos_gemini_tts_advanced"),
    layout=chat.layout.Default(
        page_title="Build an AI Chatbot App That Generates Two-Speaker Dialog With Gemini in Python | Chatnificent",
        welcome_message=welcome_message,
    ),
)

if __name__ == "__main__":
    app.run()
