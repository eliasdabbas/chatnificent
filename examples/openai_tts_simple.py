# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "chatnificent[openai]>=0.0.25",
# ]
# ///
"""
OpenAI text-to-speech — simple
==============================

The canonical recipe (see ``artifact_canonical_recipe.py``) wired to a
second OpenAI endpoint: every user message becomes an audio clip via
``audio.speech.create``.

What's identical to the recipe
------------------------------
* Subclass the nearest LLM provider (``chat.llm.OpenAI``).
* Override ``generate_response`` to call the SDK endpoint that returns bytes.
* Override ``extract_content`` to return ``Artifact(...)``.
* Override ``parse_tool_calls`` → ``None`` to disable the agentic loop.

What changes vs. ``openai_image_simple.py``
-------------------------------------------
**Even simpler extraction.** The TTS endpoint returns raw audio bytes
directly via ``response.read()`` — no base64 round-trip. We hand those
bytes to ``Artifact(ext=".mp3", folder="audio")`` and the framework
writes ``audio/0.mp3``, ``1.mp3``, … and embeds an
``<audio src=".../audio/0.mp3" controls></audio>`` player in the message
body (the engine's ``ARTIFACT_WRAPPERS`` map keys off the MIME family,
so audio bytes get an ``<audio>`` element for free).

That's the whole adaptation. Different endpoint, different folder,
different file extension — same four overrides, same Artifact contract.

Prerequisites
-------------
::

    export OPENAI_API_KEY="sk-..."
    uv run --script artifact_examples/openai_tts_simple.py

Open http://127.0.0.1:7777 and type any prompt — for example:

* "Welcome to Chatnificent. Let's build something delightful."
* "Once upon a time, in a small village by the sea..."
* "Reading you the morning headlines in the voice of a friendly robot."

Each prompt produces one MP3, persisted under
``artifact_examples/_convos_openai_tts_simple/<user>/<convo>/audio/``.
"""

import chatnificent as chat
from chatnificent.models import USER_ROLE, Artifact


class OpenAITTS(chat.llm.OpenAI):
    def generate_response(self, messages, **kwargs):
        text = next(
            (
                m.get("content") or ""
                for m in reversed(messages)
                if m.get("role") == USER_ROLE
            ),
            "",
        )
        return self.client.audio.speech.create(
            model=self.model,
            voice="alloy",
            input=text,
            response_format="mp3",
        )

    def extract_content(self, response):
        return Artifact(data=response.read(), ext=".mp3", folder="audio")

    def parse_tool_calls(self, response):
        return None


app = chat.Chatnificent(
    llm=OpenAITTS(model="tts-1", stream=False),
    store=chat.store.File(base_dir="./artifact_examples/_convos_openai_tts_simple"),
    layout=chat.layout.Default(
        page_title="Build an AI Chatbot App That Speaks With OpenAI TTS in Python | Chatnificent",
    ),
)

if __name__ == "__main__":
    app.run()
