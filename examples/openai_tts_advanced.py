# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "chatnificent[openai]>=0.0.25",
# ]
# ///
"""
OpenAI text-to-speech — advanced (streamed chat + spoken reply)
===============================================================

The *advanced* counterpart to ``openai_tts_simple.py``. Instead of
turning every user message into a single audio clip, this example
streams a normal chat reply **and then appends an audio player that
reads it aloud** — in the same assistant message, in a single turn.

The structural truth this example exists to teach
--------------------------------------------------
``extract_stream_delta`` is allowed to return ``str``, ``Artifact``, or
``None`` — that's the whole streaming contract (see
``artifact_canonical_recipe.py`` and ``openai_image_advanced.py``).
So a single assistant turn can interleave Markdown text deltas and
finished binary files in whatever order the model produces them.

Here we want both, but they come from *two different OpenAI endpoints*:

* ``chat.completions.create(stream=True)`` — text deltas
* ``audio.speech.create(...)`` — one finished MP3

The trick: ``generate_response`` is allowed to be a **generator** that
yields chunks of any shape it likes. We yield the real OpenAI chat
chunks first (so the text streams immediately), accumulate the text as
it flows by, then — after the chat completion finishes — call the TTS
endpoint once and yield the resulting bytes as a sentinel value the
engine will pass back to ``extract_stream_delta``. We answer that one
sentinel with an ``Artifact``; the engine persists it and embeds an
``<audio controls>`` player at the bottom of the same message.

How this maps to the canonical recipe
-------------------------------------
Same four-override skeleton. Two of them barely change:

* ``generate_response`` is now a generator that chains two endpoints.
* ``extract_stream_delta`` returns the ``Artifact`` for our sentinel
  chunk and otherwise delegates to ``super()`` for normal text deltas.
* ``parse_tool_calls`` stays ``None`` (no agentic loop).
* ``extract_content`` is irrelevant — the streaming path never calls it.

What's added beyond the recipe
------------------------------
A tiny generator (``_chat_then_tts``) that:

1. Iterates the streamed chat chunks, forwarding each one and
   accumulating the text content as it passes.
2. After the chat stream ends, calls ``audio.speech.create`` once
   with the full accumulated text.
3. Yields the audio bytes as a sentinel — a plain
   ``("audio", bytes)`` tuple is enough; ``extract_stream_delta``
   recognises the tuple and returns the ``Artifact``.

Why a sentinel tuple instead of yielding the ``Artifact`` directly?
-------------------------------------------------------------------
Either works — the engine just passes whatever ``generate_response``
yields to ``extract_stream_delta``. We use a tuple here to keep
``generate_response`` honest about its job (produce raw provider
chunks) and ``extract_stream_delta`` honest about its job (decide
what each chunk *means*). Yielding ``Artifact`` directly would also
work and is fine for one-off code.

Why not just run the TTS on the saved text after the turn?
----------------------------------------------------------
You could — that's a perfectly good alternative, implemented as an
``_after_save`` engine hook. We chose the in-stream approach because it
keeps everything in one pillar (the LLM) and means the audio appears
in the *first* render of the message, not after a refresh.

Prerequisites
-------------
::

    export OPENAI_API_KEY="sk-..."
    uv run --script artifact_examples/openai_tts_advanced.py

Open http://127.0.0.1:7777 and ask the model anything — for example:

* "Explain quicksort in two sentences."
* "Tell me a 3-line bedtime story about a curious raccoon."
* "Give me a one-paragraph fun fact about octopuses."

You'll see the text stream in token-by-token, then an audio player
appear underneath that reads the same reply aloud. MP3 files land
under ``artifact_examples/_convos_openai_tts_advanced/<user>/<convo>/audio/``.
"""

import chatnificent as chat
from chatnificent.models import Artifact


class OpenAIChatPlusTTS(chat.llm.OpenAI):
    def generate_response(self, messages, **kwargs):
        chat_stream = super().generate_response(messages, **kwargs)
        return self._chat_then_tts(chat_stream)

    def _chat_then_tts(self, chat_stream):
        spoken = []
        for chunk in chat_stream:
            try:
                delta = chunk.choices[0].delta.content
            except (AttributeError, IndexError):
                delta = None
            if delta:
                spoken.append(delta)
            yield chunk

        text = "".join(spoken).strip()
        if not text:
            return

        audio_response = self.client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text,
            response_format="mp3",
        )
        yield ("audio", audio_response.read())

    def extract_stream_delta(self, chunk):
        if isinstance(chunk, tuple) and len(chunk) == 2 and chunk[0] == "audio":
            return Artifact(data=chunk[1], ext=".mp3", folder="audio")
        return super().extract_stream_delta(chunk)


app = chat.Chatnificent(
    llm=OpenAIChatPlusTTS(),
    store=chat.store.File(base_dir="./artifact_examples/_convos_openai_tts_advanced"),
)

if __name__ == "__main__":
    app.run()
