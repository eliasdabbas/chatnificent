# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "chatnificent[gemini]>=0.0.25",
# ]
# ///
"""
Gemini video generation with Veo 3.1 Fast
=========================================

Same Artifact recipe, this time producing **video** via Google's Veo 3.1
Fast model. Unlike text/image/audio, Veo is **asynchronous** — the API
returns a long-running ``operation`` that must be polled until ``done``,
then the bytes are fetched separately.

This example keeps cost and latency as low as Veo allows:

* ``veo-3.1-fast-generate-preview`` — the cheapest, fastest tier.
* ``duration_seconds=4`` — the shortest clip Veo will produce.
* ``resolution="720p"`` — the default (cheaper than 1080p / 4k).

Even with those knobs, expect **15-60 seconds of wall time per clip**
(the API's published min latency is 11s, max 6 min). The browser request
will sit on the polling loop the whole time; the framework's request
timeout is generous, but be patient.

What's identical to the canonical recipe
----------------------------------------
* Subclass the nearest LLM provider (``chat.llm.Gemini``).
* Override ``generate_response`` — call the SDK, return raw bytes.
* Override ``extract_content`` — return ``Artifact(...)``.
* Override ``parse_tool_calls`` → ``None`` to disable the tool loop.

What's different from ``gemini_image_simple.py``
------------------------------------------------
**The async polling loop lives entirely inside ``generate_response``.**
The engine and the rest of the framework know nothing about it — from
their perspective this is a slow synchronous call that eventually
returns bytes. That's the whole point of keeping orchestration
synchronous: any blocking I/O pattern (polling, retries, multi-step
pipelines) fits inside one pillar method.

Zero custom wrapper needed: ``ext=".mp4"`` resolves to MIME ``video/mp4``
and the engine's default ``video/`` wrapper emits
``<video src="..." controls></video>``.

Prerequisites
-------------
::

    export GEMINI_API_KEY="..."  # or GOOGLE_API_KEY
    uv run --script artifact_examples/gemini_video_simple.py

Open http://127.0.0.1:7777 and try a suggestion chip — or anything cinematic.
Each prompt produces one ~4-second MP4, persisted under
``artifact_examples/_convos_gemini_video_simple/<user>/<convo>/videos/``.
"""

import sys
import tempfile
import time
from pathlib import Path

import chatnificent as chat
from chatnificent.models import USER_ROLE, Artifact

VIDEO_MODEL = "veo-3.1-fast-generate-preview"
POLL_INTERVAL_SECONDS = 5
POLL_TIMEOUT_SECONDS = 6 * 60  # Veo's published max latency.


class GeminiVeo(chat.llm.Gemini):
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

        # Kick off the long-running job. Veo returns an Operation handle
        # immediately; the actual video isn't ready for tens of seconds.
        operation = self.client.models.generate_videos(
            model=VIDEO_MODEL,
            prompt=prompt,
            config=types.GenerateVideosConfig(
                # Lowest-cost knobs: shortest duration, default resolution.
                duration_seconds=4,
                resolution="720p",
                aspect_ratio="16:9",
                number_of_videos=1,
            ),
        )

        # Polling loop. Synchronous on purpose — this is the engine
        # thread's problem to wait, not our framework's to orchestrate.
        # Concurrent users land in separate threads (see AGENTS.md
        # "Concurrency Awareness"), so this blocking call doesn't stall
        # the server.
        deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
        while not operation.done:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Veo generation exceeded {POLL_TIMEOUT_SECONDS}s "
                    "without completing."
                )
            time.sleep(POLL_INTERVAL_SECONDS)
            operation = self.client.operations.get(operation)

        return operation

    def extract_content(self, response):
        operation = response
        if not operation.response or not operation.response.generated_videos:
            return f"No video returned (operation state: {operation!r})."

        video = operation.response.generated_videos[0].video

        # The genai SDK exposes a `.save(path)` helper that fetches the
        # bytes from the temporary server URL and writes them to disk.
        # We round-trip through a tempfile so we hand bytes — not a path
        # or URL — to the Artifact, keeping the contract identical to
        # every other example.
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            self.client.files.download(file=video)
            video.save(str(tmp_path))
            data = tmp_path.read_bytes()
        finally:
            tmp_path.unlink(missing_ok=True)

        return Artifact(data=data, ext=".mp4", folder="videos")

    def parse_tool_calls(self, response):
        return None


welcome_message = """## Text → video (with audio) via Veo 3.1 Fast

Each prompt produces one ~4-second 720p MP4. Generation is
**asynchronous** under the hood — expect 15-60 seconds of wall time
while the server polls the Veo job. Be patient; the page is waiting.

Veo generates audio natively, but **only when the prompt asks for it**.
Cue dialogue in quotes, describe sound effects explicitly, or mention
ambient noise — otherwise the clip comes back silent. Every suggestion
below includes an audio cue.

<div id="suggestions">
  <button class="suggestion" data-insert-prompt="A close-up of melting icicles on a frozen rock wall with cool blue tones, zoomed in, water drips in slow motion. Sound: a slow, rhythmic drip-drip of water, a faint icy crackle, distant wind.">
    <span class="suggestion-label">ICICLES</span>
    <span class="suggestion-text">Close-up with dripping SFX.</span>
  </button>
  <button class="suggestion" data-insert-prompt="A tiny astronaut riding a capybara through a neon forest at night, low-angle tracking shot, synthwave style. The astronaut shouts gleefully, &quot;Faster, buddy, faster!&quot; Pulsing synthwave music and soft hoofbeats in the background.">
    <span class="suggestion-label">ASTRONAUT</span>
    <span class="suggestion-text">Dialogue + synthwave score.</span>
  </button>
  <button class="suggestion" data-insert-prompt="Aerial drone shot pulling back from a calico kitten napping in a sunlit meadow, warm afternoon light, gentle breeze in the grass. Ambient: soft chirping birds, a faint breeze rustling the grass, a single contented purr.">
    <span class="suggestion-label">KITTEN</span>
    <span class="suggestion-text">Ambient nature soundscape.</span>
  </button>
  <button class="suggestion" data-insert-prompt="Film noir style: a detective lights a cigarette in a rainy alley at night, black and white, neon reflections in puddles. He mutters to himself, &quot;She was trouble from the start.&quot; Heavy rain, a distant saxophone, the flick of a lighter.">
    <span class="suggestion-label">NOIR</span>
    <span class="suggestion-text">Monologue + rain + sax.</span>
  </button>
</div>
"""

app = chat.Chatnificent(
    llm=GeminiVeo(model=VIDEO_MODEL, stream=False),
    store=chat.store.File(base_dir="./artifact_examples/_convos_gemini_video_simple"),
    layout=chat.layout.Default(welcome_message=welcome_message),
)

if __name__ == "__main__":
    app.run()
