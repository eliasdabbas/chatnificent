# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "chatnificent[gemini]>=0.0.25",
#     "pillow>=10",
# ]
# ///
"""
Gemini music generation — advanced (text + images)
==================================================

Same recipe as ``gemini_music_simple.py``. **The only difference is in
``generate_response``**: if the user's prompt contains image URLs,
fetch them and pass them alongside the text. Lyria 3 reads the mood
and colors of the image and composes accordingly (the docs allow up
to 10 images per prompt).

The whole "advanced" payload is six lines:

* one regex for image URLs,
* one comprehension that fetches each URL through ``urllib`` and opens
  it with ``PIL.Image``,
* one ``contents=[prompt, *images]`` instead of ``contents=prompt``.

That's it. ``extract_content`` and ``parse_tool_calls`` are byte-for-byte
identical to the simple version, because the **response shape doesn't
change** when you add image inputs — Lyria still returns the same
lyrics + audio parts.

Why URLs and not uploads
------------------------
File uploads aren't wired into the framework yet, but URLs cost
nothing: the suggestion chips already pre-fill the composer with
``data-insert-prompt``, so we drop a publicly-hosted image URL right
into the prompt text. The chip renders a thumbnail so the user sees
the image they're about to send; if the host 404s, the broken
thumbnail tells them immediately. When uploads land in the framework,
this same example will work — only the chips will change.

Prerequisites
-------------
::

    export GEMINI_API_KEY="..."  # or GOOGLE_API_KEY
    uv run --script artifact_examples/gemini_music_advanced.py

Click any chip to send a prompt + image. Or type your own prompt and
paste any ``https://…/photo.jpg`` URL into it.
"""

import re
from io import BytesIO
from urllib.request import urlopen

import chatnificent as chat
from chatnificent.models import USER_ROLE, Artifact
from PIL import Image

# Per-Artifact embed: Store-served URL drops into <audio controls> so
# the clip plays inline. Without this Default would render a download
# link instead.
AUDIO_HTML = '<audio controls src="{url}"></audio>'

# Inline thumbnail for the image we sent to Lyria — echoed back in the
# assistant turn so the user sees what the model was looking at. Full
# width by default: the image is the mood, and an immersive visual
# pairs naturally with the music.
INSPIRATION_IMG = '<img src="{url}" alt="" style="width:100%;height:auto;border-radius:12px;display:block;margin:8px 0;">'

# Match URLs that look like images: either ending in a recognised
# extension (`.jpg`, `.png`, …) **or** carrying an `fm=<ext>` query
# parameter — the latter is how Unsplash, Cloudinary, and friends
# signal image format without a file extension.
IMAGE_URL_RE = re.compile(
    r"https?://\S+?(?:\.(?:png|jpg|jpeg|webp|gif)|[?&]fm=(?:png|jpg|jpeg|webp|gif))\S*",
    re.I,
)


# Google's own Lyria docs sample (desert_sunset.jpg) plus three
# Unsplash photos (free, hot-linkable). Each chip pairs a thumbnail
# with a prompt that explicitly says "inspired by this image: <url>"
# so the regex picks it up and Lyria knows what to do with it.
welcome_message = """## Lyria 3 — image + text to music

Same model as the simple version, with one tweak: Lyria 3 accepts
images alongside your prompt and composes music inspired by their
mood and colors. Click a chip below to send the image + prompt, or
paste any public image URL into your own prompt.

<style>
#suggestions .suggestion img { width: 64px; height: 64px; object-fit: cover; border-radius: 6px; float: left; margin-right: 10px; }
</style>

<div id="suggestions">
  <button class="suggestion" data-insert-prompt="A warm folk ballad with gentle female vocals about a long journey home at the end of the day, inspired by the mood and colors in this image: https://storage.googleapis.com/generativeai-downloads/images/desert_sunset.jpg . Acoustic guitar, brushed drums, slow tempo.">
    <img src="https://storage.googleapis.com/generativeai-downloads/images/desert_sunset.jpg" alt="">
    <span class="suggestion-label">DESERT SUNSET</span>
    <span class="suggestion-text">Folk ballad with vocals about coming home.</span>
  </button>
  <button class="suggestion" data-insert-prompt="A cosmic synth-wave instrumental at 80 BPM inspired by this image: https://images.unsplash.com/photo-1462331940025-496dfbfc7564?w=600&fm=jpg . Slow analog pads, shimmering arpeggios, no vocals.">
    <img src="https://images.unsplash.com/photo-1462331940025-496dfbfc7564?w=600&fm=jpg" alt="">
    <span class="suggestion-label">GALAXY</span>
    <span class="suggestion-text">Cosmic synth-wave, instrumental.</span>
  </button>
  <button class="suggestion" data-insert-prompt="A dark cinematic orchestral instrumental inspired by this image: https://images.unsplash.com/photo-1605727216801-e27ce1d0cc28?w=600&fm=jpg . Deep strings, slow crescendo, timpani, no vocals.">
    <img src="https://images.unsplash.com/photo-1605727216801-e27ce1d0cc28?w=600&fm=jpg" alt="">
    <span class="suggestion-label">STORM</span>
    <span class="suggestion-text">Cinematic strings, instrumental.</span>
  </button>
  <button class="suggestion" data-insert-prompt="A bright uplifting acoustic folk song with warm male vocals about a sunny morning in the countryside, inspired by the mood and light in this image: https://images.unsplash.com/photo-1500382017468-9049fed747ef?w=600&fm=jpg . Guitar, harmonica, gentle backing harmonies.">
    <img src="https://images.unsplash.com/photo-1500382017468-9049fed747ef?w=600&fm=jpg" alt="">
    <span class="suggestion-label">COUNTRYSIDE</span>
    <span class="suggestion-text">Acoustic folk with vocals about a sunny field.</span>
  </button>
</div>
"""


class GeminiMusicWithImages(chat.llm.Gemini):
    def generate_response(self, messages, **kwargs):
        prompt = next(
            (
                m.get("content") or ""
                for m in reversed(messages)
                if m.get("role") == USER_ROLE
            ),
            "",
        )
        # The whole "advanced" payload: fetch each image URL the user
        # mentioned and hand the PIL.Image objects to Lyria alongside
        # the text. Lyria accepts up to 10 images per prompt.
        image_urls = IMAGE_URL_RE.findall(prompt)
        images = [Image.open(BytesIO(urlopen(url).read())) for url in image_urls]
        response = self.client.models.generate_content(
            model=self.model,
            contents=[prompt, *images] if images else prompt,
        )
        # Stash the URLs on the response object (request-local, safe)
        # so extract_content can echo them back in the assistant turn.
        response._chatnificent_image_urls = image_urls
        return response

    def extract_content(self, response):
        # Identical to the simple version, plus one extra step: echo
        # the inspiration images as <img> tags so the user sees what
        # the model was looking at. Audio + image + text in one turn.
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
        for url in getattr(response, "_chatnificent_image_urls", []):
            items.append(INSPIRATION_IMG.format(url=url))
        if lyrics:
            items.append("\n\n".join(lyrics))
        return items

    def parse_tool_calls(self, response):
        return None


app = chat.Chatnificent(
    llm=GeminiMusicWithImages(model="lyria-3-clip-preview", stream=False),
    store=chat.store.File(base_dir="./artifact_examples/_convos_gemini_music_advanced"),
    layout=chat.layout.Default(
        page_title="Build an AI Chatbot App That Generates Music From Image Prompts in Python | Chatnificent",
        welcome_message=welcome_message,
    ),
)

if __name__ == "__main__":
    app.run()
