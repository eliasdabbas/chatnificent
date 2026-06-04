# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "chatnificent[openai]>=0.0.25",
# ]
# ///
"""
Image Studio — multi-turn image editing with live streaming previews
====================================================================

A conversational image studio built on OpenAI's Responses API. Type a
prompt, watch the image materialise in real time, then keep refining it
with follow-up messages. Each reply edits the same running image — no
re-uploads, no context overflow, no staring at a blank screen.

What's distinctive
------------------
**1. Progressive streaming.** The image appears in the chat *before*
   generation is complete. Three partial previews stream in as the model
   works, sharpening into the final result.

**2. Multi-turn editing.** Every follow-up edits the current image
   rather than starting from scratch. Powered by OpenAI's server-side
   ``Conversation`` object — the model remembers prior images without
   the app re-uploading them.

**3. Revised prompt visibility.** The model rewrites your prompt for
   better results. The revised version is shown below each image.

How it differs from ``file_serving_simple.py``
----------------------------------------------
``file_serving_simple.py`` returns an ``Artifact`` and the framework
handles persistence + URL minting + ``<img>`` embed. Clean and
recommended for terminal endpoints.

This example can't use ``Artifact`` for the *streaming* phase because
OpenAI emits **partial images** as separate SSE events (base64 chunks
for a progressively sharper preview). ``Artifact`` is a
final-bytes-at-once dataclass — it doesn't model "three previews then a
final". So during streaming the base64 stays inline (the user sees the
image sharpen in place), and on save we hand the *final* bytes to the
engine wrapped in an ``Artifact`` — storage, URL minting and ``<img>``
embed are all handled by the framework.

Running
-------
::

    export OPENAI_API_KEY="sk-..."
    uv run --script examples/openai_responses_image_studio.py

Then try an iterative flow::

    "a neon-lit ramen bar in Tokyo at midnight"

    "add a cat sleeping on the counter"
    "make it rain outside the window"
    "turn the whole scene into a watercolour painting"
"""

import base64
import re

import chatnificent as chat

WELCOME = """## Image Studio

Type a prompt. Watch three progressive previews stream in as the model paints. Then **keep going** — each follow-up edits the *same* image, no re-uploads, no copy-paste of the previous prompt.

State lives on OpenAI's side via the Responses API's server-side `Conversation` object. The framework stashes the conversation ID on the assistant message and sends only the new user turn next time — providers handle the canvas, Chatnificent handles the chat.

<div id="suggestions">
  <button class="suggestion" data-insert-prompt="A cozy bookshop on a rainy Sunday afternoon — warm lamplight, a sleeping cat on a leather armchair, steam rising from a forgotten teacup. Photorealistic, shallow depth of field.">
    <span class="suggestion-label">SCENE</span>
    <span class="suggestion-text">A cozy bookshop on a rainy Sunday — then iterate</span>
  </button>
  <button class="suggestion" data-insert-prompt="An astronaut floating in a sea of bioluminescent jellyfish, deep underwater, cinematic lighting, hyper-detailed, surreal.">
    <span class="suggestion-label">SURREAL</span>
    <span class="suggestion-text">Astronaut among bioluminescent jellyfish</span>
  </button>
  <button class="suggestion" data-insert-prompt="A vintage travel poster for Mars in the Art Deco style of the 1930s — geometric mountains, a colonized dome city, bold typography reading 'VISIT MARS — THE RED FRONTIER'.">
    <span class="suggestion-label">POSTER</span>
    <span class="suggestion-text">Art Deco travel poster for Mars</span>
  </button>
  <button class="suggestion" data-insert-prompt="An isometric pixel-art coffee shop, 32-bit retro game style, three floors, plants on every windowsill, soft pastel palette.">
    <span class="suggestion-label">PIXEL</span>
    <span class="suggestion-text">Isometric pixel-art coffee shop</span>
  </button>
</div>

*After the first image lands, try:* `"make it nighttime"` · `"add a fox"` · `"render it as a watercolour"` · `"turn the whole scene black and white"`"""


class OpenAIResponses(chat.llm.OpenAI):
    """OpenAI Responses API with server-side Conversation state."""

    def generate_response(self, messages, **kwargs):
        api_kwargs = self.build_request_payload(messages, **kwargs)
        if "conversation" not in api_kwargs:
            api_kwargs["conversation"] = self.client.conversations.create().id
        return self.client.responses.create(**api_kwargs)

    def build_request_payload(self, messages, model=None, tools=None, **kwargs):
        api_kwargs = {**self.default_params, **kwargs}
        api_kwargs["model"] = model or self.model
        if tools:
            api_kwargs["tools"] = tools
        api_kwargs.pop("messages", None)

        # System instruction biases the model toward preserving the prior image
        # and applying only the requested change — vital for iterative editing.
        api_kwargs["instructions"] = (
            "You are an image editor. When the user requests a change after an "
            "image already exists, edit the *previous* image and apply ONLY "
            "the requested modification. Preserve composition, subjects, "
            "colors, and details that the user did not ask to change. Keep "
            "your revised prompt minimal and surgical."
        )

        convo_id = self._find_conversation_id(messages)
        if convo_id:
            api_kwargs["conversation"] = convo_id
            api_kwargs["input"] = self._new_inputs(messages)
        else:
            api_kwargs["input"] = list(messages)
        return api_kwargs

    def extract_stream_delta(self, chunk):
        if chunk.type == "response.output_text.delta":
            return chunk.delta
        if chunk.type == "response.image_generation_call.partial_image":
            # data-gen-partial marks placeholders for the engine to strip on save.
            return (
                f'\n<img data-gen-partial="1" '
                f'src="data:image/jpeg;base64,{chunk.partial_image_b64}" '
                f'style="max-width:512px;height:auto" alt="Generating...">\n'
            )
        if chunk.type == "response.output_item.done":
            return self._format_final_image(chunk.item)
        return None

    def _format_final_image(self, item):
        if getattr(item, "type", None) != "image_generation_call":
            return None
        result = getattr(item, "result", None)
        if not result:
            return None
        # Returning an Artifact lets the engine handle storage + URL + <img>
        # embed automatically. The revised prompt goes in the HTML caption.
        revised = getattr(item, "revised_prompt", None)
        caption = f'<em>Revised prompt: "{revised}"</em><br>' if revised else ""
        return chat.models.Artifact(
            data=base64.b64decode(result),
            ext=".jpeg",
            folder="images",
            html=f'{caption}<img src="{{url}}" style="max-width:512px;height:auto" alt="Generated image">',
        )

    def _find_conversation_id(self, messages):
        for msg in reversed(messages):
            cid = msg.get("_openai_conversation_id")
            if cid:
                return cid
        return None

    def _new_inputs(self, messages):
        new = []
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                break
            new.append(msg)
        return list(reversed(new))


class ImageStudioEngine(chat.engine.Orchestrator):
    """Re-types partial-image deltas as `status` events and persists finals."""

    def handle_message_stream(self, user_input, user_id, convo_id_from_url):
        # Partial-image deltas must not accumulate in the message bubble — they'd
        # pile up above the final image. Re-typing them as `status` events makes
        # the UI replace the preview area on each tick and strip it on `done`,
        # so the image visibly sharpens in place.
        for event in super().handle_message_stream(
            user_input, user_id, convo_id_from_url
        ):
            # String-based contract with our LLM adapter: any delta carrying
            # the partial-image marker is re-typed as a status event so the UI
            # replaces the preview in place instead of stacking it.
            if event.get(
                "event"
            ) == "delta" and '<img data-gen-partial="1"' in event.get("data", ""):
                yield {"event": "status", "data": event["data"]}
            else:
                yield event

    def _save_conversation(self, conversation, user_id):
        last = conversation.messages[-1] if conversation.messages else None
        if (
            last
            and last.get("role") == "assistant"
            and isinstance(last.get("content"), str)
        ):
            # Strip the partial-preview placeholders that were re-typed as
            # status events for live display — they shouldn't be saved.
            last["content"] = re.sub(
                r'\n?<img data-gen-partial="1"[^>]*>\n?', "", last["content"]
            )

        # Stash OpenAI's server-side conversation ID on the assistant message so
        # the next turn can pick up where this one left off.
        raw_responses = self.app.store.load_raw_api_responses(user_id, conversation.id)
        if raw_responses and last and last.get("role") == "assistant":
            convo_id = self._extract_conversation_id(raw_responses[-1])
            if convo_id:
                last["_openai_conversation_id"] = convo_id

        super()._save_conversation(conversation, user_id)

    def _extract_conversation_id(self, latest):
        chunks = latest if isinstance(latest, list) else [latest]
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            if chunk.get("type") == "response.completed":
                conv = (chunk.get("response") or {}).get("conversation")
                if isinstance(conv, dict) and conv.get("id"):
                    return conv["id"]
            conv = chunk.get("conversation")
            if isinstance(conv, dict) and conv.get("id"):
                return conv["id"]
        return None


app = chat.Chatnificent(
    llm=OpenAIResponses(
        model="gpt-5.4",
        tools=[
            {
                "type": "image_generation",
                "model": "gpt-image-1",
                "partial_images": 3,
                "quality": "medium",
                "output_format": "jpeg",
                # "high" tells gpt-image-1 to preserve the input image's
                # pixels much more faithfully when editing — critical for the
                # "make it black and white" / "add a cat" iterative flow.
                # (gpt-image-2 dropped this param; pin to gpt-image-1 to use it.)
                "input_fidelity": "high",
            }
        ],
    ),
    store=chat.store.File(base_dir="image_studio"),
    engine=ImageStudioEngine(),
    layout=chat.layout.Default(
        page_title="Build an AI Chatbot App That Edits Images Multi-Turn With OpenAI in Python | Chatnificent",
        welcome_message=WELCOME,
    ),
)


if __name__ == "__main__":
    app.run()
