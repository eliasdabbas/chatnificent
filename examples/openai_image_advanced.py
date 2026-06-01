# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "chatnificent[openai]>=0.0.25",
# ]
# ///
"""
OpenAI image generation — advanced (streaming + true multi-turn editing)
========================================================================

The *advanced* counterpart to ``openai_image_simple.py``. Same Artifact
pattern, but routed through OpenAI's **Responses API** with the hosted
``image_generation`` tool and OpenAI's **server-side Conversation** state
— so follow-ups like "now make them blue" actually edit the *same*
image instead of generating an unrelated one.

The structural truth this example exists to teach
--------------------------------------------------
With the Artifact pattern, ``messages.json`` stores image URLs like
``<img src="/alice/c1/images/0.png">`` — short strings, never base64.
That's exactly what you want for storage and replay. **But the model
also only sees a string.** When the prior assistant message goes back
to the LLM, it's *text* — the model has no visual reference to the
image it produced.

There is no in-message workaround for this: round-tripping raw bytes
back to the model on every turn would blow the context window in ~5
turns. The bytes have to live *somewhere else* the model can refer to
by ID. Every major provider gives you a hook for this:

* **OpenAI Responses API** — ``client.conversations.create()`` mints a
  ``conv_*`` ID. Pass it as ``conversation=<id>`` on later calls and
  OpenAI carries the prior turns (text *and* images) server-side.
* **Gemini** — upload images via ``client.files.upload()`` and reference
  them by URI in subsequent ``generate_content`` calls.
* **Anthropic** — re-attach prior images as ``image_url`` content blocks.

This example uses the OpenAI flavor. The pattern is provider-specific
but the responsibility split is universal: **Artifact handles storage
and rendering; provider-side state handles continuity.**

How this maps to the canonical recipe
-------------------------------------
The simple recipe overrides ``extract_content``. The advanced one
overrides ``extract_stream_delta`` instead — because in streaming mode
the engine never calls ``extract_content``. It iterates the response and
asks the LLM "what is the next thing to render?" on every chunk. We
answer one of three things:

* ``str`` — a text delta (rendered as Markdown into the assistant message)
* ``Artifact`` — a finished file (engine persists + embeds in place)
* ``None`` — chunk has nothing to render (control/role/finish chunks)

That's the whole streaming contract. The engine handles the rest exactly
as in the simple flow: bytes → Store → ``<img src="…">`` in the persisted
message → served back on refresh.

What's added beyond the recipe
------------------------------
1. ``generate_response`` swaps ``chat.completions.create`` for
   ``responses.create``. On the first turn it mints a server-side
   conversation; on later turns it passes ``conversation=<conv_id>`` and
   sends only the new user messages (OpenAI prepends prior turns).
2. A tiny ``_OpenAIConvBridge`` engine subclass that reads OpenAI's
   ``conv_*`` id out of the just-saved raw response and stashes it on
   the assistant message dict so the next turn can find it.

Why a custom engine for this and not for the *simple* one?
----------------------------------------------------------
The provider's conversation ID is *its* state, not Chatnificent's, and
it has to ride along with the message that produced it. There's no LLM
seam to write it from (the streaming path bypasses
``create_assistant_message``), so a tiny engine bridge is the honest
home for it. Compare to the pre-Artifact ``openai_responses_image_studio.py``
which needed ~250 lines of custom Engine + Server + base64 plumbing —
this is the same idea pared to its essential ~10 lines.

Why ``tool_choice="required"``?
-------------------------------
Forces every turn to invoke the image tool, turning this into a pure
"prompt → image" product. Drop it if you want the model to choose
between answering in text or generating an image.

Prerequisites
-------------
::

    export OPENAI_API_KEY="sk-..."
    uv run --script artifact_examples/openai_image_advanced.py

Open http://127.0.0.1:7777 and try a real editing sequence:

* "a cat and a dog sitting together, watercolor style"
* "now make them blue"
* "add a small mouse between them"

Each turn edits the *same* scene. Bytes land under
``artifact_examples/_convos_openai_image_advanced/<user>/<convo>/images/``.
"""

import base64

import chatnificent as chat
from chatnificent.models import ASSISTANT_ROLE, Artifact

WELCOME_MESSAGE = """## Edit the same image, turn after turn

Start with a scene, then refine it — colors, subjects, mood. OpenAI's server-side conversation keeps the *same* image in context, so each follow-up edits the previous picture instead of generating a fresh one.

**Try the sequence below in order**: send the first chip, wait for the image, then send the next.

<div id="suggestions">
  <button class="suggestion" data-insert-prompt="A cat and a dog sitting together in a sunlit living room, watercolor style">
    <span class="suggestion-label">1. SCENE</span>
    <span class="suggestion-text">Establish the original picture.</span>
  </button>
  <button class="suggestion" data-insert-prompt="Now make them both blue.">
    <span class="suggestion-label">2. RECOLOR</span>
    <span class="suggestion-text">Same scene, different palette.</span>
  </button>
  <button class="suggestion" data-insert-prompt="Add a small mouse sitting between them.">
    <span class="suggestion-label">3. ADD</span>
    <span class="suggestion-text">Drop in a new character.</span>
  </button>
  <button class="suggestion" data-insert-prompt="Switch the style to vintage comic-book line art.">
    <span class="suggestion-label">4. RESTYLE</span>
    <span class="suggestion-text">Same scene, brand new look.</span>
  </button>
</div>"""


class OpenAIImageStudio(chat.llm.OpenAI):
    def generate_response(self, messages, **kwargs):
        merged = {**self.default_params, **kwargs}
        openai_conv_id = next(
            (
                msg.get("_openai_conversation_id")
                for msg in reversed(messages)
                if msg.get("_openai_conversation_id")
            ),
            None,
        )

        if openai_conv_id:
            # Continuation: OpenAI already has prior turns (text + images)
            # server-side. Send only what's new since the last assistant.
            return self.client.responses.create(
                model=self.model,
                input=self._messages_after_last_assistant(messages),
                conversation=openai_conv_id,
                **merged,
            )

        # First turn for this Chatnificent conversation — mint a server-side
        # OpenAI conversation so we can keep referring back to it.
        new_conv = self.client.conversations.create()
        return self.client.responses.create(
            model=self.model,
            input=list(messages),
            conversation=new_conv.id,
            **merged,
        )

    def extract_stream_delta(self, chunk):
        if chunk.type == "response.output_text.delta":
            return chunk.delta
        if chunk.type == "response.output_item.done":
            item = chunk.item
            if getattr(item, "type", None) == "image_generation_call" and item.result:
                return Artifact(
                    data=base64.b64decode(item.result),
                    ext=".png",
                    folder="images",
                )
        return None

    def _messages_after_last_assistant(self, messages):
        out = []
        for msg in reversed(messages):
            if msg.get("role") == ASSISTANT_ROLE:
                break
            out.append(msg)
        out.reverse()
        return out


class _OpenAIConvBridge(chat.engine.Orchestrator):
    """Stash OpenAI's ``conv_*`` ID on the assistant message so the next
    turn can pass ``conversation=<id>`` for true image continuity."""

    def _save_conversation(self, conversation, user_id):
        last = conversation.messages[-1] if conversation.messages else None
        if last and last.get("role") == ASSISTANT_ROLE:
            raws = self.app.store.load_raw_api_responses(user_id, conversation.id) or []
            if raws:
                cid = self._extract_openai_conv_id(raws[-1])
                if cid:
                    last["_openai_conversation_id"] = cid
        super()._save_conversation(conversation, user_id)

    def _extract_openai_conv_id(self, latest_response):
        chunks = (
            latest_response if isinstance(latest_response, list) else [latest_response]
        )
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
    llm=OpenAIImageStudio(
        tools=[{"type": "image_generation"}],
        tool_choice="required",
    ),
    engine=_OpenAIConvBridge(),
    store=chat.store.File(base_dir="./artifact_examples/_convos_openai_image_advanced"),
    layout=chat.layout.Default(welcome_message=WELCOME_MESSAGE),
)

if __name__ == "__main__":
    app.run()
