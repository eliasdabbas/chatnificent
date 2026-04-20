# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "chatnificent[openai]",
# ]
# ///
"""
Image Studio — Professional Multi-Turn Image Generation
========================================================

The "professional" companion to ``openai_responses_image_generator.py``.
That example is the minimum demo: every turn generates an image and
inlines a base64 data URL into ``messages.json``. It works for one
shot but breaks on follow-up edits — the prior turn's ~1.9 MB data
URL is replayed as input and blows the model's context window.

This example fixes the three things a real product needs:

1. **Server-side conversation state.** Uses OpenAI's native
   ``Conversation`` object so the model sees prior images without us
   re-uploading them each turn. The framework sends only the *new*
   user message; OpenAI prepends prior conversation items.
2. **Durable, organized image artifacts.** Each generated image is
   decoded and saved as ``images/<n>.png`` under the conversation
   directory via the Store pillar. ``messages.json`` keeps short URL
   references (``/files/<user>/<convo>/images/<n>.png``), not
   megabytes of base64.
3. **A ``/files/...`` HTTP route.** A small ``DevServer`` subclass
   adds a GET handler that streams sidecar bytes through the Store,
   scoped to the requesting user. The default UI's ``<img>`` tags
   resolve via this route.

Three small subclasses do all of it (LLM + Engine + Server). The
``File`` store is used as-is — nested filenames like ``images/0.png``
are supported out of the box since the path-canonicalization fix.
The original ``openai_responses_image_generator.py`` stays unchanged
as the "minimally complete" pitch.

Architecture
------------
- ``messages.json`` view: text + short ``/files/...`` URLs (small).
- OpenAI server view: full conversation including image items
  (held by ``conversation=conv_*``).
- The two never need to be reconciled — the model never sees our
  short URLs, and we never re-send the bytes.

Flow per turn
-------------
1. **LLM** (``OpenAIResponses``):
   - If a prior assistant message carries an
     ``_openai_conversation_id`` marker, send only the new user input
     and ``conversation=<id>`` to ``responses.create``. The server
     prepends prior items.
   - First turn: ``client.conversations.create()`` to mint a fresh
     ``conv_*`` ID, then send full input plus ``conversation=<id>``.
2. **Engine** (``ImageStudioEngine``) overrides ``_save_conversation``
   (the seam that runs *after* ``raw_api_responses.jsonl`` is written
   but *before* the canonical save). This means everything happens in
   a single conversation-save:
   - Read the just-written raw response chunks from the Store.
   - Extract ``response.conversation.id`` and any
     ``image_generation_call.result`` base64 bytes.
   - Save bytes as ``images/<n>.png`` via Store.
   - Rewrite the latest assistant message: replace inline data URLs
     with ``/files/...`` URLs, attach the conversation marker.
   - Hand off to the parent's save.
3. **Server** (``FileServingDevServer``) adds a ``/files/...`` route
   that loads bytes via Store and serves them with ``image/*``
   content type, scoped to the requesting user.

Running
-------
::

    export OPENAI_API_KEY="sk-..."
    uv run examples/openai_responses_image_studio.py

Open http://127.0.0.1:7777 and try the iterative flow:

- "give me an image of a camel in the desert"
- "now make the sky yellow"
- "now have it raining"

Each follow-up edits the running image. ``messages.json`` stays
small across turns; ``image_studio/<user>/<convo>/images/``
accumulates real PNG files; the ``/files/...`` URLs in the chat
resolve to those files.

Why ``_save_conversation`` and not ``_after_save``?
---------------------------------------------------
``_after_save`` runs after the canonical save is already on disk —
mutating then would force a *second* write. ``_save_conversation``
sits between raw-exchange persistence and the canonical save, so we
mutate in place and the parent does the single write. Lower I/O,
fewer atomicity edge cases.

Friction Log (feeds the transparent-endpoint spec)
--------------------------------------------------
1. **LLM has no Store access.** Conversation IDs and image bytes
   both need persistence keyed by ``user_id`` / ``convo_id``, which
   the LLM doesn't know. We push that work to the engine subclass
   and pass the conversation ID back through the message dicts (the
   ``_openai_conversation_id`` marker pattern).
2. ~~Store enforces flat filenames.~~ *Resolved:* ``File`` now
   canonicalizes nested paths (``images/0.png``) and enforces
   containment via ``Path.resolve()`` rather than rejecting all
   ``/``. No Store subclass needed.
3. **No file-serving primitive in DevServer.** Sidecar files exist
   on disk but aren't reachable over HTTP without a custom handler.
4. **Conversation parameter binds the conversation to OpenAI.** This
   is Tension 2 from the strategic review made concrete: switching
   providers mid-conversation now requires migrating server-side
   state.
"""

import base64
from functools import partial
from http import HTTPStatus
from http.server import HTTPServer

import chatnificent as chat
from chatnificent.server import _DevHandler

# =============================================================================
# LLM — OpenAI Responses with server-side Conversation state
# =============================================================================


class OpenAIResponses(chat.llm.OpenAI):
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
        if chunk.type == "response.output_item.done":
            return self._format_image(chunk.item)
        return None

    def _format_image(self, item):
        if getattr(item, "type", None) != "image_generation_call":
            return None
        result = getattr(item, "result", None)
        if not result:
            return None
        # Inline CSS cap — generated images are 1024–1536px; render at
        # 512px max so multi-turn edits stay comfortably on screen.
        return (
            f'\n\n<img src="data:image/png;base64,{result}" '
            f'style="max-width:512px;height:auto" alt="Generated image">\n\n'
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


# =============================================================================
# Engine — single-save mutation in `_save_conversation`
# =============================================================================


class ImageStudioEngine(chat.engine.Orchestrator):
    def _save_conversation(self, conversation, user_id):
        raw_responses = self.app.store.load_raw_api_responses(user_id, conversation.id)
        if raw_responses:
            convo_id, b64_images = self._scan_latest_response(raw_responses[-1])
            url_replacements = self._save_images(user_id, conversation.id, b64_images)

            last = conversation.messages[-1] if conversation.messages else None
            if last and last.get("role") == "assistant":
                content = last.get("content", "")
                if isinstance(content, str):
                    for b64, filename in url_replacements:
                        short_url = f"/files/{user_id}/{conversation.id}/{filename}"
                        content = content.replace(
                            f"data:image/png;base64,{b64}", short_url
                        )
                    last["content"] = content
                if convo_id:
                    last["_openai_conversation_id"] = convo_id

        super()._save_conversation(conversation, user_id)

    def _scan_latest_response(self, latest):
        chunks = latest if isinstance(latest, list) else [latest]
        convo_id = None
        b64_images = []
        for chunk in chunks:
            ctype = chunk.get("type") if isinstance(chunk, dict) else None
            if ctype == "response.completed":
                conv = (chunk.get("response") or {}).get("conversation")
                if isinstance(conv, dict):
                    convo_id = conv.get("id") or convo_id
            elif ctype == "response.output_item.done":
                item = chunk.get("item") or {}
                if item.get("type") == "image_generation_call" and item.get("result"):
                    b64_images.append(item["result"])
            elif isinstance(chunk, dict) and chunk.get("object") == "response":
                conv = chunk.get("conversation")
                if isinstance(conv, dict):
                    convo_id = conv.get("id") or convo_id
                for item in chunk.get("output", []) or []:
                    if item.get("type") == "image_generation_call" and item.get(
                        "result"
                    ):
                        b64_images.append(item["result"])
        return convo_id, b64_images

    def _save_images(self, user_id, convo_id, b64_images):
        if not b64_images:
            return []
        existing = self.app.store.list_files(user_id, convo_id)
        next_index = sum(
            1 for f in existing if f.startswith("images/") and f.endswith(".png")
        )
        replacements = []
        for offset, b64 in enumerate(b64_images):
            try:
                data = base64.b64decode(b64)
            except (ValueError, TypeError):
                continue
            filename = f"images/{next_index + offset}.png"
            self.app.store.save_file(user_id, convo_id, filename, data)
            replacements.append((b64, filename))
        return replacements


# =============================================================================
# Server — add a /files/<user>/<convo>/<filename...> GET route
# =============================================================================


class FileServingDevHandler(_DevHandler):
    def do_GET(self):
        if self.path.startswith("/files/"):
            self._handle_file_download()
        else:
            super().do_GET()

    def _handle_file_download(self):
        parts = self.path.split("/", 4)
        if len(parts) != 5 or parts[1] != "files":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        _, _, user_id, convo_id, filename = parts
        if ".." in filename or "\\" in filename or filename.startswith("/"):
            self.send_error(HTTPStatus.BAD_REQUEST)
            return
        if user_id != self._get_user_id():
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        data = self._app.store.load_file(user_id, convo_id, filename)
        if not data:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = (
            "image/png" if filename.endswith(".png") else "application/octet-stream"
        )
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "private, max-age=3600")
        self.end_headers()
        self.wfile.write(data)


class FileServingDevServer(chat.server.DevServer):
    def run(self, **kwargs):
        host = kwargs.get("host", self._host)
        port = kwargs.get("port", self._port)
        handler = partial(FileServingDevHandler, self.app)
        self.httpd = HTTPServer((host, port), handler)
        print(f"Image Studio running on http://{host}:{port}")
        print("Press Ctrl+C to stop.")
        try:
            self.httpd.serve_forever()
        except KeyboardInterrupt:
            self.httpd.server_close()


# =============================================================================
# App
# =============================================================================


app = chat.Chatnificent(
    llm=OpenAIResponses(tools=[{"type": "image_generation"}]),
    store=chat.store.File(base_dir="image_studio"),
    engine=ImageStudioEngine(),
    server=FileServingDevServer(),
)


if __name__ == "__main__":
    app.run()
