# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "chatnificent[openai]>=0.0.18",
# ]
# ///
"""
Image Studio — Multi-Turn Image Editing with Live Streaming
===========================================================

A conversational image studio built on OpenAI's Responses API. Type a
prompt, watch the image materialise in real time, then keep refining it
with follow-up messages. Each reply edits the same running image —
no re-uploads, no context overflow, no staring at a blank screen.

What You'll See
---------------
Three features that set this apart from a basic image generator:

**1. Progressive streaming**
    The image appears in the chat before generation is complete. Two
    partial previews are sent as the model works, sharpening into the
    final result. Users see something immediately instead of waiting
    in silence.

**2. Multi-turn editing**
    Every follow-up message edits the current image rather than
    starting from scratch. This is powered by OpenAI's server-side
    ``Conversation`` object — the model remembers prior images without
    the app re-uploading them each turn.

**3. Revised prompt visibility**
    The model automatically rewrites your prompt for better results.
    The revised version is shown below each image so you can see
    exactly what was sent to the image generator — useful for learning
    what kinds of prompts work well.

Running
-------
::

    export OPENAI_API_KEY="sk-..."
    uv run --script examples/openai_responses_image_studio.py

Open http://127.0.0.1:7777 and try this iterative flow::

    "a neon-lit ramen bar in Tokyo at midnight"

    "add a cat sleeping on the counter"
    "make it rain outside the window"
    "turn the whole scene into a watercolour painting"

Each follow-up edits the running image. Your conversation history and
generated images are saved in ``image_studio/`` so they survive
server restarts.

How It Works
------------
Three small subclasses extend the framework — one per pillar:

**LLM** (``OpenAIResponses``)
    Uses OpenAI's Responses API with the built-in ``image_generation``
    tool (``partial_images=2``, ``quality="medium"``, ``output_format="jpeg"``).

    On the first turn it mints a server-side ``conv_*`` conversation ID
    via ``client.conversations.create()``. On every subsequent turn it
    sends only the new user message plus ``conversation=<id>`` — OpenAI
    prepends the full prior context server-side. No image bytes ever
    leave the client after the first request.

    The revised prompt and partial images are surfaced through
    ``extract_stream_delta()``: partial events become ``<img
    data-gen-partial>`` tags (reclassified as SSE ``status`` events so
    they display progressively but don't accumulate), and the final
    ``response.output_item.done`` event produces the permanent image tag
    plus the revised prompt prefix.

**Engine** (``ImageStudioEngine``)
    Overrides two methods:

    ``handle_message_stream()`` — intercepts partial-image ``delta``
    events and re-emits them as ``status`` events. The browser UI
    replaces the status area on each new ``status`` event and removes
    it entirely on ``done``, so users see the image sharpen in place
    rather than accumulating a stack of blurry previews.

    ``_save_conversation()`` — runs between raw-log persistence and the
    canonical save. It regex-scans the latest assistant message for
    ``<img src="data:image/jpeg;base64,...">`` tags, decodes each one,
    writes it to ``images/<n>.jpeg`` via the Store pillar, and rewrites
    the src to a short ``/files/<user>/<convo>/images/<n>.jpeg`` URL.
    Partial placeholders are stripped. The result: ``messages.json``
    stays small (URLs, not megabytes of base64), and every image is a
    real file on disk.

**Server** (``FileServingDevServer``)
    Adds a single ``/files/<user>/<convo>/<filename>`` GET route. The
    handler validates the path (no ``..`` traversal), checks that the
    requesting session owns the file, loads bytes from the Store, and
    serves them with the correct ``image/jpeg`` content type. The
    default chat UI's ``<img>`` tags resolve via this route with no
    frontend changes.

Why Not Inline Base64?
----------------------
The companion example ``openai_responses_image_generator.py`` keeps
everything inline for simplicity — it's the "minimally complete" pitch.
The trade-off: a 1024×1024 JPEG base64-encoded is around 200 KB, and
it gets replayed as input on every follow-up turn. Across a multi-turn
editing session this blows the context window. This example solves that
by separating storage (the Store pillar, ``images/<n>.jpeg``) from
display (short ``/files/...`` URLs in ``messages.json``).

What to Explore Next
--------------------
- ``openai_responses_image_generator.py`` — the minimal one-file
  version; great starting point before reading this one
- ``starlette_quickstart.py`` — swap ``DevServer`` for a production
  async server with one parameter change
- ``openai_responses_website_search.py`` — same Responses API pattern
  applied to web search instead of image generation
- ``tool_calling.py`` — build your own Python-based tools using the
  ``PythonTool`` pillar
"""

import base64
import re
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
        if chunk.type == "response.image_generation_call.partial_image":
            return self._format_partial_image(
                chunk.partial_image_b64, chunk.partial_image_index
            )
        if chunk.type == "response.output_item.done":
            return self._format_image(chunk.item)
        return None

    def _format_partial_image(self, b64, index):
        # Marked with data-gen-partial so _save_conversation can strip them
        return (
            f'\n<img data-gen-partial="1" '
            f'src="data:image/jpeg;base64,{b64}" '
            f'style="max-width:512px;height:auto" alt="Generating...">\n'
        )

    def _format_image(self, item):
        if getattr(item, "type", None) != "image_generation_call":
            return None
        result = getattr(item, "result", None)
        if not result:
            return None
        revised = getattr(item, "revised_prompt", None)
        prefix = f'_Revised prompt: "{revised}"_\n\n' if revised else ""
        img = (
            f'\n\n<img src="data:image/jpeg;base64,{result}" '
            'style="max-width:512px;height:auto" alt="Generated image">\n\n'
        )
        return prefix + img

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
    def handle_message_stream(self, user_input, user_id, convo_id_from_url):
        # Partial-image deltas must not accumulate in the browser's message
        # bubble — they'd pile up above the final image. Instead, re-type them
        # as `status` events: the UI replaces the status area on every new
        # status event and strips it entirely on `done`, so users see the image
        # progressively sharpen without any redundant copies persisting.
        for event in super().handle_message_stream(
            user_input, user_id, convo_id_from_url
        ):
            if event.get(
                "event"
            ) == "delta" and '<img data-gen-partial="1"' in event.get("data", ""):
                yield {"event": "status", "data": event["data"]}
            else:
                yield event

    def _save_conversation(self, conversation, user_id):
        last = conversation.messages[-1] if conversation.messages else None
        if last and last.get("role") == "assistant":
            content = last.get("content", "")
            if isinstance(content, str):
                # Strip partial placeholders first
                content = re.sub(
                    r'\n?<img data-gen-partial="1"[^>]*>\n?',
                    "",
                    content,
                )
                # Save each final inline image directly from content,
                # replacing the base64 src with a /files/... URL in-place.
                # We extract b64 from content itself — no JSONL comparison
                # needed, so there is no risk of a serialization mismatch.
                existing = self.app.store.list_files(user_id, conversation.id)
                counter = [
                    sum(
                        1
                        for f in existing
                        if f.startswith("images/") and f.endswith(".jpeg")
                    )
                ]

                def _replace_img(match):
                    b64 = match.group(1)
                    try:
                        data = base64.b64decode(b64)
                    except (ValueError, TypeError):
                        return match.group(0)
                    filename = f"images/{counter[0]}.jpeg"
                    self.app.store.save_file(user_id, conversation.id, filename, data)
                    counter[0] += 1
                    url = f"/files/{user_id}/{conversation.id}/{filename}"
                    return (
                        f'<img src="{url}" '
                        'style="max-width:512px;height:auto" '
                        'alt="Generated image">'
                    )

                content = re.sub(
                    r'<img src="data:image/jpeg;base64,([A-Za-z0-9+/=]+)"'
                    r"[^>]*>",
                    _replace_img,
                    content,
                )
                last["content"] = content

        # Attach the OpenAI conversation ID from raw logs
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
            ctype = chunk.get("type")
            if ctype == "response.completed":
                conv = (chunk.get("response") or {}).get("conversation")
                if isinstance(conv, dict) and conv.get("id"):
                    return conv["id"]
            conv = chunk.get("conversation")
            if isinstance(conv, dict) and conv.get("id"):
                return conv["id"]
        return None


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
            "image/png"
            if filename.endswith(".png")
            else "image/jpeg"
            if filename.endswith(".jpeg")
            else "application/octet-stream"
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
    llm=OpenAIResponses(
        model="gpt-5.4",
        tools=[
            {
                "type": "image_generation",
                "partial_images": 2,
                "quality": "medium",
                "output_format": "jpeg",
            }
        ],
    ),
    store=chat.store.File(base_dir="image_studio"),
    engine=ImageStudioEngine(),
    server=FileServingDevServer(),
)


if __name__ == "__main__":
    app.run()
