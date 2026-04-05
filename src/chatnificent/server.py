"""Server implementations for Chatnificent.

The Server pillar owns the HTTP transport layer: receiving requests,
delegating to the Engine, and formatting responses for the client.
"""

import json
import logging
import uuid
from abc import ABC, abstractmethod
from functools import partial
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import HTTPServer, SimpleHTTPRequestHandler
from typing import TYPE_CHECKING, Any, Optional

from .models import ASSISTANT_ROLE, SYSTEM_ROLE, USER_ROLE

if TYPE_CHECKING:
    from . import Chatnificent

logger = logging.getLogger(__name__)


class Server(ABC):
    """Abstract Base Class for all Chatnificent servers."""

    def __init__(self, app: Optional["Chatnificent"] = None) -> None:
        """Initialize with optional app reference (bound during Chatnificent init)."""
        self.app = app

    @abstractmethod
    def create_server(self, **kwargs) -> Any:
        """Create the underlying web application object.

        Parameters
        ----------
        **kwargs
            Server-specific configuration passed through from Chatnificent.

        Returns
        -------
        Any
            The web application object (e.g. Dash app, Starlette app).
        """
        pass

    @abstractmethod
    def run(self, **kwargs) -> None:
        """Start serving HTTP requests.

        Parameters
        ----------
        **kwargs
            Runtime options (host, port, debug, etc.).
        """
        pass

    # -- Shared helpers -------------------------------------------------------
    # Non-abstract methods available to all server implementations. These
    # eliminate duplicated logic across DevServer, Starlette, etc.

    def _build_conversation_title(self, conversation):
        """Derive a display title from a Conversation's first user message.

        Parameters
        ----------
        conversation : Conversation
            The conversation to extract a title from.

        Returns
        -------
        str
            First user message content truncated to 30 chars with "…",
            or the conversation id as fallback.
        """
        if conversation and conversation.messages:
            first_user = next(
                (m for m in conversation.messages if m.get("role") == USER_ROLE),
                None,
            )
            if first_user:
                content = first_user.get("content", "")
                if isinstance(content, str) and content.strip():
                    stripped = content.strip()
                    return stripped[:30] + ("…" if len(stripped) > 30 else "")
        return conversation.id if conversation else ""

    def _extract_last_response(self, display_messages):
        """Extract the last assistant message content from display messages.

        Parameters
        ----------
        display_messages : list[dict]
            Messages already filtered for display.

        Returns
        -------
        str
            The content of the last non-empty assistant message, or "".
        """
        assistant_messages = [
            msg
            for msg in display_messages
            if msg.get("role") == ASSISTANT_ROLE
            and msg.get("content") is not None
            and str(msg.get("content", "")).strip() != ""
        ]
        return assistant_messages[-1]["content"] if assistant_messages else ""

    def _is_llm_streaming(self):
        """Check if the configured LLM is set up for streaming.

        Returns
        -------
        bool
        """
        llm = self.app.llm
        if getattr(llm, "default_params", {}).get("stream", False):
            return True
        if hasattr(llm, "_streaming") and llm._streaming:
            return True
        return False

    def _render_messages(self, user_id, conversation):
        """Delegate message filtering to the Layout pillar.

        Parameters
        ----------
        user_id : str
        conversation : Conversation

        Returns
        -------
        list[dict]
        """
        return self.app.layout.render_messages(
            conversation.messages,
            user_id=user_id,
            convo_id=conversation.id,
            conversation=conversation,
        )

    def _render_conversations(self, user_id, conversations):
        """Delegate conversation-list shaping to the Layout pillar.

        Parameters
        ----------
        user_id : str
        conversations : list[dict]

        Returns
        -------
        list[dict]
        """
        return self.app.layout.render_conversations(conversations, user_id=user_id)

    def _inject_root_into_html(self, html, root_path):
        """Inject ``window.__CHATNIFICENT_ROOT__`` into HTML when mounted.

        Parameters
        ----------
        html : str
            The rendered HTML page.
        root_path : str
            The ASGI root_path (mount prefix). Empty string when not mounted.

        Returns
        -------
        str
            HTML with the script tag injected before ``</head>``, or
            unchanged if *root_path* is empty.
        """
        if not root_path:
            return html
        safe_root = root_path.replace("\\", "\\\\").replace('"', '\\"')
        tag = f'<script>window.__CHATNIFICENT_ROOT__="{safe_root}";</script>'
        return html.replace("</head>", tag + "</head>")

    def _build_full_conversation_path(self, root_path, user_id, convo_id):
        """Build a URL path with the mount prefix prepended.

        Parameters
        ----------
        root_path : str
            The mount prefix (e.g. ``"/code"``). Empty when not mounted.
        user_id : str
        convo_id : str

        Returns
        -------
        str
        """
        return root_path + self.app.url.build_conversation_path(user_id, convo_id)

    @staticmethod
    def _cookie_path(root_path):
        """Compute the cookie ``Path`` attribute for the given mount prefix.

        Parameters
        ----------
        root_path : str
            Mount prefix (e.g. ``"/code"``). Empty when not mounted.

        Returns
        -------
        str
            ``root_path + "/"`` when mounted, ``"/"`` otherwise.
        """
        return (root_path + "/") if root_path else "/"


# =============================================================================
# DevServer — zero-dependency stdlib server for development & demonstration
# =============================================================================


class _DevHandler(SimpleHTTPRequestHandler):
    """HTTP request handler for DevServer."""

    def __init__(self, chatnificent_app, *args, **kwargs):
        self._app = chatnificent_app
        self._new_session = False
        self._session_id = None
        super().__init__(*args, **kwargs)

    def _get_user_id(self):
        """Get user identity by delegating to the Auth pillar."""
        if self._session_id:
            return self._session_id
        cookie = SimpleCookie(self.headers.get("Cookie", ""))
        cookie_value = None
        if "chatnificent_session" in cookie:
            cookie_value = cookie["chatnificent_session"].value
        self._session_id = self._app.auth.get_current_user_id(session_id=cookie_value)
        if not cookie_value:
            self._new_session = True
        return self._session_id

    def _has_session_cookie(self):
        """Check if the request carries an existing session cookie."""
        cookie = SimpleCookie(self.headers.get("Cookie", ""))
        return "chatnificent_session" in cookie

    def do_GET(self):
        if self.path == "/api/conversations":
            self._handle_list_conversations()
        elif self.path.startswith("/api/conversations/"):
            convo_id = self.path.split("/api/conversations/", 1)[1].strip("/")
            self._handle_load_conversation(convo_id)
        elif not self.path.startswith("/api/"):
            url_parts = self._app.url.parse(self.path)
            if url_parts.user_id and not self._has_session_cookie():
                self._session_id = url_parts.user_id
                self._new_session = True
            html = self._app.layout.render_page()
            if url_parts.convo_id:
                safe_id = url_parts.convo_id.replace("\\", "\\\\").replace('"', '\\"')
                tag = f'<script>window.__CHATNIFICENT_CONVO__="{safe_id}";</script>'
                html = html.replace("</head>", tag + "</head>")
            self._respond_html(html)
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):
        if self.path == "/api/chat":
            if self._server._is_llm_streaming():
                self._handle_chat_stream()
            else:
                self._handle_chat()
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    @property
    def _server(self):
        return self._app.server

    def _handle_chat(self):
        try:
            body = self._read_json_body()
            if body is None:
                return

            message = body.get("message", "").strip()
            if not message:
                self._respond_json({"error": "Empty message"}, HTTPStatus.BAD_REQUEST)
                return

            user_id = self._get_user_id()
            convo_id = body.get("conversation_id")

            conversation = self._app.engine.handle_message(message, user_id, convo_id)
            display_messages = self._server._render_messages(user_id, conversation)
            last_response = self._server._extract_last_response(display_messages)

            self._respond_json(
                {
                    "response": last_response,
                    "messages": display_messages,
                    "conversation_id": conversation.id,
                    "path": self._app.url.build_conversation_path(
                        user_id, conversation.id
                    ),
                }
            )
        except Exception as e:
            logger.exception("DevServer error in /api/chat")
            self._respond_json({"error": str(e)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_chat_stream(self):
        try:
            body = self._read_json_body()
            if body is None:
                return

            message = body.get("message", "").strip()
            if not message:
                self._respond_json({"error": "Empty message"}, HTTPStatus.BAD_REQUEST)
                return

            user_id = self._get_user_id()
            convo_id = body.get("conversation_id")

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self._maybe_set_session_cookie()
            self.end_headers()

            for event in self._app.engine.handle_message_stream(
                message, user_id, convo_id
            ):
                if event.get("event") == "done" and isinstance(event.get("data"), dict):
                    cid = event["data"].get("conversation_id")
                    if cid:
                        event["data"]["path"] = self._app.url.build_conversation_path(
                            user_id, cid
                        )
                line = f"data: {json.dumps(event)}\n\n"
                self.wfile.write(line.encode("utf-8"))
                self.wfile.flush()

        except Exception as e:
            logger.exception("DevServer error in /api/chat (stream)")
            try:
                error_event = json.dumps({"event": "error", "data": str(e)})
                self.wfile.write(f"data: {error_event}\n\n".encode("utf-8"))
                self.wfile.flush()
            except Exception:
                pass

    def _handle_list_conversations(self):
        try:
            user_id = self._get_user_id()
            convo_ids = self._app.store.list_conversations(user_id)
            conversations = []
            for cid in convo_ids:
                convo = self._app.store.load_conversation(user_id, cid)
                title = self._server._build_conversation_title(convo) if convo else cid
                conversations.append({"id": cid, "title": title})
            conversations = self._server._render_conversations(user_id, conversations)
            self._respond_json({"conversations": conversations})
        except Exception as e:
            logger.exception("DevServer error in /api/conversations")
            self._respond_json({"error": str(e)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_load_conversation(self, convo_id):
        try:
            user_id = self._get_user_id()
            conversation = self._app.store.load_conversation(user_id, convo_id)
            if not conversation:
                self._respond_json(
                    {"error": "Conversation not found"}, HTTPStatus.NOT_FOUND
                )
                return

            messages = self._server._render_messages(user_id, conversation)
            self._respond_json(
                {
                    "id": conversation.id,
                    "messages": messages,
                    "path": self._app.url.build_conversation_path(
                        user_id, conversation.id
                    ),
                }
            )
        except Exception as e:
            logger.exception("DevServer error loading conversation")
            self._respond_json({"error": str(e)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def _read_json_body(self):
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self._respond_json({"error": "Empty body"}, HTTPStatus.BAD_REQUEST)
            return None
        raw = self.rfile.read(content_length)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            self._respond_json({"error": "Invalid JSON"}, HTTPStatus.BAD_REQUEST)
            return None

    def _respond_json(self, data, status=HTTPStatus.OK):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._maybe_set_session_cookie()
        self.end_headers()
        self.wfile.write(body)

    def _respond_html(self, html):
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._maybe_set_session_cookie()
        self.end_headers()
        self.wfile.write(body)

    def _maybe_set_session_cookie(self):
        if self._new_session and self._session_id:
            self.send_header(
                "Set-Cookie",
                f"chatnificent_session={self._session_id}; Path=/; SameSite=Lax",
            )

    def log_message(self, format, *args):
        logger.info(format, *args)


class DevServer(Server):
    """Zero-dependency development server using Python's stdlib http.server.

    Serves a minimal HTML chat UI and JSON API endpoints. Not intended for
    production — analogous to Echo for the LLM pillar. Proves the full
    pipeline works without installing any extras.

    Endpoints
    ---------
    GET  /                          Minimal chat UI (vanilla JS)
    POST /api/chat                  Send message, get response
    GET  /api/conversations         List conversation IDs
    GET  /api/conversations/{id}    Load a conversation
    """

    def __init__(self, app=None):
        super().__init__(app)
        self.httpd = None
        self._host = "127.0.0.1"
        self._port = 7777

    def create_server(self, **kwargs) -> None:
        """Store configuration. The actual HTTPServer is created lazily in run()."""
        self._host = kwargs.pop("host", self._host)
        self._port = kwargs.pop("port", self._port)

    def run(self, **kwargs) -> None:
        host = kwargs.get("host", self._host)
        port = kwargs.get("port", self._port)
        debug = kwargs.get("debug", False)

        if debug:
            logging.basicConfig(level=logging.DEBUG)

        handler = partial(_DevHandler, self.app)
        self.httpd = HTTPServer((host, port), handler)

        print(f"Chatnificent DevServer running on http://{host}:{port}")
        print("Press Ctrl+C to stop.")
        try:
            self.httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down.")
            self.httpd.server_close()


# =============================================================================
# DashServer — full-stack Dash server with built-in UI
# =============================================================================


class DashServer(Server):
    """Full-stack Dash server with built-in UI.

    Uses the Layout pillar to render the chat interface and registers
    Dash callbacks that bridge user interactions to the Engine.
    """

    def create_server(self, **kwargs) -> Any:
        from dash import Dash

        from .layout import DashLayout

        layout = self.app.layout
        if not isinstance(layout, DashLayout):
            raise TypeError(
                f"DashServer requires a DashLayout (e.g. Bootstrap, Mantine, Minimal), "
                f"got {type(layout).__name__}. "
                f'Pass layout=Bootstrap() or install with: pip install "chatnificent[dash]"'
            )

        if "external_stylesheets" not in kwargs:
            kwargs["external_stylesheets"] = []
        kwargs["external_stylesheets"].extend(layout.get_external_stylesheets())

        if "external_scripts" not in kwargs:
            kwargs["external_scripts"] = []
        kwargs["external_scripts"].extend(layout.get_external_scripts())

        self.dash_app = Dash(**kwargs)
        self.dash_app.layout = layout.build_layout()

        from ._callbacks import register_callbacks

        register_callbacks(self.dash_app, self.app)

        return self.dash_app

    def run(self, **kwargs) -> None:
        self.dash_app.run(**kwargs)


# =============================================================================
# Starlette — production-grade async ASGI server
# =============================================================================


class Starlette(Server):
    """Production-grade async server using Starlette + Uvicorn.

    Implements the same endpoint contract as DevServer with async handlers
    that wrap the synchronous engine via ``anyio.to_thread.run_sync``.

    Parameters
    ----------
    debug : bool
        Forwarded to ``starlette.applications.Starlette(debug=...)``.
    routes : sequence, optional
        Additional Starlette ``Route`` objects. Prepended before framework
        routes so user routes take priority (first-match wins).
    middleware : sequence, optional
        Starlette ``Middleware`` descriptors forwarded to the app constructor.
    exception_handlers : mapping, optional
        Forwarded to ``starlette.applications.Starlette``.
    lifespan : callable, optional
        ASGI lifespan handler for startup/shutdown hooks.

    Examples
    --------
    Minimal:

    >>> import chatnificent as chat
    >>> app = chat.Chatnificent(server=chat.server.Starlette())
    >>> app.run()  # http://127.0.0.1:7777

    Direct uvicorn (enables ``--workers``, ``--reload``, etc.):

    >>> app = chat.Chatnificent(server=chat.server.Starlette())
    >>> # $ uvicorn app:app

    With CORS middleware:

    >>> from starlette.middleware import Middleware
    >>> from starlette.middleware.cors import CORSMiddleware
    >>> server = chat.server.Starlette(
    ...     middleware=[Middleware(CORSMiddleware, allow_origins=["*"])],
    ... )

    """

    def __init__(
        self,
        debug=False,
        routes=None,
        middleware=None,
        exception_handlers=None,
        lifespan=None,
    ):
        super().__init__()
        self.asgi_app = None
        self._debug = debug
        self._user_routes = routes
        self._middleware = middleware
        self._exception_handlers = exception_handlers
        self._lifespan = lifespan

    def create_server(self, **kwargs) -> Any:
        import starlette.applications
        import starlette.routing

        framework_routes = [
            starlette.routing.Route("/api/chat", self._handle_chat, methods=["POST"]),
            starlette.routing.Route(
                "/api/conversations/{convo_id:path}",
                self._handle_load_conversation,
                methods=["GET"],
            ),
            starlette.routing.Route(
                "/api/conversations",
                self._handle_list_conversations,
                methods=["GET"],
            ),
            starlette.routing.Route("/{path:path}", self._handle_page, methods=["GET"]),
            starlette.routing.Route("/", self._handle_page, methods=["GET"]),
        ]
        self.routes = list(self._user_routes or []) + framework_routes

        self.asgi_app = starlette.applications.Starlette(
            debug=self._debug,
            routes=self.routes,
            middleware=self._middleware,
            exception_handlers=self._exception_handlers,
            lifespan=self._lifespan,
        )
        return self.asgi_app

    def run(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 7777,
        workers: int | None = None,
        reload: bool = False,
        log_level: str = "info",
        ssl_keyfile: str | None = None,
        ssl_certfile: str | None = None,
        **kwargs,
    ) -> None:
        import sys
        from pathlib import Path

        import uvicorn

        # Auto-resolve import string so workers/reload can spawn fresh processes.
        app_target: object = self.asgi_app
        main = sys.modules.get("__main__")
        if main:
            for name, obj in vars(main).items():
                if obj is self.app:
                    app_target = f"{Path(sys.argv[0]).stem}:{name}"
                    break

        print(f"Chatnificent Starlette server running on http://{host}:{port}")
        uvicorn.run(
            app_target,
            host=host,
            port=port,
            workers=workers,
            reload=reload,
            log_level=log_level,
            ssl_keyfile=ssl_keyfile,
            ssl_certfile=ssl_certfile,
            **kwargs,
        )

    # -- Auth helper ----------------------------------------------------------

    def _get_session(self, request):
        """Read session cookie and resolve user identity.

        Returns (user_id, is_new_session).
        """
        cookie_value = request.cookies.get("chatnificent_session")
        user_id = self.app.auth.get_current_user_id(session_id=cookie_value)
        is_new = not cookie_value
        return user_id, is_new

    def _get_root_path(self, request):
        """Extract the ASGI root_path (mount prefix) from the request.

        Returns
        -------
        str
            The mount prefix (e.g. ``"/code"``), or ``""`` when not mounted.
        """
        return request.scope.get("root_path", "")

    def _maybe_set_cookie(self, response, user_id, is_new, root_path=""):
        if is_new and user_id:
            response.set_cookie(
                "chatnificent_session",
                user_id,
                path=self._cookie_path(root_path),
                samesite="lax",
            )

    # -- Route handlers -------------------------------------------------------

    async def _handle_page(self, request):
        from starlette.responses import HTMLResponse

        user_id, is_new = self._get_session(request)
        path = request.scope.get("path", request.url.path)
        url_parts = self.app.url.parse(path)

        if url_parts.user_id and not request.cookies.get("chatnificent_session"):
            user_id = url_parts.user_id
            is_new = True

        root_path = self._get_root_path(request)
        html = self._inject_root_into_html(self.app.layout.render_page(), root_path)

        if url_parts.convo_id:
            safe_id = url_parts.convo_id.replace("\\", "\\\\").replace('"', '\\"')
            tag = f'<script>window.__CHATNIFICENT_CONVO__="{safe_id}";</script>'
            html = html.replace("</head>", tag + "</head>")

        response = HTMLResponse(html)
        self._maybe_set_cookie(response, user_id, is_new, root_path)
        return response

    async def _handle_chat(self, request):
        import anyio
        from starlette.responses import JSONResponse

        user_id, is_new = self._get_session(request)

        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)

        message = body.get("message", "").strip() if body else ""
        if not message:
            return JSONResponse({"error": "Empty message"}, status_code=400)

        convo_id = body.get("conversation_id")

        root_path = self._get_root_path(request)

        if self._is_llm_streaming():
            return await self._handle_chat_stream(
                request, user_id, is_new, message, convo_id
            )

        try:

            def _sync_chat():
                conversation = self.app.engine.handle_message(
                    message, user_id, convo_id
                )
                display_messages = self._render_messages(user_id, conversation)
                last_response = self._extract_last_response(display_messages)
                return {
                    "response": last_response,
                    "messages": display_messages,
                    "conversation_id": conversation.id,
                    "path": self._build_full_conversation_path(
                        root_path, user_id, conversation.id
                    ),
                }

            response_data = await anyio.to_thread.run_sync(_sync_chat)
            response = JSONResponse(response_data)
            self._maybe_set_cookie(response, user_id, is_new, root_path)
            return response
        except Exception as e:
            logger.exception("Starlette error in /api/chat")
            return JSONResponse({"error": str(e)}, status_code=500)

    async def _handle_chat_stream(self, request, user_id, is_new, message, convo_id):
        import anyio
        from starlette.responses import StreamingResponse

        root_path = self._get_root_path(request)

        async def event_generator():
            try:
                stream = self.app.engine.handle_message_stream(
                    message, user_id, convo_id
                )
                _sentinel = object()

                while True:
                    if await request.is_disconnected():
                        break
                    event = await anyio.to_thread.run_sync(
                        lambda: next(stream, _sentinel)
                    )
                    if event is _sentinel:
                        break
                    if event.get("event") == "done" and isinstance(
                        event.get("data"), dict
                    ):
                        cid = event["data"].get("conversation_id")
                        if cid:
                            event["data"]["path"] = self._build_full_conversation_path(
                                root_path, user_id, cid
                            )
                    yield f"data: {json.dumps(event)}\n\n"
            except Exception as e:
                logger.exception("Starlette error in /api/chat (stream)")
                error_event = {"event": "error", "data": str(e)}
                yield f"data: {json.dumps(error_event)}\n\n"

        response = StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
        self._maybe_set_cookie(response, user_id, is_new, root_path)
        return response

    async def _handle_list_conversations(self, request):
        import anyio
        from starlette.responses import JSONResponse

        user_id, is_new = self._get_session(request)

        try:

            def _sync_list():
                convo_ids = self.app.store.list_conversations(user_id)
                conversations = []
                for cid in convo_ids:
                    convo = self.app.store.load_conversation(user_id, cid)
                    title = self._build_conversation_title(convo) if convo else cid
                    conversations.append({"id": cid, "title": title})
                return self._render_conversations(user_id, conversations)

            conversations = await anyio.to_thread.run_sync(_sync_list)
            response = JSONResponse({"conversations": conversations})
            self._maybe_set_cookie(
                response, user_id, is_new, self._get_root_path(request)
            )
            return response
        except Exception as e:
            logger.exception("Starlette error in /api/conversations")
            return JSONResponse({"error": str(e)}, status_code=500)

    async def _handle_load_conversation(self, request):
        import anyio
        from starlette.responses import JSONResponse

        user_id, is_new = self._get_session(request)

        try:
            convo_id = request.path_params["convo_id"]
            root_path = self._get_root_path(request)

            def _sync_load():
                conversation = self.app.store.load_conversation(user_id, convo_id)
                if not conversation:
                    return None
                messages = self._render_messages(user_id, conversation)
                return {
                    "id": conversation.id,
                    "messages": messages,
                    "path": self._build_full_conversation_path(
                        root_path, user_id, conversation.id
                    ),
                }

            response_data = await anyio.to_thread.run_sync(_sync_load)
            if response_data is None:
                return JSONResponse(
                    {"error": "Conversation not found"}, status_code=404
                )

            response = JSONResponse(response_data)
            self._maybe_set_cookie(response, user_id, is_new, root_path)
            return response
        except Exception as e:
            logger.exception("Starlette error loading conversation")
            return JSONResponse({"error": str(e)}, status_code=500)
