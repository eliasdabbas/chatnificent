"""Server implementations for Chatnificent.

The Server pillar owns the HTTP transport layer: receiving requests,
delegating to the Engine, and formatting responses for the client.
"""

import json
import logging
from abc import ABC, abstractmethod
from functools import partial
from http import HTTPStatus
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


# =============================================================================
# DevServer — zero-dependency stdlib server for development & demonstration
# =============================================================================


class _DevHandler(SimpleHTTPRequestHandler):
    """HTTP request handler for DevServer."""

    def __init__(self, chatnificent_app, *args, **kwargs):
        self._app = chatnificent_app
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._respond_html(self._app.layout.render_page())
        elif self.path == "/api/conversations":
            self._handle_list_conversations()
        elif self.path.startswith("/api/conversations/"):
            convo_id = self.path.split("/api/conversations/", 1)[1].strip("/")
            self._handle_load_conversation(convo_id)
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):
        if self.path == "/api/chat":
            self._handle_chat()
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def _handle_chat(self):
        try:
            body = self._read_json_body()
            if body is None:
                return

            message = body.get("message", "").strip()
            if not message:
                self._respond_json({"error": "Empty message"}, HTTPStatus.BAD_REQUEST)
                return

            user_id = self._app.auth.get_current_user_id()
            convo_id = body.get("conversation_id")

            conversation = self._app.engine.handle_message(message, user_id, convo_id)

            display_messages = [
                msg
                for msg in conversation.messages
                if msg.get("role") == ASSISTANT_ROLE
                and msg.get("content") is not None
                and str(msg.get("content", "")).strip() != ""
            ]

            last_response = display_messages[-1]["content"] if display_messages else ""

            self._respond_json(
                {
                    "response": last_response,
                    "conversation_id": conversation.id,
                }
            )
        except Exception as e:
            logger.exception("DevServer error in /api/chat")
            self._respond_json({"error": str(e)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_list_conversations(self):
        try:
            user_id = self._app.auth.get_current_user_id()
            convo_ids = self._app.store.list_conversations(user_id)
            self._respond_json({"conversations": convo_ids})
        except Exception as e:
            logger.exception("DevServer error in /api/conversations")
            self._respond_json({"error": str(e)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_load_conversation(self, convo_id):
        try:
            user_id = self._app.auth.get_current_user_id()
            conversation = self._app.store.load_conversation(user_id, convo_id)
            if not conversation:
                self._respond_json(
                    {"error": "Conversation not found"}, HTTPStatus.NOT_FOUND
                )
                return

            messages = [
                {"role": msg["role"], "content": msg.get("content", "")}
                for msg in conversation.messages
                if msg.get("role") in (USER_ROLE, ASSISTANT_ROLE)
            ]
            self._respond_json({"id": conversation.id, "messages": messages})
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
        self.end_headers()
        self.wfile.write(body)

    def _respond_html(self, html):
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

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
        self._port = 8050

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
