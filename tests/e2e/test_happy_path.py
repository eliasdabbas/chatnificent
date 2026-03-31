"""End-to-end test: full HTTP request → DevServer → Engine → Echo → Store → HTTP response.

Uses stdlib only (threading + urllib) — no test dependencies beyond pytest.
"""

import http.cookiejar
import json
import socket
import threading
import time
import urllib.error
import urllib.request

import pytest


def _find_free_port():
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_server(host, port, timeout=10):
    """Wait until the server accepts connections."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"Server did not start within {timeout}s")


@pytest.fixture
def running_devserver():
    """Start a DevServer on a random port in a background thread."""
    from functools import partial
    from http.server import HTTPServer

    from chatnificent import Chatnificent
    from chatnificent.llm import Echo
    from chatnificent.server import DevServer, _DevHandler
    from chatnificent.store import InMemory

    port = _find_free_port()
    app = Chatnificent(
        llm=Echo(stream=False),
        store=InMemory(),
        server=DevServer(),
    )
    app.server.create_server(host="127.0.0.1", port=port)

    handler = partial(_DevHandler, app)
    httpd = HTTPServer(("127.0.0.1", port), handler)

    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    _wait_for_server("127.0.0.1", port)

    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(jar)
    )

    yield {
        "host": "127.0.0.1",
        "port": port,
        "app": app,
        "httpd": httpd,
        "opener": opener,
    }

    httpd.shutdown()


def _url(server, path):
    return f"http://{server['host']}:{server['port']}{path}"


def _post_json(server, path, data):
    """POST JSON to the server, return parsed response."""
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        _url(server, path),
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with server["opener"].open(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_json(server, path):
    """GET JSON from the server, return parsed response."""
    req = urllib.request.Request(_url(server, path))
    with server["opener"].open(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_html(server, path):
    """GET HTML from the server, return response text."""
    req = urllib.request.Request(_url(server, path))
    with server["opener"].open(req) as resp:
        return resp.read().decode("utf-8")


class TestHappyPath:
    """Full-stack happy path: send message → get response → conversation persists."""

    def test_get_root_returns_html(self, running_devserver):
        """GET / returns the chat UI HTML."""
        html = _get_html(running_devserver, "/")
        assert "Chatnificent" in html
        assert "<html" in html.lower()

    def test_post_chat_creates_conversation(self, running_devserver):
        """POST /api/chat with a message creates a new conversation."""
        data = _post_json(
            running_devserver,
            "/api/chat",
            {"message": "Hello, world!"},
        )

        assert "conversation_id" in data
        assert data["conversation_id"] is not None
        assert "response" in data
        assert len(data["response"]) > 0
        assert "messages" in data
        assert "path" in data

    def test_conversation_listed_after_creation(self, running_devserver):
        """After sending a message, the conversation appears in the list."""
        chat_data = _post_json(
            running_devserver,
            "/api/chat",
            {"message": "List test"},
        )
        convo_id = chat_data["conversation_id"]

        list_data = _get_json(running_devserver, "/api/conversations")
        convo_ids = [c["id"] for c in list_data["conversations"]]
        assert convo_id in convo_ids

    def test_conversation_loadable_after_creation(self, running_devserver):
        """After sending a message, the conversation can be loaded by ID."""
        chat_data = _post_json(
            running_devserver,
            "/api/chat",
            {"message": "Load test"},
        )
        convo_id = chat_data["conversation_id"]

        convo_data = _get_json(
            running_devserver, f"/api/conversations/{convo_id}"
        )
        assert convo_data["id"] == convo_id
        assert len(convo_data["messages"]) >= 2

        roles = [m["role"] for m in convo_data["messages"]]
        assert "user" in roles
        assert "assistant" in roles

    def test_second_message_grows_conversation(self, running_devserver):
        """Sending a second message to the same conversation appends to it."""
        data1 = _post_json(
            running_devserver,
            "/api/chat",
            {"message": "First message"},
        )
        convo_id = data1["conversation_id"]

        data2 = _post_json(
            running_devserver,
            "/api/chat",
            {"message": "Second message", "conversation_id": convo_id},
        )

        assert data2["conversation_id"] == convo_id

        convo_data = _get_json(
            running_devserver, f"/api/conversations/{convo_id}"
        )
        user_messages = [
            m for m in convo_data["messages"] if m.get("role") == "user"
        ]
        assert len(user_messages) >= 2

    def test_empty_message_returns_400(self, running_devserver):
        """POST /api/chat with empty message returns HTTP 400."""
        body = json.dumps({"message": ""}).encode("utf-8")
        req = urllib.request.Request(
            _url(running_devserver, "/api/chat"),
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            running_devserver["opener"].open(req)
        assert exc_info.value.code == 400

    def test_nonexistent_conversation_returns_404(self, running_devserver):
        """GET /api/conversations/<nonexistent> returns 404."""
        req = urllib.request.Request(
            _url(running_devserver, "/api/conversations/does_not_exist")
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            running_devserver["opener"].open(req)
        assert exc_info.value.code == 404

    def test_multiple_conversations_isolated(self, running_devserver):
        """Two separate conversations don't interfere with each other."""
        data1 = _post_json(
            running_devserver,
            "/api/chat",
            {"message": "Conversation A"},
        )
        data2 = _post_json(
            running_devserver,
            "/api/chat",
            {"message": "Conversation B"},
        )

        assert data1["conversation_id"] != data2["conversation_id"]

        list_data = _get_json(running_devserver, "/api/conversations")
        convo_ids = [c["id"] for c in list_data["conversations"]]
        assert data1["conversation_id"] in convo_ids
        assert data2["conversation_id"] in convo_ids
