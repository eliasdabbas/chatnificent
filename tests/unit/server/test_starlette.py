"""Tests for the StarletteServer adapter.

Validates the Starlette implementation against the Server pillar contract
documented in AGENTS.md. Uses Starlette's TestClient (httpx-based) for
fast, in-process HTTP testing.
"""

import json
from unittest.mock import Mock

import pytest

starlette = pytest.importorskip(
    "starlette", reason="StarletteServer tests require the starlette extra"
)
httpx = pytest.importorskip("httpx", reason="StarletteServer tests require httpx")

from chatnificent import Chatnificent
from chatnificent.llm import Echo
from chatnificent.models import Conversation
from chatnificent.server import Starlette as StarletteServer
from chatnificent.store import InMemory
from starlette.testclient import TestClient


def _make_app(**kwargs):
    """Create a Chatnificent app wired to StarletteServer + Echo LLM."""
    defaults = dict(
        server=StarletteServer(),
        llm=Echo(stream=False),
        store=InMemory(),
    )
    defaults.update(kwargs)
    return Chatnificent(**defaults)


def _client(app=None):
    """Return a Starlette TestClient for the given (or default) app."""
    if app is None:
        app = _make_app()
    return TestClient(app.server.asgi_app)


# =============================================================================
# Server lifecycle
# =============================================================================


class TestStarletteServerCreation:
    """Test StarletteServer lifecycle and ABC compliance."""

    def test_creates_asgi_app(self):
        """create_server() produces a Starlette ASGI application."""
        app = _make_app()
        assert app.server.asgi_app is not None

    def test_is_not_default_server(self):
        """Without an explicit server, Chatnificent defaults to DevServer."""
        from chatnificent.server import DevServer

        app = Chatnificent()
        assert isinstance(app.server, DevServer)


# =============================================================================
# GET / — HTML page
# =============================================================================


class TestGetRoot:
    """GET / serves the chat UI HTML page."""

    def test_returns_html(self):
        client = _client()
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]

    def test_convo_injection(self):
        """GET /{user}/{convo} injects window.__CHATNIFICENT_CONVO__."""
        app = _make_app()
        user_id = "testuser"
        convo = Conversation(id="abc123", messages=[{"role": "user", "content": "hi"}])
        app.store.save_conversation(user_id, convo)

        client = TestClient(
            app.server.asgi_app, cookies={"chatnificent_session": user_id}
        )
        r = client.get(f"/{user_id}/abc123")
        assert r.status_code == 200
        assert 'window.__CHATNIFICENT_CONVO__="abc123"' in r.text


# =============================================================================
# POST /api/chat — non-streaming
# =============================================================================


class TestPostChat:
    """POST /api/chat with non-streaming LLM."""

    def test_returns_json_response(self):
        client = _client()
        r = client.post("/api/chat", json={"message": "hello"})
        assert r.status_code == 200
        data = r.json()
        assert "response" in data
        assert "conversation_id" in data
        assert "path" in data
        assert "messages" in data

    def test_empty_message_returns_400(self):
        client = _client()
        r = client.post("/api/chat", json={"message": ""})
        assert r.status_code == 400
        assert "error" in r.json()

    def test_empty_body_returns_400(self):
        client = _client()
        r = client.post(
            "/api/chat",
            content=b"",
            headers={"content-type": "application/json"},
        )
        assert r.status_code == 400

    def test_conversation_persists(self):
        """The conversation created by POST /api/chat is retrievable."""
        app = _make_app()
        client = TestClient(app.server.asgi_app)
        r = client.post("/api/chat", json={"message": "hello"})
        convo_id = r.json()["conversation_id"]

        r2 = client.get(f"/api/conversations/{convo_id}")
        assert r2.status_code == 200
        assert r2.json()["id"] == convo_id


# =============================================================================
# POST /api/chat — streaming (SSE)
# =============================================================================


class TestPostChatStream:
    """POST /api/chat with streaming LLM returns SSE."""

    def _make_streaming_client(self):
        app = _make_app(llm=Echo(stream=True))
        return app, TestClient(app.server.asgi_app)

    def test_returns_event_stream(self):
        _, client = self._make_streaming_client()
        with client.stream("POST", "/api/chat", json={"message": "hello"}) as r:
            assert r.status_code == 200
            assert "text/event-stream" in r.headers["content-type"]

    def test_stream_contains_delta_and_done(self):
        _, client = self._make_streaming_client()
        with client.stream("POST", "/api/chat", json={"message": "hello"}) as r:
            lines = list(r.iter_lines())

        events = []
        for line in lines:
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

        event_types = [e["event"] for e in events]
        assert "delta" in event_types
        assert "done" in event_types

    def test_done_event_has_conversation_id_and_path(self):
        _, client = self._make_streaming_client()
        with client.stream("POST", "/api/chat", json={"message": "hello"}) as r:
            lines = list(r.iter_lines())

        events = []
        for line in lines:
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

        done = next(e for e in events if e["event"] == "done")
        assert "conversation_id" in done["data"]
        assert "path" in done["data"]


# =============================================================================
# GET /api/conversations
# =============================================================================


class TestGetConversations:
    """GET /api/conversations returns conversation list."""

    def test_empty_list(self):
        client = _client()
        r = client.get("/api/conversations")
        assert r.status_code == 200
        assert r.json() == {"conversations": []}

    def test_lists_existing_conversations(self):
        app = _make_app()
        client = TestClient(app.server.asgi_app)

        client.post("/api/chat", json={"message": "hello"})
        r = client.get("/api/conversations")
        assert r.status_code == 200
        convos = r.json()["conversations"]
        assert len(convos) >= 1
        assert "id" in convos[0]
        assert "title" in convos[0]

    def test_title_truncation(self):
        """Long first messages are truncated to 30 chars + ellipsis."""
        app = _make_app()
        client = TestClient(app.server.asgi_app)

        long_msg = "a" * 50
        client.post("/api/chat", json={"message": long_msg})
        r = client.get("/api/conversations")
        title = r.json()["conversations"][0]["title"]
        assert len(title) <= 31  # 30 + "…"
        assert title.endswith("…")


# =============================================================================
# GET /api/conversations/{id}
# =============================================================================


class TestGetConversation:
    """GET /api/conversations/{id} returns conversation detail."""

    def test_returns_conversation(self):
        app = _make_app()
        client = TestClient(app.server.asgi_app)

        r = client.post("/api/chat", json={"message": "hello"})
        convo_id = r.json()["conversation_id"]

        r2 = client.get(f"/api/conversations/{convo_id}")
        assert r2.status_code == 200
        data = r2.json()
        assert data["id"] == convo_id
        assert "messages" in data
        assert "path" in data

    def test_not_found_returns_404(self):
        client = _client()
        r = client.get("/api/conversations/nonexistent")
        assert r.status_code == 404
        assert "error" in r.json()


# =============================================================================
# Auth contract — cookie-based session
# =============================================================================


class TestAuthContract:
    """StarletteServer must follow the same auth contract as DevServer."""

    def test_new_session_sets_cookie(self):
        """First request sets chatnificent_session cookie."""
        client = _client()
        r = client.get("/api/conversations")
        assert "chatnificent_session" in r.cookies

    def test_session_cookie_reused(self):
        """Subsequent requests with cookie don't generate a new session."""
        app = _make_app()
        client = TestClient(app.server.asgi_app)

        r1 = client.get("/api/conversations")
        session_1 = r1.cookies.get("chatnificent_session")

        r2 = client.get("/api/conversations")
        session_2 = r2.cookies.get("chatnificent_session")
        # TestClient persists cookies, so second request should not set new one
        # (cookie already sent back, so no Set-Cookie needed)
        assert session_2 is None or session_2 == session_1

    def test_auth_receives_session_id(self):
        """auth.get_current_user_id() is called with session_id from cookie."""
        app = _make_app()
        app.auth = Mock()
        app.auth.get_current_user_id.return_value = "mock_user"

        client = TestClient(
            app.server.asgi_app, cookies={"chatnificent_session": "abc123"}
        )
        client.get("/api/conversations")

        app.auth.get_current_user_id.assert_called()
        kwargs = app.auth.get_current_user_id.call_args.kwargs
        assert kwargs.get("session_id") == "abc123"


# =============================================================================
# Error handling — JSON error responses, not raw HTML
# =============================================================================


class TestErrorHandling:
    """Starlette must return JSON errors when pillars raise exceptions."""

    def test_engine_error_returns_500_json(self):
        """Engine exception in /api/chat → 500 with JSON error body."""
        app = _make_app()
        app.engine.handle_message = Mock(side_effect=RuntimeError("engine boom"))

        client = TestClient(app.server.asgi_app, raise_server_exceptions=False)
        r = client.post("/api/chat", json={"message": "hello"})
        assert r.status_code == 500
        data = r.json()
        assert "error" in data
        assert "engine boom" in data["error"]

    def test_store_error_in_list_returns_500_json(self):
        """Store exception in /api/conversations → 500 with JSON error body."""
        app = _make_app()
        app.store.list_conversations = Mock(side_effect=RuntimeError("store boom"))

        client = TestClient(app.server.asgi_app, raise_server_exceptions=False)
        r = client.get("/api/conversations")
        assert r.status_code == 500
        data = r.json()
        assert "error" in data
        assert "store boom" in data["error"]

    def test_store_error_in_load_returns_500_json(self):
        """Store exception in /api/conversations/{id} → 500 with JSON."""
        app = _make_app()
        app.store.load_conversation = Mock(side_effect=RuntimeError("load boom"))

        client = TestClient(app.server.asgi_app, raise_server_exceptions=False)
        r = client.get("/api/conversations/some-id")
        assert r.status_code == 500
        data = r.json()
        assert "error" in data
        assert "load boom" in data["error"]

    def test_stream_error_emits_sse_error_event(self):
        """Engine exception during streaming → SSE error event emitted."""
        app = _make_app(llm=Echo(stream=True))

        def _exploding_stream(*args, **kwargs):
            yield {"event": "delta", "data": "partial"}
            raise RuntimeError("stream boom")

        app.engine.handle_message_stream = _exploding_stream

        client = TestClient(app.server.asgi_app, raise_server_exceptions=False)
        with client.stream("POST", "/api/chat", json={"message": "hi"}) as r:
            lines = list(r.iter_lines())

        events = []
        for line in lines:
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

        error_events = [e for e in events if e.get("event") == "error"]
        assert len(error_events) >= 1
        assert "stream boom" in error_events[0]["data"]


# =============================================================================
# Constructor — Starlette-native parameters
# =============================================================================


class TestStarletteConstructor:
    """Constructor mirrors starlette.applications.Starlette signature."""

    def test_default_prefix_is_empty(self):
        server = StarletteServer()
        assert server._prefix == ""

    def test_prefix_strips_trailing_slash(self):
        server = StarletteServer(prefix="/chat/")
        assert server._prefix == "/chat"

    def test_user_routes_prepended(self):
        """User-supplied routes appear before framework routes."""
        from starlette.responses import PlainTextResponse
        from starlette.routing import Route

        async def health(request):
            return PlainTextResponse("ok")

        app = _make_app(server=StarletteServer(routes=[Route("/health", health)]))
        client = TestClient(app.server.asgi_app)
        r = client.get("/health")
        assert r.status_code == 200
        assert r.text == "ok"

    def test_middleware_applied(self):
        """Middleware passed to constructor is active on responses."""
        from starlette.middleware import Middleware
        from starlette.middleware.base import BaseHTTPMiddleware

        class TagMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                response = await call_next(request)
                response.headers["X-Tag"] = "chatnificent"
                return response

        app = _make_app(server=StarletteServer(middleware=[Middleware(TagMiddleware)]))
        client = TestClient(app.server.asgi_app)
        r = client.get("/api/conversations")
        assert r.headers.get("X-Tag") == "chatnificent"

    def test_framework_routes_still_work_with_user_routes(self):
        """Framework endpoints remain accessible when user routes are added."""
        from starlette.responses import PlainTextResponse
        from starlette.routing import Route

        async def custom(request):
            return PlainTextResponse("custom")

        app = _make_app(server=StarletteServer(routes=[Route("/custom", custom)]))
        client = TestClient(app.server.asgi_app)
        r = client.get("/api/conversations")
        assert r.status_code == 200
        assert "conversations" in r.json()


# =============================================================================
# Prefix — outgoing path generation
# =============================================================================


class TestPrefix:
    """prefix= prepends to all outgoing paths in JSON responses."""

    def test_chat_response_path_includes_prefix(self):
        app = _make_app(server=StarletteServer(prefix="/chat"))
        client = TestClient(app.server.asgi_app)
        r = client.post("/api/chat", json={"message": "hello"})
        assert r.status_code == 200
        assert r.json()["path"].startswith("/chat/")

    def test_conversation_detail_path_includes_prefix(self):
        app = _make_app(server=StarletteServer(prefix="/chat"))
        client = TestClient(app.server.asgi_app)

        r = client.post("/api/chat", json={"message": "hello"})
        convo_id = r.json()["conversation_id"]

        r2 = client.get(f"/api/conversations/{convo_id}")
        assert r2.json()["path"].startswith("/chat/")

    def test_stream_done_path_includes_prefix(self):
        app = _make_app(server=StarletteServer(prefix="/chat"), llm=Echo(stream=True))
        client = TestClient(app.server.asgi_app)

        with client.stream("POST", "/api/chat", json={"message": "hi"}) as r:
            lines = list(r.iter_lines())

        events = []
        for line in lines:
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

        done = next(e for e in events if e["event"] == "done")
        assert done["data"]["path"].startswith("/chat/")

    def test_no_prefix_paths_unchanged(self):
        """Without prefix, paths have no prefix added."""
        app = _make_app(server=StarletteServer())
        client = TestClient(app.server.asgi_app)
        r = client.post("/api/chat", json={"message": "hello"})
        path = r.json()["path"]
        assert not path.startswith("/chat")


# =============================================================================
# ASGI __call__ — direct uvicorn usage
# =============================================================================


class TestASGICallable:
    """Chatnificent.__call__ delegates to server.asgi_app for uvicorn."""

    def test_chatnificent_is_asgi_callable(self):
        """uvicorn app:app works — Chatnificent delegates to asgi_app."""
        app = _make_app()
        client = TestClient(app)
        r = client.get("/api/conversations")
        assert r.status_code == 200
        assert "conversations" in r.json()

    def test_chat_via_asgi_callable(self):
        """POST /api/chat works through Chatnificent.__call__."""
        app = _make_app()
        client = TestClient(app)
        r = client.post("/api/chat", json={"message": "hello"})
        assert r.status_code == 200
        assert "response" in r.json()

    def test_asgi_not_supported_for_devserver(self):
        """DevServer has no asgi_app — __call__ raises TypeError."""
        app = Chatnificent()
        client = TestClient(app, raise_server_exceptions=False)
        r = client.get("/")
        assert r.status_code == 500


# =============================================================================
# run() — uvicorn kwargs passthrough
# =============================================================================


class TestRunKwargsPassthrough:
    """StarletteServer.run() passes **kwargs transparently to uvicorn.run()."""

    def test_defaults(self, mocker):
        """Without kwargs, uses host=127.0.0.1, port=7777, log_level=info."""
        mock_run = mocker.patch("uvicorn.run")
        app = _make_app()
        app.server.run()
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs["host"] == "127.0.0.1"
        assert call_kwargs.kwargs["port"] == 7777
        assert call_kwargs.kwargs["log_level"] == "info"

    def test_custom_host_and_port(self, mocker):
        """host and port override defaults."""
        mock_run = mocker.patch("uvicorn.run")
        app = _make_app()
        app.server.run(host="0.0.0.0", port=9000)
        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs["host"] == "0.0.0.0"
        assert call_kwargs.kwargs["port"] == 9000

    def test_debug_sets_log_level(self, mocker):
        """debug=True maps to log_level='debug'."""
        mock_run = mocker.patch("uvicorn.run")
        app = _make_app()
        app.server.run(debug=True)
        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs["log_level"] == "debug"

    def test_explicit_log_level_overrides_debug(self, mocker):
        """Explicit log_level takes precedence over debug flag."""
        mock_run = mocker.patch("uvicorn.run")
        app = _make_app()
        app.server.run(debug=True, log_level="warning")
        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs["log_level"] == "warning"

    def test_extra_kwargs_passed_to_uvicorn(self, mocker):
        """Arbitrary uvicorn kwargs are forwarded transparently."""
        mock_run = mocker.patch("uvicorn.run")
        app = _make_app()
        app.server.run(
            workers=4,
            reload=True,
            timeout_keep_alive=30,
            ssl_keyfile="/path/to/key.pem",
        )
        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs["workers"] == 4
        assert call_kwargs["reload"] is True
        assert call_kwargs["timeout_keep_alive"] == 30
        assert call_kwargs["ssl_keyfile"] == "/path/to/key.pem"

    def test_debug_not_forwarded_to_uvicorn(self, mocker):
        """debug is a Chatnificent flag, not a uvicorn parameter."""
        mock_run = mocker.patch("uvicorn.run")
        app = _make_app()
        app.server.run(debug=True)
        call_kwargs = mock_run.call_args.kwargs
        assert "debug" not in call_kwargs

    def test_first_arg_is_asgi_app(self, mocker):
        """First positional arg to uvicorn.run is the ASGI app."""
        mock_run = mocker.patch("uvicorn.run")
        app = _make_app()
        app.server.run()
        assert mock_run.call_args.args[0] is app.server.asgi_app
