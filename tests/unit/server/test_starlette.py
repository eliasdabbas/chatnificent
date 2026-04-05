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
    """StarletteServer.run() passes kwargs transparently to uvicorn.run()."""

    def test_defaults(self, mocker):
        """Without kwargs, uses Chatnificent defaults for all explicit params."""
        mock_run = mocker.patch("uvicorn.run")
        app = _make_app()
        app.server.run()
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs["host"] == "127.0.0.1"
        assert call_kwargs["port"] == 7777
        assert call_kwargs["log_level"] == "info"
        assert call_kwargs["workers"] is None
        assert call_kwargs["reload"] is False
        assert call_kwargs["ssl_keyfile"] is None
        assert call_kwargs["ssl_certfile"] is None

    def test_custom_host_and_port(self, mocker):
        """host and port override defaults."""
        mock_run = mocker.patch("uvicorn.run")
        app = _make_app()
        app.server.run(host="0.0.0.0", port=9000)
        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs["host"] == "0.0.0.0"
        assert call_kwargs["port"] == 9000

    def test_extra_kwargs_passed_to_uvicorn(self, mocker):
        """Arbitrary uvicorn kwargs are forwarded transparently."""
        mock_run = mocker.patch("uvicorn.run")
        app = _make_app()
        app.server.run(
            timeout_keep_alive=30,
            limit_concurrency=100,
            proxy_headers=True,
        )
        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs["timeout_keep_alive"] == 30
        assert call_kwargs["limit_concurrency"] == 100
        assert call_kwargs["proxy_headers"] is True

    def test_explicit_workers_param(self, mocker):
        """workers= explicit param is forwarded to uvicorn."""
        mock_run = mocker.patch("uvicorn.run")
        app = _make_app()
        app.server.run(workers=8)
        assert mock_run.call_args.kwargs["workers"] == 8

    def test_explicit_reload_param(self, mocker):
        """reload= explicit param is forwarded to uvicorn."""
        mock_run = mocker.patch("uvicorn.run")
        app = _make_app()
        app.server.run(reload=True)
        assert mock_run.call_args.kwargs["reload"] is True

    def test_explicit_log_level_param(self, mocker):
        """log_level= explicit param is forwarded to uvicorn."""
        mock_run = mocker.patch("uvicorn.run")
        app = _make_app()
        app.server.run(log_level="warning")
        assert mock_run.call_args.kwargs["log_level"] == "warning"

    def test_explicit_ssl_params(self, mocker):
        """ssl_keyfile= and ssl_certfile= explicit params are forwarded."""
        mock_run = mocker.patch("uvicorn.run")
        app = _make_app()
        app.server.run(ssl_keyfile="k.pem", ssl_certfile="c.pem")
        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs["ssl_keyfile"] == "k.pem"
        assert call_kwargs["ssl_certfile"] == "c.pem"

    def test_auto_resolves_import_string_from_main(self, mocker):
        """Without explicit app kwarg, resolves import string from __main__."""
        import sys
        import types

        mock_run = mocker.patch("uvicorn.run")
        app = _make_app()
        fake_main = types.ModuleType("__main__")
        fake_main.my_chat = app
        mocker.patch.dict(sys.modules, {"__main__": fake_main})
        mocker.patch("sys.argv", ["my_script.py"])
        app.server.run()
        assert mock_run.call_args.args[0] == "my_script:my_chat"

    def test_auto_resolve_strips_path(self, mocker):
        """Auto-resolved import string uses just the module stem."""
        import sys
        import types

        mock_run = mocker.patch("uvicorn.run")
        app = _make_app()
        fake_main = types.ModuleType("__main__")
        fake_main.chatbot = app
        mocker.patch.dict(sys.modules, {"__main__": fake_main})
        mocker.patch("sys.argv", ["/full/path/to/server.py"])
        app.server.run()
        assert mock_run.call_args.args[0] == "server:chatbot"

    def test_falls_back_to_asgi_object(self, mocker):
        """Falls back to ASGI app object when not found in __main__."""
        import sys
        import types

        mock_run = mocker.patch("uvicorn.run")
        app = _make_app()
        fake_main = types.ModuleType("__main__")
        fake_main.other_var = "not the app"
        mocker.patch.dict(sys.modules, {"__main__": fake_main})
        app.server.run()
        assert mock_run.call_args.args[0] is app.server.asgi_app

    def test_two_instances_resolves_first(self, mocker):
        """With two Chatnificent instances in __main__, picks the matching one."""
        import sys
        import types

        mock_run = mocker.patch("uvicorn.run")
        app1 = _make_app()
        app2 = _make_app()
        fake_main = types.ModuleType("__main__")
        fake_main.first_app = app1
        fake_main.second_app = app2
        mocker.patch.dict(sys.modules, {"__main__": fake_main})
        mocker.patch("sys.argv", ["server.py"])
        app2.server.run()
        assert mock_run.call_args.args[0] == "server:second_app"


# =============================================================================
# root_path — ASGI mount prefix support
# =============================================================================


def _make_mounted_app(mount_path="/sub", **kwargs):
    """Create a Chatnificent app mounted under a prefix via Starlette Mount."""
    from starlette.applications import Starlette as StarletteApp
    from starlette.routing import Mount

    app = _make_app(**kwargs)
    parent = StarletteApp(routes=[Mount(mount_path, app=app)])
    return app, parent


class TestRootPath:
    """ASGI root_path (mount prefix) is propagated to HTML, JSON paths, and cookies."""

    def test_root_path_injected_in_html(self):
        """Mounted app injects window.__CHATNIFICENT_ROOT__ with the mount prefix."""
        _, parent = _make_mounted_app("/myapp")
        client = TestClient(parent)
        r = client.get("/myapp/")
        assert r.status_code == 200
        assert 'window.__CHATNIFICENT_ROOT__="/myapp"' in r.text

    def test_root_path_not_injected_when_not_mounted(self):
        """Non-mounted app does not inject a __CHATNIFICENT_ROOT__ assignment."""
        client = _client()
        r = client.get("/")
        assert '__CHATNIFICENT_ROOT__="' not in r.text

    def test_root_path_in_chat_response_path(self):
        """Non-streaming POST /api/chat returns path prefixed with mount path."""
        _, parent = _make_mounted_app("/app")
        client = TestClient(parent)
        r = client.post("/app/api/chat", json={"message": "hello"})
        assert r.status_code == 200
        assert r.json()["path"].startswith("/app/")

    def test_root_path_in_stream_done_event(self):
        """Streaming POST /api/chat done event has prefixed path."""
        _, parent = _make_mounted_app("/app", llm=Echo(stream=True))
        client = TestClient(parent)
        with client.stream("POST", "/app/api/chat", json={"message": "hello"}) as r:
            lines = list(r.iter_lines())

        events = []
        for line in lines:
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

        done = next(e for e in events if e["event"] == "done")
        assert done["data"]["path"].startswith("/app/")

    def test_root_path_in_load_conversation_path(self):
        """GET /api/conversations/{id} returns path prefixed with mount path."""
        _, parent = _make_mounted_app("/app")
        client = TestClient(parent)

        r = client.post("/app/api/chat", json={"message": "hello"})
        convo_id = r.json()["conversation_id"]

        r2 = client.get(f"/app/api/conversations/{convo_id}")
        assert r2.status_code == 200
        assert r2.json()["path"].startswith("/app/")

    def test_cookie_scoped_to_mount_path(self):
        """Mounted app sets cookie with Path scoped to the mount prefix."""
        _, parent = _make_mounted_app("/app")
        client = TestClient(parent, cookies={})
        r = client.get("/app/api/conversations", follow_redirects=False)
        set_cookie = r.headers.get("set-cookie", "")
        assert "Path=/app/" in set_cookie

    def test_cookie_path_is_root_when_not_mounted(self):
        """Non-mounted app sets cookie with Path=/."""
        client = TestClient(_make_app().server.asgi_app, cookies={})
        r = client.get("/api/conversations", follow_redirects=False)
        set_cookie = r.headers.get("set-cookie", "")
        assert "Path=/" in set_cookie

    def test_two_mounted_apps_independent_cookies(self):
        """Two apps at different mount points scope cookies independently."""
        from starlette.applications import Starlette as StarletteApp
        from starlette.routing import Mount

        app_a = _make_app()
        app_b = _make_app()
        parent = StarletteApp(
            routes=[Mount("/a", app=app_a), Mount("/b", app=app_b)]
        )
        client = TestClient(parent, cookies={})

        r_a = client.get("/a/api/conversations", follow_redirects=False)
        r_b = client.get("/b/api/conversations", follow_redirects=False)

        cookie_a = r_a.headers.get("set-cookie", "")
        cookie_b = r_b.headers.get("set-cookie", "")
        assert "Path=/a/" in cookie_a
        assert "Path=/b/" in cookie_b
