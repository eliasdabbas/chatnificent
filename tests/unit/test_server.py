"""Unit tests for the Server ABC and DevServer (core, zero dependencies).

Server adapter tests live in tests/unit/server/.
"""

import io
import json
from unittest.mock import Mock, patch

import pytest
from chatnificent.server import DevServer, Server

# =============================================================================
# Server ABC
# =============================================================================


class TestServerBase:
    """Test the abstract Server base class."""

    def test_server_is_abstract(self):
        """Server cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            Server()

    def test_server_with_app_reference(self):
        """Server can be initialized with app reference."""
        mock_app = Mock()

        class ConcreteServer(Server):
            def create_server(self, **kwargs):
                return None

            def run(self, **kwargs):
                pass

        srv = ConcreteServer(mock_app)
        assert srv.app is mock_app

    def test_server_without_app_reference(self):
        """Server can be initialized without app (lazy binding)."""

        class ConcreteServer(Server):
            def create_server(self, **kwargs):
                return None

            def run(self, **kwargs):
                pass

        srv = ConcreteServer()
        assert srv.app is None

        mock_app = Mock()
        srv.app = mock_app
        assert srv.app is mock_app


# =============================================================================
# DevServer
# =============================================================================


class TestDevServer:
    """Test the DevServer implementation."""

    def test_default_host_and_port(self):
        """DevServer defaults to 127.0.0.1:7777."""
        srv = DevServer()
        assert srv._host == "127.0.0.1"
        assert srv._port == 7777

    def test_create_server_stores_config(self):
        """create_server() stores host/port without binding a socket."""
        srv = DevServer()
        srv.create_server(host="0.0.0.0", port=3000)
        assert srv._host == "0.0.0.0"
        assert srv._port == 3000
        assert srv.httpd is None

    def test_devserver_is_default_without_dash(self):
        """Without Dash installed, Chatnificent falls back to DevServer."""
        with patch.dict("sys.modules", {"dash": None}):
            from chatnificent import Chatnificent

            app = Chatnificent()
            assert isinstance(app.server, DevServer)


class TestDevHandler:
    """Test DevServer HTTP handler endpoints."""

    @pytest.fixture
    def app(self):
        """Chatnificent instance with Echo + InMemory + DevServer."""
        from chatnificent import Chatnificent
        from chatnificent.llm import Echo
        from chatnificent.store import InMemory

        return Chatnificent(
            llm=Echo(stream=False), store=InMemory(), server=DevServer()
        )

    @pytest.fixture
    def handler_class(self, app):
        """Return a partially-bound _DevHandler class ready for testing."""
        from functools import partial

        from chatnificent.server import _DevHandler

        return partial(_DevHandler, app)

    def _make_handler(self, handler_class, method, path, body=None):
        """Simulate an HTTP request, returning raw response text."""
        from chatnificent.server import _DevHandler

        raw_body = json.dumps(body).encode("utf-8") if body is not None else b""

        wfile = io.BytesIO()

        with patch.object(_DevHandler, "__init__", lambda self, app, *a, **kw: None):
            h = _DevHandler.__new__(_DevHandler)
            h._app = handler_class.args[0] if hasattr(handler_class, "args") else None
            h._new_session = False
            h._session_id = None
            h.rfile = io.BytesIO(raw_body)
            h.wfile = wfile
            h.requestline = f"{method} {path} HTTP/1.1"
            h.command = method
            h.path = path
            h.headers = {"Content-Length": str(len(raw_body)), "Host": "localhost"}
            h.request_version = "HTTP/1.1"
            h.close_connection = True

            if method == "GET":
                h.do_GET()
            elif method == "POST":
                h.rfile = io.BytesIO(raw_body)
                h.do_POST()

            wfile.seek(0)
            return wfile.read().decode("utf-8", errors="replace")

    def test_get_root_returns_html(self, handler_class):
        """GET / returns the chat UI HTML page."""
        response = self._make_handler(handler_class, "GET", "/")
        assert "200" in response
        assert "text/html" in response
        assert "Chatnificent" in response

    def test_get_index_returns_html(self, handler_class):
        """GET /index.html also returns the chat UI."""
        response = self._make_handler(handler_class, "GET", "/index.html")
        assert "200" in response
        assert "Chatnificent" in response

    def test_get_any_path_serves_page_for_deep_links(self, handler_class):
        """GET to any non-API path serves the page (supports deep links like /001)."""
        response = self._make_handler(handler_class, "GET", "/001")
        assert "200" in response
        assert "Chatnificent" in response

    def test_get_unknown_api_path_returns_404(self, handler_class):
        """GET to unknown /api/ path returns 404."""
        response = self._make_handler(handler_class, "GET", "/api/unknown")
        assert "404" in response

    def test_post_chat_returns_response(self, handler_class):
        """POST /api/chat returns a JSON response with conversation_id."""
        response = self._make_handler(
            handler_class, "POST", "/api/chat", {"message": "hello"}
        )
        assert "200" in response
        assert "conversation_id" in response
        assert "response" in response
        assert "messages" in response

    def test_post_chat_empty_message(self, handler_class):
        """POST /api/chat with empty message returns 400."""
        response = self._make_handler(
            handler_class, "POST", "/api/chat", {"message": ""}
        )
        assert "400" in response

    def test_post_unknown_path_returns_404(self, handler_class):
        """POST to unknown path returns 404."""
        response = self._make_handler(handler_class, "POST", "/unknown")
        assert "404" in response


class TestDevHandlerAuthDelegation:
    """DevServer must delegate identity to the Auth pillar, not manage it directly."""

    @pytest.fixture
    def app(self):
        from chatnificent import Chatnificent
        from chatnificent.llm import Echo
        from chatnificent.store import InMemory

        return Chatnificent(
            llm=Echo(stream=False), store=InMemory(), server=DevServer()
        )

    def _make_handler(self, app, method, path, body=None, cookie=None):
        """Simulate an HTTP request with optional cookie, returning raw response."""
        from chatnificent.server import _DevHandler

        raw_body = json.dumps(body).encode("utf-8") if body is not None else b""
        wfile = io.BytesIO()

        with patch.object(_DevHandler, "__init__", lambda self, _app, *a, **kw: None):
            h = _DevHandler.__new__(_DevHandler)
            h._app = app
            h._new_session = False
            h._session_id = None
            h.rfile = io.BytesIO(raw_body)
            h.wfile = wfile
            h.requestline = f"{method} {path} HTTP/1.1"
            h.command = method
            h.path = path
            headers = {"Content-Length": str(len(raw_body)), "Host": "localhost"}
            if cookie:
                headers["Cookie"] = cookie
            h.headers = headers
            h.request_version = "HTTP/1.1"
            h.close_connection = True

            if method == "GET":
                h.do_GET()
            elif method == "POST":
                h.rfile = io.BytesIO(raw_body)
                h.do_POST()

            wfile.seek(0)
            return wfile.read().decode("utf-8", errors="replace")

    def test_handler_calls_auth_pillar(self, app):
        """_get_user_id should delegate to app.auth, not generate UUIDs directly."""
        app.auth = Mock()
        app.auth.get_current_user_id.return_value = "auth-decided-id"

        self._make_handler(
            app,
            "POST",
            "/api/chat",
            {"message": "hello"},
            cookie="chatnificent_session=my-cookie-val",
        )

        app.auth.get_current_user_id.assert_called()
        call_kwargs = app.auth.get_current_user_id.call_args
        assert call_kwargs.kwargs.get("session_id") == "my-cookie-val"

    def test_handler_passes_none_session_when_no_cookie(self, app):
        """Without a cookie, session_id=None should be passed to auth."""
        app.auth = Mock()
        app.auth.get_current_user_id.return_value = "new-anon-id"

        self._make_handler(app, "POST", "/api/chat", {"message": "hello"})

        app.auth.get_current_user_id.assert_called()
        call_kwargs = app.auth.get_current_user_id.call_args
        assert call_kwargs.kwargs.get("session_id") is None


class TestDevHandlerURLIntegration:
    """DevServer API responses must include paths built by the URL pillar."""

    @pytest.fixture
    def app(self):
        from chatnificent import Chatnificent
        from chatnificent.llm import Echo
        from chatnificent.store import InMemory

        return Chatnificent(
            llm=Echo(stream=False), store=InMemory(), server=DevServer()
        )

    def _make_handler(self, app, method, path, body=None, cookie=None):
        from chatnificent.server import _DevHandler

        raw_body = json.dumps(body).encode("utf-8") if body is not None else b""
        wfile = io.BytesIO()

        with patch.object(_DevHandler, "__init__", lambda self, _app, *a, **kw: None):
            h = _DevHandler.__new__(_DevHandler)
            h._app = app
            h._new_session = False
            h._session_id = None
            h.rfile = io.BytesIO(raw_body)
            h.wfile = wfile
            h.requestline = f"{method} {path} HTTP/1.1"
            h.command = method
            h.path = path
            headers = {"Content-Length": str(len(raw_body)), "Host": "localhost"}
            if cookie:
                headers["Cookie"] = cookie
            h.headers = headers
            h.request_version = "HTTP/1.1"
            h.close_connection = True

            if method == "GET":
                h.do_GET()
            elif method == "POST":
                h.rfile = io.BytesIO(raw_body)
                h.do_POST()

            wfile.seek(0)
            return wfile.read().decode("utf-8", errors="replace")

    def _extract_json(self, raw_response):
        """Extract the JSON body from raw HTTP response text."""
        body_start = raw_response.find("\r\n\r\n")
        if body_start == -1:
            return None
        return json.loads(raw_response[body_start + 4 :])

    def test_chat_response_includes_path(self, app):
        """POST /api/chat response must include 'path' built by URL pillar."""
        response = self._make_handler(
            app,
            "POST",
            "/api/chat",
            {"message": "hi"},
            cookie="chatnificent_session=user1",
        )
        data = self._extract_json(response)
        assert "path" in data, "API response must include 'path' from URL pillar"
        assert data["conversation_id"] in data["path"]

    def test_chat_response_path_matches_url_pillar(self, app):
        """The 'path' in response should match what url.build_conversation_path returns."""
        app.auth = Mock()
        app.auth.get_current_user_id.return_value = "testuser"

        response = self._make_handler(
            app,
            "POST",
            "/api/chat",
            {"message": "hi"},
            cookie="chatnificent_session=testuser",
        )
        data = self._extract_json(response)
        expected_path = app.url.build_conversation_path(
            "testuser", data["conversation_id"]
        )
        assert data["path"] == expected_path

    def test_deep_link_adopts_url_user_id_without_cookie(self, app):
        """Visiting /<user_id>/<convo_id> without a cookie should adopt the URL user_id."""
        from chatnificent.models import Conversation

        # Pre-populate a conversation for user "abc123"
        convo = Conversation(
            id="conv001",
            messages=[
                {"role": "user", "content": "hello"},
            ],
        )
        app.store.save_conversation("abc123", convo)

        # Page load with deep link — no cookie
        self._make_handler(app, "GET", "/abc123/conv001")

        # Now subsequent API call should use the adopted user_id
        response = self._make_handler(
            app,
            "GET",
            "/api/conversations/conv001",
            cookie="chatnificent_session=abc123",
        )
        data = self._extract_json(response)
        assert data.get("id") == "conv001"

    def test_deep_link_sets_session_cookie(self, app):
        """Deep link without cookie should set the session cookie to URL user_id."""
        response = self._make_handler(app, "GET", "/myuser/someconvo")
        assert "chatnificent_session=myuser" in response

    def test_deep_link_does_not_override_existing_cookie(self, app):
        """If cookie already exists, URL user_id should NOT override it."""
        response = self._make_handler(
            app, "GET", "/url_user/conv1", cookie="chatnificent_session=cookie_user"
        )
        assert (
            "chatnificent_session=cookie_user" not in response
            or "chatnificent_session=url_user" not in response
        )


class TestDevHandlerLayoutIntegration:
    @pytest.fixture
    def app(self):
        from chatnificent import Chatnificent
        from chatnificent.llm import Echo
        from chatnificent.store import InMemory

        app = Chatnificent(
            llm=Echo(stream=False),
            store=InMemory(),
            server=DevServer(),
        )
        app.layout.render_messages = Mock(
            return_value=[{"role": "assistant", "content": "Rendered"}]
        )
        app.layout.render_conversations = Mock(
            side_effect=lambda conversations, **kwargs: conversations
        )
        return app

    def _make_handler(self, app, method, path, body=None, cookie=None):
        from chatnificent.server import _DevHandler

        raw_body = json.dumps(body).encode("utf-8") if body is not None else b""
        wfile = io.BytesIO()

        with patch.object(_DevHandler, "__init__", lambda self, _app, *a, **kw: None):
            h = _DevHandler.__new__(_DevHandler)
            h._app = app
            h._new_session = False
            h._session_id = None
            h.rfile = io.BytesIO(raw_body)
            h.wfile = wfile
            h.requestline = f"{method} {path} HTTP/1.1"
            h.command = method
            h.path = path
            headers = {"Content-Length": str(len(raw_body)), "Host": "localhost"}
            if cookie:
                headers["Cookie"] = cookie
            h.headers = headers
            h.request_version = "HTTP/1.1"
            h.close_connection = True

            if method == "GET":
                h.do_GET()
            elif method == "POST":
                h.rfile = io.BytesIO(raw_body)
                h.do_POST()

            wfile.seek(0)
            return wfile.read().decode("utf-8", errors="replace")

    def _extract_json(self, raw_response):
        body_start = raw_response.find("\r\n\r\n")
        if body_start == -1:
            return None
        return json.loads(raw_response[body_start + 4 :])

    def test_post_chat_uses_layout_render_messages(self, app):
        response = self._make_handler(
            app,
            "POST",
            "/api/chat",
            {"message": "hello"},
            cookie="chatnificent_session=user1",
        )

        data = self._extract_json(response)

        app.layout.render_messages.assert_called_once()
        assert data["messages"] == [{"role": "assistant", "content": "Rendered"}]
        assert data["response"] == "Rendered"

    def test_load_conversation_uses_layout_render_messages(self, app):
        from chatnificent.models import Conversation

        app.store.save_conversation(
            "user1",
            Conversation(
                id="conv1",
                messages=[
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "world"},
                ],
            ),
        )

        response = self._make_handler(
            app,
            "GET",
            "/api/conversations/conv1",
            cookie="chatnificent_session=user1",
        )
        data = self._extract_json(response)

        app.layout.render_messages.assert_called_once()
        assert data["messages"] == [{"role": "assistant", "content": "Rendered"}]

    def test_list_conversations_uses_layout_render_conversations(self, app):
        from chatnificent.models import Conversation

        app.store.save_conversation(
            "user1",
            Conversation(
                id="conv1",
                messages=[{"role": "user", "content": "first title"}],
            ),
        )

        self._make_handler(
            app,
            "GET",
            "/api/conversations",
            cookie="chatnificent_session=user1",
        )

        app.layout.render_conversations.assert_called_once()

    def test_list_conversations_title_truncated_with_ellipsis(self, app):
        from chatnificent.models import Conversation

        long_msg = "A" * 50
        app.store.save_conversation(
            "user1",
            Conversation(
                id="conv1",
                messages=[{"role": "user", "content": long_msg}],
            ),
        )

        response = self._make_handler(
            app,
            "GET",
            "/api/conversations",
            cookie="chatnificent_session=user1",
        )
        data = self._extract_json(response)

        title = data["conversations"][0]["title"]
        assert title == "A" * 30 + "…"

    def test_list_conversations_short_title_no_ellipsis(self, app):
        from chatnificent.models import Conversation

        app.store.save_conversation(
            "user1",
            Conversation(
                id="conv1",
                messages=[{"role": "user", "content": "short"}],
            ),
        )

        response = self._make_handler(
            app,
            "GET",
            "/api/conversations",
            cookie="chatnificent_session=user1",
        )
        data = self._extract_json(response)

        title = data["conversations"][0]["title"]
        assert title == "short"
        assert "…" not in title


# =============================================================================
# Server Parity Contract Tests
# =============================================================================


class TestServerParityContract:
    """Verify that DevServer and DashServer follow the same pillar contracts.

    These tests document the expected caller contract for each server
    and surface known divergences.
    """

    def _make_dev_handler(self, app, method, path, body=None, cookie=None):
        """Simulate a DevServer HTTP request, returning the handler instance."""
        from chatnificent.server import _DevHandler

        raw_body = json.dumps(body).encode("utf-8") if body is not None else b""
        wfile = io.BytesIO()

        with patch.object(_DevHandler, "__init__", lambda self, _app, *a, **kw: None):
            h = _DevHandler.__new__(_DevHandler)
            h._app = app
            h._new_session = False
            h._session_id = None
            h.rfile = io.BytesIO(raw_body)
            h.wfile = wfile
            h.requestline = f"{method} {path} HTTP/1.1"
            h.command = method
            h.path = path
            headers = {"Content-Length": str(len(raw_body)), "Host": "localhost"}
            if cookie:
                headers["Cookie"] = cookie
            h.headers = headers
            h.request_version = "HTTP/1.1"
            h.close_connection = True

            if method == "POST":
                h.rfile = io.BytesIO(raw_body)
                h.do_POST()
            elif method == "GET":
                h.do_GET()

            return h, wfile

    def test_devserver_passes_session_id_to_auth(self):
        """DevServer passes session_id kwarg from cookie to auth.get_current_user_id."""
        from chatnificent import Chatnificent
        from chatnificent.llm import Echo
        from chatnificent.store import InMemory

        app = Chatnificent(llm=Echo(stream=False), store=InMemory(), server=DevServer())
        app.auth = Mock()
        app.auth.get_current_user_id.return_value = "user1"

        self._make_dev_handler(
            app,
            "POST",
            "/api/chat",
            {"message": "hello"},
            cookie="chatnificent_session=session_abc",
        )

        app.auth.get_current_user_id.assert_called()
        call_kwargs = app.auth.get_current_user_id.call_args
        assert call_kwargs.kwargs.get("session_id") == "session_abc"

    def test_dashserver_auth_parity_achieved(self):
        """DashServer now passes session_id to auth, matching DevServer.

        See tests/unit/server/test_dash.py::TestDashServerAuthParity for
        the full behavioral and structural contract tests.
        """
        dash = pytest.importorskip(
            "dash", reason="DashServer parity test requires dash"
        )
        import ast
        import inspect

        from chatnificent import _callbacks

        source = inspect.getsource(_callbacks.register_callbacks)
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "get_current_user_id"
            ):
                kwarg_names = [kw.arg for kw in node.keywords]
                assert "session_id" in kwarg_names, (
                    f"Line {node.lineno}: get_current_user_id() missing session_id"
                )

    def test_devserver_url_build_path_contract(self):
        """DevServer uses url.build_conversation_path for response paths."""
        from chatnificent import Chatnificent
        from chatnificent.llm import Echo
        from chatnificent.store import InMemory

        app = Chatnificent(llm=Echo(stream=False), store=InMemory(), server=DevServer())
        app.url = Mock()
        app.url.parse.return_value = Mock(user_id=None, convo_id=None)
        app.url.build_conversation_path.return_value = "/user1/conv1"

        _, wfile = self._make_dev_handler(
            app,
            "POST",
            "/api/chat",
            {"message": "hello"},
            cookie="chatnificent_session=user1",
        )

        app.url.build_conversation_path.assert_called_once()
        wfile.seek(0)
        raw = wfile.read().decode("utf-8", errors="replace")
        body_start = raw.find("\r\n\r\n")
        data = json.loads(raw[body_start + 4 :])
        assert data["path"] == "/user1/conv1"

    def test_both_servers_produce_same_path_for_same_inputs(self):
        """url.build_conversation_path produces identical results regardless of server."""
        from chatnificent.url import PathBased, QueryParams

        for impl in [PathBased(), QueryParams()]:
            path = impl.build_conversation_path("user_abc", "conv_xyz")
            path2 = impl.build_conversation_path("user_abc", "conv_xyz")
            assert path == path2, (
                f"Deterministic path building failed for {type(impl).__name__}"
            )
            assert "conv_xyz" in path


# =============================================================================
# Shared Server Helpers — base class methods
# =============================================================================


class _StubServer(Server):
    """Minimal concrete Server for testing base-class helpers."""

    def create_server(self, **kwargs):
        return None

    def run(self, **kwargs):
        pass


class TestBuildConversationTitle:
    """Server._build_conversation_title extracts a display title from a Conversation."""

    def _server(self):
        return _StubServer()

    def test_short_message_used_as_title(self):
        from chatnificent.models import Conversation

        convo = Conversation(
            id="c1", messages=[{"role": "user", "content": "Hello world"}]
        )
        assert self._server()._build_conversation_title(convo) == "Hello world"

    def test_long_message_truncated_with_ellipsis(self):
        from chatnificent.models import Conversation

        long_msg = "a" * 50
        convo = Conversation(id="c1", messages=[{"role": "user", "content": long_msg}])
        title = self._server()._build_conversation_title(convo)
        assert title == "a" * 30 + "…"
        assert len(title) == 31

    def test_exactly_30_chars_no_ellipsis(self):
        from chatnificent.models import Conversation

        msg = "x" * 30
        convo = Conversation(id="c1", messages=[{"role": "user", "content": msg}])
        assert self._server()._build_conversation_title(convo) == msg

    def test_falls_back_to_convo_id(self):
        """Conversation with no user messages falls back to id."""
        from chatnificent.models import Conversation

        convo = Conversation(
            id="abc123", messages=[{"role": "assistant", "content": "hi"}]
        )
        assert self._server()._build_conversation_title(convo) == "abc123"

    def test_empty_messages_falls_back_to_id(self):
        from chatnificent.models import Conversation

        convo = Conversation(id="abc123", messages=[])
        assert self._server()._build_conversation_title(convo) == "abc123"

    def test_whitespace_only_content_falls_back_to_id(self):
        from chatnificent.models import Conversation

        convo = Conversation(id="c1", messages=[{"role": "user", "content": "   "}])
        assert self._server()._build_conversation_title(convo) == "c1"

    def test_non_string_content_falls_back_to_id(self):
        from chatnificent.models import Conversation

        convo = Conversation(
            id="c1", messages=[{"role": "user", "content": [{"type": "text"}]}]
        )
        assert self._server()._build_conversation_title(convo) == "c1"

    def test_strips_leading_trailing_whitespace(self):
        from chatnificent.models import Conversation

        convo = Conversation(
            id="c1", messages=[{"role": "user", "content": "  hello  "}]
        )
        assert self._server()._build_conversation_title(convo) == "hello"

    def test_skips_system_messages(self):
        """First message might be system — title comes from first user message."""
        from chatnificent.models import Conversation

        convo = Conversation(
            id="c1",
            messages=[
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "actual question"},
            ],
        )
        assert self._server()._build_conversation_title(convo) == "actual question"


class TestExtractLastResponse:
    """Server._extract_last_response finds last assistant content from display messages."""

    def _server(self):
        return _StubServer()

    def test_returns_last_assistant_content(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "first"},
            {"role": "user", "content": "again"},
            {"role": "assistant", "content": "second"},
        ]
        assert self._server()._extract_last_response(msgs) == "second"

    def test_empty_when_no_assistant(self):
        msgs = [{"role": "user", "content": "hello"}]
        assert self._server()._extract_last_response(msgs) == ""

    def test_empty_list(self):
        assert self._server()._extract_last_response([]) == ""

    def test_skips_none_content(self):
        msgs = [
            {"role": "assistant", "content": "real"},
            {"role": "assistant", "content": None},
        ]
        assert self._server()._extract_last_response(msgs) == "real"

    def test_skips_whitespace_only_content(self):
        msgs = [
            {"role": "assistant", "content": "real"},
            {"role": "assistant", "content": "   "},
        ]
        assert self._server()._extract_last_response(msgs) == "real"


class TestIsLlmStreaming:
    """Server._is_llm_streaming checks the LLM's streaming configuration."""

    def _server_with_llm(self, llm):
        srv = _StubServer()
        srv.app = Mock()
        srv.app.llm = llm
        return srv

    def test_true_when_default_params_stream(self):
        llm = Mock()
        llm.default_params = {"stream": True}
        llm._streaming = False
        assert self._server_with_llm(llm)._is_llm_streaming() is True

    def test_true_when_streaming_flag(self):
        llm = Mock()
        llm.default_params = {}
        llm._streaming = True
        assert self._server_with_llm(llm)._is_llm_streaming() is True

    def test_false_by_default(self):
        llm = Mock(spec=[])  # no default_params, no _streaming
        assert self._server_with_llm(llm)._is_llm_streaming() is False

    def test_false_when_stream_explicit_false(self):
        llm = Mock()
        llm.default_params = {"stream": False}
        del llm._streaming
        assert self._server_with_llm(llm)._is_llm_streaming() is False


class TestRenderMessages:
    """Server._render_messages delegates to the Layout pillar."""

    def test_delegates_to_layout(self):
        from chatnificent.models import Conversation

        srv = _StubServer()
        srv.app = Mock()
        srv.app.layout.render_messages.return_value = [
            {"role": "assistant", "content": "filtered"}
        ]

        convo = Conversation(
            id="c1",
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "filtered"},
            ],
        )
        result = srv._render_messages("user1", convo)

        srv.app.layout.render_messages.assert_called_once_with(
            convo.messages,
            user_id="user1",
            convo_id="c1",
            conversation=convo,
        )
        assert result == [{"role": "assistant", "content": "filtered"}]


class TestRenderConversations:
    """Server._render_conversations delegates to the Layout pillar."""

    def test_delegates_to_layout(self):
        srv = _StubServer()
        srv.app = Mock()
        convos = [{"id": "c1", "title": "Hello"}]
        srv.app.layout.render_conversations.return_value = convos

        result = srv._render_conversations("user1", convos)

        srv.app.layout.render_conversations.assert_called_once_with(
            convos, user_id="user1"
        )
        assert result == convos
