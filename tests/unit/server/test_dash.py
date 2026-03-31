"""Tests for the DashServer adapter.

Same pattern as tests/unit/llm/test_openai.py — validates one concrete
implementation against the Server pillar contract.
"""

import ast
import inspect
from unittest.mock import Mock, patch

import pytest

dash = pytest.importorskip("dash", reason="DashServer tests require the dash extra")

import flask
from chatnificent import Chatnificent
from chatnificent.server import DashServer, DevServer


def _make_dash_app(**kwargs):
    """Create a Chatnificent app wired to DashServer + mock DashLayout."""
    from unittest.mock import Mock

    import dash.html as html
    from chatnificent.layout import DashLayout

    mock_layout = Mock(spec=DashLayout)
    mock_layout.build_layout.return_value = html.Div(
        [
            html.Div(id="sidebar"),
            html.Button(id="sidebar_toggle"),
            html.Ul(id="conversations_list"),
            html.Button(id="new_conversation_button"),
            html.Div(id="chat_area"),
            html.Div(id="messages_container"),
            html.Textarea(id="input_textarea"),
            html.Button(id="submit_button"),
            html.Div(id="status_indicator"),
        ]
    )
    mock_layout.get_external_stylesheets.return_value = []
    mock_layout.get_external_scripts.return_value = []
    return Chatnificent(server=DashServer(), layout=mock_layout, **kwargs)


class TestDashServerCreation:
    """Test DashServer lifecycle."""

    def test_creates_dash_app(self):
        """DashServer.create_server() creates a Dash application."""
        app = _make_dash_app()
        assert isinstance(app.server, DashServer)
        assert isinstance(app.server.dash_app, dash.Dash)

    def test_has_layout(self):
        """DashServer sets layout from app.layout."""
        app = _make_dash_app()
        assert app.server.dash_app.layout is not None

    def test_registers_callbacks(self):
        """DashServer registers callbacks during create_server."""
        with patch("chatnificent._callbacks.register_callbacks") as mock_register:
            app = _make_dash_app()
            mock_register.assert_called_once()

    def test_is_default_when_no_server_specified(self):
        """Without an explicit server, Chatnificent defaults to DevServer."""
        app = Chatnificent()
        assert isinstance(app.server, DevServer)


class TestDashServerLayout:
    """Test DashServer layout injection and stylesheet handling."""

    def test_custom_layout_injection(self):
        """Custom layout builder is wired into DashServer."""
        import dash.html as html
        from chatnificent.layout import DashLayout

        mock_layout = Mock(spec=DashLayout)
        mock_layout.build_layout.return_value = html.Div(id="test-layout")
        mock_layout.get_external_stylesheets.return_value = []
        mock_layout.get_external_scripts.return_value = []

        app = Chatnificent(server=DashServer(), layout=mock_layout)

        assert app.layout is mock_layout
        mock_layout.build_layout.assert_called_once()

    def test_layout_stylesheets_added(self):
        """Layout stylesheets are passed to the Dash app."""
        import dash.html as html
        from chatnificent.layout import DashLayout

        mock_layout = Mock(spec=DashLayout)
        mock_layout.build_layout.return_value = html.Div(id="test-layout")
        mock_layout.get_external_stylesheets.return_value = [
            "https://example.com/style.css"
        ]
        mock_layout.get_external_scripts.return_value = []

        app = Chatnificent(server=DashServer(), layout=mock_layout)
        # Verify Dash app was created (layout was used)
        assert app.server.dash_app is not None

    def test_existing_stylesheets_preserved(self):
        """Existing kwargs stylesheets are preserved alongside layout ones."""
        import dash.html as html
        from chatnificent.layout import DashLayout

        existing_stylesheet = "https://existing.com/style.css"

        mock_layout = Mock(spec=DashLayout)
        mock_layout.build_layout.return_value = html.Div(id="test-layout")
        mock_layout.get_external_stylesheets.return_value = [
            "https://layout.com/style.css"
        ]
        mock_layout.get_external_scripts.return_value = []

        app = Chatnificent(
            server=DashServer(),
            layout=mock_layout,
            external_stylesheets=[existing_stylesheet],
        )
        assert app.server.dash_app is not None


class TestDashServerAuthParity:
    """DashServer callbacks must pass session_id to auth, matching DevServer."""

    def test_get_session_id_reads_cookie(self):
        """_get_session_id extracts chatnificent_session cookie from Flask request."""
        from chatnificent._callbacks import _get_session_id

        flask_app = flask.Flask(__name__)
        with flask_app.test_request_context(
            headers={"Cookie": "chatnificent_session=abc123"}
        ):
            assert _get_session_id() == "abc123"

    def test_get_session_id_returns_none_without_cookie(self):
        """_get_session_id returns None when no session cookie exists."""
        from chatnificent._callbacks import _get_session_id

        flask_app = flask.Flask(__name__)
        with flask_app.test_request_context():
            assert _get_session_id() is None

    def test_all_auth_calls_include_session_id(self):
        """Every auth.get_current_user_id() call in register_callbacks must pass session_id.

        Structural contract test: DevServer reads the session cookie and passes
        session_id= to auth. DashServer must do the same at every call site.
        """
        from chatnificent import _callbacks

        source = inspect.getsource(_callbacks.register_callbacks)
        tree = ast.parse(source)

        violations = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "get_current_user_id"
            ):
                kwarg_names = [kw.arg for kw in node.keywords]
                if "session_id" not in kwarg_names:
                    violations.append(node.lineno)

        assert not violations, (
            f"get_current_user_id() called without session_id= at relative lines: {violations}"
        )

    def test_no_pathname_kwarg_passed_to_auth(self):
        """auth.get_current_user_id() must never receive pathname= (wrong kwarg)."""
        from chatnificent import _callbacks

        source = inspect.getsource(_callbacks.register_callbacks)
        tree = ast.parse(source)

        violations = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "get_current_user_id"
            ):
                kwarg_names = [kw.arg for kw in node.keywords]
                if "pathname" in kwarg_names:
                    violations.append(node.lineno)

        assert not violations, (
            f"get_current_user_id() called with pathname= (wrong kwarg) at relative lines: {violations}"
        )
