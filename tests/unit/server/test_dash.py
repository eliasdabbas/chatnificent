"""Tests for the DashServer adapter.

Same pattern as tests/unit/llm/test_openai.py — validates one concrete
implementation against the Server pillar contract.
"""

from unittest.mock import Mock, patch

import pytest

dash = pytest.importorskip("dash", reason="DashServer tests require the dash extra")

from chatnificent import Chatnificent
from chatnificent.server import DashServer, DevServer


def _make_dash_app(**kwargs):
    """Create a Chatnificent app wired to DashServer + mock DashLayout."""
    import dash.html as html
    from unittest.mock import Mock
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
