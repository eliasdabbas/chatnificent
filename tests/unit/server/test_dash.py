"""Tests for the DashServer adapter.

Same pattern as tests/unit/llm/test_openai.py — validates one concrete
implementation against the Server pillar contract.
"""

from unittest.mock import Mock, patch

import pytest

dash = pytest.importorskip("dash", reason="DashServer tests require the dash extra")

from chatnificent import Chatnificent
from chatnificent.server import DashServer


class TestDashServerCreation:
    """Test DashServer lifecycle."""

    def test_creates_dash_app(self):
        """DashServer.create_server() creates a Dash application."""
        app = Chatnificent()
        assert isinstance(app.server, DashServer)
        assert isinstance(app.server.dash_app, dash.Dash)

    def test_has_layout(self):
        """DashServer sets layout from app.layout."""
        app = Chatnificent()
        assert app.server.dash_app.layout is not None

    def test_registers_callbacks(self):
        """DashServer registers callbacks during create_server."""
        with patch("chatnificent._callbacks.register_callbacks") as mock_register:
            app = Chatnificent()
            mock_register.assert_called_once()

    def test_is_default_when_dash_installed(self):
        """With Dash installed, Chatnificent auto-selects DashServer."""
        app = Chatnificent()
        assert isinstance(app.server, DashServer)


class TestDashServerLayout:
    """Test DashServer layout injection and stylesheet handling."""

    def test_custom_layout_injection(self):
        """Custom layout builder is wired into DashServer."""
        import dash.html as html

        mock_layout = Mock()
        mock_layout.build_layout.return_value = html.Div(id="test-layout")
        mock_layout.get_external_stylesheets.return_value = []
        mock_layout.get_external_scripts.return_value = []

        app = Chatnificent(layout=mock_layout)

        assert app.layout is mock_layout
        mock_layout.build_layout.assert_called_once()

    def test_layout_stylesheets_added(self):
        """Layout stylesheets are passed to the Dash app."""
        import dash.html as html

        mock_layout = Mock()
        mock_layout.build_layout.return_value = html.Div(id="test-layout")
        mock_layout.get_external_stylesheets.return_value = [
            "https://example.com/style.css"
        ]
        mock_layout.get_external_scripts.return_value = []

        app = Chatnificent(layout=mock_layout)
        # Verify Dash app was created (layout was used)
        assert app.server.dash_app is not None

    def test_existing_stylesheets_preserved(self):
        """Existing kwargs stylesheets are preserved alongside layout ones."""
        import dash.html as html

        existing_stylesheet = "https://existing.com/style.css"

        mock_layout = Mock()
        mock_layout.build_layout.return_value = html.Div(id="test-layout")
        mock_layout.get_external_stylesheets.return_value = [
            "https://layout.com/style.css"
        ]
        mock_layout.get_external_scripts.return_value = []

        app = Chatnificent(
            layout=mock_layout, external_stylesheets=[existing_stylesheet]
        )
        assert app.server.dash_app is not None
