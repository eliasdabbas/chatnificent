"""Unit tests for the server module."""

from unittest.mock import Mock, patch

import pytest
from chatnificent.server import DashServer, Server


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


class TestDashServer:
    """Test the DashServer implementation."""

    def test_create_server_creates_dash_app(self):
        """DashServer.create_server() creates a Dash application."""
        from chatnificent import Chatnificent

        app = Chatnificent()
        dash_app = app.server.dash_app

        from dash import Dash

        assert isinstance(dash_app, Dash)

    def test_dash_server_has_layout(self):
        """DashServer sets layout from layout_builder."""
        from chatnificent import Chatnificent

        app = Chatnificent()
        assert app.server.dash_app.layout is not None

    def test_dash_server_registers_callbacks(self):
        """DashServer registers callbacks during create_server."""
        with patch("chatnificent.callbacks.register_callbacks") as mock_register:
            from chatnificent import Chatnificent

            app = Chatnificent()
            mock_register.assert_called_once()

    def test_server_pillar_on_chatnificent(self):
        """Chatnificent has a server attribute."""
        from chatnificent import Chatnificent

        app = Chatnificent()
        assert hasattr(app, "server")
        assert isinstance(app.server, DashServer)

    def test_custom_server_injection(self):
        """Custom Server implementation can be injected."""
        from chatnificent import Chatnificent

        class CustomServer(Server):
            def create_server(self, **kwargs):
                self.created = True
                return None

            def run(self, **kwargs):
                pass

        custom = CustomServer()
        app = Chatnificent(server=custom)

        assert app.server is custom
        assert app.server.app is app
        assert custom.created is True

    def test_chatnificent_run_delegates_to_server(self):
        """Chatnificent.run() delegates to server.run()."""

        class SpyServer(Server):
            def __init__(self):
                super().__init__()
                self.run_called = False
                self.run_kwargs = {}

            def create_server(self, **kwargs):
                return None

            def run(self, **kwargs):
                self.run_called = True
                self.run_kwargs = kwargs

        from chatnificent import Chatnificent

        spy = SpyServer()
        app = Chatnificent(server=spy)
        app.run(debug=True, port=9999)

        assert spy.run_called
        assert spy.run_kwargs == {"debug": True, "port": 9999}
