"""Unit tests for Chatnificent initialization and configuration.

These are CORE tests — they must pass with zero optional dependencies.
Server adapter tests live in tests/unit/server/,
LLM adapter tests live in tests/unit/llm/.
"""

from unittest.mock import Mock, patch

import pytest
from chatnificent import Chatnificent
from chatnificent.engine import Engine, Synchronous
from chatnificent.llm import Echo
from chatnificent.server import DevServer, Server
from chatnificent.store import InMemory


class TestChatnificentInit:
    """Test Chatnificent initialization and pillar configuration."""

    def test_default_initialization(self):
        """Test Chatnificent initializes with all pillar attributes."""
        app = Chatnificent()

        assert hasattr(app, "llm")
        assert hasattr(app, "store")
        assert hasattr(app, "layout_builder")
        assert hasattr(app, "engine")
        assert hasattr(app, "auth")
        assert hasattr(app, "tools")
        assert hasattr(app, "retrieval")
        assert hasattr(app, "url")
        assert hasattr(app, "server")

    def test_custom_llm_initialization(self):
        """Test custom LLM provider injection."""
        mock_llm = Mock()
        app = Chatnificent(llm=mock_llm)

        assert app.llm is mock_llm

    def test_custom_store_initialization(self):
        """Test custom store injection."""
        mock_store = Mock()
        app = Chatnificent(store=mock_store)

        assert app.store is mock_store

    def test_custom_server_injection(self):
        """Test custom server injection."""

        class StubServer(Server):
            def create_server(self, **kwargs):
                return None

            def run(self, **kwargs):
                pass

        stub = StubServer()
        app = Chatnificent(server=stub)

        assert app.server is stub
        assert app.server.app is app

    def test_echo_llm_fallback(self):
        """Test fallback to Echo LLM when OpenAI not available."""
        with patch.dict("sys.modules", {"openai": None}):
            with patch("warnings.warn") as mock_warn:
                app = Chatnificent()

                assert isinstance(app.llm, Echo)
                mock_warn.assert_called_once()

    def test_devserver_fallback_without_dash(self):
        """Without Dash, Chatnificent falls back to DevServer."""
        with patch.dict("sys.modules", {"dash": None}):
            app = Chatnificent()
            assert isinstance(app.server, DevServer)


class TestEngineInitialization:
    """Test engine initialization and lazy binding."""

    def test_default_engine_initialization(self):
        """Test default Synchronous engine is created."""
        app = Chatnificent()

        assert isinstance(app.engine, Synchronous)
        assert app.engine.app is app

    def test_custom_engine_instance_with_lazy_binding(self):
        """Test passing engine instance with lazy binding."""
        custom_engine = Synchronous()
        assert custom_engine.app is None

        app = Chatnificent(engine=custom_engine)

        assert app.engine is custom_engine
        assert custom_engine.app is app

    def test_custom_engine_subclass_with_lazy_binding(self):
        """Test custom engine subclass with lazy binding."""

        class CustomEngine(Synchronous):
            def __init__(self, app=None):
                super().__init__(app)
                self.custom_attr = "test"

        custom_engine = CustomEngine()
        assert custom_engine.app is None
        assert custom_engine.custom_attr == "test"

        app = Chatnificent(engine=custom_engine)

        assert app.engine is custom_engine
        assert custom_engine.app is app
        assert custom_engine.custom_attr == "test"

    def test_engine_with_pre_existing_app_reference(self):
        """Test engine that already has app reference."""
        mock_app = Mock()
        custom_engine = Synchronous(mock_app)

        new_app = Chatnificent(engine=custom_engine)

        assert custom_engine.app is new_app
        assert custom_engine.app is not mock_app


class TestPillarIntegration:
    """Test that pillars are properly integrated."""

    def test_all_pillars_accessible_by_engine(self):
        """Test engine can access all pillars through app reference."""
        mock_llm = Mock()
        mock_store = Mock()
        mock_auth = Mock()
        mock_tools = Mock()
        mock_retrieval = Mock()
        mock_url = Mock()

        custom_engine = Synchronous()

        app = Chatnificent(
            llm=mock_llm,
            store=mock_store,
            auth=mock_auth,
            tools=mock_tools,
            retrieval=mock_retrieval,
            url=mock_url,
            engine=custom_engine,
        )

        assert custom_engine.app.llm is mock_llm
        assert custom_engine.app.store is mock_store
        assert custom_engine.app.auth is mock_auth
        assert custom_engine.app.tools is mock_tools
        assert custom_engine.app.retrieval is mock_retrieval
        assert custom_engine.app.url is mock_url

    def test_run_delegates_to_server(self):
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

        spy = SpyServer()
        app = Chatnificent(server=spy)
        app.run(debug=True, port=9999)

        assert spy.run_called
        assert spy.run_kwargs == {"debug": True, "port": 9999}


class TestBackwardCompatibility:
    """Test backward compatibility considerations."""

    def test_engine_parameter_accepts_instance(self):
        """Test new 'engine' parameter works correctly."""
        custom_engine = Synchronous()
        app = Chatnificent(engine=custom_engine)

        assert app.engine is custom_engine
        assert custom_engine.app is app
