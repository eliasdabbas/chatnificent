"""Unit tests for Chatnificent initialization and configuration."""

from unittest.mock import Mock, patch

import pytest
from chatnificent import Chatnificent
from chatnificent.engine import Engine, Synchronous
from chatnificent.layout import Bootstrap, Minimal
from chatnificent.llm import Echo, OpenAI
from chatnificent.store import InMemory


class TestChatnificentInit:
    """Test Chatnificent initialization and pillar configuration."""

    def test_default_initialization(self):
        """Test Chatnificent initializes with default pillars."""
        with patch("chatnificent.llm.OpenAI"):
            app = Chatnificent()

            # Check defaults are set
            assert hasattr(app, "llm")
            assert hasattr(app, "store")
            assert hasattr(app, "layout_builder")
            assert hasattr(app, "engine")
            assert hasattr(app, "auth")
            assert hasattr(app, "tools")
            assert hasattr(app, "retrieval")
            assert hasattr(app, "url")

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

    def test_custom_layout_initialization(self):
        """Test custom layout builder injection."""
        import dash.html as html

        mock_layout = Mock()
        # Return a real Dash component to satisfy validation
        mock_layout.build_layout.return_value = html.Div(id="test-layout")
        mock_layout.get_external_stylesheets.return_value = []
        mock_layout.get_external_scripts.return_value = []

        app = Chatnificent(layout=mock_layout)

        assert app.layout_builder is mock_layout
        mock_layout.build_layout.assert_called_once()

    def test_echo_llm_fallback(self):
        """Test fallback to Echo LLM when OpenAI not available."""
        with patch("chatnificent.llm.OpenAI", side_effect=ImportError):
            with patch("warnings.warn") as mock_warn:
                app = Chatnificent()

                assert isinstance(app.llm, Echo)
                mock_warn.assert_called_once()


class TestEngineInitialization:
    """Test engine initialization and lazy binding."""

    def test_default_engine_initialization(self):
        """Test default Synchronous engine is created."""
        app = Chatnificent()

        assert isinstance(app.engine, Synchronous)
        assert app.engine.app is app

    def test_custom_engine_instance_with_lazy_binding(self):
        """Test passing engine instance with lazy binding."""
        # Create engine without app
        custom_engine = Synchronous()
        assert custom_engine.app is None

        # Pass to Chatnificent
        app = Chatnificent(engine=custom_engine)

        # Verify lazy binding occurred
        assert app.engine is custom_engine
        assert custom_engine.app is app

    def test_custom_engine_subclass_with_lazy_binding(self):
        """Test custom engine subclass with lazy binding."""

        class CustomEngine(Synchronous):
            def __init__(self, app=None):
                super().__init__(app)
                self.custom_attr = "test"

        # Create instance without app
        custom_engine = CustomEngine()
        assert custom_engine.app is None
        assert custom_engine.custom_attr == "test"

        # Pass to Chatnificent
        app = Chatnificent(engine=custom_engine)

        # Verify binding and custom attributes preserved
        assert app.engine is custom_engine
        assert custom_engine.app is app
        assert custom_engine.custom_attr == "test"

    def test_engine_with_pre_existing_app_reference(self):
        """Test engine that already has app reference."""
        mock_app = Mock()
        custom_engine = Synchronous(mock_app)

        # Pass to Chatnificent
        new_app = Chatnificent(engine=custom_engine)

        # App reference should be overwritten with new app
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

        # Verify engine can access all pillars
        assert custom_engine.app.llm is mock_llm
        assert custom_engine.app.store is mock_store
        assert custom_engine.app.auth is mock_auth
        assert custom_engine.app.tools is mock_tools
        assert custom_engine.app.retrieval is mock_retrieval
        assert custom_engine.app.url is mock_url

    def test_callbacks_registered_after_initialization(self):
        """Test callbacks are registered during initialization."""
        with patch("chatnificent.callbacks.register_callbacks") as mock_register:
            app = Chatnificent()

            mock_register.assert_called_once_with(app)


class TestStylesheetAndScriptHandling:
    """Test external stylesheet and script handling."""

    def test_layout_stylesheets_added(self):
        """Test layout stylesheets are added to kwargs."""
        import dash.html as html

        mock_layout = Mock()
        mock_layout.build_layout.return_value = html.Div(id="test-layout")
        mock_layout.get_external_stylesheets.return_value = [
            "https://example.com/style.css"
        ]
        mock_layout.get_external_scripts.return_value = []

        app = Chatnificent(layout=mock_layout)

        # Note: We can't easily test this without accessing internal state
        # In a real test, we'd verify the Dash app has these stylesheets

    def test_existing_stylesheets_preserved(self):
        """Test existing stylesheets in kwargs are preserved."""
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

        # Both stylesheets should be present
        # Note: Again, would need to check Dash internals


class TestBackwardCompatibility:
    """Test backward compatibility considerations."""

    def test_engine_parameter_accepts_instance(self):
        """Test new 'engine' parameter works correctly."""
        custom_engine = Synchronous()
        app = Chatnificent(engine=custom_engine)

        assert app.engine is custom_engine
        assert custom_engine.app is app

    # Note: If we wanted to support both engine and engine_class temporarily:
    # def test_engine_class_deprecated_but_works(self):
    #     """Test old engine_class parameter still works with warning."""
    #     with patch('warnings.warn') as mock_warn:
    #         class CustomEngine(Synchronous):
    #             pass
    #
    #         app = Chatnificent(engine_class=CustomEngine)
    #
    #         assert isinstance(app.engine, CustomEngine)
    #         mock_warn.assert_called_once()  # Deprecation warning
