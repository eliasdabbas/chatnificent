"""Integration tests for UI Interactions: Layout → Engine → LLM kwargs."""

from unittest.mock import patch

import pytest
from chatnificent import Chatnificent
from chatnificent.layout import Control, DefaultLayout
from chatnificent.llm import Echo
from chatnificent.store import InMemory


class TestUiInteractionsIntegration:
    """Full-stack: register control → set value → engine passes kwarg to LLM."""

    @pytest.fixture
    def layout(self):
        lo = DefaultLayout()
        lo.register_control(
            Control(
                id="max-tokens",
                html='<input id="max-tokens" type="number" value="256">',
                slot="toolbar",
                llm_param="max_completion_tokens",
                cast=int,
            )
        )
        return lo

    @pytest.fixture
    def app(self, layout):
        return Chatnificent(
            layout=layout,
            llm=Echo(stream=False),
            store=InMemory(),
        )

    def test_get_llm_kwargs_returns_empty_before_any_set(self, layout):
        """No state set → get_llm_kwargs returns empty dict."""
        result = layout.get_llm_kwargs("user1")
        assert result == {}

    def test_get_llm_kwargs_returns_cast_value_after_set(self, layout):
        """set_control_value → get_llm_kwargs returns cast kwarg."""
        layout.set_control_value("user1", "max-tokens", "512")
        result = layout.get_llm_kwargs("user1")
        assert result == {"max_completion_tokens": 512}
        assert isinstance(result["max_completion_tokens"], int)

    def test_engine_passes_llm_kwargs_to_llm(self, app):
        """Engine calls LLM with extra kwargs from layout control state."""
        app.layout.set_control_value("user1", "max-tokens", "100")

        captured = {}

        original_generate = app.llm.generate_response

        def capturing_generate(messages, **kwargs):
            captured.update(kwargs)
            return original_generate(messages, **kwargs)

        with patch.object(app.llm, "generate_response", side_effect=capturing_generate):
            app.engine.handle_message("hello", "user1", None)

        assert captured.get("max_completion_tokens") == 100

    def test_engine_stream_passes_llm_kwargs_to_llm(self, app):
        """Streaming engine path also forwards layout kwargs to LLM."""
        app.layout.set_control_value("user1", "max-tokens", "200")
        app.llm._streaming = True

        captured = {}
        original_generate = app.llm.generate_response

        def capturing_generate(messages, **kwargs):
            captured.update(kwargs)
            return original_generate(messages, **kwargs)

        with patch.object(app.llm, "generate_response", side_effect=capturing_generate):
            list(app.engine.handle_message_stream("hello", "user1", None))

        assert captured.get("max_completion_tokens") == 200

    def test_user_isolation(self, layout):
        """Two users get independent control values."""
        layout.set_control_value("alice", "max-tokens", "100")
        layout.set_control_value("bob", "max-tokens", "200")
        assert layout.get_llm_kwargs("alice") == {"max_completion_tokens": 100}
        assert layout.get_llm_kwargs("bob") == {"max_completion_tokens": 200}

    def test_no_cast_passes_raw_string(self):
        """Control without cast passes raw string value."""
        lo = DefaultLayout()
        lo.register_control(
            Control(
                id="model",
                html='<select id="model"><option>gpt-4o</option></select>',
                slot="toolbar",
                llm_param="model",
            )
        )
        lo.set_control_value("user1", "model", "gpt-4o")
        result = lo.get_llm_kwargs("user1")
        assert result == {"model": "gpt-4o"}
        assert isinstance(result["model"], str)
