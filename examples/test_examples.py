# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "chatnificent[dev]",
#     "pytest",
# ]
# ///
"""
Automated tests for all Chatnificent examples.

Run with::

    uv run pytest examples/test_examples.py -v
"""

import importlib
import sys
from pathlib import Path

import chatnificent as chat
import pytest

EXAMPLES_DIR = Path(__file__).parent


def _import_example(name: str):
    """Import an example module by filename (without .py)."""
    spec = importlib.util.spec_from_file_location(name, EXAMPLES_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Tier 1 — Basics
# ---------------------------------------------------------------------------


class TestQuickstart:
    def test_app_instantiates(self):
        mod = _import_example("quickstart")
        assert mod.app.llm is not None

    def test_responds(self):
        mod = _import_example("quickstart")
        convo = mod.app.engine.handle_message(
            "hello", user_id="test", convo_id_from_url=None
        )
        assert len(convo.messages) >= 2
        assert convo.messages[-1]["content"]


class TestLlmProviders:
    def test_app_uses_openai(self):
        mod = _import_example("llm_providers")
        assert isinstance(mod.app.llm, chat.llm.OpenAI)

    def test_openai_responds(self):
        mod = _import_example("llm_providers")
        convo = mod.app.engine.handle_message(
            "Say only the word 'pong'.",
            user_id="test",
            convo_id_from_url=None,
        )
        assert len(convo.messages) >= 2
        assert convo.messages[-1]["content"]

    def test_streaming(self):
        mod = _import_example("llm_providers")
        events = list(
            mod.app.engine.handle_message_stream(
                "Say only the word 'pong'.",
                user_id="test",
                convo_id_from_url=None,
            )
        )
        event_types = {e["event"] for e in events}
        assert "delta" in event_types
        assert "done" in event_types


class TestOllamaLocal:
    def test_app_uses_ollama(self):
        mod = _import_example("ollama_local")
        assert isinstance(mod.app.llm, chat.llm.Ollama)
        assert mod.app.llm.model == "llama3.2"

    def test_ollama_responds(self):
        mod = _import_example("ollama_local")
        convo = mod.app.engine.handle_message(
            "Say only the word 'pong'.",
            user_id="test",
            convo_id_from_url=None,
        )
        assert len(convo.messages) >= 2


class TestOpenrouterModels:
    def test_app_uses_openrouter(self):
        mod = _import_example("openrouter_models")
        assert isinstance(mod.app.llm, chat.llm.OpenRouter)

    def test_openrouter_responds(self):
        mod = _import_example("openrouter_models")
        convo = mod.app.engine.handle_message(
            "Say only the word 'pong'.",
            user_id="test",
            convo_id_from_url=None,
        )
        assert len(convo.messages) >= 2
        assert convo.messages[-1]["content"]


# ---------------------------------------------------------------------------
# Tier 2 — Features
# ---------------------------------------------------------------------------


class TestPersistentStorage:
    def test_file_store(self, tmp_path):
        app = chat.Chatnificent(store=chat.store.File(base_dir=str(tmp_path)))
        convo = app.engine.handle_message(
            "hello", user_id="test", convo_id_from_url=None
        )
        loaded = app.store.load_conversation("test", convo.id)
        assert loaded is not None
        assert len(loaded.messages) >= 2

    def test_sqlite_store(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        app = chat.Chatnificent(store=chat.store.SQLite(db_path=db_path))
        convo = app.engine.handle_message(
            "hello", user_id="test", convo_id_from_url=None
        )
        loaded = app.store.load_conversation("test", convo.id)
        assert loaded is not None
        assert len(loaded.messages) >= 2


class TestToolCalling:
    def test_app_has_tools(self):
        mod = _import_example("tool_calling")
        assert isinstance(mod.app.tools, chat.tools.PythonTool)
        tool_defs = mod.app.tools.get_tools()
        tool_names = {t["function"]["name"] for t in tool_defs}
        assert "get_weather" in tool_names
        assert "roll_dice" in tool_names

    def test_tool_execution(self):
        mod = _import_example("tool_calling")
        convo = mod.app.engine.handle_message(
            "What's the weather in Tokyo?",
            user_id="test",
            convo_id_from_url=None,
        )
        assert len(convo.messages) >= 2
        assert convo.messages[-1]["content"]


class TestSystemPrompt:
    def test_app_uses_custom_llm(self):
        mod = _import_example("system_prompt")
        assert isinstance(mod.app.llm, mod.PirateAI)
        assert isinstance(mod.app.llm, chat.llm.OpenAI)

    def test_system_prompt_injected(self):
        mod = _import_example("system_prompt")
        convo = mod.app.engine.handle_message(
            "What is Python?",
            user_id="test",
            convo_id_from_url=None,
        )
        assert len(convo.messages) >= 2
        assert convo.messages[-1]["content"]


class TestMultiToolAgent:
    def test_app_has_four_tools(self):
        mod = _import_example("multi_tool_agent")
        tool_defs = mod.app.tools.get_tools()
        tool_names = {t["function"]["name"] for t in tool_defs}
        assert len(tool_names) == 4
        assert "get_weather" in tool_names
        assert "roll_dice" in tool_names
        assert "get_current_time" in tool_names
        assert "calculate_bmi" in tool_names

    def test_tool_execution(self):
        mod = _import_example("multi_tool_agent")
        convo = mod.app.engine.handle_message(
            "Roll a 20-sided die.",
            user_id="test",
            convo_id_from_url=None,
        )
        assert len(convo.messages) >= 2
        assert convo.messages[-1]["content"]


class TestSingleUser:
    def test_app_uses_single_user_auth(self):
        mod = _import_example("single_user")
        assert isinstance(mod.app.auth, chat.auth.SingleUser)
        assert mod.app.auth.get_current_user_id() == "elias"

    def test_app_uses_sqlite_store(self):
        mod = _import_example("single_user")
        assert isinstance(mod.app.store, chat.store.SQLite)
