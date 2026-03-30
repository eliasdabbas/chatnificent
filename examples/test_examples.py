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
import json
import sys
from pathlib import Path
from types import SimpleNamespace

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


def _openai_response(content: str):
    """Build a minimal OpenAI-compatible response object for tests."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=None),
                finish_reason="stop",
            )
        ]
    )


def _openai_stream(*deltas: str):
    """Build minimal OpenAI-compatible streaming chunks for tests."""
    return [
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=delta))])
        for delta in deltas
    ]


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
    def test_app_uses_openai(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mod = _import_example("llm_providers")
        assert isinstance(mod.app.llm, chat.llm.OpenAI)

    def test_openai_responds(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mod = _import_example("llm_providers")
        mod.app.llm.generate_response = lambda *args, **kwargs: _openai_response("pong")
        convo = mod.app.engine.handle_message(
            "Say only the word 'pong'.",
            user_id="test",
            convo_id_from_url=None,
        )
        assert len(convo.messages) >= 2
        assert convo.messages[-1]["content"] == "pong"

    def test_streaming(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mod = _import_example("llm_providers")
        mod.app.llm.generate_response = lambda *args, **kwargs: _openai_stream(
            "po", "ng"
        )
        mod.app.store.save_raw_api_response = lambda *args, **kwargs: None
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
        assert "".join(e["data"] for e in events if e["event"] == "delta") == "pong"


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


class TestUsageDisplay:
    def test_usage_display_augments_assistant_messages(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        mod = _import_example("usage_display")
        mod.app.store = chat.store.InMemory()
        convo = chat.models.Conversation(
            id="usage-convo",
            messages=[
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
        )

        mod.app.store.save_raw_api_response(
            "test",
            convo.id,
            [
                {"object": "chat.completion.chunk", "usage": None},
                {
                    "object": "chat.completion.chunk",
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 20,
                        "total_tokens": 30,
                    },
                },
            ],
        )

        rendered = mod.app.layout.render_messages(
            convo.messages,
            user_id="test",
            convo_id=convo.id,
            conversation=convo,
        )

        assert "Usage: ↑ 10 + ↓ 20 = 30 Tokens" in rendered[-1]["content"]


class TestUsageDisplayMultiProvider:
    @pytest.mark.parametrize(
        ("raw_response", "expected_line"),
        [
            (
                [
                    {"object": "chat.completion.chunk", "usage": None},
                    {
                        "object": "chat.completion.chunk",
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 20,
                            "total_tokens": 30,
                        },
                    },
                ],
                "Usage: ↑ 10 + ↓ 20 = 30 Tokens",
            ),
            (
                [
                    {
                        "type": "message_start",
                        "message": {"usage": {"input_tokens": 8, "output_tokens": 0}},
                    },
                    {
                        "type": "message_delta",
                        "usage": {"input_tokens": 8, "output_tokens": 12},
                    },
                ],
                "Usage: ↑ 8 + ↓ 12 = 20 Tokens",
            ),
            (
                {
                    "usage_metadata": {
                        "prompt_token_count": 7,
                        "candidates_token_count": 9,
                        "total_token_count": 16,
                    }
                },
                "Usage: ↑ 7 + ↓ 9 = 16 Tokens",
            ),
        ],
    )
    def test_usage_display_multi_provider_augments_supported_payloads(
        self, monkeypatch, raw_response, expected_line
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        mod = _import_example("usage_display_multi_provider")
        mod.app.store = chat.store.InMemory()
        convo = chat.models.Conversation(
            id="usage-multi-convo",
            messages=[
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
        )

        mod.app.store.save_raw_api_response("test", convo.id, raw_response)

        rendered = mod.app.layout.render_messages(
            convo.messages,
            user_id="test",
            convo_id=convo.id,
            conversation=convo,
        )

        assert expected_line in rendered[-1]["content"]


class TestConversationTitle:
    def test_title_file_is_written_and_used_for_sidebar_rendering(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        mod = _import_example("conversation_title")
        mod.app.store = chat.store.InMemory()
        mod.app.llm.generate_response = lambda *args, **kwargs: {
            "content": "Solar System"
        }
        mod.app.llm.extract_content = lambda response: response["content"]
        mod.app.llm.create_assistant_message = lambda response: {
            "role": "assistant",
            "content": "Assistant reply",
        }
        mod.app.llm.parse_tool_calls = lambda response: None
        mod.app.llm.is_tool_message = lambda message: False

        convo = mod.app.engine.handle_message(
            "summarize the solar system",
            user_id="test",
            convo_id_from_url=None,
        )

        title_bytes = mod.app.store.load_file(
            "test", convo.id, "conversation_title.txt"
        )
        assert title_bytes is not None

        rendered = mod.app.layout.render_conversations(
            [{"id": convo.id, "title": convo.id}],
            user_id="test",
        )

        assert rendered[0]["title"] != convo.id
        assert len(rendered[0]["title"]) <= 40


class TestConversationSummary:
    def test_conversation_summary_is_written_and_rendered(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        mod = _import_example("conversation_summary")
        mod.app.store = chat.store.InMemory()

        def fake_generate_response(messages, **kwargs):
            if messages and messages[0].get("role") == "system":
                return {
                    "content": "## Key Takeaways\n\n- The solar system has eight planets."
                }
            return {"content": "Assistant reply"}

        mod.app.llm.generate_response = fake_generate_response
        mod.app.llm.extract_content = lambda response: response["content"]
        mod.app.llm.create_assistant_message = lambda response: {
            "role": "assistant",
            "content": response["content"],
        }
        mod.app.llm.parse_tool_calls = lambda response: None
        mod.app.llm.is_tool_message = lambda message: False

        convo = mod.app.engine.handle_message(
            "summarize the solar system",
            user_id="test",
            convo_id_from_url=None,
        )

        summary_bytes = mod.app.store.load_file("test", convo.id, "summaries.md")
        assert summary_bytes is not None

        convo = mod.app.engine.handle_message(
            "add one more detail",
            user_id="test",
            convo_id_from_url=convo.id,
        )

        summary_bytes = mod.app.store.load_file("test", convo.id, "summaries.md")
        assert summary_bytes is not None
        assert summary_bytes.decode("utf-8").count("<details>") == 2

        rendered = mod.app.layout.render_messages(
            convo.messages,
            user_id="test",
            convo_id=convo.id,
            conversation=convo,
        )

        assert "Conversation Summary" in rendered[0]["content"]
        assert "Key Takeaways" in rendered[0]["content"]


class TestWebSearch:
    def test_web_search_renders_sources_from_raw_responses(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        mod = _import_example("web_search")
        mod.app.store = chat.store.InMemory()

        grounded_response = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Spain won Euro 2024."}],
                    },
                    "groundingMetadata": {
                        "webSearchQueries": ["who won euro 2024"],
                        "groundingChunks": [
                            {
                                "web": {
                                    "uri": "https://www.uefa.com/euro2024/",
                                    "title": "UEFA Euro 2024",
                                }
                            },
                            {
                                "web": {
                                    "uri": "https://www.bbc.com/sport/football",
                                    "title": "BBC Sport",
                                }
                            },
                        ],
                        "groundingSupports": [
                            {
                                "segment": {
                                    "text": "**Spain** won Euro 2024 in Berlin."
                                },
                                "groundingChunkIndices": [0],
                            }
                        ],
                    },
                }
            ]
        }

        mod.app.llm.generate_response = lambda *args, **kwargs: grounded_response
        mod.app.llm.extract_content = lambda response: response["candidates"][0][
            "content"
        ]["parts"][0]["text"]
        mod.app.llm.create_assistant_message = lambda response: {
            "role": "assistant",
            "content": response["candidates"][0]["content"]["parts"][0]["text"],
        }
        mod.app.llm.parse_tool_calls = lambda response: None
        mod.app.llm.is_tool_message = lambda message: False

        convo = mod.app.engine.handle_message(
            "Who won Euro 2024?",
            user_id="test",
            convo_id_from_url=None,
        )

        raw_responses = mod.app.store.load_raw_api_responses("test", convo.id)
        assert raw_responses == [grounded_response]

        rendered = mod.app.layout.render_messages(
            convo.messages,
            user_id="test",
            convo_id=convo.id,
            conversation=convo,
        )

        assert "<details>" in rendered[-1]["content"]
        assert "<summary>🔍 Sources (2)</summary>" in rendered[-1]["content"]
        assert (
            "**[UEFA Euro 2024](https://www.uefa.com/euro2024/)**"
            in rendered[-1]["content"]
        )
        assert "https://www.uefa.com/euro2024/" in rendered[-1]["content"]
        assert "**Spain** won Euro 2024 in Berlin." in rendered[-1]["content"]


class TestDisplayRedaction:
    def test_sensitive_data_is_redacted_in_display_only(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        mod = _import_example("display_redaction")
        mod.app.store = chat.store.InMemory()
        page = mod.app.layout.render_page()

        assert "loadConvo(convoId, true);" not in page

        response_text = (
            "We then emailed robin@gmail.com, called (555) 123-9876, "
            "and charged 4111 1111 1111 8742. "
            "Backup contact was riley@protonmail.com on +1 555-222-4444, "
            "with test card 4242-4242-4242-1111."
        )
        mod.app.llm.generate_response = lambda *args, **kwargs: {
            "content": response_text
        }
        mod.app.llm.extract_content = lambda response: response["content"]
        mod.app.llm.create_assistant_message = lambda response: {
            "role": "assistant",
            "content": response["content"],
        }
        mod.app.llm.parse_tool_calls = lambda response: None
        mod.app.llm.is_tool_message = lambda message: False

        convo = mod.app.engine.handle_message(
            "Rewrite this customer support note for me.",
            user_id="test",
            convo_id_from_url=None,
        )

        rendered = mod.app.layout.render_messages(
            convo.messages,
            user_id="test",
            convo_id=convo.id,
            conversation=convo,
        )

        assert "robin@gmail.com" in convo.messages[-1]["content"]
        assert "(555) 123-9876" in convo.messages[-1]["content"]
        assert "4111 1111 1111 8742" in convo.messages[-1]["content"]
        assert "riley@protonmail.com" in convo.messages[-1]["content"]
        assert "+1 555-222-4444" in convo.messages[-1]["content"]
        assert "4242-4242-4242-1111" in convo.messages[-1]["content"]

        assert "robin@gmail.com" not in rendered[-1]["content"]
        assert "(555) 123-9876" not in rendered[-1]["content"]
        assert "4111 1111 1111 8742" not in rendered[-1]["content"]
        assert "riley@protonmail.com" not in rendered[-1]["content"]
        assert "+1 555-222-4444" not in rendered[-1]["content"]
        assert "4242-4242-4242-1111" not in rendered[-1]["content"]

        assert "r****@g****.com" in rendered[-1]["content"]
        assert "XXXX9876" in rendered[-1]["content"]
        assert "XXXX8742" in rendered[-1]["content"]
        assert "r****@p****.com" in rendered[-1]["content"]
        assert "XXXX4444" in rendered[-1]["content"]
        assert "XXXX1111" in rendered[-1]["content"]
