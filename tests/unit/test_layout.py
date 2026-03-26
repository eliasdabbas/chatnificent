"""Tests for the HTML Layout defaults used by DevServer."""

from unittest.mock import Mock

from chatnificent.layout import Layout


class ConcreteLayout(Layout):
    """Concrete Layout for testing the default DevServer seams."""

    def render_page(self) -> str:
        return "<html></html>"


class TestLayoutRenderMessages:
    def test_render_messages_filters_system_tool_and_empty_messages(self):
        layout = ConcreteLayout()
        layout.app = Mock()
        layout.app.llm.is_tool_message.side_effect = (
            lambda message: message.get("role") == "tool"
        )

        messages = [
            {"role": "system", "content": "hidden"},
            {"role": "tool", "content": "hidden tool"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "World"},
        ]

        rendered = layout.render_messages(messages)

        assert rendered == [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "World"},
        ]

    def test_render_messages_returns_shallow_copies(self):
        layout = ConcreteLayout()
        layout.app = Mock()
        layout.app.llm.is_tool_message.return_value = False

        messages = [{"role": "assistant", "content": "Original"}]

        rendered = layout.render_messages(messages)
        rendered[0]["content"] = "Changed"

        assert messages[0]["content"] == "Original"
        assert rendered[0] is not messages[0]

    def test_render_messages_serializes_non_string_content_for_display(self):
        layout = ConcreteLayout()
        layout.app = Mock()
        layout.app.llm.is_tool_message.return_value = False

        rendered = layout.render_messages(
            [{"role": "assistant", "content": {"summary": "hello"}}]
        )

        assert rendered == [
            {"role": "assistant", "content": '{"summary": "hello"}'}
        ]

    def test_render_messages_skips_tool_detection_without_app(self):
        layout = ConcreteLayout()

        rendered = layout.render_messages(
            [{"role": "assistant", "content": "Visible"}]
        )

        assert rendered == [{"role": "assistant", "content": "Visible"}]


class TestLayoutRenderConversations:
    def test_render_conversations_returns_shallow_copies(self):
        layout = ConcreteLayout()

        conversations = [{"id": "abc123", "title": "First title"}]

        rendered = layout.render_conversations(conversations)
        rendered[0]["title"] = "Changed"

        assert conversations[0]["title"] == "First title"
        assert rendered[0] is not conversations[0]
