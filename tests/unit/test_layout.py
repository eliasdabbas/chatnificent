"""
Tests for the Layout pillar implementations.

The Layout pillar handles UI rendering, component layout, message formatting,
and theming. It has the most complex interface with required component IDs
that must match callback expectations.

Note: These tests focus on interface compliance and core logic rather than
UI component rendering, since Layout implementations depend on external
libraries (dash_bootstrap_components, dash_mantine_components).
"""

import pytest
from chatnificent.layout import Layout
from chatnificent.models import ChatMessage


class TestLayoutInterface:
    """Test the Layout abstract base class interface."""

    def test_layout_is_abstract(self):
        """Test that Layout cannot be instantiated directly."""
        with pytest.raises(TypeError) as exc_info:
            Layout()

        error_message = str(exc_info.value)
        assert "abstract" in error_message.lower()

    def test_layout_requires_all_methods(self):
        """Test that Layout subclasses must implement all required methods."""

        class IncompleteLayout1(Layout):
            def build_messages(self, messages):
                return []

            def get_external_stylesheets(self):
                return []

        # Missing build_layout
        with pytest.raises(TypeError) as exc_info:
            IncompleteLayout1()

        error_message = str(exc_info.value)
        assert "build_layout" in error_message

        class IncompleteLayout2(Layout):
            def build_layout(self):
                from dash import html

                return html.Div()

            def get_external_stylesheets(self):
                return []

        # Missing build_messages
        with pytest.raises(TypeError) as exc_info:
            IncompleteLayout2()

        error_message = str(exc_info.value)
        assert "build_messages" in error_message

    def test_layout_subclass_with_all_methods_works(self):
        """Test that complete Layout subclasses work correctly."""
        from dash import html

        class MinimalLayout(Layout):
            def build_layout(self):
                return html.Div(
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

            def build_messages(self, messages):
                return [html.Div(msg.content) for msg in messages]

            def get_external_stylesheets(self):
                return []

        # Should work fine
        layout = MinimalLayout()
        assert isinstance(layout, Layout)

        # Test basic functionality
        messages = [ChatMessage(role="user", content="Hello")]
        rendered = layout.build_messages(messages)
        assert len(rendered) == 1

    def test_layout_validation_catches_missing_ids(self):
        """Test that layout validation catches missing required component IDs."""
        from dash import html

        class IncompleteIDLayout(Layout):
            def build_layout(self):
                return html.Div(
                    [
                        html.Div(id="sidebar"),
                        html.Button(id="sidebar_toggle"),
                        # Missing several required IDs
                    ]
                )

            def build_messages(self, messages):
                return []

            def get_external_stylesheets(self):
                return []

        # Should fail during initialization due to missing IDs
        with pytest.raises(ValueError) as exc_info:
            IncompleteIDLayout()

        error_message = str(exc_info.value)
        assert "missing required component id" in error_message.lower()
        # Should mention some of the missing IDs
        missing_ids = [
            "conversations_list",
            "new_conversation_button",
            "chat_area",
            "messages_container",
            "input_textarea",
            "submit_button",
            "status_indicator",
        ]
        assert any(missing_id in error_message for missing_id in missing_ids)


class TestLayoutUtilities:
    """Test Layout utility methods and helper functions."""

    def test_layout_base_utilities(self):
        """Test base Layout utility methods."""
        from dash import html

        class TestLayout(Layout):
            def build_layout(self):
                return html.Div(
                    [
                        html.Div(
                            id="sidebar",
                            style={"backgroundColor": "white"},
                            className="sidebar-class",
                        ),
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

            def build_messages(self, messages):
                return [html.Div(msg.content) for msg in messages]

            def get_external_stylesheets(self):
                return [
                    "https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css",
                    {"href": "https://example.com/style.css", "rel": "stylesheet"},
                ]

        layout = TestLayout()

        # Test style extraction methods
        styles = layout.get_current_styles()
        assert isinstance(styles, dict)
        assert "sidebar" in styles
        assert "style" in styles["sidebar"]
        assert "className" in styles["sidebar"]

        # Test utility methods
        component_keys = layout.get_component_keys()
        assert isinstance(component_keys, set)
        assert "sidebar" in component_keys

        # Test style/class getters
        sidebar_class = layout.get_class_name("sidebar")
        assert sidebar_class == "sidebar-class"

        sidebar_style = layout.get_style("sidebar")
        assert sidebar_style == {"backgroundColor": "white"}

        # Test with non-existent component
        assert layout.get_class_name("nonexistent") is None
        assert layout.get_style("nonexistent") is None

        # Test external stylesheets
        stylesheets = layout.get_external_stylesheets()
        assert isinstance(stylesheets, list)
        assert len(stylesheets) == 2
        assert isinstance(stylesheets[0], str)
        assert isinstance(stylesheets[1], dict)

        # Test external scripts (default implementation)
        scripts = layout.get_external_scripts()
        assert isinstance(scripts, list)
        assert len(scripts) == 0  # Default returns empty list

    def test_rtl_text_detection(self):
        """Test RTL (Right-To-Left) text detection utility."""
        from dash import html

        class TestLayout(Layout):
            def build_layout(self):
                return html.Div(
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

            def build_messages(self, messages):
                return []

            def get_external_stylesheets(self):
                return []

        layout = TestLayout()

        # Test cases for RTL detection
        test_cases = [
            # (text, expected_rtl)
            ("", False),  # Empty string
            ("   ", False),  # Whitespace only
            ("Hello World", False),  # English (LTR)
            ("123456", False),  # Numbers
            ("مرحبا بك", True),  # Arabic (RTL)
            ("שלום עליכם", True),  # Hebrew (RTL)
            ("Hello مرحبا", False),  # Mixed, LTR first should win
            ("مرحبا Hello", True),  # Mixed, RTL first should win
            ("こんにちは", False),  # Japanese (LTR)
            ("Привет", False),  # Russian (LTR)
            ("!@#$%", False),  # Special characters
        ]

        for text, expected_rtl in test_cases:
            result = layout._is_rtl(text)
            assert result == expected_rtl, f"RTL detection failed for '{text}'"


class TestLayoutMessageRendering:
    """Test Layout message rendering functionality."""

    def test_layout_message_contract(self):
        """Test that layout message rendering follows the contract."""
        from dash import html

        class TestLayout(Layout):
            def build_layout(self):
                return html.Div(
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

            def build_messages(self, messages):
                # Simple implementation for testing
                result = []
                for i, msg in enumerate(messages):
                    result.append(
                        html.Div(
                            [
                                html.Strong(f"{msg.role}: "),
                                html.Span(msg.content),
                            ],
                            id=f"message_{i}",
                        )
                    )
                return result

            def get_external_stylesheets(self):
                return []

        layout = TestLayout()

        # Empty messages should return empty list
        result = layout.build_messages([])
        assert isinstance(result, list)
        assert len(result) == 0

        # Single message should return one component
        messages = [ChatMessage(role="user", content="Hello")]
        result = layout.build_messages(messages)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] is not None

        # Multiple messages should return multiple components
        messages = [
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi there!"),
            ChatMessage(role="user", content="How are you?"),
        ]
        result = layout.build_messages(messages)
        assert isinstance(result, list)
        assert len(result) == 3
        for component in result:
            assert component is not None

    def test_layout_message_edge_cases(self):
        """Test message rendering with edge case content."""
        from dash import html

        class TestLayout(Layout):
            def build_layout(self):
                return html.Div(
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

            def build_messages(self, messages):
                return [html.Div(msg.content) for msg in messages]

            def get_external_stylesheets(self):
                return []

        layout = TestLayout()

        # Messages with edge case content
        edge_case_messages = [
            ChatMessage(role="user", content=""),  # Empty content
            ChatMessage(role="assistant", content="   "),  # Whitespace only
            ChatMessage(role="user", content="A" * 10000),  # Very long content
            ChatMessage(
                role="assistant", content="Line 1\nLine 2\nLine 3"
            ),  # Multiline
            ChatMessage(role="user", content="Special chars: @#$%^&*()"),
            ChatMessage(role="assistant", content="Unicode: 你好世界 🌍"),
            ChatMessage(role="user", content="مرحبا بك"),  # RTL text
        ]

        rendered = layout.build_messages(edge_case_messages)
        assert len(rendered) == len(edge_case_messages)

        # All should be renderable components
        for component in rendered:
            assert component is not None


class TestLayoutValidation:
    """Test Layout validation and error handling."""

    def test_layout_validation_comprehensive(self):
        """Test comprehensive layout validation with different missing ID combinations."""
        from dash import html

        required_ids = {
            "sidebar",
            "sidebar_toggle",
            "conversations_list",
            "new_conversation_button",
            "chat_area",
            "messages_container",
            "input_textarea",
            "submit_button",
            "status_indicator",
        }

        class TestLayout(Layout):
            def __init__(self, missing_ids=None):
                self.missing_ids = missing_ids or []
                super().__init__()

            def build_layout(self):
                components = []
                for req_id in required_ids:
                    if req_id not in self.missing_ids:
                        components.append(html.Div(id=req_id))
                return html.Div(components)

            def build_messages(self, messages):
                return []

            def get_external_stylesheets(self):
                return []

        # Test missing single ID
        for missing_id in required_ids:
            with pytest.raises(ValueError) as exc_info:
                TestLayout(missing_ids=[missing_id])

            error_message = str(exc_info.value)
            assert missing_id in error_message

        # Test missing multiple IDs
        with pytest.raises(ValueError) as exc_info:
            TestLayout(missing_ids=["sidebar", "chat_area", "submit_button"])

        error_message = str(exc_info.value)
        assert "sidebar" in error_message
        assert "chat_area" in error_message
        assert "submit_button" in error_message

        # Test all IDs present should work
        complete_layout = TestLayout(missing_ids=[])
        assert isinstance(complete_layout, Layout)

    def test_layout_tree_traversal(self):
        """Test that layout validation correctly traverses component trees."""
        from dash import html

        class NestedLayout(Layout):
            def build_layout(self):
                return html.Div(
                    [
                        html.Div(
                            [
                                html.Div(id="sidebar"),
                                html.Button(id="sidebar_toggle"),
                            ]
                        ),
                        html.Div(
                            [
                                html.Ul(id="conversations_list"),
                                html.Button(id="new_conversation_button"),
                                html.Div(
                                    [
                                        html.Div(id="chat_area"),
                                        html.Div(id="messages_container"),
                                    ]
                                ),
                            ]
                        ),
                        html.Div(
                            [
                                html.Textarea(id="input_textarea"),
                                html.Button(id="submit_button"),
                                html.Div(id="status_indicator"),
                            ]
                        ),
                    ]
                )

            def build_messages(self, messages):
                return []

            def get_external_stylesheets(self):
                return []

        # Should work even with nested structure
        layout = NestedLayout()
        assert isinstance(layout, Layout)

    def test_layout_handles_none_children(self):
        """Test that layout validation handles None children gracefully."""
        from dash import html

        class NoneChildrenLayout(Layout):
            def build_layout(self):
                return html.Div(
                    [
                        html.Div(id="sidebar"),
                        None,  # None child should be handled
                        html.Button(id="sidebar_toggle"),
                        html.Ul(id="conversations_list"),
                        None,  # Another None child
                        html.Button(id="new_conversation_button"),
                        html.Div(id="chat_area"),
                        html.Div(id="messages_container"),
                        html.Textarea(id="input_textarea"),
                        html.Button(id="submit_button"),
                        html.Div(id="status_indicator"),
                    ]
                )

            def build_messages(self, messages):
                return []

            def get_external_stylesheets(self):
                return []

        # Should work even with None children
        layout = NoneChildrenLayout()
        assert isinstance(layout, Layout)


class TestLayoutThemes:
    """Test Layout theme support and configuration."""

    def test_layout_theme_initialization(self):
        """Test that layouts handle theme initialization."""
        from dash import html

        class ThemeableLayout(Layout):
            def __init__(self, theme=None):
                super().__init__(theme)

            def build_layout(self):
                return html.Div(
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

            def build_messages(self, messages):
                return []

            def get_external_stylesheets(self):
                theme_styles = {
                    "light": ["https://cdn.example.com/light.css"],
                    "dark": ["https://cdn.example.com/dark.css"],
                }
                return theme_styles.get(self.theme_name, [])

        # Test default theme (None)
        layout = ThemeableLayout()
        assert layout.theme_name is None
        assert layout.get_external_stylesheets() == []

        # Test light theme
        layout = ThemeableLayout(theme="light")
        assert layout.theme_name == "light"
        stylesheets = layout.get_external_stylesheets()
        assert len(stylesheets) == 1
        assert "light.css" in stylesheets[0]

        # Test dark theme
        layout = ThemeableLayout(theme="dark")
        assert layout.theme_name == "dark"
        stylesheets = layout.get_external_stylesheets()
        assert len(stylesheets) == 1
        assert "dark.css" in stylesheets[0]


class TestLayoutIntegration:
    """Test Layout pillar integration patterns."""

    def test_layout_callback_readiness(self):
        """Test that layout provides what callbacks need."""
        from dash import html

        class CallbackReadyLayout(Layout):
            def build_layout(self):
                return html.Div(
                    [
                        html.Div(id="sidebar"),
                        html.Button(id="sidebar_toggle"),
                        html.Ul(id="conversations_list"),
                        html.Button(id="new_conversation_button"),
                        html.Div(id="chat_area"),
                        html.Div(id="messages_container"),  # Critical for callbacks
                        html.Textarea(id="input_textarea"),  # Critical for callbacks
                        html.Button(id="submit_button"),  # Critical for callbacks
                        html.Div(id="status_indicator"),
                    ]
                )

            def build_messages(self, messages):
                # This is what callbacks will call to render messages
                return [
                    html.Div(
                        f"{msg.role}: {msg.content}", className=f"{msg.role}-message"
                    )
                    for msg in messages
                ]

            def get_external_stylesheets(self):
                return []

        layout = CallbackReadyLayout()

        # Verify layout passed validation (has all required IDs)
        assert isinstance(layout, Layout)

        # Verify message building works as callbacks expect
        messages = [
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi there!"),
        ]
        rendered_messages = layout.build_messages(messages)
        assert len(rendered_messages) == 2

        # Verify each message is a renderable component
        for component in rendered_messages:
            assert component is not None
            assert hasattr(component, "children") or hasattr(component, "id")

    def test_layout_pillar_interface_compliance(self):
        """Test that layout implements the complete pillar interface."""
        from dash import html

        class CompliantLayout(Layout):
            def build_layout(self):
                return html.Div(
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

            def build_messages(self, messages):
                return []

            def get_external_stylesheets(self):
                return []

        layout = CompliantLayout()

        # Test all interface methods exist and are callable
        interface_methods = [
            "build_layout",
            "build_messages",
            "get_external_stylesheets",
            "get_external_scripts",
        ]

        for method_name in interface_methods:
            assert hasattr(layout, method_name)
            assert callable(getattr(layout, method_name))

        # Test utility methods exist
        utility_methods = [
            "get_class_name",
            "get_style",
            "get_component_keys",
            "get_current_styles",
        ]

        for method_name in utility_methods:
            assert hasattr(layout, method_name)
            assert callable(getattr(layout, method_name))
