"""Tests for the HTML Layout defaults used by DevServer."""

import threading
from unittest.mock import Mock, patch

import pytest
from chatnificent.layout import Control, DefaultLayout, Layout


class ConcreteLayout(Layout):
    """Concrete Layout for testing the default DevServer seams."""

    def render_page(self) -> str:
        return "<html></html>"


class TestLayoutRenderMessages:
    def test_render_messages_filters_system_tool_and_empty_messages(self):
        layout = ConcreteLayout()
        layout.app = Mock()
        layout.app.llm.is_tool_message.side_effect = lambda message: (
            message.get("role") == "tool"
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

        assert rendered == [{"role": "assistant", "content": '{"summary": "hello"}'}]

    def test_render_messages_skips_tool_detection_without_app(self):
        layout = ConcreteLayout()

        rendered = layout.render_messages([{"role": "assistant", "content": "Visible"}])

        assert rendered == [{"role": "assistant", "content": "Visible"}]


class TestLayoutRenderConversations:
    def test_render_conversations_returns_shallow_copies(self):
        layout = ConcreteLayout()

        conversations = [{"id": "abc123", "title": "First title"}]

        rendered = layout.render_conversations(conversations)
        rendered[0]["title"] = "Changed"

        assert conversations[0]["title"] == "First title"
        assert rendered[0] is not conversations[0]


# =====================================================================
# Control dataclass tests
# =====================================================================


class TestControl:
    def test_control_required_fields(self):
        c = Control(id="x", html="<input>", slot="toolbar", llm_param="temperature")
        assert c.id == "x"
        assert c.html == "<input>"
        assert c.slot == "toolbar"
        assert c.llm_param == "temperature"
        assert c.cast is None

    def test_control_with_cast(self):
        c = Control(
            id="x", html="<input>", slot="toolbar", llm_param="max_tokens", cast=int
        )
        assert c.cast is int


# =====================================================================
# Layout ABC no-op defaults
# =====================================================================


class TestLayoutControlNoOps:
    def test_register_control_is_no_op(self):
        layout = ConcreteLayout()
        control = Control(
            id="x", html="<input>", slot="toolbar", llm_param="temperature"
        )
        layout.register_control(control)  # should not raise

    def test_set_control_value_is_no_op(self):
        layout = ConcreteLayout()
        layout.set_control_value("user1", "x", "0.7")  # should not raise

    def test_get_control_values_returns_empty(self):
        layout = ConcreteLayout()
        assert layout.get_control_values("user1") == {}

    def test_get_llm_kwargs_returns_empty(self):
        layout = ConcreteLayout()
        assert layout.get_llm_kwargs("user1") == {}


# =====================================================================
# DefaultLayout control state API
# =====================================================================


class TestDefaultLayoutControls:
    def test_register_control_stores_control(self):
        layout = DefaultLayout()
        c = Control(
            id="tok", html="<select>", slot="toolbar", llm_param="max_completion_tokens"
        )
        layout.register_control(c)
        assert layout.get_control_values("u1") == {}  # no value set yet

    def test_set_control_value_stores_per_user(self):
        layout = DefaultLayout()
        c = Control(
            id="tok", html="<select>", slot="toolbar", llm_param="max_completion_tokens"
        )
        layout.register_control(c)
        layout.set_control_value("u1", "tok", "500")
        assert layout.get_control_values("u1") == {"tok": "500"}

    def test_set_control_value_isolated_between_users(self):
        layout = DefaultLayout()
        c = Control(
            id="tok", html="<select>", slot="toolbar", llm_param="max_completion_tokens"
        )
        layout.register_control(c)
        layout.set_control_value("u1", "tok", "100")
        layout.set_control_value("u2", "tok", "200")
        assert layout.get_control_values("u1") == {"tok": "100"}
        assert layout.get_control_values("u2") == {"tok": "200"}

    def test_set_control_value_upserts(self):
        layout = DefaultLayout()
        c = Control(
            id="tok", html="<select>", slot="toolbar", llm_param="max_completion_tokens"
        )
        layout.register_control(c)
        layout.set_control_value("u1", "tok", "100")
        layout.set_control_value("u1", "tok", "200")
        assert layout.get_control_values("u1") == {"tok": "200"}

    def test_get_control_values_unknown_user_returns_empty(self):
        layout = DefaultLayout()
        assert layout.get_control_values("nobody") == {}

    def test_get_llm_kwargs_maps_llm_param(self):
        layout = DefaultLayout()
        c = Control(
            id="tok", html="<select>", slot="toolbar", llm_param="max_completion_tokens"
        )
        layout.register_control(c)
        layout.set_control_value("u1", "tok", "500")
        assert layout.get_llm_kwargs("u1") == {"max_completion_tokens": "500"}

    def test_get_llm_kwargs_applies_cast(self):
        layout = DefaultLayout()
        c = Control(
            id="tok",
            html="<select>",
            slot="toolbar",
            llm_param="max_completion_tokens",
            cast=int,
        )
        layout.register_control(c)
        layout.set_control_value("u1", "tok", "500")
        result = layout.get_llm_kwargs("u1")
        assert result == {"max_completion_tokens": 500}
        assert isinstance(result["max_completion_tokens"], int)

    def test_get_llm_kwargs_skips_unset_controls(self):
        layout = DefaultLayout()
        c1 = Control(
            id="tok",
            html="<select>",
            slot="toolbar",
            llm_param="max_completion_tokens",
            cast=int,
        )
        c2 = Control(
            id="temp",
            html="<input>",
            slot="toolbar",
            llm_param="temperature",
            cast=float,
        )
        layout.register_control(c1)
        layout.register_control(c2)
        layout.set_control_value("u1", "tok", "500")
        result = layout.get_llm_kwargs("u1")
        assert "max_completion_tokens" in result
        assert "temperature" not in result

    def test_get_llm_kwargs_multiple_controls(self):
        layout = DefaultLayout()
        layout.register_control(
            Control(
                id="tok",
                html="",
                slot="toolbar",
                llm_param="max_completion_tokens",
                cast=int,
            )
        )
        layout.register_control(
            Control(
                id="temp", html="", slot="toolbar", llm_param="temperature", cast=float
            )
        )
        layout.set_control_value("u1", "tok", "300")
        layout.set_control_value("u1", "temp", "0.5")
        result = layout.get_llm_kwargs("u1")
        assert result == {"max_completion_tokens": 300, "temperature": 0.5}

    def test_get_llm_kwargs_unknown_user_returns_empty(self):
        layout = DefaultLayout()
        c = Control(
            id="tok", html="", slot="toolbar", llm_param="max_completion_tokens"
        )
        layout.register_control(c)
        assert layout.get_llm_kwargs("nobody") == {}

    def test_get_control_values_returns_copy(self):
        layout = DefaultLayout()
        layout.register_control(
            Control(id="x", html="", slot="toolbar", llm_param="model")
        )
        layout.set_control_value("u1", "x", "gpt-4o")
        values = layout.get_control_values("u1")
        values["x"] = "tampered"
        assert layout.get_control_values("u1") == {"x": "gpt-4o"}

    def test_thread_safety_concurrent_set_and_get(self):
        layout = DefaultLayout()
        layout.register_control(
            Control(
                id="tok",
                html="",
                slot="toolbar",
                llm_param="max_completion_tokens",
                cast=int,
            )
        )
        errors = []

        def writer(user_id, value):
            try:
                for _ in range(50):
                    layout.set_control_value(user_id, "tok", str(value))
            except Exception as e:
                errors.append(e)

        def reader(user_id):
            try:
                for _ in range(50):
                    layout.get_llm_kwargs(user_id)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=("u1", 100)),
            threading.Thread(target=writer, args=("u2", 200)),
            threading.Thread(target=reader, args=("u1",)),
            threading.Thread(target=reader, args=("u2",)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []


# =====================================================================
# DefaultLayout render_page slot injection
# =====================================================================


class TestDefaultLayoutSlotInjection:
    def test_render_page_contains_chatinteraction_function(self):
        layout = DefaultLayout()
        html = layout.render_page()
        assert "chatInteraction" in html

    def test_render_page_injects_control_html_at_slot(self):
        layout = DefaultLayout()
        layout.register_control(
            Control(
                id="tok",
                html='<select id="tok"></select>',
                slot="toolbar",
                llm_param="max_completion_tokens",
            )
        )
        html = layout.render_page()
        assert '<select id="tok"></select>' in html

    def test_render_page_no_controls_removes_markers(self):
        layout = DefaultLayout()
        html = layout.render_page()
        assert "<!-- SLOT:" not in html

    def test_render_page_multiple_controls_same_slot(self):
        layout = DefaultLayout()
        layout.register_control(
            Control(id="a", html="<span>A</span>", slot="toolbar", llm_param="model")
        )
        layout.register_control(
            Control(
                id="b", html="<span>B</span>", slot="toolbar", llm_param="temperature"
            )
        )
        html = layout.render_page()
        assert "<span>A</span>" in html
        assert "<span>B</span>" in html

    def test_render_page_control_injected_in_correct_slot(self):
        layout = DefaultLayout()
        layout.register_control(
            Control(id="x", html="<b>SIDEBAR</b>", slot="sidebar", llm_param="model")
        )
        html = layout.render_page()
        assert "<b>SIDEBAR</b>" in html


class TestGetLlmKwargsNullSentinel:
    def test_none_value_skips_param(self):
        """set_control_value with None clears the param from get_llm_kwargs."""
        layout = DefaultLayout()
        layout.register_control(
            Control(
                id="tok",
                html="",
                slot="toolbar",
                llm_param="max_completion_tokens",
                cast=int,
            )
        )
        layout.set_control_value("user1", "tok", "100")
        layout.set_control_value("user1", "tok", None)
        assert layout.get_llm_kwargs("user1") == {}

    def test_empty_string_is_not_skipped(self):
        """Empty string is a legitimate value and is not skipped."""
        layout = DefaultLayout()
        layout.register_control(
            Control(id="style", html="", slot="toolbar", llm_param="style")
        )
        layout.set_control_value("user1", "style", "")
        assert layout.get_llm_kwargs("user1") == {"style": ""}


class TestDefaultLayoutControlInit:
    def test_no_controls_no_init_script(self):
        """With no registered controls, no controls chatInteraction init block is injected."""
        layout = DefaultLayout()
        html = layout.render_page()
        assert "chatInteraction(el)" not in html

    def test_one_control_injects_init_script(self):
        """A registered control gets a DOMContentLoaded chatInteraction call."""
        layout = DefaultLayout()
        layout.register_control(
            Control(
                id="tok",
                html="<select id='tok'></select>",
                slot="toolbar",
                llm_param="max_completion_tokens",
            )
        )
        html = layout.render_page()
        assert "DOMContentLoaded" in html
        assert "chatInteraction" in html
        assert "tok" in html

    def test_multiple_controls_all_appear_in_init_script(self):
        """Every registered control id appears in the init script."""
        layout = DefaultLayout()
        layout.register_control(
            Control(
                id="ctrl-a",
                html="<select id='ctrl-a'></select>",
                slot="toolbar",
                llm_param="model",
            )
        )
        layout.register_control(
            Control(
                id="ctrl-b",
                html="<input id='ctrl-b'>",
                slot="sidebar",
                llm_param="temperature",
            )
        )
        html = layout.render_page()
        assert "ctrl-a" in html
        assert "ctrl-b" in html
        # welcome-message script + controls init script = 2 DOMContentLoaded blocks
        assert html.count("DOMContentLoaded") == 2


# =====================================================================
# DefaultLayout constructor controls= parameter
# =====================================================================


class TestDefaultLayoutControlsConstructorParam:
    def test_controls_registered_via_constructor(self):
        """Controls passed to the constructor are registered and usable."""
        c = Control(
            id="tok",
            html="<select>",
            slot="toolbar",
            llm_param="max_completion_tokens",
            cast=int,
        )
        layout = DefaultLayout(controls=[c])
        layout.set_control_value("u1", "tok", "500")
        assert layout.get_llm_kwargs("u1") == {"max_completion_tokens": 500}

    def test_multiple_controls_via_constructor(self):
        """Multiple controls passed to the constructor are all registered."""
        c1 = Control(
            id="tok",
            html="",
            slot="toolbar",
            llm_param="max_completion_tokens",
            cast=int,
        )
        c2 = Control(
            id="temp", html="", slot="toolbar", llm_param="temperature", cast=float
        )
        layout = DefaultLayout(controls=[c1, c2])
        layout.set_control_value("u1", "tok", "200")
        layout.set_control_value("u1", "temp", "0.7")
        assert layout.get_llm_kwargs("u1") == {
            "max_completion_tokens": 200,
            "temperature": 0.7,
        }

    def test_no_controls_arg_is_backward_compatible(self):
        """DefaultLayout() with no arguments still works as before."""
        layout = DefaultLayout()
        assert layout.get_llm_kwargs("u1") == {}

    def test_empty_controls_list_is_valid(self):
        """DefaultLayout(controls=[]) is equivalent to DefaultLayout()."""
        layout = DefaultLayout(controls=[])
        assert layout.get_llm_kwargs("u1") == {}

    def test_constructor_controls_appear_in_rendered_page(self):
        """Controls passed to the constructor are rendered into the page."""
        c = Control(
            id="tok",
            html='<select id="tok"></select>',
            slot="toolbar",
            llm_param="max_completion_tokens",
        )
        layout = DefaultLayout(controls=[c])
        html = layout.render_page()
        assert '<select id="tok"></select>' in html


# =====================================================================
# DefaultLayout branding params
# =====================================================================


class TestDefaultLayoutBranding:
    def test_default_brand_in_header_link(self):
        html = DefaultLayout().render_page()
        assert 'href="/"' in html
        assert ">Chatnificent<" in html

    def test_default_welcome_message_in_js(self):
        html = DefaultLayout().render_page()
        assert "Welcome to Chatnificent" in html

    def test_default_welcome_message_includes_version(self):
        from chatnificent import __version__

        html = DefaultLayout().render_page()
        assert f"v{__version__}" in html

    def test_default_welcome_message_has_examples_link(self):
        html = DefaultLayout().render_page()
        assert "github.com/eliasdabbas/chatnificent/tree/main/examples" in html

    def test_default_welcome_message_has_changelog_link(self):
        html = DefaultLayout().render_page()
        assert "CHANGELOG.md" in html

    def test_welcome_message_div_in_html(self):
        html = DefaultLayout().render_page()
        assert 'id="welcome-message"' in html

    def test_custom_welcome_message(self):
        html = DefaultLayout(
            welcome_message="## Ask us anything\n\nWe're here to help."
        ).render_page()
        assert "Ask us anything" in html

    def test_custom_welcome_message_replaces_default(self):
        html = DefaultLayout(
            welcome_message="## Custom Heading\n\nCustom body."
        ).render_page()
        assert "Welcome to Chatnificent" not in html
        assert "Custom body." in html

    def test_default_page_title_is_seo_string(self):
        html = DefaultLayout().render_page()
        assert (
            "<title>Build an AI/LLM Chat App with Python | Chatnificent</title>" in html
        )

    def test_default_slogan_in_page(self):
        html = DefaultLayout().render_page()
        assert "Minimally complete \u2013 Maximally hackable" in html

    def test_custom_brand_in_header_link(self):
        html = DefaultLayout(brand="MyApp").render_page()
        assert ">MyApp<" in html
        assert 'href="/"' in html

    def test_welcome_message_default_is_independent_of_brand(self):
        html = DefaultLayout(brand="MyApp").render_page()
        assert "Welcome to Chatnificent" in html

    def test_custom_brand_in_page_title_fallback(self):
        html = DefaultLayout(brand="MyApp").render_page()
        assert "<title>Build an AI/LLM Chat App with Python | MyApp</title>" in html

    def test_custom_slogan(self):
        html = DefaultLayout(slogan="Hack everything.").render_page()
        assert "Hack everything." in html

    def test_explicit_page_title_overrides_seo_default(self):
        html = DefaultLayout(page_title="Custom Title | My Site").render_page()
        assert "<title>Custom Title | My Site</title>" in html

    def test_explicit_page_title_does_not_contain_seo_template(self):
        html = DefaultLayout(page_title="Custom Title | My Site").render_page()
        assert "Build an AI/LLM Chat App with Python" not in html

    def test_logo_url_renders_img_tag(self):
        html = DefaultLayout(logo_url="/static/logo.png").render_page()
        assert '<img id="header-logo"' in html
        assert 'src="/static/logo.png"' in html

    def test_logo_url_alt_uses_brand(self):
        html = DefaultLayout(brand="MyApp", logo_url="/logo.svg").render_page()
        assert 'alt="MyApp"' in html

    def test_no_logo_url_no_img_tag(self):
        html = DefaultLayout().render_page()
        assert 'id="header-logo"' not in html

    def test_brand_is_html_escaped(self):
        html = DefaultLayout(brand="<script>alert(1)</script>").render_page()
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html

    def test_slogan_is_html_escaped(self):
        html = DefaultLayout(slogan='<img src="x" onerror="evil()">').render_page()
        assert 'onerror="evil()"' not in html
        assert "&lt;img" in html

    def test_logo_url_is_html_escaped(self):
        html = DefaultLayout(logo_url='"/><script>evil()</script>').render_page()
        assert "<script>evil()</script>" not in html

    def test_favicon_url_renders_link_tag(self):
        html = DefaultLayout(favicon_url="/static/favicon.ico").render_page()
        assert '<link rel="icon" href="/static/favicon.ico">' in html

    def test_no_favicon_url_no_link_tag(self):
        html = DefaultLayout().render_page()
        assert 'rel="icon"' not in html

    def test_favicon_url_is_html_escaped(self):
        html = DefaultLayout(favicon_url='"><script>evil()</script>').render_page()
        assert "<script>evil()</script>" not in html

    def test_favicon_remote_url_passed_through(self):
        html = DefaultLayout(
            favicon_url="https://example.com/favicon.ico"
        ).render_page()
        assert 'href="https://example.com/favicon.ico"' in html

    def test_favicon_data_uri_passed_through(self):
        uri = "data:image/png;base64,iVBORw0KGgo="
        html = DefaultLayout(favicon_url=uri).render_page()
        assert uri in html

    def test_default_welcome_message_has_sidebar_hint(self):
        html = DefaultLayout().render_page()
        assert "Start typing below" in html

    def test_custom_welcome_message_plain_text(self):
        html = DefaultLayout(welcome_message="Ask me about the weather.").render_page()
        assert "Ask me about the weather." in html

    def test_custom_welcome_message_with_html(self):
        html = DefaultLayout(
            welcome_message="Try asking: <b>bold text</b>"
        ).render_page()
        assert "Try asking:" in html
        assert "<b>bold text</b>" in html

    def test_custom_welcome_message_with_markdown(self):
        html = DefaultLayout(
            welcome_message="Ask about **the weather** or _anything else_."
        ).render_page()
        assert "the weather" in html
        assert "marked.parse" in html

    def test_welcome_message_rendered_via_marked(self):
        html = DefaultLayout().render_page()
        assert "marked.parse" in html
        assert "DOMPurify.sanitize" in html
