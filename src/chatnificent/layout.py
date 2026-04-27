"""Layout pillar — defines the chat UI's visual identity.

Provides a thin Layout ABC for HTML-based servers and a DashLayout ABC
for Dash-based servers. Each server type uses the appropriate interface.
"""

import json
import threading
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass
from html import escape as _html_escape
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union

from .models import SYSTEM_ROLE, USER_ROLE

_TEMPLATES_DIR = Path(__file__).parent / "templates"


# =====================================================================
# Control dataclass
# =====================================================================


@dataclass
class Control:
    """A UI control whose value is bound to an LLM call parameter."""

    id: str
    html: str
    slot: str
    llm_param: str
    cast: Optional[Callable] = None


# =====================================================================
# Layout ABC — thin, zero-dependency interface
# =====================================================================


class Layout(ABC):
    """Base class for all Chatnificent layouts.

    Subclass this for HTML-based servers (DevServer, Starlette, FastAPI).
    For Dash-based servers, subclass DashLayout instead.
    """

    @abstractmethod
    def render_page(self) -> str:
        """Return the full HTML page as a string.

        Called by HTTP servers (DevServer, Starlette, etc.) to serve
        the initial page load.
        """

    def render_messages(
        self, messages: List[Dict[str, Any]], **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Return display-ready messages for DevServer.

        Parameters
        ----------
        messages : List[Dict[str, Any]]
            Canonical provider-native conversation messages.
        **kwargs : Any
            Optional context such as ``user_id``, ``convo_id``, and the
            canonical ``conversation`` object. Custom layouts can use these to
            load additional files and enrich the display without mutating the
            stored conversation history.
        """
        llm = getattr(getattr(self, "app", None), "llm", None)
        rendered = []

        for message in messages:
            if message.get("role") == SYSTEM_ROLE:
                continue
            if llm and llm.is_tool_message(message):
                continue

            rendered_message = dict(message)
            content = rendered_message.get("content")
            if content is None:
                continue

            if isinstance(content, str):
                if not content.strip():
                    continue
            else:
                rendered_message["content"] = json.dumps(
                    content, ensure_ascii=False, default=str
                )

            rendered.append(rendered_message)

        return rendered

    def render_conversations(
        self, conversations: List[Dict[str, Any]], **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Return display-ready conversation list items for DevServer."""
        return [dict(conversation) for conversation in conversations]

    def register_control(self, control: "Control") -> None:
        """Register a UI control to be rendered in a slot and bound to an LLM param."""
        pass

    def set_control_value(self, user_id: str, control_id: str, value: Any) -> None:
        """Store the latest value submitted by a user for a given control."""
        pass

    def get_control_values(self, user_id: str) -> Dict[str, Any]:
        """Return raw {control_id: value} state for a user."""
        return {}

    def get_llm_kwargs(self, user_id: str) -> Dict[str, Any]:
        """Return {llm_param: cast_value} kwargs to inject into the next LLM call for a user."""
        return {}

    def _is_rtl(self, text: str) -> bool:
        """Check if text requires right-to-left rendering."""
        if not text or isinstance(text, list) or not text.strip():
            return False
        for char in text:
            bidi = unicodedata.bidirectional(char)
            if bidi in ("R", "AL"):
                return True
            elif bidi == "L":
                return False
        return False


# =====================================================================
# DefaultLayout — zero-dependency HTML layout
# =====================================================================


class DefaultLayout(Layout):
    """Full-featured HTML chat layout using vanilla JS.

    Zero external dependencies. Works with DevServer and any HTTP server
    that calls render_page(). Analogous to Echo for the LLM pillar.
    """

    def __init__(
        self,
        brand: str = "Chatnificent",
        slogan: str = "Minimally complete \u2013 Maximally hackable",
        logo_url: Optional[str] = None,
        favicon_url: Optional[str] = None,
        page_title: Optional[str] = None,
        welcome_message: Optional[str] = None,
        controls: Optional[List[Control]] = None,
    ) -> None:
        """Configure the chat page's branding and optional UI controls.

        Parameters
        ----------
        brand : str
            Display name shown in the header and used as the default page-title
            suffix. HTML-escaped on render.
        slogan : str
            Short tagline rendered next to the brand. HTML-escaped on render.
        logo_url, favicon_url : str, optional
            URLs for the header logo image and browser-tab icon. Rendered as
            ``<img src="...">`` and ``<link rel="icon" href="...">``. Any URL
            the browser can fetch works (absolute, relative, or ``data:`` URI).
            DevServer does not serve static files — for local assets during
            development, base64-encode them and pass as
            ``data:image/...;base64,...``.
        page_title : str, optional
            Browser-tab title. Defaults to a Chatnificent-branded title that
            includes ``brand``.
        welcome_message : str, optional
            Markdown shown in the empty chat area. Rendered client-side via
            marked.js + DOMPurify. Defaults to a Chatnificent welcome that
            includes the installed version and links to examples and changelog.
        controls : list of Control, optional
            UI controls bound to LLM call parameters. Each ``Control.html`` is
            injected verbatim into its slot — the caller is responsible for
            the safety of that markup.
        """
        self._controls: Dict[str, Control] = {}
        self._state: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self.brand = brand
        self.slogan = slogan
        self.logo_url = logo_url
        self.favicon_url = favicon_url
        self.page_title = page_title
        if welcome_message is None:
            # Deferred import: avoids coupling to chatnificent/__init__.py import order.
            from chatnificent import __version__

            welcome_message = f"""## Welcome to Chatnificent v{__version__}

Start typing below, or browse the [examples](https://github.com/eliasdabbas/chatnificent/tree/main/examples) to see what's possible. New features land in the [changelog](https://github.com/eliasdabbas/chatnificent/blob/main/CHANGELOG.md)."""
        self.welcome_message = welcome_message
        for control in controls or []:
            self.register_control(control)

    def register_control(self, control: Control) -> None:
        """Register a UI control to be rendered in a slot and bound to an LLM param."""
        with self._lock:
            self._controls[control.id] = control

    def set_control_value(self, user_id: str, control_id: str, value: Any) -> None:
        """Store the latest value submitted by a user for a given control."""
        with self._lock:
            if user_id not in self._state:
                self._state[user_id] = {}
            self._state[user_id][control_id] = value

    def get_control_values(self, user_id: str) -> Dict[str, Any]:
        """Return raw {control_id: value} state for a user."""
        with self._lock:
            return dict(self._state.get(user_id, {}))

    def get_llm_kwargs(self, user_id: str) -> Dict[str, Any]:
        """Return {llm_param: cast_value} kwargs to inject into the next LLM call for a user."""
        with self._lock:
            user_state = self._state.get(user_id, {})
            controls = dict(self._controls)
        result = {}
        for control_id, control in controls.items():
            if control_id not in user_state:
                continue
            value = user_state[control_id]
            if value is None:
                continue
            result[control.llm_param] = control.cast(value) if control.cast else value
        return result

    def render_page(self) -> str:
        """Return the complete HTML chat page with registered controls injected into their slots."""
        html = (_TEMPLATES_DIR / "default.html").read_text(encoding="utf-8")
        title = _html_escape(
            self.page_title or f"Build an AI/LLM Chat App with Python | {self.brand}"
        )
        logo_html = (
            f'<img id="header-logo" src="{_html_escape(self.logo_url)}" alt="{_html_escape(self.brand)}">'
            if self.logo_url
            else ""
        )
        favicon_html = (
            f'<link rel="icon" href="{_html_escape(self.favicon_url)}">'
            if self.favicon_url
            else ""
        )
        html = html.replace("<!-- PAGE_TITLE -->", title)
        html = html.replace("<!-- BRAND -->", _html_escape(self.brand))
        html = html.replace("<!-- SLOGAN -->", _html_escape(self.slogan))
        html = html.replace("<!-- LOGO_HTML -->", logo_html)
        html = html.replace("<!-- FAVICON_HTML -->", favicon_html)
        welcome_script = f"""<script>
  document.addEventListener('DOMContentLoaded', function() {{
    var _wm = document.getElementById('welcome-message');
    if (_wm) _wm.innerHTML = DOMPurify.sanitize(marked.parse({json.dumps(self.welcome_message)}));
  }});
</script>"""
        html = html.replace("</body>", welcome_script + "\n</body>")
        with self._lock:
            controls = list(self._controls.values())
        slots: Dict[str, str] = {}
        for control in controls:
            slots[control.slot] = slots.get(control.slot, "") + control.html
        for slot_name, slot_html in slots.items():
            html = html.replace(f"<!-- SLOT:{slot_name} -->", slot_html)
        # Remove any remaining empty slot markers
        import re

        html = re.sub(r"<!-- SLOT:[^>]+ -->", "", html)
        if controls:
            ids_js = ", ".join(f'"{c.id}"' for c in controls)
            init_script = f"""
<script>
  document.addEventListener('DOMContentLoaded', function() {{
    [{ids_js}].forEach(function(id) {{
      var el = document.getElementById(id);
      if (el) chatInteraction(el);
    }});
  }});
</script>"""
            html = html.replace("</body>", init_script + "\n</body>")
        return html


# =====================================================================
# DashLayout — Dash-specific layout ABC
# =====================================================================


class DashLayout(Layout):
    """Layout for Dash-based servers.

    Extends the Layout ABC with Dash-specific methods for building
    component trees. The HTML render methods raise NotImplementedError
    since Dash handles its own page rendering.
    """

    def __init__(self, theme: Optional[str] = None):
        """Initialize layout with optional theme variant."""
        import dash.dcc
        import dash.html

        self.dcc = dash.dcc
        self.html = dash.html
        self.theme_name = theme
        layout = self.build_layout()
        self._validate_layout(layout)
        self.component_styles = self.get_current_styles()

    @abstractmethod
    def build_layout(self):
        """Build the Dash component tree for the full app layout.

        Must include these IDs for callback integration:
        - sidebar, sidebar_toggle, conversations_list, new_conversation_button
        - chat_area, messages_container, input_textarea, submit_button
        """

    @abstractmethod
    def build_messages(self, messages: List[Dict[str, Any]]) -> list:
        """Return Dash components for the message list."""

    @abstractmethod
    def get_external_stylesheets(self) -> List[Union[str, Dict]]:
        """Return required external stylesheets."""

    def get_external_scripts(self) -> List[Union[str, Dict]]:
        """Return required external scripts."""
        return []

    # HTML methods — Dash handles its own rendering
    def render_page(self) -> str:
        raise NotImplementedError(
            f"{type(self).__name__} is a Dash layout. "
            "Use DashServer, or override render_page() for HTML servers."
        )

    def render_messages(self, messages: List[Dict[str, Any]]) -> str:
        raise NotImplementedError(
            f"{type(self).__name__} is a Dash layout. "
            "Use DashServer, or override render_messages() for HTML servers."
        )

    # ===== STYLING & THEMING METHODS =====
    def get_class_name(self, component_key: str) -> Optional[str]:
        """Get CSS className for component key."""
        return self.component_styles.get(component_key, {}).get("className")

    def get_style(self, component_key: str) -> Optional[Dict]:
        """Get inline style dict for component key."""
        return self.component_styles.get(component_key, {}).get("style")

    def get_component_keys(self) -> Set[str]:
        """Get all available component styling keys."""
        return set(self.component_styles.keys())

    def get_current_styles(self) -> Dict[str, Dict]:
        """Extract component styles from layout tree."""
        styles = {}
        layout = self.build_layout()

        def traverse(component):
            if hasattr(component, "id") and component.id:
                style_dict = {}
                if hasattr(component, "style") and component.style:
                    style_dict["style"] = component.style
                if hasattr(component, "className") and component.className:
                    style_dict["className"] = component.className
                if style_dict:
                    styles[component.id] = style_dict
            if hasattr(component, "children"):
                children = component.children
                if isinstance(children, list):
                    for child in children:
                        if child is not None:
                            traverse(child)
                elif children is not None:
                    traverse(children)

        traverse(layout)
        return styles

    def _validate_layout(self, layout) -> None:
        """Validate layout contains required component IDs."""
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
        found_ids = set()

        def traverse(component):
            if hasattr(component, "id") and component.id:
                found_ids.add(component.id)
            if hasattr(component, "children"):
                children = component.children
                if isinstance(children, list):
                    for child in children:
                        if child is not None:
                            traverse(child)
                elif children is not None:
                    traverse(children)

        traverse(layout)
        missing_ids = required_ids - found_ids
        if missing_ids:
            raise ValueError(f"Layout missing required component IDs: {missing_ids}")


class Bootstrap(DashLayout):
    """Bootstrap-based layout with integrated message formatting and theming."""

    def __init__(self, theme: Optional[str] = "bootstrap"):
        """Initialize with Bootstrap theme variant (bootstrap, flatly, darkly, etc.)."""
        import dash_bootstrap_components as dbc

        self.dbc = dbc
        super().__init__(theme)

    # Remove the overridden get_current_styles method - use base class implementation
    # The base class already correctly traverses the layout tree

    def get_external_stylesheets(self) -> List[Union[str, Dict]]:
        """Return Bootstrap stylesheets based on theme variant."""
        themes = {
            "bootstrap": {
                "href": "https://cdn.jsdelivr.net/npm/bootstrap@5.3.7/dist/css/bootstrap.min.css",
                "rel": "stylesheet",
                "integrity": "sha384-LN+7fdVzj6u52u30Kp6M/trliBMCMKTyK833zpbD+pXdCLuTusPj697FH4R/5mcr",
                "crossorigin": "anonymous",
            },
            # Add other Bootstrap themes...
        }
        return [
            themes.get(self.theme_name, themes["bootstrap"]),
            {
                "href": "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.13.1/font/bootstrap-icons.min.css",
                "rel": "stylesheet",
            },
            {
                "href": "data:text/css;charset=utf-8,"
                + "#messages_container::-webkit-scrollbar { display: none; } "
                + "#messages_container { scrollbar-width: none; -ms-overflow-style: none; } "
                + ".hover-effect:hover { background-color: #e9ecef !important; transform: translateY(-1px); }",
                "rel": "stylesheet",
            },
        ]

    def build_layout(self):
        """Complete Bootstrap layout - main structure visible for easy customization."""
        return self.dbc.Container(
            [
                self.dcc.Location(id="url_location", refresh=False),
                self.build_sidebar_toggle(),
                self.build_sidebar(),
                self.dbc.Row(
                    [
                        self.dbc.Col(
                            [
                                self.build_chat_area(),
                            ],
                            lg=7,
                            md=12,
                            className="mx-auto",
                            style={
                                "position": "relative",
                                "height": "calc(100vh - 160px)",
                            },
                        ),
                    ]
                ),
                self.build_input_area(),
            ],
            fluid=True,
            style={"height": "100vh"},
        )

    def build_sidebar_toggle(self):
        """Simple fixed burger menu button only."""
        return self.dbc.Button(
            self.html.I(className="bi bi-list"),
            id="sidebar_toggle",
            n_clicks=0,
            className="navbar-brand",
            style={
                "position": "fixed",
                "top": "12px",
                "left": "12px",
                "border": "none",
                "background": "transparent",
                "color": "var(--bs-dark)",
                "fontSize": "34px",
                "padding": "0.5rem",
                "cursor": "pointer",
                "zIndex": "9999",
            },
        )

    def build_sidebar(self):
        """Collapsible sidebar with new chat button above conversations list."""
        return self.html.Div(
            [
                self.html.Br(),
                self.html.Br(),
                self.html.Br(),
                self.html.Span(
                    [self.html.I(className="bi bi-pencil-square me-2"), "New Chat"],
                    id="new_conversation_button",
                    n_clicks=0,
                    style={
                        "display": "flex",
                        "alignItems": "center",
                        "justifyContent": "center",
                        "cursor": "pointer",
                    },
                ),
                self.html.Ul(id="conversations_list", className="list-unstyled"),
            ],
            id="sidebar",
            hidden=True,
            className="p-3 border-end",
            style={
                "width": "280px",
                "height": "100vh",
                "overflowY": "auto",
                "position": "fixed",
                "top": "0",
                "left": "0",
                "zIndex": "1040",
                "backgroundColor": "var(--bs-body-bg)",
            },
        )

    def build_chat_area(self):
        """Chat area with proper viewport height and scrolling."""
        return self.html.Div(
            [
                self.html.Div(
                    id="messages_container",
                    className="overflow-auto",
                    style={
                        "height": "calc(100vh - 160px)",
                        "scrollbarWidth": "none",  # Firefox
                        "msOverflowStyle": "none",  # IE and Edge
                        "padding": "16px",
                        "paddingBottom": "24px",  # Extra padding at bottom
                    },
                )
            ],
            id="chat_area",
        )

    def build_input_area(self):
        """Fixed input area at bottom of screen that never scrolls."""
        return self.html.Div(
            [
                self.dbc.Container(
                    [
                        # Status indicator positioned above input with same layout
                        self.dbc.Row(
                            [
                                self.dbc.Col(
                                    [
                                        self.html.Div(
                                            [
                                                self.html.Span(
                                                    "Working...",
                                                    style={"marginRight": "8px"},
                                                ),
                                                self.dbc.Spinner(size="sm"),
                                            ],
                                            id="status_indicator",
                                            hidden=True,
                                            style={
                                                "textAlign": "left",
                                                "color": "#888",
                                                "fontSize": "15px",
                                                "marginBottom": "8px",
                                                "fontStyle": "italic",
                                                "fontWeight": "300",
                                            },
                                        ),
                                    ],
                                    lg=7,
                                    md=12,
                                    className="mx-auto",
                                ),
                            ]
                        ),
                        self.dbc.Row(
                            [
                                self.dbc.Col(
                                    [
                                        self.html.Div(
                                            [
                                                self.dbc.Textarea(
                                                    id="input_textarea",
                                                    placeholder="Ask...",
                                                    rows=4,
                                                    className="border-0 shadow-none",
                                                    style={
                                                        "border": "0",
                                                        "flex": "1",
                                                    },
                                                ),
                                                self.dbc.Button(
                                                    self.html.I(className="bi bi-send"),
                                                    id="submit_button",
                                                    n_clicks=0,
                                                    style={
                                                        "border": "none",
                                                        "background": "transparent",
                                                        "color": "#484a4d",
                                                        "fontSize": "32px",
                                                        "padding": "4px",
                                                        "cursor": "pointer",
                                                    },
                                                ),
                                            ],
                                            style={
                                                "display": "flex",
                                                "alignItems": "center",
                                                "border": "1px solid #dee2e6",
                                                "borderRadius": "25px",
                                                "overflow": "hidden",
                                                "padding": "8px 16px",
                                            },
                                        )
                                    ],
                                    lg=7,
                                    md=12,
                                    className="mx-auto",
                                ),
                            ]
                        ),
                    ],
                    fluid=True,
                )
            ],
            style={
                "position": "fixed",
                "bottom": "0",
                "left": "0",
                "right": "0",
                "backgroundColor": "white",
                "padding": "15px 0",
                "zIndex": "1000",
            },
        )

    def build_messages(self, messages: List[Dict[str, Any]]) -> list:
        """Build all message components for display."""
        if not messages:
            return []
        return [self.build_message(msg, i) for i, msg in enumerate(messages)]

    def build_message(self, message: Dict[str, Any], index: int):
        """Build single message component."""
        content = message.get("content", "")
        direction = "rtl" if self._is_rtl(content) else "ltr"
        if message.get("role") == USER_ROLE:
            return self.build_user_message(message, index, direction)
        else:
            return self.build_assistant_message(message, index, direction)

    def build_user_message(
        self, message: Dict[str, Any], index: int, direction: str = "ltr"
    ):
        """Build user message with Bootstrap styling - right-aligned with copy button."""
        content = message.get("content", "")
        return self.html.Div(
            className="mb-3",
            dir=direction,
            children=[
                self.dbc.Row(
                    [
                        self.dbc.Col(
                            [
                                self.html.Div(
                                    [
                                        self.dcc.Markdown(
                                            content,
                                            id=f"user_msg_{index}",
                                            className="p-3 rounded-3 bg-light table",
                                            style={
                                                "lineHeight": "1.5",
                                                "wordWrap": "break-word",
                                            },
                                        ),
                                    ],
                                    style={
                                        "width": "fit-content",
                                        "marginLeft": "auto",
                                    },
                                )
                            ],
                            width=8,
                            className="ms-auto",
                        )
                    ]
                ),
                self.build_copy_button(content, "user", index),
            ],
        )

    def build_assistant_message(
        self, message: Dict[str, Any], index: int, direction: str = "ltr"
    ):
        """Build assistant message with Bootstrap styling - left-aligned with copy button."""
        content = message.get("content", "")
        return self.html.Div(
            className="mb-3",
            dir=direction,
            children=[
                self.dbc.Row(
                    [
                        self.dbc.Col(
                            [
                                self.html.Div(
                                    [
                                        self.dcc.Markdown(
                                            content,
                                            id=f"assistant_msg_{index}",
                                            className="p-3 table",
                                            style={
                                                "lineHeight": "1.72",
                                                "wordWrap": "break-word",
                                            },
                                        ),
                                    ],
                                )
                            ],
                            width=12,
                        )
                    ]
                ),
                self.build_copy_button(content, "assistant", index),
            ],
        )

    def build_copy_button(self, content: str, msg_type: str, index: int):
        """Build copy button with proper Bootstrap positioning, only if content is non-empty."""
        if content is None or str(content).strip() == "":
            return None
        return self.html.Div(
            [
                self.dcc.Clipboard(
                    content=content,
                    id=f"copy_{msg_type}_{index}",
                    title="Copy message",
                    style={
                        "display": "inline-block",
                        "fontSize": "16px",
                        "cursor": "pointer",
                        "marginLeft": "8px",
                        "marginRight": "0px",
                    },
                ),
            ],
            style={
                "textAlign": "right" if msg_type == "user" else "left",
                "marginTop": "2px",
            },
        )


class Mantine(DashLayout):
    """Mantine-based layout - systematic translation from Bootstrap."""

    def __init__(self, theme: Optional[str] = "light"):
        """Initialize with Mantine theme variant (light, dark, etc.)."""
        import dash_mantine_components as dmc
        from dash_iconify import DashIconify

        self.dmc = dmc
        self.DashIconify = DashIconify
        super().__init__(theme)

    def get_external_stylesheets(self) -> List[Union[str, Dict]]:
        """Mantine stylesheets are bundled as of 1.2.0."""
        return []

    def build_layout(self):
        """Complete Mantine layout - wraps MantineProvider around Bootstrap structure."""
        return self.dmc.MantineProvider(
            forceColorScheme=self.theme_name or "light",
            children=self.dmc.TypographyStylesProvider(
                [
                    self.dcc.Location(id="url_location", refresh=False),
                    self.build_sidebar_toggle(),
                    self.build_sidebar(),
                    self.build_chat_area(),
                    self.build_input_area(),
                ]
            ),
        )

    def build_sidebar_toggle(self):
        """Header with burger menu - uses ActionIcon but keeps same styling."""
        return self.dmc.ActionIcon(
            self.DashIconify(icon="bi-list", width=36),
            id="sidebar_toggle",  # CALLBACK COMPONENT - ActionIcon supports n_clicks
            n_clicks=0,
            size="xl",
            variant="subtle",
            color="gray",
            style={
                "position": "fixed",
                "top": "12px",
                "left": "12px",
                "zIndex": "9999",
            },
        )

    def build_sidebar(self):
        """Sidebar - wrapped in self.html.Div for hidden property."""
        return self.html.Div(  # CALLBACK COMPONENT - needs hidden property
            [
                self.dmc.Button(
                    "New Chat",
                    leftSection=self.DashIconify(icon="tabler:plus"),
                    id="new_conversation_button",
                    n_clicks=0,
                    variant="subtle",
                    color="gray",
                    mt=48,
                    mb="md",
                ),
                self.dmc.ScrollArea(
                    self.html.Ul(  # CALLBACK COMPONENT - conversations list
                        id="conversations_list",
                        style={"listStyle": "none", "padding": "0"},
                    ),
                    type="hover",
                ),
            ],
            id="sidebar",
            hidden=True,
            style={
                "width": "280px",
                "height": "100vh",
                "position": "fixed",
                "top": "0",
                "left": "0",
                "zIndex": "1040",
                "padding": "16px",
                "borderRight": "1px solid var(--mantine-color-default-border)",
                "backgroundColor": "var(--mantine-color-body)",
            },
        )

    def build_chat_area(self):
        """Chat area ."""
        return self.dmc.Grid(
            self.dmc.GridCol(
                self.dmc.ScrollArea(
                    id="messages_container",  # CALLBACK COMPONENT
                    type="never",
                    h="calc(100vh - 100px)",
                    p="md",
                ),
                span={"md": 7},
            ),
            justify="center",
            id="chat_area",
        )

    def build_input_area(self):
        """Input area - status_indicator wrapped, submit_button wrapped."""
        return self.dmc.Grid(
            children=[
                self.dmc.GridCol(
                    span={"md": 7},
                    children=self.dmc.Container(
                        [
                            self.html.Div(
                                self.dmc.Button(
                                    [self.dmc.Text("Working...")],
                                    rightSection=self.dmc.Loader(
                                        size="xs", color="gray"
                                    ),
                                    fs="italic",
                                    c="dimmed",
                                    variant="transparent",
                                ),
                                id="status_indicator",
                                hidden=True,
                            ),
                            self.dmc.Flex(
                                align="center",
                                style={
                                    "border": "1px solid var(--mantine-color-default-border)",
                                    "backgroundColor": "var(--mantine-color-body)",
                                    "borderRadius": "25px",
                                    "padding": "8px 16px",
                                    "marginBottom": "10px",
                                },
                                children=[
                                    self.dmc.Textarea(
                                        id="input_textarea",
                                        placeholder="Ask...",
                                        autosize=True,
                                        maxRows=6,
                                        variant="unstyled",
                                        style={"flex": 1},
                                    ),
                                    self.dmc.ActionIcon(
                                        self.DashIconify(icon="bi-send"),
                                        id="submit_button",
                                        n_clicks=0,
                                        variant="subtle",
                                        radius="lg",
                                        color="gray",
                                    ),
                                ],
                            ),
                        ],
                    ),
                )
            ],
            justify="center",
            pos="fixed",
            bottom=0,
            left=0,
            right=0,
            style={"zIndex": 1000},
        )

    def build_messages(self, messages: List[Dict[str, Any]]) -> list:
        """Build all message components for display."""
        if not messages:
            return []

        return [self.build_message(msg, i) for i, msg in enumerate(messages)]

    def build_message(self, message: Dict[str, Any], index: int):
        """Build single message component."""
        content = message.get("content", "")
        direction = "rtl" if self._is_rtl(content) else "ltr"
        if message.get("role") == USER_ROLE:
            return self.build_user_message(message, index, direction)
        else:
            return self.build_assistant_message(message, index, direction)

    def build_user_message(
        self, message: Dict[str, Any], index: int, direction: str = "ltr"
    ):
        """User message - direct translation from Bootstrap with Mantine colors."""
        content = message.get("content", "")
        return self.html.Div(
            style={"marginBottom": "16px", "direction": direction},
            children=[
                self.dcc.Markdown(
                    content,
                    id=f"user_msg_{index}",
                    style={
                        "padding": "8px",
                        "borderRadius": "8px",
                        "backgroundColor": "var(--mantine-color-default-border)",
                        "wordWrap": "break-word",
                        "width": "fit-content",
                        "marginLeft": "auto",
                        "maxWidth": "66.67%",
                    },
                ),
                self.build_copy_button(content, "user", index),
            ],
        )

    def build_assistant_message(
        self, message: Dict[str, Any], index: int, direction: str = "ltr"
    ):
        """Assistant message - direct translation from Bootstrap."""
        content = message.get("content", "")
        return self.html.Div(
            style={"marginBottom": "8px", "direction": direction},
            children=[
                self.dcc.Markdown(
                    content,
                    id=f"assistant_msg_{index}",
                    style={
                        "wordWrap": "break-word",
                    },
                ),
                self.build_copy_button(content, "assistant", index),
            ],
        )

    def build_copy_button(self, content: str, msg_type: str, index: int):
        """Copy button - exact translation from Bootstrap, only if content is non-empty."""
        if content is None or str(content).strip() == "":
            return None
        return self.html.Div(
            [
                self.dcc.Clipboard(
                    content=content,
                    id=f"copy_{msg_type}_{index}",
                    title="Copy message",
                    style={
                        "display": "inline-block",
                        "fontSize": "12px",
                        "cursor": "pointer",
                        "marginLeft": "8px",
                        "marginRight": "0px",
                        "color": "var(--mantine-color-dimmed)",
                    },
                ),
            ],
            style={
                "textAlign": "right" if msg_type == "user" else "left",
                "marginTop": "2px",
            },
        )


class Minimal(DashLayout):
    """Minimal layout using only standard Dash/HTML components."""

    def __init__(self, theme: Optional[str] = None):
        super().__init__(theme)

    def get_external_stylesheets(self) -> List[Union[str, Dict]]:
        return [
            "https://cdnjs.cloudflare.com/ajax/libs/normalize/8.0.1/normalize.min.css"
        ]

    # ===== LAYOUT BUILDING METHODS =====
    def build_layout(self):
        """Build simple HTML layout."""
        return self.html.Div(
            [
                self.dcc.Location(id="url_location", refresh=False),
                self.build_sidebar_toggle(),
                self.build_sidebar(),
                self.html.Div(
                    [
                        self.html.Div(
                            [
                                self.build_chat_area(),
                                self.build_input_area(),
                            ],
                            style={
                                "width": "40%",
                                "margin": "0 auto",
                                "paddingTop": "60px",  # Account for fixed header
                                "paddingBottom": "20px",
                            },
                        )
                    ],
                    style={
                        "width": "100%",
                        "minHeight": "100vh",
                    },
                ),
            ],
            style={
                "fontFamily": "Arial, sans-serif",
            },
        )

    def build_sidebar_toggle(self):
        return self.html.Button(
            "☰",
            id="sidebar_toggle",
            n_clicks=0,
            style={
                "position": "fixed",
                "top": "12px",
                "left": "12px",
                "zIndex": "9999",
                "border": "none",
                "background": "rgba(255, 255, 255, 0.9)",
                "fontSize": "44px",
                "padding": "8px",
                "borderRadius": "4px",
                "cursor": "pointer",
            },
        )

    def build_sidebar(self):
        return self.html.Div(
            [
                self.html.Br(),
                self.html.Br(),
                self.html.Br(),
                self.html.Br(),
                self.html.Br(),
                self.html.Span(
                    self.html.B("✏️ New chat"),
                    id="new_conversation_button",
                    n_clicks=0,
                    style={
                        "margin": "8px",
                        "cursor": "pointer",
                    },
                ),
                self.html.Br(),
                self.html.Br(),
                self.html.Br(),
                self.html.Ul(id="conversations_list"),
            ],
            id="sidebar",
            hidden=True,
            style={
                "width": "280px",
                "background": "#f8f9fa",
                "padding": "16px",
                "position": "fixed",
                "top": "0",
                "left": "0",
                "height": "100vh",
                "overflowY": "auto",
                "zIndex": "1040",
                "borderRight": "1px solid #dee2e6",
            },
        )

    def build_chat_area(self):
        return self.html.Div(
            id="chat_area",
            children=[
                self.html.Div(
                    id="messages_container",
                    style={
                        "minHeight": "300px",
                        "padding": "16px",
                        "background": "#fff",
                        "borderRadius": "8px",
                        "marginBottom": "16px",
                        "flexGrow": 20,
                    },
                )
            ],
        )

    def build_input_area(self):
        return self.html.Div(
            [
                self.html.Div(
                    [
                        self.html.Span(
                            "Working...",
                            style={"marginRight": "8px"},
                        ),
                    ],
                    id="status_indicator",
                    hidden=True,
                    style={
                        "textAlign": "left",
                        "color": "#888",
                        "fontSize": "15px",
                        "marginBottom": "8px",
                        "fontStyle": "italic",
                        "fontWeight": "300",
                    },
                ),
                self.html.Div(
                    [
                        self.dcc.Textarea(
                            id="input_textarea",
                            rows=4,
                            style={
                                "gridArea": "textarea",
                                "width": "100%",
                                "resize": "none",
                                "border": "1px solid #ccc",
                                "padding": "8px",
                                "boxSizing": "border-box",
                                "borderRadius": "25px 0 0 25px",
                            },
                        ),
                        self.html.Button(
                            "Send",
                            id="submit_button",
                            n_clicks=0,
                            style={
                                "gridArea": "button",
                                "width": "100%",
                                "height": "100%",
                                "border": "1px solid #ccc",
                                "cursor": "pointer",
                                "borderRadius": "0 25px 25px 0",
                            },
                        ),
                    ],
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "85% 15%",
                        "gridTemplateAreas": '"textarea button"',
                        "width": "100%",
                        "gap": "0px",
                    },
                ),
            ],
            style={
                "position": "fixed",
                "bottom": "0",
                "left": "50%",
                "transform": "translateX(-50%)",
                "width": "40%",
                "backgroundColor": "white",
                "padding": "15px",
                "zIndex": "1000",
                # "borderTop": "1px solid #eee",
            },
        )

    # ===== MESSAGE FORMATTING METHODS =====
    def build_messages(self, messages: List[Dict[str, Any]]) -> list:
        """Build simple message list."""
        if not messages:
            return []
        return [self.build_message(msg, i) for i, msg in enumerate(messages)]

    def build_message(self, message: Dict[str, Any], index: int):
        """Build simple message component."""
        content = message.get("content", "")
        direction = "rtl" if self._is_rtl(content) else "ltr"
        if message.get("role") == USER_ROLE:
            return self.build_user_message(message, index, direction)
        else:
            return self.build_assistant_message(message, index, direction)

    def build_user_message(
        self, message: Dict[str, Any], index: int, direction: str = "ltr"
    ):
        content = message.get("content", "")
        return self.html.Div(
            className="user-message mb-3",
            dir=direction,
            children=[
                self.html.Div(
                    className="user-message-content",
                    children=[
                        self.dcc.Markdown(content, id=f"user_msg_{index}"),
                        self.build_copy_button(content, "user", index),
                    ],
                )
            ],
        )

    def build_assistant_message(
        self, message: Dict[str, Any], index: int, direction: str = "ltr"
    ):
        content = message.get("content", "")
        return self.html.Div(
            className="assistant-message mb-3",
            dir=direction,
            children=[
                self.html.Div(
                    className="assistant-message-content",
                    children=[
                        self.dcc.Markdown(content, id=f"assistant_msg_{index}"),
                        self.build_copy_button(content, "assistant", index),
                    ],
                )
            ],
        )

    def build_copy_button(self, content: str, msg_type: str, index: int):
        """Build copy button with proper Bootstrap positioning, only if content is non-empty."""
        if content is None or str(content).strip() == "":
            return None
        return self.html.Div(
            [
                self.dcc.Clipboard(
                    content=content,
                    id=f"copy_{msg_type}_{index}",
                    title="Copy message",
                    style={
                        "display": "inline-block",
                        "fontSize": "16px",
                        "color": "#6c757d",
                        "cursor": "pointer",
                        "marginLeft": "8px",
                        "marginRight": "0px",
                    },
                ),
            ],
            style={
                "textAlign": "right" if msg_type == "user" else "left",
                "marginTop": "2px",
            },
        )
