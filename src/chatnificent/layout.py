"""Unified layout module - merges formatting, theming, and layout building."""

import unicodedata
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Union

import dash.dcc as dcc
import dash.html as html
from dash.development.base_component import Component as DashComponent

from .models import USER_ROLE, ChatMessage


class Layout(ABC):
    """
    Unified base class that handles:
    1. Overall layout building (header, sidebar, chat area, input)
    2. Message formatting and rendering
    3. Component styling and theming
    """

    def __init__(self, theme: Optional[str] = None):
        """Initialize layout with optional theme variant."""
        self.theme_name = theme
        layout = self.build_layout()
        self._validate_layout(layout)
        self.component_styles = self.get_current_styles()

    # ===== MINIMAL CORE INTERFACE =====
    @abstractmethod
    def build_layout(self) -> DashComponent:
        """
        Build complete app layout with required component IDs.

        Must include these IDs for framework integration:
        - sidebar, sidebar_toggle, conversations_list, new_conversation_button
        - chat_area, messages_container, input_textarea, submit_button

        Implementation strategy (decomposition, orchestration) is completely free.
        """
        pass

    @abstractmethod
    def build_messages(self, messages: List[ChatMessage]) -> List[DashComponent]:
        """
        Render messages for display in chat area.

        Core contract for how users see their conversations.
        """
        pass

    # ===== STYLING & THEMING METHODS =====
    @abstractmethod
    def get_external_stylesheets(self) -> List[Union[str, Dict]]:
        """Return required external stylesheets."""
        pass

    def get_external_scripts(self) -> List[Union[str, Dict]]:
        """Return required external scripts."""
        return []

    def get_class_name(self, component_key: str) -> Optional[str]:
        """Get CSS className for component key."""
        return self.component_styles.get(component_key, {}).get("className")

    def get_style(self, component_key: str) -> Optional[Dict]:
        """Get inline style dict for component key."""
        return self.component_styles.get(component_key, {}).get("style")

    def get_component_keys(self) -> Set[str]:
        """Get all available component styling keys."""
        return set(self.component_styles.keys())

    # ===== UTILITY METHODS =====
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

    def _validate_layout(self, layout: DashComponent) -> None:
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


class Bootstrap(Layout):
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

    def build_layout(self) -> DashComponent:
        """Complete Bootstrap layout - main structure visible for easy customization."""
        return self.dbc.Container(
            [
                dcc.Location(id="url_location", refresh=False),
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

    def build_sidebar_toggle(self) -> DashComponent:
        """Simple fixed burger menu button only."""
        return self.dbc.Button(
            html.I(className="bi bi-list"),
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

    def build_sidebar(self) -> DashComponent:
        """Collapsible sidebar with new chat button above conversations list."""
        return html.Div(
            [
                html.Br(),
                html.Br(),
                html.Br(),
                html.Span(
                    [html.I(className="bi bi-pencil-square me-2"), "New Chat"],
                    id="new_conversation_button",
                    n_clicks=0,
                    style={
                        "display": "flex",
                        "alignItems": "center",
                        "justifyContent": "center",
                        "cursor": "pointer",
                    },
                ),
                html.Ul(id="conversations_list", className="list-unstyled"),
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

    def build_chat_area(self) -> DashComponent:
        """Chat area with proper viewport height and scrolling."""
        return html.Div(
            [
                html.Div(
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

    def build_input_area(self) -> DashComponent:
        """Fixed input area at bottom of screen that never scrolls."""
        return html.Div(
            [
                self.dbc.Container(
                    [
                        # Status indicator positioned above input with same layout
                        self.dbc.Row(
                            [
                                self.dbc.Col(
                                    [
                                        html.Div(
                                            [
                                                html.Span(
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
                                        html.Div(
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
                                                    html.I(className="bi bi-send"),
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

    def build_messages(self, messages: List[ChatMessage]) -> List[DashComponent]:
        """Build all message components for display."""
        if not messages:
            return []
        return [self.build_message(msg, i) for i, msg in enumerate(messages)]

    def build_message(self, message: ChatMessage, index: int) -> DashComponent:
        """Build single message component."""
        direction = "rtl" if self._is_rtl(message.content) else "ltr"
        if message.role == USER_ROLE:
            return self.build_user_message(message, index, direction)
        else:
            return self.build_assistant_message(message, index, direction)

    def build_user_message(
        self, message: ChatMessage, index: int, direction: str = "ltr"
    ) -> DashComponent:
        """Build user message with Bootstrap styling - right-aligned with copy button."""
        return html.Div(
            className="mb-3",
            dir=direction,
            children=[
                self.dbc.Row(
                    [
                        self.dbc.Col(
                            [
                                html.Div(
                                    [
                                        dcc.Markdown(
                                            message.content,
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
                self.build_copy_button(message.content, "user", index),
            ],
        )

    def build_assistant_message(
        self, message: ChatMessage, index: int, direction: str = "ltr"
    ) -> DashComponent:
        """Build assistant message with Bootstrap styling - left-aligned with copy button."""
        return html.Div(
            className="mb-3",
            dir=direction,
            children=[
                self.dbc.Row(
                    [
                        self.dbc.Col(
                            [
                                html.Div(
                                    [
                                        dcc.Markdown(
                                            message.content,
                                            id=f"assistant_msg_{index}",
                                            className="p-3 table",
                                            style={
                                                "lineHeight": "1.5",
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
                self.build_copy_button(message.content, "assistant", index),
            ],
        )

    def build_copy_button(
        self, content: str, msg_type: str, index: int
    ) -> Optional[DashComponent]:
        """Build copy button with proper Bootstrap positioning, only if content is non-empty."""
        if content is None or str(content).strip() == "":
            return None
        return html.Div(
            [
                dcc.Clipboard(
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


class Mantine(Layout):
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

    def build_layout(self) -> DashComponent:
        """Complete Mantine layout - wraps MantineProvider around Bootstrap structure."""
        return self.dmc.MantineProvider(
            forceColorScheme= self.theme_name or "light",
            children=self.dmc.TypographyStylesProvider([
                dcc.Location(id="url_location", refresh=False),
                self.build_sidebar_toggle(),
                self.build_sidebar(),
                self.build_chat_area(),
                self.build_input_area(),
            ]),
        )

    def build_sidebar_toggle(self) -> DashComponent:
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

    def build_sidebar(self) -> DashComponent:
        """Sidebar - wrapped in html.Div for hidden property."""
        return html.Div(  # CALLBACK COMPONENT - needs hidden property
            [

                self.dmc.Button(
                    "New Chat",
                    leftSection=self.DashIconify(icon="tabler:plus"),
                    id="new_conversation_button",
                    n_clicks=0,
                    variant="subtle",
                    color="gray",
                    mt=48,
                    mb="md"
                ),
                self.dmc.ScrollArea(
                    html.Ul(  # CALLBACK COMPONENT - conversations list
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

    def build_chat_area(self) -> DashComponent:
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

    def build_input_area(self) -> DashComponent:
        """Input area - status_indicator wrapped, submit_button wrapped."""
        return self.dmc.Grid(
            children=[
                self.dmc.GridCol(
                    span={"md": 7},
                    children=self.dmc.Container(
                        [
                            html.Div(
                                self.dmc.Button(
                                    [self.dmc.Text("Working...")],
                                    rightSection=self.dmc.Loader(size="xs", color="gray"),
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
            style={"zIndex": 1000}
        )



    def build_messages(self, messages: List[ChatMessage]) -> List[DashComponent]:
        """Build all message components for display."""
        if not messages:
            return []

        return [self.build_message(msg, i) for i, msg in enumerate(messages)]

    def build_message(self, message: ChatMessage, index: int) -> DashComponent:
        """Build single message component."""
        direction = "rtl" if self._is_rtl(message.content) else "ltr"
        if message.role == USER_ROLE:
            return self.build_user_message(message, index, direction)
        else:
            return self.build_assistant_message(message, index, direction)

    def build_user_message(
        self, message: ChatMessage, index: int, direction: str = "ltr"
    ) -> DashComponent:
        """User message - direct translation from Bootstrap with Mantine colors."""
        return html.Div(
            style={"marginBottom": "16px", "direction": direction},
            children=[
                dcc.Markdown(
                    message.content,
                    id=f"user_msg_{index}",
                    style={
                        "padding": "8px",
                        "borderRadius": "8px",
                        "backgroundColor": "var(--mantine-color-default-border)",
                        "wordWrap": "break-word",
                        "width": "fit-content",
                        "marginLeft": "auto",
                        "maxWidth": "66.67%"
                    },
                ),
                self.build_copy_button(message.content, "user", index),
            ],
        )
    def build_assistant_message(
        self, message: ChatMessage, index: int, direction: str = "ltr"
    ) -> DashComponent:
        """Assistant message - direct translation from Bootstrap."""

        return html.Div(
            style={"marginBottom": "8px", "direction": direction},
            children=[
                dcc.Markdown(
                    message.content,
                    id=f"assistant_msg_{index}",
                    style={
                        "wordWrap": "break-word",
                    },
                ),
                self.build_copy_button(message.content, "assistant", index),
            ],
        )

    def build_copy_button(
        self, content: str, msg_type: str, index: int
    ) -> Optional[DashComponent]:
        """Copy button - exact translation from Bootstrap, only if content is non-empty."""
        if content is None or str(content).strip() == "":
            return None
        return html.Div(
            [
                dcc.Clipboard(
                    content=content,
                    id=f"copy_{msg_type}_{index}",
                    title="Copy message",
                    style={
                        "display": "inline-block",
                        "fontSize": "12px",
                        "cursor": "pointer",
                        "marginLeft": "8px",
                        "marginRight": "0px",
                        "color": "var(--mantine-color-dimmed)"
                    },
                ),
            ],
            style={
                "textAlign": "right" if msg_type == "user" else "left",
                "marginTop": "2px",
            },
        )


class Minimal(Layout):
    """Minimal layout using only standard Dash/HTML components."""

    def __init__(self, theme: Optional[str] = None):
        super().__init__(theme)

    def get_external_stylesheets(self) -> List[Union[str, Dict]]:
        return [
            "https://cdnjs.cloudflare.com/ajax/libs/normalize/8.0.1/normalize.min.css"
        ]

    # ===== LAYOUT BUILDING METHODS =====
    def build_layout(self) -> DashComponent:
        """Build simple HTML layout."""
        return html.Div(
            [
                dcc.Location(id="url_location", refresh=False),
                self.build_sidebar_toggle(),
                self.build_sidebar(),
                html.Div(
                    [
                        html.Div(
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

    def build_sidebar_toggle(self) -> DashComponent:
        return html.Button(
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

    def build_sidebar(self) -> DashComponent:
        return html.Div(
            [
                html.Br(),
                html.Br(),
                html.Br(),
                html.Br(),
                html.Br(),
                html.Span(
                    html.B("✏️ New chat"),
                    id="new_conversation_button",
                    n_clicks=0,
                    style={
                        "margin": "8px",
                        "cursor": "pointer",
                    },
                ),
                html.Br(),
                html.Br(),
                html.Br(),
                html.Ul(id="conversations_list"),
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

    def build_chat_area(self) -> DashComponent:
        return html.Div(
            id="chat_area",
            children=[
                html.Div(
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

    def build_input_area(self) -> DashComponent:
        return html.Div(
            [
                html.Div(
                    [
                        html.Span(
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
                html.Div(
                    [
                        dcc.Textarea(
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
                        html.Button(
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
    def build_messages(self, messages: List[ChatMessage]) -> List[DashComponent]:
        """Build simple message list."""
        if not messages:
            return []
        return [self.build_message(msg, i) for i, msg in enumerate(messages)]

    def build_message(self, message: ChatMessage, index: int) -> DashComponent:
        """Build simple message component."""
        direction = "rtl" if self._is_rtl(message.content) else "ltr"
        if message.role == USER_ROLE:
            return self.build_user_message(message, index, direction)
        else:
            return self.build_assistant_message(message, index, direction)

    def build_user_message(
        self, message: ChatMessage, index: int, direction: str = "ltr"
    ) -> DashComponent:
        return html.Div(
            className="user-message mb-3",
            dir=direction,
            children=[
                html.Div(
                    className="user-message-content",
                    children=[
                        dcc.Markdown(message.content, id=f"user_msg_{index}"),
                        self.build_copy_button(message.content, "user", index),
                    ],
                )
            ],
        )

    def build_assistant_message(
        self, message: ChatMessage, index: int, direction: str = "ltr"
    ) -> DashComponent:
        return html.Div(
            className="assistant-message mb-3",
            dir=direction,
            children=[
                html.Div(
                    className="assistant-message-content",
                    children=[
                        dcc.Markdown(message.content, id=f"assistant_msg_{index}"),
                        self.build_copy_button(message.content, "assistant", index),
                    ],
                )
            ],
        )

    def build_copy_button(
        self, content: str, msg_type: str, index: int
    ) -> Optional[DashComponent]:
        """Build copy button with proper Bootstrap positioning, only if content is non-empty."""
        if content is None or str(content).strip() == "":
            return None
        return html.Div(
            [
                dcc.Clipboard(
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
