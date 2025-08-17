"""Default implementation for the layout builder."""

from abc import ABC, abstractmethod

import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.development.base_component import Component as DashComponent


class Layout(ABC):
    """Interface for building the Dash component layout."""

    @abstractmethod
    def build_layout(self) -> DashComponent:
        """Constructs and returns the entire Dash component tree for the UI."""
        pass


class Default(Layout):
    """Builds the standard, default layout for the chat application."""

    def build_layout(self) -> DashComponent:
        """Constructs the main layout Div."""
        return html.Div(
            className="d-flex flex-column vh-100",
            children=[
                dcc.Location(id="url_location", refresh=False),
                self.build_header(),
                dbc.Container(
                    fluid=True,
                    className="flex-grow-1 d-flex flex-column",
                    style={"minHeight": "0"},
                    children=[
                        dbc.Row(
                            justify="center",
                            className="flex-grow-1",
                            style={"minHeight": "0"},
                            children=[
                                dbc.Col(
                                    lg=7,
                                    md=12,
                                    className="d-flex flex-column",
                                    style={"minHeight": "0"},
                                    children=[
                                        self.build_chat_area(),
                                        self.build_input_area(),
                                    ],
                                )
                            ],
                        )
                    ],
                ),
                self.build_sidebar(),
            ],
        )

    def build_header(self) -> DashComponent:
        """Builds the header component."""
        return html.Header(
            className="p-2",
            style={"width": "300px"},
            children=[
                html.Button(
                    html.I(className="bi bi-list"),
                    id="sidebar_toggle_button",
                    n_clicks=0,
                    style={
                        "border": "none",
                        "background": "none",
                        "fontSize": "48px",
                        "display": "block",
                        "margin": "0 auto",
                    },
                ),
            ],
        )

    def build_sidebar(self) -> DashComponent:
        """Builds the sidebar component."""
        return dbc.Offcanvas(
            id="sidebar_offcanvas",
            is_open=False,
            backdrop=False,
            style={"width": "300px"},
            children=[
                html.Div(
                    [html.I(className="bi bi-pencil-square"), " New chat"],
                    id="new_chat_button",
                    n_clicks=0,
                    className="w-100 mb-3 p-2",
                    style={"cursor": "pointer", "textAlign": "center"},
                ),
                html.Div(id="convo_list_div", children=[]),
            ],
        )

    def build_chat_area(self) -> DashComponent:
        """Builds the main chat display area."""
        return html.Main(
            id="chat_area_main",
            className="flex-grow-1 p-3",
            style={"overflowY": "auto"},
        )

    def build_input_area(self) -> DashComponent:
        """Builds the user input area."""
        return html.Footer(
            className="p-3 position-sticky bottom-0 bg-white",
            style={"zIndex": "100"},
            children=[
                html.Div(
                    className="d-flex",
                    style={
                        "border": "1px solid #dee2e6",
                        "borderRadius": "25px",
                        "overflow": "hidden",
                    },
                    children=[
                        dbc.Textarea(
                            id="user_input_textarea",
                            placeholder="Ask...",
                            rows=4,
                            autoFocus=True,
                            style={"border": "none", "resize": "none"},
                            className="flex-grow-1",
                        ),
                        html.Button(
                            html.I(className="bi bi-send"),
                            id="chat_send_button",
                            n_clicks=0,
                            style={
                                "border": "none",
                                "background": "none",
                                "fontSize": "32px",
                                "padding": "4px",
                                "outline": "none",
                                "boxShadow": "none",
                            },
                        ),
                    ],
                )
            ],
        )
