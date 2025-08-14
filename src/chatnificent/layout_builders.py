"""Default implementation for the layout builder."""

from abc import ABC, abstractmethod

import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.development.base_component import Component as DashComponent


class BaseLayoutBuilder(ABC):
    """Interface for building the Dash component layout."""

    @abstractmethod
    def build_layout(self) -> DashComponent:
        """Constructs and returns the entire Dash component tree for the UI."""
        pass


class DefaultLayoutBuilder(BaseLayoutBuilder):
    """Builds the standard, default layout for the chat application."""

    # def build_layout(self) -> DashComponent:
    #     return html.Div([html.H1("Hello world")])

    def build_layout(self) -> DashComponent:
        """Constructs the main layout Div."""
        return html.Div(
            className="d-flex flex-column vh-100",
            children=[
                dcc.Location(id="url", refresh=False),
                self.build_header(),
                html.Div(
                    className="d-flex flex-grow-1",
                    style={"overflow": "hidden"},
                    children=[
                        self.build_sidebar(),
                        self.build_chat_area(),
                    ],
                ),
                self.build_input_area(),
            ],
        )

    def build_header(self) -> DashComponent:
        """Builds the header component."""
        return html.Header(
            className="p-2 bg-light border-bottom",
            children=[
                dbc.Container(
                    fluid=True,
                    children=[
                        dbc.Row(
                            align="center",
                            children=[
                                dbc.Col(
                                    dbc.Button(
                                        "☰", id="open-sidebar-button", n_clicks=0
                                    ),
                                    width="auto",
                                ),
                                dbc.Col(html.H4("Chatnificent", className="m-0")),
                            ],
                        )
                    ],
                )
            ],
        )

    def build_sidebar(self) -> DashComponent:
        """Builds the sidebar component."""
        return dbc.Offcanvas(
            id="sidebar",
            is_open=False,
            title="Conversations",
            children=[
                dbc.ListGroup(id="conversation-list", children=[]),
                dbc.Button(
                    "New Chat",
                    id="new-chat-button",
                    color="primary",
                    className="w-100 mt-3",
                ),
            ],
        )

    def build_chat_area(self) -> DashComponent:
        """Builds the main chat display area."""
        return html.Main(
            id="chat-display", className="flex-grow-1 p-3", style={"overflowY": "auto"}
        )

    def build_input_area(self) -> DashComponent:
        """Builds the user input area."""
        return html.Footer(
            className="p-3 bg-light border-top",
            children=[
                dbc.InputGroup(
                    [
                        dbc.Textarea(id="user-input", placeholder="Type a message..."),
                        dbc.Button("Send", id="send-button", color="primary"),
                    ]
                )
            ],
        )
