"""Concrete implementations for message formatters."""

import unicodedata
from abc import ABC, abstractmethod
from typing import List

import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.development.base_component import Component as DashComponent

from .models import USER_ROLE, ChatMessage


class Fmt(ABC):
    """Interface for converting message data into Dash components."""

    @abstractmethod
    def format_messages(self, messages: List[ChatMessage]) -> List[DashComponent]:
        """Converts a list of message models into renderable Dash components."""
        pass


class Default(Fmt):
    """The default formatter, rendering messages as simple styled divs."""

    def format_messages(self, messages: List[ChatMessage]) -> List[DashComponent]:
        if not messages:
            return []
        return [self.format_message(msg) for msg in messages]

    def format_message(self, message: ChatMessage) -> DashComponent:
        """Formats a single message."""
        style = {
            "padding": "10px",
            "borderRadius": "15px",
            "marginBottom": "10px",
            "maxWidth": "70%",
            "width": "fit-content",
        }
        if message.role == USER_ROLE:
            style["marginLeft"] = "auto"
            style["backgroundColor"] = "#efefef"
        else:
            style["marginRight"] = "auto"
            style["backgroundColor"] = "#ffffff"
            style["border"] = "1px solid #eee"

        return html.Div([dcc.Markdown(message.content)], style=style)


class Markdown(Fmt):
    """Enhanced formatter with markdown rendering, RTL support, and copy functionality."""

    def format_messages(self, messages: List[ChatMessage]) -> List[DashComponent]:
        """Converts messages into rich markdown components with copy functionality."""
        if not messages:
            return []

        formatted = []
        current_msg_direction = "ltr"

        for i, msg in enumerate(messages):
            if msg.role == USER_ROLE:
                current_msg_direction = "rtl" if self._is_rtl(msg.content) else "ltr"
                formatted.append(
                    self._format_user_message(msg, i, current_msg_direction)
                )
            else:
                # Use same direction as conversation for assistant messages
                formatted.append(
                    self._format_assistant_message(msg, i, current_msg_direction)
                )

        return formatted

    def _format_user_message(
        self, message: ChatMessage, index: int, direction: str
    ) -> DashComponent:
        """Formats user message with right alignment and RTL support."""
        return html.Div(
            className="mb-3",
            dir=direction,
            children=[
                html.Div(
                    className="row",
                    children=[
                        dbc.Col(
                            width=8,
                            className="ms-auto",
                            children=[
                                html.Div(
                                    dcc.Markdown(
                                        message.content,
                                        id=f"user_msg_{index}",
                                        className="p-3 rounded-3 bg-light text-end table",
                                    ),
                                    style={
                                        "width": "fit-content",
                                        "marginLeft": "auto",
                                    },
                                )
                            ],
                        )
                    ],
                ),
                html.Div(
                    className="row",
                    children=[
                        dbc.Col(
                            width=1,
                            className="ms-auto text-end",
                            children=[
                                self._create_copy_button(message.content, "user", index)
                            ],
                        )
                    ],
                ),
            ],
        )

    def _format_assistant_message(
        self, message: ChatMessage, index: int, direction: str
    ) -> DashComponent:
        """Formats assistant message with left alignment."""
        return html.Div(
            className="mb-3",
            dir=direction,
            children=[
                html.Div(
                    className="row",
                    children=[
                        dbc.Col(
                            width=12,
                            children=[
                                dcc.Markdown(
                                    message.content,
                                    id=f"assistant_msg_{index}",
                                    className="p-3 table",
                                )
                            ],
                        )
                    ],
                ),
                html.Div(
                    className="row",
                    children=[
                        dbc.Col(
                            width=1,
                            children=[
                                self._create_copy_button(
                                    message.content, "assistant", index
                                )
                            ],
                        )
                    ],
                ),
            ],
        )

    def _create_copy_button(
        self, content: str, msg_type: str, index: int
    ) -> DashComponent:
        """Creates a copy button with tooltip for message content."""
        button_id = f"clipboard_{msg_type}_{index}"
        return html.Div(
            [
                dcc.Clipboard(
                    content=content, id=button_id, style={"cursor": "pointer"}
                ),
                dbc.Tooltip("Copy", target=button_id, placement="top"),
            ]
        )

    def _is_rtl(self, text: str) -> bool:
        """Detects if text should be rendered right-to-left."""
        if not text or not text.strip():
            return False
        for char in text:
            bidi = unicodedata.bidirectional(char)
            if bidi in ("R", "AL"):
                return True
            elif bidi == "L":
                return False
        return False
