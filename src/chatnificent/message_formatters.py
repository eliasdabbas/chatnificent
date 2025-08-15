"""Concrete implementations for message formatters."""

from abc import ABC, abstractmethod
from typing import List

from dash import dcc, html
from dash.development.base_component import Component as DashComponent

from .models import USER_ROLE, ChatMessage


class BaseMessageFormatter(ABC):
    """Interface for converting message data into Dash components."""

    @abstractmethod
    def format_messages(self, messages: List[ChatMessage]) -> List[DashComponent]:
        """Converts a list of message models into renderable Dash components."""
        pass


class DefaultMessageFormatter(BaseMessageFormatter):
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
            style["backgroundColor"] = "#dcf8c6"
        else:
            style["marginRight"] = "auto"
            style["backgroundColor"] = "#ffffff"
            style["border"] = "1px solid #eee"

        return html.Div(dcc.Markdown(message.content), style=style)
