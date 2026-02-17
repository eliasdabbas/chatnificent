"""Server implementations for Chatnificent.

The Server pillar owns the HTTP transport layer: receiving requests,
delegating to the Engine, and formatting responses for the client.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from . import Chatnificent


class Server(ABC):
    """Abstract Base Class for all Chatnificent servers."""

    def __init__(self, app: Optional["Chatnificent"] = None) -> None:
        """Initialize with optional app reference (bound during Chatnificent init)."""
        self.app = app

    @abstractmethod
    def create_server(self, **kwargs) -> Any:
        """Create the underlying web application object.

        Parameters
        ----------
        **kwargs
            Server-specific configuration passed through from Chatnificent.

        Returns
        -------
        Any
            The web application object (e.g. Dash app, Starlette app).
        """
        pass

    @abstractmethod
    def run(self, **kwargs) -> None:
        """Start serving HTTP requests.

        Parameters
        ----------
        **kwargs
            Runtime options (host, port, debug, etc.).
        """
        pass


class DashServer(Server):
    """Full-stack Dash server with built-in UI.

    Uses the Layout pillar to render the chat interface and registers
    Dash callbacks that bridge user interactions to the Engine.
    """

    def create_server(self, **kwargs) -> Any:
        from dash import Dash

        layout_builder = self.app.layout_builder

        if "external_stylesheets" not in kwargs:
            kwargs["external_stylesheets"] = []
        kwargs["external_stylesheets"].extend(layout_builder.get_external_stylesheets())

        if "external_scripts" not in kwargs:
            kwargs["external_scripts"] = []
        kwargs["external_scripts"].extend(layout_builder.get_external_scripts())

        self.dash_app = Dash(**kwargs)
        self.dash_app.layout = layout_builder.build_layout()

        from .callbacks import register_callbacks

        register_callbacks(self.dash_app, self.app)

        return self.dash_app

    def run(self, **kwargs) -> None:
        self.dash_app.run(**kwargs)
