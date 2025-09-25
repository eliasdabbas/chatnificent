"""
The main entrypoint for the Chatnificent package.

This module contains the primary Chatnificent class and the abstract base classes
(interfaces) for each of the extensible pillars. These interfaces form the
contract that enables the package's hackability.
"""

__version__ = "0.0.10"

from typing import TYPE_CHECKING, Optional, Type

from dash import Dash

from . import auth, engine, layout, llm, models, retrieval, store, tools, url

if TYPE_CHECKING:
    from .engine import Engine


class Chatnificent(Dash):
    """
    The main class for the Chatnificent LLM Chat UI Framework.

    This class acts as the central orchestrator, using the injected pillar
    components to manage the application's behavior. The constructor uses
    concrete default implementations, making it easy to get started while
    remaining fully customizable.
    """

    def __init__(
        self,
        layout: Optional["layout.Layout"] = None,
        llm: Optional["llm.LLM"] = None,
        store: Optional["store.Store"] = None,
        auth: Optional["auth.Auth"] = None,
        tools: Optional["tools.Tool"] = None,
        retrieval: Optional["retrieval.Retrieval"] = None,
        url: Optional["url.URL"] = None,
        engine: Optional["engine.Engine"] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the Chatnificent application with configurable pillars.

        Parameters
        ----------
        layout : layout.Layout, optional
        llm : llm.LLM, optional
        store : store.Store, optional
        auth : auth.Auth, optional
        tools : tools.Tool, optional
        retrieval : retrieval.Retrieval, optional
        url : url.URL, optional
        engine : engine.Engine, optional
            The orchestration engine class to use for managing the request lifecycle.
            Defaults to engine.Synchronous.
        **kwargs
            Additional arguments passed to the Dash constructor.

        Notes
        -----
        The constructor automatically adds Bootstrap CSS and Bootstrap Icons
        to external_stylesheets if not already present.

        Raises
        ------
        ValueError
            If the layout is missing required component IDs needed for the chat
            functionality.

        Examples
        --------
        Basic usage with defaults:

        >>> app = Chatnificent()

        Custom configuration:

        >>> app = Chatnificent(
        ...     llm=llm.Anthropic(api_key="your-key"),
        ...     store=store.File(directory="./conversations"),
        ... )
        """
        if layout:
            self.layout_builder = layout
        else:
            try:
                from .layout import Bootstrap

                self.layout_builder = Bootstrap()
            except ImportError:
                import warnings

                warnings.warn(
                    "Chatnificent is running with a minimal layout because 'dash-bootstrap-components' is not installed. "
                    'For the default UI, install with: pip install "chatnificent[default]"',
                    UserWarning,
                )
                from .layout import Minimal

                self.layout_builder = Minimal()

        if llm:
            self.llm = llm
        else:
            try:
                from .llm import OpenAI

                self.llm = OpenAI()
            except ImportError:
                import warnings

                warnings.warn(
                    "Chatnificent is running with a simple EchoLLM because the 'openai' package is not installed. "
                    'For the default OpenAI integration, install with: pip install "chatnificent[default]"',
                    UserWarning,
                )
                from .llm import Echo

                self.llm = Echo()

        if "external_stylesheets" not in kwargs:
            kwargs["external_stylesheets"] = []
        kwargs["external_stylesheets"].extend(
            self.layout_builder.get_external_stylesheets()
        )

        if "external_scripts" not in kwargs:
            kwargs["external_scripts"] = []
        kwargs["external_scripts"].extend(self.layout_builder.get_external_scripts())

        super().__init__(**kwargs)

        if store is not None:
            self.store = store
        else:
            from .store import InMemory

            self.store = InMemory()

        if auth is not None:
            self.auth = auth
        else:
            from .auth import Anonymous

            self.auth = Anonymous()

        if tools is not None:
            self.tools = tools
        else:
            from .tools import NoTool

            self.tools = NoTool()
        if retrieval is not None:
            self.retrieval = retrieval
        else:
            from .retrieval import NoRetrieval

            self.retrieval = NoRetrieval()

        if url is not None:
            self.url = url
        else:
            from .url import PathBased

            self.url = PathBased()

        if engine:
            self.engine = engine
            self.engine.app = self
        else:
            from .engine import Synchronous

            self.engine = Synchronous(self)

        self.layout = self.layout_builder.build_layout()
        self._register_callbacks()

    def _register_callbacks(self) -> None:
        """Registers all the callbacks that orchestrate the pillars."""
        from .callbacks import register_callbacks

        register_callbacks(self)
