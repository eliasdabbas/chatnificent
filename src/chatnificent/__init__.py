"""
The main entrypoint for the Chatnificent package.

This module contains the primary Chatnificent class and the abstract base classes
(interfaces) for each of the extensible pillars. These interfaces form the
contract that enables the package's hackability.
"""

from typing import Optional

from dash import Dash

from . import auth, layout, llm, retrieval, store, tools, url


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
        llm: Optional[llm.LLM] = None,
        store: Optional[store.Store] = None,
        auth: Optional[auth.Auth] = None,
        tools: Optional[tools.Tool] = None,
        retrieval: Optional[retrieval.Retrieval] = None,
        url: Optional["url.URL"] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the Chatnificent application with configurable pillars.

        This constructor implements dependency injection for the framework's
        6 core pillars, each responsible for a specific aspect of the chat application.
        Default implementations are provided for immediate use, while custom
        implementations can be injected for specialized behavior.

        Parameters
        ----------
        layout : layout.Layout, optional
            Layout builder for constructing the Dash component tree.
            Defaults to layout.Default() which provides a standard chat UI.
        llm : llm.LLM, optional
            LLM provider for generating responses from language models.
            Defaults to llm.OpenAI() for OpenAI API integration.
        store : store.Store, optional
            Persistence manager for saving/loading conversations.
            Defaults to store.InMemory() for session-only storage.
        auth : auth.Auth, optional
            Authentication manager for user identification.
            Defaults to auth.SingleUser() for single-user mode.
        tools : tools.Tool, optional
            Tool handler for LLM function calling capabilities.
            Defaults to tools.NoTool() (no function calling).
        retrieval : retrieval.Retrieval, optional
            Knowledge retriever for RAG/context capabilities.
            Defaults to retrieval.NoRetrieval() (no RAG).
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

        llm_module = globals()["llm"]
        store_module = globals()["store"]
        auth_module = globals()["auth"]
        tools_module = globals()["tools"]
        retrieval_module = globals()["retrieval"]
        url_module = globals()["url"]

        if "external_stylesheets" not in kwargs:
            kwargs["external_stylesheets"] = []
        kwargs["external_stylesheets"].extend(
            self.layout_builder.get_external_stylesheets()
        )

        if "external_scripts" not in kwargs:
            kwargs["external_scripts"] = []
        kwargs["external_scripts"].extend(self.layout_builder.get_external_scripts())

        super().__init__(**kwargs)

        self.store = store if store is not None else store_module.InMemory()
        self.auth = auth if auth is not None else auth_module.SingleUser()
        self.tools = tools if tools is not None else tools_module.NoTool()
        self.retrieval = (
            retrieval if retrieval is not None else retrieval_module.NoRetrieval()
        )
        self.url = url if url is not None else url_module.PathBased()

        self.layout = self.layout_builder.build_layout()
        self._register_callbacks()

    def _register_callbacks(self) -> None:
        """Registers all the callbacks that orchestrate the pillars."""
        from .callbacks import register_callbacks

        register_callbacks(self)
