"""
The main entrypoint for the Chatnificent package.

This module contains the primary Chatnificent class and the abstract base classes
(interfaces) for each of the extensible pillars. These interfaces form the
contract that enables the package's hackability.
"""

from importlib.metadata import version as _get_version

__version__ = _get_version("chatnificent")

from typing import TYPE_CHECKING, Optional, Type

from . import auth, engine, llm, models, retrieval, server, store, tools, url

if TYPE_CHECKING:
    from .engine import Engine


class Chatnificent:
    """
    The main class for the Chatnificent LLM Chat UI Framework.

    This class acts as the central orchestrator, using the injected pillar
    components to manage the application's behavior. The constructor uses
    concrete default implementations, making it easy to get started while
    remaining fully customizable.
    """

    def __init__(
        self,
        server: Optional["server.Server"] = None,
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
        server : server.Server, optional
            The HTTP transport layer. Defaults to DashServer if Dash is
            installed, otherwise DevServer (zero-dependency stdlib server).
        layout : layout.Layout, optional
        llm : llm.LLM, optional
        store : store.Store, optional
        auth : auth.Auth, optional
        tools : tools.Tool, optional
        retrieval : retrieval.Retrieval, optional
        url : url.URL, optional
        engine : engine.Engine, optional
            The orchestration engine. Defaults to engine.Synchronous.
        **kwargs
            Additional arguments passed to the Server's create_server method.

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
            self.layout_builder = None

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

        if server is not None:
            self.server = server
        else:
            try:
                import dash  # noqa: F401
                from .server import DashServer

                self.server = DashServer()
            except ImportError:
                from .server import DevServer

                self.server = DevServer()

        # Auto-resolve layout for DashServer if not explicitly provided
        try:
            from .server import DashServer as _DashServer
        except Exception:
            _DashServer = None

        if _DashServer is not None and isinstance(self.server, _DashServer) and self.layout_builder is None:
            try:
                from .layout import Bootstrap

                self.layout_builder = Bootstrap()
            except ImportError:
                try:
                    from .layout import Minimal

                    self.layout_builder = Minimal()
                except ImportError:
                    import warnings

                    warnings.warn(
                        "DashServer requires a Layout pillar but Dash components are not installed. "
                        'Install with: pip install "chatnificent[dash]"',
                        UserWarning,
                    )

        self.server.app = self
        self.server.create_server(**kwargs)

    def run(self, **kwargs) -> None:
        """Start the application server.

        Parameters
        ----------
        **kwargs
            Runtime options passed to the server (host, port, debug, etc.).
        """
        self.server.run(**kwargs)
