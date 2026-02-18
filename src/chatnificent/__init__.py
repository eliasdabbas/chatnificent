"""
The main entrypoint for the Chatnificent package.

This module contains the primary Chatnificent class and the abstract base classes
(interfaces) for each of the extensible pillars. These interfaces form the
contract that enables the package's hackability.
"""

from importlib.metadata import version as _get_version

__version__ = _get_version("chatnificent")

from typing import Optional

from . import auth, engine, layout, llm, models, retrieval, server, store, tools, url


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
            The HTTP transport layer. Defaults to DevServer
            (zero-dependency stdlib server).
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
            self.layout = layout
        else:
            from .layout import DefaultLayout

            self.layout = DefaultLayout()

        if llm:
            self.llm = llm
        else:
            try:
                from .llm import OpenAI

                self.llm = OpenAI()
            except ImportError:
                import warnings

                warnings.warn(
                    "No LLM provider SDK found — falling back to EchoLLM (mirrors your input). "
                    "Install a provider, e.g.: pip install 'chatnificent[openai]', "
                    "'chatnificent[anthropic]', 'chatnificent[gemini]', or 'chatnificent[ollama]'",
                    UserWarning,
                    stacklevel=2,
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
            from .server import DevServer

            self.server = DevServer()

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
