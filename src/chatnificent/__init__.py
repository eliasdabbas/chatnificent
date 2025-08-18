"""
The main entrypoint for the Chatnificent package.

This module contains the primary Chatnificent class and the abstract base classes
(interfaces) for each of the extensible pillars. These interfaces form the
contract that enables the package's "hackability."
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import dash_bootstrap_components as dbc
from dash import Dash

from . import (
    auth,
    fmt,
    layout,
    llm,
    retrieval,
    store,
    themes,
    tools,
)


class Chatnificent(Dash):
    """
    The main class for the Chatnificent LLM Chat UI Framework.

    This class acts as the central orchestrator, using the injected "pillar"
    components to manage the application's behavior. The constructor uses
    concrete default implementations, making it easy to get started while
    remaining fully customizable.
    """

    def __init__(
        self,
        layout: Optional[layout.Layout] = None,
        llm: Optional[llm.LLM] = None,
        store: Optional[store.Store] = None,
        fmt: Optional[fmt.Fmt] = None,
        auth: Optional[auth.Auth] = None,
        tools: Optional[tools.Tool] = None,
        retrieval: Optional[retrieval.Retrieval] = None,
        # --- Other Dash kwargs ---
        **kwargs,
    ):
        """
        Initialize the Chatnificent application with configurable pillars.

        This constructor implements dependency injection for the framework's
        7 core "pillars",
        each responsible for a specific aspect of the chat application.
        Default implementations
        are provided for immediate use, while custom implementations can be injected for
        specialized behavior.

        Parameters
        ----------
        layout : layout.Layout, optional
            Layout builder for constructing the Dash component tree.
            Defaults to layout.Default() which provides a standard chat UI.
        llm : llm.LLM, optional
            LLM provider for generating responses from language models.
            Defaults to llm.OpenAI() for OpenAI GPT integration.
        store : store.Store, optional
            Persistence manager for saving/loading conversations.
            Defaults to store.InMemory() for session-only storage.
        fmt : fmt.Fmt, optional
            Message formatter for converting messages to Dash components.
            Defaults to fmt.Markdown() for rich text rendering.
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
            If the layout is missing required component IDs needed
            for the chat functionality.

        Examples
        --------
        Basic usage with defaults:

        >>> app = Chatnificent()

        Custom configuration:

        >>> app = Chatnificent(
        ...     llm=llm.Anthropic(api_key="your-key"),
        ...     store=store.File(directory="./conversations"),
        ...     fmt=fmt.Code(),  # For code-focused formatting
        ...     tools=tools.Calculator(),  # Add function calling
        ... )
        """
        if "external_stylesheets" not in kwargs:
            kwargs["external_stylesheets"] = []

        if dbc.themes.BOOTSTRAP not in kwargs["external_stylesheets"]:
            kwargs["external_stylesheets"].append(dbc.themes.BOOTSTRAP)

        bootstrap_icons_cdn = "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.0/font/bootstrap-icons.css"
        if bootstrap_icons_cdn not in kwargs["external_stylesheets"]:
            kwargs["external_stylesheets"].append(bootstrap_icons_cdn)
        super().__init__(**kwargs)
        # This pattern avoids the mutable default argument issue.
        # Use globals() to access modules that are shadowed by parameters
        layout_module = globals()["layout"]
        llm_module = globals()["llm"]
        store_module = globals()["store"]
        fmt_module = globals()["fmt"]
        auth_module = globals()["auth"]
        tools_module = globals()["tools"]
        retrieval_module = globals()["retrieval"]

        self.layout_builder = layout if layout is not None else layout_module.Default()
        self.llm = llm if llm is not None else llm_module.OpenAI()
        self.store = store if store is not None else store_module.InMemory()
        self.fmt = fmt if fmt is not None else fmt_module.Markdown()
        self.auth = auth if auth is not None else auth_module.SingleUser()
        self.tools = tools if tools is not None else tools_module.NoTool()
        self.retrieval = (
            retrieval if retrieval is not None else retrieval_module.NoRetrieval()
        )

        # Build the layout using the injected builder
        self.layout = self.layout_builder.build_layout()
        self._validate_layout()
        self._register_callbacks()

    def _validate_layout(self):
        """Ensures the layout contains all required component IDs."""
        required_ids = {
            "url_location",
            "chat_area_main",
            "user_input_textarea",
            "convo_list_div",
            "chat_send_button",
            "new_chat_button",
            "sidebar_offcanvas",
            "sidebar_toggle_button",
        }

        found_ids = set()

        def collect_ids(component):
            """Recursively collects all component IDs from the layout tree."""
            if hasattr(component, "id") and component.id:
                if isinstance(component.id, str):
                    found_ids.add(component.id)

            if hasattr(component, "children"):
                children = component.children
                if children is None:
                    return
                elif isinstance(children, list):
                    for child in children:
                        if child is not None:
                            collect_ids(child)
                else:
                    collect_ids(children)

        collect_ids(self.layout)

        missing_ids = required_ids - found_ids
        if missing_ids:
            raise ValueError(
                f"Layout validation failed. Missing required component IDs: "
                f"{sorted(missing_ids)}"
            )

    def _register_callbacks(self):
        """Registers all the callbacks that orchestrate the pillars."""
        from .callbacks import register_callbacks

        register_callbacks(self)
