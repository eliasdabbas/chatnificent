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
    actions,
    auth,
    fmt,
    layout,
    llm,
    retrieval,
    store,
    themes,
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
        action: Optional[actions.Action] = None,
        retrieval: Optional[retrieval.Retrieval] = None,
        # --- Other Dash kwargs ---
        **kwargs,
    ):
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
        actions_module = globals()["actions"]
        retrieval_module = globals()["retrieval"]

        self.layout_builder = layout if layout is not None else layout_module.Default()
        self.llm = llm if llm is not None else llm_module.OpenAI()
        self.store = store if store is not None else store_module.InMemory()
        self.fmt = fmt if fmt is not None else fmt_module.Markdown()
        self.auth = auth if auth is not None else auth_module.SingleUser()
        self.actions = action if action is not None else actions_module.NoAction()
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
