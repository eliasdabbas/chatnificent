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

from .action_handlers import BaseActionHandler, NoActionHandler
from .auth_managers import BaseAuthManager, SingleUserAuthManager
from .knowledge_retrievers import BaseKnowledgeRetriever, NoKnowledgeRetriever
from .layout_builders import BaseLayoutBuilder, DefaultLayoutBuilder
from .llm_providers import BaseLLMProvider, OpenAIProvider
from .message_formatters import (
    BaseMessageFormatter,
    DefaultMessageFormatter,
    MarkdownFormatter,
)
from .persistence_managers import (
    BasePersistenceManager,
    FilePersistenceManager,
    InMemoryPersistenceManager,
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
        # --- Pillar Injection using Pythonic None-as-default pattern ---
        layout_builder: Optional[BaseLayoutBuilder] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
        persistence_manager: Optional[BasePersistenceManager] = None,
        message_formatter: Optional[BaseMessageFormatter] = None,
        auth_manager: Optional[BaseAuthManager] = None,
        action_handler: Optional[BaseActionHandler] = None,
        knowledge_retriever: Optional[BaseKnowledgeRetriever] = None,
        # --- Other Dash kwargs ---
        **kwargs,
    ):
        if "external_stylesheets" not in kwargs:
            kwargs["external_stylesheets"] = []

        if dbc.themes.BOOTSTRAP not in kwargs["external_stylesheets"]:
            kwargs["external_stylesheets"].append(dbc.themes.BOOTSTRAP)

        # Add Bootstrap Icons
        bootstrap_icons_cdn = "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.0/font/bootstrap-icons.css"
        if bootstrap_icons_cdn not in kwargs["external_stylesheets"]:
            kwargs["external_stylesheets"].append(bootstrap_icons_cdn)
        super().__init__(**kwargs)
        # This pattern avoids the mutable default argument issue.
        self.layout_builder = (
            layout_builder if layout_builder is not None else DefaultLayoutBuilder()
        )
        self.llm_provider = (
            llm_provider if llm_provider is not None else OpenAIProvider()
        )
        self.persistence_manager = (
            persistence_manager
            if persistence_manager is not None
            else InMemoryPersistenceManager()
        )
        self.message_formatter = (
            message_formatter if message_formatter is not None else MarkdownFormatter()
        )
        self.auth_manager = (
            auth_manager if auth_manager is not None else SingleUserAuthManager()
        )
        self.action_handler = (
            action_handler if action_handler is not None else NoActionHandler()
        )
        self.knowledge_retriever = (
            knowledge_retriever
            if knowledge_retriever is not None
            else NoKnowledgeRetriever()
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
                f"Layout validation failed. Missing required component IDs: {sorted(missing_ids)}"
            )

    def _register_callbacks(self):
        """Registers all the callbacks that orchestrate the pillars."""
        from .callbacks import register_callbacks

        register_callbacks(self)
