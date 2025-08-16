"""
The main entrypoint for the Chatnificent package.

This module contains the primary Chatnificent class and the abstract base classes
(interfaces) for each of the extensible pillars. These interfaces form the
contract that enables the package's "hackability."
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import dash_bootstrap_components as dbc
from dash import (
    ALL,
    Dash,
    Input,
    Output,
    State,
    callback,
    callback_context,
    html,
    no_update,
)
from dash.development.base_component import Component as DashComponent

from .action_handlers import BaseActionHandler, NoActionHandler
from .auth_managers import BaseAuthManager, SingleUserAuthManager
from .knowledge_retrievers import BaseKnowledgeRetriever, NoKnowledgeRetriever
from .layout_builders import BaseLayoutBuilder, DefaultLayoutBuilder
from .llm_providers import BaseLLMProvider, OpenAIProvider
from .message_formatters import (
    BaseMessageFormatter,
    DefaultMessageFormatter,
)
from .models import ASSISTANT_ROLE, USER_ROLE, ChatMessage, Conversation
from .persistence_managers import (
    BasePersistenceManager,
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
            message_formatter
            if message_formatter is not None
            else DefaultMessageFormatter()
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

        @self.callback(
            Output("chat_area_main", "children"),
            Output("user_input_textarea", "value"),
            Output("convo_list_div", "children"),
            Input("chat_send_button", "n_clicks"),
            Input("url_location", "pathname"),
            State("user_input_textarea", "value"),
        )
        def handle_interaction(n_clicks, pathname, user_input_value):
            """Orchestrates the main chat interaction loop and view updates."""
            ctx = callback_context
            triggered_id = ctx.triggered_id

            user_id = self.auth_manager.get_current_user_id(pathname=pathname)

            # Determine conversation ID from URL
            try:
                convo_id = pathname.strip("/").split("/")[-1]
            except IndexError:
                convo_id = self.persistence_manager.get_next_conversation_id(user_id)

            # --- Handle User Message Submission ---
            if triggered_id == "chat_send_button" and n_clicks and user_input_value:
                conversation = self.persistence_manager.load_conversation(
                    user_id, convo_id
                )

                # Create new conversation if it doesn't exist
                if conversation is None:
                    conversation = Conversation(id=convo_id)

                user_message = ChatMessage(role=USER_ROLE, content=user_input_value)
                conversation.messages.append(user_message)

                message_dicts = [msg.model_dump() for msg in conversation.messages]

                assistant_response = self.llm_provider.generate_response(message_dicts)

                assistant_response_content = self.llm_provider.extract_content(
                    assistant_response
                )

                assistant_message = ChatMessage(
                    role=ASSISTANT_ROLE, content=assistant_response_content
                )
                conversation.messages.append(assistant_message)

                self.persistence_manager.save_conversation(user_id, conversation)

                formatted_messages = self.message_formatter.format_messages(
                    conversation.messages
                )

                # Build conversation list UI components
                conversation_ids = self.persistence_manager.list_conversations(user_id)
                conversation_list = []
                for convo_id in conversation_ids:
                    conv = self.persistence_manager.load_conversation(user_id, convo_id)
                    # Only include conversations that have messages
                    if conv and conv.messages:
                        first_message = next(
                            (msg for msg in conv.messages if msg.role == "user"), None
                        )
                        if first_message:
                            content = first_message.content
                            title = content[:50] + ("..." if len(content) > 50 else "")

                            conversation_list.append(
                                html.Div(
                                    title,
                                    id={"type": "convo-item", "id": convo_id},
                                    n_clicks=0,
                                    style={"cursor": "pointer"},
                                )
                            )

                return formatted_messages, "", conversation_list

            # --- Handle Page Load / URL Change ---
            conversation = self.persistence_manager.load_conversation(user_id, convo_id)
            if conversation is None:
                conversation = Conversation(id=convo_id)
            formatted_messages = self.message_formatter.format_messages(
                conversation.messages
            )

            conversation_ids = self.persistence_manager.list_conversations(user_id)
            conversation_list = []
            for convo_id in conversation_ids:
                conv = self.persistence_manager.load_conversation(user_id, convo_id)
                # Only include conversations that have messages
                if conv and conv.messages:
                    first_message = next(
                        (msg for msg in conv.messages if msg.role == "user"), None
                    )
                    if first_message:
                        content = first_message.content
                        title = content[:40] + ("..." if len(content) > 40 else "")

                        conversation_list.append(
                            html.Div(
                                title,
                                id={"type": "convo-item", "id": convo_id},
                                n_clicks=0,
                                style={"cursor": "pointer"},
                            )
                        )

            return formatted_messages, "", conversation_list

        @self.callback(
            Output("url_location", "pathname", allow_duplicate=True),
            Input({"type": "convo-item", "id": ALL}, "n_clicks"),
            State("url_location", "pathname"),
            prevent_initial_call=True,
        )
        def handle_conversation_selection(n_clicks, pathname):
            """Navigates to a selected conversation."""
            if not any(n_clicks):
                return no_update

            ctx = callback_context
            convo_id = ctx.triggered_id["id"]
            user_id = self.auth_manager.get_current_user_id(pathname=pathname)

            return f"/{user_id}/{convo_id}"

        @self.callback(
            Output("url_location", "pathname", allow_duplicate=True),
            Input("new_chat_button", "n_clicks"),
            State("url_location", "pathname"),
            prevent_initial_call=True,
        )
        def handle_new_chat(n_clicks, pathname):
            """Creates a new chat and navigates to it."""
            if not n_clicks:
                return no_update

            user_id = self.auth_manager.get_current_user_id(pathname=pathname)
            path_parts = [p for p in pathname.strip("/").split("/") if p]
            current_convo_id = path_parts[-1] if len(path_parts) >= 2 else None

            if current_convo_id:
                current_conversation = self.persistence_manager.load_conversation(
                    user_id, current_convo_id
                )

                if current_conversation and current_conversation.messages:
                    # Current conversation has messages, create a new one
                    new_convo_id = self.persistence_manager.get_next_conversation_id(
                        user_id
                    )
                    return f"/{user_id}/{new_convo_id}"
                else:
                    return no_update
            else:
                new_convo_id = self.persistence_manager.get_next_conversation_id(
                    user_id
                )
                return f"/{user_id}/{new_convo_id}"

        @self.callback(
            Output("sidebar_offcanvas", "is_open"),
            Input("sidebar_toggle_button", "n_clicks"),
            Input("new_chat_button", "n_clicks"),
            State("sidebar_offcanvas", "is_open"),
            prevent_initial_call=True,
        )
        def toggle_sidebar(hamburger_clicks, new_chat_clicks, is_open):
            """Toggles the sidebar's visibility."""
            ctx = callback_context
            triggered_id = ctx.triggered_id

            if triggered_id == "sidebar_toggle_button":
                return not is_open
            elif triggered_id == "new_chat_button":
                return False

            return is_open
