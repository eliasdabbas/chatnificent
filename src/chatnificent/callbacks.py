"""Default callback handlers for the Chatnificent application."""

from dash import ALL, Input, Output, State, callback_context, html, no_update

from .models import ASSISTANT_ROLE, USER_ROLE, ChatMessage, Conversation


def register_callbacks(app):
    """Registers all default callbacks with the Chatnificent app instance."""

    @app.callback(
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

        user_id = app.auth_manager.get_current_user_id(pathname=pathname)

        # Determine conversation ID from URL
        try:
            convo_id = pathname.strip("/").split("/")[-1]
        except IndexError:
            convo_id = app.persistence_manager.get_next_conversation_id(user_id)

        # --- Handle User Message Submission ---
        if triggered_id == "chat_send_button" and n_clicks and user_input_value:
            conversation = app.persistence_manager.load_conversation(user_id, convo_id)

            # Create new conversation if it doesn't exist
            if conversation is None:
                conversation = Conversation(id=convo_id)

            user_message = ChatMessage(role=USER_ROLE, content=user_input_value)
            conversation.messages.append(user_message)

            message_dicts = [msg.model_dump() for msg in conversation.messages]

            assistant_response = app.llm_provider.generate_response(message_dicts)

            assistant_response_content = app.llm_provider.extract_content(
                assistant_response
            )

            assistant_message = ChatMessage(
                role=ASSISTANT_ROLE, content=assistant_response_content
            )
            conversation.messages.append(assistant_message)

            app.persistence_manager.save_conversation(user_id, conversation)

            formatted_messages = app.message_formatter.format_messages(
                conversation.messages
            )

            # Build conversation list UI components
            conversation_ids = app.persistence_manager.list_conversations(user_id)
            conversation_list = []
            for convo_id in conversation_ids:
                conv = app.persistence_manager.load_conversation(user_id, convo_id)
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
        conversation = app.persistence_manager.load_conversation(user_id, convo_id)
        if conversation is None:
            conversation = Conversation(id=convo_id)
        formatted_messages = app.message_formatter.format_messages(
            conversation.messages
        )

        conversation_ids = app.persistence_manager.list_conversations(user_id)
        conversation_list = []
        for convo_id in conversation_ids:
            conv = app.persistence_manager.load_conversation(user_id, convo_id)
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

    @app.callback(
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
        user_id = app.auth_manager.get_current_user_id(pathname=pathname)

        return f"/{user_id}/{convo_id}"

    @app.callback(
        Output("url_location", "pathname", allow_duplicate=True),
        Input("new_chat_button", "n_clicks"),
        State("url_location", "pathname"),
        prevent_initial_call=True,
    )
    def handle_new_chat(n_clicks, pathname):
        """Creates a new chat and navigates to it."""
        if not n_clicks:
            return no_update

        user_id = app.auth_manager.get_current_user_id(pathname=pathname)
        path_parts = [p for p in pathname.strip("/").split("/") if p]
        current_convo_id = path_parts[-1] if len(path_parts) >= 2 else None

        if current_convo_id:
            current_conversation = app.persistence_manager.load_conversation(
                user_id, current_convo_id
            )

            if current_conversation and current_conversation.messages:
                # Current conversation has messages, create a new one
                new_convo_id = app.persistence_manager.get_next_conversation_id(user_id)
                return f"/{user_id}/{new_convo_id}"
            else:
                return no_update
        else:
            new_convo_id = app.persistence_manager.get_next_conversation_id(user_id)
            return f"/{user_id}/{new_convo_id}"

    @app.callback(
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
