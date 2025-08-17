"""Atomic callback architecture for Chatnificent."""

from dash import ALL, Input, Output, State, callback_context, no_update

from .models import ASSISTANT_ROLE, USER_ROLE, ChatMessage, Conversation


def register_callbacks(app):

    @app.callback(
        [
            Output("chat_area_main", "children"),
            Output("user_input_textarea", "value"),
            Output("chat_send_button", "disabled"),
        ],
        [Input("chat_send_button", "n_clicks")],
        [State("user_input_textarea", "value"), State("url_location", "pathname")],
    )
    def send_message(n_clicks, user_input, pathname):
        if not n_clicks or not user_input or not user_input.strip():
            return no_update, no_update, no_update

        try:
            user_id = app.auth.get_current_user_id(pathname=pathname)
            path_parts = pathname.strip("/").split("/")

            if len(path_parts) >= 2 and path_parts[-1] and path_parts[-1] != "NEW":
                convo_id = path_parts[-1]
            else:
                convo_id = app.store.get_next_conversation_id(user_id)

            conversation = app.store.load_conversation(user_id, convo_id)
            if not conversation:
                conversation = Conversation(id=convo_id)

            user_message = ChatMessage(role=USER_ROLE, content=user_input.strip())
            conversation.messages.append(user_message)

            message_dicts = [msg.model_dump() for msg in conversation.messages]
            result = app.llm.generate_response(message_dicts)

            # Handle both old and new response formats for backward compatibility
            if isinstance(result, dict) and "content" in result:
                ai_content = result["content"]
                if "raw_response" in result and hasattr(
                    app.store, "save_raw_api_response"
                ):
                    app.store.save_raw_api_response(
                        user_id, convo_id, result["raw_response"]
                    )
            else:
                ai_content = app.llm.extract_content(result)

            ai_message = ChatMessage(role=ASSISTANT_ROLE, content=ai_content)
            conversation.messages.append(ai_message)

            app.store.save_conversation(user_id, conversation)

            formatted_messages = app.fmt.format_messages(
                conversation.messages
            )

            return formatted_messages, "", False

        except Exception as e:
            error_message = f"I encountered an error: {str(e)}. Please try again."
            error_response = ChatMessage(role=ASSISTANT_ROLE, content=error_message)

            if "conversation" in locals():
                conversation.messages.append(error_response)
                app.store.save_conversation(user_id, conversation)
                formatted_messages = app.fmt.format_messages(
                    conversation.messages
                )
                return formatted_messages, "", False

            return [{"role": ASSISTANT_ROLE, "content": error_message}], "", False

    @app.callback(
        Output("chat_area_main", "children", allow_duplicate=True),
        [Input("url_location", "pathname")],
        prevent_initial_call="initial_duplicate",
    )
    def load_conversation(pathname):
        try:
            user_id = app.auth.get_current_user_id(pathname=pathname)
            path_parts = pathname.strip("/").split("/")

            if not path_parts or not path_parts[-1] or path_parts[-1] == "NEW":
                return []

            convo_id = path_parts[-1]
            conversation = app.store.load_conversation(user_id, convo_id)

            if not conversation or not conversation.messages:
                return []

            return app.fmt.format_messages(conversation.messages)

        except Exception:
            return []

    @app.callback(
        [
            Output("url_location", "pathname", allow_duplicate=True),
            Output("sidebar_offcanvas", "is_open", allow_duplicate=True),
        ],
        [Input("new_chat_button", "n_clicks")],
        [State("url_location", "pathname")],
        prevent_initial_call=True,
    )
    def create_new_chat(n_clicks, current_pathname):
        if not n_clicks:
            return no_update, no_update

        try:
            user_id = app.auth.get_current_user_id(pathname=current_pathname)
            new_convo_id = app.store.get_next_conversation_id(user_id)
            new_path = f"/{user_id}/{new_convo_id}"
            return new_path, False
        except Exception:
            return no_update, no_update

    @app.callback(
        Output("url_location", "pathname", allow_duplicate=True),
        [Input({"type": "convo-item", "id": ALL}, "n_clicks")],
        [State("url_location", "pathname")],
        prevent_initial_call=True,
    )
    def switch_conversation(n_clicks, current_pathname):
        if not any(n_clicks):
            return no_update

        try:
            ctx = callback_context
            selected_convo_id = ctx.triggered_id["id"]
            user_id = app.auth.get_current_user_id(pathname=current_pathname)
            return f"/{user_id}/{selected_convo_id}"
        except Exception:
            return no_update

    @app.callback(
        Output("sidebar_offcanvas", "is_open"),
        [
            Input("sidebar_toggle_button", "n_clicks"),
            Input({"type": "convo-item", "id": ALL}, "n_clicks"),
        ],
        [State("sidebar_offcanvas", "is_open")],
        prevent_initial_call=True,
    )
    def toggle_sidebar(toggle_clicks, convo_clicks, is_open):
        ctx = callback_context
        triggered_id = ctx.triggered_id

        if triggered_id == "sidebar_toggle_button":
            return not is_open
        elif (
            isinstance(triggered_id, dict) and triggered_id.get("type") == "convo-item"
        ):
            return False

        return no_update

    @app.callback(
        Output("convo_list_div", "children"),
        [
            Input("url_location", "pathname"),
            Input("chat_area_main", "children"),
        ],  # Update when messages change
    )
    def update_conversation_list(pathname, chat_messages):
        from dash import html

        try:
            user_id = app.auth.get_current_user_id(pathname=pathname)
            conversation_ids = app.store.list_conversations(user_id)

            conversation_items = []
            for convo_id in conversation_ids:
                conv = app.store.load_conversation(user_id, convo_id)

                if conv and conv.messages:
                    first_user_msg = next(
                        (msg for msg in conv.messages if msg.role == USER_ROLE), None
                    )
                    if first_user_msg:
                        title = first_user_msg.content[:40] + (
                            "..." if len(first_user_msg.content) > 40 else ""
                        )

                        conversation_items.append(
                            html.Div(
                                title,
                                id={"type": "convo-item", "id": convo_id},
                                n_clicks=0,
                                style={
                                    "cursor": "pointer",
                                    "padding": "8px",
                                    "borderBottom": "1px solid #eee",
                                    "wordWrap": "break-word",
                                },
                            )
                        )

            return conversation_items

        except Exception:
            return []

    _register_clientside_callbacks(app)


def _register_clientside_callbacks(app):
    # ENTER key → click send button
    app.clientside_callback(
        """
        function(n_submit) {
            if (n_submit > 0) {
                const sendButton = document.getElementById('chat_send_button');
                if (sendButton && !sendButton.disabled) {
                    sendButton.click();
                }
            }
            return {};
        }
        """,
        Output("user_input_textarea", "style", allow_duplicate=True),
        [Input("user_input_textarea", "n_submit")],
        prevent_initial_call=True,
    )

    # Auto-scroll to bottom
    app.clientside_callback(
        """
        function(chat_content) {
            if (chat_content && chat_content.length > 0) {
                setTimeout(() => {
                    const chatArea = document.getElementById('chat_area_main');
                    if (chatArea) {
                        chatArea.scrollTop = chatArea.scrollHeight;
                    }
                }, 100);
            }
            return {};
        }
        """,
        Output("chat_area_main", "style", allow_duplicate=True),
        [Input("chat_area_main", "children")],
        prevent_initial_call=True,
    )

    # Focus input after sending
    app.clientside_callback(
        """
        function(input_value) {
            if (input_value === "") {
                setTimeout(() => {
                    const textarea = document.getElementById('user_input_textarea');
                    if (textarea) {
                        textarea.focus();
                    }
                }, 100);
            }
            return {};
        }
        """,
        Output("user_input_textarea", "style", allow_duplicate=True),
        [Input("user_input_textarea", "value")],
        prevent_initial_call=True,
    )

    app.clientside_callback(
        """
        function(textarea_value) {
            if (textarea_value) {
                const rtlPattern = '[\\u0590-\\u05ff\\u0600-\\u06ff\\u0750-\\u077f' +
                                   '\\u08a0-\\u08ff\\ufb1d-\\ufb4f\\ufb50-\\ufdff\\ufe70-\\ufeff]';
                const rtlRegex = new RegExp(rtlPattern);
                const isRTL = rtlRegex.test(textarea_value);
                return isRTL ? 'rtl' : 'ltr';
            }
            return 'ltr';
        }
        """,
        Output("user_input_textarea", "dir", allow_duplicate=True),
        Input("user_input_textarea", "value"),
        prevent_initial_call=True,
    )
