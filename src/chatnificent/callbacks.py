"""Atomic callback architecture for Chatnificent."""

from dash import ALL, Input, Output, State, callback_context, no_update

from .models import ASSISTANT_ROLE, USER_ROLE, ChatMessage, Conversation


def register_callbacks(app):
    @app.callback(
        [
            Output("messages_container", "children"),
            Output("input_textarea", "value"),
            Output("submit_button", "disabled"),
        ],
        [Input("submit_button", "n_clicks")],
        [
            State("input_textarea", "value"),
            State("url_location", "pathname"),
            State("url_location", "search"),
        ],
        running=[(Output("status_indicator", "hidden"), False, True)],
    )
    def send_message(n_clicks, user_input, pathname, search):
        if not n_clicks or not user_input or not user_input.strip():
            return no_update, no_update, no_update

        try:
            url_parts = app.url.parse(pathname, search)
            user_id = url_parts.user_id or app.auth.get_current_user_id(
                pathname=pathname
            )
            convo_id = url_parts.convo_id or app.store.get_next_conversation_id(user_id)

            conversation = app.store.load_conversation(user_id, convo_id)
            if not conversation:
                conversation = Conversation(id=convo_id)

            user_message = ChatMessage(role=USER_ROLE, content=user_input.strip())
            conversation.messages.append(user_message)

            message_dicts = [msg.model_dump() for msg in conversation.messages]
            raw_response = app.llm.generate_response(message_dicts)
            ai_content = app.llm.extract_content(raw_response)

            if hasattr(app.store, "save_raw_api_response"):
                try:
                    response_to_save = raw_response.model_dump()
                except AttributeError:
                    response_to_save = raw_response
                app.store.save_raw_api_response(user_id, convo_id, response_to_save)

            ai_message = ChatMessage(role=ASSISTANT_ROLE, content=ai_content)
            conversation.messages.append(ai_message)

            app.store.save_conversation(user_id, conversation)

            formatted_messages = app.layout_builder.build_messages(
                conversation.messages
            )

            return formatted_messages, "", False

        except Exception as e:
            error_message = f"I encountered an error: {str(e)}. Please try again."
            error_response = ChatMessage(role=ASSISTANT_ROLE, content=error_message)

            if "conversation" in locals():
                conversation.messages.append(error_response)
                app.store.save_conversation(user_id, conversation)

                formatted_messages = app.layout_builder.build_messages(
                    conversation.messages
                )
                return formatted_messages, "", False

            return [{"role": ASSISTANT_ROLE, "content": error_message}], "", False

    @app.callback(
        Output("messages_container", "children", allow_duplicate=True),
        [
            Input("url_location", "pathname"),
            Input("url_location", "search"),  # Add search for query params
        ],
        prevent_initial_call="initial_duplicate",
    )
    def load_conversation(pathname, search):
        try:
            url_parts = app.url.parse(pathname, search)
            convo_id = url_parts.convo_id
            if not convo_id:
                return []
            user_id = url_parts.user_id or app.auth.get_current_user_id(
                pathname=pathname
            )
            conversation = app.store.load_conversation(user_id, convo_id)

            if not conversation or not conversation.messages:
                return []

            return app.layout_builder.build_messages(conversation.messages)

        except Exception:
            return []

    @app.callback(
        [
            Output("url_location", "pathname", allow_duplicate=True),
            Output("sidebar", "hidden", allow_duplicate=True),
        ],
        [Input("new_conversation_button", "n_clicks")],
        [State("url_location", "pathname")],
        prevent_initial_call=True,
    )
    def create_new_chat(n_clicks, current_pathname):
        if not n_clicks:
            return no_update, no_update

        try:
            user_id = app.auth.get_current_user_id(pathname=current_pathname)
            new_path = app.url.build_new_chat_path(user_id)
            return new_path, True
        except Exception:
            return no_update, no_update

    @app.callback(
        [
            Output("url_location", "pathname", allow_duplicate=True),
            Output("sidebar", "hidden", allow_duplicate=True),
        ],
        [Input({"type": "convo-item", "id": ALL}, "n_clicks")],
        [State("url_location", "pathname")],
        prevent_initial_call=True,
    )
    def switch_conversation(n_clicks, current_pathname):
        if not any(n_clicks):
            return no_update, no_update

        try:
            ctx = callback_context
            selected_convo_id = ctx.triggered_id["id"]
            user_id = app.auth.get_current_user_id(pathname=current_pathname)
            new_path = app.url.build_conversation_path(user_id, selected_convo_id)
            return new_path, True
        except Exception:
            return no_update, no_update

    @app.callback(
        Output("sidebar", "hidden"),
        [Input("sidebar_toggle", "n_clicks")],
        [State("sidebar", "hidden")],
        prevent_initial_call=True,
    )
    def toggle_sidebar(toggle_clicks, is_hidden):
        if not toggle_clicks:
            return no_update
        return not is_hidden

    @app.callback(
        Output("conversations_list", "children"),
        [
            Input("url_location", "pathname"),
            Input("url_location", "search"),
            Input("messages_container", "children"),
        ],
    )
    def update_conversation_list(pathname, search, chat_messages):
        from dash import html

        try:
            # === REFACTORED URL LOGIC ===
            url_parts = app.url.parse(pathname, search)
            user_id = url_parts.user_id or app.auth.get_current_user_id(
                pathname=pathname
            )
            # ============================

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
    app.clientside_callback(
        """
        function(pathname) {
            // Set up enter to send functionality when page loads/changes
            setTimeout(function() {
                const textarea = document.getElementById('input_textarea');
                const submitButton = document.getElementById('submit_button');

                if (textarea && submitButton && !window.enterListenerSetup) {
                    // Set flag to avoid setting up multiple listeners
                    window.enterListenerSetup = true;

                    // Define the handler function
                    window.enterToSendHandler = function(e) {
                        if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();
                            // Only send if there's text content
                            if (textarea.value.trim()) {
                                submitButton.click();
                            }
                        }
                        // Shift+Enter will naturally create a newline (default behavior)
                    };

                    // Add the event listener
                    textarea.addEventListener('keydown', window.enterToSendHandler);
                }
            }, 100);

            return window.dash_clientside.no_update;
        }
        """,
        Output("submit_button", "n_clicks", allow_duplicate=True),
        [Input("url_location", "pathname")],
        prevent_initial_call=True,
    )

    # Auto-scroll to bottom
    app.clientside_callback(
        """
        function(messages_content) {
            if (messages_content && messages_content.length > 0) {
                setTimeout(function() {
                    const messagesContainer = document.getElementById('messages_container');
                    if (messagesContainer) {
                        messagesContainer.scrollTop = messagesContainer.scrollHeight;
                    }
                }, 100);
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output("messages_container", "data-scroll-trigger", allow_duplicate=True),
        [Input("messages_container", "children")],
        prevent_initial_call=True,
    )

    # Focus input after sending
    app.clientside_callback(
        """
        function(input_value) {
            if (input_value === "") {
                setTimeout(() => {
                    const textarea = document.getElementById('input_textarea');
                    if (textarea) {
                        textarea.focus();
                    }
                }, 100);
            }
            return {};
        }
        """,
        Output("input_textarea", "style", allow_duplicate=True),
        [Input("input_textarea", "value")],
        prevent_initial_call=True,
    )

    # Auto-detect RTL text
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
        Output("input_textarea", "dir", allow_duplicate=True),
        Input("input_textarea", "value"),
        prevent_initial_call=True,
    )
