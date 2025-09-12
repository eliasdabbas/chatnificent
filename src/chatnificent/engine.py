"""
Defines the core orchestration logic for the Chatnificent application.
"""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

# Import dash.no_update for use in the output builder
try:
    from dash import no_update
except ImportError:
    # Fallback for environments where Dash might not be present (e.g. unit tests).
    # This allows the engine logic to be tested independently of the Dash framework.
    class _NoUpdate:
        def __repr__(self):
            return "no_update"

    no_update = _NoUpdate()


from .models import (
    ASSISTANT_ROLE,
    MODEL_ROLE,
    SYSTEM_ROLE,
    TOOL_ROLE,
    USER_ROLE,
    ChatMessage,
    Conversation,
    ToolCall,
    ToolResult,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from . import Chatnificent


class Engine(ABC):
    """Abstract Base Class for all Chatnificent Engines."""

    def __init__(self, app: Optional["Chatnificent"] = None) -> None:
        """Initialize with optional app reference (can be bound later via app setter)."""
        self.app = app

    @abstractmethod
    def handle_message(
        self,
        user_input: str,
        user_id: str,
        convo_id_from_url: Optional[str],
    ) -> Dict[str, Any]:
        """The main public entry point for processing a user's message."""
        pass


class Synchronous(Engine):
    """
    The default engine that processes a request synchronously with an agentic loop.
    """

    MAX_AGENTIC_TURNS = 5

    def handle_message(
        self,
        user_input: str,
        user_id: str,
        convo_id_from_url: Optional[str],
    ) -> Dict[str, Any]:
        """Orchestrates the synchronous, multi-turn agentic lifecycle."""

        conversation = None
        try:
            # 1. Initialization
            conversation = self._resolve_conversation(user_id, convo_id_from_url)
            conversation = self._add_user_message(conversation, user_input)

            # 2. Contextualization (RAG)
            self._before_retrieval(user_input, conversation)
            retrieval_context = self._retrieve_context(
                user_input, user_id, conversation.id
            )
            self._after_retrieval(retrieval_context)

            if retrieval_context and not any(
                msg.role == SYSTEM_ROLE for msg in conversation.messages
            ):
                system_message = ChatMessage(
                    role=SYSTEM_ROLE, content=f"Context:\n---\n{retrieval_context}\n---"
                )
                conversation.messages.insert(0, system_message)

            # 3. Agentic Loop
            llm_response = None
            tool_calls = None  # Initialize scope

            for turn in range(self.MAX_AGENTIC_TURNS):
                self._before_llm_call(conversation)
                llm_payload = self._prepare_llm_payload(conversation, retrieval_context)

                # Generation
                llm_response = self._generate_response(llm_payload)
                self._after_llm_call(llm_response)

                # Parsing (Adapter)
                tool_calls = self.app.llm.parse_tool_calls(llm_response)

                # Decision Point
                if not tool_calls:
                    break

                # Add Assistant Message (Tool Request) using Adapter
                assistant_message = self.app.llm.create_assistant_message(llm_response)
                conversation.messages.append(assistant_message)

                # Execution (Runtime)
                tool_results = self._execute_tools(tool_calls)

                # Add Tool Results using Adapter
                # Note: This uses the plural method to allow provider-specific batching (e.g., Anthropic)
                tool_result_messages = self.app.llm.create_tool_result_messages(
                    tool_results, conversation
                )
                conversation.messages.extend(tool_result_messages)

            else:
                # Loop finished without breaking (Max turns reached)
                self._handle_max_turns(conversation)
                # Clear tool_calls to ensure the final response logic below is triggered correctly
                tool_calls = None

            # 4. Finalization
            # Add the final assistant message if the loop broke normally and a response exists
            if not tool_calls and llm_response is not None:
                # Use the LLM adapter to create the final message with text content
                text_content = self.app.llm.extract_content(llm_response)
                assistant_message = ChatMessage(
                    role=ASSISTANT_ROLE, content=text_content
                )
                conversation.messages.append(assistant_message)

            # 5. Persistence
            self._before_save(conversation)
            self._save_conversation(conversation, user_id, llm_response)

            # 6. Output
            return self._build_output(conversation, convo_id_from_url, user_id)

        except Exception as e:
            return self._handle_error(e, user_id, conversation)

    # =========================================================================
    # Seams (Core Logic - Overridable)
    # =========================================================================

    def _resolve_conversation(
        self, user_id: str, convo_id: Optional[str]
    ) -> Conversation:
        conversation = None
        if convo_id:
            conversation = self.app.store.load_conversation(user_id, convo_id)
        if not conversation:
            new_convo_id = self.app.store.get_next_conversation_id(user_id)
            conversation = Conversation(id=new_convo_id)
        return conversation

    def _add_user_message(
        self, conversation: Conversation, user_input: str
    ) -> Conversation:
        user_message = ChatMessage(role=USER_ROLE, content=user_input.strip())
        conversation.messages.append(user_message)
        return conversation

    def _retrieve_context(
        self, query: str, user_id: str, convo_id: str
    ) -> Optional[str]:
        if self.app.retrieval:
            return self.app.retrieval.retrieve(query, user_id, convo_id)
        return None

    def _prepare_llm_payload(
        self, conversation: Conversation, retrieval_context: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        messages = [msg.model_dump(exclude_none=True) for msg in conversation.messages]
        return messages

    def _generate_response(self, llm_payload: List[Dict[str, Any]]) -> Any:
        """[Seam] Executes the synchronous call to the LLM pillar."""
        # Future enhancement: Determine required tool format based on LLM type if Tools pillar supports translation.
        tools = self.app.tools.get_tools()
        if tools:
            return self.app.llm.generate_response(llm_payload, tools=tools)
        else:
            return self.app.llm.generate_response(llm_payload)

    def _execute_tools(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """[Seam] Executes tools using the standardized protocol."""
        results = []
        for tool_call in tool_calls:
            # The Tool pillar handles execution and internal error management.
            result = self.app.tools.execute_tool_call(tool_call)
            results.append(result)
        return results

    def _handle_max_turns(self, conversation: Conversation):
        """[Seam] Handles the scenario where the agent loop reaches the limit."""
        logger.warning(f"Max agentic turns reached for conversation {conversation.id}")
        limit_message = ChatMessage(
            role=ASSISTANT_ROLE,
            content="I reached the maximum number of steps allowed. Please try rephrasing or simplifying your request.",
        )
        conversation.messages.append(limit_message)

    def _save_conversation(
        self, conversation: Conversation, user_id: str, llm_response: Any
    ) -> None:
        self.app.store.save_conversation(user_id, conversation)
        if hasattr(self.app.store, "save_raw_api_response") and llm_response:
            try:
                response_to_save = llm_response.model_dump()
            except (AttributeError, TypeError):
                response_to_save = llm_response  # Fallback for non-Pydantic objects

            # Basic check for serializability
            if isinstance(response_to_save, (dict, list)):
                self.app.store.save_raw_api_response(
                    user_id, conversation.id, response_to_save
                )

    def _build_output(
        self, conversation: Conversation, convo_id_from_url: Optional[str], user_id: str
    ) -> Dict[str, Any]:
        display_messages = [
            msg
            for msg in conversation.messages
            if (
                not self.app.llm.is_tool_message(msg)
                and msg.role != SYSTEM_ROLE
                and msg.content is not None
                and str(msg.content).strip() != ""
            )
        ]
        formatted_messages = self.app.layout_builder.build_messages(display_messages)

        new_pathname = no_update
        if convo_id_from_url != conversation.id:
            new_pathname = self.app.url.build_conversation_path(
                user_id, conversation.id
            )
        return {
            "messages": formatted_messages,
            "input_value": "",
            "submit_disabled": False,
            "pathname": new_pathname,
        }

    # =========================================================================
    # Hooks (Extensibility Points - Empty by default)
    # =========================================================================

    def _before_retrieval(self, user_input: str, conversation: Conversation) -> None:
        pass

    def _after_retrieval(self, retrieval_context: Optional[str]) -> None:
        pass

    def _before_llm_call(self, conversation: Conversation) -> None:
        pass

    def _after_llm_call(self, llm_response: Any) -> None:
        pass

    def _before_save(self, conversation: Conversation) -> None:
        pass

    def _handle_error(
        self, error: Exception, user_id: str, conversation: Optional[Conversation]
    ) -> Dict[str, Any]:
        logger.exception("Error during message handling")
        error_message = f"I encountered an error: {str(error)}. Please try again."

        if conversation:
            # If we have a conversation context, append error and use the standard output builder
            error_response = ChatMessage(role=ASSISTANT_ROLE, content=error_message)
            conversation.messages.append(error_response)
            # Pass None for llm_response as the call failed
            self._save_conversation(conversation, user_id, None)
            # Use the existing conversation ID for the URL context (convo_id_from_url)
            return self._build_output(conversation, conversation.id, user_id)
        else:
            # If the error occurred before the conversation was established.
            # CRITICAL: We must format the error using the layout builder to return Dash components.
            error_msg_obj = ChatMessage(role=ASSISTANT_ROLE, content=error_message)
            formatted_error = self.app.layout_builder.build_messages([error_msg_obj])

            return {
                "messages": formatted_error,
                "input_value": "",
                "submit_disabled": False,
                "pathname": no_update,
            }


class Streaming(Engine):
    def handle_message(self, *args, **kwargs) -> Any:
        raise NotImplementedError("StreamingEngine is not yet implemented.")
