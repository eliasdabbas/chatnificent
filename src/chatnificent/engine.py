"""
Defines the core orchestration logic for the Chatnificent application.
"""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional

from .models import (
    ASSISTANT_ROLE,
    SYSTEM_ROLE,
    USER_ROLE,
    Conversation,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from . import Chatnificent


class Engine(ABC):
    """Abstract Base Class for all Chatnificent Engines."""

    def __init__(
        self,
        app: Optional["Chatnificent"] = None,
        max_agentic_turns: int = 5,
    ) -> None:
        """Initialize with optional app reference.

        Can be bound later via app setter.

        Parameters
        ----------
        app : Optional[Chatnificent]
            The Chatnificent application instance.
        max_agentic_turns : int
            Maximum number of tool-calling loop iterations before forcing a
            final response. By default 5.
        """
        self.app = app
        self.max_agentic_turns = max_agentic_turns

    @abstractmethod
    def handle_message(
        self,
        user_input: str,
        user_id: str,
        convo_id_from_url: Optional[str],
    ) -> Conversation:
        """Process a user message and return the updated Conversation.

        Parameters
        ----------
        user_input : str
            The user's message text.
        user_id : str
            The authenticated user's identifier.
        convo_id_from_url : Optional[str]
            The conversation ID from the current URL, or None for a new chat.

        Returns
        -------
        Conversation
            The conversation with the new user and assistant messages appended.
        """
        pass

    @abstractmethod
    def handle_message_stream(
        self,
        user_input: str,
        user_id: str,
        convo_id_from_url: Optional[str],
    ) -> Generator[Dict[str, Any], None, None]:
        """Process a user message and yield SSE event dicts as the response streams.

        Each yielded dict has ``event`` and ``data`` keys:

        - ``{"event": "status", "data": "Calling tool: ..."}``
        - ``{"event": "delta",  "data": "token text"}``
        - ``{"event": "done",   "data": {"conversation_id": "..."}}``
        - ``{"event": "error",  "data": "error message"}``

        Parameters
        ----------
        user_input : str
            The user's message text.
        user_id : str
            The authenticated user's identifier.
        convo_id_from_url : Optional[str]
            The conversation ID from the current URL, or None for a new chat.

        Yields
        ------
        Dict[str, Any]
            Server-Sent Event dicts with ``event`` and ``data`` keys.
        """
        pass


class Orchestrator(Engine):
    """
    The default engine that processes requests with an agentic loop.

    Supports both non-streaming (``handle_message``) and streaming
    (``handle_message_stream``) request lifecycles.
    """

    def _initialize_conversation(
        self, user_input: str, user_id: str, convo_id_from_url: Optional[str]
    ) -> Conversation:
        """Resolve or create a conversation and append the user message."""
        conversation = self._resolve_conversation(user_id, convo_id_from_url)
        conversation = self._add_user_message(conversation, user_input)
        return conversation

    def _apply_retrieval_context(
        self, user_input: str, user_id: str, conversation: Conversation
    ) -> Optional[str]:
        """Run the retrieval pipeline and inject context into messages."""
        self._before_retrieval(user_input, conversation)
        retrieval_context = self._retrieve_context(
            user_input, user_id, conversation.id
        )
        self._after_retrieval(retrieval_context)

        if retrieval_context and not any(
            msg.get("role") == SYSTEM_ROLE for msg in conversation.messages
        ):
            system_message = {
                "role": SYSTEM_ROLE,
                "content": f"Context:\n---\n{retrieval_context}\n---",
            }
            conversation.messages.insert(0, system_message)

        return retrieval_context

    def handle_message(
        self,
        user_input: str,
        user_id: str,
        convo_id_from_url: Optional[str],
    ) -> Conversation:
        """Orchestrates the non-streaming, multi-turn agentic lifecycle."""

        conversation = None
        try:
            # 1. Initialization
            conversation = self._initialize_conversation(
                user_input, user_id, convo_id_from_url
            )

            # 2. Contextualization (RAG)
            retrieval_context = self._apply_retrieval_context(
                user_input, user_id, conversation
            )

            # 3. Agentic Loop
            llm_response = None
            tool_calls = None

            for _turn in range(self.max_agentic_turns):
                self._before_llm_call(conversation)
                llm_payload = self._prepare_llm_payload(conversation, retrieval_context)

                # Generation
                llm_response = self._generate_response(llm_payload)
                self._after_llm_call(llm_response)

                # Persist raw request/response pair for this turn
                self._save_raw_exchange(user_id, conversation.id, llm_response)

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
                tool_result_messages = self.app.llm.create_tool_result_messages(
                    tool_results, conversation
                )
                conversation.messages.extend(tool_result_messages)

            else:
                # Loop finished without breaking (Max turns reached)
                self._handle_max_turns(conversation)
                tool_calls = None

            # 4. Finalization
            if not tool_calls and llm_response is not None:
                text_content = self.app.llm.extract_content(llm_response)
                assistant_message = {"role": ASSISTANT_ROLE, "content": text_content}
                conversation.messages.append(assistant_message)

            # 5. Persistence
            self._before_save(conversation)
            self._save_conversation(conversation, user_id)

            return conversation

        except Exception as e:
            return self._handle_error(e, user_id, conversation)

    def handle_message_stream(
        self,
        user_input: str,
        user_id: str,
        convo_id_from_url: Optional[str],
    ) -> Generator[Dict[str, Any], None, None]:
        """Orchestrates the streaming, multi-turn agentic lifecycle."""

        conversation = None
        try:
            # 1. Initialization
            conversation = self._initialize_conversation(
                user_input, user_id, convo_id_from_url
            )

            # 2. Contextualization (RAG)
            retrieval_context = self._apply_retrieval_context(
                user_input, user_id, conversation
            )

            # 3. Agentic Loop
            llm_response = None
            tool_calls = None
            accumulated_text = ""

            for _turn in range(self.max_agentic_turns):
                self._before_llm_call(conversation)
                llm_payload = self._prepare_llm_payload(conversation, retrieval_context)

                has_tools = bool(self.app.tools.get_tools())

                if has_tools:
                    # Tool-calling turns always run non-streamed
                    llm_response = self._generate_response(
                        llm_payload, stream=False
                    )
                    self._after_llm_call(llm_response)
                    self._save_raw_exchange(user_id, conversation.id, llm_response)

                    tool_calls = self.app.llm.parse_tool_calls(llm_response)

                    if not tool_calls:
                        text_content = self.app.llm.extract_content(llm_response)
                        if text_content:
                            accumulated_text = text_content
                            yield {"event": "delta", "data": text_content}
                            self._on_stream_delta(text_content, accumulated_text)
                        break

                    # Emit any text content before executing tools
                    # (Anthropic can return text + tool_use in the same response)
                    text_content = self.app.llm.extract_content(llm_response)
                    if text_content:
                        accumulated_text += text_content
                        yield {"event": "delta", "data": text_content}
                        self._on_stream_delta(text_content, accumulated_text)

                    assistant_message = self.app.llm.create_assistant_message(
                        llm_response
                    )
                    conversation.messages.append(assistant_message)

                    for tool_call in tool_calls:
                        fn_name = tool_call.get("function_name", "unknown")
                        yield {
                            "event": "status",
                            "data": f"Calling tool: {fn_name}...",
                        }

                        result = self._execute_tools([tool_call])[0]

                        yield {"event": "status", "data": "Tool result received."}

                        tool_result_messages = (
                            self.app.llm.create_tool_result_messages(
                                [result], conversation
                            )
                        )
                        conversation.messages.extend(tool_result_messages)

                else:
                    # No tools — stream the response token-by-token
                    stream = self._generate_response(llm_payload)
                    for chunk in stream:
                        delta = self.app.llm.extract_stream_delta(chunk)
                        if delta:
                            accumulated_text += delta
                            yield {"event": "delta", "data": delta}
                            self._on_stream_delta(delta, accumulated_text)
                    break

            else:
                # Max turns reached — save before returning
                self._handle_max_turns(conversation)
                self._before_save(conversation)
                self._save_conversation(conversation, user_id)
                yield {
                    "event": "done",
                    "data": {"conversation_id": conversation.id},
                }
                return

            # 4. Finalization
            if accumulated_text:
                assistant_message = {
                    "role": ASSISTANT_ROLE,
                    "content": accumulated_text,
                }
                conversation.messages.append(assistant_message)

            # Save raw exchange for the streaming path
            self._save_raw_exchange(user_id, conversation.id, None)

            # 5. Persistence
            self._before_save(conversation)
            self._save_conversation(conversation, user_id)

            yield {"event": "done", "data": {"conversation_id": conversation.id}}

        except Exception as e:
            logger.exception("Error during streaming message handling")
            yield {"event": "error", "data": str(e)}

            if conversation and conversation.id:
                error_response = {
                    "role": ASSISTANT_ROLE,
                    "content": f"I encountered an error: {str(e)}. Please try again.",
                }
                conversation.messages.append(error_response)
                try:
                    self._save_conversation(conversation, user_id)
                except Exception:
                    logger.exception("Failed to save error conversation")

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
        user_message = {"role": USER_ROLE, "content": user_input.strip()}
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
        return list(conversation.messages)

    def _generate_response(self, llm_payload: List[Dict[str, Any]], **kwargs) -> Any:
        """[Seam] Executes the call to the LLM pillar.

        Parameters
        ----------
        llm_payload : List[Dict[str, Any]]
            The message list to send to the LLM.
        **kwargs : Any
            Additional keyword arguments forwarded to ``generate_response()``
            (e.g. ``stream=False`` to override a streaming-configured LLM).
        """
        tools = self.app.tools.get_tools()
        if tools:
            return self.app.llm.generate_response(llm_payload, tools=tools, **kwargs)
        else:
            return self.app.llm.generate_response(llm_payload, **kwargs)

    def _execute_tools(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """[Seam] Executes tools using the standardized protocol."""
        results = []
        for tool_call in tool_calls:
            result = self.app.tools.execute_tool_call(tool_call)
            results.append(result)
        return results

    def _handle_max_turns(self, conversation: Conversation):
        """[Seam] Handles the scenario where the agent loop reaches the limit."""
        logger.warning(f"Max agentic turns reached for conversation {conversation.id}")
        limit_message = {
            "role": ASSISTANT_ROLE,
            "content": (
                "I reached the maximum number of steps allowed. "
                "Please try rephrasing or simplifying your request."
            ),
        }
        conversation.messages.append(limit_message)

    def _save_conversation(self, conversation: Conversation, user_id: str) -> None:
        self.app.store.save_conversation(user_id, conversation)

    def _save_raw_exchange(
        self, user_id: str, convo_id: str, llm_response: Any
    ) -> None:
        """[Seam] Persist the raw API request/response pair for one LLM call."""
        request_payload = self.app.llm.get_last_request_payload()
        if hasattr(self.app.store, "save_raw_api_request") and request_payload:
            self.app.store.save_raw_api_request(user_id, convo_id, request_payload)

        if hasattr(self.app.store, "save_raw_api_response") and llm_response:
            try:
                response_to_save = llm_response.model_dump()
            except (AttributeError, TypeError):
                response_to_save = llm_response

            if isinstance(response_to_save, (dict, list)):
                self.app.store.save_raw_api_response(
                    user_id, convo_id, response_to_save
                )

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

    def _on_stream_delta(self, delta: str, accumulated: str) -> None:
        """[Hook] Called after each streamed text delta.

        Parameters
        ----------
        delta : str
            The new text chunk just received.
        accumulated : str
            The full accumulated text so far.
        """
        pass

    def _handle_error(
        self, error: Exception, user_id: str, conversation: Optional[Conversation]
    ) -> Conversation:
        logger.exception("Error during message handling")
        error_message = f"I encountered an error: {str(error)}. Please try again."

        if not conversation:
            conversation = Conversation(id="")

        error_response = {"role": ASSISTANT_ROLE, "content": error_message}
        conversation.messages.append(error_response)

        if conversation.id:
            try:
                self._save_conversation(conversation, user_id)
            except Exception:
                logger.exception("Failed to save error conversation")

        return conversation



