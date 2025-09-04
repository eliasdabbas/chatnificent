"""Concrete implementations for LLM providers."""

import json
import logging
import os
import secrets
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Union

from .models import (
    ASSISTANT_ROLE,
    MODEL_ROLE,
    TOOL_ROLE,
    USER_ROLE,
    ChatMessage,
    ToolCall,
    ToolResult,
)

logger = logging.getLogger(__name__)


class LLM(ABC):
    """Abstract Base Class for all LLM providers."""

    @abstractmethod
    def generate_response(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Communicates with the LLM SDK and returns the native response object."""
        pass

    @abstractmethod
    def extract_content(self, response: Any) -> Optional[str]:
        """Extracts human-readable text from the native response."""
        pass

    def parse_tool_calls(self, response: Any) -> Optional[List[ToolCall]]:
        """Translates the native response into the standardized format."""
        return None

    def create_assistant_message(self, response: Any) -> ChatMessage:
        """Converts the native response int a ChatMessage for persistence."""
        content = self.extract_content(response)
        return ChatMessage(role=ASSISTANT_ROLE, content=content)

    def create_tool_result_messages(
        self, results: List[ToolResult]
    ) -> List[ChatMessage]:
        """Converts tool result objects into ChatMessage instances for persistence."""
        if results:
            raise NotImplementedError(
                f"{self.__class__.__name__} This method must be implemented by subclasses."
            )
        return []


class _OpenAICompatible(LLM):
    def generate_response(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> Any:
        api_kwargs = {"messages": messages, "model": model or self.model, **kwargs}

        if tools:
            api_kwargs["tools"] = tools
        return self.client.chat.completions.create(**api_kwargs)

    def extract_content(self, response: Any) -> Optional[str]:
        if not response.choices:
            return None
        return response.choices[0].message.content

    def parse_tool_calls(self, response: Any) -> Optional[List[ToolCall]]:
        if not response.choices:
            return None
        message = response.choices[0].message
        if not hasattr(message, "tool_calls") or not message.tool_calls:
            return None
        tool_calls = []
        for tool_call in message.tool_calls:
            if tool_call.type == "function" and tool_call.function:
                tool_calls.append(
                    {
                        "id": tool_call.id,
                        "function_name": tool_call.function.name,
                        "function_args": tool_call.function.arguments,
                    }
                )
        return tool_calls if tool_calls else None

    def create_assistant_message(self, response: Any) -> ChatMessage:
        """Create a ChatMessage mirroring the OpenAI structure."""
        if not response.choices:
            return ChatMessage(role=ASSISTANT_ROLE, content="[No response generated]")
        message = response.choices[0].message
        raw_tool_calls = None
        if hasattr(message, "tool_calls") and message.tool_calls:
            raw_tool_calls = [tc.model_dump() for tc in message.tool_calls]
        return ChatMessage(
            role=ASSISTANT_ROLE,
            content=message.content,
            tool_calls=raw_tool_calls,
        )

    def create_tool_result_messages(
        self, results: List[ToolResult]
    ) -> List[ChatMessage]:
        """Creates an OpenAI-compatible tool result message (role=tool)."""
        messages = []
        for result in results:
            messages.append(
                ChatMessage(
                    role=TOOL_ROLE,
                    content=result.content,
                    tool_call_id=result.tool_call_id,
                )
            )
        return messages


class OpenAI(_OpenAICompatible):
    def __init__(self, default_model: str = "gpt-4o"):
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.client = client
        self.model = default_model


class OpenRouter(_OpenAICompatible):
    def __init__(self, default_model: str = "openai/gpt-4o"):
        from openai import OpenAI

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
        self.model = default_model

    def generate_response(self, *args, **kwargs):
        headers = kwargs.pop("extra_headers", {})
        headers.update(
            {"HTTP-Referer": "https://chatnificent.com", "X-Title": "Chatnificent"}
        )
        kwargs["extra_headers"] = headers
        return super().generate_response(*args, **kwargs)


class DeepSeek(_OpenAICompatible):
    def __init__(self, default_model: str = "deepseek-chat"):
        from openai import OpenAI

        self.client = OpenAI(
            base_url="https://api.deepseek.com",
            api_key=os.environ["DEEPSEEK_API_KEY"],
        )
        self.model = default_model


class Gemini(LLM):
    """
    LLM provider for Google Gemini models.
    Requires significant adaptation due to different message structures.
    """

    def __init__(self, default_model: str = "gemini-1.5-flash-latest"):
        import google.generativeai as genai
        from google.generativeai.types import content_types

        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)

        self.genai = genai
        self.content_types = content_types
        self.model_name = default_model

    def generate_response(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> Any:
        system_instruction, gemini_messages = self._process_messages(messages)

        model_instance = self.genai.GenerativeModel(
            model_name=model or self.model_name,
            tools=tools if tools else None,
            system_instruction=system_instruction,
        )

        valid_config_args = [
            "temperature",
            "top_p",
            "top_k",
            "max_output_tokens",
            "stop_sequences",
        ]
        gen_config_kwargs = {k: v for k, v in kwargs.items() if k in valid_config_args}

        generation_config = (
            self.genai.GenerationConfig(**gen_config_kwargs)
            if gen_config_kwargs
            else None
        )

        try:
            return model_instance.generate_content(
                contents=gemini_messages,
                generation_config=generation_config,
            )
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            raise

    def _process_messages(self, messages: List[Dict[str, Any]]):
        """
        CRITICAL ADAPTER LOGIC: Translates history structure (content -> parts),
        roles (assistant -> model), and extracts system instructions for Gemini.
        """
        system_instruction = None
        gemini_messages = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            # 1. Extract System Instructions
            if role == "system":
                if isinstance(content, str):
                    if system_instruction is None:
                        system_instruction = content
                    else:
                        system_instruction += "\n" + content
                else:
                    logger.warning(
                        "Skipping non-string system message content for Gemini."
                    )
                continue

            # 2. Translate Roles
            if role == ASSISTANT_ROLE:
                gemini_role = MODEL_ROLE  # Gemini uses 'model'
            elif role == USER_ROLE:
                gemini_role = USER_ROLE
            elif role == MODEL_ROLE:
                # Already in Gemini format (from history)
                gemini_role = MODEL_ROLE
            else:
                # Skip other roles (like OpenAI 'tool' role)
                continue

            # 3. Restructure Content to Parts (THE FIX)
            # Gemini strictly requires the 'parts' key instead of 'content'.
            parts = []
            if isinstance(content, str):
                # Simple string content -> wrap in a text part
                parts = [{"text": content}]
            elif isinstance(content, list):
                # Structured content (e.g., from high-fidelity persistence)
                # Assume it's already a list of valid parts (text, function_call, function_response)
                parts = content
            elif content is None:
                # Handle cases where content might be None (e.g. OpenAI tool call message)
                pass
            else:
                logger.warning(
                    f"Unexpected content type in history for Gemini: {type(content)}"
                )
                parts = [{"text": str(content)}]

            # Gemini requires non-empty parts list if the message is included
            if parts:
                # Construct the message in the required format
                gemini_messages.append({"role": gemini_role, "parts": parts})
            else:
                logger.warning(
                    f"Skipping message with empty content/parts for Gemini: Role={role}"
                )

        return system_instruction, gemini_messages

    # (extract_content, parse_tool_calls, create_assistant_message,
    # and create_tool_result_messages remain the same as the previous implementation, as they were correct.)

    def extract_content(self, response: Any) -> Optional[str]:
        try:
            # response.text concatenates all text parts.
            return response.text
        except ValueError:
            # Occurs if no text (e.g., only tool calls or safety block)
            if (
                hasattr(response, "prompt_feedback")
                and response.prompt_feedback
                and response.prompt_feedback.block_reason
            ):
                return f"[Response blocked by safety filters. Reason: {response.prompt_feedback.block_reason}]"
            return None

    def parse_tool_calls(self, response: Any) -> Optional[List[ToolCall]]:
        if not response.candidates or not response.candidates[0].content.parts:
            return None

        tool_calls = []
        for part in response.candidates[0].content.parts:
            # Check if the part is a function_call
            if hasattr(part, "function_call") and part.function_call:
                # CRITICAL: Gemini does not provide a unique ID. We generate one securely.
                tool_id = f"gemini-tool-call-{secrets.token_hex(8)}"

                # Arguments are in a dictionary-like structure.
                args_dict = dict(part.function_call.args)

                tool_calls.append(
                    ToolCall(
                        id=tool_id,
                        function_name=part.function_call.name,
                        function_args=json.dumps(args_dict),
                    )
                )
        return tool_calls if tool_calls else None

    def create_assistant_message(self, response: Any) -> ChatMessage:
        """Creates a ChatMessage mirroring the Gemini structure (role=model, content=parts)."""
        if not response.candidates:
            return ChatMessage(role=MODEL_ROLE, content="[No response generated]")

        # Convert SDK objects (Content/Part) into dicts for storage.
        try:
            # Use the SDK's utility function for robust conversion
            parts = [
                self.content_types.to_dict(p)
                for p in response.candidates[0].content.parts
            ]
        except Exception as e:
            logger.warning(
                f"Could not serialize Gemini response content using to_dict: {e}"
            )
            # Fallback to text extraction if serialization fails
            text_content = self.extract_content(response)
            parts = [{"text": text_content or "[Error extracting content]"}]

        return ChatMessage(role=MODEL_ROLE, content=parts)

    def create_tool_result_messages(
        self, results: List[ToolResult]
    ) -> List[ChatMessage]:
        """Creates a Gemini-compatible tool result message (role=user, function_response)."""
        # Gemini best practice: Batch tool results into a single role='user' message.

        parts = []
        for result in results:
            # Gemini requires the output to be a dictionary for function_response.

            if result.is_error:
                # Gemini doesn't have a specific error flag, embed it in the result structure.
                output_content = {"error": result.content}
            else:
                try:
                    output_content = json.loads(result.content)
                    # If it loads but isn't a dict (e.g., a list or primitive), wrap it.
                    if not isinstance(output_content, dict):
                        output_content = {"result": output_content}
                except json.JSONDecodeError:
                    # If it's not valid JSON (e.g., a raw string), wrap it.
                    output_content = {"result": result.content}

            # Structure required by the Gemini API
            part = {
                "function_response": {
                    # CRITICAL: Gemini correlates results using the function name, not the ID.
                    "name": result.function_name,
                    "response": output_content,
                }
            }
            parts.append(part)

        if not parts:
            return []

        return [ChatMessage(role=USER_ROLE, content=parts)]


class Anthropic(LLM):
    def __init__(self, default_model: str = "claude-3-5-sonnet-20241022"):
        from anthropic import Anthropic

        self.client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = default_model

    def generate_response(self, messages, model=None, tools=None, **kwargs) -> Any:
        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = 4096
        system_prompt, filtered_messages = self._process_messages(messages)
        api_kwargs = {
            "model": model or self.model,
            "messages": filtered_messages,
            **kwargs,
        }
        if system_prompt:
            api_kwargs["system"] = system_prompt
        if tools:
            api_kwargs["tools"] = tools
        return self.client.messages.create(**api_kwargs)

    def _process_messages(self, messages: List[Dict[str, Any]]):
        system_prompt = None
        filtered_messages = []
        for msg in messages:
            role = msg.get("role")
            if role == "system":
                content = msg.get("content", "")
                system_prompt = (
                    content if system_prompt is None else system_prompt + "\n" + content
                )
            elif role in (USER_ROLE, ASSISTANT_ROLE):
                filtered_messages.append(msg)

        return system_prompt, filtered_messages

    def extract_content(self, response: Any) -> Optional[str]:
        text_content = [
            block.text for block in response.content if block.type == "text"
        ]
        return "\n".join(text_content) if text_content else ""

    def parse_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        """Parse tool calls from Anthropic response."""
        tool_calls = []

        for content_block in response.content:
            if hasattr(content_block, "type") and content_block.type == "tool_use":
                tool_calls.append(
                    {
                        "id": content_block.id,
                        "function_name": content_block.name,
                        "arguments": content_block.input,
                    }
                )

        return tool_calls

    def create_assistant_message(self, response: Any) -> ChatMessage:
        """Creates a ChatMessage mirroring the Anthropic structure (content blocks)."""
        content_blocks = [block.model_dump() for block in response.content]
        return ChatMessage(role=ASSISTANT_ROLE, content=content_blocks)

    def create_tool_result_messages(
        self, results: List[ToolResult]
    ) -> List[ChatMessage]:
        """Creates an Anthropic-compatible tool result message (role=user)."""
        content_blocks = []
        for result in results:
            block = {
                "type": "tool_result",
                "tool_use_id": result.tool_call_id,
                "content": result.content,
            }
            if result.is_error:
                block["is_error"] = True
            content_blocks.append(block)

        if not content_blocks:
            return []

        return [ChatMessage(role=USER_ROLE, content=content_blocks)]


class Ollama(LLM):
    def __init__(self, default_model: str = "llama3.1"):
        from ollama import Client

        self.client = Client()
        self.model = default_model

    def generate_response(self, messages, model=None, tools=None, **kwargs) -> Any:
        api_kwargs = {
            "model": model or self.model,
            "messages": messages,
            **kwargs,
        }
        if tools:
            api_kwargs["tools"] = tools
        return self.client.chat(**api_kwargs)

    def extract_content(self, response: Any) -> Optional[str]:
        return response.get("message", {}).get("content")

    def parse_tool_calls(self, response: Any) -> Optional[List[ToolCall]]:
        message = response.get("message", {})
        raw_tool_calls = message.get("tool_calls")
        if not raw_tool_calls:
            return None
        tool_calls = []
        for tool_call in raw_tool_calls:
            function_data = tool_call.get("function")
            if function_data:
                import secrets

                tool_id = f"ollama-tool-call-{secrets.token_hex(8)}"
                args = function_data.get("arguments", {})
                tool_calls.append(
                    ToolCall(
                        id=tool_id,
                        function_name=function_data.get("name", ""),
                        function_args=args,
                    )
                )
        return tool_calls if tool_calls else None

    def create_assistant_message(self, response: Any) -> ChatMessage:
        message = response.get("message", {})
        return ChatMessage(
            role=ASSISTANT_ROLE,
            content=message.get("content", ""),
            tool_calls=message.get("tool_calls"),
        )

    def create_tool_result_messages(
        self, results: List[ToolResult]
    ) -> List[ChatMessage]:
        messages = []
        for result in results:
            messages.append(
                ChatMessage(
                    role=TOOL_ROLE,
                    content=result.content,
                    tool_call_id=result.tool_call_id,
                )
            )
        return messages


class Echo(LLM):
    """Mock LLM for testing purposes and fallback."""

    def __init__(self, default_model: str = "echo-v1"):
        self.model = default_model

    def generate_response(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> Any:
        import time

        time.sleep(0.8)

        user_prompt = ""
        for msg in reversed(messages):
            if msg.get("role") == USER_ROLE:
                content = msg.get("content")
                if isinstance(content, str):
                    user_prompt = content
                elif isinstance(content, list):
                    # Handle structured content if necessary (e.g., previous tool results)
                    user_prompt = "[Structured Input]"
                else:
                    user_prompt = str(content) if content else ""
                break

        if not user_prompt:
            user_prompt = "No user message found."

        content = f"**Echo LLM - static response**\n\n_Your prompt:_\n\n{user_prompt}"

        if tools:
            content += "\n\n_Note: Tools were provided but ignored by Echo LLM._"

        return {
            "content": content,
            "model": model or self.model,
            "type": "echo_response",
        }

    def extract_content(self, response: Any) -> Optional[str]:
        if isinstance(response, dict) and response.get("type") == "echo_response":
            return response.get("content")
        return str(response)

    def parse_tool_calls(self, response: Any) -> Optional[List[ToolCall]]:
        return None  # Echo does not generate tool calls

    def create_assistant_message(self, response: Any) -> ChatMessage:
        return ChatMessage(role=ASSISTANT_ROLE, content=self.extract_content(response))

    def create_tool_result_messages(
        self, results: List[ToolResult]
    ) -> List[ChatMessage]:
        return []
