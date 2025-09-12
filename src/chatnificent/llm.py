"""Concrete implementations for LLM providers."""

import json
import logging
import os
import secrets
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .models import (
    ASSISTANT_ROLE,
    MODEL_ROLE,
    TOOL_ROLE,
    USER_ROLE,
    ChatMessage,
    Conversation,
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
        self, results: List[ToolResult], conversation: Conversation
    ) -> List[ChatMessage]:
        """Converts ToolResult objects into ChatMessage instances for persistence."""
        if results:
            if (
                type(self).create_tool_result_messages
                == LLM.create_tool_result_messages
            ):
                raise NotImplementedError(
                    f"{self.__class__.__name__} must implement this method if tools are supported."
                )
        return []

    def is_tool_message(self, message: "ChatMessage") -> bool:
        return False


class _OpenAICompatible(LLM):
    def generate_response(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> Any:
        cleaned_messages = []
        for msg in messages:
            cleaned_msg = dict(msg)
            if cleaned_msg.get("content") is None:
                if cleaned_msg.get("role") == ASSISTANT_ROLE and cleaned_msg.get(
                    "tool_calls"
                ):
                    pass
                elif cleaned_msg.get("role") == TOOL_ROLE:
                    cleaned_msg["content"] = ""
                elif cleaned_msg.get("role") in [
                    USER_ROLE,
                    ASSISTANT_ROLE,
                ] and not cleaned_msg.get("tool_calls"):
                    cleaned_msg["content"] = ""
            cleaned_messages.append(cleaned_msg)
        api_kwargs = {
            "messages": cleaned_messages,
            "model": model or self.model,
            **kwargs,
        }
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
                    ToolCall(
                        id=tool_call.id,
                        function_name=tool_call.function.name,
                        function_args=tool_call.function.arguments,
                    )
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
        self, results: List[ToolResult], conversation: Conversation
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

    def is_tool_message(self, message: "ChatMessage") -> bool:
        """OpenAI tool messages have role='tool'."""
        return message.role == TOOL_ROLE


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


class Anthropic(LLM):
    """Concrete implementation for Anthropic Claude models."""

    def __init__(self, default_model: str = "claude-3-opus-20240229"):
        from anthropic import Anthropic

        self.client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = default_model

    def _translate_tool_schema(
        self, tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Helper to translate OpenAI schema to Anthropic's format."""
        translated_tools = []
        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                func = tool["function"]
                translated_tools.append(
                    {
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "input_schema": func.get(
                            "parameters", {"type": "object", "properties": {}}
                        ),
                    }
                )
        return translated_tools

    def generate_response(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> Any:
        # Anthropic requires a system prompt to be at the top level
        system_prompt = None
        if messages and messages[0].get("role") == "system":
            system_prompt = messages.pop(0)["content"]

        api_kwargs = {
            "messages": messages,
            "model": model or self.model,
            "max_tokens": 4096,  # Recommended default for Anthropic
            **kwargs,
        }
        if system_prompt:
            api_kwargs["system"] = system_prompt
        if tools:
            api_kwargs["tools"] = self._translate_tool_schema(tools)

        return self.client.messages.create(**api_kwargs)

    def extract_content(self, response: Any) -> Optional[str]:
        # Find the first text block in the response content
        if not response.content:
            return None
        for block in response.content:
            if block.type == "text":
                return block.text
        return None

    def parse_tool_calls(self, response: Any) -> Optional[List[ToolCall]]:
        if response.stop_reason != "tool_use":
            return None
        tool_calls = []
        for block in response.content:
            if block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        function_name=block.name,
                        # Anthropic provides args as a dict, convert to JSON string for consistency
                        function_args=json.dumps(block.input),
                    )
                )
        return tool_calls if tool_calls else None

    def create_assistant_message(self, response: Any) -> ChatMessage:
        """For Anthropic, we persist the raw content blocks for later use."""
        if response.stop_reason == "tool_use":
            return ChatMessage(
                role=ASSISTANT_ROLE, content=response.model_dump()["content"]
            )
        return ChatMessage(
            role=ASSISTANT_ROLE,
            # Persist the raw model_dump for high fidelity
            content=self.extract_content(response),
        )

    def create_tool_result_messages(
        self, results: List[ToolResult], conversation: "Conversation"
    ) -> List[ChatMessage]:
        """
        Anthropic requires a specific message structure for tool results:
        1. The original assistant message containing the tool_use requests.
        2. A new user message containing the tool_result blocks.
        The engine will add this user message, and our adapter must prepare it.
        """
        tool_result_content = []
        for result in results:
            tool_result_content.append(
                {
                    "type": "tool_result",
                    "tool_use_id": result.tool_call_id,
                    "content": result.content,
                    "is_error": result.is_error,
                }
            )

        # Anthropic expects a single message with all tool results
        return [ChatMessage(role=USER_ROLE, content=tool_result_content)]

    def is_tool_message(self, message: "ChatMessage") -> bool:
        message_dict = message.model_dump()
        content_data = message_dict.get("content")
        role = message_dict.get("role")

        # The primary indicator of a special tool message is list-based content.
        if not isinstance(content_data, list):
            return False

        # Apply filtering logic based on the reliable data from the dumped model.
        if role == USER_ROLE:
            return all(item.get("type") == "tool_result" for item in content_data)

        if role == ASSISTANT_ROLE:
            return any(item.get("type") == "tool_use" for item in content_data)

        return False


class Gemini(LLM):
    """Concrete implementation for Google Gemini models."""

    def __init__(self, default_model: str = "gemini-1.5-flash"):
        from google import generativeai as genai
        from google.generativeai import types as genai_types

        self.genai = genai
        self.genai_types = genai_types
        api_key = os.environ.get("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(default_model)
        self.model = default_model
        self.system_instruction = None

    def _translate_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Translate roles and structure for Gemini."""
        translated = []
        for msg in messages:
            role = msg.get("role")
            if role == "system":
                self.system_instruction = msg.get("content", "")
                continue

            if role == ASSISTANT_ROLE:
                role = MODEL_ROLE  # Gemini uses 'model' for assistant
            elif role == TOOL_ROLE:
                # Gemini expects tool results in a specific format
                translated.append(
                    self.genai_types.Part.from_function_response(
                        name=msg.get("name"), response={"content": msg.get("content")}
                    )
                )
                continue

            content = msg.get("content", "")
            if isinstance(content, str):
                translated.append({"role": role, "parts": [content]})

        return translated

    def generate_response(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> Any:
        gemini_tools = None
        if tools:
            # Extract the 'function' part of the OpenAI schema
            function_declarations = [
                t["function"] for t in tools if t.get("type") == "function"
            ]
            if function_declarations:
                gemini_tools = [
                    self.genai_types.Tool(function_declarations=function_declarations)
                ]

        # Select the correct model client with system instruction if needed
        client = self.client
        if model and model != self.model:
            client = self.genai.GenerativeModel(
                model, system_instruction=self.system_instruction
            )
        elif self.system_instruction:
            client = self.genai.GenerativeModel(
                self.model, system_instruction=self.system_instruction
            )

        response = client.generate_content(
            self._translate_messages(messages),
            tools=gemini_tools,
            **kwargs,
        )
        return response.to_dict()

    def extract_content(self, response: Any) -> Optional[str]:
        try:
            candidates = response.get("candidates", [])
            if candidates and "content" in candidates[0]:
                parts = candidates[0]["content"].get("parts", [])
                for part in parts:
                    if "text" in part:
                        return part["text"]
            return None
        except Exception:
            logger.warning(
                "Could not extract text from Gemini response.", exc_info=True
            )
            return None

    def parse_tool_calls(self, response: Any) -> Optional[List[ToolCall]]:
        # response is now a dict from to_dict()
        candidates = response.get("candidates", [])
        if not candidates:
            return None
        tool_calls = []
        if "content" in candidates[0]:
            parts = candidates[0]["content"].get("parts", [])
            for part in parts:
                if "function_call" in part:
                    fc = part["function_call"]
                    # Generate tool_call_id for internal tracking
                    tool_call_id = f"call_{secrets.token_hex(8)}"
                    tool_calls.append(
                        ToolCall(
                            id=tool_call_id,
                            function_name=fc.get("name", ""),
                            function_args=json.dumps(fc.get("args", {})),
                        )
                    )
        return tool_calls if tool_calls else None

    def create_assistant_message(self, response: Any) -> ChatMessage:
        """Create a ChatMessage mirroring Gemini's structure, using MODEL_ROLE."""
        # response is now a dict from to_dict()
        candidates = response.get("candidates", [])
        if not candidates:
            return ChatMessage(role=MODEL_ROLE, content="[No response generated]")

        # Extract raw parts for high fidelity storage
        if "content" in candidates[0]:
            raw_parts = candidates[0]["content"].get("parts", [])
            return ChatMessage(
                role=MODEL_ROLE,
                content=raw_parts,
            )
        return ChatMessage(role=MODEL_ROLE, content="[No response generated]")

    def create_tool_result_messages(
        self, results: List[ToolResult], conversation: "Conversation"
    ) -> List[ChatMessage]:
        """Creates Gemini-compatible tool result messages."""
        messages = []
        for result in results:
            messages.append(
                ChatMessage(
                    role=TOOL_ROLE,
                    name=result.function_name,
                    content=result.content,
                    # We don't need tool_call_id here as Gemini tracks by function name in context
                )
            )
        return messages

    def is_tool_message(self, message: "ChatMessage") -> bool:
        """Gemini tool messages have role='tool'."""
        return message.role == TOOL_ROLE


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

    def parse_tool_calls(self, response: Any) -> Optional[List["ToolCall"]]:
        return None

    def create_assistant_message(self, response: Any) -> "ChatMessage":
        return ChatMessage(role=ASSISTANT_ROLE, content=self.extract_content(response))

    def create_tool_result_messages(
        self, results: List["ToolResult"]
    ) -> List["ChatMessage"]:
        return []
