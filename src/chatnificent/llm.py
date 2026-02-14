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
        """Communicate with the LLM SDK and return the native response object.

        Parameters
        ----------
        messages : List[Dict[str, Any]]
            A list of message dictionaries conforming to the provider's format.
        model : Optional[str], optional
            The specific model to use for this request, overriding the instance's
            default model. By default None.
        tools : Optional[List[Any]], optional
            A list of tools the model can call. By default None.
        **kwargs : Any
            Additional keyword arguments to pass to the provider's API.

        Returns
        -------
        Any
            The native response object from the LLM provider's SDK.

        """
        pass

    @abstractmethod
    def extract_content(self, response: Any) -> Optional[str]:
        """Extract human-readable text from the native response.

        Parameters
        ----------
        response : Any
            The native response object from the provider's SDK.

        Returns
        -------
        Optional[str]
            The extracted text content from the response.

        """
        pass

    def parse_tool_calls(self, response: Any) -> Optional[List[ToolCall]]:
        """Translate the native response into the standardized ToolCall format.

        Parameters
        ----------
        response : Any
            The native response object from the provider's SDK.

        Returns
        -------
        Optional[List[ToolCall]]
            A list of standardized ToolCall objects, or None if no tools were called.

        """
        return None

    def create_assistant_message(self, response: Any) -> ChatMessage:
        """Convert the native response into a ChatMessage for persistence.

        Parameters
        ----------
        response : Any
            The native response object from the provider's SDK.

        Returns
        -------
        ChatMessage
            A standardized ChatMessage object representing the assistant's response.

        """
        content = self.extract_content(response)
        return ChatMessage(role=ASSISTANT_ROLE, content=content)

    def create_tool_result_messages(
        self, results: List[ToolResult], conversation: Conversation
    ) -> List[ChatMessage]:
        """Convert ToolResult objects into ChatMessage instances for persistence.

        Parameters
        ----------
        results : List[ToolResult]
            A list of ToolResult objects from executed tools.
        conversation : Conversation
            The conversation context, which may be needed by some providers.

        Returns
        -------
        List[ChatMessage]
            A list of ChatMessage objects to be sent back to the LLM.

        Raises
        ------
        NotImplementedError
            If the provider supports tools but this method is not implemented.

        """
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
        """Check if a message is a special tool-related message for a provider.

        Parameters
        ----------
        message : ChatMessage
            The ChatMessage object to check.

        Returns
        -------
        bool
            True if the message is a tool message, False otherwise.

        """
        return False


class _OpenAICompatible(LLM):
    """Mixin class for providers with OpenAI-compatible APIs."""

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
            **self.default_params,
            **kwargs,
        }
        api_kwargs["model"] = model or self.model
        api_kwargs["messages"] = cleaned_messages

        if tools:
            api_kwargs["tools"] = tools
        return self.client.chat.completions.create(**api_kwargs)

    def extract_content(self, response: Any) -> Optional[str]:
        if not response.choices:
            return None
        choice = response.choices[0]
        if choice.message.content:
            return choice.message.content
        finish_reason = getattr(choice, "finish_reason", "UNKNOWN")
        return f"Empty response from {self.model} — finish_reason: {finish_reason}"

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
        return message.role == TOOL_ROLE


class OpenAI(_OpenAICompatible):
    """Concrete implementation for OpenAI models."""

    def __init__(self, model: str = "gpt-4.1", api_key: Optional[str] = None, **kwargs):
        """
        Initializes the OpenAI client.

        Parameters
        ----------
        model : str, optional
            The default model to use for chat completions, by default "gpt-4.1".
        api_key : Optional[str], optional
            Your OpenAI API key. If not provided, the `OPENAI_API_KEY`
            environment variable will be used. By default None.
        **kwargs : Any
            Default parameters for the chat completions API (e.g., `temperature`).

        Raises
        ------
        ValueError
            If the API key is not provided via argument or environment variable.
        """
        from openai import OpenAI

        resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "OpenAI API key not found. Please pass the `api_key` argument "
                "or set the 'OPENAI_API_KEY' environment variable."
            )

        self.client = OpenAI(api_key=resolved_api_key)
        self.model = model
        self.default_params = kwargs


class OpenRouter(_OpenAICompatible):
    """Concrete implementation for OpenRouter models."""

    def __init__(
        self, model: str = "openai/gpt-4.1", api_key: Optional[str] = None, **kwargs
    ):
        """
        Initializes the OpenRouter client.

        Parameters
        ----------
        model : str, optional
            The default model to use (e.g., 'openai/gpt-4.1'),
            by default "openai/gpt-4.1".
        api_key : Optional[str], optional
            Your OpenRouter API key. If not provided, the `OPENROUTER_API_KEY`
            environment variable will be used. By default None.
        **kwargs : Any
            Default parameters for the chat completions API.

        Raises
        ------
        ValueError
            If the API key is not provided via argument or environment variable.
        """
        from openai import OpenAI

        resolved_api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "OpenRouter API key not found. Please pass the `api_key` argument "
                "or set the 'OPENROUTER_API_KEY' environment variable."
            )

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=resolved_api_key,
        )
        self.model = model
        self.default_params = kwargs

    def generate_response(self, *args, **kwargs):
        headers = kwargs.pop("extra_headers", {})
        headers.update(
            {"HTTP-Referer": "https://chatnificent.com", "X-Title": "Chatnificent"}
        )
        kwargs["extra_headers"] = headers
        return super().generate_response(*args, **kwargs)


class DeepSeek(_OpenAICompatible):
    """Concrete implementation for DeepSeek models."""

    def __init__(
        self, model: str = "deepseek-chat", api_key: Optional[str] = None, **kwargs
    ):
        """
        Initializes the DeepSeek client.

        Parameters
        ----------
        model : str, optional
            The default model to use, by default "deepseek-chat".
        api_key : Optional[str], optional
            Your DeepSeek API key. If not provided, the `DEEPSEEK_API_KEY`
            environment variable will be used. By default None.
        **kwargs : Any
            Default parameters for the chat completions API.

        Raises
        ------
        ValueError
            If the API key is not provided via argument or environment variable.
        """
        from openai import OpenAI

        resolved_api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "DeepSeek API key not found. Please pass the `api_key` argument "
                "or set the 'DEEPSEEK_API_KEY' environment variable."
            )

        self.client = OpenAI(
            base_url="https://api.deepseek.com",
            api_key=resolved_api_key,
        )
        self.model = model
        self.default_params = kwargs


class Anthropic(LLM):
    """Concrete implementation for Anthropic Claude models."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-5",
        api_key: Optional[str] = None,
        **kwargs,
    ):
        """
        Initializes the Anthropic client.

        Parameters
        ----------
        model : str, optional
            The default model to use, by default "claude-3-opus-20240229".
        api_key : Optional[str], optional
            Your Anthropic API key. If not provided, the `ANTHROPIC_API_KEY`
            environment variable will be used. By default None.
        **kwargs : Any
            Default parameters for the messages API (e.g., `temperature`).

        Raises
        ------
        ValueError
            If the API key is not provided via argument or environment variable.
        """
        from anthropic import Anthropic

        resolved_api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "Anthropic API key not found. Please pass the `api_key` argument "
                "or set the 'ANTHROPIC_API_KEY' environment variable."
            )

        self.client = Anthropic(api_key=resolved_api_key)
        self.model = model
        self.default_params = {"max_tokens": 4096}
        self.default_params.update(kwargs)

    def _translate_tool_schema(
        self, tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Translate OpenAI tool schema to Anthropic's format."""
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
        system_prompt = None
        if messages and messages[0].get("role") == "system":
            system_prompt = messages.pop(0)["content"]

        api_kwargs = {
            **self.default_params,
            **kwargs,
        }
        api_kwargs["model"] = model or self.model
        api_kwargs["messages"] = messages

        if system_prompt:
            api_kwargs["system"] = system_prompt
        if tools:
            api_kwargs["tools"] = self._translate_tool_schema(tools)

        return self.client.messages.create(**api_kwargs)

    def extract_content(self, response: Any) -> Optional[str]:
        if not response.content:
            stop_reason = getattr(response, "stop_reason", "UNKNOWN")
            return f"Empty response from {self.model} — stop_reason: {stop_reason}"
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
                        function_args=json.dumps(block.input),
                    )
                )
        return tool_calls if tool_calls else None

    def create_assistant_message(self, response: Any) -> ChatMessage:
        if response.stop_reason == "tool_use":
            return ChatMessage(
                role=ASSISTANT_ROLE, content=response.model_dump()["content"]
            )
        return ChatMessage(
            role=ASSISTANT_ROLE,
            content=self.extract_content(response),
        )

    def create_tool_result_messages(
        self, results: List[ToolResult], conversation: "Conversation"
    ) -> List[ChatMessage]:
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
        return [ChatMessage(role=USER_ROLE, content=tool_result_content)]

    def is_tool_message(self, message: "ChatMessage") -> bool:
        message_dict = message.model_dump()
        content_data = message_dict.get("content")
        role = message_dict.get("role")

        if not isinstance(content_data, list):
            return False
        if role == USER_ROLE:
            return all(item.get("type") == "tool_result" for item in content_data)
        if role == ASSISTANT_ROLE:
            return any(item.get("type") == "tool_use" for item in content_data)
        return False


class Gemini(LLM):
    """Concrete implementation for Google Gemini models.

    Examples
    --------
    >>> import chatnificent as chat
    >>> from chatnificent import Chatnificent
    >>> llm = chat.llm.Gemini(model="gemini-3-flash", temperature=0.7)
    >>> llm = chat.llm.Gemini(
    ...     vertexai=True, project="my-project", location="us-central1"
    ... )
    >>> app = Chatnificent(llm=llm)
    """

    _CLIENT_KEYS = frozenset(
        {"api_key", "vertexai", "project", "location", "http_options", "client_options"}
    )

    def __init__(self, model: str = "gemini-2.5-flash", **kwargs):
        """
        Initializes the Gemini client.

        Parameters
        ----------
        model : str, optional
            The default model to use, by default "gemini-2.5-flash".
        **kwargs : Any
            Client keys (``api_key``, ``vertexai``, ``project``, ``location``,
            ``http_options``, ``client_options``) are forwarded to
            ``genai.Client``; everything else becomes default generation
            parameters.

        Raises
        ------
        ValueError
            If no API key is discoverable and Vertex AI mode is not enabled.
        """
        from google import genai
        from google.genai import types as genai_types

        self._genai_types = genai_types

        client_kwargs = {k: v for k, v in kwargs.items() if k in self._CLIENT_KEYS}
        generation_kwargs = {
            k: v for k, v in kwargs.items() if k not in self._CLIENT_KEYS
        }

        if not client_kwargs.get("vertexai"):
            if "api_key" not in client_kwargs:
                resolved_key = os.environ.get("GEMINI_API_KEY") or os.environ.get(
                    "GOOGLE_API_KEY"
                )
                if not resolved_key:
                    raise ValueError(
                        "Gemini API key not found. Please pass the `api_key` argument "
                        "or set the 'GEMINI_API_KEY' or 'GOOGLE_API_KEY' environment variable."
                    )
                client_kwargs["api_key"] = resolved_key

        try:
            from importlib.metadata import version as _get_pkg_version

            pkg_version = _get_pkg_version("chatnificent")
        except Exception:
            pkg_version = "unknown"

        http_options = client_kwargs.get("http_options", {})
        if isinstance(http_options, dict):
            headers = http_options.setdefault("headers", {})
            headers.setdefault("x-goog-api-client", f"chatnificent/{pkg_version}")
            client_kwargs["http_options"] = http_options

        self.client = genai.Client(**client_kwargs)
        self.model = model
        self.default_params = generation_kwargs

    def _translate_request(self, messages: List[Dict[str, Any]]) -> tuple:
        """Translate OpenAI-format messages to google-genai Content objects.

        Returns
        -------
        tuple[list, Optional[str]]
            A ``(contents, system_instruction)`` pair.
        """
        types = self._genai_types
        contents: list = []
        system_instruction: Optional[str] = None
        pending_tool_parts: list = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if role == "system":
                system_instruction = content
                continue

            if role != TOOL_ROLE and pending_tool_parts:
                contents.append(types.Content(role="user", parts=pending_tool_parts))
                pending_tool_parts = []

            if role == TOOL_ROLE:
                pending_tool_parts.append(
                    types.Part.from_function_response(
                        name=msg.get("name", "unknown"),
                        response={"content": content or ""},
                    )
                )
                continue

            google_role = "model" if role in (ASSISTANT_ROLE, MODEL_ROLE) else "user"

            parts = self._build_parts(content)
            if parts:
                contents.append(types.Content(role=google_role, parts=parts))

        if pending_tool_parts:
            contents.append(types.Content(role="user", parts=pending_tool_parts))

        return contents, system_instruction

    def _build_parts(self, content: Any) -> list:
        """Convert a message's content field into a list of Part objects."""
        types = self._genai_types
        parts: list = []

        if isinstance(content, str):
            parts.append(types.Part.from_text(text=content))
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, str):
                    parts.append(types.Part.from_text(text=item))
                elif isinstance(item, dict):
                    if item.get("function_call"):
                        fc = item["function_call"]
                        parts.append(
                            types.Part.from_function_call(
                                name=fc.get("name", ""),
                                args=fc.get("args", {}),
                            )
                        )
                    elif item.get("function_response"):
                        fr = item["function_response"]
                        parts.append(
                            types.Part.from_function_response(
                                name=fr.get("name", ""),
                                response=fr.get("response", {}),
                            )
                        )
                    elif item.get("text") is not None:
                        parts.append(types.Part.from_text(text=item["text"]))

        return parts

    def _translate_tool_schema(self, tools: List[Dict[str, Any]]) -> List[Any]:
        """Translate OpenAI tool schema to Gemini FunctionDeclaration format."""
        types = self._genai_types
        declarations = []
        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                func = tool["function"]
                decl_kwargs: Dict[str, Any] = {
                    "name": func["name"],
                    "description": func.get("description", ""),
                }
                if func.get("parameters"):
                    decl_kwargs["parameters_json_schema"] = func["parameters"]
                declarations.append(types.FunctionDeclaration(**decl_kwargs))
        if declarations:
            return [types.Tool(function_declarations=declarations)]
        return []

    def generate_response(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> Any:
        types = self._genai_types
        config_dict: Dict[str, Any] = {**self.default_params, **kwargs}

        contents, system_instruction = self._translate_request(messages)

        if system_instruction is not None:
            config_dict["system_instruction"] = system_instruction

        if tools:
            translated = self._translate_tool_schema(tools)
            if translated:
                config_dict["tools"] = translated

        config = types.GenerateContentConfig(**config_dict) if config_dict else None

        response = self.client.models.generate_content(
            model=model or self.model,
            contents=contents,
            config=config,
        )
        return response.model_dump(mode="json")

    def extract_content(self, response: Any) -> Optional[str]:
        try:
            candidates = response.get("candidates") or []
            if not candidates:
                return None
            candidate = candidates[0]
            parts = (candidate.get("content") or {}).get("parts") or []
            text_pieces = [p["text"] for p in parts if p.get("text") is not None]
            if text_pieces:
                return "".join(text_pieces)
            finish_reason = candidate.get("finish_reason", "UNKNOWN")
            return f"Empty response from Gemini — finish_reason: {finish_reason}"
        except Exception:
            logger.warning(
                "Could not extract text from Gemini response.", exc_info=True
            )
            return None

    def parse_tool_calls(self, response: Any) -> Optional[List[ToolCall]]:
        candidates = response.get("candidates") or []
        if not candidates:
            return None

        parts = (candidates[0].get("content") or {}).get("parts") or []
        function_calls = [p["function_call"] for p in parts if p.get("function_call")]
        if not function_calls:
            return None

        tool_calls = []
        for fc in function_calls:
            tool_calls.append(
                ToolCall(
                    id=f"call_{secrets.token_hex(8)}",
                    function_name=fc.get("name", ""),
                    function_args=json.dumps(fc.get("args", {})),
                )
            )
        return tool_calls if tool_calls else None

    def create_assistant_message(self, response: Any) -> ChatMessage:
        candidates = response.get("candidates") or []
        if not candidates:
            return ChatMessage(role=MODEL_ROLE, content="[No response generated]")

        parts = (candidates[0].get("content") or {}).get("parts") or []
        if not parts:
            return ChatMessage(role=MODEL_ROLE, content="[No response generated]")

        cleaned_parts = [
            {k: v for k, v in part.items() if v is not None} for part in parts
        ]
        return ChatMessage(role=MODEL_ROLE, content=cleaned_parts)

    def create_tool_result_messages(
        self, results: List[ToolResult], conversation: "Conversation"
    ) -> List[ChatMessage]:
        messages = []
        for result in results:
            messages.append(
                ChatMessage(
                    role=TOOL_ROLE,
                    name=result.function_name,
                    content=result.content,
                )
            )
        return messages

    def is_tool_message(self, message: "ChatMessage") -> bool:
        if message.role == TOOL_ROLE:
            return True
        if message.role == MODEL_ROLE and isinstance(message.content, list):
            return any(
                isinstance(item, dict) and item.get("function_call") is not None
                for item in message.content
            )
        return False


class Ollama(LLM):
    def __init__(self, model: str = "llama3.2"):
        from ollama import Client

        self.client = Client()
        self.model = model

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
        content = response.get("message", {}).get("content")
        if content:
            return content
        done_reason = response.get("done_reason", "UNKNOWN")
        return f"Empty response from {self.model} — done_reason: {done_reason}"

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
                        function_args=json.dumps(args),
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

    def __init__(self, model: str = "echo-v1", **kwargs):
        """
        Initializes the Echo mock LLM.

        Parameters
        ----------
        model : str, optional
            The model name to echo in the response, by default "echo-v1".
        **kwargs : Any
            Accepted for signature consistency but not used.
        """
        self.model = model
        self.default_params = kwargs

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
