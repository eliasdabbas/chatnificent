"""Concrete implementations for LLM providers."""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Union


class LLM(ABC):
    """Abstract Base Class for all LLM providers."""

    @abstractmethod
    def generate_response(
        self, messages: List[Dict[str, Any]], model: str, **kwargs: Any
    ) -> Any:
        """Generates a response from the LLM provider.

        This method should return the provider's native, rich response object
        directly from their SDK.

        Parameters
        ----------
        messages : List[Dict[str, Any]]
            A list of message dictionaries, conforming to the provider's
            expected format.
        model : str
            The specific model to use for the generation.
        **kwargs : Any
            Provider-specific parameters (e.g., stream, temperature) to be
            passed directly to the SDK.

        Returns
        -------
        Any
            The provider's native, rich response object.
        """
        pass

    @abstractmethod
    def extract_content(self, response: Any) -> str:
        """Extracts the text content from the provider's native response object.

        Parameters
        ----------
        response : Any
            The provider's native response object from generate_response.

        Returns
        -------
        str
            The extracted text content from the response.
        """
        pass


class OpenAI(LLM):
    def __init__(self, default_model: str = "gpt-4o"):
        from openai import OpenAI

        self.client = OpenAI()
        self.model = default_model

    def generate_response(
        self, messages: List[Dict[str, Any]], model=None, **kwargs: Any
    ) -> Any:
        return self.client.chat.completions.create(
            messages=messages, model=model or self.model, **kwargs
        )

    def extract_content(self, response: Any) -> str:
        return response.choices[0].message.content


class Gemini(LLM):
    def __init__(self, default_model: str = "gemini-1.5-flash"):
        from google import genai

        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model = default_model

    def generate_response(self, messages, model=None, **kwargs):
        chat = self.client.chats.create(model=model or self.model)
        current_message = messages[-1]["content"]
        return chat.send_message(current_message)

    def extract_content(self, response: Any) -> str:
        return response.text


class Anthropic(LLM):
    def __init__(self, default_model: str = "claude-3-5-sonnet-20241022"):
        from anthropic import Anthropic

        self.client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = default_model

    def generate_response(self, messages, model=None, **kwargs):
        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = 4096
        return self.client.messages.create(
            model=model or self.model, messages=messages, **kwargs
        )

    def extract_content(self, response: Any) -> str:
        return response.content[0].text


class Ollama(LLM):
    def __init__(self, default_model: str = "llama3.1"):
        from ollama import Client

        self.client = Client()
        self.model = default_model

    def generate_response(self, messages, model=None, **kwargs):
        return self.client.chat(model=model or self.model, messages=messages, **kwargs)

    def extract_content(self, response: Any) -> str:
        return response["message"]["content"]


class OpenRouter(LLM):
    def __init__(self, default_model: str = "openai/gpt-4o-mini"):
        from openai import OpenAI

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
        self.model = default_model

    def generate_response(self, messages, model=None, **kwargs):
        return self.client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            extra_headers={
                "HTTP-Referer": "Chatnificent.com",
                "X-Title": "Chatnificent",
            },
            **kwargs,
        )

    def extract_content(self, response: Any) -> str:
        return response.choices[0].message.content


class DeepSeek(LLM):
    def __init__(self, default_model: str = "deepseek/deepseek-chat"):
        from openai import OpenAI

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
        self.model = default_model

    def generate_response(self, messages, model=None, **kwargs):
        return self.client.chat.completions.create(
            model=model or self.model, messages=messages, **kwargs
        )

    def extract_content(self, response: Any) -> str:
        return response.choices[0].message.content


class Echo(LLM):
    def __init__(self, default_model: str = "echo-v1"):
        self.model = default_model

    def generate_response(self, messages, model=None, **kwargs):
        import time

        time.sleep(0.8)
        user_prompt = messages[-1]["content"] if messages else "No message provided"
        content = f"**Echo LLM - static response for testing**\n\n_Your prompt:_\n\n\n\n{user_prompt}"

        return {
            "content": content,
            "raw_response": "Echo LLM - static response for testing",
        }

    def extract_content(self, response: Any) -> str:
        if isinstance(response, dict) and "content" in response:
            return response["content"]
        return str(response)
