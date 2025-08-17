"""Concrete implementations for LLM providers."""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Union


class LLM(ABC):
    """Abstract Base Class for all LLM providers."""

    @abstractmethod
    def generate_response(
        self, messages: List[Dict[str, Any]], model: str, **kwargs: Any
    ) -> Union[Dict[str, Any], Any, Iterator[Any]]:
        """Generates a response from the LLM provider.

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
        Union[Any, Iterator[Any]]
            The provider's native, rich response object for a non-streaming
            call, or an iterator of native chunk objects for a streaming call.
        """
        pass

    @abstractmethod
    def extract_content(self, response: Any) -> str:
        """Extracts the text content from the provider's response object.

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
    ) -> Dict[str, Any]:
        raw_response = self.client.chat.completions.create(
            messages=messages, model=model or self.model, **kwargs
        )

        return {
            "content": raw_response.choices[0].message.content,
            "raw_response": raw_response.model_dump(),
        }

    def extract_content(self, response: Any) -> str:
        return response.choices[0].message.content


class Gemini(LLM):
    def __init__(self):
        from google import genai

        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def generate_response(self, messages, model, **kwargs):
        chat = self.client.chats.create(model=model)
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
        raw_response = self.client.messages.create(
            model=model or self.model, messages=messages, **kwargs
        )

        return {
            "content": raw_response.content[0].text,
            "raw_response": raw_response.model_dump(),
        }

    def extract_content(self, response: Any) -> str:
        return response.content[0].text


class Ollama(LLM):
    def __init__(self):
        from ollama import Client

        self.client = Client()

    def generate_response(self, messages, model, **kwargs):
        raw_response = self.client.chat(model=model, messages=messages, **kwargs)

        return {
            "content": raw_response["message"]["content"],
            "raw_response": raw_response,
        }

    def extract_content(self, response: Any) -> str:
        return response["message"]["content"]


class OpenRouter(LLM):
    def __init__(self):
        from openai import OpenAI

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
            extra_headers={
                "HTTP-Referer": "Chatnificent.com",
                "X-Title": "Chatnificent",
            },
        )

    def generate_response(self, messages, model, **kwargs):
        return self.client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )

    def extract_content(self, response: Any) -> str:
        return response.choices[0].message.content


class DeepSeek(LLM):
    def __init__(self):
        from openai import OpenAI

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )

    def generate_response(self, messages, model, **kwargs):
        return self.client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )

    def extract_content(self, response: Any) -> str:
        return response.choices[0].message.content
