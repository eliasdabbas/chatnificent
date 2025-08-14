"""Concrete implementations for LLM providers."""

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Union


class BaseLLMProvider(ABC):
    """Abstract Base Class for all LLM providers."""

    @abstractmethod
    def generate_response(
        self, messages: List[Dict[str, Any]], model: str, **kwargs: Any
    ) -> Union[Any, Iterator[Any]]:
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


class OpenAIProvider(BaseLLMProvider):
    def __init__(self):
        from openai import OpenAI

        self.client = OpenAI()

    def generate_response(
        self, messages: List[Dict[str, Any]], model: str, **kwargs: Any
    ) -> Any:
        print("Generating response!!!!!!")
        return self.client.chat.completions.create(
            messages=messages, model=model, **kwargs
        )


class GeminiProvider(BaseLLMProvider):
    def __init__(self):
        from google import genai

        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def generate_response(self, messages, model, **kwargs):
        chat = self.client.chats.create(model=model)
        current_message = messages[-1]["content"]
        return chat.send_message(current_message)


class AnthropicProvider(BaseLLMProvider):
    def __init__(self):
        from anthropic import Anthropic

        self.client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def generate_response(self, messages, model, **kwargs):
        return self.client.messages.create(model=model, messages=messages, **kwargs)


class OllamaProvider(BaseLLMProvider):
    def __init__(self):
        from ollama import Client

        self.client = Client()

    def generate_response(self, messages, model, **kwargs):
        return self.client.chat(model=model, messages=messages, **kwargs)


class OpenRouterProvider(BaseLLMProvider):
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


class DeepSeekProvider(BaseLLMProvider):
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
