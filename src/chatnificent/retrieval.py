"""Concrete implementations for knowledge retrievers."""

from abc import ABC, abstractmethod
from typing import Optional


class Retrieval(ABC):
    """Interface for retrieving context for RAG."""

    @abstractmethod
    def retrieve(self, query: str, user_id: str, convo_id: str) -> Optional[str]:
        """Retrieves relevant context to augment a user's query."""
        return None


class NoRetrieval(Retrieval):
    """Default retriever that performs no action."""

    def retrieve(self, query, user_id, convo_id):
        pass
