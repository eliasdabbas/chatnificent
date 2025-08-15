"""Concrete implementations for persistence managers."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from .models import Conversation


class BasePersistenceManager(ABC):
    """Interface for saving and loading conversation data."""

    @abstractmethod
    def load_conversation(self, user_id: str, convo_id: str) -> Optional[Conversation]:
        """Loads a single conversation from the persistence layer."""
        pass

    @abstractmethod
    def save_conversation(self, user_id: str, conversation: Conversation):
        """Saves a single conversation to the persistence layer."""
        pass

    @abstractmethod
    def list_conversations(self, user_id: str) -> List[str]:
        """Lists all conversation IDs for a given user."""
        pass

    @abstractmethod
    def get_next_conversation_id(self, user_id: str) -> str:
        """Generates a new, unique conversation ID for a user."""
        pass


class InMemoryPersistenceManager(BasePersistenceManager):
    """Saves and loads conversations from an in-memory dictionary."""

    def __init__(self):
        self._store: Dict[str, Conversation] = {}

    def load_conversation(self, user_id: str, convo_id: str) -> Optional[Conversation]:
        return self._store.get(convo_id)

    def save_conversation(self, user_id: str, conversation: Conversation):
        self._store[conversation.id] = conversation.copy(deep=True)

    def list_conversations(self, user_id: str) -> List[str]:
        """Lists all conversation IDs for a given user."""
        return list(self._store.keys())

    def get_next_conversation_id(self, user_id: str) -> str:
        """Generates a new, unique conversation ID for a user."""
        return str(len(self._store) + 1)


class FilePersistenceManager(BasePersistenceManager):
    """Saves and loads conversations from the local file system as JSON."""

    def __init__(self, base_dir: str):
        pass

    def load_conversation(self, user_id: str, convo_id: str) -> Optional[Conversation]:
        pass

    def save_conversation(self, user_id: str, conversation: Conversation):
        pass

    def list_conversations(self, user_id: str) -> List[str]:
        pass

    def get_next_conversation_id(self, user_id: str) -> str:
        pass
