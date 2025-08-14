"""Concrete implementations for persistence managers."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict

from .models import Conversation


class BasePersistenceManager(ABC):
    """Interface for saving and loading conversation data."""

    @abstractmethod
    def load_conversation(self, convo_id: str, user_id: str) -> Conversation:
        """Loads a single conversation from the persistence layer."""
        pass

    @abstractmethod
    def save_conversation(self, conversation: Conversation, user_id: str):
        """Saves a single conversation to the persistence layer."""
        pass


class InMemoryPersistenceManager(BasePersistenceManager):
    """Saves and loads conversations from an in-memory dictionary."""

    def __init__(self):
        self._store: Dict[str, Conversation] = {}

    def load_conversation(self, convo_id: str, user_id: str) -> Conversation:
        return self._store.get(convo_id, Conversation(id=convo_id))

    def save_conversation(self, conversation: Conversation, user_id: str):
        self._store[conversation.id] = conversation.copy(deep=True)


class FilePersistenceManager(BasePersistenceManager):
    """Saves and loads conversations from the local file system as JSON."""

    def __init__(self, base_dir: str):
        pass

    def load_conversation(self, convo_id: str, user_id: str) -> Conversation:
        pass

    def save_conversation(self, conversation: Conversation, user_id: str):
        pass
