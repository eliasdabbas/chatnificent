"""Concrete implementations for conversation managers."""

from abc import ABC, abstractmethod
from typing import Dict, List


class BaseConversationManager(ABC):
    """Interface for managing conversation lifecycle."""

    @abstractmethod
    def list_conversations(self, user_id: str) -> List[Dict[str, str]]:
        """Lists all conversations for a given user."""
        pass

    @abstractmethod
    def get_next_conversation_id(self, user_id: str) -> str:
        """Generates a new, unique conversation ID for a user."""
        pass


class InMemoryConversationManager(BaseConversationManager):
    """Manages conversation lifecycle in memory."""

    def __init__(self):
        self._conversations: Dict[str, List[Dict[str, str]]] = {}

    def list_conversations(self, user_id: str) -> List[Dict[str, str]]:
        return self._conversations.get(user_id, [])

    def get_next_conversation_id(self, user_id: str) -> str:
        if user_id not in self._conversations:
            self._conversations[user_id] = []
        next_id = len(self._conversations[user_id]) + 1

        # Add a placeholder title for the new conversation
        new_convo = {"id": str(next_id), "title": f"Chat {next_id}"}
        if not any(c["id"] == str(next_id) for c in self._conversations[user_id]):
            self._conversations[user_id].append(new_convo)

        return str(next_id)


class FileConversationManager(BaseConversationManager):
    """Manages conversation lifecycle based on file system directories."""

    def __init__(self, base_dir: str):
        pass

    def list_conversations(self, user_id: str) -> List[Dict[str, str]]:
        pass

    def get_next_conversation_id(self, user_id: str) -> str:
        pass
