"""Concrete implementations for authentication managers."""

from abc import ABC, abstractmethod


class Auth(ABC):
    """Interface for identifying the current user."""

    @abstractmethod
    def get_current_user_id(self, **kwargs) -> str:
        """Determines and returns the ID of the current user."""
        pass


class SingleUser(Auth):
    """A simple auth manager for single-user apps."""

    def __init__(self, user_id: str = "chat"):
        """Initialize with a user ID.

        Parameters
        ----------
        user_id : str, default="chat"
            User identifier. Non-string values will be converted to strings
            to enforce the Auth interface contract.
        """
        self._user_id = str(user_id)

    def get_current_user_id(self, **kwargs) -> str:
        return self._user_id
