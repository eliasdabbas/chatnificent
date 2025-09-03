"""Concrete implementations for authentication managers."""

import uuid
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


class Anonymous(Auth):
    """Anonymous user authentication with short UUID-based session isolation.

    Each browser session gets a unique short UUID (first segment) that persists
    in the URL path. When users visit the root path, they are redirected to
    /<short-uuid>/new for clean, manageable URLs.

    Perfect for:
    - Documentation chatbots
    - Demo applications
    - Public tools requiring privacy
    - Development/testing
    """

    def get_current_user_id(self, **kwargs) -> str:
        """Generate a short unique user ID for anonymous sessions.

        Returns the first segment of a UUID4 (8 characters) for clean URLs
        like /a1b2c3d4/new instead of /a1b2c3d4-e5f6-7g8h-9i0j-k1l2m3n4o5p6/new

        Parameters
        ----------
        **kwargs
            May contain 'pathname' from URL routing

        Returns
        -------
        str
            A short UUID segment for user identification
        """
        return str(uuid.uuid4()).split("-")[0]
