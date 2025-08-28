"""
Defines the interface and default implementation for URL management.

This pillar is responsible for parsing application state (like user and
conversation IDs) from the URL pathname and for building valid pathnames
from application state. Abstracting this logic allows developers to easily
customize the application's routing scheme without modifying core callbacks.
"""

from abc import ABC, abstractmethod
from typing import Optional
from urllib.parse import parse_qs, urlencode

from pydantic import BaseModel, Field


class URLParts(BaseModel):
    """
    A structured representation of the state parsed from a URL.

    This model serves as the data contract between the URL pillar and the
    rest of the application, providing a clear, validated structure for
    the components of a URL path.
    """

    user_id: Optional[str] = Field(
        None,
        description="The user ID, if present in the URL. If None, the auth pillar should determine the user.",
    )
    convo_id: Optional[str] = Field(
        None,
        description="The conversation ID, if present. If None, it signifies a new chat.",
    )


class URL(ABC):
    """
    Abstract base class (interface) for URL management.

    This class defines the contract for parsing pathnames into structured
    data and building pathnames from application state.
    """

    @abstractmethod
    def parse(self, pathname: str, search: Optional[str] = None) -> URLParts:
        """
        Parses a URL into a structured URLParts object.

        This method is the single source of truth for interpreting the URL. It
        must gracefully handle various path formats.

        Parameters
        ----------
        pathname : str
            The pathname string from the dcc.Location component.
        search : str, optional
            The search/query string from the dcc.Location component (e.g., "?user=123").

        Returns
        -------
        URLParts
            A Pydantic model containing the parsed user_id and optional convo_id.
        """
        pass

    @abstractmethod
    def build_conversation_path(self, user_id: str, convo_id: str) -> str:
        """
        Builds the URL for a specific, existing conversation.

        Parameters
        ----------
        user_id : str
            The ID of the user who owns the conversation.
        convo_id : str
            The ID of the conversation to link to.

        Returns
        -------
        str
            A formatted URL string.
        """
        pass

    @abstractmethod
    def build_new_chat_path(self, user_id: str) -> str:
        """
        Builds the URL that signifies the creation of a new chat.

        Parameters
        ----------
        user_id : str
            The ID of the user for whom to create a new chat.

        Returns
        -------
        str
            A formatted URL string for a new chat.
        """
        pass


class PathBased(URL):
    """
    URL manager using a simple path-based routing scheme.

    - /<user_id>/<convo_id> for existing conversations.
    - /<user_id>/new for creating a new conversation.
    """

    def parse(self, pathname: str, search: Optional[str] = None) -> URLParts:
        """
        Parses pathnames in the format /<user_id>/<convo_id>/.
        """
        path_parts = pathname.strip("/").split("/")

        if not path_parts or not path_parts[0]:
            return URLParts(user_id=None, convo_id=None)

        user_id = path_parts[0]

        if len(path_parts) == 1:
            return URLParts(user_id=user_id, convo_id=None)

        convo_id_part = path_parts[1]

        if not convo_id_part or convo_id_part.lower() == "new":
            return URLParts(user_id=user_id, convo_id=None)

        return URLParts(user_id=user_id, convo_id=convo_id_part)

    def build_conversation_path(self, user_id: str, convo_id: str) -> str:
        """
        Builds a path in the format /<user_id>/<convo_id>.
        """
        return f"/{user_id}/{convo_id}"

    def build_new_chat_path(self, user_id: str) -> str:
        """
        Builds a path in the format /<user_id>/new.
        """
        return f"/{user_id}/new"


class QueryParams(URL):
    """
    URL manager using query parameters.

    - /chat?user=<user_id>&convo=<convo_id> for existing conversations.
    - /chat?user=<user_id> for a new conversation.
    """

    def parse(self, pathname: str, search: Optional[str] = None) -> URLParts:
        """
        Parses query strings in the format ?user=...&convo=...
        """
        if not search:
            return URLParts(user_id=None, convo_id=None)

        query_params = parse_qs(search.lstrip("?"))

        user_id = query_params.get("user", [None])[0]
        convo_id = query_params.get("convo", [None])[0]

        return URLParts(user_id=user_id, convo_id=convo_id)

    def build_conversation_path(self, user_id: str, convo_id: str) -> str:
        """
        Builds a path in the format /chat?user=...&convo=...
        """
        params = {"user": user_id, "convo": convo_id}
        return f"/chat?{urlencode(params)}"

    def build_new_chat_path(self, user_id: str) -> str:
        """
        Builds a path in the format /chat?user=...
        """
        params = {"user": user_id}
        return f"/chat?{urlencode(params)}"
