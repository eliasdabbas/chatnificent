"""
Core data models for the Chatnificent framework.

Messages are plain dicts — each LLM provider owns its own format.
Conversation is a lightweight dataclass grouping an id with a message list.
Role constants prevent typos across pillars.
"""

import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

USER_ROLE = "user"
ASSISTANT_ROLE = "assistant"
SYSTEM_ROLE = "system"
TOOL_ROLE = "tool"
MODEL_ROLE = "model"


@dataclass
class Conversation:
    """A chat conversation: an id and a list of message dicts."""

    id: str
    messages: List[Dict[str, Any]] = field(default_factory=list)

    def copy(self, deep: bool = False) -> "Conversation":
        """Return a copy of this conversation."""
        if deep:
            return Conversation(id=self.id, messages=copy.deepcopy(self.messages))
        return Conversation(id=self.id, messages=list(self.messages))
