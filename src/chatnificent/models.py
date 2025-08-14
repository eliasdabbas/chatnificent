"""
Defines the core Pydantic data models for the application.

These models serve as the formal, validated data contract between all other pillars,
aligning with conventions from industry-standard libraries like the OpenAI SDK.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

# --- Constants ---
USER_ROLE = "user"
ASSISTANT_ROLE = "assistant"
SYSTEM_ROLE = "system"
TOOL_ROLE = "tool"
Role = Literal[USER_ROLE, ASSISTANT_ROLE, SYSTEM_ROLE, TOOL_ROLE]


# --- Models ---
class ChatMessage(BaseModel):
    """Represents a single message within a conversation."""

    role: Role
    content: str
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Conversation(BaseModel):
    """Represents a complete chat conversation session."""

    id: str
    messages: List[ChatMessage] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
