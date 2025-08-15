"""
Defines the core Pydantic data models for the application.

These models serve as the formal, validated data contract between all other pillars,
aligning with conventions from industry-standard libraries like the OpenAI SDK.
"""

from typing import List, Literal, TypeAlias

from pydantic import BaseModel, Field

USER_ROLE = "user"
ASSISTANT_ROLE = "assistant"
SYSTEM_ROLE = "system"
TOOL_ROLE = "tool"


Role: TypeAlias = Literal["user", "assistant", "system", "tool"]


class ChatMessage(BaseModel):
    """Represents a single message within a conversation."""

    role: Role
    content: str


class Conversation(BaseModel):
    """Represents a complete chat conversation session."""

    id: str
    messages: List[ChatMessage] = Field(default_factory=list)
