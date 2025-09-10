"""
Defines the core Pydantic data models for the application.

These models serve as the formal, validated data contract between all other pillars,
aligning with conventions from industry-standard libraries like the OpenAI SDK.
"""

import json
import logging
from typing import Any, Dict, List, Literal, Optional, TypeAlias, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

USER_ROLE = "user"
ASSISTANT_ROLE = "assistant"
SYSTEM_ROLE = "system"
TOOL_ROLE = "tool"
MODEL_ROLE = "model"

Role: TypeAlias = Literal["system", "user", "assistant", "tool", "model"]
ContentBlock: TypeAlias = Dict[str, Any]
MessageContent: TypeAlias = Union[str, List[ContentBlock], None]


class ChatMessage(BaseModel):
    """Represents a single message within a conversation.

    This model is designed to be compatible with several LLM providers, supporting both
    simple text content and structured tool-calling messages.
    """

    role: Role
    content: MessageContent = Field(
        default=None,
        description="The message content. Can be a string, a list of dicts (blocks), or null.",
    )
    tool_calls: Optional[List[Dict[str, Any]]] = Field(default=None)
    tool_call_id: Optional[str] = Field(default=None)
    name: Optional[str] = Field(default=None)


class Conversation(BaseModel):
    """Represents a complete chat conversation session."""

    id: str = Field(..., min_length=1, description="Non-empty conversation identifier")
    messages: List[ChatMessage] = Field(default_factory=list)


class ToolCall(BaseModel):
    """Standardized request to execute a tool (Produced by LLM)."""

    id: str = Field(..., description="Unique identifier for the call.")
    function_name: str = Field(..., description="Name of the tool to execute.")
    function_args: str = Field(
        ..., description="Arguments serialized as a JSON string."
    )

    def get_args_dict(self) -> Dict[str, Any]:
        """Helper to deserialize arguments."""
        try:
            return json.loads(self.function_args)
        except json.JSONDecodeError:
            logger.error(
                f"Failed to parse JSON arguments for tool call {self.id}: {self.function_args}"
            )
            return {}


class ToolResult(BaseModel):
    """Standardized result of a tool execution (Produced by Tools)."""

    tool_call_id: str
    function_name: str
    content: str = Field(..., description="The result of the tool execution.")
    is_error: bool = False
