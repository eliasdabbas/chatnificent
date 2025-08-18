"""Concrete implementations for tool handlers."""

from abc import ABC, abstractmethod
from typing import Any, List


class Tool(ABC):
    """Interface for executing agentic tools."""

    @abstractmethod
    def get_tools(self) -> List[Any]:
        """Returns a list of tool specifications for the LLM."""
        return []

    @abstractmethod
    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Executes a tool with the given name and arguments."""
        pass


class NoTool(Tool):
    """Default handler that provides no tools and does nothing."""

    def get_tools(self) -> List:
        return []

    def execute_tool(self, tool_name: str, **kwargs) -> str:
        pass
