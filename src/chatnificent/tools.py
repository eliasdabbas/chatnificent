"""Concrete implementations for tool handlers."""

import json
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from .models import ToolCall, ToolResult


class Tool(ABC):
    """Interface for defining and executing agentic tools."""

    @abstractmethod
    def get_tools(self) -> List[Dict[str, Any]]:
        """Returns a list of tool specifications for the LLM.

        Returns
        -------
        List[Dict[str, Any]]
            A list of tool definitions, conforming to a format like OpenAI's
            JSON schema.
        """
        return []

    @abstractmethod
    def execute_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Executes a single tool call and returns the result.

        Parameters
        ----------
        tool_call : Dict[str, Any]
            A dictionary representing a single tool call request from the LLM.
            Expected format: {"id": "...", "function_name": "...", "function_args": "..."}

        Returns
        -------
        Dict[str, Any]
            A dictionary representing the result of the tool execution.
            Expected format: {"tool_call_id": "...", "content": "..."}
        """
        pass


class NoTool(Tool):
    """Default handler that provides no tools and performs no actions."""

    def get_tools(self) -> List[Dict[str, Any]]:
        return []

    def execute_tool_call(self, tool_call: Dict[str, Any]) -> ToolResult:
        return ToolResult(
            tool_call_id=tool_call.id,
            function_name=tool_call.function_name,
            content="Error: Tool execution attempted, but NoTool handler is active",
            is_error=True,
        )


class PythonTool(Tool):
    """Flagship implementation for registering and executing Python functions."""

    def __init__(self):
        self._registry: Dict[str, Callable] = {}

    def register_function(self, func: Callable) -> None:
        """Registers a Python function and its corresponding JSON schema as a tool.
        Parameters
        ----------
        func : Callable
            The Python function to be executed.
        """
        if not callable(func):
            raise ValueError("Provided object is not callable.")
        self._registry[func.__name__] = func

    def get_tools(self, format: str = "openai") -> List[Dict[str, Any]]:
        """Attempts to generate schemas for all registered functions.."""
        schemas = []
        for func in self._registry:
            schema = self._generate_schema(func, format)
            if schema:
                schemas.append(schema)
        return schemas

    def _generate_schema(self, func: Callable, format: str) -> Optional[Dict[str, Any]]:
        return None

    def execute_tool_call(self, tool_call: ToolCall) -> ToolResult:
        """Execute the requested function"""
        func_name = tool_call.function_name
        tool_call_id = tool_call.id

        if func_name not in self._registry:
            return ToolResult(
                tool_call_id=tool_call_id,
                function_name=func_name,
                content=f"Error: Tool '{func_name}' not found.",
                is_error=True,
            )

        func = self._registry[func_name]
        args = tool_call.get_args_dict()

        if not args and tool_call.function_args.strip() not in ["{}", "null"]:
            return ToolResult(
                tool_call_id=tool_call_id,
                function_name=func_name,
                content=f"Error: Failed to parse arguments for tool '{func_name}'.",
                is_error=True,
            )
        try:
            result = func(**args)
            if not isinstance(result, str):
                try:
                    result_str = json.dumps(result)
                except TypeError:
                    result_str = str(result)
            else:
                result_str = result
            return ToolResult(
                tool_call_id=tool_call_id,
                function_name=func_name,
                content=result_str,
                is_error=False,
            )
        except TypeError as e:
            return ToolResult(
                tool_call_id=tool_call_id,
                function_name=func_name,
                content=f"Error: Invalid arguments provided for tool '{func_name}': {e}",
                is_error=True,
            )
