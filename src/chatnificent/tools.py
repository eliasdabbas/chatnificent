"""Concrete implementations for tool handlers."""

import inspect
import json
import logging
import typing
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from .models import ToolCall, ToolResult

logger = logging.getLogger(__name__)


def _parse_docstring(docstring: str) -> Dict[str, Any]:
    if not docstring:
        return {"description": "", "params": {}}

    cleaned_doc = inspect.cleandoc(docstring)
    lines = cleaned_doc.strip().splitlines()
    description = []
    params = {}
    current_section = "description"

    for i, line in enumerate(lines):
        stripped_line = line.strip()

        if not stripped_line:
            if current_section == "description" and description:
                current_section = "other"
            continue

        if (
            stripped_line.lower() == "parameters"
            and i + 1 < len(lines)
            and lines[i + 1].strip().startswith(("---", "==="))
        ):
            current_section = "params"
            continue
        elif stripped_line.lower().startswith(("args:", "parameters:", "arguments:")):
            current_section = "params"
            continue

        elif stripped_line.lower().startswith(
            ("returns:", "raises:", "return", "raise")
        ):
            current_section = "other"
            continue

        if current_section == "description":
            description.append(stripped_line)
        elif current_section == "params":
            # Try to parse "arg_name (type): Description" or "arg_name: Description"
            parts = stripped_line.split(":", 1)
            if len(parts) == 2:
                arg_name_and_type = parts[0].strip()
                arg_desc = parts[1].strip()

                arg_name = arg_name_and_type.split("(")[0].strip()
                if arg_name:
                    params[arg_name] = arg_desc

    return {"description": " ".join(description), "params": params}


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

    def get_tools(self) -> List[Dict[str, Any]]:
        """Attempts to generate schemas for all registered functions.."""
        schemas = []
        for func in self._registry.values():
            schema = self._generate_schema(func)
            if schema:
                schemas.append(schema)
        return schemas

    def _generate_schema(self, func: Callable) -> Optional[Dict[str, Any]]:
        """Generates the canonical (OpenAI) tool specification using standard libraries."""

        # Use the native parser
        doc_info = _parse_docstring(func.__doc__ or "")
        description = doc_info["description"]
        param_docs = doc_info["params"]

        parameters = {"type": "object", "properties": {}, "required": []}

        try:
            sig = inspect.signature(func)
            type_hints = typing.get_type_hints(func)
        except Exception as e:
            logger.debug(
                f"Could not resolve type hints for {func.__name__}: {e}. Using fallback."
            )
            try:
                sig = inspect.signature(func)
                type_hints = {}
            except ValueError:
                logger.warning(
                    f"Could not inspect signature for {func.__name__}. Skipping tool registration."
                )
                return None

        for name, param in sig.parameters.items():
            if name == "self":
                continue

            if name in type_hints:
                param_type = type_hints[name]
            elif param.annotation != inspect.Parameter.empty:
                param_type = param.annotation
            else:
                param_type = Any

            prop = self._map_type_to_json_schema(param_type)

            if name in param_docs and param_docs[name]:
                prop["description"] = param_docs[name]

            parameters["properties"][name] = prop

            if param.default == inspect.Parameter.empty:
                parameters["required"].append(name)

        # Construct the OpenAI format (Canonical)
        function_def = {
            "name": func.__name__,
            "description": description,
        }

        # Only include parameters if there are any (avoid empty object for Gemini compatibility)
        if parameters["properties"]:
            function_def["parameters"] = parameters

        return {
            "type": "function",
            "function": function_def,
        }

    def _map_type_to_json_schema(self, py_type: Any) -> Dict[str, Any]:
        """Helper to map Python types to JSON schema types using standard library."""
        origin = typing.get_origin(py_type)
        args = typing.get_args(py_type)

        if origin is typing.Union and type(None) in args and len(args) == 2:
            actual_type = next(arg for arg in args if arg is not type(None))
            return self._map_type_to_json_schema(actual_type)

        if py_type == str:
            return {"type": "string"}
        elif py_type == int:
            return {"type": "integer"}
        elif py_type == float:
            return {"type": "number"}
        elif py_type == bool:
            return {"type": "boolean"}
        elif origin is list or py_type == list:
            item_schema = (
                self._map_type_to_json_schema(args[0]) if args else {"type": "string"}
            )
            return {"type": "array", "items": item_schema}
        elif origin is dict or py_type == dict:
            return {"type": "object"}
        elif origin is typing.Literal:
            return {"type": "string", "enum": list(args)}
        else:
            # Fallback for complex types (Pydantic models, Any, etc.) or unresolvable types
            # Defaulting to string is often the safest fallback for LLM input
            return {
                "type": "string",
                "description": f"Type: {getattr(py_type, '__name__', 'Any')} (Treated as string)",
            }

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

        # if not args and tool_call.function_args.strip() not in ["{}", "null"]:
        if (
            not args
            and tool_call.function_args
            and tool_call.function_args.strip() not in ["{}", "null", ""]
        ):
            args = self._attempt_argument_recovery(func, tool_call.function_args)
            if args is None:
                return ToolResult(
                    tool_call_id=tool_call_id,
                    function_name=func_name,
                    content=f"Error: Failed to parse arguments for tool '{func_name}'.",
                    is_error=True,
                )
        if args is None:
            args = {}
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
            logger.exception(f"Error during execution of tool '{func_name}'")
            return ToolResult(
                tool_call_id=tool_call_id,
                function_name=func_name,
                content=f"Error: Invalid arguments provided for tool '{func_name}': {e}",
                is_error=True,
            )
        except Exception as e:
            logger.exception(f"Error during execution of tool '{func_name}'")
            return ToolResult(
                tool_call_id=tool_call_id,
                function_name=func_name,
                content=f"Error during execution of tool '{func_name}': {type(e).__name__}: {e}",
                is_error=True,
            )

    def _attempt_argument_recovery(
        self, func: Callable, raw_args: str
    ) -> Optional[Dict[str, Any]]:
        """Attempts to map raw string arguments if JSON parsing failed."""
        try:
            sig = inspect.signature(func)
        except Exception:
            return None

        params = [p for p in sig.parameters.keys() if p != "self"]

        if len(params) == 1:
            try:
                value = json.loads(raw_args)
                if isinstance(value, dict):
                    return None
                return {params[0]: value}
            except json.JSONDecodeError:
                # If it looks like malformed JSON (starts with { or [), don't recover
                stripped = raw_args.strip()
                if stripped.startswith(("{", "[")):
                    return None
                return {params[0]: raw_args}
        return None
