"""
Tests for the Tools pillar implementations.

The Tools pillar handles agentic tool execution. It has a clearly defined
contract, making it a good candidate for validating our testing patterns.
"""

import json
from typing import Any, Dict, List

import pytest
from chatnificent.models import ToolCall, ToolResult
from chatnificent.tools import NoTool, PythonTool, Tool


@pytest.fixture
def sample_tool_call() -> ToolCall:
    """Fixture for a sample valid ToolCall."""
    return ToolCall(
        id="call_123",
        function_name="test_function",
        function_args='{"arg1": "value1"}',
    )


class TestToolContract:
    """Test that all Tool implementations follow the same contract."""

    @pytest.fixture(params=[NoTool, PythonTool])
    def tool_implementation(self, request):
        """Parametrized fixture providing all Tool implementations."""
        return request.param()

    def test_tool_implements_interface(self, tool_implementation):
        """Test that all implementations properly inherit from Tool base class."""
        assert isinstance(tool_implementation, Tool)
        assert hasattr(tool_implementation, "get_tools")
        assert hasattr(tool_implementation, "execute_tool_call")

    def test_get_tools_returns_list(self, tool_implementation):
        """Test that get_tools always returns a list of dictionaries."""
        tools = tool_implementation.get_tools()
        assert isinstance(tools, list)
        assert all(isinstance(tool, dict) for tool in tools)

    def test_execute_tool_call_signature(self, tool_implementation, sample_tool_call):
        """Test that execute_tool_call accepts a ToolCall and returns a ToolResult."""
        result = tool_implementation.execute_tool_call(sample_tool_call)
        assert isinstance(result, ToolResult)


class TestNoTool:
    """Test the NoTool implementation specifically."""

    @pytest.fixture
    def tool(self) -> NoTool:
        return NoTool()

    def test_get_tools_returns_empty_list(self, tool):
        """Test that NoTool returns an empty tools list."""
        assert tool.get_tools() == []

    def test_execute_tool_call_returns_error(self, tool, sample_tool_call):
        """Test that execute_tool_call consistently returns an error message."""
        result = tool.execute_tool_call(sample_tool_call)
        assert isinstance(result, ToolResult)
        assert result.tool_call_id == sample_tool_call.id
        assert result.is_error is True
        assert "NoTool handler is active" in result.content


class TestPythonTool:
    """Tests for the PythonTool implementation."""

    @pytest.fixture
    def tool(self) -> PythonTool:
        return PythonTool()

    def test_initialization(self, tool):
        """Test that PythonTool initializes with an empty registry."""
        assert tool.get_tools() == []
        assert tool._registry == {}

    def test_register_callable(self, tool):
        """Test that a callable function can be registered."""

        def my_func():
            pass

        tool.register_function(my_func)
        assert "my_func" in tool._registry
        assert tool._registry["my_func"] == my_func

    def test_register_non_callable_raises_error(self, tool):
        """Test that registering a non-callable raises a ValueError."""
        with pytest.raises(ValueError):
            tool.register_function("not_a_function")

    def test_generate_schema_basic_function(self, tool):
        """Test that the private _generate_schema method generates correct schema for functions with and without parameters."""

        def no_params_func():
            pass

        def params_func(a=10, b="hello"):
            return a, b

        # Test schema for function with no parameters
        schema_no_params = tool._generate_schema(no_params_func)
        assert schema_no_params is not None
        assert schema_no_params["type"] == "function"
        assert schema_no_params["function"]["name"] == "no_params_func"
        assert schema_no_params["function"]["description"] == ""
        # Only check parameters if present
        if "parameters" in schema_no_params["function"]:
            assert schema_no_params["function"]["parameters"]["type"] == "object"
            assert schema_no_params["function"]["parameters"]["properties"] == {}
            assert schema_no_params["function"]["parameters"]["required"] == []

        # Test schema for function with parameters
        schema_params = tool._generate_schema(params_func)
        assert schema_params is not None
        assert schema_params["type"] == "function"
        assert schema_params["function"]["name"] == "params_func"
        assert schema_params["function"]["description"] == ""
        assert "parameters" in schema_params["function"]
        params = schema_params["function"]["parameters"]
        assert params["type"] == "object"
        assert set(params["properties"].keys()) == {"a", "b"}
        assert isinstance(params["required"], list)

    def test_execute_unknown_tool_returns_error(self, tool, sample_tool_call):
        """Test that executing a non-existent tool returns a ToolResult error."""
        result = tool.execute_tool_call(sample_tool_call)
        assert result.is_error is True
        assert "not found" in result.content
        assert result.tool_call_id == sample_tool_call.id

    def test_execute_tool_with_valid_args(self, tool):
        """Test executing a registered function with valid arguments."""

        def add(a: int, b: int) -> int:
            return a + b

        tool.register_function(add)
        tool_call = ToolCall(
            id="add_1", function_name="add", function_args='{"a": 5, "b": 10}'
        )
        result = tool.execute_tool_call(tool_call)

        assert result.is_error is False
        assert result.tool_call_id == "add_1"
        assert result.content == "15"

    def test_execute_tool_with_no_args(self, tool):
        """Test executing a registered function that takes no arguments."""

        def get_pi():
            return 3.14159

        tool.register_function(get_pi)
        tool_call = ToolCall(id="pi_1", function_name="get_pi", function_args="{}")
        result = tool.execute_tool_call(tool_call)

        assert result.is_error is False
        assert result.content == "3.14159"

    def test_execute_tool_with_invalid_args_returns_error(self, tool):
        """Test that providing incorrect arguments returns a TypeError error."""

        def greet(name: str):
            return f"Hello, {name}"

        tool.register_function(greet)
        # Calling with a missing 'name' argument
        tool_call = ToolCall(
            id="greet_1", function_name="greet", function_args='{"wrong_arg": "World"}'
        )
        result = tool.execute_tool_call(tool_call)

        assert result.is_error is True
        assert "Invalid arguments" in result.content
        assert isinstance(result.content, str)

    def test_execute_tool_with_malformed_json_args(self, tool):
        """Test that malformed JSON in function_args returns an error."""

        def my_func(a):
            pass

        tool.register_function(my_func)
        tool_call = ToolCall(
            id="json_err", function_name="my_func", function_args='{"a": 1,'
        )  # Malformed
        result = tool.execute_tool_call(tool_call)

        assert result.is_error is True
        assert "Failed to parse arguments" in result.content

    def test_tool_result_serialization(self, tool):
        """Test that non-string results are properly serialized."""

        def get_data():
            return {"key": "value", "numbers": [1, 2, 3]}

        tool.register_function(get_data)
        tool_call = ToolCall(id="data_1", function_name="get_data", function_args="{}")
        result = tool.execute_tool_call(tool_call)

        assert result.is_error is False
        # The dict result should be converted to a JSON string
        assert result.content == '{"key": "value", "numbers": [1, 2, 3]}'
        assert isinstance(json.loads(result.content), dict)

    def test_tool_result_non_json_serializable(self, tool):
        """Test that non-JSON-serializable results are converted to strings."""

        class NonSerializable:
            def __str__(self):
                return "NonSerializableObject"

        def get_non_serializable():
            return NonSerializable()

        tool.register_function(get_non_serializable)
        tool_call = ToolCall(
            id="non_serial_1", function_name="get_non_serializable", function_args="{}"
        )
        result = tool.execute_tool_call(tool_call)

        assert result.is_error is False
        assert result.content == "NonSerializableObject"


class TestToolInterface:
    """Test the Tool abstract base class interface."""

    def test_tool_is_abstract(self):
        """Test that Tool cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            Tool()

    def test_subclass_must_implement_get_tools(self):
        """Test that a subclass must implement the `get_tools` method."""
        with pytest.raises(TypeError, match="get_tools"):

            class IncompleteTool(Tool):
                def execute_tool_call(self, tool_call: ToolCall) -> ToolResult:
                    pass

            IncompleteTool()  # Attempt to instantiate

    def test_subclass_must_implement_execute_tool_call(self):
        """Test that a subclass must implement the `execute_tool_call` method."""
        with pytest.raises(TypeError, match="execute_tool_call"):

            class IncompleteTool2(Tool):
                def get_tools(self) -> List[Dict[str, Any]]:
                    return []

            IncompleteTool2()  # Attempt to instantiate

    def test_complete_subclass_works(self):
        """Test that a complete subclass can be instantiated."""

        class CompleteTool(Tool):
            def get_tools(self) -> List[Dict[str, Any]]:
                return []

            def execute_tool_call(self, tool_call: ToolCall) -> ToolResult:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    function_name=tool_call.function_name,
                    content="Success",
                )

        tool = CompleteTool()
        assert isinstance(tool, Tool)
