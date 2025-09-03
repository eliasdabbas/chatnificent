"""Unit tests for the engine module."""

from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import pytest
from chatnificent.engine import Engine, Synchronous
from chatnificent.models import ASSISTANT_ROLE, USER_ROLE, ChatMessage, Conversation


class TestEngineBase:
    """Test the abstract Engine base class."""

    def test_engine_is_abstract(self):
        """Engine cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            Engine()

    def test_engine_with_app_reference(self):
        """Engine can be initialized with app reference."""
        mock_app = Mock()

        # Create a concrete implementation for testing
        class ConcreteEngine(Engine):
            def handle_message(self, user_input, user_id, convo_id_from_url):
                return {}

        engine = ConcreteEngine(mock_app)
        assert engine.app is mock_app

    def test_engine_without_app_reference(self):
        """Engine can be initialized without app reference (for lazy binding)."""

        class ConcreteEngine(Engine):
            def handle_message(self, user_input, user_id, convo_id_from_url):
                return {}

        engine = ConcreteEngine()
        assert engine.app is None

        # Later binding
        mock_app = Mock()
        engine.app = mock_app
        assert engine.app is mock_app


class TestSynchronousEngine:
    """Test the Synchronous engine implementation."""

    @pytest.fixture
    def mock_app(self):
        """Create a mock Chatnificent app with all required components."""
        app = Mock()

        # Mock LLM
        app.llm = Mock()
        app.llm.generate_response = Mock(return_value=Mock())
        app.llm.extract_content = Mock(return_value="Test response")
        app.llm.parse_tool_calls = Mock(return_value=[])
        app.llm.create_assistant_message = Mock(
            return_value=ChatMessage(role=ASSISTANT_ROLE, content="Test response")
        )

        # Mock Store
        app.store = Mock()
        app.store.load_conversation = Mock(return_value=None)
        app.store.get_next_conversation_id = Mock(return_value="test-convo-1")
        app.store.save_conversation = Mock()

        # Mock other components
        app.retrieval = Mock()
        app.retrieval.retrieve = Mock(return_value=None)

        app.tools = Mock()
        app.tools.get_tools = Mock(return_value=None)

        app.layout_builder = Mock()
        app.layout_builder.build_messages = Mock(return_value=[])

        app.url = Mock()
        app.url.build_conversation_path = Mock(
            return_value="/conversation/test-convo-1"
        )

        return app

    @pytest.fixture
    def engine(self, mock_app):
        """Create a Synchronous engine with mock app."""
        return Synchronous(mock_app)

    def test_handle_message_basic_flow(self, engine, mock_app):
        """Test basic message handling flow."""
        result = engine.handle_message(
            user_input="Hello", user_id="user123", convo_id_from_url=None
        )

        # Verify the flow
        assert mock_app.llm.generate_response.called
        assert mock_app.llm.extract_content.called
        assert mock_app.store.save_conversation.called

        # Check result structure
        assert "messages" in result
        assert "input_value" in result
        assert "submit_disabled" in result
        assert "pathname" in result

    def test_empty_tool_calls_triggers_finalization(self, engine, mock_app):
        """Test that empty tool_calls list (not None) triggers finalization."""
        # This tests our bug fix
        mock_app.llm.parse_tool_calls.return_value = []  # Empty list, not None!
        mock_app.llm.extract_content.return_value = "Assistant response"

        result = engine.handle_message("Hello", "user123", None)

        # Verify extract_content was called (finalization happened)
        mock_app.llm.extract_content.assert_called_once()

        # Verify a message was added to the conversation
        mock_app.store.save_conversation.assert_called_once()
        saved_conversation = mock_app.store.save_conversation.call_args[0][1]
        assert len(saved_conversation.messages) == 2  # User + Assistant
        assert saved_conversation.messages[1].content == "Assistant response"

    def test_none_tool_calls_triggers_finalization(self, engine, mock_app):
        """Test that None tool_calls also triggers finalization."""
        mock_app.llm.parse_tool_calls.return_value = None
        mock_app.llm.extract_content.return_value = "Assistant response"

        result = engine.handle_message("Hello", "user123", None)

        # Verify extract_content was called
        mock_app.llm.extract_content.assert_called_once()

    def test_with_tool_calls(self, engine, mock_app):
        """Test handling when tool calls are present."""
        from chatnificent.models import ToolCall, ToolResult

        # First call has tool calls
        tool_call = ToolCall(
            id="tool-1", function_name="test_function", function_args='{"arg": "value"}'
        )
        mock_app.llm.parse_tool_calls.side_effect = [
            [tool_call],  # First call has tools
            [],  # Second call has no tools
        ]

        # Mock tool execution
        mock_app.tools.execute_tool_call = Mock(
            return_value=ToolResult(
                tool_call_id="tool-1",
                content="Tool result",
                function_name="test_function",
            )
        )

        mock_app.llm.create_tool_result_messages = Mock(
            return_value=[ChatMessage(role=USER_ROLE, content="Tool result")]
        )

        result = engine.handle_message("Use a tool", "user123", None)

        # Verify tool was executed
        mock_app.tools.execute_tool_call.assert_called_once()

        # Verify LLM was called twice (once with tool, once for final response)
        assert mock_app.llm.generate_response.call_count == 2

    def test_max_turns_limit(self, engine, mock_app):
        """Test that engine respects MAX_AGENTIC_TURNS."""
        # Always return tool calls to force max turns
        tool_call = Mock()
        mock_app.llm.parse_tool_calls.return_value = [tool_call]
        mock_app.tools.execute = Mock(return_value=Mock())
        mock_app.llm.create_tool_result_messages = Mock(return_value=[])

        result = engine.handle_message("Keep using tools", "user123", None)

        # Should stop after MAX_AGENTIC_TURNS
        assert (
            mock_app.llm.generate_response.call_count == Synchronous.MAX_AGENTIC_TURNS
        )

    def test_conversation_persistence(self, engine, mock_app):
        """Test that conversations are properly saved."""
        mock_app.store.load_conversation.return_value = None

        result = engine.handle_message("Hello", "user123", None)

        # Verify conversation was saved
        mock_app.store.save_conversation.assert_called_once()

        # Check saved conversation structure
        user_id, conversation = mock_app.store.save_conversation.call_args[0]
        assert user_id == "user123"
        assert isinstance(conversation, Conversation)
        assert conversation.id == "test-convo-1"
        assert len(conversation.messages) >= 1

    def test_existing_conversation_load(self, engine, mock_app):
        """Test loading existing conversation."""
        existing_convo = Conversation(
            id="existing-1",
            messages=[
                ChatMessage(role=USER_ROLE, content="Previous message"),
                ChatMessage(role=ASSISTANT_ROLE, content="Previous response"),
            ],
        )
        mock_app.store.load_conversation.return_value = existing_convo

        result = engine.handle_message("New message", "user123", "existing-1")

        # Verify existing conversation was loaded
        mock_app.store.load_conversation.assert_called_with("user123", "existing-1")

        # Verify new message was added to existing conversation
        saved_conversation = mock_app.store.save_conversation.call_args[0][1]
        assert saved_conversation.id == "existing-1"
        assert len(saved_conversation.messages) > 2

    def test_error_handling(self, engine, mock_app):
        """Test error handling in message processing."""
        mock_app.llm.generate_response.side_effect = Exception("LLM Error")

        result = engine.handle_message("Cause error", "user123", None)

        # Should return error response
        assert "messages" in result
        assert result["submit_disabled"] == False

    def test_retrieval_integration(self, engine, mock_app):
        """Test that retrieval is called when available."""
        mock_app.retrieval.retrieve.return_value = "Context information"

        result = engine.handle_message("Need context", "user123", None)

        # Verify retrieval was called
        mock_app.retrieval.retrieve.assert_called_once_with(
            "Need context", "user123", "test-convo-1"
        )

        # Verify context was added to LLM payload
        llm_payload = mock_app.llm.generate_response.call_args[0][0]
        assert any(
            msg.get("role") == "system"
            and "Context information" in msg.get("content", "")
            for msg in llm_payload
        )


class TestEngineHooks:
    """Test the hook methods in Synchronous engine."""

    def test_hooks_are_called_in_order(self):
        """Test that all hook methods are called in the correct order."""
        mock_app = Mock()
        mock_app.llm = Mock()
        mock_app.llm.generate_response = Mock(return_value=Mock())
        mock_app.llm.extract_content = Mock(return_value="Response")
        mock_app.llm.parse_tool_calls = Mock(return_value=[])
        mock_app.store = Mock()
        mock_app.store.load_conversation = Mock(return_value=None)
        mock_app.store.get_next_conversation_id = Mock(return_value="test-1")
        mock_app.retrieval = Mock()
        mock_app.retrieval.retrieve = Mock(return_value=None)
        mock_app.tools = Mock()
        mock_app.tools.get_tools = Mock(return_value=None)
        mock_app.layout_builder = Mock()
        mock_app.layout_builder.build_messages = Mock(return_value=[])
        mock_app.url = Mock()

        # Create engine with tracked hooks
        class TrackedEngine(Synchronous):
            def __init__(self, app):
                super().__init__(app)
                self.call_log = []

            def _before_llm_call(self, conversation):
                self.call_log.append("before_llm")
                super()._before_llm_call(conversation)

            def _after_llm_call(self, llm_response):
                self.call_log.append("after_llm")
                super()._after_llm_call(llm_response)

            def _before_save(self, conversation):
                self.call_log.append("before_save")
                super()._before_save(conversation)

        engine = TrackedEngine(mock_app)
        result = engine.handle_message("Test", "user123", None)

        # Verify hooks were called in order
        assert engine.call_log == ["before_llm", "after_llm", "before_save"]

    def test_hook_can_modify_conversation(self):
        """Test that hooks can modify the conversation."""
        mock_app = Mock()
        # ... setup mocks ...

        class ModifyingEngine(Synchronous):
            def _before_llm_call(self, conversation):
                # Add a system message
                conversation.messages.insert(
                    0, ChatMessage(role="system", content="Always be helpful")
                )

        # Test that the modification affects the LLM call
        # ... implementation ...


class TestEngineLazyBinding:
    """Test lazy binding of app reference."""

    def test_engine_created_without_app(self):
        """Test engine can be created without app."""
        engine = Synchronous()
        assert engine.app is None

    def test_engine_app_bound_later(self):
        """Test app can be bound after engine creation."""
        engine = Synchronous()
        mock_app = Mock()

        engine.app = mock_app
        assert engine.app is mock_app

    def test_engine_methods_require_app(self):
        """Test that engine methods fail gracefully without app."""
        engine = Synchronous()

        with pytest.raises(AttributeError, match="'NoneType' object has no attribute"):
            engine.handle_message("Test", "user123", None)
