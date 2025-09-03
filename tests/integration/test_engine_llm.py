"""Integration tests for Engine + LLM interaction."""

from unittest.mock import Mock, patch

import pytest
from chatnificent import Chatnificent
from chatnificent.engine import Synchronous
from chatnificent.llm import Anthropic, Echo, Gemini
from chatnificent.models import ASSISTANT_ROLE, USER_ROLE, ChatMessage


class TestEngineLLMIntegration:
    """Test Engine and LLM working together with real implementations."""

    def test_basic_conversation_flow_with_echo(self, test_app):
        """Test complete conversation flow with Echo LLM."""
        # Use the test_app which has Echo LLM
        result = test_app.engine.handle_message(
            user_input="Hello, Echo!", user_id="test_user", convo_id_from_url=None
        )

        # Verify response structure
        assert "messages" in result
        assert "input_value" in result
        assert result["input_value"] == ""  # Input should be cleared
        assert not result["submit_disabled"]

        # Get the actual conversation ID
        conversations = test_app.store.list_conversations("test_user")
        assert len(conversations) > 0
        convo_id = conversations[0]  # Get most recent

        # Verify conversation was saved
        conversation = test_app.store.load_conversation("test_user", convo_id)
        assert conversation is not None
        assert len(conversation.messages) == 2  # User + Assistant
        assert conversation.messages[0].content == "Hello, Echo!"
        assert "Echo LLM" in conversation.messages[1].content

    def test_multi_turn_conversation(self, test_app):
        """Test multiple back-and-forth messages."""
        # First message
        test_app.engine.handle_message(
            user_input="First message", user_id="test_user", convo_id_from_url=None
        )

        # Get the actual conversation ID
        conversations = test_app.store.list_conversations("test_user")
        convo_id = conversations[0]

        # Second message in same conversation
        test_app.engine.handle_message(
            user_input="Second message", user_id="test_user", convo_id_from_url=convo_id
        )

        # Verify conversation has all messages
        conversation = test_app.store.load_conversation("test_user", convo_id)
        assert len(conversation.messages) == 4  # 2 user + 2 assistant
        assert conversation.messages[0].content == "First message"
        assert conversation.messages[2].content == "Second message"

    def test_empty_tool_calls_list_finalization(self):
        """Test our critical bug fix: empty tool_calls list triggers finalization."""
        # Create app with mock LLM that returns empty tool calls list
        mock_llm = Mock()
        mock_llm.generate_response.return_value = Mock()
        mock_llm.extract_content.return_value = "Assistant response"
        mock_llm.parse_tool_calls.return_value = []  # Empty list, not None!

        app = Chatnificent(llm=mock_llm)

        app.engine.handle_message(
            user_input="Test message", user_id="test_user", convo_id_from_url=None
        )

        # Verify extract_content was called (finalization happened)
        mock_llm.extract_content.assert_called_once()

        # Verify message was added
        conversations = app.store.list_conversations("test_user")
        conversation = app.store.load_conversation("test_user", conversations[0])
        assert len(conversation.messages) == 2
        assert conversation.messages[1].content == "Assistant response"

    def test_none_tool_calls_finalization(self):
        """Test that None tool_calls also triggers finalization."""
        mock_llm = Mock()
        mock_llm.generate_response.return_value = Mock()
        mock_llm.extract_content.return_value = "Assistant response"
        mock_llm.parse_tool_calls.return_value = None  # None, not empty list

        app = Chatnificent(llm=mock_llm)

        app.engine.handle_message(
            user_input="Test message", user_id="test_user", convo_id_from_url=None
        )

        # Verify finalization happened
        mock_llm.extract_content.assert_called_once()
        conversations = app.store.list_conversations("test_user")
        conversation = app.store.load_conversation("test_user", conversations[0])
        assert conversation.messages[1].content == "Assistant response"

    def test_anthropic_list_content_handling(self):
        """Test that Anthropic's list content format is handled correctly."""
        # Mock Anthropic to return list content format
        mock_response = Mock()
        mock_response.content = [Mock(type="text", text="Hello from Anthropic")]

        mock_anthropic = Mock(spec=Anthropic)
        mock_anthropic.generate_response.return_value = mock_response
        mock_anthropic.extract_content.return_value = "Hello from Anthropic"
        mock_anthropic.parse_tool_calls.return_value = []

        app = Chatnificent(llm=mock_anthropic)

        app.engine.handle_message(
            user_input="Test", user_id="test_user", convo_id_from_url=None
        )

        # Verify the text was extracted correctly
        mock_anthropic.extract_content.assert_called_with(mock_response)

        # Verify message content is a string, not a list
        conversations = app.store.list_conversations("test_user")
        conversation = app.store.load_conversation("test_user", conversations[0])
        assert isinstance(conversation.messages[1].content, str)
        assert conversation.messages[1].content == "Hello from Anthropic"

    def test_gemini_parts_content_handling(self):
        """Test that Gemini's parts content format is handled correctly."""
        # Mock Gemini to return parts format
        mock_response = Mock()
        mock_response.text = "Hello from Gemini"

        mock_gemini = Mock(spec=Gemini)
        mock_gemini.generate_response.return_value = mock_response
        mock_gemini.extract_content.return_value = "Hello from Gemini"
        mock_gemini.parse_tool_calls.return_value = []

        app = Chatnificent(llm=mock_gemini)

        app.engine.handle_message(
            user_input="Test", user_id="test_user", convo_id_from_url=None
        )

        # Verify the text was extracted correctly
        mock_gemini.extract_content.assert_called_with(mock_response)

        # Verify message content is a string
        conversations = app.store.list_conversations("test_user")
        conversation = app.store.load_conversation("test_user", conversations[0])
        assert isinstance(conversation.messages[1].content, str)
        assert conversation.messages[1].content == "Hello from Gemini"

    def test_error_handling_in_llm_call(self, test_app):
        """Test error handling when LLM call fails."""
        # Mock the LLM to raise an error
        with patch.object(
            test_app.llm, "generate_response", side_effect=Exception("LLM Error")
        ):
            result = test_app.engine.handle_message(
                user_input="Cause an error", user_id="test_user", convo_id_from_url=None
            )

            # Should return error response
            assert "messages" in result
            assert not result["submit_disabled"]

            # Error should be saved in conversation
            conversations = test_app.store.list_conversations("test_user")
            conversation = test_app.store.load_conversation(
                "test_user", conversations[0]
            )
            assert len(conversation.messages) == 2
            assert "error" in conversation.messages[1].content.lower()

    def test_llm_response_saved_to_store(self):
        """Test that raw LLM responses are saved if store supports it."""
        mock_llm = Mock()
        mock_response = {"model": "test", "content": "response"}
        mock_llm.generate_response.return_value = mock_response
        mock_llm.extract_content.return_value = "response"
        mock_llm.parse_tool_calls.return_value = []

        mock_store = Mock()
        mock_store.get_next_conversation_id.return_value = "001"
        mock_store.load_conversation.return_value = None
        mock_store.save_conversation.return_value = None
        # Store supports raw response saving
        mock_store.save_raw_api_response = Mock()

        app = Chatnificent(llm=mock_llm, store=mock_store)

        app.engine.handle_message(
            user_input="Test", user_id="test_user", convo_id_from_url=None
        )

        # Verify raw response was saved
        mock_store.save_raw_api_response.assert_called()


class TestEngineHooksIntegration:
    """Test that engine hooks work correctly with real components."""

    def test_hooks_called_with_real_components(self):
        """Test hooks are called in correct order with real LLM."""

        class TrackedEngine(Synchronous):
            def __init__(self, app=None):
                super().__init__(app)
                self.call_log = []

            def _before_llm_call(self, conversation):
                self.call_log.append(f"before_llm:{len(conversation.messages)}")

            def _after_llm_call(self, response):
                _ = response  # Hook receives response but we only track the call
                self.call_log.append("after_llm")

            def _before_save(self, conversation):
                self.call_log.append(f"before_save:{len(conversation.messages)}")

        engine = TrackedEngine()
        app = Chatnificent(llm=Echo(), engine=engine)

        app.engine.handle_message(
            user_input="Test", user_id="test_user", convo_id_from_url=None
        )

        # Verify hooks were called in order with correct state
        assert engine.call_log == [
            "before_llm:1",  # User message added
            "after_llm",
            "before_save:2",  # Assistant message added
        ]

    def test_hook_can_modify_conversation(self):
        """Test that hooks can modify the conversation."""

        class ModifyingEngine(Synchronous):
            def _before_llm_call(self, conversation):
                # Add a system message
                conversation.messages.insert(
                    0, ChatMessage(role="system", content="Always be helpful")
                )

        engine = ModifyingEngine()
        app = Chatnificent(llm=Echo(), engine=engine)

        app.engine.handle_message(
            user_input="Test", user_id="test_user", convo_id_from_url=None
        )

        # Verify system message was added
        conversations = app.store.list_conversations("test_user")
        conversation = app.store.load_conversation("test_user", conversations[0])
        assert len(conversation.messages) == 3  # System + User + Assistant
        assert conversation.messages[0].role == "system"
        assert conversation.messages[0].content == "Always be helpful"


class TestEngineWithRealProviders:
    """Test engine with different real LLM provider implementations."""

    @pytest.mark.parametrize(
        "llm_class,expected_content",
        [
            (Echo, "Echo LLM"),
            # We could add more real providers here if we want to test them
        ],
    )
    def test_different_llm_providers(self, llm_class, expected_content):
        """Test engine works with different LLM providers."""
        llm = llm_class()
        app = Chatnificent(llm=llm)

        app.engine.handle_message(
            user_input="Hello", user_id="test_user", convo_id_from_url=None
        )

        conversations = app.store.list_conversations("test_user")
        conversation = app.store.load_conversation("test_user", conversations[0])
        assert len(conversation.messages) == 2
        assert expected_content in conversation.messages[1].content
