"""
Tests for the Retrieval pillar implementations.

The Retrieval pillar handles knowledge retrieval for RAG (Retrieval-Augmented Generation).
It has the simplest interface with just one method, making it perfect for validating patterns.
"""

from typing import Optional

import pytest
from chatnificent.retrieval import NoRetrieval, Retrieval


class TestRetrievalContract:
    """Test that all Retrieval implementations follow the same contract."""

    @pytest.fixture(
        params=[
            ("NoRetrieval", lambda: NoRetrieval()),
        ]
    )
    def retrieval_implementation(self, request):
        """Parametrized fixture providing all Retrieval implementations."""
        impl_name, impl_factory = request.param
        return impl_name, impl_factory()

    def test_retrieval_implements_interface(self, retrieval_implementation):
        """Test that all implementations properly inherit from Retrieval base class."""
        impl_name, retrieval = retrieval_implementation

        # Should inherit from Retrieval ABC
        assert isinstance(retrieval, Retrieval)

        # Should implement required method
        assert hasattr(retrieval, "retrieve")
        assert callable(getattr(retrieval, "retrieve"))

    def test_retrieve_accepts_required_parameters(self, retrieval_implementation):
        """Test that retrieve accepts query, user_id, and convo_id parameters."""
        impl_name, retrieval = retrieval_implementation

        # Should accept all required parameters without error
        try:
            result = retrieval.retrieve("test query", "test_user", "test_convo")
            # Result can be None or string, both valid for different implementations
            assert result is None or isinstance(result, str)
        except Exception as e:
            # Some implementations might raise errors, but we'll document this
            pytest.fail(f"Unexpected exception from {impl_name}: {e}")

    def test_retrieve_return_type(self, retrieval_implementation):
        """Test that retrieve returns Optional[str]."""
        impl_name, retrieval = retrieval_implementation

        result = retrieval.retrieve("test", "user", "convo")

        # Should return None or string
        assert result is None or isinstance(result, str)

    def test_retrieve_consistency(self, retrieval_implementation):
        """Test that retrieve behaves consistently."""
        impl_name, retrieval = retrieval_implementation

        query = "consistent query"
        user_id = "consistent_user"
        convo_id = "consistent_convo"

        # Multiple calls with same parameters should be deterministic
        result1 = retrieval.retrieve(query, user_id, convo_id)
        result2 = retrieval.retrieve(query, user_id, convo_id)

        # For deterministic implementations, should be same
        # For NoRetrieval, both should be None
        assert result1 == result2


class TestNoRetrieval:
    """Test the NoRetrieval implementation specifically."""

    def test_retrieve_returns_none(self):
        """Test that NoRetrieval always returns None."""
        retrieval = NoRetrieval()

        # Test with basic parameters
        result = retrieval.retrieve("test query", "user123", "conv001")
        assert result is None

        # Test with different parameters
        result = retrieval.retrieve("different query", "other_user", "other_conv")
        assert result is None

    def test_retrieve_accepts_any_string_parameters(self):
        """Test that retrieve accepts any string values for its parameters."""
        retrieval = NoRetrieval()

        # Different query types
        queries = [
            "simple query",
            "How to use Python?",
            "complex query with special chars: @#$%^&*()",
            "",  # Empty query
            "very " * 1000 + "long query",  # Very long query
            "unicode query: 你好世界 🌍",
            "multiline\nquery\nwith\nbreaks",
        ]

        for query in queries:
            result = retrieval.retrieve(query, "user", "convo")
            assert result is None

    def test_retrieve_accepts_any_user_ids(self):
        """Test that retrieve accepts any user ID formats."""
        retrieval = NoRetrieval()

        user_ids = [
            "user123",
            "user-with-dashes",
            "user_with_underscores",
            "user.with.dots",
            "user@domain.com",
            "",  # Empty user ID
            "αβγ_unicode_user",
            "测试用户",
            "very_long_user_id_" * 100,
        ]

        for user_id in user_ids:
            result = retrieval.retrieve("test", user_id, "convo")
            assert result is None

    def test_retrieve_accepts_any_conversation_ids(self):
        """Test that retrieve accepts any conversation ID formats."""
        retrieval = NoRetrieval()

        convo_ids = [
            "conv001",
            "conversation-123",
            "conv_with_underscores",
            "conv.with.dots",
            "",  # Empty conversation ID
            "αβγ_unicode_conv",
            "对话123",
            "very_long_conversation_id_" * 50,
        ]

        for convo_id in convo_ids:
            result = retrieval.retrieve("test", "user", convo_id)
            assert result is None

    def test_retrieve_parameter_combinations(self):
        """Test various combinations of parameters."""
        retrieval = NoRetrieval()

        test_combinations = [
            # Normal usage
            ("What is Python?", "developer_user", "python_discussion"),
            # Empty parameters
            ("", "", ""),
            # Mixed languages
            ("Python编程", "用户123", "对话001"),
            # Special characters
            ("query@#$%", "user!@#", "conv&*()"),
            # Very long parameters
            ("x" * 1000, "y" * 500, "z" * 200),
            # Numbers as strings (converted by auth pillar)
            ("query123", "456", "789"),
        ]

        for query, user_id, convo_id in test_combinations:
            result = retrieval.retrieve(query, user_id, convo_id)
            assert result is None

    def test_multiple_instances_independent(self):
        """Test that multiple NoRetrieval instances behave independently."""
        retrieval1 = NoRetrieval()
        retrieval2 = NoRetrieval()

        # Both should return None
        assert retrieval1.retrieve("test", "user", "convo") is None
        assert retrieval2.retrieve("test", "user", "convo") is None

        # They should be separate instances
        assert retrieval1 is not retrieval2

    def test_retrieve_stability_across_calls(self):
        """Test that NoRetrieval behavior is stable across many calls."""
        retrieval = NoRetrieval()

        # Test many calls with same parameters
        for i in range(100):
            result = retrieval.retrieve(f"query_{i}", f"user_{i}", f"convo_{i}")
            assert result is None

    def test_retrieve_with_none_like_strings(self):
        """Test retrieve with string representations of None/null."""
        retrieval = NoRetrieval()

        none_like_strings = [
            "None",
            "null",
            "NULL",
            "nil",
            "undefined",
            "NaN",
        ]

        for none_string in none_like_strings:
            # Test as query
            result = retrieval.retrieve(none_string, "user", "convo")
            assert result is None

            # Test as user_id
            result = retrieval.retrieve("query", none_string, "convo")
            assert result is None

            # Test as convo_id
            result = retrieval.retrieve("query", "user", none_string)
            assert result is None


class TestRetrievalInterface:
    """Test the Retrieval abstract base class interface."""

    def test_retrieval_is_abstract(self):
        """Test that Retrieval cannot be instantiated directly."""
        with pytest.raises(TypeError) as exc_info:
            Retrieval()

        error_message = str(exc_info.value)
        assert "abstract" in error_message.lower()

    def test_retrieval_requires_retrieve_method(self):
        """Test that Retrieval subclasses must implement retrieve."""

        # Incomplete implementation missing retrieve
        class IncompleteRetrieval(Retrieval):
            pass

        # Should not be able to instantiate
        with pytest.raises(TypeError) as exc_info:
            IncompleteRetrieval()

        error_message = str(exc_info.value)
        assert "retrieve" in error_message

    def test_retrieval_subclass_with_implementation_works(self):
        """Test that complete Retrieval subclasses work correctly."""

        class CustomRetrieval(Retrieval):
            def retrieve(
                self, query: str, user_id: str, convo_id: str
            ) -> Optional[str]:
                return f"Retrieved for query='{query}', user='{user_id}', convo='{convo_id}'"

        # Should work fine
        retrieval = CustomRetrieval()
        assert isinstance(retrieval, Retrieval)

        result = retrieval.retrieve("test query", "user123", "conv001")
        expected = "Retrieved for query='test query', user='user123', convo='conv001'"
        assert result == expected

    def test_custom_retrieval_return_none(self):
        """Test that custom retrievals can return None."""

        class AlwaysNoneRetrieval(Retrieval):
            def retrieve(
                self, query: str, user_id: str, convo_id: str
            ) -> Optional[str]:
                return None

        retrieval = AlwaysNoneRetrieval()
        result = retrieval.retrieve("any", "query", "params")
        assert result is None

    def test_custom_retrieval_return_strings(self):
        """Test that custom retrievals can return strings."""

        class SimpleRetrieval(Retrieval):
            def retrieve(
                self, query: str, user_id: str, convo_id: str
            ) -> Optional[str]:
                if query == "hello":
                    return "Hello response"
                elif query == "python":
                    return "Python is a programming language"
                else:
                    return None

        retrieval = SimpleRetrieval()

        assert retrieval.retrieve("hello", "u", "c") == "Hello response"
        assert (
            retrieval.retrieve("python", "u", "c") == "Python is a programming language"
        )
        assert retrieval.retrieve("unknown", "u", "c") is None


class TestRetrievalIntegration:
    """Test Retrieval pillar integration patterns."""

    def test_retrieval_in_llm_context(self):
        """Test retrieval usage in LLM context."""
        retrieval = NoRetrieval()

        # Simulate LLM workflow asking for context
        user_query = "What is machine learning?"
        user_id = "student_123"
        conversation_id = "ml_discussion_001"

        # Retrieve relevant context
        context = retrieval.retrieve(user_query, user_id, conversation_id)

        # NoRetrieval returns None, so no context available
        assert context is None

        # LLM would proceed without additional context
        if context:
            # Would use context to enhance response
            enhanced_prompt = f"Context: {context}\n\nQuery: {user_query}"
        else:
            # Would use original query only
            enhanced_prompt = user_query

        assert enhanced_prompt == user_query  # NoRetrieval case

    def test_retrieval_with_conversation_history(self):
        """Test retrieval with conversation context."""
        retrieval = NoRetrieval()

        # Simulate multi-turn conversation
        conversation_turns = [
            ("What is Python?", "user123", "python_chat"),
            ("How do I install it?", "user123", "python_chat"),
            ("What about virtual environments?", "user123", "python_chat"),
        ]

        retrieved_contexts = []
        for query, user_id, convo_id in conversation_turns:
            context = retrieval.retrieve(query, user_id, convo_id)
            retrieved_contexts.append(context)

        # NoRetrieval should return None for all
        assert all(context is None for context in retrieved_contexts)

    def test_retrieval_error_handling(self):
        """Test retrieval error handling patterns."""
        retrieval = NoRetrieval()

        # Test with potentially problematic inputs
        problematic_inputs = [
            # Very long strings
            ("x" * 10000, "user", "convo"),
            # Special characters that might break parsers
            ("query\x00\x01\x02", "user", "convo"),
            # SQL-injection-like strings (should be safe)
            ("'; DROP TABLE users; --", "user", "convo"),
            # JSON-like strings
            ('{"query": "test"}', "user", "convo"),
            # HTML/XML-like strings
            ("<script>alert('test')</script>", "user", "convo"),
        ]

        for query, user_id, convo_id in problematic_inputs:
            try:
                result = retrieval.retrieve(query, user_id, convo_id)
                # NoRetrieval should handle gracefully
                assert result is None
            except Exception as e:
                pytest.fail(f"NoRetrieval raised unexpected exception: {e}")

    def test_retrieval_with_auth_integration(self):
        """Test retrieval working with auth pillar patterns."""
        retrieval = NoRetrieval()

        # Simulate auth providing user context
        from chatnificent.auth import SingleUser

        auth = SingleUser(user_id="authenticated_user")
        current_user = auth.get_current_user_id()

        # Use auth-provided user ID in retrieval
        result = retrieval.retrieve("test query", current_user, "convo001")
        assert result is None

    def test_retrieval_interface_stability(self):
        """Test that retrieval interface remains stable."""
        retrieval = NoRetrieval()

        # Interface should remain consistent across calls
        for i in range(10):
            result = retrieval.retrieve(f"query_{i}", f"user_{i}", f"convo_{i}")
            assert result is None

        # Method should exist and be callable
        assert callable(retrieval.retrieve)

        # Method signature should be as expected
        import inspect

        retrieve_sig = inspect.signature(retrieval.retrieve)
        params = list(retrieve_sig.parameters.keys())

        expected_params = ["query", "user_id", "convo_id"]
        for param in expected_params:
            assert param in params, (
                f"Expected parameter '{param}' not found in signature"
            )
