"""
Tests for the Auth pillar implementations.

The Auth pillar is responsible for identifying the current user. It's the simplest
pillar with only one method, making it perfect for establishing our testing patterns.
"""

import pytest
from chatnificent.auth import Auth, SingleUser


class TestAuthContract:
    """Test that all Auth implementations follow the same contract."""

    @pytest.fixture(
        params=[
            ("SingleUser", lambda: SingleUser()),
            ("SingleUserCustom", lambda: SingleUser(user_id="custom_user")),
        ]
    )
    def auth_implementation(self, request):
        """Parametrized fixture providing all Auth implementations."""
        impl_name, impl_factory = request.param
        return impl_name, impl_factory()

    def test_auth_implements_interface(self, auth_implementation):
        """Test that all implementations properly inherit from Auth base class."""
        impl_name, auth = auth_implementation

        # Should inherit from Auth ABC
        assert isinstance(auth, Auth)

        # Should implement required method
        assert hasattr(auth, "get_current_user_id")
        assert callable(getattr(auth, "get_current_user_id"))

    def test_get_current_user_id_returns_string(self, auth_implementation):
        """Test that get_current_user_id returns a string (per interface contract)."""
        impl_name, auth = auth_implementation

        # All Auth implementations must return strings per interface
        user_id = auth.get_current_user_id()
        assert isinstance(user_id, str)
        assert len(user_id) > 0  # Should not be empty

        # Test with keyword arguments (should work)
        user_id_with_kwargs = auth.get_current_user_id(pathname="/test")
        assert isinstance(user_id_with_kwargs, str)
        assert len(user_id_with_kwargs) > 0

    def test_get_current_user_id_consistency(self, auth_implementation):
        """Test that get_current_user_id returns consistent results."""
        impl_name, auth = auth_implementation

        # Should return same value on multiple calls
        first_call = auth.get_current_user_id()
        second_call = auth.get_current_user_id()
        third_call = auth.get_current_user_id(pathname="/different")

        assert first_call == second_call
        # For current implementations, should be same regardless of kwargs
        assert first_call == third_call

    def test_get_current_user_id_accepts_kwargs(self, auth_implementation):
        """Test that get_current_user_id accepts arbitrary keyword arguments."""
        impl_name, auth = auth_implementation

        # Should accept various kwargs without error
        test_kwargs = [
            {},
            {"pathname": "/user/123"},
            {"pathname": "/chat", "search": "?param=value"},
            {"custom_arg": "value", "another_arg": 42},
            {"pathname": None, "user_data": {"id": "test"}},
        ]

        for kwargs in test_kwargs:
            user_id = auth.get_current_user_id(**kwargs)
            assert isinstance(user_id, str)
            assert len(user_id) > 0


class TestSingleUser:
    """Test the SingleUser implementation specifically."""

    def test_default_user_id(self):
        """Test SingleUser with default user ID."""
        auth = SingleUser()
        assert auth.get_current_user_id() == "chat"

        # Should be consistent
        assert auth.get_current_user_id() == "chat"
        assert auth.get_current_user_id(pathname="/test") == "chat"

    def test_custom_user_id(self):
        """Test SingleUser with custom user ID."""
        custom_id = "my_custom_user"
        auth = SingleUser(user_id=custom_id)

        assert auth.get_current_user_id() == custom_id
        assert auth.get_current_user_id(pathname="/test") == custom_id

    def test_empty_user_id_allowed(self):
        """Test that empty user ID is allowed (documents current behavior)."""
        auth = SingleUser(user_id="")
        assert auth.get_current_user_id() == ""

    def test_special_character_user_ids(self):
        """Test user IDs with special characters."""
        special_ids = [
            "user-123",
            "user_with_underscores",
            "user.with.dots",
            "user@domain.com",
            "αβγ",  # Greek
            "测试用户",  # Chinese
            "🔥user",  # Emoji
        ]

        for user_id in special_ids:
            auth = SingleUser(user_id=user_id)
            assert auth.get_current_user_id() == user_id

    def test_user_id_string_conversion(self):
        """Test that SingleUser converts all inputs to strings."""
        # Numbers should be converted to strings
        auth = SingleUser(user_id=123)
        result = auth.get_current_user_id()
        assert isinstance(result, str)
        assert result == "123"

        # None should be converted to string
        auth = SingleUser(user_id=None)
        result = auth.get_current_user_id()
        assert isinstance(result, str)
        assert result == "None"

        # Strings should remain strings
        auth = SingleUser(user_id="test")
        result = auth.get_current_user_id()
        assert isinstance(result, str)
        assert result == "test"

        # Booleans should be converted
        auth = SingleUser(user_id=True)
        result = auth.get_current_user_id()
        assert isinstance(result, str)
        assert result == "True"

    def test_kwargs_ignored_consistently(self):
        """Test that all kwargs are ignored consistently."""
        auth = SingleUser(user_id="test_user")
        base_result = auth.get_current_user_id()

        # All these should return the same result
        kwargs_variations = [
            {"pathname": "/user/different_user"},
            {"user_hint": "should_be_ignored"},
            {"auth_token": "bearer xyz"},
            {"request_data": {"user_id": "hacker_attempt"}},
        ]

        for kwargs in kwargs_variations:
            result = auth.get_current_user_id(**kwargs)
            assert result == base_result == "test_user"

    def test_immutable_behavior(self):
        """Test that the auth instance behaves consistently."""
        auth = SingleUser(user_id="immutable_user")

        # User ID should not change
        original_id = auth.get_current_user_id()

        # Multiple calls with different kwargs
        for i in range(10):
            result = auth.get_current_user_id(call_number=i)
            assert result == original_id

        # Final check
        assert auth.get_current_user_id() == original_id


class TestAuthInterface:
    """Test the Auth abstract base class interface."""

    def test_auth_is_abstract(self):
        """Test that Auth cannot be instantiated directly."""
        with pytest.raises(TypeError) as exc_info:
            Auth()

        # Should mention that it can't instantiate abstract class
        error_message = str(exc_info.value)
        assert "abstract" in error_message.lower()

    def test_auth_requires_get_current_user_id(self):
        """Test that Auth subclasses must implement get_current_user_id."""

        # Create a class that doesn't implement the required method
        class IncompleteAuth(Auth):
            pass

        # Should not be able to instantiate
        with pytest.raises(TypeError) as exc_info:
            IncompleteAuth()

        error_message = str(exc_info.value)
        assert "get_current_user_id" in error_message

    def test_auth_subclass_with_implementation_works(self):
        """Test that proper Auth subclasses work correctly."""

        class CustomAuth(Auth):
            def get_current_user_id(self, **kwargs) -> str:
                return "custom_implementation"

        # Should be able to instantiate and use
        auth = CustomAuth()
        assert isinstance(auth, Auth)
        assert auth.get_current_user_id() == "custom_implementation"


class TestAuthIntegration:
    """Test Auth pillar integration patterns."""

    def test_auth_in_callback_simulation(self):
        """Test auth usage patterns similar to callback usage."""
        auth = SingleUser(user_id="callback_user")

        # Simulate how callbacks use auth
        simulated_callback_calls = [
            # New chat creation
            auth.get_current_user_id(pathname="/callback_user/new"),
            # Existing conversation loading
            auth.get_current_user_id(pathname="/callback_user/001"),
            # Conversation switching
            auth.get_current_user_id(pathname="/callback_user/002"),
        ]

        # All should return the same user ID
        for call_result in simulated_callback_calls:
            assert call_result == "callback_user"

    def test_auth_with_url_pillar_integration(self):
        """Test auth working with URL parsing patterns."""
        auth = SingleUser(user_id="url_test_user")

        # Simulate URL pillar providing different path info
        url_scenarios = [
            {"pathname": "/url_test_user/001", "search": ""},
            {"pathname": "/different_user/002", "search": "?param=value"},
            {"pathname": "/", "search": "?user=someone_else"},
        ]

        for url_data in url_scenarios:
            # Auth should return consistent user regardless of URL data
            result = auth.get_current_user_id(**url_data)
            assert result == "url_test_user"

    def test_multiple_auth_instances_independent(self):
        """Test that multiple Auth instances work independently."""
        auth1 = SingleUser(user_id="user_one")
        auth2 = SingleUser(user_id="user_two")

        # Should return different values
        assert auth1.get_current_user_id() == "user_one"
        assert auth2.get_current_user_id() == "user_two"

        # Should remain independent
        assert auth1.get_current_user_id(test="data") == "user_one"
        assert auth2.get_current_user_id(test="data") == "user_two"
