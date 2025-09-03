"""
Tests for the URL pillar implementations.

The URL pillar handles parsing application state from URLs and building URLs
from application state. It has two implementations with different routing schemes.
"""

from urllib.parse import quote

import pytest
from chatnificent.url import URL, PathBased, QueryParams, URLParts


class TestURLContract:
    """Test that all URL implementations follow the same contract."""

    @pytest.fixture(
        params=[
            ("PathBased", lambda: PathBased()),
            ("QueryParams", lambda: QueryParams()),
        ]
    )
    def url_implementation(self, request):
        """Parametrized fixture providing all URL implementations."""
        impl_name, impl_factory = request.param
        return impl_name, impl_factory()

    def test_url_implements_interface(self, url_implementation):
        """Test that all implementations properly inherit from URL base class."""
        impl_name, url = url_implementation

        assert isinstance(url, URL)

        required_methods = ["parse", "build_conversation_path", "build_new_chat_path"]
        for method in required_methods:
            assert hasattr(url, method)
            assert callable(getattr(url, method))

    def test_parse_returns_url_parts(self, url_implementation):
        """Test that parse always returns URLParts objects."""
        impl_name, url = url_implementation

        result = url.parse("/")
        assert isinstance(result, URLParts)

        test_paths = ["/user123", "/user123/conv001", "/user123/new"]
        for path in test_paths:
            result = url.parse(path)
            assert isinstance(result, URLParts)
            assert hasattr(result, "user_id")
            assert hasattr(result, "convo_id")

    def test_build_methods_return_strings(self, url_implementation):
        """Test that build methods always return strings."""
        impl_name, url = url_implementation

        # Test conversation path building
        conv_path = url.build_conversation_path("user123", "conv001")
        assert isinstance(conv_path, str)
        assert len(conv_path) > 0

        # Test new chat path building
        new_path = url.build_new_chat_path("user123")
        assert isinstance(new_path, str)
        assert len(new_path) > 0

    def test_round_trip_consistency(self, url_implementation):
        """Test that building and parsing are consistent."""
        impl_name, url = url_implementation

        # Test conversation round-trip
        original_user = "test_user"
        original_convo = "conv_001"

        built_path = url.build_conversation_path(original_user, original_convo)

        # Parse correctly based on implementation type
        if "?" in built_path:
            path_part, query_part = built_path.split("?", 1)
            parsed = url.parse(path_part, "?" + query_part)
        else:
            parsed = url.parse(built_path)

        assert parsed.user_id == original_user
        assert parsed.convo_id == original_convo

        # Test new chat round-trip
        new_path = url.build_new_chat_path(original_user)

        if "?" in new_path:
            path_part, query_part = new_path.split("?", 1)
            parsed_new = url.parse(path_part, "?" + query_part)
        else:
            parsed_new = url.parse(new_path)

        assert parsed_new.user_id == original_user
        assert parsed_new.convo_id is None  # New chats have no convo_id

    def test_user_id_extraction_consistency(self, url_implementation):
        """Test that user_id is extracted consistently across different paths."""
        impl_name, url = url_implementation

        user_id = "consistent_user"

        # Build different paths for same user
        conv_path = url.build_conversation_path(user_id, "conv1")
        new_path = url.build_new_chat_path(user_id)

        # Parse both paths correctly
        if "?" in conv_path:
            path_part, query_part = conv_path.split("?", 1)
            conv_parsed = url.parse(path_part, "?" + query_part)
        else:
            conv_parsed = url.parse(conv_path)

        if "?" in new_path:
            path_part, query_part = new_path.split("?", 1)
            new_parsed = url.parse(path_part, "?" + query_part)
        else:
            new_parsed = url.parse(new_path)

        # User ID should be consistent
        assert conv_parsed.user_id == user_id
        assert new_parsed.user_id == user_id


class TestURLParts:
    """Test the URLParts data model."""

    def test_url_parts_creation(self):
        """Test URLParts object creation and attributes."""
        # Test with both values
        parts = URLParts(user_id="user123", convo_id="conv001")
        assert parts.user_id == "user123"
        assert parts.convo_id == "conv001"

        # Test with only user_id
        parts = URLParts(user_id="user123", convo_id=None)
        assert parts.user_id == "user123"
        assert parts.convo_id is None

        # Test with both None
        parts = URLParts(user_id=None, convo_id=None)
        assert parts.user_id is None
        assert parts.convo_id is None

    def test_url_parts_serialization(self):
        """Test URLParts serialization behavior."""
        parts = URLParts(user_id="test_user", convo_id="test_convo")

        # Should be serializable to dict
        serialized = parts.model_dump()
        expected = {"user_id": "test_user", "convo_id": "test_convo"}
        assert serialized == expected

        # Should be deserializable from dict
        reconstructed = URLParts(**serialized)
        assert reconstructed.user_id == parts.user_id
        assert reconstructed.convo_id == parts.convo_id


class TestPathBased:
    """Test the PathBased URL implementation specifically."""

    def test_basic_path_parsing(self):
        """Test basic path parsing patterns."""
        url = PathBased()

        # Root path
        result = url.parse("/")
        assert result.user_id is None
        assert result.convo_id is None

        # User only
        result = url.parse("/user123")
        assert result.user_id == "user123"
        assert result.convo_id is None

        # User and conversation
        result = url.parse("/user123/conv001")
        assert result.user_id == "user123"
        assert result.convo_id == "conv001"

        # User with "new" keyword
        result = url.parse("/user123/new")
        assert result.user_id == "user123"
        assert result.convo_id is None

    def test_path_parsing_edge_cases(self):
        """Test edge cases in path parsing."""
        url = PathBased()

        # Trailing slashes
        result = url.parse("/user123/")
        assert result.user_id == "user123"
        assert result.convo_id is None

        # Multiple slashes
        result = url.parse("//user123//conv001//")
        assert result.user_id == "user123"  # Should handle gracefully

        # Empty segments
        result = url.parse("/user123//")
        assert result.user_id == "user123"
        assert result.convo_id is None

    def test_path_building(self):
        """Test path building methods."""
        url = PathBased()

        # Conversation path
        path = url.build_conversation_path("user123", "conv001")
        assert path == "/user123/conv001"

        # New chat path
        path = url.build_new_chat_path("user123")
        assert path == "/user123/new"

    def test_special_characters_in_paths(self):
        """Test handling of special characters."""
        url = PathBased()

        # Special characters in user ID
        special_user_ids = [
            "user-123",
            "user_with_underscores",
            "user.with.dots",
            "αβγ",  # Greek
            "测试",  # Chinese
        ]

        for user_id in special_user_ids:
            # Build and parse back
            path = url.build_conversation_path(user_id, "conv001")
            parsed = url.parse(path)
            assert parsed.user_id == user_id
            assert parsed.convo_id == "conv001"

    def test_case_sensitivity(self):
        """Test case sensitivity in path parsing."""
        url = PathBased()

        # "new" keyword should be case insensitive
        test_cases = ["new", "NEW", "New", "nEw"]
        for case in test_cases:
            result = url.parse(f"/user123/{case}")
            assert result.user_id == "user123"
            assert result.convo_id is None  # Should be treated as "new"


class TestQueryParams:
    """Test the QueryParams URL implementation specifically."""

    def test_basic_query_parsing(self):
        """Test basic query parameter parsing."""
        url = QueryParams()

        # No query string
        result = url.parse("/chat")
        assert result.user_id is None
        assert result.convo_id is None

        # User only
        result = url.parse("/chat", "?user=user123")
        assert result.user_id == "user123"
        assert result.convo_id is None

        # User and conversation
        result = url.parse("/chat", "?user=user123&convo=conv001")
        assert result.user_id == "user123"
        assert result.convo_id == "conv001"

    def test_query_parameter_variations(self):
        """Test different query parameter formats."""
        url = QueryParams()

        # Different parameter orders
        result = url.parse("/chat", "?convo=conv001&user=user123")
        assert result.user_id == "user123"
        assert result.convo_id == "conv001"

        # URL-encoded values
        result = url.parse("/chat", "?user=user%20with%20spaces&convo=conv%20001")
        assert result.user_id == "user with spaces"
        assert result.convo_id == "conv 001"

        # Additional parameters (should be ignored)
        result = url.parse("/chat", "?user=user123&convo=conv001&extra=ignored")
        assert result.user_id == "user123"
        assert result.convo_id == "conv001"

    def test_query_building(self):
        """Test query parameter building."""
        url = QueryParams()

        # Conversation path
        path = url.build_conversation_path("user123", "conv001")
        assert path == "/chat?user=user123&convo=conv001"

        # New chat path
        path = url.build_new_chat_path("user123")
        assert path == "/chat?user=user123"

    def test_query_special_characters(self):
        """Test URL encoding of special characters."""
        url = QueryParams()

        # Characters that need URL encoding
        user_id = "user with spaces & symbols"
        convo_id = "conv/with/slashes"

        path = url.build_conversation_path(user_id, convo_id)
        parsed = url.parse("/chat", path.split("?", 1)[1])

        # Should round-trip correctly
        assert parsed.user_id == user_id
        assert parsed.convo_id == convo_id

    def test_empty_query_values(self):
        """Test handling of empty query parameter values."""
        url = QueryParams()

        # Empty user parameter (parse_qs drops empty values, so becomes None)
        result = url.parse("/chat", "?user=&convo=conv001")
        assert result.user_id is None  # parse_qs drops empty values
        assert result.convo_id == "conv001"

        # Missing parameters
        result = url.parse("/chat", "?other=value")
        assert result.user_id is None
        assert result.convo_id is None


class TestURLInterface:
    """Test the URL abstract base class interface."""

    def test_url_is_abstract(self):
        """Test that URL cannot be instantiated directly."""
        with pytest.raises(TypeError) as exc_info:
            URL()

        error_message = str(exc_info.value)
        assert "abstract" in error_message.lower()

    def test_url_requires_all_methods(self):
        """Test that URL subclasses must implement all required methods."""

        # Incomplete implementation missing build_new_chat_path
        class IncompleteURL(URL):
            def parse(self, pathname, search=None):
                return URLParts(user_id=None, convo_id=None)

            def build_conversation_path(self, user_id, convo_id):
                return f"/{user_id}/{convo_id}"

        # Should not be able to instantiate
        with pytest.raises(TypeError) as exc_info:
            IncompleteURL()

        error_message = str(exc_info.value)
        assert "build_new_chat_path" in error_message

    def test_url_subclass_with_all_methods_works(self):
        """Test that complete URL subclasses work correctly."""

        class CustomURL(URL):
            def parse(self, pathname, search=None):
                return URLParts(user_id="custom", convo_id="parsed")

            def build_conversation_path(self, user_id, convo_id):
                return f"/custom/{user_id}/{convo_id}"

            def build_new_chat_path(self, user_id):
                return f"/custom/{user_id}/new"

        # Should work fine
        url = CustomURL()
        assert isinstance(url, URL)

        result = url.parse("/anything")
        assert result.user_id == "custom"
        assert result.convo_id == "parsed"


class TestURLIntegration:
    """Test URL pillar integration patterns."""

    def test_url_with_auth_integration(self):
        """Test URL parsing with auth fallback patterns."""
        url = PathBased()

        # Simulate callback pattern: URL provides user, auth provides fallback
        url_parts = url.parse("/explicit_user/conv001")
        auth_fallback_user = "auth_provided_user"

        # URL should provide user_id when present
        effective_user = url_parts.user_id or auth_fallback_user
        assert effective_user == "explicit_user"

        # URL should fall back to auth when user not in URL
        url_parts_no_user = url.parse("/")
        effective_user_fallback = url_parts_no_user.user_id or auth_fallback_user
        assert effective_user_fallback == "auth_provided_user"

    def test_url_conversation_loading_patterns(self):
        """Test URL patterns used for conversation loading."""
        for impl_name, url in [
            ("PathBased", PathBased()),
            ("QueryParams", QueryParams()),
        ]:
            # New conversation pattern
            new_path = url.build_new_chat_path("user123")
            parsed_new = url.parse(
                new_path, new_path.split("?", 1)[1] if "?" in new_path else None
            )

            assert parsed_new.user_id == "user123"
            assert parsed_new.convo_id is None  # Signals new conversation

            # Existing conversation pattern
            conv_path = url.build_conversation_path("user123", "conv001")
            parsed_conv = url.parse(
                conv_path, conv_path.split("?", 1)[1] if "?" in conv_path else None
            )

            assert parsed_conv.user_id == "user123"
            assert parsed_conv.convo_id == "conv001"  # Signals load existing

    def test_url_building_for_redirects(self):
        """Test URL building patterns used in callback redirects."""
        for impl_name, url in [
            ("PathBased", PathBased()),
            ("QueryParams", QueryParams()),
        ]:
            user_id = "redirect_user"
            convo_id = "new_conversation_123"

            # Build redirect URL (used when new conversation is created)
            redirect_path = url.build_conversation_path(user_id, convo_id)

            # Verify it parses back correctly
            if "?" in redirect_path:
                path_part, query_part = redirect_path.split("?", 1)
                parsed = url.parse(path_part, query_part)
            else:
                parsed = url.parse(redirect_path)

            assert parsed.user_id == user_id
            assert parsed.convo_id == convo_id

    def test_multiple_url_implementations_coexist(self):
        """Test that different URL implementations work independently."""
        path_url = PathBased()
        query_url = QueryParams()

        user_id = "multi_test"
        convo_id = "conv_001"

        # Build with different implementations
        path_built = path_url.build_conversation_path(user_id, convo_id)
        query_built = query_url.build_conversation_path(user_id, convo_id)

        # Should produce different formats
        assert path_built != query_built
        assert "/multi_test/conv_001" in path_built
        assert "user=multi_test" in query_built
        assert "convo=conv_001" in query_built

        # But both should parse their own format correctly
        path_parsed = path_url.parse(path_built)
        query_parsed = query_url.parse("/chat", query_built.split("?", 1)[1])

        assert path_parsed.user_id == user_id
        assert path_parsed.convo_id == convo_id
        assert query_parsed.user_id == user_id
        assert query_parsed.convo_id == convo_id
