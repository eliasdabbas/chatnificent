"""
Core pytest configuration and fixtures for Chatnificent testing.

This module provides shared test fixtures, configuration, and utilities
that support the pillar-based testing architecture.
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock

import pytest
from chatnificent.models import ASSISTANT_ROLE, USER_ROLE, ChatMessage, Conversation

# ===== TEST DATA FIXTURES =====


@pytest.fixture
def sample_messages() -> List[ChatMessage]:
    """Sample chat messages for testing."""
    return [
        ChatMessage(role=USER_ROLE, content="Hello, how are you?"),
        ChatMessage(
            role=ASSISTANT_ROLE,
            content="I'm doing well, thank you! How can I help you today?",
        ),
        ChatMessage(role=USER_ROLE, content="Can you explain quantum computing?"),
        ChatMessage(
            role=ASSISTANT_ROLE,
            content="Quantum computing uses quantum mechanics principles...",
        ),
    ]


@pytest.fixture
def sample_conversation(sample_messages) -> Conversation:
    """Sample conversation for testing."""
    return Conversation(id="001", messages=sample_messages)


@pytest.fixture
def empty_conversation() -> Conversation:
    """Empty conversation for testing edge cases."""
    return Conversation(id="empty", messages=[])


@pytest.fixture
def conversation_dict(sample_messages) -> List[Dict]:
    """Sample conversation as dict format (for LLM providers)."""
    return [msg.model_dump() for msg in sample_messages]


@pytest.fixture
def mock_llm_response() -> Dict:
    """Mock LLM response object."""
    return {
        "content": "This is a mock response from the LLM",
        "model": "test-model-v1",
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
    }


# ===== DIRECTORY FIXTURES =====


@pytest.fixture
def temp_dir():
    """Temporary directory for file-based tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def conversations_dir(temp_dir):
    """Directory structure for conversation storage."""
    conv_dir = temp_dir / "conversations"
    conv_dir.mkdir()
    return conv_dir


# ===== MOCK FIXTURES =====


@pytest.fixture
def mock_llm():
    """Mock LLM provider for testing."""
    mock = MagicMock()
    mock.generate_response.return_value = {
        "choices": [{"message": {"content": "Mock LLM response"}}]
    }
    mock.extract_content.return_value = "Mock LLM response"
    return mock


@pytest.fixture
def mock_auth():
    """Mock auth provider for testing."""
    mock = MagicMock()
    mock.get_current_user_id.return_value = "test_user"
    return mock


@pytest.fixture
def mock_store():
    """Mock store provider for testing."""
    mock = MagicMock()
    mock.get_next_conversation_id.return_value = "001"
    return mock


@pytest.fixture
def mock_layout():
    """Mock layout provider for testing."""
    mock = MagicMock()
    mock.build_messages.return_value = []
    mock.get_external_stylesheets.return_value = []
    mock.get_external_scripts.return_value = []
    return mock


@pytest.fixture
def mock_url():
    """Mock URL provider for testing."""
    mock = MagicMock()
    return mock


# ===== PILLAR IMPLEMENTATION FIXTURES =====


@pytest.fixture
def all_store_implementations(temp_dir):
    """All store implementations for contract testing."""
    from chatnificent import store

    return [
        ("InMemory", store.InMemory()),
        ("File", store.File(str(temp_dir / "file_store"))),
        ("SQLite", store.SQLite(str(temp_dir / "test.db"))),
    ]


@pytest.fixture
def all_auth_implementations():
    """All auth implementations for contract testing."""
    from chatnificent import auth

    return [
        ("SingleUser", auth.SingleUser()),
        ("SingleUserCustom", auth.SingleUser(user_id="custom_user")),
    ]


@pytest.fixture
def all_url_implementations():
    """All URL implementations for contract testing."""
    from chatnificent import url

    return [
        ("PathBased", url.PathBased()),
        ("QueryParams", url.QueryParams()),
    ]


# ===== TEST UTILITIES =====


def create_test_conversation_file(
    directory: Path, user_id: str, convo_id: str, messages: List[ChatMessage]
) -> None:
    """Helper to create test conversation files."""
    user_dir = directory / user_id
    user_dir.mkdir(exist_ok=True)
    convo_dir = user_dir / convo_id
    convo_dir.mkdir(exist_ok=True)

    messages_file = convo_dir / "messages.json"
    messages_data = [msg.model_dump() for msg in messages]

    with open(messages_file, "w") as f:
        json.dump(messages_data, f, indent=2)


@pytest.fixture
def create_conversation_file():
    """Helper function to create conversation files in tests."""
    return create_test_conversation_file


# ===== APP FIXTURES =====


@pytest.fixture
def test_app():
    """
    Provides a Chatnificent app instance with simple, predictable pillars.

    This fixture is ideal for integration tests where we need a running app
    but want to avoid external dependencies like filesystems or actual LLM APIs.
    """
    from chatnificent import Chatnificent
    from chatnificent.auth import SingleUser
    from chatnificent.llm import Echo
    from chatnificent.store import InMemory

    app = Chatnificent(
        store=InMemory(),
        llm=Echo(),
        auth=SingleUser(),
    )
    return app


# ===== DASH TESTING FIXTURES =====


@pytest.fixture
def dash_duo():
    """Dash testing utilities."""
    try:
        from dash.testing import DashComposite

        return DashComposite()
    except ImportError:
        pytest.skip("dash[testing] not available")


# ===== CONFIGURATION =====


def pytest_configure(config):
    """Pytest configuration."""
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end tests")
    config.addinivalue_line("markers", "slow: marks tests as slow-running")


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on location."""
    for item in items:
        # Mark integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Mark e2e tests
        if "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
            item.add_marker(pytest.mark.slow)
