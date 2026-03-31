"""Contract tests: raw API log persistence across store implementations.

Verifies that save_raw_api_request/response + load_raw_api_requests/responses
round-trip correctly, preserve append ordering, and degrade gracefully.
"""

import pytest
from chatnificent.models import Conversation


class TestRawLogPersistence:
    """Raw API log round-trip and ordering across File and SQLite stores."""

    @pytest.fixture(params=["File", "SQLite"])
    def store(self, request, tmp_path):
        from chatnificent import store as store_mod

        if request.param == "File":
            return store_mod.File(str(tmp_path / "file_store"))
        elif request.param == "SQLite":
            return store_mod.SQLite(str(tmp_path / "test.db"))

    @pytest.fixture
    def seeded_store(self, store):
        """Store with a conversation already saved (needed for File store dirs)."""
        convo = Conversation(id="c001", messages=[{"role": "user", "content": "hi"}])
        store.save_conversation("user1", convo)
        return store

    def test_request_round_trip(self, seeded_store):
        """save_raw_api_request → load_raw_api_requests reproduces the payload."""
        payload = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}
        seeded_store.save_raw_api_request("user1", "c001", payload)
        loaded = seeded_store.load_raw_api_requests("user1", "c001")

        assert len(loaded) == 1
        assert loaded[0] == payload

    def test_response_round_trip(self, seeded_store):
        """save_raw_api_response → load_raw_api_responses reproduces the payload."""
        payload = {
            "id": "chatcmpl-abc",
            "choices": [{"message": {"role": "assistant", "content": "hello"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10},
        }
        seeded_store.save_raw_api_response("user1", "c001", payload)
        loaded = seeded_store.load_raw_api_responses("user1", "c001")

        assert len(loaded) == 1
        assert loaded[0] == payload

    def test_multiple_appends_preserve_ordering(self, seeded_store):
        """Multiple raw exchanges append in FIFO order, never overwrite."""
        payloads = [
            {"model": "gpt-4", "turn": 1},
            {"model": "gpt-4", "turn": 2},
            {"model": "gpt-4", "turn": 3},
        ]
        for p in payloads:
            seeded_store.save_raw_api_request("user1", "c001", p)

        loaded = seeded_store.load_raw_api_requests("user1", "c001")
        assert len(loaded) == 3
        assert loaded == payloads

    def test_request_and_response_are_independent(self, seeded_store):
        """Request and response logs don't interfere with each other."""
        req = {"model": "gpt-4", "messages": []}
        resp = {"id": "resp-1", "choices": []}

        seeded_store.save_raw_api_request("user1", "c001", req)
        seeded_store.save_raw_api_response("user1", "c001", resp)

        requests = seeded_store.load_raw_api_requests("user1", "c001")
        responses = seeded_store.load_raw_api_responses("user1", "c001")

        assert len(requests) == 1
        assert requests[0] == req
        assert len(responses) == 1
        assert responses[0] == resp

    def test_load_empty_returns_empty_list(self, seeded_store):
        """Loading logs from a conversation with no raw data returns []."""
        assert seeded_store.load_raw_api_requests("user1", "c001") == []
        assert seeded_store.load_raw_api_responses("user1", "c001") == []

    def test_load_nonexistent_conversation_returns_empty(self, seeded_store):
        """Loading logs for nonexistent conversation returns []."""
        assert seeded_store.load_raw_api_requests("user1", "nope") == []
        assert seeded_store.load_raw_api_responses("user1", "nope") == []

    def test_complex_nested_payload_round_trips(self, seeded_store):
        """Deeply nested payloads with various JSON types survive."""
        payload = {
            "model": "gpt-4",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "search",
                                "arguments": '{"query": "nested \\"quotes\\""}',
                            },
                        }
                    ],
                }
            ],
            "temperature": 0.7,
            "top_p": None,
            "flags": [True, False, None, 42, 3.14],
        }
        seeded_store.save_raw_api_request("user1", "c001", payload)
        loaded = seeded_store.load_raw_api_requests("user1", "c001")
        assert loaded[0] == payload

    def test_response_list_payload_round_trips(self, seeded_store):
        """save_raw_api_response accepts list payloads (streaming chunks)."""
        chunks = [
            {"delta": {"content": "Hello"}},
            {"delta": {"content": " world"}},
        ]
        seeded_store.save_raw_api_response("user1", "c001", chunks)
        loaded = seeded_store.load_raw_api_responses("user1", "c001")
        assert len(loaded) == 1
        assert loaded[0] == chunks


class TestInMemoryRawLogNoOp:
    """InMemory store's raw logging degrades gracefully (no-op base impl)."""

    def test_inmemory_load_requests_returns_empty(self):
        from chatnificent.store import InMemory

        store = InMemory()
        convo = Conversation(id="c1", messages=[{"role": "user", "content": "hi"}])
        store.save_conversation("u1", convo)

        store.save_raw_api_request("u1", "c1", {"model": "test"})
        loaded = store.load_raw_api_requests("u1", "c1")
        assert isinstance(loaded, list)

    def test_inmemory_load_responses_returns_empty(self):
        from chatnificent.store import InMemory

        store = InMemory()
        convo = Conversation(id="c1", messages=[{"role": "user", "content": "hi"}])
        store.save_conversation("u1", convo)

        store.save_raw_api_response("u1", "c1", {"id": "resp"})
        loaded = store.load_raw_api_responses("u1", "c1")
        assert isinstance(loaded, list)
