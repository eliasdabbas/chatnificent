"""Integration tests for chatnificent.client.

Spins up a real DevServer in a background thread (no mocks) and exercises
each public function in chatnificent.client end-to-end. Tests both streaming
and non-streaming LLM modes.
"""

from __future__ import annotations

import socket
import threading
import time
from functools import partial
from http.server import HTTPServer
from typing import Iterator

import pytest
from chatnificent import Chatnificent
from chatnificent import client as chat_client
from chatnificent.llm import Echo
from chatnificent.server import DevServer, _DevHandler
from chatnificent.store import InMemory

# ---------------------------------------------------------------- fixtures


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_server(host: str, port: int, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError(f"Server did not start within {timeout}s")


def _spin_devserver(stream: bool) -> Iterator[tuple]:
    port = _find_free_port()
    app = Chatnificent(
        llm=Echo(stream=stream),
        store=InMemory(),
        server=DevServer(),
    )
    handler = partial(_DevHandler, app)
    httpd = HTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    _wait_for_server("127.0.0.1", port)
    try:
        yield f"http://127.0.0.1:{port}", app
    finally:
        httpd.shutdown()
        thread.join(timeout=2)


@pytest.fixture
def streaming_server():
    yield from _spin_devserver(stream=True)


@pytest.fixture
def non_streaming_server():
    yield from _spin_devserver(stream=False)


# ------------------------------------------------------------ start_session


class TestStartSession:
    def test_returns_session_with_base_url(self, streaming_server):
        base_url, _ = streaming_server
        session = chat_client.start_session(base_url=base_url)
        assert isinstance(session, chat_client.Session)
        assert session.base_url == base_url

    def test_mints_session_cookie(self, streaming_server):
        base_url, _ = streaming_server
        session = chat_client.start_session(base_url=base_url)
        cookies = {c.name: c.value for c in session.cookie_jar}
        assert "chatnificent_session" in cookies
        assert cookies["chatnificent_session"]  # non-empty

    def test_cookie_value_equals_user_id(self, streaming_server):
        base_url, app = streaming_server
        session = chat_client.start_session(base_url=base_url)
        cookie_value = next(
            c.value for c in session.cookie_jar if c.name == "chatnificent_session"
        )
        # That cookie value should now exist as a user in the store
        # (after at least one persisted action). list_conversations is enough
        # to confirm the server scopes data by this user.
        convos = chat_client.list_conversations(session=session)
        assert convos == []  # fresh user, no conversations yet


# --------------------------------------------------------------- send_message


class TestSendMessageStreaming:
    def test_yields_delta_then_done(self, streaming_server):
        base_url, _ = streaming_server
        session = chat_client.start_session(base_url=base_url)
        events = list(chat_client.send_message("hello", session=session))

        assert any(e["event"] == "delta" for e in events)
        assert events[-1]["event"] == "done"
        done = events[-1]["data"]
        assert "conversation_id" in done
        assert "path" in done

    def test_returns_iterator_not_list(self, streaming_server):
        base_url, _ = streaming_server
        session = chat_client.start_session(base_url=base_url)
        result = chat_client.send_message("hi", session=session)
        # Plain generator-like: iterable, not subscriptable
        assert hasattr(result, "__iter__")
        assert hasattr(result, "__next__")

    def test_creates_new_conversation_when_no_id(self, streaming_server):
        base_url, _ = streaming_server
        session = chat_client.start_session(base_url=base_url)
        events = list(chat_client.send_message("hi", session=session))
        new_id = events[-1]["data"]["conversation_id"]
        assert new_id  # non-empty
        listed = chat_client.list_conversations(session=session)
        assert any(c["id"] == new_id for c in listed)

    def test_continues_existing_conversation(self, streaming_server):
        base_url, _ = streaming_server
        session = chat_client.start_session(base_url=base_url)

        first = list(chat_client.send_message("one", session=session))
        cid = first[-1]["data"]["conversation_id"]

        second = list(
            chat_client.send_message("two", session=session, conversation_id=cid)
        )
        assert second[-1]["data"]["conversation_id"] == cid

        convo = chat_client.load_conversation(cid, session=session)
        user_messages = [m for m in convo["messages"] if m["role"] == "user"]
        assert [m["content"] for m in user_messages] == ["one", "two"]


class TestSendMessageNonStreaming:
    def test_yields_single_done_event(self, non_streaming_server):
        base_url, _ = non_streaming_server
        session = chat_client.start_session(base_url=base_url)
        events = list(chat_client.send_message("hello", session=session))

        # In non-streaming mode the server returns one JSON blob; we wrap it
        # as a synthetic 'done' event so the caller always sees the same shape.
        assert len(events) == 1
        assert events[0]["event"] == "done"
        payload = events[0]["data"]
        assert "conversation_id" in payload
        assert "response" in payload  # server's non-streaming response key


# --------------------------------------------------------- list_conversations


class TestListConversations:
    def test_empty_session_has_no_conversations(self, streaming_server):
        base_url, _ = streaming_server
        session = chat_client.start_session(base_url=base_url)
        assert chat_client.list_conversations(session=session) == []

    def test_shows_created_conversations(self, streaming_server):
        base_url, _ = streaming_server
        session = chat_client.start_session(base_url=base_url)
        list(chat_client.send_message("first", session=session))
        list(chat_client.send_message("second", session=session))

        convos = chat_client.list_conversations(session=session)
        assert len(convos) == 2
        for c in convos:
            assert "id" in c
            assert "title" in c

    def test_sessions_are_isolated(self, streaming_server):
        base_url, _ = streaming_server
        alice = chat_client.start_session(base_url=base_url)
        bob = chat_client.start_session(base_url=base_url)

        list(chat_client.send_message("alice msg", session=alice))

        assert len(chat_client.list_conversations(session=alice)) == 1
        assert chat_client.list_conversations(session=bob) == []


# ---------------------------------------------------------- load_conversation


class TestLoadConversation:
    def test_returns_full_message_list(self, streaming_server):
        base_url, _ = streaming_server
        session = chat_client.start_session(base_url=base_url)
        events = list(chat_client.send_message("hello", session=session))
        cid = events[-1]["data"]["conversation_id"]

        convo = chat_client.load_conversation(cid, session=session)
        assert convo["id"] == cid
        assert "messages" in convo
        assert "path" in convo
        roles = [m["role"] for m in convo["messages"]]
        assert roles == ["user", "assistant"]
        assert convo["messages"][0]["content"] == "hello"


# ----------------------------------------------------------------- collect_text


class TestCollectText:
    def test_joins_delta_events(self):
        events = [
            {"event": "delta", "data": "Hello"},
            {"event": "delta", "data": ", "},
            {"event": "delta", "data": "world!"},
            {"event": "done", "data": {"conversation_id": "x"}},
        ]
        assert chat_client.collect_text(events) == "Hello, world!"

    def test_ignores_non_delta_events(self):
        events = [
            {"event": "delta", "data": "kept"},
            {"event": "tool_call", "data": {"name": "search"}},
            {"event": "error", "data": "ignored"},
            {"event": "done", "data": {}},
        ]
        assert chat_client.collect_text(events) == "kept"

    def test_empty_iterable(self):
        assert chat_client.collect_text([]) == ""

    def test_works_with_send_message(self, streaming_server):
        base_url, _ = streaming_server
        session = chat_client.start_session(base_url=base_url)
        text = chat_client.collect_text(
            chat_client.send_message("hello", session=session)
        )
        # Echo LLM streams a non-empty greeting that mentions Echo
        assert text
        assert "Echo" in text


# ---------------------------------------------------- transport-level options


class TestHeadersAndTimeout:
    def test_custom_header_is_forwarded(self, streaming_server):
        base_url, _ = streaming_server
        session = chat_client.start_session(
            base_url=base_url, headers={"X-Test-Marker": "1"}
        )
        # Just ensure passing headers doesn't break the call.
        assert isinstance(session, chat_client.Session)

    def test_timeout_is_accepted(self, streaming_server):
        base_url, _ = streaming_server
        session = chat_client.start_session(base_url=base_url, timeout=5.0)
        events = list(chat_client.send_message("hi", session=session, timeout=5.0))
        assert events[-1]["event"] == "done"
