"""
HTTP client for talking to a running Chatnificent server.

Use this from scripts, tests, and other programs to drive any Chatnificent
app over the wire. **This is the canonical Python interface** for
programmatic access to a Chatnificent app.

Write the same Python whether the server is on localhost, behind nginx,
on your-app.com, or in CI — the only thing that changes is ``base_url``.

Not a pillar — this module is a *consumer* of the server's HTTP API, not an
extension point of the framework itself. Pillars (LLM, Store, Engine,
Auth, ...) live on the server; this client lives outside.

Five functions cover the surface:

================================  =========================
Function                           HTTP call
================================  =========================
``start_session``                  ``GET  /api/conversations``
``send_message``                   ``POST /api/chat``
``list_conversations``             ``GET  /api/conversations``
``load_conversation``              ``GET  /api/conversations/{id}``
``collect_text``                   (pure helper, no HTTP)
================================  =========================

Errors
------

All functions propagate ``urllib.error.HTTPError`` on non-2xx responses
and ``urllib.error.URLError`` on connection failures. We do not wrap or
translate these — handle them like any other ``urllib`` call.

Examples
--------

Send one message and print the reply::

    import chatnificent as chat

    s = chat.client.start_session()
    text = chat.client.collect_text(chat.client.send_message("Hi!", session=s))
    print(text)

Stream tokens as they arrive::

    s = chat.client.start_session()
    for event in chat.client.send_message("Tell me a joke.", session=s):
        if event["event"] == "delta":
            print(event["data"], end="", flush=True)

Continue an existing conversation::

    s = chat.client.start_session()
    first = list(chat.client.send_message("Pick a number 1-10.", session=s))
    convo_id = first[-1]["data"]["conversation_id"]

    chat.client.collect_text(
        chat.client.send_message("Now double it.", session=s, conversation_id=convo_id)
    )

Two independent users against the same server::

    alice = chat.client.start_session()
    bob   = chat.client.start_session()

    chat.client.send_message("Hi from Alice", session=alice)
    chat.client.send_message("Hi from Bob",   session=bob)

    assert chat.client.list_conversations(session=alice) \\
        != chat.client.list_conversations(session=bob)

Talk to a remote app::

    s = chat.client.start_session(base_url="https://your-app.com/")
    chat.client.list_conversations(session=s)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from http.cookiejar import CookieJar
from typing import Any, Dict, Iterable, Iterator, List, Optional
from urllib.parse import urljoin
from urllib.request import HTTPCookieProcessor, Request, build_opener

DEFAULT_BASE_URL = "http://127.0.0.1:7777"
DEFAULT_TIMEOUT = 60.0


@dataclass
class Session:
    """Per-server state carried across every call in this module.

    Holds:

    - ``base_url`` — where the Chatnificent app is reachable.
    - ``cookie_jar`` — stores the ``chatnificent_session`` cookie the
      server sets on first contact. That cookie's value *is* the
      ``user_id`` the server uses to scope conversations.
    - ``_opener`` — a cached ``urllib`` opener (cookie-aware HTTP
      executor) created lazily on first use. Internal; treat as private.

    Always construct via :func:`start_session` so the jar is primed
    against the target server. Then pass the returned ``Session`` as the
    ``session=`` keyword argument to every other function in this module.
    """

    base_url: str = DEFAULT_BASE_URL
    cookie_jar: CookieJar = field(default_factory=CookieJar)
    _opener: Any = field(default=None, init=False, repr=False, compare=False)


def _opener_for(session: Session):
    if session._opener is None:
        session._opener = build_opener(HTTPCookieProcessor(session.cookie_jar))
    return session._opener


def _build_request(
    session: Session,
    path: str,
    *,
    method: str = "GET",
    body: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
) -> Request:
    url = urljoin(session.base_url.rstrip("/") + "/", path.lstrip("/"))
    final_headers: Dict[str, str] = {}
    data: Optional[bytes] = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        final_headers["Content-Type"] = "application/json"
    if headers:
        final_headers.update(headers)
    return Request(url, data=data, headers=final_headers, method=method)


def start_session(
    base_url: str = DEFAULT_BASE_URL,
    *,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Session:
    """Open a session against ``base_url`` and return it primed with a cookie.

    Performs the lightest possible GET (``/api/conversations``) so the
    server mints a ``chatnificent_session`` cookie that subsequent calls
    reuse.
    """
    session = Session(base_url=base_url)
    req = _build_request(session, "/api/conversations", headers=headers)
    with _opener_for(session).open(req, timeout=timeout) as resp:
        resp.read()  # drain
    return session


def list_conversations(
    *,
    session: Session,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> List[Dict[str, str]]:
    """Return the list of conversations the current session can see."""
    req = _build_request(session, "/api/conversations", headers=headers)
    with _opener_for(session).open(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return payload.get("conversations", []) if isinstance(payload, dict) else []


def load_conversation(
    convo_id: str,
    *,
    session: Session,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """Load a single conversation by id.

    Returns the server's JSON response verbatim. Typically that's
    ``{"id": ..., "messages": [...], "path": ...}`` where each message
    is ``{"role": ..., "content": ...}``, but the exact shape depends on
    the app — custom servers, stores, or engines may add or change fields.
    """
    req = _build_request(session, f"/api/conversations/{convo_id}", headers=headers)
    with _opener_for(session).open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def send_message(
    message: str,
    *,
    session: Session,
    conversation_id: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Iterator[Dict[str, Any]]:
    """Send a message and yield each SSE event as a dict.

    **Returns a generator** — iterate it with a ``for`` loop, or wrap it
    in ``list(...)`` to collect every event. Nothing is sent to the
    server until you start iterating.

    Each streamed event has the shape ``{"event": "<name>", "data":
    <payload>}``. The server emits one ``delta`` event per token and a
    final ``done`` event whose ``data`` includes ``conversation_id`` and
    ``path``.

    When the server returns a non-streaming JSON response (the LLM has
    ``stream=False``), this function yields a single ``done`` event
    wrapping that JSON in ``data``. The reply text in that case lives at
    ``data["response"]``, not in any ``delta`` event — so
    :func:`collect_text` will return an empty string.
    """
    body: Dict[str, Any] = {"message": message}
    if conversation_id is not None:
        body["conversation_id"] = conversation_id

    req = _build_request(
        session,
        "/api/chat",
        method="POST",
        body=body,
        headers=headers,
    )
    resp = _opener_for(session).open(req, timeout=timeout)

    content_type = resp.headers.get("Content-Type", "")
    if "text/event-stream" not in content_type:
        with resp:
            raw = resp.read().decode("utf-8")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"raw": raw}
        yield {"event": "done", "data": payload}
        return

    with resp:
        yield from _iter_sse(resp)


def _iter_sse(resp) -> Iterator[Dict[str, Any]]:
    """Parse Chatnificent's SSE stream into ``{event, data}`` dicts.

    Chatnificent encodes the full ``{event, data}`` object inside each
    ``data:`` line, so we just JSON-decode that line and yield it as-is.
    Comment lines (``:keep-alive``) and ``event:`` headers are ignored.
    """
    data_lines: List[str] = []
    for raw_line in resp:
        line = raw_line.decode("utf-8").rstrip("\r\n")
        if line == "":
            if data_lines:
                yield json.loads("\n".join(data_lines))
                data_lines = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data_lines.append(line[len("data:") :].lstrip())


def collect_text(events: Iterable[Dict[str, Any]]) -> str:
    """Concatenate the text from every ``delta`` event in ``events``.

    Convenience for "I just want the assistant's reply as a string." Pass
    the iterator returned by :func:`send_message` directly::

        events = chat.client.send_message("Hi", session=s)
        text = chat.client.collect_text(events)

    Other event kinds (``done``, ``error``, future ``tool_call`` /
    ``usage`` / etc.) are ignored — write your own loop over
    :func:`send_message` if you need them.
    """
    return "".join(
        event["data"]
        for event in events
        if event.get("event") == "delta" and isinstance(event.get("data"), str)
    )
