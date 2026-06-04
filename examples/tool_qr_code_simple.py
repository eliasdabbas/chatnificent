# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "chatnificent[openai]>=0.0.25",
#     "segno>=1.6",
# ]
# ///
"""
Tool-produced artifacts — QR codes on demand
============================================

Other ``artifact_examples/*_simple.py`` files produce the artifact from
the **LLM's own response** (``extract_content`` returns an ``Artifact``,
the engine's ``_finalize_content`` saves it). This file produces it
from a **Python tool the LLM decides to call**.

The new pattern
---------------
::

    def make_qr_code(text: str) -> Artifact:  # ← the only thing devs write
        ...

Two thin subclasses make that one line possible *today*; both can
collapse into the framework later.

Why two subclasses (current limitation)
---------------------------------------
1. ``ArtifactPythonTool`` — the base ``PythonTool`` calls
   ``json.dumps(result)`` to embed the result in the LLM-bound tool
   message. ``Artifact`` is a frozen dataclass with raw ``bytes`` and
   crashes ``json.dumps``. So we intercept any function whose return
   annotation is ``Artifact``: stash it by ``tool_call_id`` (the
   framework's own foreign key — no invented tokens) and return an
   empty tool message.
2. ``ArtifactOrchestrator`` — the engine's ``_finalize_content`` only
   runs on the **final** LLM response, not on tool returns mid-loop.
   So we need our own save-time hook. ``_save_conversation`` is the
   smallest seam where ``user_id`` is in scope. It walks the tool
   messages, pops each stashed Artifact, embeds them, and replaces the
   trailing assistant text — the user asked for the QR, not chitchat
   about the QR.

Known cost: the agentic loop still makes one extra LLM call (with an
empty tool result) before we replace its output. When this pattern is
promoted to the framework, that round-trip goes away too.

Prerequisites
-------------
::

    export OPENAI_API_KEY="sk-..."
    uv run --script artifact_examples/tool_qr_code_simple.py
"""

import io
import json
import threading
from typing import Any, Dict, List, Optional, get_type_hints

import chatnificent as chat
import segno
from chatnificent.models import ASSISTANT_ROLE, TOOL_ROLE, Artifact, Conversation


def make_qr_code(text: str) -> Artifact:
    """Generate a scannable QR code PNG for arbitrary text.

    Works for URLs, Wi-Fi join strings (``WIFI:T:WPA;S:<ssid>;P:<pass>;;``),
    vCards, plain text — anything that fits in a QR code.

    For map locations you MUST use the ``geo:`` URI scheme (RFC 5870) so
    phones open the native Maps app and drop a pin:
    ``geo:<lat>,<lon>?q=<url-encoded label>``, e.g.
    ``geo:40.7829,-73.9654?q=Central%20Park``. Use your own knowledge of
    well-known landmarks to supply approximate coordinates. Never encode
    a ``maps.google.com`` URL — those only open a browser tab.

    Parameters
    ----------
    text : str
        The string to encode into the QR code.
    """
    buf = io.BytesIO()
    segno.make(text, error="m").save(buf, kind="png", scale=8, border=2)
    return Artifact(data=buf.getvalue(), ext=".png", folder="qrcodes")


class ArtifactPythonTool(chat.tools.PythonTool):
    """``PythonTool`` that supports functions returning ``Artifact``."""

    def __init__(self):
        super().__init__()
        self._pending: Dict[str, Artifact] = {}
        self._lock = threading.Lock()

    def execute_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        func = self._registry.get(tool_call.get("function_name", ""))
        if not func or get_type_hints(func).get("return") is not Artifact:
            return super().execute_tool_call(tool_call)

        call_id = tool_call["id"]
        args = json.loads(tool_call.get("function_args") or "{}")
        with self._lock:
            self._pending[call_id] = func(**args)
        return {
            "tool_call_id": call_id,
            "function_name": func.__name__,
            "content": "",
            "is_error": False,
        }

    def pop_artifact(self, call_id: str) -> Optional[Artifact]:
        with self._lock:
            return self._pending.pop(call_id, None)


class ArtifactOrchestrator(chat.engine.Orchestrator):
    """Resolves stashed tool Artifacts at save time."""

    def _save_conversation(self, conversation: Conversation, user_id: str) -> None:
        embeds: List[str] = []
        for msg in conversation.messages:
            if msg.get("role") != TOOL_ROLE:
                continue
            artifact = self.app.tools.pop_artifact(msg.get("tool_call_id") or "")
            if artifact is not None:
                embeds.append(
                    self._artifact_to_html(artifact, user_id, conversation.id)
                )

        if embeds:
            for msg in reversed(conversation.messages):
                if msg.get("role") == ASSISTANT_ROLE and isinstance(
                    msg.get("content"), str
                ):
                    msg["content"] = "\n\n".join(embeds)
                    break
        return super()._save_conversation(conversation, user_id)


tools = ArtifactPythonTool()
tools.register_function(make_qr_code)


welcome_message = """## QR codes, on demand — from a tool, not the model

Ask for a QR code in plain English. The LLM calls the `make_qr_code`
tool — which knows nothing about the framework, it just returns an
`Artifact` — and the rendered QR appears inline. Scan it with your
phone; it's a real QR code.

<div id="suggestions">
  <button class="suggestion" data-insert-prompt="Make a QR code for https://chatnificent.com">
    <span class="suggestion-label">URL</span>
    <span class="suggestion-text">A QR for the Chatnificent homepage.</span>
  </button>
  <button class="suggestion" data-insert-prompt="Generate a QR code that joins my Wi-Fi: SSID is Chatnificent, password is Hackable123, WPA2.">
    <span class="suggestion-label">WI-FI</span>
    <span class="suggestion-text">A QR that auto-joins a Wi-Fi network.</span>
  </button>
  <button class="suggestion" data-insert-prompt="Make a vCard QR code for Ada Lovelace, founder of computing, email ada@analytical.engine, phone +44 20 7946 0958.">
    <span class="suggestion-label">VCARD</span>
    <span class="suggestion-text">A QR that adds a contact to your phone.</span>
  </button>
  <button class="suggestion" data-insert-prompt="Make a QR code that opens the Eiffel Tower in the Maps app.">
    <span class="suggestion-label">GEO</span>
    <span class="suggestion-text">A QR that drops a pin on the Eiffel Tower.</span>
  </button>
</div>
"""


SYSTEM_PROMPT = (
    "You generate QR codes by calling the make_qr_code tool. For map or "
    "location requests, always pass a geo: URI of the form "
    "`geo:<lat>,<lon>?q=<url-encoded label>` — use your own knowledge of "
    "well-known landmarks to supply approximate coordinates. Never pass a "
    "maps.google.com URL for locations."
)


class QRCodeLLM(chat.llm.OpenAI):
    def generate_response(self, messages, **kwargs):
        if not messages or messages[0].get("role") != "system":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, *messages]
        return super().generate_response(messages, **kwargs)


app = chat.Chatnificent(
    llm=QRCodeLLM(model="gpt-4o-mini", stream=False),
    engine=ArtifactOrchestrator(),
    tools=tools,
    store=chat.store.File(base_dir="./artifact_examples/_convos_tool_qr_code_simple"),
    layout=chat.layout.Default(
        page_title="Build an AI Chatbot App That Generates QR Codes in Python | Chatnificent",
        welcome_message=welcome_message,
    ),
)


if __name__ == "__main__":
    app.run()
