# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "chatnificent[openai]",
# ]
# ///
"""
UI Interactions — Binding Controls to LLM Parameters
=====================================================

This example shows how to wire a UI control so that a user's selection is
silently injected into the next LLM API call — no page reload, no extra
parameters in the chat message.

The pattern has two parts:

1. **A ``Control``** — declares the HTML to render, which slot to place it in,
   and which LLM kwarg it maps to. An optional ``cast`` converts the raw
   browser string to the right Python type before it reaches the API.

2. **Pass it to ``DefaultLayout(controls=[...])``** — no subclassing needed.
   Everything else (state storage, thread-safety, ``POST /api/interactions``
   endpoint, engine injection) is handled by the framework.

The JS side
-----------
The built-in ``chatInteraction(element, data?)`` helper is already injected by
the framework. Call it from any standard DOM event attribute::

    onchange = "chatInteraction(this)"  # uses element.value
    onclick = "chatInteraction(this, 'pirate')"  # custom data

It fires ``POST /api/interactions`` with ``{"id": element.id, "data": value}``
and the server updates the user's control state.

Null sentinel
-------------
Sending ``null`` from JS clears the parameter so it is not forwarded to the
API. The ``-- no limit --`` option uses this pattern::

    chatInteraction(this, this.value === '' ? null : this.value)

Running
-------
::

    uv run examples/ui_interactions.py

Open http://127.0.0.1:7777, choose a token limit from the toolbar, send a
message, then check ``raw_api_requests.jsonl`` in the conversation folder to
confirm ``max_completion_tokens`` reached the API.
"""

import chatnificent as chat
from chatnificent.layout import Control

TOKEN_OPTIONS = "\n".join(
    f'<option value="{n}"{"  selected" if n == 20 else ""}>{n}</option>'
    for n in list(range(10, 110, 10)) + list(range(200, 600, 100))
)

TOOLBAR_HTML = f"""
<div style="padding:8px 16px;border-bottom:1px solid var(--border);
     background:var(--bg);display:flex;align-items:center;gap:8px;font-size:13px;">
  <label for="token-limit">Max tokens</label>
  <select id="token-limit" onchange="chatInteraction(this, this.value === '' ? null : this.value)"
          style="font-size:13px;padding:2px 6px;border-radius:4px;
                 border:1px solid var(--border);background:var(--bg);color:var(--text);">
    <option value="">-- no limit --</option>
    {TOKEN_OPTIONS}
  </select>
</div>
"""

control = Control(
    id="token-limit",
    html=TOOLBAR_HTML,
    slot="toolbar",
    llm_param="max_completion_tokens",
    cast=int,
)

app = chat.Chatnificent(layout=chat.layout.DefaultLayout(controls=[control]))

if __name__ == "__main__":
    app.run()
