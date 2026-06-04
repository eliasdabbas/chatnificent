# /// script
# requires-python = ">=3.10"
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

2. **Pass it to ``Default(controls=[...])``** — no subclassing needed.
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

TOKEN_OPTIONS = "\n".join(
    f'<option value="{n}"{"  selected" if n == 20 else ""}>{n}</option>'
    for n in list(range(10, 110, 10)) + list(range(200, 600, 100))
)

# Styling is kept lean: the default template already styles ``<select>`` and
# ``<label>`` (custom chevron, focus ring, dark-mode aware), so we only add a
# thin flex toolbar to position the label next to the dropdown. All sizes,
# colors, and spacing come from the framework's CSS custom properties.
TOOLBAR_HTML = f"""
<div style="display:flex;align-items:center;gap:var(--space-3);
            padding:var(--space-3) var(--space-4);
            border-bottom:1px solid var(--border);
            background:var(--bg);font-size:var(--text-sm);
            color:var(--text-secondary);">
  <label for="token-limit">Max tokens</label>
  <select id="token-limit"
          onchange="chatInteraction(this, this.value === '' ? null : this.value)"
          style="width:auto;min-width:160px;">
    <option value="">-- no limit --</option>
    {TOKEN_OPTIONS}
  </select>
</div>
"""

control = chat.layout.Control(
    id="token-limit",
    html=TOOLBAR_HTML,
    slot="messages-begin",
    llm_param="max_completion_tokens",
    cast=int,
)

welcome_message = """## UI controls bound to LLM kwargs

The dropdown above the messages directly controls `max_completion_tokens`. Pick a small cap, send a long-form prompt, then change to a larger cap and resend the **same** prompt — the answer length tracks the dropdown live.

<div id="suggestions">
  <button class="suggestion" data-insert-prompt="Set the dropdown to a small cap, then send: &quot;Explain how an espresso machine works in detail.&quot;">
    <span class="suggestion-label">SHORT</span>
    <span class="suggestion-text">Test a small token cap.</span>
  </button>
  <button class="suggestion" data-insert-prompt="Now bump the dropdown up and resend the same prompt to see a richer answer.">
    <span class="suggestion-label">MEDIUM</span>
    <span class="suggestion-text">Same prompt, more room.</span>
  </button>
  <button class="suggestion" data-insert-prompt="Write a poem about coffee — try multiple dropdown values to feel the cap.">
    <span class="suggestion-label">POEM</span>
    <span class="suggestion-text">Watch the cap mid-stream.</span>
  </button>
</div>"""

app = chat.Chatnificent(
    layout=chat.layout.Default(
        page_title="Python AI Chatbot App Example: Custom UI Controls | Chatnificent",
        controls=[control],
        welcome_message=welcome_message,
    )
)

if __name__ == "__main__":
    app.run()
