# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "chatnificent[openai]",
# ]
# ///
"""
Interactive Web Search — Live UI Controls for the Responses API
===============================================================

An AI research assistant where every search parameter is a live UI
control. Three knobs, wired directly to ``responses.create`` kwargs
through the ``Control`` + ``DefaultLayout`` system — no custom server
code, no per-request plumbing.

Controls
--------
All three controls live in the toolbar, side by side.

**Reasoning effort** → ``reasoning``
    How hard the model thinks before answering. Matters most for
    complex, ambiguous queries. Try asking "Why is the sky blue?" with
    *None* vs *High* and watch the reasoning depth change. Supported
    on ``o-series`` and ``gpt-5.1+`` models; silently ignored on
    others. The Responses API takes ``reasoning={"effort": "low"}``,
    not a plain string — ``cast=_build_reasoning`` wraps the value.
    All six API values are available: ``none``, ``minimal``, ``low``,
    ``medium``, ``high``, ``xhigh``. Defaults to ``none`` on page load.
    Note: ``minimal`` is excluded here — it cannot be used with the
    ``web_search`` tool and returns a 400 error.

**Domain restriction** → ``tools`` (multi-select)
    Pick one or more domains to restrict the search. The ``cast``
    function builds the full ``tools`` list-of-dicts from the
    comma-joined selected values::

        [{"type": "web_search", "filters": {"allowed_domains": [...]}}]

    When nothing is selected, ``cast("")`` returns the unfiltered
    ``[{"type": "web_search"}]`` — same as ``default_params``. No null
    sentinel needed: empty = unrestricted. This is the canonical
    pattern for overriding a list-typed LLM kwarg from a ``Control``:
    ``cast`` can return any Python object, not just a scalar.

**Force search** → ``tool_choice`` (checkbox)
    When checked, ``tool_choice="required"`` forces a web search on
    every message. When unchecked, ``tool_choice="auto"`` lets the
    model decide. The checkbox uses ``value="auto"`` so the
    framework's DOMContentLoaded auto-init reads ``el.value = "auto"``
    and sends the correct initial state without special handling.

Citations
---------
The ``extract_stream_delta`` override catches the
``response.output_item.done`` event that carries URL annotations and
formats them as a collapsible ``<details>`` block appended to the
streamed answer.

Why no Streaming toggle?
------------------------
The server's routing decision (SSE vs JSON response) is made once at
startup from ``llm.default_params.get("stream")``. Toggling it per-
request via a ``Control`` *would* reach ``generate_response`` — but
the server is already committed to the SSE code path, so a
``stream=False`` kwarg would hand a non-streaming response object to
the streaming accumulator loop and break. Fixing this cleanly requires
a new ``_should_stream(user_id)`` seam on the engine that the server
consults before routing. Tracked on the roadmap; not yet supported.

Checkbox + auto-init trick
--------------------------
The framework's DOMContentLoaded auto-init calls
``chatInteraction(el)`` for each registered control, which reads
``el.value``. A checkbox's ``el.value`` is always ``"on"`` by default
— independent of checked state. Fix: set ``value="auto"`` on the
element. Auto-init reads ``el.value = "auto"`` (the correct
default); ``onchange`` overrides with ``this.checked ? 'required' :
'auto'``.

Verify
------
::

    cat <convo-folder>/raw_api_requests.jsonl | python -m json.tool

With "Wikipedia" selected you should see::

    "tools": [{"type": "web_search", "filters": {"allowed_domains": ["en.wikipedia.org"]}}]

With no domains selected the ``tools`` value is the unfiltered
``[{"type": "web_search"}]`` — same as ``default_params``.

Running
-------
::

    export OPENAI_API_KEY="sk-..."
    uv run examples/openai_responses_interactive_search.py

Then open http://127.0.0.1:7777.

What to hack next
-----------------
* Add a **model selector** control (``llm_param="model"``, ``cast=None``)
  to compare ``gpt-4o`` vs ``o4-mini`` on the same query.
* Replace the domain presets with a free-text ``<input>`` and a
  ``cast`` that splits on commas — instant custom domain filtering.
* Add a fourth control for ``max_output_tokens`` to cap response
  length when you only need a quick fact.
* Swap ``allowed_domains`` for ``blocked_domains`` to exclude noisy
  sources while keeping the rest of the web available.
"""

import chatnificent as chat
from chatnificent.layout import Control, DefaultLayout

# ---------------------------------------------------------------------------
# LLM subclass — routes through the Responses API + citation extraction
# ---------------------------------------------------------------------------


class InteractiveSearch(chat.llm.OpenAI):
    default_params = {
        "stream": True,
        "tools": [{"type": "web_search"}],
    }

    def generate_response(self, messages, **kwargs):
        return self.client.responses.create(
            model=self.model, input=messages, **{**self.default_params, **kwargs}
        )

    def extract_stream_delta(self, chunk):
        if chunk.type == "response.output_text.delta":
            return chunk.delta
        if chunk.type == "response.output_item.done":
            return self._format_citations(chunk.item)
        return None

    def _format_citations(self, item):
        if getattr(item, "type", None) != "message":
            return None
        urls, seen = [], set()
        for part in getattr(item, "content", []) or []:
            for annotation in getattr(part, "annotations", []) or []:
                if getattr(annotation, "type", None) != "url_citation":
                    continue
                url = annotation.url
                if url in seen:
                    continue
                seen.add(url)
                urls.append((annotation.title or url, url))
        if not urls:
            return None
        body = "\n".join(f"- [{title}]({url})" for title, url in urls)
        return (
            f"\n\n<details><summary>Sources ({len(urls)})</summary>\n\n"
            f"{body}\n\n</details>"
        )


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------

TOOLBAR_HTML = """
<style>
  #search-toolbar { display:flex; align-items:center; justify-content:center; gap:0; padding:10px 16px;
    border-bottom:1px solid var(--border); background:var(--surface);
    box-shadow:var(--shadow-sm); flex-wrap:wrap; }
  #search-toolbar .tb-section { display:flex; flex-direction:column; gap:7px; padding:0 20px; }
  #search-toolbar .tb-section:first-child { padding-left:0; }
  #search-toolbar .tb-divider { width:1px; background:var(--border); align-self:stretch; margin:4px 0; }
  #search-toolbar .tb-label { font-size:10px; font-weight:700; letter-spacing:0.06em;
    text-transform:uppercase; color:var(--text-secondary); }
  #search-toolbar .tb-row { display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
  #search-toolbar select { font-size:13px; padding:5px 28px 5px 10px; border-radius:8px;
    border:1.5px solid var(--border); background:var(--bg); color:var(--text);
    appearance:none; -webkit-appearance:none; cursor:pointer;
    background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%236b7280' stroke-width='2.5'%3E%3Cpolyline points='6 9 12 15 18 9'/%3E%3C/svg%3E");
    background-repeat:no-repeat; background-position:right 8px center; }
  .dp { display:inline-flex; align-items:center; gap:4px; padding:4px 11px;
    border-radius:20px; border:1.5px solid var(--border); cursor:pointer;
    font-size:12px; font-weight:500; color:var(--text-secondary); background:var(--bg);
    transition:all 0.15s; user-select:none; }
  .dp:hover { border-color:var(--accent); color:var(--accent); background:var(--accent-subtle); }
  .dp input[type=checkbox] { display:none; }
  .dp.on { border-color:var(--accent); background:var(--accent-subtle);
    color:var(--accent); box-shadow:0 0 0 3px var(--accent-ring); }
  .fn-toggle { position:relative; display:inline-flex; align-items:center;
    width:38px; height:22px; flex-shrink:0; cursor:pointer; }
  .fn-toggle input { opacity:0; width:0; height:0; position:absolute; }
  .fn-toggle-track { position:absolute; inset:0; background:var(--border);
    border-radius:22px; transition:background 0.2s; }
  .fn-toggle-thumb { position:absolute; height:16px; width:16px; left:3px; top:3px;
    background:var(--btn-text); border-radius:50%; transition:transform 0.2s;
    box-shadow:0 1px 3px rgba(0,0,0,0.25); pointer-events:none; }
  .fn-toggle input:checked ~ .fn-toggle-track { background:var(--accent); }
  .fn-toggle input:checked ~ .fn-toggle-thumb { transform:translateX(16px); }
</style>

<div id="search-toolbar">

  <div class="tb-section">
    <span class="tb-label">Reasoning effort</span>
    <select id="reasoning-effort" onchange="chatInteraction(this)">
      <option value="none" selected>None</option>
      <option value="low">Low</option>
      <option value="medium">Medium</option>
      <option value="high">High</option>
      <option value="xhigh">Max</option>
    </select>
  </div>

  <div class="tb-divider"></div>

  <div class="tb-section">
    <span class="tb-label">Domain filter</span>
    <div class="tb-row" id="domain-pills">
      <label class="dp"><input type="checkbox" value="en.wikipedia.org"
        onchange="this.closest('.dp').classList.toggle('on',this.checked);var s=document.querySelectorAll('#domain-pills input:checked');chatInteraction({id:'domain-select'},Array.from(s).map(function(c){return c.value;}).join(',')||null)">Wikipedia</label>
      <label class="dp"><input type="checkbox" value="reuters.com"
        onchange="this.closest('.dp').classList.toggle('on',this.checked);var s=document.querySelectorAll('#domain-pills input:checked');chatInteraction({id:'domain-select'},Array.from(s).map(function(c){return c.value;}).join(',')||null)">Reuters</label>
      <label class="dp"><input type="checkbox" value="apnews.com"
        onchange="this.closest('.dp').classList.toggle('on',this.checked);var s=document.querySelectorAll('#domain-pills input:checked');chatInteraction({id:'domain-select'},Array.from(s).map(function(c){return c.value;}).join(',')||null)">AP News</label>
      <label class="dp"><input type="checkbox" value="github.com"
        onchange="this.closest('.dp').classList.toggle('on',this.checked);var s=document.querySelectorAll('#domain-pills input:checked');chatInteraction({id:'domain-select'},Array.from(s).map(function(c){return c.value;}).join(',')||null)">GitHub</label>
      <label class="dp"><input type="checkbox" value="stackoverflow.com"
        onchange="this.closest('.dp').classList.toggle('on',this.checked);var s=document.querySelectorAll('#domain-pills input:checked');chatInteraction({id:'domain-select'},Array.from(s).map(function(c){return c.value;}).join(',')||null)">Stack Overflow</label>
      <label class="dp"><input type="checkbox" value="arxiv.org"
        onchange="this.closest('.dp').classList.toggle('on',this.checked);var s=document.querySelectorAll('#domain-pills input:checked');chatInteraction({id:'domain-select'},Array.from(s).map(function(c){return c.value;}).join(',')||null)">arXiv</label>
      <label class="dp"><input type="checkbox" value="nature.com"
        onchange="this.closest('.dp').classList.toggle('on',this.checked);var s=document.querySelectorAll('#domain-pills input:checked');chatInteraction({id:'domain-select'},Array.from(s).map(function(c){return c.value;}).join(',')||null)">Nature</label>
      <label class="dp"><input type="checkbox" value="bbc.com"
        onchange="this.closest('.dp').classList.toggle('on',this.checked);var s=document.querySelectorAll('#domain-pills input:checked');chatInteraction({id:'domain-select'},Array.from(s).map(function(c){return c.value;}).join(',')||null)">BBC</label>
      <label class="dp"><input type="checkbox" value="theguardian.com"
        onchange="this.closest('.dp').classList.toggle('on',this.checked);var s=document.querySelectorAll('#domain-pills input:checked');chatInteraction({id:'domain-select'},Array.from(s).map(function(c){return c.value;}).join(',')||null)">The Guardian</label>
    </div>
  </div>

  <div class="tb-divider"></div>

  <div class="tb-section" style="flex-direction:row;align-items:center;gap:10px;">
    <label class="fn-toggle">
      <input type="checkbox" id="search-mode" value="auto"
             onchange="chatInteraction(this, this.checked ? 'required' : 'auto')">
      <span class="fn-toggle-track"></span>
      <span class="fn-toggle-thumb"></span>
    </label>
    <span style="font-size:13px;font-weight:500;cursor:pointer;"
          onclick="var cb=document.getElementById('search-mode');cb.checked=!cb.checked;cb.dispatchEvent(new Event('change'))">
      Force search
    </span>
  </div>

</div>
"""

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def _build_reasoning(v: str) -> dict:
    return {"effort": v}


def _build_tools(v):
    if not v:
        return [{"type": "web_search"}]
    return [{"type": "web_search", "filters": {"allowed_domains": v.split(",")}}]


app = chat.Chatnificent(
    llm=InteractiveSearch(),
    layout=DefaultLayout(
        controls=[
            Control(
                id="reasoning-effort",
                html=TOOLBAR_HTML,
                slot="toolbar",
                llm_param="reasoning",
                cast=_build_reasoning,
            ),
            Control(
                id="domain-select",
                html="",  # pills rendered inside TOOLBAR_HTML; chatInteraction fires with this id
                slot="toolbar",
                llm_param="tools",
                cast=_build_tools,
            ),
            Control(
                id="search-mode",
                html="",  # rendered inside TOOLBAR_HTML above
                slot="toolbar",
                llm_param="tool_choice",
                cast=None,
            ),
        ]
    ),
)

if __name__ == "__main__":
    app.run()
