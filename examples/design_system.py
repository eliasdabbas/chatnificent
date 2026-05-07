# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "chatnificent",
#     "openai",
# ]
# ///
"""
Design system playground — Theme × Brand × Font × Scale × Mode
===============================================================

This example demonstrates Chatnificent's token system end-to-end. Five
orthogonal controls live in the toolbar; pick any combination and the whole
interface re-skins through CSS variables.

- **Theme** — a pack of token overrides (surfaces, borders, text, sidebar).
  Most themes leave ``--brand`` alone.
- **Brand** — a single ``--brand`` token used on the wordmark, the active
  conversation marker, and the streaming dots.
- **Font** — a system-font stack (no @font-face, no Google Fonts).
- **Type scale** — a single ``--scale`` multiplier on ``html`` ``font-size``;
  every size in the stylesheet is in ``rem`` so one number scales it all.
- **Mode** — the built-in sun/moon toggle flips ``html[data-theme]``; each
  theme ships matching ``light`` and ``dark`` variants.

A collapsible **Elements gallery** lives below the pickers and renders every
native HTML control (buttons, inputs, selects, dialogs, meters, etc.) styled
by the active token set — useful for spotting weak spots in a theme.

Running
-------
::

    uv run examples/design_system.py

Then open http://127.0.0.1:7777, flip controls, and click *Gallery* to audit
every styled element against the active tokens.
"""

import json

import chatnificent as chat

###############################################################################
# Elements gallery markup — rendered inside the collapsible #element-gallery
# panel. Every section showcases a different group of native HTML controls so
# we can audit how the active token set treats them.
###############################################################################

GALLERY_BODY = """
<main class="gallery">
  <section class="gallery-section">
    <h2>Buttons</h2>
    <p class="gallery-caption">Variants via <code>data-variant</code>; sizes via <code>data-size</code>.</p>
    <div class="gallery-row">
      <button>Primary</button>
      <button data-variant="secondary">Secondary</button>
      <button data-variant="ghost">Ghost</button>
      <button data-variant="danger">Danger</button>
      <button disabled>Disabled</button>
    </div>
    <div class="gallery-row">
      <button data-size="sm">Small</button>
      <button>Medium</button>
      <button data-size="lg">Large</button>
    </div>
  </section>
  <section class="gallery-section">
    <h2>Text inputs &amp; textarea</h2>
    <p class="gallery-caption">Twelve input types share one ruleset; <code>textarea</code> joins it.</p>
    <div class="gallery-grid">
      <label class="field"><span>Name</span><input type="text" placeholder="Jane Doe"></label>
      <label class="field"><span>Email</span><input type="email" placeholder="jane@example.com"></label>
      <label class="field"><span>Password</span><input type="password" value="hunter2"></label>
      <label class="field"><span>Tokens</span><input type="number" value="1024" min="1" max="32000"></label>
      <label class="field"><span>Search</span><input type="search" placeholder="Search…"></label>
      <label class="field"><span>Disabled</span><input type="text" value="read only" disabled></label>
      <label class="field"><span>Date</span><input type="date" value="2026-04-30"></label>
      <label class="field"><span>Time</span><input type="time" value="14:30"></label>
    </div>
    <label class="field" style="margin-top:14px"><span>System prompt</span>
      <textarea rows="3" placeholder="You are a helpful assistant…">You are a helpful assistant. Answer concisely and cite sources.</textarea>
    </label>
  </section>
  <section class="gallery-section">
    <h2>Select</h2>
    <p class="gallery-caption">Pure-CSS chevron — no SVG, no font, works offline.</p>
    <div class="gallery-grid">
      <label class="field"><span>Model</span>
        <select>
          <option>gpt-5-mini</option>
          <option>gpt-5</option>
          <option>claude-opus-4</option>
          <option>gemini-2.5-pro</option>
        </select>
      </label>
      <label class="field"><span>Provider (optgroups)</span>
        <select>
          <optgroup label="OpenAI"><option>gpt-5</option><option>gpt-5-mini</option></optgroup>
          <optgroup label="Anthropic"><option>claude-opus-4</option><option>claude-sonnet-4</option></optgroup>
          <optgroup label="Google"><option>gemini-2.5-pro</option></optgroup>
        </select>
      </label>
      <label class="field"><span>Disabled</span><select disabled><option>locked</option></select></label>
    </div>
  </section>
  <section class="gallery-section">
    <h2>Checkbox &amp; radio</h2>
    <p class="gallery-caption"><code>appearance: none</code> with a custom-drawn checkmark via SVG mask.</p>
    <div class="gallery-row">
      <label class="check"><input type="checkbox" checked> Streaming</label>
      <label class="check"><input type="checkbox"> Tools</label>
      <label class="check"><input type="checkbox" checked> Persist</label>
      <label class="check"><input type="checkbox" disabled> Disabled</label>
      <label class="check"><input type="checkbox" disabled checked> Disabled checked</label>
    </div>
    <div class="gallery-row">
      <label class="check"><input type="radio" name="g-r1" checked> Auto</label>
      <label class="check"><input type="radio" name="g-r1"> Manual</label>
      <label class="check"><input type="radio" name="g-r1"> Off</label>
      <label class="check"><input type="radio" name="g-r1" disabled> Disabled</label>
    </div>
  </section>
  <section class="gallery-section">
    <h2>Range, color, file</h2>
    <p class="gallery-caption">Cross-browser thumbs and tracks; file picker button uses E1's pill shape.</p>
    <div class="gallery-grid">
      <label class="field"><span>Temperature <output id="temp-val" style="color:var(--text-muted);font-weight:400">0.7</output></span>
        <input type="range" min="0" max="2" step="0.1" value="0.7"
               oninput="document.getElementById('temp-val').textContent=this.value">
      </label>
      <label class="field"><span>Accent</span><input type="color" value="#14b8a6"></label>
      <label class="field"><span>Upload</span><input type="file"></label>
    </div>
  </section>
  <section class="gallery-section">
    <h2>Progress &amp; meter</h2>
    <p class="gallery-caption">Meter exposes three bands: optimum (green), suboptimum (amber), even-less-good (red).</p>
    <div class="gallery-row">
      <progress value="32" max="100"></progress>
      <progress value="68" max="100"></progress>
      <progress max="100"></progress>
    </div>
    <div class="gallery-row" style="margin-top:14px">
      <meter value="2" min="0" max="10" low="3" high="7" optimum="5"></meter>
      <meter value="6" min="0" max="10" low="3" high="7" optimum="5"></meter>
      <meter value="9" min="0" max="10" low="3" high="7" optimum="5"></meter>
    </div>
  </section>
  <section class="gallery-section">
    <h2>Disclosure &amp; dialog</h2>
    <p class="gallery-caption">Native <code>&lt;details&gt;</code>; <code>&lt;dialog&gt;</code> with a blurred backdrop.</p>
    <details>
      <summary>Show advanced settings</summary>
      <p>Hidden by default. The chevron rotates on <code>[open]</code>. No JavaScript.</p>
    </details>
    <details open style="margin-top:10px">
      <summary>Already open</summary>
      <p>Use the <code>open</code> attribute to start expanded.</p>
    </details>
    <div class="gallery-row" style="margin-top:14px">
      <button onclick="document.getElementById('demo-dialog').showModal()">Open dialog</button>
      <button data-variant="ghost" onclick="document.getElementById('demo-dialog').showModal()">Open from ghost</button>
    </div>
    <dialog id="demo-dialog">
      <h3 style="margin-bottom:8px;font-weight:700">Confirm action</h3>
      <p style="color:var(--text-secondary);margin-bottom:18px;line-height:1.5">
        Are you sure you want to delete this conversation? This action cannot be undone.
      </p>
      <form method="dialog" class="gallery-row" style="justify-content:flex-end;margin:0">
        <button data-variant="ghost" value="cancel">Cancel</button>
        <button data-variant="danger" value="ok">Delete</button>
      </form>
    </dialog>
  </section>
  <section class="gallery-section">
    <h2>Inline text</h2>
    <p class="gallery-caption">
      Press <kbd>⌘</kbd>+<kbd>K</kbd> to open. Use <code>&lt;mark&gt;</code> to <mark>highlight</mark>
      key terms, and <abbr title="Cascading Style Sheets">CSS</abbr> for tooltips on abbreviations.
      Inline <code>code</code> uses the mono font.
    </p>
  </section>
  <section class="gallery-section">
    <h2>Block &amp; form layout</h2>
    <p class="gallery-caption"><code>&lt;hr&gt;</code>, <code>&lt;blockquote&gt;</code>, and a <code>&lt;fieldset&gt;</code> with a <code>&lt;legend&gt;</code>.</p>
    <hr>
    <blockquote>
      Minimally complete. Maximally hackable. — every element ships from a token,
      so re-theming is one CSS variable away.
    </blockquote>
    <fieldset style="margin-top:14px">
      <legend>Notification preferences</legend>
      <div class="gallery-row" style="margin:0">
        <label class="check"><input type="checkbox" checked> Email</label>
        <label class="check"><input type="checkbox"> SMS</label>
        <label class="check"><input type="checkbox" checked> Push</label>
      </div>
    </fieldset>
  </section>
</main>
"""

THEMES = {
    "default": {"light": {}, "dark": {}},
    "carbon": {
        # strict monochrome — paper / ink
        "light": {
            "--bg": "#ffffff",
            "--bg-elev": "#fafafa",
            "--surface": "#ffffff",
            "--border": "#e0e0e0",
            "--border-strong": "#a0a0a0",
            "--text": "#0a0a0a",
            "--text-secondary": "#4a4a4a",
            "--text-muted": "#8a8a8a",
            "--accent": "#000000",
            "--accent-hover": "#2a2a2a",
            "--accent-subtle": "rgba(0,0,0,0.05)",
            "--accent-ring": "rgba(0,0,0,0.12)",
            "--user-bubble": "#000000",
            "--user-text": "#ffffff",
            "--assistant-bg": "#f5f5f5",
            "--assistant-text": "#0a0a0a",
            "--sidebar-bg": "#f5f5f5",
            "--btn-text": "#ffffff",
        },
        "dark": {
            "--bg": "#050505",
            "--bg-elev": "#161616",
            "--surface": "#0a0a0a",
            "--border": "#262626",
            "--border-strong": "#404040",
            "--text": "#f5f5f5",
            "--text-secondary": "#b0b0b0",
            "--text-muted": "#6e6e6e",
            "--accent": "#ffffff",
            "--accent-hover": "#d0d0d0",
            "--accent-subtle": "rgba(255,255,255,0.05)",
            "--accent-ring": "rgba(255,255,255,0.12)",
            "--user-bubble": "#f5f5f5",
            "--user-text": "#050505",
            "--assistant-bg": "#161616",
            "--assistant-text": "#f5f5f5",
            "--sidebar-bg": "#0d0d0d",
            "--btn-text": "#050505",
        },
    },
    "sunlight": {
        # warm cream paper / candlelit warm wood
        "light": {
            "--bg": "#fdf6e3",
            "--bg-elev": "#fffaf0",
            "--surface": "#fdf6e3",
            "--border": "#ede0bb",
            "--border-strong": "#c9b886",
            "--text": "#3a2c14",
            "--text-secondary": "#7a6b4a",
            "--text-muted": "#a89878",
            "--accent": "#3a2c14",
            "--accent-hover": "#1f1808",
            "--accent-subtle": "rgba(58,44,20,0.06)",
            "--accent-ring": "rgba(58,44,20,0.18)",
            "--user-bubble": "#3a2c14",
            "--user-text": "#fdf6e3",
            "--assistant-bg": "#f5ecc7",
            "--assistant-text": "#3a2c14",
            "--sidebar-bg": "#f5ecc7",
            "--btn-text": "#fdf6e3",
        },
        "dark": {
            "--bg": "#1a1408",
            "--bg-elev": "#241c0e",
            "--surface": "#1a1408",
            "--border": "#3a2e18",
            "--border-strong": "#5c4a28",
            "--text": "#f5e6c8",
            "--text-secondary": "#c9b886",
            "--text-muted": "#8a7550",
            "--accent": "#f5e6c8",
            "--accent-hover": "#ffffff",
            "--accent-subtle": "rgba(245,230,200,0.08)",
            "--accent-ring": "rgba(245,230,200,0.20)",
            "--user-bubble": "#f5e6c8",
            "--user-text": "#1a1408",
            "--assistant-bg": "#241c0e",
            "--assistant-text": "#f5e6c8",
            "--sidebar-bg": "#1f1810",
            "--btn-text": "#1a1408",
        },
    },
    "winter": {
        # frost paper / arctic night
        "light": {
            "--bg": "#eef2f7",
            "--bg-elev": "#ffffff",
            "--surface": "#eef2f7",
            "--border": "#d3dce6",
            "--border-strong": "#a8b8cc",
            "--text": "#0f1c2e",
            "--text-secondary": "#4a5b73",
            "--text-muted": "#8090a8",
            "--accent": "#0f1c2e",
            "--accent-hover": "#050d18",
            "--accent-subtle": "rgba(15,28,46,0.06)",
            "--accent-ring": "rgba(15,28,46,0.18)",
            "--user-bubble": "#0f1c2e",
            "--user-text": "#eef2f7",
            "--assistant-bg": "#e3eaf3",
            "--assistant-text": "#0f1c2e",
            "--sidebar-bg": "#e3eaf3",
            "--btn-text": "#eef2f7",
        },
        "dark": {
            "--bg": "#0a1220",
            "--bg-elev": "#131d2e",
            "--surface": "#0a1220",
            "--border": "#1f2d44",
            "--border-strong": "#3a4d6b",
            "--text": "#e6edf7",
            "--text-secondary": "#a8b8cc",
            "--text-muted": "#6e7d96",
            "--accent": "#e6edf7",
            "--accent-hover": "#ffffff",
            "--accent-subtle": "rgba(230,237,247,0.07)",
            "--accent-ring": "rgba(230,237,247,0.20)",
            "--user-bubble": "#e6edf7",
            "--user-text": "#0a1220",
            "--assistant-bg": "#131d2e",
            "--assistant-text": "#e6edf7",
            "--sidebar-bg": "#0d1626",
            "--btn-text": "#0a1220",
        },
    },
    "forest": {
        # sage paper / mossy night
        "light": {
            "--bg": "#eef3ed",
            "--bg-elev": "#ffffff",
            "--surface": "#eef3ed",
            "--border": "#c8d3c5",
            "--border-strong": "#94a692",
            "--text": "#1a2e1f",
            "--text-secondary": "#4a6553",
            "--text-muted": "#8a9c8d",
            "--accent": "#1a2e1f",
            "--accent-hover": "#0c1810",
            "--accent-subtle": "rgba(26,46,31,0.06)",
            "--accent-ring": "rgba(26,46,31,0.18)",
            "--user-bubble": "#1a2e1f",
            "--user-text": "#eef3ed",
            "--assistant-bg": "#e2eadf",
            "--assistant-text": "#1a2e1f",
            "--sidebar-bg": "#e2eadf",
            "--btn-text": "#eef3ed",
        },
        "dark": {
            "--bg": "#0d1610",
            "--bg-elev": "#16201a",
            "--surface": "#0d1610",
            "--border": "#243028",
            "--border-strong": "#3d5040",
            "--text": "#e2eadf",
            "--text-secondary": "#94a692",
            "--text-muted": "#6a7a6c",
            "--accent": "#e2eadf",
            "--accent-hover": "#ffffff",
            "--accent-subtle": "rgba(226,234,223,0.07)",
            "--accent-ring": "rgba(226,234,223,0.20)",
            "--user-bubble": "#e2eadf",
            "--user-text": "#0d1610",
            "--assistant-bg": "#16201a",
            "--assistant-text": "#e2eadf",
            "--sidebar-bg": "#0f1812",
            "--btn-text": "#0d1610",
        },
    },
    "plum": {
        # pale plum / deep aubergine
        "light": {
            "--bg": "#f5edf3",
            "--bg-elev": "#ffffff",
            "--surface": "#f5edf3",
            "--border": "#d8c8d2",
            "--border-strong": "#a890a4",
            "--text": "#2a1929",
            "--text-secondary": "#6b4f68",
            "--text-muted": "#a08899",
            "--accent": "#2a1929",
            "--accent-hover": "#160c16",
            "--accent-subtle": "rgba(42,25,41,0.06)",
            "--accent-ring": "rgba(42,25,41,0.18)",
            "--user-bubble": "#2a1929",
            "--user-text": "#f5edf3",
            "--assistant-bg": "#ede2eb",
            "--assistant-text": "#2a1929",
            "--sidebar-bg": "#ede2eb",
            "--btn-text": "#f5edf3",
        },
        "dark": {
            "--bg": "#1a0f1a",
            "--bg-elev": "#251828",
            "--surface": "#1a0f1a",
            "--border": "#382a3a",
            "--border-strong": "#5a4a5e",
            "--text": "#f0e5ee",
            "--text-secondary": "#b89cb4",
            "--text-muted": "#806a7e",
            "--accent": "#f0e5ee",
            "--accent-hover": "#ffffff",
            "--accent-subtle": "rgba(240,229,238,0.07)",
            "--accent-ring": "rgba(240,229,238,0.20)",
            "--user-bubble": "#f0e5ee",
            "--user-text": "#1a0f1a",
            "--assistant-bg": "#251828",
            "--assistant-text": "#f0e5ee",
            "--sidebar-bg": "#1d1220",
            "--btn-text": "#1a0f1a",
        },
    },
    # Solarized-light and Solarized-dark map to the same canonical pair (Ethan
    # Schoonover). Picking one selects the default landing mode; the toggle
    # reveals the other half of the pair.
    "solarized-light": {
        "light": {
            "--bg": "#fdf6e3",
            "--bg-elev": "#eee8d5",
            "--surface": "#fdf6e3",
            "--border": "#ddd6c1",
            "--border-strong": "#93a1a1",
            "--text": "#073642",
            "--text-secondary": "#586e75",
            "--text-muted": "#93a1a1",
            "--accent": "#073642",
            "--accent-hover": "#002b36",
            "--accent-subtle": "rgba(7,54,66,0.06)",
            "--accent-ring": "rgba(7,54,66,0.18)",
            "--user-bubble": "#073642",
            "--user-text": "#fdf6e3",
            "--assistant-bg": "#eee8d5",
            "--assistant-text": "#073642",
            "--sidebar-bg": "#eee8d5",
            "--btn-text": "#fdf6e3",
        },
        "dark": {
            "--bg": "#002b36",
            "--bg-elev": "#073642",
            "--surface": "#002b36",
            "--border": "#103a45",
            "--border-strong": "#586e75",
            "--text": "#eee8d5",
            "--text-secondary": "#93a1a1",
            "--text-muted": "#657b83",
            "--accent": "#eee8d5",
            "--accent-hover": "#fdf6e3",
            "--accent-subtle": "rgba(238,232,213,0.06)",
            "--accent-ring": "rgba(238,232,213,0.18)",
            "--user-bubble": "#eee8d5",
            "--user-text": "#002b36",
            "--assistant-bg": "#073642",
            "--assistant-text": "#eee8d5",
            "--sidebar-bg": "#073642",
            "--btn-text": "#002b36",
        },
    },
    "solarized-dark": {
        "light": {
            "--bg": "#fdf6e3",
            "--bg-elev": "#eee8d5",
            "--surface": "#fdf6e3",
            "--border": "#ddd6c1",
            "--border-strong": "#93a1a1",
            "--text": "#073642",
            "--text-secondary": "#586e75",
            "--text-muted": "#93a1a1",
            "--accent": "#073642",
            "--accent-hover": "#002b36",
            "--accent-subtle": "rgba(7,54,66,0.06)",
            "--accent-ring": "rgba(7,54,66,0.18)",
            "--user-bubble": "#073642",
            "--user-text": "#fdf6e3",
            "--assistant-bg": "#eee8d5",
            "--assistant-text": "#073642",
            "--sidebar-bg": "#eee8d5",
            "--btn-text": "#fdf6e3",
        },
        "dark": {
            "--bg": "#002b36",
            "--bg-elev": "#073642",
            "--surface": "#002b36",
            "--border": "#103a45",
            "--border-strong": "#586e75",
            "--text": "#eee8d5",
            "--text-secondary": "#93a1a1",
            "--text-muted": "#657b83",
            "--accent": "#eee8d5",
            "--accent-hover": "#fdf6e3",
            "--accent-subtle": "rgba(238,232,213,0.06)",
            "--accent-ring": "rgba(238,232,213,0.18)",
            "--user-bubble": "#eee8d5",
            "--user-text": "#002b36",
            "--assistant-bg": "#073642",
            "--assistant-text": "#eee8d5",
            "--sidebar-bg": "#073642",
            "--btn-text": "#002b36",
        },
    },
    "dim": {
        # cool slate paper (Bluestone) / canonical Twitter-dim
        "light": {
            "--bg": "#e8edf2",
            "--bg-elev": "#f4f7fa",
            "--surface": "#e8edf2",
            "--border": "#cdd5de",
            "--border-strong": "#8899a6",
            "--text": "#15202b",
            "--text-secondary": "#4a5d72",
            "--text-muted": "#7a8a9a",
            "--accent": "#15202b",
            "--accent-hover": "#050a10",
            "--accent-subtle": "rgba(21,32,43,0.06)",
            "--accent-ring": "rgba(21,32,43,0.18)",
            "--user-bubble": "#15202b",
            "--user-text": "#e8edf2",
            "--assistant-bg": "#dde4ec",
            "--assistant-text": "#15202b",
            "--sidebar-bg": "#dde4ec",
            "--btn-text": "#e8edf2",
        },
        "dark": {
            "--bg": "#15202b",
            "--bg-elev": "#1c2733",
            "--surface": "#15202b",
            "--border": "#2a3744",
            "--border-strong": "#4a5d72",
            "--text": "#f7f9fa",
            "--text-secondary": "#8899a6",
            "--text-muted": "#5c7080",
            "--accent": "#f7f9fa",
            "--accent-hover": "#ffffff",
            "--accent-subtle": "rgba(247,249,250,0.06)",
            "--accent-ring": "rgba(247,249,250,0.18)",
            "--user-bubble": "#f7f9fa",
            "--user-text": "#15202b",
            "--assistant-bg": "#192734",
            "--assistant-text": "#f7f9fa",
            "--sidebar-bg": "#192734",
            "--btn-text": "#15202b",
        },
    },
    "newsprint": {
        # bold print on cream / inkwell night with sharp white borders
        "light": {
            "--bg": "#fbf6e9",
            "--bg-elev": "#ffffff",
            "--surface": "#fbf6e9",
            "--border": "#3a3a3a",
            "--border-strong": "#000000",
            "--text": "#000000",
            "--text-secondary": "#1a1a1a",
            "--text-muted": "#4a4a4a",
            "--accent": "#000000",
            "--accent-hover": "#2a2a2a",
            "--accent-subtle": "rgba(0,0,0,0.06)",
            "--accent-ring": "rgba(0,0,0,0.18)",
            "--user-bubble": "#000000",
            "--user-text": "#fbf6e9",
            "--assistant-bg": "#f0e9d2",
            "--assistant-text": "#000000",
            "--sidebar-bg": "#f0e9d2",
            "--btn-text": "#fbf6e9",
        },
        "dark": {
            "--bg": "#0a0a08",
            "--bg-elev": "#141412",
            "--surface": "#0a0a08",
            "--border": "#c8c8c0",
            "--border-strong": "#ffffff",
            "--text": "#ffffff",
            "--text-secondary": "#e8e8e0",
            "--text-muted": "#b0b0a8",
            "--accent": "#ffffff",
            "--accent-hover": "#f0f0e8",
            "--accent-subtle": "rgba(255,255,255,0.08)",
            "--accent-ring": "rgba(255,255,255,0.22)",
            "--user-bubble": "#ffffff",
            "--user-text": "#0a0a08",
            "--assistant-bg": "#141412",
            "--assistant-text": "#ffffff",
            "--sidebar-bg": "#0f0f0d",
            "--btn-text": "#0a0a08",
        },
    },
}

THEME_ORDER = [
    ("default", "Default"),
    ("carbon", "Carbon"),
    ("sunlight", "Sunlight"),
    ("solarized-light", "Solarized Light"),
    ("solarized-dark", "Solarized Dark"),
    ("dim", "Dim"),
    ("newsprint", "Newsprint"),
]

WELCOME = """
## Design system playground

Mix and match between theme, brand, font, type scale, mode (sun/moon)

Or try a chip below to see how chat copy renders against the active token set.
<br>
<div id="suggestions">
  <button class="suggestion" data-insert-prompt="Show me a long markdown response with headings, a bulleted list, a numbered list, a blockquote, and a small code block, so I can audit the type scale.">
    <span class="suggestion-label">Type scale</span>
    <span class="suggestion-text">Stress-test headings, lists, code</span>
  </button>
  <button class="suggestion" data-insert-prompt="Reply with a markdown table that has exactly 4 columns — Theme, Mood, Accent, Surface — and 3 rows for Warm, Cool, and Monochrome. Make sure the header separator has 4 dashes-segments so the table is well-formed.">
    <span class="suggestion-label">Tables</span>
    <span class="suggestion-text">Four-column theme comparison</span>
  </button>
  <button class="suggestion" data-insert-prompt="Write a multi-paragraph reply about color tokens — one paragraph each on surfaces, borders, text, and accents — so I can see how the wall of text breathes.">
    <span class="suggestion-label">Density</span>
    <span class="suggestion-text">Walls of paragraph text</span>
  </button>
  <button class="suggestion" data-insert-prompt="Give me a Python snippet (~15 lines) using a class, a decorator, and a context manager so I can read the mono font in code blocks.">
    <span class="suggestion-label">Code</span>
    <span class="suggestion-text">Python with class + decorator</span>
  </button>
</div>
"""


# Brand color tuples for use in pickers, etc.
BRANDS = [
    ("Teal", "#0f766e"),
    ("Indigo", "#5b5cff"),
    ("Amber", "#f59e0b"),
    ("Magenta", "#ec4899"),
    ("Teal bright", "#14b8a6"),
    ("Indigo soft", "#6366f1"),
    ("Gold", "#d4a017"),
    ("Coral", "#fb7185"),
    ("Slate", "#64748b"),
    ("Sage", "#84cc16"),
    ("Plum", "#a855f7"),
    ("Rose", "#f43f5e"),
    ("Mono", "MONO"),
]

# Font choices — every option resolves on stock OS installs (macOS, Windows,
# iOS, Android, ChromeOS, major Linux). No @font-face, no Google Fonts, no
# downloads. Each entry pairs a label with the full fallback chain.
FONTS = [
    (
        "Trebuchet (default)",
        "'Trebuchet MS', 'Lucida Grande', 'Lucida Sans Unicode', "
        "ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif",
    ),
    (
        "System UI (native everywhere)",
        "ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "
        "'Segoe UI', Roboto, sans-serif",
    ),
    (
        "Verdana (universal, screen-tuned)",
        "Verdana, Geneva, Tahoma, sans-serif",
    ),
    (
        "Georgia (serif, distinctive)",
        "Georgia, 'Times New Roman', Times, serif",
    ),
    (
        "Tahoma (compact humanist)",
        "Tahoma, 'Lucida Sans Unicode', sans-serif",
    ),
]

# Type scale slider — drives a single --scale multiplier on html's font-size.
# Because every size in the stylesheet is in rem, one multiplier scales the
# whole interface uniformly. No presets needed; users tune to taste.
SCALE_MIN = 0.80
SCALE_MAX = 1.30
SCALE_STEP = 0.05
SCALE_DEFAULT = 1.00


# ---------------------------------------------------------------------------
# Picker HTML + JS — injected into the "toolbar" slot.
# ---------------------------------------------------------------------------

THEME_OPTIONS = "".join(
    f'<option value="{key}">{label}</option>' for key, label in THEME_ORDER
)
BRAND_OPTIONS = "".join(
    f'<option value="{value}">{name}</option>' for name, value in BRANDS
)
FONT_OPTIONS = "".join(
    f'<option value="{i}">{label}</option>' for i, (label, _stack) in enumerate(FONTS)
)

FONT_STACKS_JSON = json.dumps([stack for _label, stack in FONTS])

PICKER_HTML = f"""
<style>
  /* Type-scale slider \u2014 selects a point in a range, so we don't paint the
     "from-min" portion. Track is a thin neutral bar; thumb is a tall slim
     pill in brand color, with a subtle inset notch. Distinct but quiet. */
  #scale-picker {{
    -webkit-appearance: none;
    appearance: none;
    background: transparent;
    height: 24px;
    padding: 0;
    margin: 0;
    cursor: pointer;
    outline: none;
  }}
  #scale-picker::-webkit-slider-runnable-track {{
    height: 2px;
    background: var(--border-strong);
    border: none;
    border-radius: 0;
  }}
  #scale-picker::-moz-range-track {{
    height: 2px;
    background: var(--border-strong);
    border: none;
    border-radius: 0;
  }}
  #scale-picker::-moz-range-progress {{
    background: var(--border-strong);
    height: 2px;
  }}
  #scale-picker::-webkit-slider-thumb {{
    -webkit-appearance: none;
    appearance: none;
    width: 6px;
    height: 18px;
    border-radius: 2px;
    background: var(--brand);
    border: none;
    margin-top: -8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.18);
    transition: transform 0.12s ease;
  }}
  #scale-picker::-moz-range-thumb {{
    width: 6px;
    height: 18px;
    border-radius: 2px;
    background: var(--brand);
    border: none;
    box-shadow: 0 1px 3px rgba(0,0,0,0.18);
    transition: transform 0.12s ease;
  }}
  #scale-picker:active::-webkit-slider-thumb {{ transform: scaleY(1.1); }}
  #scale-picker:active::-moz-range-thumb {{ transform: scaleY(1.1); }}
  #scale-picker:focus-visible::-webkit-slider-thumb {{
    box-shadow: 0 0 0 3px var(--brand-ring);
  }}
  #scale-picker:focus-visible::-moz-range-thumb {{
    box-shadow: 0 0 0 3px var(--brand-ring);
  }}
</style>
<div id="theme-pickers" style="display:grid;grid-template-columns:repeat(2, minmax(0, 1fr));gap:14px 20px;align-items:end;padding:12px 20px;max-width:560px;margin:0 auto;width:100%;font-size:0.85rem;">
  <label style="display:flex;flex-direction:column;gap:4px;min-width:0;">
    <span style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em;color:var(--text-muted);font-weight:700;">Theme</span>
    <select id="theme-picker" onchange="setTheme(this.value)" style="padding:8px 12px;border-radius:10px;border:1px solid var(--border);background:var(--bg-elev);color:var(--text);font-family:inherit;font-size:0.9rem;cursor:pointer;width:100%;">
      {THEME_OPTIONS}
    </select>
  </label>
  <label style="display:flex;flex-direction:column;gap:4px;min-width:0;">
    <span style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em;color:var(--text-muted);font-weight:700;display:inline-flex;align-items:center;gap:6px;">
      Brand
      <span id="brand-swatch" aria-hidden="true" style="display:inline-block;width:10px;height:10px;border-radius:50%;background:var(--brand);box-shadow:0 0 0 1px var(--border-strong);"></span>
    </span>
    <select id="brand-picker" onchange="setBrand(this.value)" style="padding:8px 12px;border-radius:10px;border:1px solid var(--border);background:var(--bg-elev);color:var(--text);font-family:inherit;font-size:0.9rem;cursor:pointer;width:100%;">
      {BRAND_OPTIONS}
    </select>
  </label>
  <label style="display:flex;flex-direction:column;gap:4px;min-width:0;">
    <span style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em;color:var(--text-muted);font-weight:700;">Font</span>
    <select id="font-picker" onchange="setFont(this.value)" style="padding:8px 12px;border-radius:10px;border:1px solid var(--border);background:var(--bg-elev);color:var(--text);font-family:inherit;font-size:0.9rem;cursor:pointer;width:100%;">
      {FONT_OPTIONS}
    </select>
  </label>
  <div style="display:flex;flex-direction:column;gap:4px;min-width:0;">
    <span style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em;color:var(--text-muted);font-weight:700;display:inline-flex;align-items:center;gap:8px;">
      Type scale
      <span id="scale-readout" style="font-family:var(--mono-font);font-size:0.72rem;color:var(--text-secondary);font-weight:500;letter-spacing:0;text-transform:none;">{SCALE_DEFAULT:.2f}×</span>
    </span>
    <div style="display:flex;align-items:center;gap:8px;height:36px;">
      <button type="button" onclick="bumpScale(-1)" aria-label="Decrease type scale" style="flex:none;width:28px;height:28px;padding:0;border-radius:8px;border:1px solid var(--border);background:var(--bg-elev);color:var(--text);font-family:inherit;font-size:0.8rem;font-weight:700;cursor:pointer;line-height:1;box-shadow:none;">A−</button>
      <input id="scale-picker" type="range" min="{SCALE_MIN}" max="{SCALE_MAX}" step="{SCALE_STEP}" value="{SCALE_DEFAULT}" oninput="setScale(this.value)" style="flex:1;min-width:0;">
      <button type="button" onclick="bumpScale(1)" aria-label="Increase type scale" style="flex:none;width:28px;height:28px;padding:0;border-radius:8px;border:1px solid var(--border);background:var(--bg-elev);color:var(--text);font-family:inherit;font-size:1rem;font-weight:700;cursor:pointer;line-height:1;box-shadow:none;">A+</button>
    </div>
  </div>
</div>
<script>
(function () {{
  var THEMES = {json.dumps(THEMES)};
  var FONT_STACKS = {FONT_STACKS_JSON};
  var SCALE_MIN = {SCALE_MIN};
  var SCALE_MAX = {SCALE_MAX};
  var SCALE_STEP = {SCALE_STEP};

  var ALL_KEYS = new Set();
  Object.keys(THEMES).forEach(function (t) {{
    ["light", "dark"].forEach(function (mode) {{
      var v = (THEMES[t] || {{}})[mode] || {{}};
      Object.keys(v).forEach(function (k) {{ ALL_KEYS.add(k); }});
    }});
  }});

  function currentMode() {{
    return document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light";
  }}

  window.setTheme = function (name) {{
    var entry = THEMES[name] || {{"light": {{}}, "dark": {{}}}};
    var variant = entry[currentMode()] || {{}};
    var root = document.documentElement;
    ALL_KEYS.forEach(function (k) {{
      if (k in variant) {{
        root.style.setProperty(k, variant[k]);
      }} else {{
        root.style.removeProperty(k);
      }}
    }});
    var bp = document.getElementById("brand-picker");
    if (bp && bp.value) window.setBrand(bp.value);
  }};

  function withAlpha(hex, alphaHex) {{
    if (!/^#[0-9a-fA-F]{{6}}$/.test(hex)) return hex;
    return hex + alphaHex;
  }}

  window.setBrand = function (color) {{
    var root = document.documentElement;
    if (color === "MONO") {{
      root.style.setProperty("--brand", "var(--accent)");
      root.style.setProperty("--brand-hover", "var(--accent-hover)");
      root.style.setProperty("--brand-subtle", "var(--accent-subtle)");
      root.style.setProperty("--brand-ring", "var(--accent-ring)");
    }} else {{
      root.style.setProperty("--brand", color);
      root.style.setProperty("--brand-hover", color);
      root.style.setProperty("--brand-subtle", withAlpha(color, "1a"));
      root.style.setProperty("--brand-ring", withAlpha(color, "47"));
    }}
  }};

  window.setFont = function (idx) {{
    var stack = FONT_STACKS[parseInt(idx, 10)] || FONT_STACKS[0];
    document.documentElement.style.setProperty("--font", stack);
  }};

  // Type scale: one --scale multiplier on html font-size. Because every
  // sizing rule is in rem, a single number scales the whole interface.
  window.setScale = function (value) {{
    var v = parseFloat(value);
    if (isNaN(v)) return;
    if (v < SCALE_MIN) v = SCALE_MIN;
    if (v > SCALE_MAX) v = SCALE_MAX;
    document.documentElement.style.setProperty("--scale", v);
    var slider = document.getElementById("scale-picker");
    if (slider && parseFloat(slider.value) !== v) slider.value = v;
    var readout = document.getElementById("scale-readout");
    if (readout) readout.textContent = v.toFixed(2) + "×";
  }};

  window.bumpScale = function (direction) {{
    var slider = document.getElementById("scale-picker");
    var current = slider ? parseFloat(slider.value) : 1;
    window.setScale(current + direction * SCALE_STEP);
  }};

  // Sun/moon flips html[data-theme] directly — re-apply the active theme's
  // matching variant so Theme × Mode stays orthogonal.
  var observer = new MutationObserver(function () {{
    var tp = document.getElementById("theme-picker");
    window.setTheme(tp ? tp.value : "default");
  }});
  observer.observe(document.documentElement, {{
    attributes: true,
    attributeFilter: ["data-theme"]
  }});
}})();
</script>
"""

# Visual-only control: the pickers mutate CSS custom properties via JS and
# never POST to /api/interactions, so no llm_param is needed.
PICKER_CONTROL = chat.layout.Control(
    id="theme-pickers",
    html=PICKER_HTML,
    slot="toolbar",
)


# Gallery toggle + collapsible gallery, centered to the chat column.
# We wrap both in a #gallery-wrap container that mirrors the chat area's
# max-width via the --chat-max-w token, so the gallery feels like a natural
# extension of the page rather than a left-aligned banner. The collapsible
# panel itself is a scroll container so a long gallery never pushes the
# input bar off-screen.
GALLERY_TOGGLE_CONTROL = chat.layout.Control(
    id="gallery-toggle-toolbar",
    slot="toolbar",
    html="""
<style>
#gallery-wrap {
  max-width: var(--chat-max-w);
  margin: 24px auto 0;
  padding: 0 24px;
  width: 100%;
  box-sizing: border-box;
  text-align: center;
}
#gallery-toggle-btn:focus-visible {
  border-color: var(--accent-hover);
  box-shadow: 0 0 0 2px var(--accent-ring);
}
#gallery-chevron {
  transition: transform 0.22s cubic-bezier(.4,0,.2,1);
}
#element-gallery {
  margin: 20px auto 0;
  max-width: var(--chat-max-w);
  width: 100%;
  box-sizing: border-box;
  border: 1px solid var(--border);
  border-radius: 14px;
  background: var(--bg-elev);
  text-align: left;
}
#element-gallery .gallery-header,
#element-gallery .gallery-pickers { display: none; }
#element-gallery .gallery {
  max-width: 100%;
  margin: 0;
  padding: 8px 24px 32px;
  display: flex;
  flex-direction: column;
  gap: 28px;
}
#element-gallery .gallery-section {
  border-top: 1px solid var(--border);
  padding-top: 20px;
}
#element-gallery .gallery-section:first-child { border-top: 0; padding-top: 8px; }
#element-gallery .gallery-section h2 {
  font-size: 1rem;
  font-weight: 700;
  margin: 0 0 4px;
  letter-spacing: -0.01em;
}
#element-gallery .gallery-caption {
  color: var(--text-muted);
  font-size: 0.85rem;
  margin: 0 0 14px;
}
#element-gallery .gallery-row {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  align-items: center;
  margin-bottom: 8px;
}
#element-gallery .gallery-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 14px;
}
#element-gallery .field {
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-width: 0;
}
#element-gallery .field > span:first-child {
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-muted);
  font-weight: 600;
}
#element-gallery .check {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-size: 0.92rem;
  cursor: pointer;
}
#element-gallery .check input { margin: 0; }
</style>
<div id="gallery-wrap">
  <button id="gallery-toggle-btn" type="button" style="
    display: inline-flex;
    align-items: center;
    gap: 8px;
    border: 1.5px solid var(--accent);
    background: var(--bg-elev);
    color: var(--accent);
    font-weight: 600;
    font-size: 1rem;
    border-radius: 10px;
    padding: 8px 16px;
    cursor: pointer;
    transition: border-color 0.18s, background 0.18s, color 0.18s;
    outline: none;" onclick="
    var el = document.getElementById('element-gallery');
    var chev = document.getElementById('gallery-chevron');
    var open = el.style.display !== 'block';
    el.style.display = open ? 'block' : 'none';
    if (chev) chev.style.transform = open ? 'rotate(90deg)' : 'rotate(0deg)';">
    <span style="font-weight:700;">Gallery</span>
    <svg id="gallery-chevron" viewBox="0 0 20 20" width="16" height="16" style="display:inline-block;vertical-align:middle;"><polyline points="7 8 13 10 7 12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
    <span style="font-size:0.95rem;font-weight:400;">Show every styled native element</span>
  </button>
  <div id="element-gallery" style="display:none;">
    """
    + GALLERY_BODY
    + """
  </div>
</div>
<script>
  // Relocate the gallery panel into the scrollable #messages area so the
  // page can scroll past a tall gallery to reveal the welcome copy and
  // suggestion chips below. The button stays in the toolbar; getElementById
  // is global so the toggle keeps working after the move.
  document.addEventListener('DOMContentLoaded', function () {
    var panel = document.getElementById('element-gallery');
    var messages = document.getElementById('messages');
    if (panel && messages && panel.parentNode !== messages) {
      messages.insertBefore(panel, messages.firstChild);
    }
  });
</script>
""",
)


# ---------------------------------------------------------------------------
# Wire it up — Chatnificent with DefaultLayout and gallery controls.
# ---------------------------------------------------------------------------

app = chat.Chatnificent(
    layout=chat.layout.DefaultLayout(
        brand="Chatnificent design system",
        slogan="Theme × Brand × Font × Scale × Mode",
        welcome_message=WELCOME,
        controls=[
            PICKER_CONTROL,
            GALLERY_TOGGLE_CONTROL,
        ],
    ),
)


if __name__ == "__main__":
    app.run()
