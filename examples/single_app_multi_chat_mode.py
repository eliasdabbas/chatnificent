# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "chatnificent[openai]",
# ]
# ///
"""
Single App, Multiple Chat Modes
===============================

One Chatnificent app. Four kinds of conversation. Each mode is a different
LLM call under the hood, but the user just picks a pill, types, and sends.

Modes
-----
- **Plain chat** (default) — your usual OpenAI Chat Completions, no studio.
- **TTS Studio** (indigo) — calls the OpenAI ``audio.speech`` endpoint.
  Choose model, voice, speed, format, and optional style instructions.
  Each turn renders an inline ``<audio>`` player with the generated speech.
- **Image Studio** (rose) — calls ``images.generate``. Pick size, quality,
  and background; one prompt → one image, persisted under
  ``/files/<user>/<convo>/images/<n>.png``.
- **Search Studio** (teal) — calls the Responses API with the built-in
  ``web_search`` tool. Replies arrive with inline ``[¹]`` citations and a
  numbered Sources block.

How a mode is locked to a conversation
--------------------------------------
The mode is captured the first time a conversation is saved, written to
``mode.txt`` next to the messages via the Store pillar's ``save_file``
sidecar API, and that file is canonical from then on. Mid-conversation
pill clicks update the UI for the next *new* chat — they cannot retarget
an existing one. Reload any old conversation and its studio reappears
automatically (the page fetches ``mode.txt`` on load).

What is hackable here
---------------------
Three pillar customizations carry the entire example:

- ``ModeRouter(LLM)`` — a delegating LLM proxy that forwards every method
  to the active mode's sub-LLM. The active mode is a ``contextvars.ContextVar``
  set per request — thread-safe under any server.
- ``ModeAwareEngine(Orchestrator)`` — resolves the mode from the sidecar
  (or, on the home page, from the user's last pill click), stamps
  ``mode.txt`` on first save, and rewrites inline base64 placeholders into
  stable ``/files/...`` URLs after each turn.
- ``MultiChatModeLayout(DefaultLayout)`` — drops the studio HTML above the
  input bar and injects the interactivity script after ``</body>``.

Running
-------
::

    export OPENAI_API_KEY="sk-..."
    uv run --script examples/single_app_multi_chat_mode.py

Then open http://127.0.0.1:7777 and try each pill. Conversations land in
``single_app_data/`` so they survive restarts.
"""

import base64
import contextvars
import re as _re
from functools import partial
from http import HTTPStatus
from http.server import HTTPServer

import chatnificent as chat

PILLS_HTML = """
<h2 style="text-align:center;margin-bottom:1em;">What would you like to do?</h2>

<style>
.mode-pills {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 10px;
    margin: 0.6em 0 0.4em;
}
.mode-pill {
    position: relative;
    width: 158px;
    padding: 14px 12px 12px;
    border-radius: 22px;
    background: linear-gradient(
        135deg,
        color-mix(in srgb, var(--pill-accent) 22%, white) 0%,
        color-mix(in srgb, var(--pill-accent) 8%, white) 100%
    );
    border: 1px solid color-mix(in srgb, var(--pill-accent) 35%, white);
    box-shadow:
        0 6px 22px color-mix(in srgb, var(--pill-accent) 18%, transparent),
        inset 0 1px 0 rgba(255, 255, 255, 0.8);
    backdrop-filter: blur(14px) saturate(140%);
    -webkit-backdrop-filter: blur(14px) saturate(140%);
    cursor: pointer;
    transition: transform 0.18s ease, box-shadow 0.18s ease;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 8px;
    font: inherit;
    color: inherit;
    text-align: left;
    overflow: hidden;
}
.mode-pill::before {
    /* glass shine highlight in the top-left */
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 60%;
    height: 50%;
    background: linear-gradient(
        135deg,
        rgba(255, 255, 255, 0.7) 0%,
        rgba(255, 255, 255, 0) 70%
    );
    border-top-left-radius: 22px;
    pointer-events: none;
}
.mode-pill:hover {
    transform: translateY(-2px);
    box-shadow:
        0 10px 30px color-mix(in srgb, var(--pill-accent) 28%, transparent),
        inset 0 1px 0 rgba(255, 255, 255, 0.9);
}
.mode-pill.is-active {
    border-color: var(--pill-accent);
    box-shadow:
        0 0 0 2px color-mix(in srgb, var(--pill-accent) 55%, transparent),
        0 10px 30px color-mix(in srgb, var(--pill-accent) 30%, transparent),
        inset 0 1px 0 rgba(255, 255, 255, 0.9);
}
html[data-theme="dark"] .mode-pill {
    background: linear-gradient(
        135deg,
        color-mix(in srgb, var(--pill-accent) 18%, transparent) 0%,
        color-mix(in srgb, var(--pill-accent) 4%, transparent) 100%
    );
    border-color: color-mix(in srgb, var(--pill-accent) 30%, transparent);
    box-shadow:
        0 6px 22px rgba(0, 0, 0, 0.4),
        inset 0 1px 0 rgba(255, 255, 255, 0.12);
}
html[data-theme="dark"] .mode-pill:hover {
    box-shadow:
        0 10px 30px color-mix(in srgb, var(--pill-accent) 25%, rgba(0, 0, 0, 0.4)),
        inset 0 1px 0 rgba(255, 255, 255, 0.18);
}
html[data-theme="dark"] .mode-pill::before {
    background: linear-gradient(
        135deg,
        rgba(255, 255, 255, 0.18) 0%,
        rgba(255, 255, 255, 0) 70%
    );
}
.mode-pill .mode-icon {
    width: 26px;
    height: 26px;
    color: var(--pill-accent, currentColor);
    display: block;
}
.mode-pill .mode-label {
    font-weight: 600;
    font-size: 0.92rem;
    line-height: 1.2;
}
</style>

<input type="hidden" id="active-mode" value="chat">

<div class="mode-pills">
  <button class="mode-pill" data-mode="search" style="--pill-accent:#0d9488;">
    <svg class="mode-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor"
         stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <circle cx="12" cy="12" r="10"/>
      <line x1="2" y1="12" x2="22" y2="12"/>
      <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>
    </svg>
    <span class="mode-label">Web search</span>
  </button>
  <button class="mode-pill" data-mode="tts" style="--pill-accent:#6366f1;">
    <svg class="mode-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor"
         stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/>
      <path d="M15.54 8.46a5 5 0 0 1 0 7.07"/>
      <path d="M19.07 4.93a10 10 0 0 1 0 14.14"/>
    </svg>
    <span class="mode-label">Text to speech</span>
  </button>
  <button class="mode-pill" data-mode="image" style="--pill-accent:#db2777;">
    <svg class="mode-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor"
         stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
      <circle cx="8.5" cy="8.5" r="1.5"/>
      <polyline points="21 15 16 10 5 21"/>
    </svg>
    <span class="mode-label">Image studio</span>
  </button>
</div>
"""


# --- TTS Studio Panel ----------------------------------------------------
# Phase-1 visual scaffold for the Text-to-speech mode. In phase 2 this
# will appear above the chat textarea when the TTS pill is clicked. For
# now it's appended below the pills so we can iterate on the look in
# isolation. Styled as a "studio" rack: dark panel, mono labels, LCD
# readouts — meant to feel like the front face of audio hardware.

TTS_STUDIO_HTML = """
<style>
.tts-studio {
    --tts-accent: #6366f1;
    --tts-accent-dim: color-mix(in srgb, var(--tts-accent) 35%, transparent);
    --tts-text: var(--text);
    --tts-muted: var(--text-secondary);
    --tts-surface: color-mix(in srgb, var(--tts-accent) 8%, var(--bg));
    --tts-rail: color-mix(in srgb, var(--tts-accent) 22%, transparent);
    --tts-lcd-bg: color-mix(in srgb, var(--tts-accent) 14%, var(--bg));
    --tts-lcd-fg: var(--tts-accent);
    display: none;
    position: relative;
    margin: 0 auto 10px;
    max-width: 560px;
    padding: 22px 22px 24px;
    border-radius: 28px;
    background: linear-gradient(
        135deg,
        color-mix(in srgb, var(--tts-accent) 22%, white) 0%,
        color-mix(in srgb, var(--tts-accent) 8%, white) 100%
    );
    border: 1px solid color-mix(in srgb, var(--tts-accent) 35%, white);
    box-shadow:
        0 12px 40px color-mix(in srgb, var(--tts-accent) 22%, transparent),
        inset 0 1px 0 rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(16px) saturate(140%);
    -webkit-backdrop-filter: blur(16px) saturate(140%);
    color: var(--tts-text);
    text-align: left;
    overflow: hidden;
}
/* Visible whenever the active mode for this user/convo matches the
   studio's modifier class — set on <body data-mode="..."> by the
   script. The studio panels are part of the persistent layout (above
   #input-bar), so these CSS toggles are the only thing that determines
   which (if any) studio is shown on home page or convo. */
body[data-mode="tts"]    .tts-studio.studio--tts    { display: block; }
body[data-mode="image"]  .tts-studio.studio--image  { display: block; }
body[data-mode="search"] .tts-studio.studio--search { display: block; }

/* Per-mode accent overrides — every other rule reads --tts-accent. */
.tts-studio.studio--image  { --tts-accent: #db2777; }
.tts-studio.studio--search { --tts-accent: #0d9488; }
.tts-studio::before {
    /* same diagonal shine as a mode-pill, scaled up */
    content: "";
    position: absolute;
    inset: 0;
    background: radial-gradient(
        circle at 0% 0%,
        rgba(255, 255, 255, 0.55) 0%,
        rgba(255, 255, 255, 0.18) 25%,
        rgba(255, 255, 255, 0) 60%
    );
    border-radius: inherit;
    pointer-events: none;
}
html[data-theme="dark"] .tts-studio {
    background: linear-gradient(
        135deg,
        color-mix(in srgb, var(--tts-accent) 18%, transparent) 0%,
        color-mix(in srgb, var(--tts-accent) 4%, transparent) 100%
    );
    border-color: color-mix(in srgb, var(--tts-accent) 30%, transparent);
    box-shadow:
        0 12px 40px rgba(0, 0, 0, 0.45),
        inset 0 1px 0 rgba(255, 255, 255, 0.12);
}
html[data-theme="dark"] .tts-studio::before {
    background: radial-gradient(
        circle at 0% 0%,
        rgba(255, 255, 255, 0.14) 0%,
        rgba(255, 255, 255, 0.05) 30%,
        rgba(255, 255, 255, 0) 65%
    );
}

.tts-studio__head {
    position: relative;
    display: flex;
    align-items: center;
    gap: 10px;
    padding-bottom: 16px;
    margin-bottom: 18px;
    border-bottom: 1px dashed color-mix(in srgb, var(--tts-accent) 25%, transparent);
    cursor: pointer;
    user-select: none;
}
.tts-studio__chevron {
    margin-left: auto;
    width: 18px;
    height: 18px;
    color: var(--tts-muted);
    transition: transform 0.18s ease, color 0.15s ease;
}
.tts-studio__head:hover .tts-studio__chevron { color: var(--tts-text); }
.tts-studio.is-collapsed { padding: 14px 22px; }
.tts-studio.is-collapsed .tts-studio__head {
    padding-bottom: 0;
    margin-bottom: 0;
    border-bottom: 0;
}
.tts-studio.is-collapsed .tts-studio__chevron { transform: rotate(-90deg); }
.tts-studio.is-collapsed .tts-studio__body { display: none; }
.tts-studio__icon {
    width: 22px;
    height: 22px;
    color: var(--tts-accent);
    flex-shrink: 0;
}
.tts-studio__title {
    font-size: 0.78rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    font-weight: 700;
    color: var(--tts-text);
}
.tts-studio__led {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--tts-accent);
    box-shadow: 0 0 8px var(--tts-accent);
    animation: tts-pulse 2.4s ease-in-out infinite;
    margin-left: 4px;
}
@keyframes tts-pulse {
    0%, 100% { opacity: 0.45; }
    50%      { opacity: 1; }
}

.tts-row { position: relative; margin-bottom: 18px; }
.tts-row:last-child { margin-bottom: 0; }
.tts-row__label {
    display: flex;
    align-items: baseline;
    gap: 10px;
    font-size: 0.62rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--tts-muted);
    margin-bottom: 10px;
    font-weight: 600;
}
.tts-row__hint {
    margin-left: auto;
    font-size: 0.6rem;
    color: var(--tts-muted);
    opacity: 0.7;
    letter-spacing: 0.12em;
    font-weight: 500;
}

/* Generic chip — used by voice & model rows */
.tts-chip {
    appearance: none;
    background: color-mix(in srgb, var(--tts-accent) 6%, var(--bg));
    border: 1px solid color-mix(in srgb, var(--tts-accent) 18%, transparent);
    color: var(--tts-text);
    padding: 6px 11px;
    border-radius: 999px;
    font: inherit;
    font-size: 0.74rem;
    letter-spacing: 0.04em;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    transition: border-color 0.15s ease, background 0.15s ease, color 0.15s ease;
}
.tts-chip::before {
    content: "";
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: color-mix(in srgb, var(--tts-accent) 35%, transparent);
}
.tts-chip:hover { border-color: var(--tts-accent-dim); }
.tts-chip[aria-pressed="true"] {
    background: color-mix(in srgb, var(--tts-accent) 18%, var(--bg));
    border-color: var(--tts-accent);
    color: var(--tts-text);
    font-weight: 600;
}
.tts-chip[aria-pressed="true"]::before {
    background: var(--tts-accent);
    box-shadow: 0 0 6px var(--tts-accent);
}

.tts-voices, .tts-models {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}

/* Speed slider */
.tts-speed {
    display: block;
}
.tts-speed__track {
    position: relative;
    height: 28px;
    display: flex;
    align-items: center;
}
.tts-speed__track input[type="range"] {
    -webkit-appearance: none;
    appearance: none;
    width: 100%;
    height: 4px;
    background: var(--tts-rail);
    border-radius: 999px;
    outline: none;
    margin: 0;
}
.tts-speed__track input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: radial-gradient(circle at 35% 30%,
        #ffffff 0%,
        color-mix(in srgb, var(--tts-accent) 25%, white) 60%,
        var(--tts-accent) 100%);
    border: 1px solid color-mix(in srgb, var(--tts-accent) 40%, white);
    box-shadow: 0 2px 6px color-mix(in srgb, var(--tts-accent) 30%, transparent);
    cursor: pointer;
}
.tts-speed__track input[type="range"]::-moz-range-thumb {
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: radial-gradient(circle at 35% 30%,
        #ffffff 0%,
        color-mix(in srgb, var(--tts-accent) 25%, white) 60%,
        var(--tts-accent) 100%);
    border: 1px solid color-mix(in srgb, var(--tts-accent) 40%, white);
    box-shadow: 0 2px 6px color-mix(in srgb, var(--tts-accent) 30%, transparent);
    cursor: pointer;
}
.tts-speed__lcd {
    background: var(--tts-lcd-bg);
    color: var(--tts-lcd-fg);
    font-family: 'SF Mono', 'JetBrains Mono', ui-monospace, monospace;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    padding: 6px 12px;
    border-radius: 8px;
    min-width: 68px;
    text-align: center;
    border: 1px solid color-mix(in srgb, var(--tts-accent) 25%, transparent);
    box-shadow: inset 0 1px 3px color-mix(in srgb, var(--tts-accent) 12%, transparent);
    text-shadow: 0 0 8px color-mix(in srgb, var(--tts-accent) 50%, transparent);
}
.tts-speed__scale {
    position: relative;
    height: 14px;
    margin-top: 4px;
    font-size: 0.58rem;
    letter-spacing: 0.12em;
    color: var(--tts-muted);
    opacity: 0.65;
    font-family: 'SF Mono', 'JetBrains Mono', ui-monospace, monospace;
}
.tts-speed__scale span {
    position: absolute;
    transform: translateX(-50%);
    top: 0;
}
.tts-speed__scale span:first-child { transform: none; }
.tts-speed__scale span:last-child  { transform: translateX(-100%); }

/* Format selector */
.tts-formats {
    display: inline-flex;
    background: color-mix(in srgb, var(--tts-accent) 6%, var(--bg));
    border: 1px solid color-mix(in srgb, var(--tts-accent) 18%, transparent);
    border-radius: 12px;
    padding: 3px;
    gap: 2px;
}
.tts-format {
    appearance: none;
    background: transparent;
    border: none;
    color: var(--tts-muted);
    padding: 6px 12px;
    border-radius: 9px;
    font: inherit;
    font-size: 0.7rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.12s ease, color 0.12s ease;
}
.tts-format:hover { color: var(--tts-text); }
.tts-format[aria-pressed="true"] {
    background: color-mix(in srgb, var(--tts-accent) 22%, var(--bg));
    color: var(--tts-text);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.4);
}
html[data-theme="dark"] .tts-format[aria-pressed="true"] {
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08);
}

/* Instructions textarea */
.tts-instructions {
    width: 100%;
    background: color-mix(in srgb, var(--tts-accent) 5%, var(--bg));
    border: 1px solid color-mix(in srgb, var(--tts-accent) 18%, transparent);
    border-radius: 12px;
    padding: 10px 12px;
    color: var(--tts-text);
    font-family: inherit;
    font-size: 0.8rem;
    line-height: 1.5;
    resize: vertical;
    min-height: 64px;
    outline: none;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
}
.tts-instructions::placeholder {
    color: var(--tts-muted);
    opacity: 0.7;
    font-style: italic;
}
.tts-instructions:focus {
    border-color: var(--tts-accent);
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--tts-accent) 18%, transparent);
}
</style>

<div class="tts-studio studio--tts is-collapsed" role="group" aria-label="Text-to-speech controls">
    <div class="tts-studio__head" role="button" tabindex="0" aria-expanded="false" aria-controls="tts-studio-body">
        <svg class="tts-studio__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor"
             stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/>
            <path d="M15.54 8.46a5 5 0 0 1 0 7.07"/>
            <path d="M19.07 4.93a10 10 0 0 1 0 14.14"/>
        </svg>
        <span class="tts-studio__title">TTS Studio</span>
        <span class="tts-studio__led" aria-hidden="true"></span>
        <svg class="tts-studio__chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor"
             stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <polyline points="6 9 12 15 18 9"/>
        </svg>
    </div>
    <input type="hidden" id="tts-model"  value="gpt-4o-mini-tts">
    <input type="hidden" id="tts-voice"  value="alloy">
    <input type="hidden" id="tts-format" value="mp3">
    <div class="tts-studio__body" id="tts-studio-body">
    <div class="tts-row">
        <div class="tts-row__label">
            <span>Model</span>
        </div>
        <div class="tts-models" role="radiogroup" aria-label="Model">
            <button type="button" class="tts-chip" data-model="gpt-4o-mini-tts" aria-pressed="true">gpt-4o-mini-tts</button>
            <button type="button" class="tts-chip" data-model="tts-1-hd"        aria-pressed="false">tts-1-hd</button>
            <button type="button" class="tts-chip" data-model="tts-1"           aria-pressed="false">tts-1</button>
        </div>
    </div>
    <div class="tts-row">
        <div class="tts-row__label">
            <span>Voice</span>
        </div>
        <div class="tts-voices" role="radiogroup" aria-label="Voice">
            <button type="button" class="tts-chip" data-voice="alloy"   aria-pressed="true">alloy</button>
            <button type="button" class="tts-chip" data-voice="ash"     aria-pressed="false">ash</button>
            <button type="button" class="tts-chip" data-voice="ballad"  aria-pressed="false">ballad</button>
            <button type="button" class="tts-chip" data-voice="coral"   aria-pressed="false">coral</button>
            <button type="button" class="tts-chip" data-voice="echo"    aria-pressed="false">echo</button>
            <button type="button" class="tts-chip" data-voice="fable"   aria-pressed="false">fable</button>
            <button type="button" class="tts-chip" data-voice="onyx"    aria-pressed="false">onyx</button>
            <button type="button" class="tts-chip" data-voice="nova"    aria-pressed="false">nova</button>
            <button type="button" class="tts-chip" data-voice="sage"    aria-pressed="false">sage</button>
            <button type="button" class="tts-chip" data-voice="shimmer" aria-pressed="false">shimmer</button>
            <button type="button" class="tts-chip" data-voice="verse"   aria-pressed="false">verse</button>
            <button type="button" class="tts-chip" data-voice="marin"   aria-pressed="false">marin</button>
            <button type="button" class="tts-chip" data-voice="cedar"   aria-pressed="false">cedar</button>
        </div>
    </div>
    <div class="tts-row">
        <div class="tts-row__label">
            <span>Speed</span>
        </div>
        <div class="tts-speed">
            <div class="tts-speed__track">
                <input type="range" id="tts-speed" min="0.25" max="4" step="0.05" value="1.0"
                       style="--tts-fill: 20%;" aria-label="Playback speed">
            </div>
            <div class="tts-speed__scale">
                <span style="left: 0%;">0.25</span>
                <span style="left: 20%;">1.0</span>
                <span style="left: 46.67%;">2.0</span>
                <span style="left: 73.33%;">3.0</span>
                <span style="left: 100%;">4.0</span>
            </div>
        </div>
    </div>
    <div class="tts-row">
        <div class="tts-row__label">
            <span>Format</span>
        </div>
        <div class="tts-formats" role="radiogroup" aria-label="Audio format">
            <button type="button" class="tts-format" data-format="mp3"  aria-pressed="true">mp3</button>
            <button type="button" class="tts-format" data-format="opus" aria-pressed="false">opus</button>
            <button type="button" class="tts-format" data-format="aac"  aria-pressed="false">aac</button>
            <button type="button" class="tts-format" data-format="flac" aria-pressed="false">flac</button>
            <button type="button" class="tts-format" data-format="wav"  aria-pressed="false">wav</button>
            <button type="button" class="tts-format" data-format="pcm"  aria-pressed="false">pcm</button>
        </div>
    </div>
    <div class="tts-row">
        <div class="tts-row__label">
            <span>Style instructions</span>
            <span class="tts-row__hint">optional · gpt-4o-mini-tts only</span>
        </div>
        <textarea class="tts-instructions" id="tts-instructions" rows="2"
            placeholder="Speak with a warm, conspiratorial tone. Slightly slower at the punchline."></textarea>
    </div>
    </div>
</div>
"""


# --- Image Studio Panel ------------------------------------------------------
# Same accordion shell as the TTS studio (.tts-studio CSS is fully reused), but
# with the rose accent (--tts-accent override on .studio--image) and a tighter
# control set tuned to OpenAI's images.generate endpoint: Size, Quality,
# Background. Single-shot generation, no edits / partials — the user gets one
# image per send, and starting a new chat is the way to start a new image.

IMAGE_STUDIO_HTML = """
<div class="tts-studio studio--image is-collapsed" role="group" aria-label="Image generation controls">
    <div class="tts-studio__head" role="button" tabindex="0" aria-expanded="false" aria-controls="image-studio-body">
        <svg class="tts-studio__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor"
             stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
            <circle cx="8.5" cy="8.5" r="1.5"/>
            <polyline points="21 15 16 10 5 21"/>
        </svg>
        <span class="tts-studio__title">Image Studio</span>
        <span class="tts-studio__led" aria-hidden="true"></span>
        <svg class="tts-studio__chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor"
             stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <polyline points="6 9 12 15 18 9"/>
        </svg>
    </div>
    <input type="hidden" id="image-size"       value="1024x1024">
    <input type="hidden" id="image-quality"    value="auto">
    <input type="hidden" id="image-background" value="auto">
    <div class="tts-studio__body" id="image-studio-body">
    <div class="tts-row">
        <div class="tts-row__label">
            <span>Size</span>
            <span class="tts-row__hint">aspect ratio</span>
        </div>
        <div class="tts-models" role="radiogroup" aria-label="Size">
            <button type="button" class="tts-chip" data-image-size="1024x1024" aria-pressed="true">Square · 1024</button>
            <button type="button" class="tts-chip" data-image-size="1024x1536" aria-pressed="false">Portrait · 1024×1536</button>
            <button type="button" class="tts-chip" data-image-size="1536x1024" aria-pressed="false">Landscape · 1536×1024</button>
            <button type="button" class="tts-chip" data-image-size="auto"      aria-pressed="false">auto</button>
        </div>
    </div>
    <div class="tts-row">
        <div class="tts-row__label">
            <span>Quality</span>
            <span class="tts-row__hint">higher = slower &amp; pricier</span>
        </div>
        <div class="tts-formats" role="radiogroup" aria-label="Quality">
            <button type="button" class="tts-format" data-image-quality="auto"   aria-pressed="true">auto</button>
            <button type="button" class="tts-format" data-image-quality="low"    aria-pressed="false">low</button>
            <button type="button" class="tts-format" data-image-quality="medium" aria-pressed="false">medium</button>
            <button type="button" class="tts-format" data-image-quality="high"   aria-pressed="false">high</button>
        </div>
    </div>
    <div class="tts-row">
        <div class="tts-row__label">
            <span>Background</span>
            <span class="tts-row__hint">transparent works best with PNG</span>
        </div>
        <div class="tts-formats" role="radiogroup" aria-label="Background">
            <button type="button" class="tts-format" data-image-background="auto"        aria-pressed="true">auto</button>
            <button type="button" class="tts-format" data-image-background="opaque"      aria-pressed="false">opaque</button>
            <button type="button" class="tts-format" data-image-background="transparent" aria-pressed="false">transparent</button>
        </div>
    </div>
    </div>
</div>
"""


# --- Search Studio Panel -----------------------------------------------------
# Same accordion shell as TTS / Image, with the teal accent override
# (.studio--search). Two controls: Model + Search depth (search_context_size).
# The Responses API + web_search tool returns text with URL annotations —
# SearchLLM.extract_content() formats those as inline superscripts + a Sources
# footnote block. No file artifacts to persist (text-only reply), so the
# engine's _rewrite_artifacts is untouched for this mode.

SEARCH_STUDIO_HTML = """
<div class="tts-studio studio--search is-collapsed" role="group" aria-label="Web search controls">
    <div class="tts-studio__head" role="button" tabindex="0" aria-expanded="false" aria-controls="search-studio-body">
        <svg class="tts-studio__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor"
             stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <circle cx="12" cy="12" r="10"/>
            <line x1="2" y1="12" x2="22" y2="12"/>
            <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>
        </svg>
        <span class="tts-studio__title">Search Studio</span>
        <span class="tts-studio__led" aria-hidden="true"></span>
        <svg class="tts-studio__chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor"
             stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <polyline points="6 9 12 15 18 9"/>
        </svg>
    </div>
    <input type="hidden" id="search-model" value="gpt-4o-mini">
    <input type="hidden" id="search-depth" value="medium">
    <div class="tts-studio__body" id="search-studio-body">
    <div class="tts-row">
        <div class="tts-row__label">
            <span>Model</span>
            <span class="tts-row__hint">all support web_search</span>
        </div>
        <div class="tts-models" role="radiogroup" aria-label="Model">
            <button type="button" class="tts-chip" data-search-model="gpt-4o-mini" aria-pressed="true">gpt-4o-mini</button>
            <button type="button" class="tts-chip" data-search-model="gpt-4o"      aria-pressed="false">gpt-4o</button>
            <button type="button" class="tts-chip" data-search-model="gpt-4.1-mini" aria-pressed="false">gpt-4.1-mini</button>
            <button type="button" class="tts-chip" data-search-model="gpt-4.1"      aria-pressed="false">gpt-4.1</button>
        </div>
    </div>
    <div class="tts-row">
        <div class="tts-row__label">
            <span>Search depth</span>
            <span class="tts-row__hint">higher = slower &amp; pricier</span>
        </div>
        <div class="tts-formats" role="radiogroup" aria-label="Search depth">
            <button type="button" class="tts-format" data-search-depth="low"    aria-pressed="false">low</button>
            <button type="button" class="tts-format" data-search-depth="medium" aria-pressed="true">medium</button>
            <button type="button" class="tts-format" data-search-depth="high"   aria-pressed="false">high</button>
        </div>
    </div>
    </div>
</div>
"""


# --- Phase 2: interactivity --------------------------------------------------
# DOMPurify (the sanitizer that runs over `welcome_message`) strips <script>
# tags and inline event handlers. To wire interactivity we extend DefaultLayout
# and inject our JS once, after </body>. Document-level event delegation then
# handles clicks even on nodes added later by the welcome render.

PILLS_SCRIPT = """
<script>
(function () {
  function press(group, target) {
    group.querySelectorAll('[aria-pressed]').forEach(function (b) {
      b.setAttribute('aria-pressed', b === target ? 'true' : 'false');
    });
  }
  function pushControl(id, value) {
    var el = document.getElementById(id);
    if (el) {
      el.value = value;
      chatInteraction(el);
    }
  }
  function setStudioExpanded(studio, expanded) {
    studio.classList.toggle('is-collapsed', !expanded);
    var head = studio.querySelector('.tts-studio__head');
    if (head) head.setAttribute('aria-expanded', expanded ? 'true' : 'false');
  }
  function setActiveMode(mode) {
    document.body.dataset.mode = mode;
    var modeInput = document.getElementById('active-mode');
    if (modeInput) {
      modeInput.value = mode;
      chatInteraction(modeInput);
    }
    document.querySelectorAll('.mode-pill').forEach(function (p) {
      p.classList.toggle('is-active', p.dataset.mode === mode);
    });
  }
  document.addEventListener('click', function (e) {
    var pill = e.target.closest('.mode-pill');
    if (pill) {
      // Toggle behavior: clicking the active pill returns to plain chat.
      var nextMode = (document.body.dataset.mode === pill.dataset.mode)
        ? 'chat'
        : pill.dataset.mode;
      setActiveMode(nextMode);
      // If a studio for this mode exists, expand it for convenience.
      var studio = document.querySelector('.studio--' + pill.dataset.mode);
      if (studio && nextMode !== 'chat') setStudioExpanded(studio, true);
      var input = document.getElementById('input');
      if (input) input.focus();
      return;
    }
    var head = e.target.closest('.tts-studio__head');
    if (head) {
      var studio = head.closest('.tts-studio');
      if (studio) setStudioExpanded(studio, studio.classList.contains('is-collapsed'));
      return;
    }
    var chip = e.target.closest('.tts-chip, .tts-format');
    if (chip) {
      press(chip.parentElement, chip);
      if (chip.dataset.model)  pushControl('tts-model',  chip.dataset.model);
      if (chip.dataset.voice)  pushControl('tts-voice',  chip.dataset.voice);
      if (chip.dataset.format) pushControl('tts-format', chip.dataset.format);
      if (chip.dataset.imageSize)       pushControl('image-size',       chip.dataset.imageSize);
      if (chip.dataset.imageQuality)    pushControl('image-quality',    chip.dataset.imageQuality);
      if (chip.dataset.imageBackground) pushControl('image-background', chip.dataset.imageBackground);
      if (chip.dataset.searchModel)     pushControl('search-model',     chip.dataset.searchModel);
      if (chip.dataset.searchDepth)     pushControl('search-depth',     chip.dataset.searchDepth);
    }
  });
  document.addEventListener('keydown', function (e) {
    if ((e.key === 'Enter' || e.key === ' ') && e.target.classList.contains('tts-studio__head')) {
      e.preventDefault();
      var studio = e.target.closest('.tts-studio');
      if (studio) setStudioExpanded(studio, studio.classList.contains('is-collapsed'));
    }
  });
  document.addEventListener('change', function (e) {
    if (e.target.id === 'tts-speed' || e.target.id === 'tts-instructions') {
      chatInteraction(e.target);
    }
  });
  // Restore the convo's mode from its sidecar mode.txt (written by the
  // engine on first turn). The /files/ route serves it as text/plain.
  function restoreModeFromSidecar() {
    var match = window.location.pathname.match(/^\\/?([^\\/]+)\\/([^\\/]+)\\/?$/);
    if (!match) return;
    fetch('/files/' + match[1] + '/' + match[2] + '/mode.txt')
      .then(function (r) { return r.ok ? r.text() : null; })
      .then(function (txt) {
        if (txt) setActiveMode(txt.trim());
      })
      .catch(function () { /* no sidecar → stay on whatever pill was clicked */ });
  }
  // After every send, the URL changes to /<user>/<convo> — that's our cue
  // that this convo's mode is now sealed. Re-fetch the sidecar to confirm.
  function hookAfterSend() {
    if (!window.chatnificent) return;
    var prev = window.chatnificent.afterSend || function () {};
    window.chatnificent.afterSend = function (convoId) {
      prev(convoId);
      restoreModeFromSidecar();
    };
  }
  document.addEventListener('DOMContentLoaded', function () {
    if (!document.body.dataset.mode) document.body.dataset.mode = 'chat';
    hookAfterSend();
    restoreModeFromSidecar();
  });
})();
</script>
"""


class MultiChatModeLayout(chat.layout.DefaultLayout):
    """DefaultLayout + the persistent TTS studio + the interactivity script.

    The studio panel must live *outside* the welcome message: when the user
    sends their first message the framework hides ``#welcome``, and we want
    the studio to remain visible so they can keep tweaking voice/speed and
    re-sending the same text. So we splice it into the persistent layout
    just above ``#input-bar``. CSS in ``TTS_STUDIO_HTML`` keeps it hidden
    until either ``body.in-convo`` is set (auto, by the script's
    MutationObserver) or the user explicitly opens it from the home page
    by clicking the TTS pill (``.tts-open``).
    """

    def render_page(self) -> str:
        html = super().render_page()
        html = html.replace(
            '<div id="input-bar">',
            TTS_STUDIO_HTML
            + IMAGE_STUDIO_HTML
            + SEARCH_STUDIO_HTML
            + '<div id="input-bar">',
        )
        return html.replace("</body>", PILLS_SCRIPT + "\n</body>")


# --- TTS LLM -----------------------------------------------------------------
# A minimal LLM pillar that calls OpenAI's audio.speech endpoint instead of
# chat completions. The control values from the studio (model / voice / speed /
# response_format / instructions) flow in as kwargs via DefaultLayout's
# `_get_llm_kwargs(user_id)` seam — nothing else to wire on this side.
#
# We don't stream and we don't speak chat — every assistant turn is exactly
# one audio file. The bytes are base64-embedded in the assistant message so
# the engine can write them to the Store pillar after the turn completes.


class TTSLLM(chat.llm.LLM):
    """OpenAI TTS as a Chatnificent LLM pillar."""

    def __init__(self, model: str = "gpt-4o-mini-tts", voice: str = "alloy"):
        from openai import OpenAI as _OpenAIClient

        self.client = _OpenAIClient()
        self.model = model
        self.voice = voice
        self.default_params = {"stream": False}

    def build_request_payload(self, messages, model=None, tools=None, **kwargs):
        return {
            "model": model or self.model,
            "input": _last_user_text(messages),
            **kwargs,
        }

    def generate_response(self, messages, model=None, tools=None, **kwargs):
        text = _last_user_text(messages) or "(empty)"
        api_kwargs = {
            "model": kwargs.get("model", model or self.model),
            "voice": kwargs.get("voice", self.voice),
            "input": text,
            "response_format": kwargs.get("response_format", "mp3"),
        }
        if "speed" in kwargs:
            api_kwargs["speed"] = kwargs["speed"]
        instructions = kwargs.get("instructions")
        if instructions and api_kwargs["model"] == "gpt-4o-mini-tts":
            api_kwargs["instructions"] = instructions
        response = self.client.audio.speech.create(**api_kwargs)
        return {
            "audio_b64": base64.b64encode(response.read()).decode("ascii"),
            "format": api_kwargs["response_format"],
            "transcript": text,
        }

    def extract_content(self, response):
        # Inline placeholder the engine will rewrite into a real <audio> tag
        # once the bytes are persisted to a stable URL. The transcript is
        # already visible in the user's message bubble right above, so we
        # don't repeat it here.
        b64 = response["audio_b64"]
        fmt = response["format"]
        return f'<audio data-tts-b64="{b64}" data-format="{fmt}"></audio>'


def _last_user_text(messages):
    for msg in reversed(messages):
        if msg.get("role") == chat.models.USER_ROLE:
            content = msg.get("content")
            return content if isinstance(content, str) else ""
    return ""


# --- Image LLM ---------------------------------------------------------------
# Calls OpenAI's images.generate endpoint directly (not the Responses API
# image_generation tool). One prompt → one image, no edits, no partials.
# Bytes are returned base64-encoded in the API response and embedded as a
# placeholder tag that the engine rewrites into a stable /files/ URL.


class ImageLLM(chat.llm.LLM):
    """OpenAI image generation as a Chatnificent LLM pillar."""

    def __init__(self, model: str = "gpt-image-1"):
        from openai import OpenAI as _OpenAIClient

        self.client = _OpenAIClient()
        self.model = model
        self.default_params = {"stream": False}

    def build_request_payload(self, messages, model=None, tools=None, **kwargs):
        return {
            "model": model or self.model,
            "prompt": _last_user_text(messages),
            **kwargs,
        }

    def generate_response(self, messages, model=None, tools=None, **kwargs):
        prompt = _last_user_text(messages) or "(empty)"
        # Note: we deliberately ignore the `model` kwarg here. The TTS
        # studio's tts-model Control writes to the same `model` LLM param,
        # and a stale TTS model name would leak across modes if honored.
        # The image studio is gpt-image-1-only by design.
        api_kwargs = {
            "model": self.model,
            "prompt": prompt,
            "size": kwargs.get("size", "1024x1024"),
            "n": 1,
        }
        if api_kwargs["model"] == "gpt-image-1":
            if kwargs.get("quality"):
                api_kwargs["quality"] = kwargs["quality"]
            if kwargs.get("background"):
                api_kwargs["background"] = kwargs["background"]
        response = self.client.images.generate(**api_kwargs)
        item = response.data[0]
        b64 = getattr(item, "b64_json", None)
        if not b64:
            # Defensive fallback — gpt-image-1 always returns b64_json today,
            # but make the failure mode visible rather than silent.
            return {"image_b64": "", "format": "png", "error": "no image data"}
        return {"image_b64": b64, "format": "png"}

    def extract_content(self, response):
        if response.get("error"):
            return f"_(image generation failed: {response['error']})_"
        b64 = response["image_b64"]
        fmt = response["format"]
        return f'<img data-img-b64="{b64}" data-format="{fmt}" alt="Generated image">'


# --- Search LLM --------------------------------------------------------------
# Calls OpenAI's Responses API with the built-in `web_search` tool. The
# response carries `output_text` plus structured URL annotations — we walk the
# annotations, dedupe by URL, and render Markdown with inline superscripts
# linking to a numbered Sources block at the bottom. Non-streaming.

_SUPERSCRIPTS = "⁰¹²³⁴⁵⁶⁷⁸⁹"


def _to_superscript(n: int) -> str:
    return "".join(_SUPERSCRIPTS[int(d)] for d in str(n))


class SearchLLM(chat.llm.LLM):
    """OpenAI Responses API + web_search tool as a Chatnificent LLM pillar."""

    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI as _OpenAIClient

        self.client = _OpenAIClient()
        self.model = model
        self.default_params = {"stream": False}

    def build_request_payload(self, messages, model=None, tools=None, **kwargs):
        return {
            "model": model or self.model,
            "input": messages,
            **kwargs,
        }

    def generate_response(self, messages, model=None, tools=None, **kwargs):
        chosen_model = kwargs.get("model", model or self.model)
        depth = kwargs.get("search_context_size", "medium")
        return self.client.responses.create(
            model=chosen_model,
            input=messages,
            tools=[{"type": "web_search", "search_context_size": depth}],
            tool_choice="auto",
        )

    def extract_content(self, response):
        # Walk the response output for the assistant message item; collect
        # both the text and the annotation list, then weave them together.
        text_parts = []
        annotations = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) != "message":
                continue
            for part in getattr(item, "content", []) or []:
                if getattr(part, "type", None) != "output_text":
                    continue
                text_parts.append(getattr(part, "text", "") or "")
                for ann in getattr(part, "annotations", []) or []:
                    if getattr(ann, "type", None) == "url_citation":
                        annotations.append(ann)
        text = "".join(text_parts)
        if not text:
            return "_(no response text)_"

        # Dedupe URLs in first-seen order; build {url: index_starting_at_1}.
        seen = {}
        sources = []
        for ann in annotations:
            url = getattr(ann, "url", None)
            if not url or url in seen:
                continue
            seen[url] = len(sources) + 1
            title = getattr(ann, "title", None) or url
            sources.append((title, url))

        if not sources:
            return text

        # Insert inline superscripts at each annotation's end_index. Walk
        # right-to-left so earlier indices stay valid as we mutate the string.
        # Annotations point at their underlying URL, which we already mapped.
        insertions = []
        for ann in annotations:
            url = getattr(ann, "url", None)
            end = getattr(ann, "end_index", None)
            if url not in seen or end is None:
                continue
            idx = seen[url]
            marker = f"[{_to_superscript(idx)}]({url})"
            insertions.append((end, marker))
        # Stable sort so multiple annotations at the same end_index keep order.
        insertions.sort(key=lambda x: x[0], reverse=True)
        chars = list(text)
        for pos, marker in insertions:
            pos = max(0, min(pos, len(chars)))
            chars.insert(pos, marker)
        text = "".join(chars)

        sources_md = "\n".join(
            f"{i}. [{title}]({url})" for i, (title, url) in enumerate(sources, 1)
        )
        return f"{text}\n\n**Sources**\n\n{sources_md}"


# --- Mode router ------------------------------------------------------------
# A single LLM proxy that forwards every method to the active mode's sub-LLM.
# The active mode is request-scoped via contextvars (thread-safe under any
# server). The engine sets the var at the top of handle_message() based on
# the convo's mode.txt sidecar, and resets it in finally.


_active_mode: contextvars.ContextVar = contextvars.ContextVar(
    "single_app_multi_chat_mode_active_mode", default="chat"
)


class ModeRouter(chat.llm.LLM):
    """Delegating LLM that routes every call to the active sub-LLM."""

    def __init__(self, modes):
        self.modes = modes
        # Used by the server's pre-flight streaming check; we declare all
        # modes non-streaming for this example.
        self.default_params = {"stream": False}

    @property
    def model(self):
        return getattr(self._active(), "model", "mode-router")

    def _active(self):
        return self.modes.get(_active_mode.get(), self.modes["chat"])

    def build_request_payload(self, messages, model=None, tools=None, **kwargs):
        return self._active().build_request_payload(
            messages, model=model, tools=tools, **kwargs
        )

    def generate_response(self, messages, model=None, tools=None, **kwargs):
        return self._active().generate_response(
            messages, model=model, tools=tools, **kwargs
        )

    def extract_content(self, response):
        return self._active().extract_content(response)

    def parse_tool_calls(self, response):
        return self._active().parse_tool_calls(response)

    def create_assistant_message(self, response):
        return self._active().create_assistant_message(response)

    def create_tool_result_messages(self, results, conversation):
        return self._active().create_tool_result_messages(results, conversation)

    def extract_stream_delta(self, chunk):
        return self._active().extract_stream_delta(chunk)

    def is_tool_message(self, message):
        return self._active().is_tool_message(message)


# --- Engine ------------------------------------------------------------------
# Resolves the mode at the top of handle_message() and stamps a permanent
# `mode.txt` sidecar on first turn. Then post-processes any modality-specific
# artifacts (audio for now; image will be added in Phase 2).


_VALID_MODES = {"chat", "tts", "image", "search"}

# Per-mode mapping from control id → (LLM kwarg, cast). Layout state is one
# shared bag per user, so we can't rely on the framework's built-in
# `llm_param` injection: tts-model and search-model would both want to write
# `model` and race each other. The engine instead reads raw control values
# by id, casts them, and remaps them under the names the active mode's
# sub-LLM expects. Unregistered controls (like `active-mode`) are filtered
# implicitly — they never appear in any mode's table.
_MODE_CONTROLS = {
    "chat": [],
    "tts": [
        ("tts-model", "model", str),
        ("tts-voice", "voice", str),
        ("tts-speed", "speed", float),
        ("tts-format", "response_format", str),
        ("tts-instructions", "instructions", str),
    ],
    "image": [
        ("image-size", "size", str),
        ("image-quality", "quality", str),
        ("image-background", "background", str),
    ],
    "search": [
        ("search-model", "model", str),
        ("search-depth", "search_context_size", str),
    ],
}


class ModeAwareEngine(chat.engine.Orchestrator):
    def handle_message(self, user_input, user_id, convo_id_from_url):
        token = _active_mode.set(self._resolve_mode(user_id, convo_id_from_url))
        try:
            return super().handle_message(user_input, user_id, convo_id_from_url)
        finally:
            _active_mode.reset(token)

    def handle_message_stream(self, user_input, user_id, convo_id_from_url):
        token = _active_mode.set(self._resolve_mode(user_id, convo_id_from_url))
        try:
            yield from super().handle_message_stream(
                user_input, user_id, convo_id_from_url
            )
        finally:
            _active_mode.reset(token)

    def _resolve_mode(self, user_id, convo_id):
        # Existing convo: mode.txt is canonical and immutable.
        if convo_id:
            stored = self.app.store.load_file(user_id, convo_id, "mode.txt")
            if stored:
                mode = stored.decode("utf-8").strip()
                if mode in _VALID_MODES:
                    return mode
        # New convo: fall back to whatever pill the user clicked on the
        # home page (stored as a regular layout interaction).
        values = self.app.layout.get_control_values(user_id)
        candidate = values.get("active-mode", "chat")
        return candidate if candidate in _VALID_MODES else "chat"

    def _get_llm_kwargs(self, user_id):
        # Build per-mode kwargs from raw control values. We bypass the
        # framework's `llm_param` injection because two modes (tts, search)
        # want to set `model` from different controls — the framework would
        # race them through one shared key. Each mode's table maps its own
        # control ids to the LLM kwargs the active sub-LLM expects.
        raw = self.app.layout.get_control_values(user_id)
        out = {}
        for cid, param, cast in _MODE_CONTROLS.get(_active_mode.get(), []):
            value = raw.get(cid)
            if value in (None, ""):
                continue
            try:
                out[param] = cast(value)
            except (ValueError, TypeError):
                continue
        return out

    def _save_conversation(self, conversation, user_id):
        # Stamp the convo's mode permanently on first save.
        existing_mode = self.app.store.load_file(user_id, conversation.id, "mode.txt")
        if not existing_mode:
            self.app.store.save_file(
                user_id,
                conversation.id,
                "mode.txt",
                _active_mode.get().encode("utf-8"),
            )

        # Post-process modality artifacts in the latest assistant message.
        last = conversation.messages[-1] if conversation.messages else None
        if last and last.get("role") == chat.models.ASSISTANT_ROLE:
            content = last.get("content", "")
            if isinstance(content, str):
                last["content"] = self._rewrite_artifacts(
                    content, user_id, conversation.id
                )

        super()._save_conversation(conversation, user_id)

    def _rewrite_artifacts(self, content, user_id, convo_id):
        existing = self.app.store.list_files(user_id, convo_id)

        # Audio (TTS mode)
        audio_counter = [sum(1 for f in existing if f.startswith("audio/"))]

        def _replace_audio(match):
            b64 = match.group(1)
            fmt = match.group(2)
            try:
                data = base64.b64decode(b64)
            except (ValueError, TypeError):
                return match.group(0)
            filename = f"audio/{audio_counter[0]}.{fmt}"
            self.app.store.save_file(user_id, convo_id, filename, data)
            audio_counter[0] += 1
            url = f"/files/{user_id}/{convo_id}/{filename}"
            return f'<audio controls preload="metadata" src="{url}"></audio>'

        content = _re.sub(
            r'<audio data-tts-b64="([A-Za-z0-9+/=]+)" data-format="([a-z0-9]+)"></audio>',
            _replace_audio,
            content,
        )

        # Images (Image mode)
        image_counter = [sum(1 for f in existing if f.startswith("images/"))]

        def _replace_image(match):
            b64 = match.group(1)
            fmt = match.group(2)
            try:
                data = base64.b64decode(b64)
            except (ValueError, TypeError):
                return match.group(0)
            filename = f"images/{image_counter[0]}.{fmt}"
            self.app.store.save_file(user_id, convo_id, filename, data)
            image_counter[0] += 1
            url = f"/files/{user_id}/{convo_id}/{filename}"
            return (
                f'<img src="{url}" alt="Generated image" '
                f'style="max-width:100%;border-radius:12px;">'
            )

        content = _re.sub(
            r'<img data-img-b64="([A-Za-z0-9+/=]+)" data-format="([a-z0-9]+)"[^>]*>',
            _replace_image,
            content,
        )
        return content


# --- Server ------------------------------------------------------------------
# Adds GET /files/<user>/<convo>/<filename...> so saved audio + sidecars are
# reachable. mode.txt is served as text/plain so the page-load script can
# read it back to restore the studio state on convo replay.

_FILE_MIME = {
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "application/octet-stream",
    "txt": "text/plain; charset=utf-8",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
}


class _FileServingDevHandler(chat.server._DevHandler):
    def do_GET(self):
        if self.path.startswith("/files/"):
            self._serve_file()
        else:
            super().do_GET()

    def _serve_file(self):
        parts = self.path.split("/", 4)
        if len(parts) != 5 or parts[1] != "files":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        _, _, user_id, convo_id, filename = parts
        if ".." in filename or "\\" in filename or filename.startswith("/"):
            self.send_error(HTTPStatus.BAD_REQUEST)
            return
        if user_id != self._get_user_id():
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        data = self._app.store.load_file(user_id, convo_id, filename)
        if not data:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        ext = filename.rsplit(".", 1)[-1].lower()
        self.send_response(HTTPStatus.OK)
        self.send_header(
            "Content-Type", _FILE_MIME.get(ext, "application/octet-stream")
        )
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "private, max-age=3600")
        self.end_headers()
        self.wfile.write(data)


class _FileServingDevServer(chat.server.DevServer):
    def run(self, **kwargs):
        host = kwargs.get("host", self._host)
        port = kwargs.get("port", self._port)
        handler = partial(_FileServingDevHandler, self.app)
        self.httpd = HTTPServer((host, port), handler)
        print(f"Single App, Multiple Chat Modes running on http://{host}:{port}")
        try:
            self.httpd.serve_forever()
        except KeyboardInterrupt:
            self.httpd.server_close()


# --- Default chat LLM --------------------------------------------------------
# Used when the user types without picking a pill. Matches the framework's
# auto-detection: real OpenAI if a key is set, else Echo.


def _default_chat_llm():
    import os

    if os.environ.get("OPENAI_API_KEY"):
        try:
            return chat.llm.OpenAI()
        except Exception:
            pass
    from chatnificent.llm import Echo

    return Echo()


# --- Controls ----------------------------------------------------------------
# Each Control's `html` is empty — the studios live in the persistent layout
# (above #input-bar), so the framework only needs the id, llm_param, and cast
# for state flow. `active-mode` is also a Control so its value is stored in
# the layout's per-user state — the engine reads it to decide the mode.

CONTROLS = [
    # Mode dispatch (read by ModeAwareEngine, stripped before LLM call).
    chat.layout.Control(
        id="active-mode", html="", slot="toolbar", llm_param="active-mode"
    ),
    # TTS studio.
    chat.layout.Control(id="tts-model", html="", slot="toolbar", llm_param="model"),
    chat.layout.Control(id="tts-voice", html="", slot="toolbar", llm_param="voice"),
    chat.layout.Control(
        id="tts-speed", html="", slot="toolbar", llm_param="speed", cast=float
    ),
    chat.layout.Control(
        id="tts-format", html="", slot="toolbar", llm_param="response_format"
    ),
    chat.layout.Control(
        id="tts-instructions", html="", slot="toolbar", llm_param="instructions"
    ),
    # Image studio. The studio is gpt-image-1-only, so we don't expose a
    # model picker — ImageLLM's __init__ default is canonical.
    chat.layout.Control(id="image-size", html="", slot="toolbar", llm_param="size"),
    chat.layout.Control(
        id="image-quality", html="", slot="toolbar", llm_param="quality"
    ),
    chat.layout.Control(
        id="image-background", html="", slot="toolbar", llm_param="background"
    ),
    # Search studio.
    chat.layout.Control(id="search-model", html="", slot="toolbar", llm_param="model"),
    chat.layout.Control(
        id="search-depth", html="", slot="toolbar", llm_param="search_context_size"
    ),
]


app = chat.Chatnificent(
    llm=ModeRouter(
        {
            "chat": _default_chat_llm(),
            "tts": TTSLLM(),
            "image": ImageLLM(),
            "search": SearchLLM(),
        }
    ),
    store=chat.store.File(base_dir="single_app_data"),
    engine=ModeAwareEngine(),
    server=_FileServingDevServer(),
    layout=MultiChatModeLayout(
        brand="Chatnificent",
        slogan="Single App, Multiple Chat Modes",
        welcome_message=PILLS_HTML,
        controls=CONTROLS,
    ),
)


if __name__ == "__main__":
    app.run()
