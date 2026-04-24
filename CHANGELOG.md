# Changelog

All notable changes to Chatnificent will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.0.19] — 2026-04-24

UI Interactions — bind HTML controls directly to LLM call parameters with zero custom server code.

### Added

- **Layout**: `Control` dataclass (`id`, `html`, `slot`, `llm_param`, `cast`) — declarative binding of any HTML element to a `generate_response()` kwarg; `cast` can return any Python object (scalars, dicts, lists-of-dicts)
- **Layout**: `DefaultLayout(controls=[...])` constructor param — register controls at construction without subclassing; `register_control()` still available for conditional runtime registration
- **Layout**: `get_llm_kwargs(user_id)` — returns cast, per-user control state ready to inject into every LLM call; skips `None` values (null sentinel: send `null` from JS to reset a param to its default)
- **Layout**: template slot system — `<!-- SLOT:name -->` markers (`toolbar`, `sidebar`, `input-bar`); `render_page()` injects control HTML and a DOMContentLoaded auto-init script that fires `chatInteraction(el)` for each registered control
- **Engine**: `_get_llm_kwargs(user_id)` seam — delegates to `layout.get_llm_kwargs()` and merges the result into every `generate_response()` call
- **Server**: `POST /api/interactions` endpoint on both DevServer and StarletteServer — fire-and-forget; accepts `{"id": ..., "data": ...}` (JSON `null` clears the param back to default)
- **Frontend**: `chatInteraction(element, data?)` JS helper injected by `DefaultLayout.render_page()` — POSTs `{"id": element.id, "data": value}` to `/api/interactions`; `data` defaults to `element.value`
- **Examples**: `ui_interactions.py` — single `<select>` bound to `max_completion_tokens`; the minimal one-control reference
- **Examples**: `openai_responses_interactive_search.py` — three controls wired to the Responses API: reasoning effort (`reasoning={"effort": ...}`), domain-filter pill checkboxes (`tools`), and a CSS toggle switch for force-search (`tool_choice`)
- **Docs**: AGENTS.md updated — `DefaultLayout(controls=[...])` documented as the primary pattern with multi-control example, JS helper API, and slot reference

## [0.0.18] — 2026-04-20

OpenAI Responses API examples; `File` store accepts nested filenames for per-conversation media.

### Added

- **Store**: `File` store now accepts nested filenames (e.g. `images/foo.png`) in `save_file()` / `load_file()` / `list_files()` — `_get_file_path` resolves the joined path with `Path.resolve()` and asserts containment via `is_relative_to(convo_dir)`, safely supporting subdirectories while blocking traversal and symlink escapes. `list_files()` recurses with `rglob` and excludes reserved framework files (`messages.json`, `raw_api_*.jsonl`). Enables per-conversation media storage (images, audio, documents) without subclassing
- **Examples**: Tier 6 OpenAI Responses API suite — four examples showing how to route Chatnificent through OpenAI's `responses.create` endpoint in a small subclass:
  - `openai_responses` — minimal 8-line subclass baseline
  - `openai_responses_website_search` — domain-restricted research assistant via the hosted web-search tool
  - `openai_responses_image_generator` — inline image generation every turn via the hosted `image_generation` tool
  - `openai_responses_image_studio` — multi-turn image studio with per-conversation image persistence and context-safe replay
- **Agents**: `.agents/skills/example-app/SKILL.md` — agent-skill convention for authoring `/examples/*.py`, so AI coding assistants can reliably produce examples that match the project's style ([Roadmap Phase 1: "Ship all examples as agent skills"](ROADMAP.md))
- **Dev**: `pytest-xdist` added to the `dev` extra — full suite drops from ~127s to ~38s via `pytest -n auto`

### Fixed

- **Tests**: `test_starlette.py` imports that follow `pytest.importorskip` now carry explicit `# noqa: E402` (module must still be cleanly skipped when the `starlette` extra is missing)

## [0.0.17] — 2026-04-05

ASGI root_path mounting — run multiple Chatnificent apps on one website.

### Added

- **Server**: ASGI `root_path` mounting support — Chatnificent auto-detects the mount prefix from `scope["root_path"]` and adjusts API paths, browser URLs, session cookies, and frontend fetch calls accordingly
- **Server**: base class helpers `_inject_root_into_html()`, `_build_full_conversation_path()`, `_cookie_path()` — reusable by any future ASGI server implementation
- **Frontend**: `apiBase` variable in `default.html` reads `window.__CHATNIFICENT_ROOT__` for prefix-aware API calls and URL history
- **Examples**: Tier 5 Starlette suite — `starlette_quickstart`, `starlette_server_options`, `starlette_uvicorn_options`, `starlette_multi_mount`

### Changed

- **Server**: `Starlette.__init__` simplified — removed `prefix` param, removed defensive `list()`/`dict()` copies (Starlette accepts `None`)
- **Server**: `Starlette.run()` simplified — explicit params (`host`, `port`, `workers`, `reload`, `log_level`, `ssl_keyfile`, `ssl_certfile`) with transparent `**kwargs` passthrough to uvicorn; removed `debug` flag
- **Server**: `Starlette.create_server()` simplified — passes all 5 params directly to `starlette.applications.Starlette()`, no intermediate dict
- **Server**: removed `httpx` from `[starlette]` extras — not needed at runtime, only for testing via Starlette's `TestClient`

## [0.0.16] — 2026-04-02

StarletteServer — production-grade async ASGI server with full uvicorn integration.

### Added

- **Server**: `StarletteServer` — production-grade async server using Starlette + Uvicorn with SSE streaming, thread-offloaded sync pillar code via `anyio.to_thread.run_sync`, route prefix support, and ASGI `__call__` on `Chatnificent` for direct `uvicorn app:app` usage
- **Server**: `Starlette.run(**kwargs)` passes all keyword arguments transparently to `uvicorn.run()` — port, reload, workers, SSL, timeouts, etc. Auto-resolves import string from `__main__` so `reload=True` and `workers=N` just work
- **Server**: shared helper methods on `Server` base class — `_build_conversation_title()`, `_extract_last_response()`, `_is_llm_streaming()`, `_render_messages()`, `_render_conversations()` — eliminating duplication across DevServer and StarletteServer
- **Server**: `starlette` optional dependency group (`pip install chatnificent[starlette]`)
- **Init**: `Chatnificent.__call__` ASGI protocol support — delegates to `server.asgi_app` for StarletteServer
- **Store**: thread-safe locking for InMemory store (`threading.Lock` around all dict mutations)
- **Docs**: concurrency guidelines added to AGENTS.md
- **Docs**: prioritization principles and examples design constraints moved from ROADMAP to AGENTS.md
- **Docs**: ROADMAP cleaned up — all completed Phase 0 items removed (forward-looking only), examples tables synced with shipped state
- **Init**: graceful LLM fallback — `Chatnificent.__init__` now catches both `ImportError` (SDK not installed) and `Exception` (SDK misconfigured, e.g. missing API key), falling back to Echo with distinct warnings
- **Init**: test coverage for misconfigured-provider fallback path (`test_echo_llm_fallback_on_misconfigured_provider`)

### Changed

- **Server**: DevServer refactored to use shared base class helpers via `_server` property
- **LLM**: replaced `_last_request_payload` class variable with `build_request_payload()` method — eliminates shared mutable state across requests

### Fixed

- **Server**: DashServer auth aligned with DevServer `session_id` contract
- **Layout**: added missing `USER_ROLE` import in DashLayout
- **LLM**: Gemini `create_assistant_message()` now preserves `thought_signature` on function-call parts, fixing `400 INVALID_ARGUMENT` when replaying tool-calling conversations

## [0.0.15] — 2026-03-26

Display enrichment architecture — layouts can now transform messages and conversations for display without mutating the canonical conversation history.

### Added

- **Layout**: `render_messages()` and `render_conversations()` methods for display-time transformation of messages and sidebar entries
- **Layout**: `_is_rtl()` helper for RTL text detection
- **Store**: `save_raw_api_request()` and `save_raw_api_response()` for raw API payload persistence (File and SQLite)
- **Store**: `save_file()`, `load_file()`, `list_files()` for conversation-scoped sidecar file storage (File and SQLite)
- **Store**: `load_raw_api_requests()` and `load_raw_api_responses()` for reading back raw payloads
- **Engine**: `_after_save()` hook — fires after conversation persistence, useful for generating sidecar files (titles, summaries)
- **Engine**: `_save_raw_exchange()` seam and `_normalize_raw_payload()` for raw API logging
- **Server**: `_render_messages_for_display()` and `_render_conversations_for_display()` — delegates to layout for display enrichment
- **DefaultLayout**: `default.html` supports sidecar file rendering in sidebar and message area
- **Examples**: 6 Tier 4 display enrichment examples — `usage_display`, `usage_display_multi_provider`, `conversation_title`, `conversation_summary`, `display_redaction`, `web_search`
- **Examples**: smoke tests for all new examples
- **Examples**: Tier 4 section in examples README

### Fixed

- **Store**: SQLite `_connect()` context manager now properly closes connections (previously `with sqlite3.connect()` only committed transactions, never closed)

## [0.0.14] — 2026-03-22

Canonical examples, documentation alignment, and foundation hardening.

### Added

- **Examples**: 10 standalone examples with PEP 723 metadata across Tiers 1–3 (quickstart, llm_providers, ollama_local, openrouter_models, persistent_storage, tool_calling, system_prompt, multi_tool_agent, single_user, auto_title)
- **Examples**: README with accurate index and run commands
- **Examples**: `test_examples.py` smoke tests for all examples
- **LLM**: Gemini streaming support — `extract_stream_delta()`, `generate_content_stream()`
- **Layout**: RTL auto-detection in DevServer (`default.html` vanilla JS)

### Changed

- **Server**: default port changed to 7777
- **Store**: removed `get_next_conversation_id()` from Store contract — replaced with short UUIDs (`uuid4().hex[:8]`)

### Fixed

- **Store**: persist full message dicts as JSON blobs in SQLite (preserving `tool_calls`, thinking blocks, provider-specific keys)
- **Store**: path traversal validation in File store — sanitize IDs, reject absolute paths, verify resolved paths stay inside `base_dir`
- **Server**: DevServer now delegates identity and deep-link behavior through Auth and URL pillars
- **Auth**: Anonymous provider respects `session_id`
- **LLM**: `extract_content()` returns `None` on tool-call-only responses (no text content)
- **Init**: docstring drift `File(directory=...)` → `File(base_dir=...)`
- **Engine**: `create_assistant_message()` verified consistent in both streaming and non-streaming paths
- **Engine**: streaming raw response saving — collect `raw_chunks` in streaming path

## [0.0.13] — 2026-03-05

The release that establishes Chatnificent's identity: zero dependencies, streaming by default, and a stdlib server that just works.

### Added

- **Engine**: `handle_message_stream()` — SSE streaming with token-by-token delivery and status updates for tool calls
- **Server**: DevServer `/api/chat` endpoint serves streaming responses via Server-Sent Events
- **Layout**: DefaultLayout SSE client — vanilla JS consumes SSE stream with live rendering
- **LLM**: `extract_stream_delta()` on all providers (OpenAI-compat, Anthropic, Ollama, Echo)
- **LLM**: streaming as default — all providers use `stream=True`; opt out with `stream=False`

### Changed

- **Engine**: renamed `Synchronous` → `Orchestrator`
- **README**: rewritten with problem/solution positioning, progressive disclosure, DevServer-first language

### Fixed

- **Store**: InMemory `save_conversation` handles new users via `setdefault`

[Unreleased]: https://github.com/eliasdabbas/chatnificent/compare/v0.0.18...HEAD
[0.0.18]: https://github.com/eliasdabbas/chatnificent/compare/v0.0.17...v0.0.18
[0.0.17]: https://github.com/eliasdabbas/chatnificent/compare/v0.0.16...v0.0.17
[0.0.16]: https://github.com/eliasdabbas/chatnificent/compare/v0.0.15...v0.0.16
[0.0.15]: https://github.com/eliasdabbas/chatnificent/compare/v0.0.14...v0.0.15
[0.0.14]: https://github.com/eliasdabbas/chatnificent/compare/v0.0.13...v0.0.14
[0.0.13]: https://github.com/eliasdabbas/chatnificent/releases/tag/v0.0.13
