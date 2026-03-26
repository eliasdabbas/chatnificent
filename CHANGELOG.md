# Changelog

All notable changes to Chatnificent will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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

Canonical examples and documentation alignment.

### Added

- **Examples**: 10 standalone examples with PEP 723 metadata across Tiers 1–3 (quickstart, llm_providers, ollama_local, openrouter_models, persistent_storage, tool_calling, system_prompt, multi_tool_agent, single_user, auto_title)
- **Examples**: README with accurate index and run commands
- **Examples**: `test_examples.py` smoke tests for all examples
- **LLM**: Gemini streaming support — `extract_stream_delta()`, `generate_content_stream()`

### Fixed

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

[Unreleased]: https://github.com/eliasdabbas/chatnificent/compare/v0.0.15...HEAD
[0.0.15]: https://github.com/eliasdabbas/chatnificent/compare/v0.0.14...v0.0.15
[0.0.14]: https://github.com/eliasdabbas/chatnificent/compare/v0.0.13...v0.0.14
[0.0.13]: https://github.com/eliasdabbas/chatnificent/releases/tag/v0.0.13
