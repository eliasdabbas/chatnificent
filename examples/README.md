# Chatnificent Examples

Every example is a standalone Python script with [PEP 723](https://peps.python.org/pep-0723/) inline metadata. Run any of them directly — no cloning or installing required.

## Running Examples

**From a cloned repo:**
```bash
uv run --script examples/quickstart.py
```

**Remotely (no download needed — installs from PyPI automatically):**
```bash
uv run --script https://github.com/eliasdabbas/chatnificent/raw/main/examples/quickstart.py
```

---

## Tier 1 — Basics

Get up and running with different LLM providers.

| # | Example | Purpose | Dependencies |
|---|---------|---------|--------------|
| 1 | [quickstart.py](quickstart.py) | Zero-dep Echo chat — the 3-line app | `chatnificent` |
| 2 | [llm_providers.py](llm_providers.py) | Switch between OpenAI, Anthropic, Gemini | `chatnificent[openai,anthropic,gemini]` |
| 3 | [ollama_local.py](ollama_local.py) | Local inference via Ollama — no API key needed | `chatnificent[ollama]` |
| 4 | [openrouter_models.py](openrouter_models.py) | Access many models through one API via OpenRouter | `chatnificent[openai]` |

## Tier 2 — Features

Core capabilities: storage, tools, system prompts.

| # | Example | Purpose | Dependencies |
|---|---------|---------|--------------|
| 5 | [persistent_storage.py](persistent_storage.py) | File and SQLite stores — conversations survive restarts | `chatnificent` |
| 6 | [tool_calling.py](tool_calling.py) | Register Python functions as LLM tools | `chatnificent[openai]` |
| 7 | [system_prompt.py](system_prompt.py) | Give your AI a personality in 3 lines | `chatnificent[openai]` |
| 8 | [multi_tool_agent.py](multi_tool_agent.py) | Multi-tool agentic loop — LLM picks the right tool | `chatnificent[openai]` |
| 9 | [memory_tool.py](memory_tool.py) | Persistent LLM memory — remember facts and preferences across conversations | `chatnificent[anthropic,gemini]` |
| 10 | [memory_tool_multi_user.py](memory_tool_multi_user.py) | Per-user memory isolation via `ContextVar` + `UserAwareAuth` | `chatnificent[anthropic,gemini]` |

## Tier 3 — Customization

Extend the framework: custom engines, hooks, auth.

| # | Example | Purpose | Dependencies |
|---|---------|---------|--------------|
| 11 | [single_user.py](single_user.py) | SingleUser auth + SQLite — personal chat with history | `chatnificent[openai]` |
| 12 | [auto_title.py](auto_title.py) | Auto-name conversations from the first exchange via `_before_save` hook | `chatnificent[openai]` |

## Tier 4 — Display Enrichment

Keep canonical history clean while enriching what the UI shows.

| # | Example | Purpose | Dependencies |
|---|---------|---------|--------------|
| 13 | [usage_display.py](usage_display.py) | Minimal OpenAI-only example: read raw API responses and append token usage in the transcript | `chatnificent[openai]` |
| 14 | [usage_display_multi_provider.py](usage_display_multi_provider.py) | Production-style version: support OpenAI, Anthropic, and Gemini usage payloads | `chatnificent[openai,anthropic,gemini]` |
| 15 | [conversation_title.py](conversation_title.py) | Generate `conversation_title.txt` with a real LLM in `_after_save` and render it in the sidebar | `chatnificent[openai,anthropic,gemini]` |
| 16 | [conversation_summary.py](conversation_summary.py) | Append summaries to `summaries.md` in `_after_save` and render the latest one above the transcript | `chatnificent[openai,anthropic,gemini]` |
| 17 | [web_search.py](web_search.py) | Use Gemini web search, read raw API responses in `render_messages()`, and render sources as a simple Markdown list under the answer | `chatnificent[gemini]` |
| 18 | [display_redaction.py](display_redaction.py) | Use built-in regex rules to mask emails, phones, and card numbers in the visible transcript only | `chatnificent[anthropic]` |

## Tier 5 — Production & Deployment

Run Chatnificent on a production-grade async server with Starlette and Uvicorn.

| # | Example | Purpose | Dependencies |
|---|---------|---------|--------------|
| 19 | [starlette_quickstart.py](starlette_quickstart.py) | One-line swap to a production async server | `chatnificent[starlette]` |
| 20 | [starlette_server_options.py](starlette_server_options.py) | Custom routes, middleware, lifespan hooks, and error handlers | `chatnificent[starlette]` |
| 21 | [starlette_uvicorn_options.py](starlette_uvicorn_options.py) | Configure uvicorn: workers, reload, SSL, host/port | `chatnificent[starlette]` |
| 22 | [starlette_multi_mount.py](starlette_multi_mount.py) | Mount multiple independent chat apps on one website | `chatnificent[starlette]` |

## Tier 6 — OpenAI Responses API

Route Chatnificent through OpenAI's `responses.create` endpoint via a small subclass. Each example adds one hosted tool on top.

| # | Example | Purpose | Dependencies |
|---|---------|---------|--------------|
| 23 | [openai_responses.py](openai_responses.py) | Route through OpenAI's `responses.create` in an 8-line subclass | `chatnificent[openai]` |
| 24 | [openai_responses_website_search.py](openai_responses_website_search.py) | Domain-restricted research assistant via the hosted web-search tool | `chatnificent[openai]` |
| 25 | [openai_responses_image_generator.py](openai_responses_image_generator.py) | Inline image generation every turn via the hosted `image_generation` tool | `chatnificent[openai]` |
| 26 | [openai_responses_image_studio.py](openai_responses_image_studio.py) | Multi-turn image studio: live streaming previews, iterative editing, revised prompt display, JPEG sidecar persistence | `chatnificent[openai]` |
| 27 | [openai_responses_interactive_search.py](openai_responses_interactive_search.py) | Interactive web search: reasoning effort, domain presets, and force-search toggle — live UI controls wired to `responses.create` kwargs | `chatnificent[openai]` |

## Tier 7 — UI Interactions

Bind HTML controls directly to LLM call parameters — no page reload, no custom server code.

| # | Example | Purpose | Dependencies |
|---|---------|---------|--------------|
| 28 | [ui_interactions.py](ui_interactions.py) | Bind a single `<select>` to `max_completion_tokens` — the minimal one-control pattern | `chatnificent[openai]` |

## OpenAI Cookbook — From Cookbook to Production

Production implementations of [OpenAI Cookbook](https://github.com/openai/openai-cookbook) notebooks. The cookbook teaches the mechanics; these files are what you actually ship.

| # | Example | Cookbook Notebook | Dependencies |
|---|---------|------------------|--------------|
| 29 | [How_to_call_functions_with_chat_models.py](How_to_call_functions_with_chat_models.py) | [How_to_call_functions_with_chat_models.ipynb](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_call_functions_with_chat_models.ipynb) | `chatnificent[openai]` |
