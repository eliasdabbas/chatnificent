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

## Tier 3 — Customization

Extend the framework: custom engines, hooks, auth.

| # | Example | Purpose | Dependencies |
|---|---------|---------|--------------|
| 9 | [single_user.py](single_user.py) | SingleUser auth + SQLite — personal chat with history | `chatnificent[openai]` |
| 10 | [auto_title.py](auto_title.py) | Auto-name conversations from the first exchange via `_before_save` hook | `chatnificent[openai]` |

## Tier 4 — Display Enrichment

Keep canonical history clean while enriching what the UI shows.

| # | Example | Purpose | Dependencies |
|---|---------|---------|--------------|
| 11 | [usage_display.py](usage_display.py) | Minimal OpenAI-only example: read raw API responses and append token usage in the transcript | `chatnificent[openai]` |
| 12 | [usage_display_multi_provider.py](usage_display_multi_provider.py) | Production-style version: support OpenAI, Anthropic, and Gemini usage payloads | `chatnificent[openai,anthropic,gemini]` |
| 13 | [conversation_title.py](conversation_title.py) | Generate `conversation_title.txt` with a real LLM in `_after_save` and render it in the sidebar | `chatnificent[openai,anthropic,gemini]` |
| 14 | [conversation_summary.py](conversation_summary.py) | Append summaries to `summaries.md` in `_after_save` and render the latest one above the transcript | `chatnificent[openai,anthropic,gemini]` |
| 15 | [web_search.py](web_search.py) | Use Gemini web search, read raw API responses in `render_messages()`, and render sources as a simple Markdown list under the answer | `chatnificent[gemini]` |
| 16 | [display_redaction.py](display_redaction.py) | Use built-in regex rules to mask emails, phones, and card numbers in the visible transcript only | `chatnificent[anthropic]` |
