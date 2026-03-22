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
