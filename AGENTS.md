# Coding Agent Instructions for Chatnificent

## Quick Start for Agents

**Chatnificent** 

> LLM chat app framework
> Minimally complete. Maximally hackable.

```python
import chatnificent as chat
app = chat.Chatnificent()
app.run()  # http://127.0.0.1:7777
```

Run with: `uv run app.py`

## Core Architecture: 9 Pillars

Chatnificent uses **dependency injection** with 9 configurable "pillars". Each pillar has an abstract base class and concrete implementations:

| Pillar | File | Purpose | Default | Available |
|--------|------|---------|---------|-----------|
| **Server** | `server.py` | HTTP transport | `DevServer` | DevServer, DashServer |
| **Layout** | `layout.py` | UI rendering | `DefaultLayout` | DefaultLayout, Bootstrap, Mantine, Minimal |
| **LLM** | `llm.py` | LLM API calls | `OpenAI` / `Echo` | OpenAI, Anthropic, Gemini, OpenRouter, DeepSeek, Ollama, Echo |
| **Store** | `store.py` | Conversation persistence | `InMemory` | InMemory, File, SQLite |
| **Auth** | `auth.py` | User identification | `Anonymous` | Anonymous, SingleUser |
| **Engine** | `engine.py` | Request orchestration | `Orchestrator` | Orchestrator |
| **Tools** | `tools.py` | Function calling | `NoTool` | PythonTool, NoTool |
| **Retrieval** | `retrieval.py` | RAG/context | `NoRetrieval` | NoRetrieval |
| **URL** | `url.py` | Route parsing | `PathBased` | PathBased, QueryParams |

The LLM default is auto-detected: if the OpenAI SDK is installed and `OPENAI_API_KEY` is set, it uses `OpenAI`. Otherwise it falls back to `Echo` (zero-dep mock). All LLM providers stream by default (`stream=True` in `default_params`).

## Philosophy: Minimally Complete, Maximally Hackable

Each pillar is designed with **atomic methods** that operate independently without dependencies on other pillars. This enables true pluggability where any implementation can be swapped without affecting others. Pillars communicate only through abstract interfaces.

For detailed pillar interfaces, engine orchestration, data models, and customization patterns, load the **pillar-dev** skill.

For server endpoint contracts and HTTP/SSE transport details, load the **server-dev** skill.

## Development Commands

Use `uv` to run all installation, running apps, and any Python commands.

```bash
# Run main app
uv run app.py

# Run tests
uv run pytest tests/

# Install dependencies
uv sync

# Install in development mode
uv pip install -e .
```

## Agent Development Guidelines

### Core Principles
- **Atomic Methods**: Each pillar method does ONE thing only
- **Minimal Changes**: Implement exactly what's requested, nothing more
- **Incremental Development**: Build → Test → Verify → Next feature
- **Challenge Bad Ideas**: Question requirements if there's a better approach
- **Concurrency Awareness**: Design every feature as if it will run under a multi-threaded server

### Prioritization Principles

1. **Fix before you build.** Technical debt erodes trust. A broken SQLite store or a stale AGENTS.md costs more credibility than a missing feature.
2. **Protect the zero-dep core.** Every decision should ask: "Does this keep the zero-dependency experience intact?" It's the single sharpest differentiator.
3. **The server handles async, not the engine.** Don't over-engineer the engine with async internals. Let async servers (FastAPI, Starlette) handle concurrency — the engine focuses on per-request orchestration.
4. **Streaming is the default.** Every LLM provider streams by default. The delightful experience is immediate. Opt out with `stream=False`.
5. **MCP over most feature work.** MCP is an ecosystem play (force multiplier). Voice is lower priority; file upload can move earlier when it closes genuine multimodal parity gaps across major providers.
6. **Examples are documentation.** A working example teaches more than a page of API docs. Invest heavily in `/examples` and ship them as agent skills.
7. **Pillar hardening is continuous.** Don't treat it as a one-time phase gate — every release should improve edge cases, error messages, and validation.
8. **Ship the boring stuff.** Thread safety, proper error handling, accurate docs — these are what make a framework trustworthy. They're not exciting, but they're what separates a toy from a tool.
9. **Server extensibility matters.** Provide a solid reference implementation with clear extension points. The goal is consistent behavior across all servers while remaining easy to extend or replace entirely.
10. **Name the contract precisely.** Be explicit about which layer is canonical for exact provider fidelity (raw logs) and which layer exists to make orchestration pluggable (`Conversation.messages`).

### Persistence Contract

- **Raw API logs** (`raw_api_requests.jsonl`, `raw_api_responses.jsonl`) are the exact provider-fidelity layer — auditing, debugging, and advanced hacking
- **`Conversation.messages`** is the orchestration layer — the framework's working state for replaying conversations through the engine. Provider-native dict shapes are preserved best-effort, but raw logs are the canonical fidelity source

### Concurrency Awareness

The current `DevServer` is single-threaded, but Chatnificent is designed to run behind production servers (FastAPI, Starlette, etc.) that dispatch requests across threads or async tasks. **Every new feature must be safe under concurrent access.**

**Design philosophy:** The server handles concurrency, not the engine. The engine stays synchronous and stateless per-request. Async servers wrap engine calls in their thread pools (e.g. `asyncio.to_thread()`). This means non-server pillar code (Engine, LLM, Store, Auth, Tools, Retrieval) doesn't need `async/await` — but it *does* need to be thread-safe, because multiple threads will call into it concurrently. The Server pillar itself may use `async def` freely (e.g. FastAPI route handlers, Starlette endpoints).

**Rules for all pillars:**
- **No per-request state on `self`.** Never stash request-specific data on the instance (class variables, module globals, or instance variables like the old `_last_request_payload`). All pillar instances are shared across requests — mutable instance state is a race condition. Pass per-request data through the call stack or return it.
- **Immutable config on `self` is fine.** Constructor parameters (model name, API key, db path, feature flags) are set once and never mutated — no lock needed.
- **Lock intentionally shared state.** When a pillar *must* hold shared mutable state (e.g., InMemory store's conversation dict), protect it with `threading.Lock` or finer-grained locks. This is **data-integrity locking**, not concurrency orchestration. Reference patterns: `InMemory` store uses a single lock; `File` store uses per-conversation write locks.
- **Don't orchestrate concurrency.** No request queuing, no `async/await`, no thread pools inside pillars. Concurrency orchestration (thread dispatch, async event loops, request routing) is the server's job.
- **Engine methods must stay stateless per-request.** No side effects on `self` during `handle_message()` / `handle_message_stream()` beyond delegating to the Store pillar.
- **New pillar implementations must be thread-safe.** Whether it's a new Store, LLM, Auth, or any other pillar — if it holds mutable instance state, protect it.
- **Ask the concurrency question.** For every new piece of state: *"What happens if two requests hit this at the same time?"* If the answer isn't "nothing, it's request-local," reconsider the design — prefer stateless approaches over adding locks.
- **Test concurrent access.** New stateful components should include a multi-threaded test (using `threading.Thread` or `concurrent.futures`) that exercises parallel access. Sequential "concurrent" tests are insufficient.

### Code Style
- Dash component IDs: `snake_case`
- CSS classes/IDs: `kebab-case`  
- No unnecessary comments (code should be self-explanatory)
- If we are to use comments, they should be use to explain *why* we are doing something not what.
- Use `uv run` instead of `python`

### File Structure
```
src/chatnificent/
├── __init__.py          # Main Chatnificent class + pillar imports
├── _callbacks.py        # Dash callbacks (DashServer only)
├── server.py            # Server pillar (DevServer, DashServer)
├── models.py            # Conversation dataclass, role constants
├── auth.py              # User identification (Anonymous, SingleUser)
├── store.py             # Conversation persistence (InMemory, File, SQLite)
├── layout.py            # UI rendering (DefaultLayout, Bootstrap, Mantine, Minimal)
├── llm.py               # LLM providers (OpenAI, Anthropic, Gemini, etc.)
├── engine.py            # Request orchestration (Orchestrator)
├── tools.py             # Function calling (PythonTool, NoTool)
├── retrieval.py         # RAG/context retrieval (NoRetrieval)
├── url.py               # URL parsing (PathBased, QueryParams)
└── templates/
    └── default.html     # DevServer HTML/JS chat UI
```

### Git Guidelines

**Staging:**
- Never use `git add .` or `git add -A` — always add files individually and consciously
- Review what you're staging: each file should belong to the commit's logical purpose

**Meaningful units of change:**
- Each commit should represent **one logical step** — a feature + its tests is one commit; an unrelated formatting fix belongs in a separate commit
- Ask: "if I revert this commit, does exactly one coherent thing disappear?" If not, split it
- Tests and implementation can (and usually should) live in the same commit when they're part of the same feature

**Conventional Commits with pillar scopes:**

Use [Conventional Commits](https://www.conventionalcommits.org/) format. Use pillar names as scopes where applicable:

```
feat(llm): add Gemini streaming support
fix(store): handle empty conversation edge case
test(engine): add tool loop coverage
refactor(auth): simplify user ID resolution
docs: update AGENTS.md with TDD guidelines
chore: upgrade pytest-cov dependency
```

Allowed prefixes: `feat`, `fix`, `test`, `docs`, `refactor`, `chore`

### Red/Green TDD

All new features and bug fixes **must** follow Red/Green TDD. This is a workflow discipline — seeing tests fail before writing implementation proves the tests are valid and not tautological.

**Step-by-step recipe:**

1. **Write failing tests first** — cover the expected interface, edge cases, and error paths. Mock pillar dependencies so tests are isolated.
2. **Run tests → confirm red** — every new test must fail. If a test passes immediately, it's not testing new behavior.
3. **Write minimal code to pass** — implement just enough to make the tests green. Resist the urge to add unrequested functionality.
4. **Run tests → confirm green** — all tests (new and existing) must pass.
5. **Refactor if needed** — clean up the implementation while keeping tests green.

**What "comprehensive" means:**
- Test the abstract interface contract (not internal implementation details)
- Mock pillar dependencies to keep tests isolated and fast
- Cover edge cases: empty inputs, missing data, invalid state
- For LLM pillars: test both streaming and non-streaming paths

**Test coverage:**

Maintain **90%+ project-wide** test coverage. After making changes, run:

```bash
uv run pytest --cov=chatnificent --cov-report=term-missing
```

The `term-missing` flag highlights uncovered lines — use it to identify what still needs tests. Do not merge changes that drop coverage below the threshold.

### Development Conventions
- **Modular Architecture**: Consider which pillar new functionality belongs to
- **Docstrings**: Use [Numpy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html) for consistency

### Component IDs

**DevServer / DefaultLayout** (`templates/default.html`) — kebab-case HTML IDs:
`sidebar-toggle`, `new-chat-btn`, `convo-list`, `messages`, `input`, `send`, `sidebar`, `theme-toggle`

**DashServer / Dash layouts** (`_callbacks.py`) — snake_case Dash component IDs:
`url_location`, `messages_container`, `input_textarea`, `submit_button`, `new_conversation_button`, `conversations_list`, `sidebar`, `sidebar_toggle`, `status_indicator`

## Examples Design Constraints

All examples live in `/examples/` as standalone scripts. For detailed guidelines on writing examples, load the **example-app** skill.
