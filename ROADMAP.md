# Chatnificent — Roadmap

## Vision

Chatnificent should be the go-to framework for Python developers who want a robust, hackable solution for building LLM chat apps. The guiding principle:

> **Minimally complete. Maximally hackable.**

The North Star: one or more major LLM providers (OpenAI, Anthropic, Google) adopt Chatnificent in their official cookbooks — the way many have adopted `uv` as their reference tool runner. Their cookbook examples currently stitch together minimally complete prototypes that *just work* as simple demos. That's great, but it's not maximally hackable. Chatnificent fills that gap: a framework that lets you build any LLM chat app without getting in the way.

If a major provider starts using Chatnificent for their demos, we know we've arrived.

## Strategic Capabilities

These are non-negotiable features the framework *must* support to fulfill its promise. They cut across all phases — they're constraints, not tasks.

1. **Zero-dependency quickstart** — `pip install chatnificent` (no extras) must always produce a fully working chat app. The Echo LLM + DevServer + DefaultLayout stack is the proof point.
2. **Zero-code LLM upgrade** — The default constructor auto-detects installed providers: if the OpenAI SDK is present and `OPENAI_API_KEY` is set, it just works — no code change needed. If nothing is installed, it falls back to Echo. This zero-code upgrade path must be preserved and extended to other providers.
3. **Streaming everywhere** — Streaming is the default for all LLM providers. Token-by-token response delivery is the expected UX. Non-streaming is the opt-in exception (`stream=False`).
4. **Multi-provider parity** — Every feature (tool calling, streaming, message persistence, raw API logging) must work identically across all supported LLM providers. Provider-native message formats are preserved — no lossy translation layer.
5. **True pillar independence** — Any pillar implementation can be swapped without touching any other pillar. The engine never inspects message internals. Callbacks use only abstract methods.
6. **Programmatic layout composition** — Developers must be able to insert, append, and modify any part of the UI (sidebar, chat area, input area, header) from Python, plus inject JS/CSS declaratively — without touching raw HTML (although default.html is always available should they need manually hack any way they want).
7. **Server-agnostic orchestration** — The engine handles per-request orchestration. The server handles transport and concurrency. Async comes from the server (FastAPI, Starlette), not from rewriting the engine.
8. **MCP ecosystem integration** — The framework must connect to the MCP ecosystem as both client (consume MCP tools) and eventually host (expose capabilities via MCP).
9. **Production-grade persistence** — All store implementations must round-trip full conversations faithfully, including tool calls, thinking blocks, and any provider-specific message keys.

## Current State (v0.0.x — Alpha)

Chatnificent is a young framework with a genuinely zero-dependency core. `pip install chatnificent` (no extras) gives you a working chat app with the stdlib-only `DevServer`, `Echo` LLM, and pure-HTML `DefaultLayout`. For real use, install optional extras to unlock provider SDKs.

### Architecture clarifications
- **Raw API logs are the fidelity layer** — `raw_api_requests.jsonl` and `raw_api_responses.jsonl` preserve the exact provider payloads for auditing, debugging, and advanced hacking
- **`Conversation.messages` is the orchestration layer** — it is the framework's working state for replaying conversations through the engine; preserving exact provider-native shapes here too is desirable, but should be treated as a separate, testable quality bar
- **Server parity is not complete yet** — Dash and DevServer do not currently share identical auth/routing semantics, so "swap the server only" is not yet fully true in practice

### What's working well
- **9-pillar architecture** with clean ABCs and dependency injection
- **LLM pillar**: OpenAI, Anthropic, Gemini, Ollama, OpenRouter, DeepSeek, Echo — all with tool calling and streaming
- **Store pillar**: InMemory, File (atomic writes, per-conversation locks), SQLite (full message dict persistence as JSON blobs)
- **Engine**: `Orchestrator` with agentic loop and seam/hook extensibility pattern; stateless short-UUID conversation IDs (`uuid4().hex[:8]`)
- **Server**: DevServer (stdlib, zero-dep) is the primary server; DashServer available for Dash-based UIs
- **Layout**: Zero-dep DefaultLayout (HTML/JS) is the primary layout; Bootstrap, Mantine, Minimal available for Dash
- **Tools**: PythonTool with auto-schema generation from type hints + docstrings
- **Auth**: Anonymous (UUID-based), SingleUser
- **URL**: PathBased, QueryParams
- **RTL support** — auto-detect via clientside callback in Dash layouts and via vanilla JS in DevServer (`default.html`): textarea input direction + rendered messages + streaming bubbles

### What's incomplete or stubbed
- **Retrieval pillar**: only `NoRetrieval` stub — no real RAG implementation
- **Retrieval semantics are underspecified**: current context injection is effectively "insert once" rather than explicitly per-turn, ephemeral, or persisted by contract
- **Optional pillar capabilities are implicit**: raw API logging, streaming detection, and Dash layout requirements live in implementations rather than in a documented capability layer
- **No e2e tests** (empty `tests/e2e/`)
- **Async server implementations**: DevServer is stdlib single-threaded — FastAPI, Starlette, and Flask servers are missing
- **Layout composition API**: no programmatic way to insert/append/modify UI regions from Python; escape hatch is manual HTML editing only
- **Production-grade auth**: only trivial Anonymous and SingleUser implementations
- **File upload & multimodal**: no support for image/document inputs across the pipeline
- **MCP documentation**: tools work but lack canonical reference implementations and integration guide

### What's aspirational (in README/docs but not yet built)
- FastAPI, Flask, Starlette server implementations
- MCP integration (note: MCP tools already work through the existing agentic tool loop — what's missing is documentation and canonical reference implementations)
- File upload / multimodal support
- Voice input
- Production-grade auth (OAuth, JWT)

## Competitive Landscape

Understanding why someone would choose Chatnificent over alternatives should drive every priority decision.

| Alternative | Relationship to Chatnificent |
|-------------|------------------------------|
| **Chainlit** | Most direct competitor. More mature, async-first, FastAPI-based, richer UI. More features but more opinionated. |
| **Streamlit** | Simpler but less hackable. Re-run-on-every-interaction model is fundamentally different. |
| **Gradio** | ML-focused, good for demos, less suitable for production chat apps. |
| **Open WebUI** | Full product, not a framework. Much heavier. |
| **Mesop** | Similar philosophy (Python UI) but less chat-specific. |

### Chatnificent's differentiators (protect these)
1. **Zero-dependency core** — unique in the space
2. **True pillar pluggability** — swap any component independently without touching others
3. **Provider-native message formats** — no lossy translation layer
4. **Streaming by default** — delightful out-of-the-box UX
5. **3-line quickstart** that actually works with zero installs

### Weaknesses to address
1. No async server implementations yet (Chainlit is async-first) — DevServer is stdlib single-threaded; async comes via FastAPI/Starlette server subclasses (Phase 1)
2. No real auth beyond trivial implementations
3. No RAG implementation (just an interface)
4. No multimodal support
5. Very early alpha with small community

---

## Phased Milestones

### Phase 0 — Solid Foundation ✅ (mostly complete)
> Make what exists *actually* production-quality. No new features until the base is trustworthy.

**Remaining technical debt:**
- [ ] Decide retrieval semantics before adding a real backend: ephemeral per-turn context, persisted system context, retrieval-as-tool, or an explicit combination. Test the chosen contract across multi-turn conversations
- [ ] Add an explicit seam for per-turn retrieval injection so retrieval can evolve without rewriting the agent loop

**Core capability gaps:**
- [ ] Implement at least one real Retrieval backend (e.g., simple in-memory vector search, or ChromaDB integration)
- [ ] Add file upload support to the Layout + Engine (multimodal messages are now standard across all major LLMs)

**Quality:**
- [ ] Write e2e tests for the critical happy path (send message → get response → conversation persists)
- [ ] Add contract tests for server parity (Auth/URL behavior), final assistant-message persistence across providers, raw-log persistence, and File store path hardening
- [ ] Add proper API reference documentation (beyond AGENTS.md)

**Gate:** All existing tests pass. SQLite round-trips agentic conversations correctly. At least one example runs zero-dep, at least one with each major provider.

---

### Examples

> Examples are documentation. A working example teaches more than a page of API docs.

All examples live in `/examples/` as standalone scripts with [PEP 723](https://peps.python.org/pep-0723/) inline metadata — run any of them with `uv run examples/<name>.py`. Design constraints for authoring examples are in AGENTS.md.

#### Tier 1 — Basics (Getting Started)

| # | File | Purpose | Deps | Status |
|---|------|---------|------|--------|
| 1 | `quickstart.py` | Zero-dep Echo chat — the 3-line app | `chatnificent` | ✅ |
| 2 | `llm_providers.py` | Switch between OpenAI, Anthropic, Gemini (one file, comment/uncomment) | `chatnificent[openai,anthropic,gemini]` | ✅ |
| 3 | `ollama_local.py` | Local Ollama inference — no API key needed | `chatnificent[ollama]` | ✅ |
| 4 | `openrouter_models.py` | OpenRouter with model selection — access many models through one API | `chatnificent[openai]` | ✅ |

#### Tier 2 — Features (Core Capabilities)

| # | File | Purpose | Deps | Status |
|---|------|---------|------|--------|
| 5 | `persistent_storage.py` | File and SQLite stores — conversations survive restarts | `chatnificent` | ✅ |
| 6 | `tool_calling.py` | PythonTool with useful functions (weather, dice, datetime) | `chatnificent[openai]` | ✅ |
| 7 | `system_prompt.py` | Custom system prompt via engine seam (`_prepare_llm_payload`) | `chatnificent[openai]` | ✅ |
| 8 | `multi_tool_agent.py` | Multi-tool agentic loop — multiple tools, LLM picks the right one | `chatnificent[openai]` | ✅ |

#### Tier 3 — Customization (Extending the Framework)

| # | File | Purpose | Deps | Status |
|---|------|---------|------|--------|
| 9 | `single_user.py` | SingleUser auth + SQLite — personal chat with history | `chatnificent[openai]` | ✅ |
| 10 | `auto_title.py` | Auto-name conversations from the first exchange via `_before_save` hook | `chatnificent[openai]` | ✅ |

#### Tier 4 — Display Enrichment

Keep canonical history clean while enriching what the UI shows.

| # | File | Purpose | Deps | Status |
|---|------|---------|------|--------|
| 11 | `usage_display.py` | Minimal OpenAI-only: read raw API responses and append token usage | `chatnificent[openai]` | ✅ |
| 12 | `usage_display_multi_provider.py` | Production-style: OpenAI, Anthropic, and Gemini usage payloads | `chatnificent[openai,anthropic,gemini]` | ✅ |
| 13 | `conversation_title.py` | Generate title sidecar with LLM in `_after_save`, render in sidebar | `chatnificent[openai,anthropic,gemini]` | ✅ |
| 14 | `conversation_summary.py` | Append summaries to sidecar in `_after_save`, render above transcript | `chatnificent[openai,anthropic,gemini]` | ✅ |
| 15 | `web_search.py` | Gemini web search grounding, render sources from raw API responses | `chatnificent[gemini]` | ✅ |
| 16 | `display_redaction.py` | Mask emails/phones/cards in visible transcript only (stored messages unchanged) | `chatnificent[anthropic]` | ✅ |

#### Planned (not yet shipped)

| File | Purpose | Deps |
|------|---------|------|
| `streaming_control.py` | Toggle streaming on/off, show both UX modes | `chatnificent[openai]` |
| `conversation_export.py` | Export/import conversations as JSON using Store methods | `chatnificent` |
| `custom_engine.py` | Subclass Orchestrator — add hooks, logging, system prompts | `chatnificent[openai]` |
| `token_counter.py` | Read `raw_api_responses.jsonl` to track token usage | `chatnificent[openai]` |
| `dash_ui.py` | DashServer + Bootstrap layout — swap to a Dash-based UI | `chatnificent[dash]` |
| `cookbook_function_calling.py` | Function calling (cf. [OpenAI cookbook](https://cookbook.openai.com/examples/how_to_call_functions_with_chat_models)) | `chatnificent[openai]` |
| `cookbook_customer_support.py` | Customer support agent (cf. [Anthropic guide](https://docs.anthropic.com/en/docs/about-claude/use-case-guides/customer-support-agent)) | `chatnificent[anthropic]` |
| `cookbook_system_instructions.py` | System instructions (cf. [Gemini docs](https://ai.google.dev/gemini-api/docs/system-instructions)) | `chatnificent[gemini]` |

#### Navigation

`examples/README.md` groups all shipped examples by tier with one-line descriptions and run commands. It serves as the index page for browsing examples.

### Phase 1 — Developer Experience & Ecosystem
> Make it delightful to build with, and easy to discover.

**Server implementations:**
- [ ] Make DevServer complete and feature-rich as the canonical reference implementation
- [ ] Make DevServer's stream-`done` handler efficient: the JS client now upserts the sidebar entry directly from the done event data instead of making two extra HTTP requests (`refreshConvoList` + `loadConvo`). The full `refreshConvoList` fetch is only used on page load and explicit navigation
- [ ] Add `ThreadedDevServer` (stdlib `ThreadingHTTPServer`) as an additive server option that reuses the existing Dev handler and endpoint contract; position as medium-priority (interesting, not urgent)
- [ ] Decide server extensibility architecture: (a) extract a transport-agnostic request/service core that all servers (including DevServer) share as thin adapters, or (b) make DevServer the reference to subclass with shared utilities. The goal is consistent endpoint semantics, auth flow, URL handling, and streaming contract across all servers — while keeping it easy for developers to build their own server from scratch
- [ ] Implement `FastAPIServer`, `StarletteServer`, and `FlaskServer` using the chosen architecture
- [ ] These servers handle async request concurrency natively, wrapping engine calls appropriately — framework is "async out of the box" just by installing the server of your choice
- [ ] Evaluate `ThreadedDevServer` as the default zero-dependency server only after parity + safety gates pass (stable tests under concurrent load, InMemory thread-safety, and no regression in zero-dependency quickstart ergonomics)

**Pillar hardening (ongoing):**
- [ ] Treat this as a continuous effort, not a one-time task — every release should improve edge cases, error messages, and validation across all pillars
- [ ] Document optional capability protocols/mixins for extension points that intentionally live outside the minimal ABCs (raw API logging, streaming support detection, Dash layout contract, etc.)

**Auth:**
- [ ] Add OAuth/JWT auth implementation (even a basic one unlocks real multi-user deployments)

**URL pillar — versatility & customization:**
- [ ] Explore query-parameter-based URL schemes (`QueryParams`) as an alternative/addition to path-based routing — useful for passing additional state, supporting embedded iframes, or integrating with apps that control the URL path
- [ ] Ensure all servers work transparently with any URL pillar implementation (PathBased, QueryParams, or custom) — the URL pillar is the sole authority for parsing and building URLs

**Layout — programmatic composition API:**
- [ ] Design conceptual areas in the app (sidebar, input area, chat area, header, etc.) as named regions with stable IDs/markers
- [ ] Provide a Pythonic API for inserting/appending/prepending components to any region: `layout.sidebar.append(MyComponent())`, `layout.chat_area.prepend(Banner("Welcome!"))`
- [ ] Add `layout.add_scripts([...])` and `layout.add_css([...])` for declarative JS/CSS injection without touching HTML
- [ ] Keep `default.html` as the fully hackable escape hatch for those who want raw HTML/JS control

**Conversation titles:** (potential approach that needs to be verified)
- [ ] **Known debt: title derivation is duplicated in 3 places** — DevServer `_handle_list_conversations` (`[:30]`), DashServer callbacks (`[:40] + "..."`), and `default.html` JS `upsertSidebarEntry` (`slice(0, 30)`). All derive title from first user message with inconsistent truncation. The fix below collapses all three to one
- [ ] Add `title: Optional[str] = None` to `Conversation` dataclass — backward-compatible, no impact on existing code
- [ ] File store: change `messages.json` from raw array to `{"messages": [...], "title": "..."}` with transparent backward-compat (detect old format via `isinstance(data, list)`)
- [ ] SQLite store: add `title` column to `conversations` table (same migration pattern as `message_data`)
- [ ] Servers: prefer `convo.title` over truncated first user message in sidebar rendering (one-line fallback each in DevServer and Dash callbacks)
- [ ] Title generation stays in userland — the framework persists and displays titles, users set them via existing hooks like `_before_save`. No `_generate_title` seam needed

**Agent & AI integration:**
- [ ] Ship all examples as [agent skills](https://agentskills.io/) so AI IDEs can scaffold apps
- [ ] Keep AGENTS.md accurate and comprehensive (this is the AI-facing documentation — treat it as a first-class artifact)

**MCP integration:**
- [ ] MCP tools already work through the existing agentic tool loop — document this properly
- [ ] Create canonical reference implementations showing how to wire an MCP client as a `Tool` subclass
- [ ] Explore Chatnificent as MCP host (expose the chat app's capabilities via MCP)

**Documentation:**
- [ ] Dedicated documentation site (even a simple one — MkDocs/Sphinx with the pillar API reference)
- [ ] "Why Chatnificent?" page with explicit competitive positioning
- [ ] Migration/upgrade guide as the API stabilizes

**Gate:** A developer can `pip install chatnificent[default]`, scaffold an app with an AI agent, add MCP tools, and deploy it — all within an afternoon.

### Phase 2 — Adoption & Proof Points
> Show, don't tell. Prove the framework works by building real things with it.

**Cookbook examples:**
- [ ] Reimplement 5–10 cookbook examples from OpenAI, Anthropic, and Gemini using Chatnificent
- [ ] Side-by-side comparisons showing code reduction and hackability gains
- [ ] Publish as blog posts / tutorials

**Live showcase:**
- [ ] Launch chatnificent.com with hosted, interactive demos
- [ ] Gallery of community-built apps and custom pillars

**Community:**
- [ ] Publish to community channels (Python Weekly, Dash community, Hacker News, relevant subreddits)
- [ ] Pillar registry — a way to discover and share community implementations (Redis store, Postgres store, custom layouts, etc.)
- [ ] Contribution guide and issue templates

**Gate:** At least 3 cookbook examples live. Website up with interactive demos. Framework mentioned in at least one external publication or community discussion.

### Phase 3 — Production & Standards
> The framework is trusted for real workloads.

**Production apps:**
- [ ] Build 2–3 standalone production applications using real data sources
- [ ] Demonstrate real workflows: customer support bot, research assistant, data analysis chat, etc.

**Standards & emerging tech:**
- [ ] [MCP apps](https://apps.extensions.modelcontextprotocol.io/api/documents/Overview.html) support
- [ ] Voice input (speech-to-text prompt creation)
- [ ] Streaming tool calls (as providers roll out support)
- [ ] Structured output / JSON mode support across providers

**Scale & reliability:**
- [ ] Connection pooling, rate limiting, retry logic in LLM pillar
- [ ] Proper error taxonomy (retriable vs. fatal, user-facing vs. internal)
- [ ] Performance benchmarks and optimization

**Gate:** At least one production app serving real users. Framework handles concurrent users reliably. Versioned API with backward compatibility guarantees.

### Phase 4 — Cookbook Adoption (the North Star)
> LLM providers choose Chatnificent for their official examples.

- [ ] Maintain parity with each provider's latest API features (new models, new tool-calling modes, etc.)
- [ ] Provide PR-ready cookbook examples to provider DevRel teams
- [ ] Ensure the framework is so frictionless that using it is obviously easier than the alternative

This phase is an *outcome* of executing Phases 0–3 well, not a set of tasks to force.
