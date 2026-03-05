# Coding Agent Instructions for Chatnificent

## Quick Start for Agents

**Chatnificent** 

> LLM chat app framework
> Minimally complete. Maximally hackable.

```python
import chatnificent as chat
app = chat.Chatnificent()
app.run()  # http://127.0.0.1:8050
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

### Atomic Architecture Design

Each pillar is designed with **atomic methods** that operate independently without dependencies on other pillars. This enables true pluggability where any implementation can be swapped without affecting others.

**Pillar Independence:**
- **LLM**: `generate_response()` and `extract_content()` work with any message format
- **Store**: `save_conversation()`, `load_conversation()` are storage-agnostic
- **Auth**: `get_current_user_id()` works independently of other components
- **Layout**: `render()` renders regardless of LLM or storage backend
- **Server**: `create_server()` and `run()` work regardless of layout or LLM

**True Pluggability:**
Pillars communicate only through abstract interfaces, making the framework truly "minimally complete, maximally hackable."

## Key Interfaces for Agents

### 1. LLM Pillar (`llm.py`)
```python
class LLM(ABC):
    @abstractmethod
    def generate_response(self, messages: List[Dict[str, Any]], **kwargs) -> Any:
        """Generate response from LLM. Returns native provider response."""

    @abstractmethod
    def extract_content(self, llm_response: Any) -> str:
        """Extract text content from provider response."""

# Usage
import chatnificent as chat
app = chat.Chatnificent(llm=chat.llm.Anthropic(api_key="sk-..."))
```

### 2. Store Pillar (`store.py`)
```python
class Store(ABC):
    @abstractmethod
    def save_conversation(self, user_id: str, conversation: Conversation) -> None:
        """Save conversation to storage."""
        
    @abstractmethod 
    def load_conversation(self, user_id: str, convo_id: str) -> Optional[Conversation]:
        """Load conversation from storage."""

# Usage
app = chat.Chatnificent(store=chat.store.SQLite(db_path="chats.db"))
```

### 3. Server Pillar (`server.py`)
```python
class Server(ABC):
    @abstractmethod
    def create_server(self, **kwargs) -> None:
        """Initialize the HTTP server."""

    @abstractmethod
    def run(self, **kwargs) -> None:
        """Start serving requests."""
```

`DevServer` is the primary server — zero-dependency stdlib HTTP server with SSE streaming.
`DashServer` wraps Plotly Dash for use with Dash-based layouts (Bootstrap, Mantine, Minimal).

### 4. Layout Pillar (`layout.py`)
```python
class Layout(ABC):
    @abstractmethod
    def render(self) -> Any:
        """Render the layout. Returns HTML string (DevServer) or Dash component tree (DashServer)."""
```

`DefaultLayout` renders `templates/default.html` — a zero-dep vanilla HTML/JS chat UI for DevServer.
Dash-based layouts (`Bootstrap`, `Mantine`, `Minimal`) build Dash component trees and require `DashServer`.

## Engine Orchestration

The **Engine** pillar manages the request lifecycle and enables complex, multi-step agentic workflows. It orchestrates the interaction between pillars and handles tool calling loops.

### Agentic Loop Architecture

The engine can loop multiple times to allow the LLM to use tools, process results, and form a final answer:

1. **LLM generates response** (may include tool calls)
2. **Engine detects tool calls** and executes them via Tools pillar
3. **Tool results** are added to conversation history
4. **Loop continues** until LLM provides final answer

**Pillar Contracts:**
- The **LLM** pillar is responsible for parsing tool calls from its own native response format
- The **Tools** pillar is responsible for defining and executing tools
- The **Engine** provides status updates to the UI showing agent's internal state (e.g., "Running tool: ...")

### Two Engine Entry Points

- `handle_message()` — non-streaming path, returns a complete `Conversation`
- `handle_message_stream()` — streaming path, yields SSE event dicts (`{"event": "delta", "data": "..."}`)

The server routes between these based on `llm.default_params.get("stream", False)`.

### Customization via Hooks and Seams

```python
class CustomEngine(chat.engine.Orchestrator):
    # Override HOOKS for monitoring
    def _after_llm_call(self, llm_response: Any) -> None:
        tokens = getattr(llm_response, 'usage', 'N/A')
        print(f"Tokens used: {tokens}")

    # Override SEAMS for custom logic  
    def _prepare_llm_payload(self, conversation, context: Optional[str]):
        payload = super()._prepare_llm_payload(conversation, context)
        payload.insert(0, {"role": "system", "content": "Be concise."})
        return payload

app = chat.Chatnificent(engine=CustomEngine())
```

## Data Models (`models.py`)

### Native Message Dicts + Conversation Dataclass
```python
# Role constants
USER_ROLE = "user"
ASSISTANT_ROLE = "assistant"
SYSTEM_ROLE = "system"
TOOL_ROLE = "tool"
MODEL_ROLE = "model"  # Gemini uses "model" instead of "assistant"

@dataclass
class Conversation:
    id: str
    messages: list  # List[Dict[str, Any]] — provider-native dicts

    def copy(self, deep: bool = False) -> "Conversation": ...
```

Messages are **plain dicts** in each provider's native format. There is no
universal message schema — an OpenAI message looks different from an Anthropic
one, and that's intentional. The LLM pillar owns the shape of its own messages
via `create_assistant_message()` and `create_tool_result_messages()`.

## Integration and Orchestration

Chatnificent stores messages in each **provider's native dict format**. There is no universal intermediary format — an OpenAI conversation and an Anthropic conversation look different on disk, and that's by design.

The engine never inspects message internals. It only touches the minimal universal contract:
- Adds user messages via `{"role": "user", "content": text}`
- Delegates everything else to the LLM pillar's abstract methods

Each LLM concrete class is responsible for:
- `create_assistant_message()` — converting its native response into a persistable dict
- `create_tool_result_messages()` — formatting tool results for its own API
- `extract_content()` — pulling display text from its native response
- `is_tool_message()` — identifying its own tool-related messages

The **Tools** pillar outputs a standard JSON Schema tool definition. Each LLM's
`_translate_tool_schema()` converts that into the provider's native tool format
before calling the API.

This approach:
- Preserves full provider fidelity (thinking blocks, citations, etc.)
- Avoids lossy format translations
- Raw API responses are also saved to `raw_api_requests.jsonl` for auditing

## Customization Patterns

### Mix and Match Examples

**Research Setup:**
```python
import chatnificent as chat

app = chat.Chatnificent(
    llm=chat.llm.Anthropic(),
    store=chat.store.File(directory="./research_chats"),
    tools=chat.tools.PythonTool(),
)
```

**Enterprise Setup:**
```python
app = chat.Chatnificent(
    llm=chat.llm.OpenAI(),
    store=chat.store.SQLite(db_path="enterprise.db"),
    auth=chat.auth.SingleUser(user_id="corp_user"),
)
```

**Local Development:**
```python
app = chat.Chatnificent(
    llm=chat.llm.Ollama(model="llama3.2"),
    store=chat.store.InMemory(),
    tools=chat.tools.PythonTool(),
)
```

**Dash UI with Bootstrap:**
```python
from chatnificent.server import DashServer

app = chat.Chatnificent(
    server=DashServer(),
    layout=chat.layout.Bootstrap(),
    llm=chat.llm.Gemini(),
    store=chat.store.SQLite(db_path="global.db"),
)
```

### Custom Pillar Implementation

```python
import json

class RedisStore(chat.store.Store):
    def __init__(self, redis_url: str):
        self.client = redis.from_url(redis_url)
    
    def save_conversation(self, user_id: str, conversation: Conversation):
        key = f"chat:{user_id}:{conversation.id}"
        data = {"id": conversation.id, "messages": conversation.messages}
        self.client.set(key, json.dumps(data))
    
    def load_conversation(self, user_id: str, convo_id: str):
        key = f"chat:{user_id}:{convo_id}"
        raw = self.client.get(key)
        if not raw:
            return None
        data = json.loads(raw)
        return Conversation(id=data["id"], messages=data["messages"])
    
    # Implement other required methods...

app = chat.Chatnificent(store=RedisStore("redis://localhost:6379"))
```

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

## Quick Reference for Agents

**Need to add a new LLM provider?** → Subclass `llm.LLM`, implement `generate_response()` and `extract_content()`

**Need custom storage?** → Subclass `store.Store`, implement save/load methods

**Need UI changes?** → For DevServer: edit `templates/default.html`. For Dash: subclass `layout.DashLayout`

**Need request lifecycle customization?** → Subclass `engine.Orchestrator`, override hooks/seams

**Need tool integration?** → Subclass `tools.Tool`, handle tool call dicts → tool result dicts

**Need a different HTTP server?** → Subclass `server.Server`, implement `create_server()` and `run()`
