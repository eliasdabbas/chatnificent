# Coding Agent Instructions for Chatnificent

## Quick Start for Agents

**Chatnificent** 

> LLM chat app framework
> Minimally complete. Maximally hackable.

```python
import chatnificent as chat
app = chat.Chatnificent()
app.run(debug=True)  # Visit http://127.0.0.1:8050
```

Run with: `uv run app.py`

## Core Architecture: 8 Pillars

Chatnificent uses **dependency injection** with 8 configurable "pillars". Each pillar has an abstract base class and concrete implementations:

| Pillar | File | Purpose | Default | Available |
|--------|------|---------|---------|-----------|
| **Layout** | `layout.py` | UI components | `Bootstrap` | Bootstrap, Mantine, Minimal |
| **LLM** | `llm.py` | LLM API calls | `OpenAI` | OpenAI, Anthropic, Gemini, Ollama, Echo |
| **Store** | `store.py` | Conversation persistence | `InMemory` | InMemory, File, SQLite |
| **Auth** | `auth.py` | User identification | `Anonymous` | Anonymous, SingleUser |
| **Engine** | `engine.py` | Request orchestration | `Synchronous` | Synchronous |
| **Tools** | `tools.py` | Function calling | `NoTool` | PythonTool, NoTool |
| **Retrieval** | `retrieval.py` | RAG/context | `NoRetrieval` | NoRetrieval |
| **URL** | `url.py` | Route parsing | `PathBased` | PathBased, QueryParams |

## Philosophy: Minimally Complete, Maximally Hackable

### Atomic Architecture Design

Each pillar is designed with **atomic methods** that operate independently without dependencies on other pillars. This enables true pluggability where any implementation can be swapped without affecting others.

**Pillar Independence:**
- **LLM**: `generate_response()` and `extract_content()` work with any message format
- **Store**: `save_conversation()`, `load_conversation()` are storage-agnostic
- **Auth**: `get_current_user_id()` works independently of other components
- **Layout**: `build_layout()` renders regardless of LLM or storage backend

**True Pluggability:**
All callbacks use **only abstract methods** from the pillar interfaces, ensuring complete backend interchangeability. Pillars communicate only through abstract interfaces, making the framework truly "minimally complete, maximally hackable."

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

### 3. Layout Pillar (`layout.py`)
```python
class Layout(ABC):
    @abstractmethod
    def build_layout(self) -> DashComponent:
        """Build the complete UI component tree."""

# Required component IDs in any custom layout:
# - user_input_textarea, chat_send_button, new_chat_button
# - chat_area_main, convo_list_div, sidebar_offcanvas
```

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

### Customization via Hooks and Seams

```python
class CustomEngine(chat.engine.Synchronous):
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

**Scientific Research Setup:**
```python
import chatnificent as chat

app = chat.Chatnificent(
    llm=chat.llm.Anthropic(model="claude-3-5-sonnet-20240620"),
    store=chat.store.File(directory="./research_chats"),
    layout=chat.layout.Minimal(),  # Clean, distraction-free
    tools=chat.tools.PythonTool(),  # Code execution
)
```

**Enterprise Setup:**
```python
app = chat.Chatnificent(
    llm=chat.llm.OpenAI(),
    store=chat.store.SQLite(db_path="enterprise.db"),
    auth=chat.auth.SingleUser(user_id="corp_user"),
    layout=chat.layout.Bootstrap(),  # Professional look
)
```

**Development Assistant:**
```python
app = chat.Chatnificent(
    llm=chat.llm.Ollama(model="llama3"),  # Local model
    store=chat.store.InMemory(),  # Fast prototyping
    tools=chat.tools.PythonTool(),  # Code execution
    layout=chat.layout.Minimal(),
)
```

**Multilingual Support:**
```python
app = chat.Chatnificent(
    llm=chat.llm.Gemini(model="gemini-pro"),  # Multilingual model
    store=chat.store.SQLite(db_path="global.db"),
    auth=chat.auth.SingleUser(user_id="global_user"),
    layout=chat.layout.Bootstrap(),
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
├── models.py            # Conversation dataclass, role constants
├── callbacks.py         # Dash callbacks orchestrating pillars
├── auth.py             # User identification (Anonymous, SingleUser)
├── store.py            # Conversation persistence (InMemory, File, SQLite)
├── layout.py           # UI builders (Bootstrap, Mantine, Minimal)
├── llm.py              # LLM providers (OpenAI, Anthropic, Gemini, etc.)
├── engine.py           # Request orchestration (Synchronous)
├── tools.py            # Function calling (PythonTool, NoTool)  
├── retrieval.py        # RAG/context retrieval (NoRetrieval)
└── url.py              # URL parsing (PathBased, QueryParams)
```

### Git Guidelines
- Never `git add .` - add files individually
- Commit only relevant changes for the specific feature
- Use atomic commits with clear messages

### Development Conventions
- **Modular Architecture**: Consider which pillar new functionality belongs to
- **Docstrings**: Use [Numpy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html) for consistency
- **Testing**: Write comprehensive unit tests, mocking pillar dependencies
- **Required Component IDs**: Custom layouts must include: `url_location`, `chat_area_main`, `user_input_textarea`, `convo_list_div`, `chat_send_button`, `new_chat_button`, `sidebar_offcanvas`, `sidebar_toggle_button`

## Quick Reference for Agents

**Need to add a new LLM provider?** → Subclass `llm.LLM`, implement `generate_response()` and `extract_content()`

**Need custom storage?** → Subclass `store.Store`, implement save/load methods

**Need UI changes?** → Override `layout.Layout.build_layout()` (keep required component IDs)

**Need request lifecycle customization?** → Subclass `engine.Synchronous`, override hooks/seams

**Need tool integration?** → Subclass `tools.Tool`, handle tool call dicts → tool result dicts

The framework validates required component IDs at startup and handles pillar orchestration automatically via `callbacks.py`.
