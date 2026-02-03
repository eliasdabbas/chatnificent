# Coding Agent Instructions for Chatnificent

## Quick Start for Agents

**Chatnificent** 

> LLM chat app framework
> Minimally complete. Maximally hackable.

```python
from chatnificent import Chatnificent
app = Chatnificent()
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
app = Chatnificent(llm=chat.llm.Anthropic(api_key="sk-..."))
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
app = Chatnificent(store=chat.store.SQLite(db_path="chats.db"))
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

app = Chatnificent(engine=CustomEngine())
```

## Data Models (`models.py`)

### ChatMessage - Enhanced for Tool Calling
```python
class ChatMessage(BaseModel):
    role: Role  # "system" | "user" | "assistant" | "tool" | "model"
    content: MessageContent  # str | List[Dict] | None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

class Conversation(BaseModel):
    id: str
    messages: List[ChatMessage] = []
```

## Integration and Orchestration

To support any LLM provider (API formats, conversation structures, etc.), Chatnificent uses **OpenAI formats as the universal standard**. Each LLM concrete class translates from/to that format.

This enables the engine to work seamlessly without worrying about how to ingest, display, or process any message or tool call—it delegates this to the respective provider who has the necessary translation methods.

This approach:
- Enables easy extension to add any LLM
- Uses the most popular Python package in the LLM space (`openai`)
- Provides a common interface for all providers

## Customization Patterns

### Mix and Match Examples

**Scientific Research Setup:**
```python
import chatnificent as chat

app = Chatnificent(
    llm=chat.llm.Anthropic(model="claude-3-5-sonnet-20240620"),
    store=chat.store.File(directory="./research_chats"),
    layout=chat.layout.Minimal(),  # Clean, distraction-free
    tools=chat.tools.PythonTool(),  # Code execution
)
```

**Enterprise Setup:**
```python
app = Chatnificent(
    llm=chat.llm.OpenAI(),
    store=chat.store.SQLite(db_path="enterprise.db"),
    auth=chat.auth.SingleUser(user_id="corp_user"),
    layout=chat.layout.Bootstrap(),  # Professional look
)
```

**Development Assistant:**
```python
app = Chatnificent(
    llm=chat.llm.Ollama(model="llama3"),  # Local model
    store=chat.store.InMemory(),  # Fast prototyping
    tools=chat.tools.PythonTool(),  # Code execution
    layout=chat.layout.Minimal(),
)
```

**Multilingual Support:**
```python
app = Chatnificent(
    llm=chat.llm.Gemini(model="gemini-pro"),  # Multilingual model
    store=chat.store.SQLite(db_path="global.db"),
    auth=chat.auth.SingleUser(user_id="global_user"),
    layout=chat.layout.Bootstrap(),
)
```

### Custom Pillar Implementation

```python
class RedisStore(chat.store.Store):
    def __init__(self, redis_url: str):
        self.client = redis.from_url(redis_url)
    
    def save_conversation(self, user_id: str, conversation: Conversation):
        key = f"chat:{user_id}:{conversation.id}"
        self.client.set(key, conversation.model_dump_json())
    
    def load_conversation(self, user_id: str, convo_id: str):
        key = f"chat:{user_id}:{convo_id}"
        data = self.client.get(key)
        return Conversation.model_validate_json(data) if data else None
    
    # Implement other required methods...

app = Chatnificent(store=RedisStore("redis://localhost:6379"))
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
├── models.py            # ChatMessage, Conversation, ToolCall, ToolResult  
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

**Need tool integration?** → Subclass `tools.Tool`, handle `ToolCall` → `ToolResult`

The framework validates required component IDs at startup and handles pillar orchestration automatically via `callbacks.py`.
