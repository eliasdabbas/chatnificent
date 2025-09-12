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

Run with: `source .venv/bin/activate && uv run app.py`

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

## Engine Orchestration (`engine.py`)

The **Engine** manages the request lifecycle. Override hooks and seams for customization:

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

## Development Commands for Agents

Use `uv` to run all installation, running apps, and any Python commands.

```bash
# Run main app
uv run app.py

uv run pytest tests/
```

## Intergration and orchestration

In general and in order to support any LLM provider (API, file formats, conversation
structure, etc) the default implementations will be according to the OpenAI formats.

Each LLM concrete class will have to provide translate from/to that format. 

This way the engine works seamlessly without having to worry about how to ingest,
display, or process any message or tools call, it will send this to the respective
provider who should have the necessary methods to enable this.

This enables easy extension to add any LLM, and uses the most popular Python package in
the LLM space `openai`.

## Customization Patterns

### Mix and Match Example
```python
import chatnificent as chat

# Scientific research setup
app = Chatnificent(
    llm=chat.llm.Anthropic(model="claude-3-5-sonnet-20240620"),
    store=chat.store.File(directory="./research_chats"),
    layout=chat.layout.Minimal(),  # Clean, distraction-free
    tools=chat.tools.PythonTool(),  # Code execution
)

# Enterprise setup  
app = Chatnificent(
    llm=chat.llm.OpenAI(),
    store=chat.store.SQLite(db_path="enterprise.db"),
    auth=chat.auth.SingleUser(user_id="corp_user"),
    layout=chat.layout.Bootstrap(),  # Professional look
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

## Quick Reference for Agents

**Need to add a new LLM provider?** → Subclass `llm.LLM`, implement `generate_response()` and `extract_content()`

**Need custom storage?** → Subclass `store.Store`, implement save/load methods

**Need UI changes?** → Override `layout.Layout.build_layout()` (keep required component IDs)

**Need request lifecycle customization?** → Subclass `engine.Synchronous`, override hooks/seams

**Need tool integration?** → Subclass `tools.Tool`, handle `ToolCall` → `ToolResult`

The framework validates required component IDs at startup and handles pillar orchestration automatically via `callbacks.py`.