---
name: pillar-dev
description: Implement or extend Chatnificent pillars (LLM, Store, Engine, Auth, Tools, etc.)
---

# Pillar Development Reference

Use this skill when implementing a new pillar, extending an existing one, or working on cross-pillar integration (Engine orchestration, data models, customization).

## Key Interfaces

### LLM Pillar (`llm.py`)
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

### Store Pillar (`store.py`)
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

### Server Pillar (`server.py`)
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

### Layout Pillar (`layout.py`)
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

Retrieval (RAG context) runs **once per request, before** the loop — not inside it. The loop itself is bounded by `Orchestrator.max_agentic_turns` (default 5) to prevent runaway tool invocations.

The engine can loop multiple times to allow the LLM to use tools, process results, and form a final answer:

1. **LLM generates response** (may include tool calls)
2. **Engine detects tool calls** and executes them via Tools pillar
3. **Tool results** are added to conversation history
4. **Loop continues** until LLM provides final answer or `max_agentic_turns` is reached

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

## Quick Reference

| Task | How |
|------|-----|
| New LLM provider | Subclass `llm.LLM`, implement `generate_response()` and `extract_content()` |
| Custom storage | Subclass `store.Store`, implement save/load methods |
| UI changes (DevServer) | Edit `templates/default.html` |
| UI changes (Dash) | Subclass `layout.DashLayout` |
| Request lifecycle | Subclass `engine.Orchestrator`, override hooks/seams |
| Tool integration | Subclass `tools.Tool`, handle tool call dicts -> tool result dicts |
| Different HTTP server | Subclass `server.Server`, implement `create_server()` and `run()` |
