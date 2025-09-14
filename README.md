
<img src="chatnificent_logo.png" width="350">

# 🗯️ Chatnificent

### LLM chat app framework
### Minimally complete. Maximally hackable.

Build production-ready, full-stack chat applications in minutes. Customize everything in hours.


Chatnificent is a Python framework built on [Plotly's Dash](https://dash.plotly.com/) designed to get your LLM chat applications up and running instantly, while providing a robust, decoupled architecture for unlimited customization.

Stop wrestling with UI components, state management, and backend integrations. Start building magnificent chat apps.


[![PyPI version](https://img.shields.io/pypi/v/chatnificent.svg)](https://pypi.python.org/pypi/chatnificent) [![DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/eliasdabbas/chatnificent)

## The Ethos

Frameworks should get out of your way.

  * **Minimally Complete:** Out of the box, `Chatnificent` provides a fully functional, stateful, multi-user chat application with sensible defaults.
  * **Maximally Hackable:** Every core pillar—the UI, the LLM provider, the database, the authentication, the RAG pipeline, and the core orchestration—is  swappable. Customize or replace any part without fighting the framework.

## Features

  * **LLM Agnostic:** Built-in support for OpenAI, Anthropic, Gemini, Ollama, OpenRouter, DeepSeek, and any other LLM API.
  * **Flexible UI:** Default Bootstrap layout, with built-in Mantine and Minimal (pure HTML) layouts. Easily customizable with any Dash components.
  * **Pluggable Storage:** InMemory, File-system, and SQLite included. Easily extendable to Redis, Postgres, etc.
  * **Agentic Engine:** The core engine manages multi-turn conversations and standardized tool calling across providers.
  * **Auth Ready:** Abstracted authentication layer for easy integration. No-login anonymous user auth enabled by default.
  * **RTL Support:** Automatic detection and rendering of Right-to-Left languages.
  * **Dash Native:** Leverage the full power of Plotly's Dash to integrate complex data visualizations and analytics.

## Installation

To get started quickly with the default UI (Bootstrap) and the default LLM provider (OpenAI):

```bash
pip install "chatnificent[default]"

export OPENAI_API_KEY="YOUR_API_KEY"
```

For a minimal installation (no UI libraries or LLM SDKs included):

```bash
pip install chatnificent
```



## Quickstart: Hello World (3 Lines)

This is a complete, working chat application.

Create a file `app.py`:

```python
from chatnificent import Chatnificent

app = Chatnificent()

if __name__ == "__main__":
    app.run(debug=True)
```

Run it:

```bash
python app.py
```

Open your browser to [`http://127.0.0.1:8050`](http://127.0.0.1:8050). That's it. You have a fully functional chat UI with conversation history, mobile responsiveness, and URL-based session management.

## The Pillars of Hackability

Chatnificent's architecture is built around extensible Pillars. Every major function is handled by a dedicated component adhering to a strict interface.

| Pillar | Description | Defaults | Included Implementations |
| :--- | :--- | :--- | :--- |
| **`LLM`** | The brain (API calls, parsing). | `OpenAI` (or `Echo`) | OpenAI, Anthropic, Gemini, OpenRouter, DeepSeek, Ollama, Echo |
| **`Layout`** | The look and feel (UI components). | `Bootstrap` (or `Minimal`) | Bootstrap, Mantine, Minimal (HTML) |
| **`Store`** | The memory (Persistence). | `InMemory` | InMemory, File, SQLite |
| **`Auth`** | The gatekeeper (User identification). | `Anonymous` | Anonymous, SingleUser |
| **`Engine`** | The orchestrator (Request lifecycle). | `Synchronous` | Synchronous |
| **`Tools`** | Tool/function calling capabilities. | `NoTool` | PythonTool, NoTool |
| **`Retrieval`** | RAG knowledge retrieval. | `NoRetrieval` | NoRetrieval |
| **`URL`** | URL parsing and routing. | `PathBased` | PathBased, QueryParams |

You customize the app by injecting the implementations you need during initialization:

```python
from chatnificent import Chatnificent
import chatnificent as chat

app = Chatnificent(
    llm=chat.llm.Anthropic(),
    store=chat.store.SQLite(db_path="conversations.db"),
    layout=chat.layout.Mantine()
)
```

## Progressive Power: Swapping the Pillars

Let's evolve the "Hello World" example by swapping pillars.

### Level 1: Swapping the LLM 🧠

Want to use Anthropic's Claude 3.5 Sonnet? Just swap the `llm` pillar.

*(Requires `pip install anthropic` and setting `ANTHROPIC_API_KEY`)*

```python
from chatnificent import Chatnificent
import chatnificent as chat


app = Chatnificent(
    llm=chat.llm.Anthropic(model="claude-3-5-sonnet-20240620")
)

# Or try Gemini: app = Chatnificent(llm=chat.llm.Gemini())
# Or local Ollama: app = Chatnificent(llm=chat.llm.Ollama(model="llama3.1"))
```

Chatnificent handles the translation of message formats and tool-calling protocols automatically.

### Level 2: Adding Persistent Storage

The default `InMemory` store is ephemeral. Let's use `SQLite` for persistence.

```python
from chatnificent import Chatnificent
import chatnificent as chat

app = Chatnificent(
    store=store.SQLite(db_path="conversations.db")
)
# Or use the filesystem: store=chat.store.File(base_dir="./chat_data")
```

Conversations are now persisted across server restarts, and the sidebar automatically loads your history.

### Level 3: Changing the Look and Feel 🎨

Don't want Bootstrap? Let's try the Mantine layout.

*(Requires `pip install dash-mantine-components`)*

```python
from chatnificent import Chatnificent
import chatnificent as chat

app = Chatnificent(layout=chat.layout.Mantine())

# Or use the barebones HTML layout: layout=layout.Minimal()
```

Want a completely custom design? Implement the `layout.Layout` abstract base class. The framework ensures your custom layout integrates seamlessly, provided you include the required component IDs (e.g., `input_textarea`, `messages_container`, etc.).

### Level 4: Custom Authentication

The default `Anonymous` auth isolates users by random user ID. You can easily implement custom logic.

```python
from chatnificent import Chatnificent, auth

class HeaderAuth(auth.Auth):
    def get_current_user_id(self, **kwargs) -> str:
        from flask import request
        # Identify user based on a header (e.g., provided by an auth proxy)
        return request.headers.get("X-User-Id", "unknown_user")

app = Chatnificent(auth=HeaderAuth())
```

### Level 5: The Engine (Advanced Orchestration)

The `Engine` orchestrates the entire request lifecycle: resolving the conversation, RAG retrieval, the agentic loop (Tools + LLM calls), and persistence.

The default `Synchronous` engine provides "hooks" (empty methods called at specific points) and "seams" (core logic methods) that you can override to deeply customize behavior without rewriting the core logic.

```python
from chatnificent import Chatnificent
import chatnificent as chat
from typing import Any, Optional

# Create a custom engine by inheriting from the default
class CustomEngine(chat.engine.Synchronous):

    # 1. Override a HOOK to add monitoring/logging
    def _after_llm_call(self, llm_response: Any) -> None:
        # Example: Extract token usage if the LLM response object has a 'usage' attribute
        tokens = getattr(llm_response, 'usage', 'N/A')
        print(f"[MONITORING] LLM call complete. Tokens: {tokens}")

    # 2. Override a SEAM to modify core logic (e.g., prompt engineering)
    def _prepare_llm_payload(self, conversation, retrieval_context: Optional[str]):
        # Get the default payload (which already includes the context if present)
        payload = super()._prepare_llm_payload(conversation, retrieval_context)

        # Inject a custom system prompt if none exists
        if not any(m['role'] == 'system' for m in payload):
            payload.insert(0, {"role": "system", "content": "Be brief and professional."})
        return payload


# Initialize the app, passing the engine instance.
# Chatnificent's constructor will automatically bind the app reference to the engine.
app = Chatnificent(engine=CustomEngine())
```

## Architecture Overview

How the pillars work together during a request:

1.  **User Input**: The user submits a message via the `Layout`.
2.  **Callback Trigger**: A Dash callback delegates the input to the `Engine`.
3.  **Context Resolution**: The `Engine` uses `Auth`, `URL`, and `Store` to identify the user and load the conversation history.
4.  **Agentic Loop**:
      * The `Engine` calls `Retrieval` to gather context (RAG).
      * The `Engine` sends the history and context to the `LLM`.
      * If the `LLM` requests a tool call, the `Engine` executes it via `Tools` and loops back.
      * If the `LLM` returns a final response, the loop exits.
5.  **Persistence**: The `Engine` saves the updated conversation via the `Store`.
6.  **Rendering**: The `Engine` formats the messages using the `Layout` and updates the client UI.

## Building Your Own Pillars

The ultimate hackability comes from implementing your own pillars. Want to use MongoDB? Just implement the `store.Store` interface.

### Example: Custom Storage Implementation

```python
from chatnificent import Chatnificent
import chatnificent as chat
from typing import Optional, List

class MongoDBStore(chat.store.Store):
    def __init__(self, connection_string):
        # Initialize MongoDB client...
        print(f"Connecting to MongoDB at {connection_string}...")
        pass

    def load_conversation(self, user_id: str, convo_id: str) -> Optional[Conversation]:
        # Implement loading logic...
        return None

    # Implement the other required methods...
    def save_conversation(self, user_id: str, conversation: Conversation):
        pass
    def list_conversations(self, user_id: str) -> List[str]:
        return []
    def get_next_conversation_id(self, user_id: str) -> str:
        return "1"

# Use your custom implementation
# app = Chatnificent(store=MongoDBStore(connection_string="mongodb://..."))
```
