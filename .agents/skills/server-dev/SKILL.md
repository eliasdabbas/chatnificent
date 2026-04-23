---
name: server-dev
description: Implement or modify a Chatnificent server (DevServer, Starlette, FastAPI, etc.)
---

# Server Development Reference

Use this skill when implementing a new server, modifying endpoint behavior, or working on the HTTP/SSE transport layer.

## Server Endpoint Contract

Every Chatnificent server implementation must expose these endpoints. DevServer is the reference implementation — StarletteServer, FastAPIServer, and any custom server must produce identical behavior for the same inputs.

> **This contract is provisional.** It reflects DevServer's current behavior and will be refined as async server implementations (Starlette, FastAPI) are built. The e2e and parity tests are the authoritative spec.

### Endpoints

| Method | Path | Request Body | Response | Notes |
|--------|------|-------------|----------|-------|
| GET | `/` | — | HTML page (`layout.render_page()`) | Serves the chat UI |
| GET | `/{user_id}/{convo_id}` | — | HTML page with `<script>window.__CHATNIFICENT_CONVO__="{convo_id}"</script>` injected | Pre-loads conversation |
| GET | `/api/conversations` | — | `{"conversations": [{"id": "...", "title": "..."}]}` | Titles derived from first user message, truncated to 30 chars + "..." |
| GET | `/api/conversations/{id}` | — | `{"id": "...", "messages": [...], "path": "..."}` | Messages filtered through `layout.render_messages()`. 404 if not found |
| POST | `/api/chat` | `{"message": "...", "conversation_id": "..."}` | JSON or SSE stream (see below) | Dispatches based on `llm.default_params["stream"]` |

### Non-Streaming Response (POST /api/chat)

```json
{
  "response": "assistant text",
  "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
  "conversation_id": "abc123",
  "path": "/<user_id>/<convo_id>"
}
```

Errors: `{"error": "message"}` with HTTP 400 or 500.

### SSE Streaming Response (POST /api/chat)

`Content-Type: text/event-stream`. Each event is `data: {json}\n\n`.

| Event | `data` Payload | Purpose |
|-------|---------------|---------|
| `delta` | `"token text"` (string) | Streamed content token |
| `status` | `"Calling tool: ..."` (string) | Agentic loop status |
| `done` | `{"conversation_id": "...", "path": "..."}` (object) | Stream complete |
| `error` | `"error message"` (string) | Error during processing |

### Auth Contract

| Property | Value |
|----------|-------|
| Cookie name | `chatnificent_session` |
| Cookie attributes | `Path=/; SameSite=Lax` (when mounted under a `root_path`, `Path=<root_path>/`) |
| Session resolution | `auth.get_current_user_id(session_id=<cookie_value>)` |
| Set-Cookie | Only on new sessions (`_new_session=True`) |

### Pillar Delegation

- **Auth**: All endpoints resolve user via `auth.get_current_user_id(session_id=...)` — never bare `get_current_user_id()`
- **URL**: Path parsing via `url.parse(path)`, path building via `url.build_conversation_path(user_id, convo_id)`
- **Layout**: `render_page()` for HTML, `render_messages()` for message filtering, `render_conversations()` for sidebar data
- **Engine**: `handle_message()` (non-streaming) or `handle_message_stream()` (SSE) — server checks `llm.default_params.get("stream", False)`
