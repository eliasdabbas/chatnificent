# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "chatnificent[starlette]",
# ]
# ///
"""
Starlette Quickstart — Production-Ready in One Line
====================================================

The fastest way to upgrade your Chatnificent app from a development server to a
production-grade async server. One parameter change — ``server=chat.server.Starlette()``
— swaps the stdlib ``DevServer`` for a full ASGI server powered by Starlette and
Uvicorn.

Why Starlette?
--------------
The default ``DevServer`` is a zero-dependency stdlib HTTP server. It's great for
getting started, but it's single-threaded and not designed for production traffic.
Switching to ``Starlette`` gives you:

- **Async request handling** — multiple requests are served concurrently via ASGI,
  no more blocking while waiting for LLM responses
- **Uvicorn** — a lightning-fast ASGI server that powers your app in production,
  with support for multiple workers, auto-reload, and graceful shutdown
- **ASGI ecosystem** — add middleware (CORS, auth, logging), mount multiple apps,
  use ``TestClient`` for integration testing — all standard Starlette patterns
- **Direct uvicorn CLI** — run ``uvicorn app:app --workers 4`` for multi-process
  deployment without writing any extra code

Everything else stays the same — your LLM, Store, Auth, Layout, and Engine pillars
are completely unchanged. The Starlette server speaks the same endpoint contract
as DevServer, so the built-in chat UI works identically.

How It Works
------------
1. ``chat.server.Starlette()`` creates a server pillar that builds a Starlette
   ASGI application internally
2. ``app.run()`` delegates to ``uvicorn.run()``, starting the server on
   http://127.0.0.1:7777 (same default port as DevServer)
3. All Chatnificent endpoints (``/api/chat``, ``/api/conversations``, etc.) are
   registered as async Starlette routes that wrap the synchronous engine via
   ``anyio.to_thread.run_sync()``
4. The ``Chatnificent`` instance is also a valid ASGI callable, so you can run it
   directly with ``uvicorn starlette_quickstart:app`` from the command line

Running
-------
::

    uv run --script examples/starlette_quickstart.py

Then open http://127.0.0.1:7777 in your browser — the same chat UI you know from
DevServer, now running on an async production server.

Alternatively, run via the uvicorn CLI (from the examples directory)::

    uvicorn starlette_quickstart:app

What to Explore Next
--------------------
- Customize host, port, workers, and reload: see ``starlette_uvicorn_options.py``
- Add CORS middleware and custom routes: see ``starlette_server_options.py``
- Mount multiple chat apps on one website: see ``starlette_multi_mount.py``
"""

import chatnificent as chat

app = chat.Chatnificent(
    server=chat.server.Starlette(),
)


if __name__ == "__main__":
    app.run()
