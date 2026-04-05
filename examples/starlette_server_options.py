# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "chatnificent[starlette]",
#     "openai",
# ]
# ///
"""
Starlette Server Options — Customize Your ASGI Application
============================================================

The ``chat.server.Starlette`` class accepts several constructor parameters that
map directly to Starlette's ``Application`` constructor. This gives you full
control over your ASGI application — add middleware, inject custom routes,
configure error handlers, and hook into the application lifecycle.

Constructor Parameters
----------------------
``chat.server.Starlette()`` accepts these keyword arguments:

- **debug** (``bool``, default ``False``) — enables Starlette's debug mode,
  which shows detailed tracebacks in the browser on errors. For verbose
  uvicorn logging, pass ``log_level="debug"`` to ``app.run()`` separately.

- **routes** (``list``, optional) — a list of Starlette ``Route`` objects that
  are **prepended** before Chatnificent's framework routes. Since Starlette uses
  first-match routing, your routes take priority. This lets you add health
  checks, version endpoints, or even override framework routes entirely.

- **middleware** (``list``, optional) — a list of ``Middleware`` descriptors.
  These are applied in order around every request, including Chatnificent's own
  endpoints. Common uses: CORS headers, request logging, authentication,
  rate limiting.

- **exception_handlers** (``dict``, optional) — a dict mapping status codes
  or exception classes to handler functions. Useful for custom error pages
  (e.g., a branded 404 page).

- **lifespan** (async context manager, optional) — runs code on application
  startup and shutdown. Perfect for initializing database connections, loading
  ML models, or flushing logs on exit.

How Routes Work
---------------
User routes are prepended before framework routes::

    your routes:      /api/health, /api/version
    framework routes: /api/chat, /api/conversations, /{path}, /

Starlette matches top-to-bottom, so your ``/api/health`` wins over the framework
catch-all. You can even override ``/api/conversations`` to add custom metadata.

How Middleware Works
--------------------
Middleware wraps every request/response. The order matters — the first middleware
in the list is the outermost wrapper::

    Request → CORS → Logging → Chatnificent handler → Logging → CORS → Response

This example demonstrates ``CORSMiddleware`` (cross-origin requests) and a custom
timing middleware that logs how long each request takes.

Running
-------
::

    uv run --script examples/starlette_server_options.py

Then try:

- http://127.0.0.1:7777 — the chat UI (standard Chatnificent)
- http://127.0.0.1:7777/api/health — custom health check endpoint
- http://127.0.0.1:7777/api/version — custom version endpoint

Check the terminal for per-request timing logs from the middleware.

What to Explore Next
--------------------
- Start with the simplest Starlette app: see ``starlette_quickstart.py``
- Configure uvicorn options (workers, reload, SSL): see ``starlette_uvicorn_options.py``
- Mount multiple chat apps on one website: see ``starlette_multi_mount.py``
"""

import time
from contextlib import asynccontextmanager

import chatnificent as chat
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.types import ASGIApp, Receive, Scope, Send

# ---------------------------------------------------------------------------
# Custom routes — these are prepended before Chatnificent's framework routes
# ---------------------------------------------------------------------------


async def health_check(request: Request) -> JSONResponse:
    """Simple health check for load balancers and uptime monitors."""
    return JSONResponse({"status": "healthy"})


async def version_info(request: Request) -> JSONResponse:
    """Return the app version — useful for deployment verification."""
    return JSONResponse(
        {"app": "Chatnificent Starlette server demo", "version": f"{chat.__version__}"}
    )


# ---------------------------------------------------------------------------
# Custom middleware — wraps every request/response
# ---------------------------------------------------------------------------


class TimingMiddleware:
    """Log how long each request takes to process."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        start = time.perf_counter()
        await self.app(scope, receive, send)
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"  {scope['method']} {scope['path']} — Elapsed: {elapsed_ms:.1f}ms")


# ---------------------------------------------------------------------------
# Lifespan — startup and shutdown hooks
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app):
    print(
        "[LIFESPAN]: >>>> Starting up — initialize resources here (DB pools, caches, etc.)"
    )
    yield
    print(
        "[LIFESPAN]: >>>> Shutting down — clean up resources here (flush logs, close connections)"
    )


# ---------------------------------------------------------------------------
# Exception handlers — custom error responses
# ---------------------------------------------------------------------------


async def not_found(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        {"error": "not found", "path": str(request.url.path)},
        status_code=404,
    )


# ---------------------------------------------------------------------------
# Build the app with all options
# ---------------------------------------------------------------------------

app = chat.Chatnificent(
    server=chat.server.Starlette(
        debug=True,
        routes=[
            Route("/api/health", health_check),
            Route("/api/version", version_info),
        ],
        middleware=[
            Middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_methods=["*"],
                allow_headers=["*"],
            ),
            Middleware(TimingMiddleware),
        ],
        exception_handlers={404: not_found},
        lifespan=lifespan,
    ),
)

if __name__ == "__main__":
    app.run()
