# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "chatnificent[starlette]",
# ]
# ///
"""
Starlette Uvicorn Options — Configure Your Production Server
=============================================================

Chatnificent's Starlette server delegates to `Uvicorn <https://www.uvicorn.org/>`_
under the hood. The most common options — ``host``, ``port``, ``workers``,
``reload``, ``log_level``, ``ssl_keyfile``, ``ssl_certfile`` — are explicit
parameters on ``app.run()`` for discoverability. Every other
``uvicorn.run()`` keyword is accepted via ``**kwargs``, so the full
`uvicorn settings docs <https://www.uvicorn.org/settings/>`_ are always
available.

How It Works
------------
When you call ``app.run()``, the Starlette server:

1. Sets sensible defaults (``host="127.0.0.1"``, ``port=7777``)
2. Auto-resolves an import string (e.g., ``"starlette_uvicorn_options:app"``)
   so that multi-worker mode and reload work correctly — uvicorn needs an
   import string, not an object, to spawn fresh workers
3. Forwards **all remaining kwargs** to ``uvicorn.run()``

This means anything in the
`uvicorn settings docs <https://www.uvicorn.org/settings/>`_ is available
as a keyword argument.

Explicit Parameters
-------------------
These are first-class ``app.run()`` parameters with autocomplete and type hints:

**Host & port** — bind to all interfaces for external access::

    app.run(host="0.0.0.0", port=8000)

**Multiple workers** — spawn 4 OS processes for true parallelism::

    app.run(workers=4)

.. tip:: Workers require an import string (auto-resolved). Can't be
   combined with ``reload=True``.

**Auto-reload** — restart on code changes during development::

    app.run(reload=True)

**Log level** — control verbosity (debug, info, warning, error, critical)::

    app.run(log_level="warning")

.. tip:: For verbose output during development, use ``log_level="debug"``.
   Starlette debug error pages are a separate concern — set
   ``Starlette(debug=True)`` on the constructor.

**SSL/TLS** — serve over HTTPS (provide cert and key files)::

    app.run(
        ssl_keyfile="./key.pem",
        ssl_certfile="./cert.pem",
    )

Advanced Options (via **kwargs)
-------------------------------
These are passed through to ``uvicorn.run()`` directly:

**Reload scoping** — limit which files trigger a reload::

    app.run(
        reload=True,
        reload_dirs=["src"],  # only watch src/
        reload_includes=["*.py"],  # only .py files
        reload_excludes=["test_*.py"],  # skip test files
    )

**Timeouts** — tune connection and shutdown behaviour::

    app.run(
        timeout_keep_alive=30,  # seconds before closing idle connections
        timeout_graceful_shutdown=10,  # seconds to drain in-flight requests
    )

**Request limits** — protect against overload::

    app.run(
        limit_concurrency=100,  # max simultaneous connections
        limit_max_requests=1000,  # restart worker after N requests (leak guard)
        backlog=2048,  # OS-level listen backlog
    )

**Proxy headers** — trust ``X-Forwarded-For`` behind a reverse proxy::

    app.run(
        proxy_headers=True,
        forwarded_allow_ips="*",  # or a comma-separated allowlist
    )

.. warning:: Only enable ``proxy_headers`` when behind a trusted reverse proxy.

**Root path** — set the ASGI root path for reverse proxy deployments::

    app.run(root_path="/chat")

.. tip:: Prefer Starlette ``Mount()`` for multi-app setups — it sets
   ``root_path`` automatically. See ``starlette_multi_mount.py``.

**Access logs** — toggle request logging::

    app.run(access_log=False)  # disable in production if logging elsewhere

**Security headers** — control server identity leakage::

    app.run(
        server_header=False,  # omit "server: uvicorn" header
        date_header=False,  # omit "date" header
    )

Recipe: Production
------------------
::

    app.run(
        host="0.0.0.0",
        port=443,
        workers=4,
        ssl_keyfile="./key.pem",
        ssl_certfile="./cert.pem",
        log_level="warning",
        proxy_headers=True,
        forwarded_allow_ips="10.0.0.0/8",
        limit_concurrency=200,
        access_log=False,
        server_header=False,
    )

Recipe: Development
-------------------
::

    app.run(
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["src"],
        log_level="debug",
    )

CLI Alternative
---------------
Instead of ``app.run()``, you can use the ``uvicorn`` CLI directly. This is
useful in Docker or systemd deployments where you want the process manager
to control the server::

    uvicorn starlette_uvicorn_options:app --host 0.0.0.0 --port 8000 --workers 4

This works because ``Chatnificent`` implements ``__call__`` as an ASGI
callable, delegating to ``self.server.asgi_app``. So ``app`` in your script
*is* a valid ASGI application.

Running
-------
Run with the defaults (host=0.0.0.0, port=8000, reload on)::

    uv run --script examples/starlette_uvicorn_options.py

Or override from the CLI::

    uvicorn starlette_uvicorn_options:app --workers 4 --port 9000

What to Explore Next
--------------------
- Start with the simplest Starlette app: see ``starlette_quickstart.py``
- Add CORS middleware and custom routes: see ``starlette_server_options.py``
- Mount multiple chat apps on one website: see ``starlette_multi_mount.py``
"""

import chatnificent as chat

app = chat.Chatnificent(
    server=chat.server.Starlette(),
)

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["src"],
    )
