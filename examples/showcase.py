# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "chatnificent[starlette,openai,anthropic,gemini,ollama]>=0.0.25",
#     "fpdf2>=2.8.7",
#     "markdown>=3.10.2",
#     "pillow>=12.2.0",
#     "segno>=1.6.6",
# ]
# ///
"""
Showcase - Minimal Multi-Mount for Selected Examples
====================================================

Mounts every stem listed in ``INCLUDED_EXAMPLES`` under ``/chat/<slug>/`` on a
single Starlette parent app. The landing page at ``/`` lists every example
that mounted successfully. Examples that fail to import or adapt are skipped
with a printed message so the rest of the site keeps working.

This is the lean cousin of ``chatnificent_website.py``: no store remapping,
no per-example shims, no docstring scraping. The goal is the smallest
possible scaffolding that proves the multi-mount pattern works for the
curated set.

Running
-------
::

    uv run --script examples/showcase.py

Or via uvicorn::

    uvicorn examples.showcase:app --port 7777
"""

import importlib.util
import os
import re
import sys
from pathlib import Path

import chatnificent as chat
import chatnificent.templates as _chat_templates
from starlette.applications import Starlette
from starlette.responses import HTMLResponse
from starlette.routing import Mount, Route

EXAMPLES_DIR = Path(__file__).parent.resolve()

# Reuse the framework's stylesheet so the landing page matches the chat UI.
_DEFAULT_STYLES = (
    Path(_chat_templates.__file__).parent / "default" / "styles.css"
).read_text(encoding="utf-8")

# Splice the framework's stylesheet inline so we don't need a static mount.
_LANDING_HTML = (EXAMPLES_DIR / "showcase_index.html").read_text(encoding="utf-8")
_LANDING_HTML = _LANDING_HTML.replace(
    '<link rel="stylesheet" href="styles.css">',
    f"<style>{_DEFAULT_STYLES}</style>",
)

# Examples write to bare relative paths; chdir once so they all land under
# a single root. Override via CHATNIFICENT_WEBSITE_DATA in production.
_DATA_ROOT = Path(
    os.environ.get("CHATNIFICENT_WEBSITE_DATA", "showcase_data")
).resolve()
_DATA_ROOT.mkdir(parents=True, exist_ok=True)
os.chdir(_DATA_ROOT)

INCLUDED_EXAMPLES = {
    # Basics
    "quickstart",
    "custom_branding",
    # Tools & agents
    "tool_calling",
    "how_to_call_functions_with_chat_models",
    "multi_tool_agent",
    "tool_qr_code_simple",
    # Memory & titles
    "memory_tool_multi_user",
    "auto_title",
    # Display enrichment
    "usage_display",
    "conversation_summary",
    "web_search",
    "display_redaction",
    # OpenAI Responses API
    "openai_responses_website_search",
    "openai_responses_image_studio",
    "openai_responses_interactive_search",
    # UI interactions
    "ui_interactions",
    "single_app_multi_chat_mode",
    # File serving
    "file_serving_simple",
    "file_serving_advanced",
    # PDF export
    "conversation_export_pdf_simple",
    "conversation_export_pdf_advanced",
    # OpenAI artifacts
    "openai_image_simple",
    "openai_image_variations",
    "openai_tts_simple",
    "openai_tts_advanced",
    # Gemini artifacts
    "gemini_image_simple",
    "gemini_image_advanced",
    "gemini_tts_simple",
    "gemini_tts_advanced",
    "gemini_music_simple",
    "gemini_music_advanced",
    "gemini_video_simple",
    "gemini_multimodal_advanced",
}


def _slugify(stem):
    return stem.replace("_", "-")


class _StripMountPrefix:
    """Strip ``root_path`` from ``scope['path']`` before delegating.

    Starlette's ``Mount`` keeps the mount prefix in both ``scope['path']``
    and ``scope['root_path']``. Chatnificent's routes are defined relative
    to the app, so the prefix has to be stripped from ``path`` for them to
    match. ``root_path`` is left in place — Chatnificent uses it to build
    absolute URLs that include the mount prefix.
    """

    def __init__(self, inner_app):
        self._inner = inner_app

    async def __call__(self, scope, receive, send):
        if scope.get("type") in ("http", "websocket"):
            root_path = scope.get("root_path", "")
            path = scope.get("path", "")
            if root_path and path.startswith(root_path):
                scope = dict(scope)
                scope["path"] = path[len(root_path) :] or "/"
        await self._inner(scope, receive, send)


def _load(stem):
    """Import ``examples/<stem>.py`` and return its Chatnificent ``app``.

    Swaps the example's server for ``chat.server.Starlette`` so the app can
    be mounted under the parent Starlette router.
    """
    path = EXAMPLES_DIR / f"{stem}.py"
    if not path.is_file():
        raise FileNotFoundError(path)

    module_name = f"_showcase_{stem.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not create import spec for {path.name}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    app = getattr(module, "app", None)
    if not isinstance(app, chat.Chatnificent):
        raise RuntimeError(
            "expected a top-level `app = chat.Chatnificent(...)` in the module"
        )

    # Preserve any custom Starlette subclass (e.g. one with extra routes or
    # ASGI middleware) the example ships with.
    if not isinstance(app.server, chat.server.Starlette):
        server = chat.server.Starlette()
        app.server = server
        server.app = app
        server.create_server()
    return app


def _prefix_artifact_urls(app, mount_prefix):
    """Make ``engine._save_artifact`` return URLs prefixed with ``mount_prefix``.

    The engine builds artifact URLs as ``/<user>/<convo>/<file>`` (root-relative),
    which 404s when the app is mounted under a sub-path. Wrap the method so the
    minted URL — and therefore every ``href`` baked into persisted HTML —
    includes the mount prefix.
    """
    engine = app.engine
    original = engine._save_artifact

    def _save_with_prefix(artifact, user_id, convo_id):
        return mount_prefix + original(artifact, user_id, convo_id)

    engine._save_artifact = _save_with_prefix


def _build():
    mounted = []
    skipped = []
    routes = []

    for stem in sorted(INCLUDED_EXAMPLES):
        slug = _slugify(stem)
        try:
            child = _load(stem)
        except Exception as exc:
            skipped.append((stem, exc))
            print(f"[showcase] skip {stem}: {exc}")
            continue

        mounted.append((stem, slug))
        mount_prefix = f"/chat/{slug}"
        _prefix_artifact_urls(child, mount_prefix)
        routes.append(Mount(mount_prefix, app=_StripMountPrefix(child)))

    async def landing(request):
        return HTMLResponse(_LANDING_HTML)

    # Warn (don't fail) when the landing page drifts from what mounted.
    mounted_slugs = {slug for _, slug in mounted}
    linked_slugs = set(re.findall(r'href="/chat/([^/"]+)/?"', _LANDING_HTML))
    missing_cards = mounted_slugs - linked_slugs
    broken_cards = linked_slugs - mounted_slugs
    if missing_cards:
        print(f"[showcase] mounted but not linked on landing: {sorted(missing_cards)}")
    if broken_cards:
        print(f"[showcase] linked on landing but not mounted: {sorted(broken_cards)}")

    return Starlette(routes=[Route("/", landing), *routes])


app = _build()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=7777)
