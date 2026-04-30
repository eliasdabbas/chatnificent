# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "chatnificent[starlette,openai,anthropic,gemini,ollama]",
# ]
# ///
"""
Chatnificent Website - One Local Website for All Non-Starlette Examples
=======================================================================

This example turns the repository's standalone examples into one small
Starlette website. Each eligible example is imported, adapted for mounting,
and exposed under its own URL:

- ``/`` - simple home page with links to every mounted example
- ``/chat/quickstart/`` - zero-dependency Echo app
- ``/chat/openai-responses-interactive-search/`` - interactive web search
- ``/chat/memory-tool/`` - persistent memory example

Discovery is automatic: the script scans ``examples/*.py`` and mounts every
non-Starlette example it can adapt. Files named ``starlette_*.py``,
``test_examples.py``, and this script itself are excluded.

If a child example cannot be imported or adapted, this website does not abort.
It prints a short error message, skips that app, and keeps going so the rest of
the site still works. That makes it a good iterative workbench while examples
are still in flight.

A few compatibility shims are applied automatically:

- File and SQLite stores are remapped into ``./examples_hub_data/<slug>/...``
- ``memory_tool.py`` gets its module-level ``MEMORY_ROOT`` redirected into that
  same sandbox
- ``openai_responses_image_studio.py`` gets a mount-aware file-serving route so
  its saved images keep working under ``/chat/<slug>/``

Running
-------
::

    uv run --script examples/chatnificent_website.py

Then open http://127.0.0.1:7777.

This example expects a checked-out repository because it discovers sibling
example files from the local ``examples/`` directory.
"""

import base64
import html
import importlib.util
import inspect
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path

import chatnificent as chat
from starlette.applications import Starlette
from starlette.responses import HTMLResponse, Response
from starlette.routing import Mount, Route

EXAMPLES_DIR = Path(__file__).parent.resolve()
# Where File/SQLite stores and uploaded files for mounted examples are written.
# Override with the CHATNIFICENT_WEBSITE_DATA env var so production deployments
# can keep state outside the git checkout (e.g. /srv/chatnificent/data).
DATA_ROOT = Path(os.environ.get("CHATNIFICENT_WEBSITE_DATA", "examples_hub_data"))
WEBSITE_FILENAME = "chatnificent_website.py"
MEMORY_TOOL_STEM = "memory_tool"
IMAGE_STUDIO_STEM = "openai_responses_image_studio"


def build_site(example_paths=None, data_root=DATA_ROOT):
    """Build the parent Starlette app plus its mounted Chatnificent children."""
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    mounted_examples = []
    failed_examples = []
    routes = []

    for path in _discover_example_paths(example_paths):
        stem = path.stem
        slug = _slugify(stem)
        try:
            entry = _load_example(path, slug, data_root)
        except Exception as exc:
            failure = {
                "stem": stem,
                "slug": slug,
                "path": str(path),
                "error": str(exc),
            }
            failed_examples.append(failure)
            print(f"[chatnificent_website] Skipping {path.name}: {exc}")
            continue

        mounted_examples.append(entry)
        routes.append(Mount(f"/chat/{slug}", app=_StripMountPrefix(entry["app"])))

    async def landing_page(request):
        return HTMLResponse(_render_landing_page(mounted_examples, failed_examples))

    app = Starlette(routes=[Route("/", landing_page), *routes])
    return app, mounted_examples, failed_examples


class _StripMountPrefix:
    """ASGI shim: strip ``root_path`` from ``scope['path']`` before delegating.

    Starlette 1.0's ``Mount`` keeps the mount prefix in both ``scope['path']``
    and ``scope['root_path']``. Chatnificent's framework routes
    (``/api/conversations/...``) are defined relative to the app, so without
    stripping the prefix none of them match when mounted under ``/chat/<slug>``.
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


def _discover_example_paths(example_paths=None):
    """Return eligible example paths in deterministic order."""
    if example_paths is not None:
        return sorted(
            (Path(path) for path in example_paths), key=lambda path: path.stem
        )

    return sorted(
        (path for path in EXAMPLES_DIR.glob("*.py") if _is_mountable_example(path)),
        key=lambda path: path.stem,
    )


def _is_mountable_example(path):
    """Filter out non-example and Starlette-only entrypoints."""
    return (
        path.suffix == ".py"
        and path.name != WEBSITE_FILENAME
        and path.name != "test_examples.py"
        and not path.stem.startswith("starlette_")
    )


def _load_example(path, slug, data_root):
    """Import, adapt, and describe a single child example."""
    source = path.read_text(encoding="utf-8")
    module = _import_example_module(path, slug, data_root)
    child_app = getattr(module, "app", None)

    if not isinstance(child_app, chat.Chatnificent):
        raise RuntimeError(
            "expected a top-level `app = chat.Chatnificent(...)` in the example module"
        )

    _remap_store(child_app, slug, data_root)

    if path.stem == MEMORY_TOOL_STEM:
        _remap_memory_root(module, slug, data_root)

    if path.stem == IMAGE_STUDIO_STEM:
        _adapt_image_studio(module, child_app, slug)
    else:
        _reject_unsupported_mounts(source, child_app)
        _attach_starlette_server(child_app)

    return {
        "stem": path.stem,
        "slug": slug,
        "title": _example_title(module, path.stem),
        "summary": _example_summary(module, path.stem),
        "path": str(path),
        "module": module,
        "app": child_app,
    }


def _slugify(stem):
    """Turn an example filename stem into a public URL slug."""
    return stem.replace("_", "-").replace(" ", "-")


def _example_title(module, stem):
    """Derive a landing-page title from the module docstring when possible."""
    doc = inspect.getdoc(module) or ""
    for line in doc.splitlines():
        line = line.strip()
        if line:
            return line
    return stem.replace("_", " ").replace("-", " ").title()


def _example_summary(module, stem):
    """Extract a short summary for the home-page cards."""
    doc = inspect.getdoc(module) or ""
    lines = [line.strip() for line in doc.splitlines()]
    parts = []

    for line in lines[1:]:
        if not line:
            if parts:
                break
            continue
        if set(line) <= {"=", "-"}:
            continue
        parts.append(line)

    text = " ".join(parts)
    if not text:
        text = f"Open the {stem.replace('_', ' ')} example inside the shared website."
    return _truncate_text(text, 160)


def _truncate_text(text, limit):
    """Trim long text without chopping words awkwardly."""
    if len(text) <= limit:
        return text
    trimmed = text[: limit - 3].rsplit(" ", 1)[0].rstrip(" ,.;:")
    return (trimmed or text[: limit - 3]) + "..."


@contextmanager
def _remapped_store_constructors(slug, data_root):
    """Redirect example-level File/SQLite constructors into the website sandbox."""
    original_file = chat.store.File
    original_sqlite = chat.store.SQLite

    def file_factory(base_dir):
        del base_dir
        return original_file(str(_file_store_dir(data_root, slug)))

    def sqlite_factory(db_path):
        del db_path
        return original_sqlite(str(_sqlite_store_path(data_root, slug)))

    chat.store.File = file_factory
    chat.store.SQLite = sqlite_factory
    try:
        yield
    finally:
        chat.store.File = original_file
        chat.store.SQLite = original_sqlite


def _import_example_module(path, slug, data_root):
    """Import an example under a synthetic module name."""
    module_name = f"_chatnificent_website_{slug.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not create an import spec for {path.name}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    with _remapped_store_constructors(slug, data_root):
        spec.loader.exec_module(module)

    return module


def _file_store_dir(data_root, slug):
    """Return the per-example File store root."""
    return Path(data_root) / slug / "files"


def _sqlite_store_path(data_root, slug):
    """Return the per-example SQLite database path."""
    return Path(data_root) / slug / "app.db"


def _remap_store(app, slug, data_root):
    """Replace persistent stores with website-scoped locations."""
    if isinstance(app.store, chat.store.File):
        app.store = chat.store.File(str(_file_store_dir(data_root, slug)))
    elif isinstance(app.store, chat.store.SQLite):
        app.store = chat.store.SQLite(str(_sqlite_store_path(data_root, slug)))


def _remap_memory_root(module, slug, data_root):
    """Move memory_tool.py's module-level directory into the website sandbox."""
    target = (Path(data_root) / slug / "memory").resolve()
    target.mkdir(parents=True, exist_ok=True)
    module.MEMORY_ROOT = target


def _reject_unsupported_mounts(source, app):
    """Catch examples that need an explicit adapter instead of a server swap."""
    if type(app.server) not in (chat.server.DevServer, chat.server.Starlette):
        raise RuntimeError(
            f"uses custom server {type(app.server).__name__}; add a website adapter"
        )
    if "/files/" in source:
        raise RuntimeError(
            "uses a hardcoded /files/ route; add a website adapter before mounting"
        )


def _attach_starlette_server(app, routes=None):
    """Swap an example onto Starlette so it can be mounted under the parent app."""
    server = chat.server.Starlette(routes=routes)
    app.server = server
    server.app = app
    server.create_server()
    return server


def _adapt_image_studio(module, app, slug):
    """Keep image-studio file URLs and downloads working under /chat/<slug>/."""
    mount_prefix = f"/chat/{slug}"

    class MountedImageStudioEngine(module.ImageStudioEngine):
        def __init__(self, mount_prefix):
            super().__init__()
            self._mount_prefix = mount_prefix

        def _save_conversation(self, conversation, user_id):
            last = conversation.messages[-1] if conversation.messages else None
            if last and last.get("role") == "assistant":
                content = last.get("content", "")
                if isinstance(content, str):
                    content = re.sub(
                        r'\n?<img data-gen-partial="1"[^>]*>\n?',
                        "",
                        content,
                    )
                    existing = self.app.store.list_files(user_id, conversation.id)
                    counter = [
                        sum(
                            1
                            for filename in existing
                            if filename.startswith("images/")
                            and filename.endswith(".jpeg")
                        )
                    ]

                    def _replace_img(match):
                        b64 = match.group(1)
                        try:
                            data = base64.b64decode(b64)
                        except (ValueError, TypeError):
                            return match.group(0)
                        filename = f"images/{counter[0]}.jpeg"
                        self.app.store.save_file(
                            user_id, conversation.id, filename, data
                        )
                        counter[0] += 1
                        url = (
                            f"{self._mount_prefix}/files/"
                            f"{user_id}/{conversation.id}/{filename}"
                        )
                        return (
                            f'<img src="{url}" '
                            'style="max-width:512px;height:auto" '
                            'alt="Generated image">'
                        )

                    content = re.sub(
                        r'<img src="data:image/jpeg;base64,([A-Za-z0-9+/=]+)"'
                        r"[^>]*>",
                        _replace_img,
                        content,
                    )
                    last["content"] = content

            raw_responses = self.app.store.load_raw_api_responses(
                user_id, conversation.id
            )
            if raw_responses and last and last.get("role") == "assistant":
                convo_id = self._extract_conversation_id(raw_responses[-1])
                if convo_id:
                    last["_openai_conversation_id"] = convo_id

            chat.engine.Orchestrator._save_conversation(self, conversation, user_id)

    async def serve_file(request):
        user_id = request.path_params["user_id"]
        convo_id = request.path_params["convo_id"]
        filename = request.path_params["filename"]

        if ".." in filename or "\\" in filename or filename.startswith("/"):
            return Response(status_code=400)

        session_id = request.cookies.get("chatnificent_session")
        current_user = app.auth.get_current_user_id(session_id=session_id)
        if user_id != current_user:
            return Response(status_code=403)

        data = app.store.load_file(user_id, convo_id, filename)
        if not data:
            return Response(status_code=404)

        content_type = (
            "image/png"
            if filename.endswith(".png")
            else "image/jpeg"
            if filename.endswith(".jpeg")
            else "application/octet-stream"
        )
        return Response(
            data,
            media_type=content_type,
            headers={"Cache-Control": "private, max-age=3600"},
        )

    app.engine = MountedImageStudioEngine(mount_prefix)
    app.engine.app = app
    _attach_starlette_server(
        app,
        routes=[
            Route(
                "/files/{user_id:str}/{convo_id:str}/{filename:path}",
                serve_file,
                methods=["GET"],
            )
        ],
    )


def _load_framework_theme_css():
    """Read :root and dark-mode CSS variable blocks from the framework template."""
    from chatnificent.layout import _TEMPLATES_DIR

    template = (_TEMPLATES_DIR / "default.html").read_text(encoding="utf-8")
    match = re.search(
        r'(:root \{.*?\})\s*(html\[data-theme=["\']dark["\']\] \{.*?\})',
        template,
        re.DOTALL,
    )
    if not match:
        return ""
    combined = match.group(1).strip() + "\n\n" + match.group(2).strip()
    lines = combined.splitlines()
    base = next((len(l) - len(l.lstrip()) for l in lines if l.strip()), 0)
    return "\n".join("    " + l[base:] if l.strip() else "" for l in lines)


def _render_landing_page(mounted_examples, failed_examples):
    """Render the website home page."""
    theme_css = _load_framework_theme_css()
    mounted_count = len(mounted_examples)
    failed_count = len(failed_examples)
    primary_entry = next(
        (entry for entry in mounted_examples if entry["slug"] == "quickstart"),
        mounted_examples[0] if mounted_examples else None,
    )
    search_entry = next(
        (
            entry
            for entry in mounted_examples
            if entry["slug"] == "openai-responses-interactive-search"
        ),
        None,
    )
    studio_entry = next(
        (
            entry
            for entry in mounted_examples
            if entry["slug"] == "openai-responses-image-studio"
        ),
        None,
    )

    TIERS = [
        (
            "Basics",
            ["quickstart", "llm_providers", "ollama_local", "openrouter_models"],
        ),
        (
            "Features",
            [
                "persistent_storage",
                "tool_calling",
                "system_prompt",
                "multi_tool_agent",
                "memory_tool",
                "memory_tool_multi_user",
            ],
        ),
        ("Customization", ["single_user", "auto_title"]),
        (
            "Display Enrichment",
            [
                "usage_display",
                "usage_display_multi_provider",
                "conversation_title",
                "conversation_summary",
                "web_search",
                "display_redaction",
            ],
        ),
        (
            "Production & Deployment",
            [
                "starlette_quickstart",
                "starlette_server_options",
                "starlette_uvicorn_options",
                "starlette_multi_mount",
            ],
        ),
        (
            "OpenAI Responses API",
            [
                "openai_responses",
                "openai_responses_website_search",
                "openai_responses_image_generator",
                "openai_responses_image_studio",
                "openai_responses_interactive_search",
            ],
        ),
        ("UI Interactions", ["ui_interactions"]),
    ]

    by_stem = {entry["stem"]: entry for entry in mounted_examples}

    def _make_card(entry):
        return (
            f'<a class="app-card" href="/chat/{entry["slug"]}/">'
            f'<div class="app-card-kicker">/{html.escape(entry["slug"])}/</div>'
            f'<div class="app-card-title">{html.escape(entry["title"])}</div>'
            f'<div class="app-card-copy">{html.escape(entry["summary"])}</div>'
            '<div class="app-card-footer">'
            f'<span class="app-card-pill">{html.escape(entry["stem"])}</span>'
            '<span class="app-card-arrow">Open</span>'
            "</div>"
            "</a>"
        )

    app_sections_html = []
    nav_links = []
    seen = set()
    for tier_label, stems in TIERS:
        cards = [_make_card(by_stem[s]) for s in stems if s in by_stem]
        if not cards:
            continue
        for s in stems:
            seen.add(s)
        tier_id = "tier-" + _slugify(tier_label.lower())
        nav_links.append(
            f'<a class="section-nav-pill" href="#{tier_id}">{html.escape(tier_label)}</a>'
        )
        app_sections_html.append(
            f'<section class="home-section" id="{tier_id}">'
            f'<div class="section-head"><h2>{html.escape(tier_label)}</h2></div>'
            f'<div class="app-grid">{chr(10).join(cards)}</div>'
            f"</section>"
        )

    # Catch any examples not assigned to a tier
    uncategorized = [entry for entry in mounted_examples if entry["stem"] not in seen]
    if uncategorized:
        cards = [_make_card(e) for e in uncategorized]
        nav_links.append('<a class="section-nav-pill" href="#tier-other">Other</a>')
        app_sections_html.append(
            '<section class="home-section" id="tier-other">'
            '<div class="section-head"><h2>Other</h2></div>'
            f'<div class="app-grid">{chr(10).join(cards)}</div>'
            "</section>"
        )

    app_sections = "\n".join(app_sections_html)
    section_nav = (
        f'<nav class="section-nav" aria-label="Sections">{"".join(nav_links)}</nav>'
        if nav_links
        else ""
    )

    failed_panel = ""
    if failed_examples:
        failed_items = "\n".join(
            (
                '<li class="failure-item">'
                f'<span class="failure-name">{html.escape(item["stem"])}</span>'
                f'<span class="failure-error">{html.escape(item["error"])}</span>'
                "</li>"
            )
            for item in failed_examples
        )
        failed_panel = (
            '<div class="home-panel warning-panel">'
            '<div class="panel-label">Skipped For Now</div>'
            "<h3>Some examples still need adapters</h3>"
            "<p>The website stays up and skips anything that does not mount cleanly yet.</p>"
            f'<ul class="failure-list">{failed_items}</ul>'
            "</div>"
        )

    empty_panel = ""
    if not mounted_examples:
        empty_panel = (
            '<div class="home-panel empty-panel">'
            "<h3>No examples loaded</h3>"
            "<p>Every discovered example failed to mount, so there is nothing to browse yet.</p>"
            "</div>"
        )

    primary_href = f"/chat/{primary_entry['slug']}/" if primary_entry else "#"
    search_href = f"/chat/{search_entry['slug']}/" if search_entry else primary_href
    studio_href = f"/chat/{studio_entry['slug']}/" if studio_entry else primary_href

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Chatnificent Website</title>
  <style>
    {theme_css}

    *,
    *::before,
    *::after {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    html {{
      background: var(--bg);
      scroll-behavior: smooth;
    }}

    body {{
      font-family: 'Trebuchet MS', 'Gill Sans', 'Gill Sans MT', Calibri, sans-serif;
      background:
        radial-gradient(circle at top left, var(--accent-subtle), transparent 28%),
        radial-gradient(circle at top right, rgba(37, 99, 235, 0.05), transparent 24%),
        var(--bg);
      color: var(--text);
      height: 100vh;
      overflow: hidden;
    }}

    #header {{
      height: var(--header-h);
      background: color-mix(in srgb, var(--surface) 92%, transparent);
      border-bottom: 1px solid var(--border);
      color: var(--text);
      display: flex;
      align-items: center;
      gap: 14px;
      padding: 0 20px;
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      z-index: 100;
      backdrop-filter: blur(14px);
    }}

    #theme-toggle {{
      background: none;
      border: none;
      color: var(--text-secondary);
      cursor: pointer;
      padding: 6px;
      border-radius: 8px;
      transition: background var(--transition), color var(--transition);
      display: flex;
      align-items: center;
      justify-content: center;
      width: 36px;
      height: 36px;
      flex-shrink: 0;
    }}

    #theme-toggle:hover {{
      background: var(--accent-subtle);
      color: var(--accent);
    }}

    #header-title {{
      font-size: 1.05rem;
      font-weight: 600;
      letter-spacing: -0.01em;
      text-decoration: none;
      color: inherit;
    }}

    #header-actions {{
      margin-left: auto;
      display: flex;
      align-items: center;
      gap: 6px;
    }}

    .theme-icon {{
      display: none;
    }}

    html:not([data-theme="dark"]) .theme-icon-light {{
      display: block;
    }}

    html[data-theme="dark"] .theme-icon-dark {{
      display: block;
    }}

    #new-chat-btn {{
      background: var(--accent);
      border: none;
      color: var(--btn-text);
      padding: 7px 16px;
      border-radius: 20px;
      font-size: 0.85rem;
      font-weight: 500;
      cursor: pointer;
      transition: background var(--transition), transform 0.15s ease;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      text-decoration: none;
    }}

    #new-chat-btn:hover {{
      background: var(--accent-hover);
    }}

    #new-chat-btn:active {{
      transform: scale(0.98);
    }}

    #chat-wrap {{
      position: fixed;
      top: var(--header-h);
      bottom: 0;
      left: 0;
      right: 0;
      display: flex;
      flex-direction: column;
    }}

    #messages {{
      flex: 1;
      overflow-y: auto;
      padding: 24px 20px 12px;
      display: flex;
      flex-direction: column;
      gap: 16px;
      max-width: 960px;
      margin: 0 auto;
      width: 100%;
    }}

    #messages::-webkit-scrollbar {{
      width: 5px;
    }}

    #messages::-webkit-scrollbar-thumb {{
      background: var(--border);
      border-radius: 5px;
    }}

    .home-hero {{
      position: relative;
      overflow: hidden;
      border-radius: 28px;
      border: 1px solid var(--border);
      background:
        linear-gradient(135deg, color-mix(in srgb, var(--accent) 13%, var(--surface)) 0%, var(--surface) 42%, color-mix(in srgb, var(--accent) 5%, var(--surface)) 100%);
      box-shadow: 0 24px 60px rgba(17, 24, 39, 0.08);
      padding: 28px;
    }}

    .home-hero::after {{
      content: "";
      position: absolute;
      inset: auto -90px -90px auto;
      width: 220px;
      height: 220px;
      border-radius: 50%;
      background: radial-gradient(circle, var(--accent-ring), transparent 72%);
      pointer-events: none;
    }}

    .hero-kicker {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-size: 0.78rem;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--accent);
      margin-bottom: 14px;
    }}

    .hero-title {{
      font-size: clamp(2rem, 5vw, 3.4rem);
      line-height: 0.98;
      letter-spacing: -0.04em;
      max-width: 12ch;
      margin-bottom: 14px;
    }}

    .hero-copy {{
      max-width: 54ch;
      color: var(--text-secondary);
      line-height: 1.7;
      font-size: 1rem;
    }}

    .hero-stats {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 18px;
    }}

    .hero-stat {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 10px 14px;
      border-radius: 999px;
      background: color-mix(in srgb, var(--surface) 86%, transparent);
      border: 1px solid var(--border);
      box-shadow: var(--shadow-sm);
      font-size: 0.88rem;
      color: var(--text-secondary);
    }}

    .hero-actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 24px;
    }}

    .hero-action {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      min-height: 44px;
      padding: 0 18px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: var(--surface);
      color: var(--text);
      text-decoration: none;
      font-size: 0.92rem;
      font-weight: 600;
      transition: transform 0.15s ease, border-color var(--transition), background var(--transition), color var(--transition);
      box-shadow: var(--shadow-sm);
    }}

    .hero-action:hover {{
      transform: translateY(-1px);
      border-color: var(--accent);
      color: var(--accent);
    }}

    .hero-action.primary {{
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
    }}

    .hero-action.primary:hover {{
      background: var(--accent-hover);
      border-color: var(--accent-hover);
      color: #fff;
    }}

    .msg {{
      max-width: 80%;
      padding: 12px 16px;
      line-height: 1.65;
      word-wrap: break-word;
      white-space: pre-wrap;
      animation: msgIn 0.3s ease-out;
    }}

    @keyframes msgIn {{
      from {{
        opacity: 0;
        transform: translateY(8px);
      }}

      to {{
        opacity: 1;
        transform: translateY(0);
      }}
    }}

    .msg.user {{
      align-self: flex-end;
      background: var(--user-bubble);
      color: var(--user-text);
      border-radius: var(--radius) var(--radius) 4px var(--radius);
      box-shadow: var(--shadow-sm);
    }}

    .msg.assistant {{
      align-self: flex-start;
      background: var(--assistant-bg);
      color: var(--text);
      border-radius: var(--radius) var(--radius) var(--radius) 4px;
      border: 1px solid var(--border);
      box-shadow: var(--shadow-sm);
      white-space: normal;
    }}

    .msg.assistant p + p {{
      margin-top: 0.6em;
    }}

    .home-section {{
      display: flex;
      flex-direction: column;
      gap: 14px;
      scroll-margin-top: calc(var(--header-h) + 16px);
    }}

    .section-nav {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 4px 0 8px;
    }}

    .section-nav-pill {{
      display: inline-flex;
      align-items: center;
      padding: 8px 14px;
      border-radius: 999px;
      background: var(--accent-subtle);
      color: var(--accent);
      text-decoration: none;
      font-size: 0.85rem;
      font-weight: 600;
      border: 1px solid transparent;
      transition: background var(--transition), color var(--transition), border-color var(--transition);
    }}

    .section-nav-pill:hover {{
      background: var(--accent);
      color: var(--btn-text);
      border-color: var(--accent);
    }}

    .section-head {{
      padding: 0 4px;
    }}

    .section-head h2 {{
      font-size: 1.35rem;
      letter-spacing: -0.02em;
    }}

    .section-head p {{
      color: var(--text-secondary);
      max-width: 52ch;
      line-height: 1.6;
      font-size: 0.92rem;
    }}

    .app-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
    }}

    .app-card {{
      display: flex;
      flex-direction: column;
      gap: 12px;
      min-height: 220px;
      padding: 18px;
      border-radius: 22px;
      border: 1px solid var(--border);
      background: color-mix(in srgb, var(--surface) 88%, transparent);
      color: inherit;
      text-decoration: none;
      box-shadow: var(--shadow-sm);
      transition: transform 0.16s ease, border-color var(--transition), box-shadow var(--transition), background var(--transition);
    }}

    .app-card:hover {{
      transform: translateY(-3px);
      border-color: var(--accent);
      box-shadow: 0 16px 38px rgba(17, 24, 39, 0.1);
      background: var(--surface);
    }}

    .app-card-kicker {{
      font-size: 0.76rem;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--accent);
    }}

    .app-card-title {{
      font-size: 1.04rem;
      font-weight: 700;
      line-height: 1.25;
      letter-spacing: -0.02em;
    }}

    .app-card-copy {{
      color: var(--text-secondary);
      line-height: 1.65;
      font-size: 0.92rem;
      flex: 1;
    }}

    .app-card-footer {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }}

    .app-card-pill {{
      display: inline-flex;
      align-items: center;
      min-height: 30px;
      padding: 0 10px;
      border-radius: 999px;
      background: var(--accent-subtle);
      color: var(--accent);
      font-size: 0.78rem;
      font-weight: 600;
    }}

    .app-card-arrow {{
      font-size: 0.84rem;
      font-weight: 700;
      color: var(--text-secondary);
    }}

    .home-panel {{
      border-radius: 24px;
      border: 1px solid var(--border);
      background: color-mix(in srgb, var(--surface) 90%, transparent);
      padding: 20px;
      box-shadow: var(--shadow-sm);
    }}

    .panel-label {{
      display: inline-flex;
      align-items: center;
      min-height: 28px;
      padding: 0 10px;
      border-radius: 999px;
      background: var(--accent-subtle);
      color: var(--accent);
      font-size: 0.76rem;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-bottom: 12px;
    }}

    .warning-panel {{
      border-style: dashed;
    }}

    .home-panel h3 {{
      font-size: 1.1rem;
      margin-bottom: 6px;
      letter-spacing: -0.02em;
    }}

    .home-panel p {{
      color: var(--text-secondary);
      line-height: 1.65;
    }}

    .failure-list {{
      list-style: none;
      display: grid;
      gap: 10px;
      margin-top: 16px;
    }}

    .failure-item {{
      display: flex;
      flex-direction: column;
      gap: 4px;
      padding: 12px 14px;
      border-radius: 16px;
      border: 1px solid var(--border);
      background: var(--surface);
    }}

    .failure-name {{
      font-weight: 700;
      letter-spacing: -0.01em;
    }}

    .failure-error {{
      color: var(--text-secondary);
      font-size: 0.9rem;
      line-height: 1.55;
    }}

    @media (max-width: 980px) {{
      .app-grid {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
    }}

    @media (max-width: 768px) {{
      #messages {{
        padding: 16px 14px 8px;
      }}

      .home-hero {{
        padding: 22px 18px;
      }}

      .hero-title {{
        max-width: none;
      }}

      .msg {{
        max-width: 92%;
      }}

      .app-grid {{
        grid-template-columns: 1fr;
      }}

      #input-inner {{
        align-items: flex-start;
      }}
    }}
  </style>
</head>
<body>
  <div id="header">
    <a id="header-title" href="/">Chatnificent</a>
    <div id="header-actions">
      <button id="theme-toggle" aria-label="Toggle theme"><svg class="theme-icon theme-icon-light" viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
        <circle cx="12" cy="12" r="5" />
        <line x1="12" y1="1" x2="12" y2="3" />
        <line x1="12" y1="21" x2="12" y2="23" />
        <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
        <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
        <line x1="1" y1="12" x2="3" y2="12" />
        <line x1="21" y1="12" x2="23" y2="12" />
        <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
        <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
      </svg><svg class="theme-icon theme-icon-dark" viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
      </svg></button>
      <a id="new-chat-btn" href="{html.escape(primary_href)}"><svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round">
        <line x1="12" y1="5" x2="12" y2="19" />
        <line x1="5" y1="12" x2="19" y2="12" />
      </svg> New chat</a>
    </div>
  </div>
  </div>
  <div id="chat-wrap">
    <div id="messages">
      <section class="home-hero">
        <div class="hero-kicker">Unified example browser</div>
        <h1 class="hero-title">One homepage. Every chat feels native.</h1>
        <p class="hero-copy">
          This landing page reuses the same visual system as the built-in chat UI so the
          website feels like one cohesive Chatnificent product instead of a list of scripts.
          Pick a demo from the cards below or jump straight into a featured app.
        </p>
        <div class="hero-stats">
          <div class="hero-stat"><strong>{mounted_count}</strong> mounted apps</div>
          <div class="hero-stat"><strong>{failed_count}</strong> skipped apps</div>
          <div class="hero-stat">Route prefix: <code>/chat/&lt;app&gt;/</code></div>
        </div>
        <div class="hero-actions">
          <a class="hero-action primary" href="{html.escape(primary_href)}">Start with quickstart</a>
          <a class="hero-action" href="{html.escape(search_href)}">Open interactive search</a>
          <a class="hero-action" href="{html.escape(studio_href)}">Browse image studio</a>
        </div>
      </section>

      <div class="msg assistant">
        <p><strong>Welcome.</strong> Every mounted example keeps its own pillars and storage, but the landing page speaks the same design language as the built-in chat apps.</p>
        <p>Use this as the front door to the whole examples collection, then drop into any mounted chat without feeling like you left the website.</p>
      </div>

      <div class="msg user">
        Try quickstart for the smallest possible app, interactive search for live controls, or image studio for the richest multimodal flow.
      </div>

      {section_nav}

      {app_sections}

      {failed_panel}
      {empty_panel}
    </div>

  </div>
  <script>
    function $(selector) {{
      return document.querySelector(selector);
    }}

    function setTheme(dark) {{
      document.documentElement.setAttribute("data-theme", dark ? "dark" : "light");
      try {{ localStorage.setItem("chatnificent-theme", dark ? "dark" : "light"); }} catch (e) {{}}
    }}

    (function initTheme() {{
      let saved = null;
      try {{ saved = localStorage.getItem("chatnificent-theme"); }} catch (e) {{}}
      if (saved === "dark" || saved === "light") {{
        setTheme(saved === "dark");
        return;
      }}
      const prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
      setTheme(prefersDark);
    }})();

    $("#theme-toggle").addEventListener("click", function () {{
      setTheme(document.documentElement.getAttribute("data-theme") !== "dark");
    }});
  </script>
</body>
</html>"""


app, mounted_examples, failed_examples = build_site()


if __name__ == "__main__":
    import uvicorn

    print("Chatnificent Website running on http://127.0.0.1:7777")
    print("  /             - landing page")
    print("  /chat/<name>/ - mounted example app")
    if failed_examples:
        print(f"  skipped {len(failed_examples)} example(s); see messages above")
    uvicorn.run(app, host="127.0.0.1", port=7777)
