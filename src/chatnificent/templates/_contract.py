"""Template contract — the locked vocabulary every Chatnificent template honors.

A Chatnificent page is an HTML string with named, addressable locations.
This module publishes those names as immutable tuples so the framework, its
tests, and downstream forks can all agree on the same surface.

Three categories of contract:

* **Body slots** (`BODY_SLOTS`) — content addresses expressed as
  ``<div data-slot="name">`` in the template HTML. Devs replace what's inside
  via constructor kwargs (build-time) or by overriding ``render_page()``
  (runtime). Any HTML-valid `<div>` location may be a slot.
* **Required element IDs** (`REQUIRED_ELEMENT_IDS`) — JavaScript-addressable
  behavior anchors (e.g. ``#messages``, ``#send``). Every template must
  include these so framework JS can wire up event handlers.
* **Build markers** (`BUILD_MARKERS`) — template-author plumbing for
  inlining vendor scripts, styles, and first-party JS at construction time.
  They live in the template HTML as plain HTML comments
  (``<!-- VENDOR -->``, ``<!-- STYLES -->``, ``<!-- SCRIPTS -->``) outside
  any wrapper tag so embedded-CSS/JS linters stay quiet on the source;
  the framework wraps the inlined CSS/JS in ``<style>`` / ``<script>``
  tags at build time.

Standard HTML elements (``<title>``, ``<link rel="icon">``, ``<meta>``) are
their own addresses where ``<div>`` isn't valid — devs target them with
universally known selectors, no framework-invented names.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Union

__all__ = [
    "BODY_SLOTS",
    "REQUIRED_ELEMENT_IDS",
    "BUILD_MARKERS",
    "ValidationError",
    "validate_template",
]


# ---------------------------------------------------------------------------
# Locked public vocabulary
# ---------------------------------------------------------------------------

BODY_SLOTS: tuple[str, ...] = (
    "brand",
    "slogan",
    "header-begin",
    "header-end",
    "sidebar-begin",
    "sidebar-end",
    "welcome-message",
    "messages-begin",
    "messages-end",
    "composer-leading",
    "composer-primary",
    "composer-trailing",
    "composer-send",
    "composer-attachments",
    "footer-begin",
    "footer-end",
)

REQUIRED_ELEMENT_IDS: tuple[str, ...] = (
    "messages",
    "input",
    "send",
    "sidebar",
    "convo-list",
    "chat-wrap",
    "welcome",
    "welcome-message",
    "sidebar-toggle",
    "theme-toggle",
    "new-chat-btn",
)

BUILD_MARKERS: tuple[str, ...] = (
    "VENDOR",
    "STYLES",
    "SCRIPTS",
)


_BUILTIN_TEMPLATES_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class ValidationError(Exception):
    """Raised when a template folder violates the contract."""


def _resolve(template: Union[str, Path]) -> Path:
    """Resolve a template name or path to a folder, raising on missing built-ins."""
    if isinstance(template, Path):
        return template
    candidate = _BUILTIN_TEMPLATES_DIR / template
    if not candidate.is_dir():
        raise ValidationError(
            f"Unknown built-in template {template!r}: {candidate} does not exist"
        )
    return candidate


def validate_template(template: Union[str, Path]) -> None:
    """Validate a template folder against the contract.

    Parameters
    ----------
    template : str or Path
        Either a built-in template name (e.g. ``"default"``) resolved
        against ``src/chatnificent/templates/`` or an absolute/relative
        ``Path`` to any folder.

    Raises
    ------
    ValidationError
        If the folder is missing ``template.html``, lacks a required
        ``<div data-slot="...">``, omits a required element ID, drops a
        build marker, or has no ``<title>`` in ``<head>``.

    Notes
    -----
    The ``<title>`` check is scoped to the ``<head>`` block so that stray
    ``<title>`` elements elsewhere in the document (notably inside SVGs,
    where ``<title>`` is a valid accessibility annotation) cannot falsely
    satisfy the contract.
    """
    folder = _resolve(template)
    html_path = folder / "template.html"
    if not html_path.is_file():
        raise ValidationError(f"{folder}: missing template.html")
    html = html_path.read_text(encoding="utf-8")

    missing_slots = [
        name
        for name in BODY_SLOTS
        if not re.search(rf'<div\b[^>]*\bdata-slot=["\']{re.escape(name)}["\']', html)
    ]
    if missing_slots:
        raise ValidationError(
            f"{folder}: missing body slots: {', '.join(missing_slots)}"
        )

    missing_ids = [
        name
        for name in REQUIRED_ELEMENT_IDS
        if not re.search(rf'\bid=["\']{re.escape(name)}["\']', html)
    ]
    if missing_ids:
        raise ValidationError(
            f"{folder}: missing required element IDs: {', '.join(missing_ids)}"
        )

    missing_markers = [name for name in BUILD_MARKERS if f"<!-- {name} -->" not in html]
    if missing_markers:
        raise ValidationError(
            f"{folder}: missing build markers: {', '.join(missing_markers)}"
        )

    head_match = re.search(
        r"<head\b[^>]*>(.*?)</head>", html, re.IGNORECASE | re.DOTALL
    )
    head_content = head_match.group(1) if head_match else ""
    if not re.search(
        r"<title\b[^>]*>.*?</title>", head_content, re.IGNORECASE | re.DOTALL
    ):
        raise ValidationError(f"{folder}: missing <title> in <head>")
