# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "chatnificent[anthropic]",
# ]
# ///
"""
Display Redaction — Mask Sensitive Data in the UI Without Touching History
==========================================================================

This example upgrades the old ``blocked_words.txt`` idea into something more
practical:

1. let Anthropic answer normally
2. keep the stored conversation completely canonical
3. use ``Layout.render_messages(...)`` to redact common sensitive data only in
   the visible transcript

The important idea is that the conversation history is not the UI.

Here the layout uses a few built-in regex rules to mask email addresses, phone
numbers, and payment-card numbers. The stored messages and raw provider payloads
stay untouched, which means the next model turn still sees the original text.

Why this is useful
------------------
Redaction, highlighting, moderation badges, and translation overlays are often
display concerns, not canonical-history concerns. When you keep them in the
Layout layer, you get a few nice properties for free:

- you can change the display rule later without rewriting history
- different layouts can interpret the same stored messages differently
- the saved conversation folder remains easy to inspect and debug

This example intentionally keeps Anthropic streaming enabled, and it also
overrides the page rendering slightly so the first streamed bubble stays raw.
The page does not auto-reload the conversation after that first response. Once
you manually refresh the page, or reopen the conversation later,
``render_messages(...)`` applies the display redaction and the transcript
becomes masked. That makes the example very clearly a Layout concern.

Prerequisites
-------------
Set your Anthropic API key before running the script::

    export ANTHROPIC_API_KEY="sk-ant-..."

Running
-------
::

    uv run --script examples/display_redaction.py

The saved chats go into ``display_redaction_convos`` in the current directory
so you can inspect the unredacted canonical files directly.

Prompts to Try
--------------
Use prompts that encourage the assistant to repeat fake sensitive-looking data
in its answer, for example:

- ``Rewrite this support note in a friendlier tone: Email robin@gmail.com, call (555) 123-9876, and mention the test card 4111 1111 1111 8742.``
- ``Turn this billing note into a customer update: We tried calling +1 555-222-9876 and then emailed riley@protonmail.com about card 4242-4242-4242-8742.``
- ``Summarize this follow-up exactly, keeping the identifiers: Customer email is sam@example.com and callback number is 555 444 9876.``

What to Explore Next
--------------------
- Add more patterns such as SSNs or API keys
- Use different masking styles for different audiences
- Combine this with ``conversation_summary.py`` or ``usage_display.py``
"""

import re

import chatnificent as chat

EMAIL_RE = re.compile(r"\b([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+)\.([A-Za-z]{2,})\b")
PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?1[-.\s]*)?(?:\(\d{3}\)|\d{3})[-.\s]*\d{3}[-.\s]*\d{4}(?!\d)"
)
CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")


def _mask_hint(value: str) -> str:
    """Return a short masked hint that preserves only the first character."""
    if not value:
        return "****"
    return value[0] + "****"


def _mask_email(match: re.Match[str]) -> str:
    """Mask an email address while keeping a small recognizable hint."""
    local, domain, tld = match.groups()
    return f"{_mask_hint(local)}@{_mask_hint(domain)}.{tld}"


def _mask_phone(match: re.Match[str]) -> str:
    """Mask a phone number while keeping the last 4 digits."""
    digits = re.sub(r"\D", "", match.group())
    return f"XXXX{digits[-4:]}"


def _mask_card(match: re.Match[str]) -> str:
    """Mask a card-like number while keeping the last 4 digits."""
    digits = re.sub(r"\D", "", match.group())
    if not 13 <= len(digits) <= 19:
        return match.group()
    return f"XXXX{digits[-4:]}"


def _redact_text(text: str) -> str:
    """Apply the example's built-in display-only redaction rules."""
    redacted = EMAIL_RE.sub(_mask_email, text)
    redacted = PHONE_RE.sub(_mask_phone, redacted)
    redacted = CARD_RE.sub(_mask_card, redacted)
    return redacted


class RedactionLayout(chat.layout.DefaultLayout):
    """Redact common sensitive data at display time only."""

    def render_page(self) -> str:
        """Keep the first streamed response raw until a manual refresh."""
        page = super().render_page()
        return page.replace(
            "loadConvo(convoId, true);",
            "/* display_redaction: manual refresh applies layout redaction */",
        )

    def render_messages(self, messages, **kwargs):
        rendered = super().render_messages(messages, **kwargs)

        for message in rendered:
            content = message.get("content")
            if not isinstance(content, str):
                continue
            message["content"] = _redact_text(content)

        return rendered


app = chat.Chatnificent(
    llm=chat.llm.Anthropic(),
    layout=RedactionLayout(),
    store=chat.store.File(base_dir="display_redaction_convos"),
)

if __name__ == "__main__":
    app.run()
