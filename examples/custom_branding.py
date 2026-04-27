# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "chatnificent",
# ]
# ///
"""
Custom Branding — Make Chatnificent Look Like Your Product
==========================================================

The default chat UI ships under the *Chatnificent* brand. With one constructor
call you can rebrand it to look like *your* product — name, slogan, logo,
favicon, browser-tab title, and the welcome message that greets users on an
empty chat — without subclassing anything or touching the HTML template.

This example uses the zero-dependency ``Echo`` LLM so you can see the new
branding immediately, without any API keys.

What You Configure
------------------
``DefaultLayout`` accepts six pure-content branding parameters:

- ``brand`` — header display name (also used as the default page-title suffix)
- ``slogan`` — short tagline rendered next to the brand
- ``logo_url`` — image shown left of the brand text in the header
- ``favicon_url`` — browser-tab icon
- ``page_title`` — full ``<title>`` override (defaults to a Chatnificent title)
- ``welcome_message`` — Markdown rendered in the empty chat area, sanitized
  client-side via marked.js + DOMPurify

DevServer-Friendly Local Assets
-------------------------------
The built-in ``DevServer`` doesn't serve static files — it only serves the
chat page and JSON API endpoints. So *any* URL the browser can fetch works
here: an absolute ``https://`` URL, a relative path your reverse proxy will
serve, or — for self-contained DevServer demos like this one — a ``data:``
URI. SVG ``data:`` URIs are tiny and don't even need base64 encoding.

Running
-------
::

    uv run examples/custom_branding.py

Then open http://127.0.0.1:7777 and notice:

- The browser tab shows "Acme Support — Chat with us" and the orange dot favicon
- The header reads "Acme" + "Support" with the orange logo on the left
- The empty chat shows your custom welcome message with a working link

What to Explore Next
--------------------
- Swap ``Echo`` for a real provider — see ``llm_providers.py``
- Combine branding with a system prompt to define the assistant's voice as
  well as its appearance — see ``system_prompt.py``
- Add interactive UI controls to the same layout — see ``ui_interactions.py``
"""

import chatnificent as chat

# Inline SVG data URIs keep the example self-contained — no static file server
# required. Any URL the browser can fetch works here (https://, relative, etc).
LOGO = """data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'><circle cx='12' cy='12' r='10' fill='%23ff6b35'/><text x='12' y='17' text-anchor='middle' font-size='14' font-family='sans-serif' font-weight='700' fill='white'>A</text></svg>"""

WELCOME = """
## Welcome to Acme Support 👋

How can we help you today? Common topics:

- **Order status** — find a recent order
- **Returns & refunds** — start a return
- **Account help** — login, billing, settings

### Other options

Or just type your question below. A real person is one click away if you'd
rather [talk to a human](https://example.com/contact)."""


app = chat.Chatnificent(
    layout=chat.layout.DefaultLayout(
        brand="Acme Support",
        slogan="We're here to help",
        logo_url=LOGO,
        favicon_url=LOGO,
        page_title="Acme Support — Chat with us",
        welcome_message=WELCOME,
    ),
)


if __name__ == "__main__":
    app.run()
