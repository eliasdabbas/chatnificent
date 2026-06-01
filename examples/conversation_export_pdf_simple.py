# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "chatnificent[openai]>=0.0.25",
#     "fpdf2>=2.7",
#     "markdown>=3.5",
# ]
# ///
"""
Export each turn as a PDF — minimally complete
==============================================

After every assistant reply, render the running conversation as a PDF
and embed a download link **inside the same streamed message** — no
refresh, no second turn.

The structural truth this example exists to teach
--------------------------------------------------
``extract_stream_delta`` is allowed to return ``str``, ``Artifact``, or
``None`` — that's the whole streaming contract. A single assistant
turn can interleave text deltas and finished binary files. We use that
to stream the chat reply normally, then — once the stream ends —
synthesise a PDF of the entire conversation (including the reply we
just accumulated) and yield it as a sentinel the engine turns into
an ``<a>`` download link in the same message.

Same shape as ``openai_tts_advanced.py``: the LLM is a generator that
chains the real provider stream with one extra "thing" at the end.

Markdown rendering
------------------
The model speaks Markdown; PDFs don't. We convert each message with
the ``markdown`` library and feed the resulting HTML to ``write_html``,
so headers, bold/italic, and lists render as actual formatting rather
than literal ``##`` and ``**``.

Prerequisites
-------------
::

    export OPENAI_API_KEY="sk-..."
    uv run --script artifact_examples/conversation_export_pdf_simple.py
"""

from datetime import datetime
from html import escape

import chatnificent as chat
import markdown as md
from chatnificent.models import ASSISTANT_ROLE, USER_ROLE, Artifact
from fpdf import FPDF

# Built-in Helvetica is latin-1 only. Smart quotes, em-dashes, emoji,
# anything else crashes the PDF renderer. For "minimally complete" we
# fold the most common offenders to ASCII and drop the rest. The
# advanced version ships a Unicode TTF and keeps everything intact.
_FOLDS = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
        "\u00a0": " ",
    }
)


def _to_latin1(text: str) -> str:
    return text.translate(_FOLDS).encode("latin-1", "replace").decode("latin-1")


# Custom HTML wrapper for the PDF link. ``{url}`` is the only placeholder
# we use — ``{filename}`` would be ``0.pdf``, ``1.pdf``, etc. (counter-based
# per folder), which is true on disk but not what we want users to see.
# The leading marker comment lets us strip prior links back out when we
# re-render the conversation (otherwise every PDF would contain links to
# every previous PDF).
_PDF_LINK_MARKER = "<!--cn-pdf-link-->"
_PDF_LINK_HTML = f"""{_PDF_LINK_MARKER}<div style="margin-top:1em">\
<a href="{{url}}" style="display:inline-flex;align-items:center;gap:0.4em;\
font-weight:bold;font-style:italic;text-decoration:none">\
<svg width="16" height="16" viewBox="0 0 24 24" fill="none" \
stroke="currentColor" stroke-width="2" stroke-linecap="round" \
stroke-linejoin="round">\
<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>\
<polyline points="14 2 14 8 20 8"/>\
<text x="7" y="18" font-size="6" font-weight="bold" stroke="none" \
fill="currentColor">PDF</text>\
</svg>\
PDF Conversation Transcript\
</a></div>"""


def _strip_prior_links(text: str) -> str:
    """Remove previously-appended PDF links so each export is link-free."""
    if _PDF_LINK_MARKER not in text:
        return text
    out = []
    rest = text
    while _PDF_LINK_MARKER in rest:
        head, _, tail = rest.partition(_PDF_LINK_MARKER)
        out.append(head)
        # Drop everything up to and including the closing </div> of our wrapper.
        _, _, rest = tail.partition("</div>")
    out.append(rest)
    return "".join(out).rstrip()


def _render_pdf(messages) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Chatnificent - Conversation Export", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(120)
    pdf.cell(
        0,
        6,
        datetime.now().strftime("Exported %B %d, %Y at %H:%M"),
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.set_text_color(0)
    pdf.ln(4)

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role not in (USER_ROLE, ASSISTANT_ROLE):
            continue
        if not isinstance(content, str) or not content.strip():
            continue

        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*((30, 80, 160) if role == USER_ROLE else (40, 120, 60)))
        pdf.cell(
            0,
            6,
            "YOU" if role == USER_ROLE else "ASSISTANT",
            new_x="LMARGIN",
            new_y="NEXT",
        )
        pdf.set_text_color(0)
        pdf.set_font("Helvetica", "", 11)

        if role == USER_ROLE:
            # User input is plain text — render as a paragraph, no markdown.
            html = f"<p>{escape(_to_latin1(content))}</p>"
        else:
            html = md.markdown(_to_latin1(_strip_prior_links(content)))

        pdf.write_html(html)
        pdf.ln(4)

    return bytes(pdf.output())


class OpenAIChatPlusPDF(chat.llm.OpenAI):
    def generate_response(self, messages, **kwargs):
        chat_stream = super().generate_response(messages, **kwargs)
        return self._chat_then_pdf(messages, chat_stream)

    def _chat_then_pdf(self, messages, chat_stream):
        spoken = []
        for chunk in chat_stream:
            try:
                delta = chunk.choices[0].delta.content
            except (AttributeError, IndexError):
                delta = None
            if delta:
                spoken.append(delta)
            yield chunk

        reply = "".join(spoken).strip()
        if not reply:
            return

        full = list(messages) + [{"role": ASSISTANT_ROLE, "content": reply}]
        yield ("pdf", _render_pdf(full))

    def extract_stream_delta(self, chunk):
        if isinstance(chunk, tuple) and len(chunk) == 2 and chunk[0] == "pdf":
            return Artifact(
                data=chunk[1],
                ext=".pdf",
                folder="exports",
                html=_PDF_LINK_HTML,
            )
        return super().extract_stream_delta(chunk)


welcome_message = """## Export every turn as a PDF

Just chat. After each assistant reply, a **PDF Conversation Transcript**
link appears underneath — click it to grab the full conversation up to
that point as a properly formatted PDF.

<div id="suggestions">
  <button class="suggestion" data-insert-prompt="Give me three tips for writing a great cover letter.">
    <span class="suggestion-label">ADVICE</span>
    <span class="suggestion-text">Get a multi-point answer.</span>
  </button>
  <button class="suggestion" data-insert-prompt="Plan a 3-day trip to Lisbon for a foodie.">
    <span class="suggestion-label">TRAVEL</span>
    <span class="suggestion-text">Build up over several turns.</span>
  </button>
</div>
"""


app = chat.Chatnificent(
    llm=OpenAIChatPlusPDF(model="gpt-4o-mini"),
    store=chat.store.File(
        base_dir="./artifact_examples/_convos_conversation_export_pdf_simple"
    ),
    layout=chat.layout.Default(welcome_message=welcome_message),
)


if __name__ == "__main__":
    app.run()
