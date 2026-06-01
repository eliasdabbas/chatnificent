# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "chatnificent[openai]>=0.0.25",
#     "fpdf2>=2.7",
#     "markdown>=3.5",
#     "pydantic>=2",
# ]
# ///
"""
Export the conversation as a styled PDF \u2014 maximally hackable
============================================================

The on-demand counterpart to ``conversation_export_pdf_simple.py``.
Same streaming-sentinel shape, three additions:

1. **A panel of controls above the composer** (Length, Depth, Tone)
   shapes the *structured summary* that goes on the cover page.
2. **An Export button** types ``"Export PDF"`` into the composer and
   clicks Send \u2014 a normal chat turn, no new endpoint, no JS plumbing.
3. **Structured outputs** (``client.chat.completions.parse`` +
   Pydantic schema) keep the summary section's typography
   deterministic. The schema is constant; the prompt tells the model
   which sections to fill.

The PDF leads with the summary (what you actually want to read),
followed by the full transcript on later pages (the source of truth,
in case you need to confirm a detail).

How the controls reach the LLM
-------------------------------
Each ``Control`` with an ``llm_param`` is auto-merged into the kwargs
that arrive at ``generate_response``. The framework looks up
``user_id`` for us via the engine seam ``_get_llm_kwargs(user_id)`` \u2014
we just name the params (``_pdf_length`` etc.) and pop them off
``kwargs`` in the LLM subclass before calling the real OpenAI client.
No engine override needed.

Prerequisites
-------------
::

    export OPENAI_API_KEY="sk-..."
    uv run --script artifact_examples/conversation_export_pdf_advanced.py
"""

from datetime import datetime
from html import escape
from typing import List

import chatnificent as chat
import markdown as md
from chatnificent.models import ASSISTANT_ROLE, USER_ROLE, Artifact
from fpdf import FPDF
from pydantic import BaseModel, Field

# =============================================================================
# Structured output schema + prompt
# =============================================================================


class PDFSummary(BaseModel):
    title: str = Field(description="Short title for the conversation, 3-8 words.")
    tldr: str = Field(description="One-paragraph summary capturing the essence.")
    key_points: List[str] = Field(default_factory=list)
    further_considerations: List[str] = Field(default_factory=list)


_TONES = {
    "professional": "Neutral, professional register. Crisp, no filler.",
    "conversational": "Direct, second-person ('you'). Warm but concise.",
}
_DEPTHS = {
    "tldr": "Fill ONLY `title` and `tldr`. Leave the lists empty.",
    "key_points": (
        "Fill `title`, `tldr`, and `key_points` (3-6 items). Leave "
        "`further_considerations` empty."
    ),
    "considerations": (
        "Fill all four fields. `key_points`: 3-6 items. "
        "`further_considerations`: 2-5 open questions or next steps."
    ),
}


def _build_summary_prompt(transcript: str, length: str, tone: str, depth: str) -> str:
    return f"""Summarise the conversation below for a standalone PDF.

Style: {_TONES.get(tone, _TONES["professional"])}
Target length: roughly {length} words across all fields.
{_DEPTHS.get(depth, _DEPTHS["key_points"])}

Synthesise \u2014 do not replay turn-by-turn.

CONVERSATION
---
{transcript}
---
"""


# =============================================================================
# PDF rendering \u2014 same shape as the simple version, with a summary cover
# =============================================================================

# Helvetica is latin-1 only. Same fold/drop trade-off as the simple version.
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
PDF Conversation Summary\
</a></div>"""


def _strip_prior_links(text: str) -> str:
    if _PDF_LINK_MARKER not in text:
        return text
    out, rest = [], text
    while _PDF_LINK_MARKER in rest:
        head, _, tail = rest.partition(_PDF_LINK_MARKER)
        out.append(head)
        _, _, rest = tail.partition("</div>")
    out.append(rest)
    return "".join(out).rstrip()


def _render_summary_section(pdf: FPDF, s: PDFSummary) -> None:
    """Cover section: title, date, italic TL;DR, optional bullet lists."""
    pdf.set_font("Helvetica", "B", 20)
    pdf.write_html(f"<h1>{escape(_to_latin1(s.title))}</h1>")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(120)
    pdf.cell(
        0,
        6,
        datetime.now().strftime("Summary generated %B %d, %Y at %H:%M"),
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.set_text_color(0)
    pdf.ln(4)

    pdf.set_font("Helvetica", "I", 11)
    pdf.write_html(f"<p>{escape(_to_latin1(s.tldr))}</p>")
    pdf.ln(2)

    if s.key_points:
        pdf.set_font("Helvetica", "B", 12)
        pdf.write_html("<h3>Key Points</h3>")
        items = "".join(f"<li>{escape(_to_latin1(x))}</li>" for x in s.key_points)
        pdf.write_html(f"<ul>{items}</ul>")

    if s.further_considerations:
        pdf.set_font("Helvetica", "B", 12)
        pdf.write_html("<h3>Further Considerations</h3>")
        items = "".join(
            f"<li>{escape(_to_latin1(x))}</li>" for x in s.further_considerations
        )
        pdf.write_html(f"<ul>{items}</ul>")


def _render_pdf(messages, summary: PDFSummary) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    _render_summary_section(pdf, summary)

    # Transcript on its own page \u2014 read the summary first, consult the
    # transcript to verify.
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, "Full Transcript", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(120)
    pdf.cell(
        0,
        6,
        "Source of truth for the summary above.",
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
            html = f"<p>{escape(_to_latin1(content))}</p>"
        else:
            html = md.markdown(_to_latin1(_strip_prior_links(content)))
        pdf.write_html(html)
        pdf.ln(4)

    return bytes(pdf.output())


# =============================================================================
# LLM \u2014 branch on "Export PDF", everything else is the regular OpenAI path
# =============================================================================

_EXPORT_TRIGGER = "export pdf"


class OpenAIChatPlusPDF(chat.llm.OpenAI):
    """Same streaming-sentinel shape as the simple version, on-demand only.

    When the latest user message is exactly ``"Export PDF"``, run a
    structured-output summary call and yield a single ``("pdf", bytes)``
    sentinel. Otherwise pass straight through to the real chat stream.
    """

    # Control values arrive as kwargs thanks to ``Control(llm_param=...)``
    # being auto-merged by ``Default.get_llm_kwargs(user_id)``. They aren't
    # valid OpenAI parameters, so pop them off before calling super() either
    # way.
    def generate_response(self, messages, **kwargs):
        length = kwargs.pop("_pdf_length", "500")
        tone = kwargs.pop("_pdf_tone", "professional")
        depth = kwargs.pop("_pdf_depth", "key_points")

        last = messages[-1] if messages else {}
        is_export = (
            isinstance(last.get("content"), str)
            and last["content"].strip().lower() == _EXPORT_TRIGGER
        )
        if not is_export:
            return super().generate_response(messages, **kwargs)
        return self._export(messages, length, tone, depth)

    def _export(self, messages, length, tone, depth):
        transcript = self._format_transcript(messages[:-1])
        prompt = _build_summary_prompt(transcript, length, tone, depth)

        # Non-streaming structured-output call. The model returns a
        # parsed PDFSummary instance directly.
        completion = self.client.chat.completions.parse(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format=PDFSummary,
        )
        summary = completion.choices[0].message.parsed

        yield ("pdf", _render_pdf(messages[:-1], summary))

    def _format_transcript(self, messages) -> str:
        lines = []
        for msg in messages:
            role = msg.get("role")
            if role not in (USER_ROLE, ASSISTANT_ROLE):
                continue
            content = msg.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            text = _strip_prior_links(content).strip()
            if not text:
                continue
            lines.append(f"{'User' if role == USER_ROLE else 'Assistant'}: {text}")
        return "\n\n".join(lines)

    def extract_stream_delta(self, chunk):
        if isinstance(chunk, tuple) and len(chunk) == 2 and chunk[0] == "pdf":
            return Artifact(
                data=chunk[1],
                ext=".pdf",
                folder="exports",
                html=_PDF_LINK_HTML,
            )
        return super().extract_stream_delta(chunk)


# =============================================================================
# UI \u2014 panel of dropdowns + plain export button (uses default token styles)
# =============================================================================

_PANEL_STYLE = """
<style>
.cn-pdf-bar { display:flex; justify-content:center; padding:0 20px; margin-bottom: var(--space-2); }
.cn-pdf-panel {
  display:flex; flex-wrap:wrap; align-items:end;
  gap: var(--space-2) var(--space-3);
  padding: var(--space-3) var(--space-4);
  width:100%; max-width: var(--chat-max-w);
  border:1px solid var(--border); border-radius: var(--radius);
  background: var(--bg-elev); box-shadow: var(--shadow-sm);
  font-family: var(--font);
}
.cn-pdf-panel-title {
  flex:1 0 100%; font-size: var(--text-xs); font-weight:600;
  letter-spacing:0.06em; text-transform:uppercase; color: var(--text-secondary);
}
.cn-pdf-panel label { font-size: var(--text-xs); color: var(--text-secondary); }
.cn-pdf-panel select { min-width: 9rem; }
</style>
"""


def _select(cid: str, label: str, options: list[tuple[str, str]]) -> str:
    opts = "".join(f'<option value="{v}">{t}</option>' for v, t in options)
    return (
        f'<div><label for="{cid}">{label}</label><br>'
        f'<select id="{cid}" onchange="chatInteraction(this)">{opts}</select></div>'
    )


# Plain <button> \u2014 Default's CSS turns it into the black accent pill.
# The onclick submits a normal chat turn; no interactions endpoint, no
# JS plumbing.
_EXPORT_BUTTON = (
    '<button id="pdf-export" '
    "onclick=\"var i=document.getElementById('input');"
    "i.value='Export PDF';document.getElementById('send').click();\">"
    "Export PDF</button>"
)


CONTROLS = [
    chat.layout.Control(
        id="cn-pdf-style-block",
        slot="messages-end",
        html=(
            _PANEL_STYLE + '<div class="cn-pdf-bar"><div class="cn-pdf-panel">'
            '<div class="cn-pdf-panel-title">PDF Conversation Summary</div>'
        ),
    ),
    chat.layout.Control(
        id="pdf-length",
        slot="messages-end",
        llm_param="_pdf_length",
        html=_select(
            "pdf-length",
            "Length",
            [
                ("250", "Brief (~250 words)"),
                ("500", "Standard (~500 words)"),
                ("1000", "Detailed (~1000 words)"),
            ],
        ),
    ),
    chat.layout.Control(
        id="pdf-depth",
        slot="messages-end",
        llm_param="_pdf_depth",
        html=_select(
            "pdf-depth",
            "Depth",
            [
                ("tldr", "TL;DR only"),
                ("key_points", "+ Key Points"),
                ("considerations", "+ Further Considerations"),
            ],
        ),
    ),
    chat.layout.Control(
        id="pdf-tone",
        slot="messages-end",
        llm_param="_pdf_tone",
        html=_select(
            "pdf-tone",
            "Tone",
            [
                ("professional", "Professional"),
                ("conversational", "Conversational"),
            ],
        ),
    ),
    chat.layout.Control(
        id="pdf-export",
        slot="messages-end",
        html=_EXPORT_BUTTON + "</div></div>",
    ),
]


welcome_message = """## Maximally hackable PDF export

Chat normally. When ready, tune the **PDF Conversation Summary**
controls above the composer and hit **Export PDF** \u2014 the model
summarises the conversation, the PDF leads with that summary and
keeps the full transcript on the following pages.

<div id="suggestions">
  <button class="suggestion" data-insert-prompt="Let's plan a small open-source CLI tool together.">
    <span class="suggestion-label">PLAN</span>
    <span class="suggestion-text">Multi-turn planning.</span>
  </button>
  <button class="suggestion" data-insert-prompt="Critique the architecture of a typical FastAPI + SQLAlchemy app.">
    <span class="suggestion-label">CRITIQUE</span>
    <span class="suggestion-text">Long, opinionated reply.</span>
  </button>
</div>
"""


app = chat.Chatnificent(
    llm=OpenAIChatPlusPDF(model="gpt-4o-mini"),
    store=chat.store.File(
        base_dir="./artifact_examples/_convos_conversation_export_pdf_advanced"
    ),
    layout=chat.layout.Default(welcome_message=welcome_message, controls=CONTROLS),
)


if __name__ == "__main__":
    app.run()
