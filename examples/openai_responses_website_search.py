# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "chatnificent[openai]",
# ]
# ///
"""
AI-Powered Website Search — Battle-Test for Transparent Endpoints
==================================================================

A domain-restricted research assistant built on OpenAI's hosted web
search. The default scope is Wikipedia; swap it for your own docs,
a news site, or any internal knowledge base indexed on the web.

Two Chatnificent principles on display:

**Minimally Complete**
    Enabling domain-restricted, always-on web search requires **zero
    new code** beyond the prior ``openai_responses.py`` subclass. Two
    constructor kwargs flow through ``default_params`` straight into
    ``responses.create``::

        OpenAIResponses(
            tools=[
                {
                    "type": "web_search",
                    "filters": {"allowed_domains": ["en.wikipedia.org"]},
                }
            ],
            tool_choice="required",
        )

    Together they fully describe the product: the only tool exists,
    and it must be used. No prose instructions needed — the
    constraints define the behavior. This is the "transparent"
    promise: vendor-specific capabilities reached through
    provider-native parameters, no framework plumbing.

    A note on ``tool_choice``: it accepts ``"none" | "auto" |
    "required"`` (generic), or a typed object for forcing a specific
    hosted tool. The typed form lists ``web_search_preview`` but not
    the newer ``web_search`` (which is the only variant that supports
    ``filters.allowed_domains``). So for domain-restricted search,
    ``"required"`` is the forcing mechanism — works because only one
    tool is configured.

**Maximally Hackable**
    The hosted web search annotates text with URL citations. The
    engine's default streaming path only captures text deltas, so
    citations would ordinarily be discarded. One extra branch in
    ``extract_stream_delta`` catches the ``response.output_item.done``
    event, formats the annotations as a collapsible ``<details>``
    block, and appends it to the accumulated stream. Every answer now
    ends with a "Sources" disclosure users can click to verify
    provenance.

    Cost of the hackability layer: ~15 lines, one helper method.

What This Example Does NOT Do (Yet)
------------------------------------
- **No in-text citation markers.** Annotations include
  ``start_index`` / ``end_index`` positions — we could render them as
  inline superscripts, but don't.
- **No progress status.** ``web_search_call`` lifecycle events
  (searching, completed) aren't surfaced to the UI.
  ``extract_stream_delta`` can only return text today; a status
  channel would be a framework change.
- **Citations aren't persisted structurally.** They're baked into the
  assistant message's markdown ``content``. That round-trips through
  history correctly, but the structured annotation objects are lost.

Running
-------
::

    export OPENAI_API_KEY="sk-..."
    uv run examples/openai_responses_website_search.py

Then open http://127.0.0.1:7777 and ask a research question. Try:

- "What is the tallest mountain in South America?"
- "Who wrote the book Dune?"
- "When did the Hubble Space Telescope launch?"

Every query triggers a Wikipedia search — that's what
``tool_choice="required"`` buys you. Each answer ends with a
collapsible "Sources" section.

Customizing the Domain
----------------------
Change ``allowed_domains`` in the tool config. Multiple domains are
supported — e.g. restrict to your product docs plus your changelog::

    "filters": {"allowed_domains": ["docs.example.com", "example.com/blog"]}

Friction Log (feeds the transparent-endpoint spec)
--------------------------------------------------
1. Annotations are a *structured* field on the response, but the
   engine's streaming finalization keeps only accumulated text.
   Preserving structure cleanly requires either a finalization seam
   or an extended ``extract_stream_delta`` contract.
2. Non-text events (tool call status, refusals, reasoning summaries)
   can only ride the text-delta channel today. Proper status
   surfacing needs a new return type or a separate engine pathway.
3. Domain restriction works on ``web_search`` but not on
   ``web_search_preview`` — the SDK keeps both. Provider-specific
   drift we should not normalize away.
"""

import chatnificent as chat


class OpenAIResponses(chat.llm.OpenAI):
    def generate_response(self, messages, **kwargs):
        return self.client.responses.create(
            model=self.model, input=messages, **{**self.default_params, **kwargs}
        )

    def extract_stream_delta(self, chunk):
        if chunk.type == "response.output_text.delta":
            return chunk.delta
        if chunk.type == "response.output_item.done":
            return self._format_citations(chunk.item)
        return None

    def _format_citations(self, item):
        if getattr(item, "type", None) != "message":
            return None
        urls, seen = [], set()
        for part in getattr(item, "content", []) or []:
            for annotation in getattr(part, "annotations", []) or []:
                if getattr(annotation, "type", None) != "url_citation":
                    continue
                url = annotation.url
                if url in seen:
                    continue
                seen.add(url)
                urls.append((annotation.title or url, url))
        if not urls:
            return None
        body = "\n".join(f"- [{title}]({url})" for title, url in urls)
        return (
            f"\n\n<details><summary>Sources ({len(urls)})</summary>\n\n"
            f"{body}\n\n</details>"
        )


app = chat.Chatnificent(
    llm=OpenAIResponses(
        tools=[
            {
                "type": "web_search",
                "filters": {"allowed_domains": ["en.wikipedia.org"]},
            }
        ],
        tool_choice="required",
    )
)


if __name__ == "__main__":
    app.run()
