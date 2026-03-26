# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "chatnificent[gemini]",
# ]
# ///
"""
Web Search — Display-Only Gemini Google Search Example
======================================================

This example mirrors ``usage_display.py`` and teaches one pattern only:

1. ask Gemini with Google Search grounding enabled
2. let Chatnificent save the raw API response as usual
3. in ``render_messages()``, read ``raw_api_responses.jsonl``
4. parse Gemini grounding metadata into search-result-style cards
5. append those results beneath the matching assistant message

``messages.json`` and the raw response files are never modified — we only
enrich the visible transcript at render time.

How Grounding Works in Chatnificent
------------------------------------
The Gemini SDK enables Google Search via
``GenerateContentConfig(tools=[Tool(google_search=GoogleSearch())])``.
In Chatnificent, extra ``chat.llm.Gemini(...)`` kwargs become default
generation config, so ``tools=[GROUNDING_TOOL]`` is forwarded automatically.

The raw API response's final streaming chunk contains
``grounding_metadata`` with two key arrays:

- ``grounding_chunks`` — each entry has ``web.title`` (the source domain)
  and ``web.uri`` (a redirect URL to the source page).
- ``grounding_supports`` — text segments from the answer linked to chunk
  indices, serving as snippets for each source.

Prerequisites
-------------
::

    export GOOGLE_API_KEY="AI..."

Running
-------
::

    uv run --script examples/web_search.py

What to Explore Next
--------------------
- Combine with ``usage_display.py`` to show both sources and token usage
- Add ``web_search_queries`` from grounding metadata as search pills
- Use ``search_entry_point.rendered_content`` for the native Google widget
"""

from typing import Any, Dict, List, Optional

import chatnificent as chat
from google.genai import types as gemini_types

GROUNDING_TOOL = gemini_types.Tool(google_search=gemini_types.GoogleSearch())


class WebSearchLayout(chat.layout.DefaultLayout):
    """Append Gemini search sources beneath assistant messages."""

    @staticmethod
    def _grounding_metadata(raw_response: Any) -> Dict[str, Any]:
        """Return grounding metadata from the last streaming chunk that has it."""
        chunks = raw_response if isinstance(raw_response, list) else [raw_response]
        for chunk in reversed(chunks):
            if not isinstance(chunk, dict):
                continue
            candidates = chunk.get("candidates") or []
            if not candidates:
                continue
            metadata = (
                candidates[0].get("grounding_metadata")
                or candidates[0].get("groundingMetadata")
                or {}
            )
            if metadata:
                return metadata
        return {}

    @staticmethod
    def _source_block(raw_response: Any) -> Optional[str]:
        """Format grounding metadata as a collapsible Markdown source block."""
        metadata = WebSearchLayout._grounding_metadata(raw_response)
        if not metadata:
            return None

        chunks = (
            metadata.get("grounding_chunks") or metadata.get("groundingChunks") or []
        )
        supports = (
            metadata.get("grounding_supports")
            or metadata.get("groundingSupports")
            or []
        )

        snippets: Dict[int, str] = {}
        for support in supports:
            if not isinstance(support, dict):
                continue
            segment = support.get("segment") or {}
            text = segment.get("text")
            indices = (
                support.get("grounding_chunk_indices")
                or support.get("groundingChunkIndices")
                or []
            )
            if not text or not indices:
                continue
            snippet = str(text).strip()
            if not snippet:
                continue
            for index in indices:
                snippets.setdefault(index, snippet)

        cards: List[str] = []
        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                continue
            web = chunk.get("web") or {}
            title = web.get("title")
            url = web.get("uri")
            if not title or not url:
                continue

            lines = [f"**[{title}]({url})**"]
            snippet = snippets.get(index)
            if snippet:
                lines.append(snippet)
            cards.append("\n\n".join(lines))

        if not cards:
            return None

        return (
            "---\n\n"
            "<details>\n"
            f"<summary>🔍 Sources ({len(cards)})</summary>\n"
            "<br>\n\n" + "\n\n---\n\n".join(cards) + "\n\n</details>"
        )

    def render_messages(self, messages, **kwargs):
        rendered = super().render_messages(messages, **kwargs)
        user_id = kwargs["user_id"]
        conversation = kwargs["conversation"]
        raw_responses = self.app.store.load_raw_api_responses(user_id, conversation.id)

        source_blocks = [self._source_block(r) for r in raw_responses]

        assistant_index = 0
        for message in rendered:
            if message.get("role") != "assistant":
                continue
            if assistant_index >= len(source_blocks):
                break
            block = source_blocks[assistant_index]
            if block:
                message["content"] = f"{message['content']}\n\n{block}"
            assistant_index += 1

        return rendered


app = chat.Chatnificent(
    llm=chat.llm.Gemini(
        model="gemini-3-flash-preview",
        tools=[GROUNDING_TOOL],
    ),
    layout=WebSearchLayout(),
    store=chat.store.File(base_dir="web_search_convos"),
)

if __name__ == "__main__":
    app.run()
