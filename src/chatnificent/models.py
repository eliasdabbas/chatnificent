"""
Core data models for the Chatnificent framework.

Messages are plain dicts — each LLM provider owns its own format.
Conversation is a lightweight dataclass grouping an id with a message list.
Role constants prevent typos across pillars.
"""

import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

USER_ROLE = "user"
ASSISTANT_ROLE = "assistant"
SYSTEM_ROLE = "system"
TOOL_ROLE = "tool"
MODEL_ROLE = "model"


@dataclass
class Conversation:
    """A chat conversation: an id and a list of message dicts."""

    id: str
    messages: List[Dict[str, Any]] = field(default_factory=list)

    def copy(self, deep: bool = False) -> "Conversation":
        """Return a copy of this conversation."""
        if deep:
            return Conversation(id=self.id, messages=copy.deepcopy(self.messages))
        return Conversation(id=self.id, messages=list(self.messages))


@dataclass(frozen=True)
class Artifact:
    """A binary payload destined for the Store and an HTML embed in a message.

    The engine receives ``Artifact`` instances (returned by LLM adapters or
    user code), persists ``data`` via ``Store.save_file``, and rewrites the
    message body with an HTML snippet referencing the absolute URL
    ``/<user_id>/<convo_id>/<folder>/<filename>``.

    Parameters
    ----------
    data : bytes
        Raw bytes to persist.
    ext : str
        File extension including the leading dot (e.g. ``".mp3"``). Matches
        ``pathlib.Path.suffix`` / ``os.path.splitext`` so ``f"{n}{ext}"``
        composes a valid filename.
    folder : str
        Subdirectory under the conversation root. Empty string places the
        file at the conversation root.
    filename : str, optional
        Override for the auto-counter. When ``None``, the engine picks
        ``<folder>/<N><ext>`` with ``N`` incrementing per folder. When set,
        the file is saved at exactly that leaf name (last-write-wins on
        collisions — the "pinned" semantics used for streaming overwrite).
    html : str, optional
        Full embed snippet with ``{url}`` and ``{filename}`` placeholders.
        When ``None`` the engine picks a default wrapper by MIME family
        (see ``ARTIFACT_WRAPPERS`` on the Engine).
    """

    data: bytes
    ext: str
    folder: str = ""
    filename: Optional[str] = None
    html: Optional[str] = None
