"""Chunk layout: refused characters, positions and pages.

Contract airweave-neo/1, section 3. Positions are Unicode code points, end-exclusive, in the
document text: the entity's text with page markers removed and, with positions on, every
character Vespa refuses replaced by one space (so lengths, and with them every offset, hold).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict

from airweave.core.config import settings
from airweave.domains.converters.page_markers import PageStarts, page_at
from airweave.platform.entities._base import BaseEntity

# Code points Vespa 8 refuses in a string field (contract 3.2, C19): C0 controls other than
# tab, line feed and carriage return; noncharacters; lone surrogates.
_REFUSED_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufdd0-\ufdef\ufffe\uffff"
    + "".join(rf"\U{plane:04X}FFFE\U{plane:04X}FFFF" for plane in range(1, 17))
    + "]"
)


def replace_refused_characters(text: str) -> str:
    """Replace every code point Vespa refuses with one space; the length never changes."""
    return _REFUSED_RE.sub(" ", text)


@dataclass(frozen=True)
class ChunkLayout:
    """What the chunk processor adds to each text chunk, decided once per batch from settings.

    ``positions``: exact ranges and pages (sentence_pack mode only; semantic chunks keep
    upstream text and get none).
    """

    positions: bool

    @classmethod
    def from_settings(cls) -> "ChunkLayout":
        """Read the layout from the chunk settings."""
        return cls(positions=settings.CHUNKER_MODE == "sentence_pack")

    @property
    def active(self) -> bool:
        """Whether anything is added; with upstream settings nothing is."""
        return self.positions

    def apply(
        self, chunk_entity: BaseEntity, chunk: Dict[str, Any], page_starts: PageStarts
    ) -> None:
        """Set positions, pages and ``chunk_body_offset`` on one chunk entity."""
        if self.positions:
            start, end = chunk["start_index"], chunk["end_index"]
            chunk_entity.chunk_char_start = start
            chunk_entity.chunk_char_end = end
            chunk_entity.chunk_own_start = chunk["own_start"]
            chunk_entity.page_from = page_at(start, page_starts)
            chunk_entity.page_to = page_at(end - 1, page_starts)
            chunk_entity.chunk_body_offset = 0
