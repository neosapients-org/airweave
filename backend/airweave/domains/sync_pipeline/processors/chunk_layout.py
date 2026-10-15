"""Chunk layout: positions on each text chunk.

Contract airweave-neo/1, section 3. Positions are Unicode code points, end-exclusive, in the
document text (the entity's text as the text builder produced it).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from airweave.core.config import settings
from airweave.platform.entities._base import BaseEntity


@dataclass(frozen=True)
class ChunkLayout:
    """What the chunk processor adds to each text chunk, decided once per batch from settings.

    ``positions``: exact ranges (sentence_pack mode only; semantic chunks keep upstream text
    and get none).
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

    def apply(self, chunk_entity: BaseEntity, chunk: Dict[str, Any]) -> None:
        """Set positions and ``chunk_body_offset`` on one chunk entity."""
        if self.positions:
            chunk_entity.chunk_char_start = chunk["start_index"]
            chunk_entity.chunk_char_end = chunk["end_index"]
            chunk_entity.chunk_own_start = chunk["own_start"]
            chunk_entity.chunk_body_offset = 0
