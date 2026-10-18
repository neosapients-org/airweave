"""Chunk layout: refused characters, header line, positions and pages.

Contract airweave-neo/1, section 3. Positions are Unicode code points, end-exclusive, in the
document text: the entity's text with page markers removed and, with positions on, every
character Vespa refuses replaced by one space (so lengths, and with them every offset, hold).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from airweave.core.config import settings
from airweave.domains.converters.page_markers import PageStarts, page_at
from airweave.platform.entities._base import BaseEntity, Breadcrumb

# Code points Vespa 8 refuses in a string field (contract 3.2, C19): C0 controls other than
# tab, line feed and carriage return; noncharacters; lone surrogates.
_REFUSED_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufdd0-\ufdef\ufffe\uffff"
    + "".join(rf"\U{plane:04X}FFFE\U{plane:04X}FFFF" for plane in range(1, 17))
    + "]"
)


# Every line boundary str.splitlines() knows, so the header is one line by any definition.
_LINE_BREAKS = str.maketrans({c: " " for c in "\n\r\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029"})


def replace_refused_characters(text: str) -> str:
    """Replace every code point Vespa refuses with one space; the length never changes."""
    return _REFUSED_RE.sub(" ", text)


def build_header(
    name: Optional[str],
    source_name: Optional[str],
    breadcrumbs: Optional[List[Breadcrumb]],
    page_from: Optional[int] = None,
    page_to: Optional[int] = None,
) -> str:
    """Build the one-line header: ``{name} · {source} · {path}`` plus the pages when known.

    The path (breadcrumb names joined by " / ") is left out when there is none. Line breaks
    in the name or path become spaces and refused characters are replaced, one for one.
    """
    parts = [name or "", source_name or ""]
    path = " / ".join(bc.name for bc in breadcrumbs or [])
    if path:
        parts.append(path)
    header = " · ".join(p.translate(_LINE_BREAKS) for p in parts)
    if page_from is not None:
        if page_to is None or page_to == page_from:
            header += f" · page {page_from}"
        else:
            header += f" · pages {page_from}–{page_to}"
    return replace_refused_characters(header)


_tokenizer: Any = None


def _header_tokenizer() -> Any:
    """The tokenizer the header is measured with (the embedder's family), loaded once."""
    global _tokenizer
    if _tokenizer is None:
        from airweave.platform.tokenizers import get_tokenizer

        _tokenizer = get_tokenizer("cl100k_base")
    return _tokenizer


def fit_header(header: str, max_tokens: int, count: Any = None) -> str:
    """Cut a header to ``max_tokens`` tokens, ending in an ellipsis; a short one is unchanged.

    The header is embedded with the body, and the body is chunked to its own token target before
    the header is known, so the header must have a budget of its own: a long file name or path
    would otherwise push a full chunk past the embedder's limit and fail the whole entity. A
    character limit cannot bound tokens (CJK text costs a token or more per character).

    Args:
        header: The one-line header.
        max_tokens: The most tokens it may take.
        count: Counts tokens in a text; the embedder-family tokenizer by default.
    """
    count = count or _header_tokenizer().count_tokens
    if count(header) <= max_tokens:
        return header
    low, high = 0, len(header)
    while low < high:  # the longest prefix that still fits with the ellipsis
        mid = (low + high + 1) // 2
        if count(header[:mid] + "…") <= max_tokens:
            low = mid
        else:
            high = mid - 1
    return header[:low].rstrip() + "…"


@dataclass(frozen=True)
class ChunkLayout:
    """What the chunk processor adds to each text chunk, decided once per batch from settings.

    ``positions``: exact ranges and pages (sentence_pack mode only; semantic chunks keep
    upstream text and get none). ``header``: the header line before the body.
    """

    positions: bool
    header: bool = False

    @classmethod
    def from_settings(cls) -> "ChunkLayout":
        """Read the layout from the chunk settings."""
        return cls(
            positions=settings.CHUNKER_MODE == "sentence_pack",
            header=settings.CHUNK_HEADER_ENABLED,
        )

    @property
    def active(self) -> bool:
        """Whether anything is added; with upstream settings nothing is."""
        return self.positions or self.header

    def apply(
        self,
        chunk_entity: BaseEntity,
        chunk: Dict[str, Any],
        page_starts: PageStarts,
        source_name: Optional[str] = None,
    ) -> None:
        """Set positions, pages, the header and ``chunk_body_offset`` on one chunk entity."""
        if self.positions:
            start, end = chunk["start_index"], chunk["end_index"]
            chunk_entity.chunk_char_start = start
            chunk_entity.chunk_char_end = end
            chunk_entity.chunk_own_start = chunk["own_start"]
            chunk_entity.page_from = page_at(start, page_starts)
            chunk_entity.page_to = page_at(end - 1, page_starts)
        if self.header:
            header = build_header(
                chunk_entity.name,
                source_name,
                chunk_entity.breadcrumbs,
                chunk_entity.page_from,
                chunk_entity.page_to,
            )
            header = fit_header(header, settings.CHUNK_HEADER_MAX_TOKENS)
            chunk_entity.textual_representation = f"{header}\n{chunk_entity.textual_representation}"
            chunk_entity.chunk_body_offset = len(header) + 1
        else:
            chunk_entity.chunk_body_offset = 0
