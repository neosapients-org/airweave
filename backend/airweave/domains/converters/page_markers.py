"""Page markers in converted text (contract airweave-neo/1, sections 3.1 and 3.2).

Converters that know page or slide boundaries (PDF text layer, Mistral OCR of PDFs, PowerPoint)
put a marker line before each page's text. The marker is wrapped in a private-use code point,
so it cannot occur in real text, and it is lower case, so ``text_to_markdown`` leaves it alone.

The chunk processor removes every marker before chunking, so positions are counted without
them, and records where each page starts. Removing a marker leaves exactly the text the
converter produced before markers existed.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from typing import List, Optional, Tuple

PAGE_SENTINEL = "\ue000"
_MARKER_RE = re.compile(rf"{PAGE_SENTINEL}page (\d+){PAGE_SENTINEL}\n?")

# (offset in the marker-free text, page number), in text order
PageStarts = List[Tuple[int, int]]


def page_marker(page: int) -> str:
    """Return the marker line put before the text of ``page`` (1-based)."""
    return f"{PAGE_SENTINEL}page {page}{PAGE_SENTINEL}\n"


def strip_page_markers(text: str) -> Tuple[str, PageStarts]:
    """Remove every page marker; return the clean text and where each page starts in it."""
    if PAGE_SENTINEL not in text:
        return text, []
    parts: List[str] = []
    starts: PageStarts = []
    length, pos = 0, 0
    for m in _MARKER_RE.finditer(text):
        parts.append(text[pos : m.start()])
        length += m.start() - pos
        starts.append((length, int(m.group(1))))
        pos = m.end()
    parts.append(text[pos:])
    return "".join(parts), starts


def page_at(offset: int, starts: PageStarts) -> Optional[int]:
    """Return the page holding ``offset``: the last page start at or before it.

    Text before the first marker counts as the first page; without markers there are no pages.
    """
    if not starts:
        return None
    i = bisect_right(starts, offset, key=lambda s: s[0]) - 1
    return starts[max(i, 0)][1]
