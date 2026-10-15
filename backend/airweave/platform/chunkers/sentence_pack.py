"""Sentence-packing chunker with exact positions (contract airweave-neo/1, section 3).

The chunker POC 06 measured, as the shared fixture's reference implementation specifies it
(``fixtures/make_chunk_fixture.py``):

- sentences are cut after ``. ! ?`` plus whitespace, or at newlines; pieces shorter than
  ``MIN_SENTENCE_CHARS`` are merged forward (POC 06's ``corpus.sentences``);
- sentences are packed in order while the chunk body, **including** its overlap, stays within
  the target (cl100k_base tokens; the header line is not counted);
- the overlap is whole sentences from the end of the previous chunk, at most the overlap
  setting, and it shrinks while overlap plus the next new sentence would pass the target, so
  every chunk adds new text;
- a sentence longer than the target is split into target-size token windows. Window edges are
  exact code-point positions (tiktoken's token offsets); the reference derives them from the
  decoded window length, which drifts when an edge falls inside a multi-byte character.

Every chunk is a ``[start, end)`` range of the text in Unicode code points; ``own_start`` is the
previous chunk's end (the chunk's own text never overlaps another chunk's).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from tiktoken import Encoding

from airweave.domains.sync_pipeline.async_helpers import run_in_thread_pool
from airweave.platform.chunkers._base import BaseChunker
from airweave.platform.tokenizers import TikTokenTokenizer, get_tokenizer

_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")
MIN_SENTENCE_CHARS = 24

# (start, end, tokens): a sentence, or one window of a sentence longer than the target
Unit = Tuple[int, int, int]


def sentences(text: str) -> List[Tuple[int, int]]:
    """Return ``(start, end)`` spans that cover the text, one per sentence."""
    spans, start = [], 0
    for m in _BOUNDARY.finditer(text):
        if m.end() - start >= MIN_SENTENCE_CHARS:
            spans.append((start, m.end()))
            start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def units(text: str, target: int, enc: Encoding) -> List[Unit]:
    """Sentences with token counts; a sentence longer than the target becomes token windows."""
    spans = sentences(text)
    token_lists = enc.encode_ordinary_batch([text[s:e] for s, e in spans])
    out: List[Unit] = []
    for (s, e), tokens in zip(spans, token_lists, strict=True):
        if len(tokens) <= target:
            out.append((s, e, len(tokens)))
        else:
            out.extend(_windows(text, s, e, tokens, target, enc))
    return out


def _windows(
    text: str, start: int, end: int, tokens: List[int], target: int, enc: Encoding
) -> List[Unit]:
    """Split one sentence into windows of ``target`` tokens at exact code-point positions.

    ``edges[k]`` is where token ``k`` starts in the text; a token that starts inside a
    multi-byte character points at that character's start, so the character goes whole into
    one window. A window's count is its number of tokens, as in the reference.
    """
    _, offsets = enc.decode_with_offsets(tokens)
    edges = [start + o for o in offsets] + [end]
    out: List[Unit] = []
    i, n = 0, len(tokens)
    while i < n:
        j = _window_end(text, edges, i, min(i + target, n), target, enc)
        out.append((edges[i], edges[j], j - i))
        i = j
    return out


def _window_end(text: str, edges: List[int], i: int, j: int, target: int, enc: Encoding) -> int:
    """Return the largest token index up to ``j`` that ends a non-empty window within target.

    A window cut inside a word can re-encode to more tokens than it holds; it is then pulled
    back one token at a time. Only when no shorter window works (a character spread over more
    tokens than the target) does the window grow to the first non-empty end instead.
    """
    first = j
    while j > i:
        a, b = edges[i], edges[j]
        if b > a and len(enc.encode_ordinary(text[a:b])) <= target:
            return j
        j -= 1
    j = first
    while j < len(edges) - 1 and edges[j] <= edges[i]:
        j += 1
    return j


def pack(us: List[Unit], target: int, overlap: int) -> List[Tuple[int, int]]:
    """Pack units into ``(start, end)`` chunk ranges; the overlap counts inside the target."""
    chunks: List[Tuple[int, int]] = []
    i, n = 0, len(us)
    while i < n:
        j, total = i, us[i][2]
        while j + 1 < n and total + us[j + 1][2] <= target:
            j += 1
            total += us[j][2]
        chunks.append((us[i][0], us[j][1]))
        if j == n - 1:
            break
        k, ov = j + 1, 0
        while k - 1 > i and ov + us[k - 1][2] <= overlap:
            k -= 1
            ov += us[k][2]
        # The next chunk must add new text: shrink the overlap while overlap plus the next new
        # sentence would pass the target (not in POC 06, whose packer can repeat a chunk).
        while k <= j and ov + us[j + 1][2] > target:
            ov -= us[k][2]
            k += 1
        i = k
    return chunks


class SentencePackChunker(BaseChunker):
    """Chunks text by packing whole sentences up to a token target, with exact positions.

    Used when ``CHUNKER_MODE=sentence_pack``. Each chunk dict carries ``text`` (the exact slice
    ``text[start_index:end_index]``), ``start_index``, ``end_index``, ``own_start`` and
    ``token_count`` (cl100k_base tokens of the text).
    """

    TOKENIZER = "cl100k_base"

    def __init__(self, target_tokens: int, overlap_tokens: int) -> None:
        """Initialize with the token target (overlap included) and the overlap budget."""
        if not 0 <= overlap_tokens < target_tokens:
            raise ValueError("overlap_tokens must be >= 0 and below target_tokens")
        self._target = target_tokens
        self._overlap = overlap_tokens
        tokenizer = get_tokenizer(self.TOKENIZER)
        if not isinstance(tokenizer, TikTokenTokenizer):
            raise ValueError(f"{self.TOKENIZER} must be a tiktoken encoding")
        self._enc: Encoding = tokenizer.encoding

    async def chunk_batch(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        """Chunk a batch of texts (CPU-bound, so in the shared thread pool)."""
        return await run_in_thread_pool(self.chunk_texts, texts)

    def chunk_texts(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        """Chunk a batch of texts synchronously."""
        return [self.chunk_text(text) for text in texts]

    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """Chunk one text into contiguous, overlapping ranges that cover it."""
        if not text:
            return []
        ranges = pack(units(text, self._target, self._enc), self._target, self._overlap)
        bodies = [text[s:e] for s, e in ranges]
        counts = [len(t) for t in self._enc.encode_ordinary_batch(bodies)]
        chunks: List[Dict[str, Any]] = []
        prev_end = None
        for (s, e), body, count in zip(ranges, bodies, counts, strict=True):
            chunks.append(
                {
                    "text": body,
                    "start_index": s,
                    "end_index": e,
                    "own_start": s if prev_end is None else prev_end,
                    "token_count": count,
                }
            )
            prev_end = e
        return chunks
