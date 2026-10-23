"""~300-token taxonomy analysis chunks, created INSIDE each Airweave retrieval chunk.

Units are sentences, paragraphs and table rows ('|'-flattened tables are split only at row boundaries). Over-long units
are split at token boundaries. A trailing group under `min_tokens_merge` tokens is merged into the previous group of
the same parent. Every taxonomy chunk therefore has exactly one parent Airweave chunk and character offsets inside it.

Taxonomy chunks are temporary analysis units. They never replace the Airweave retrieval chunks.
Taxonomy chunk id = "<airweave chunk_id>::tc<NNN>" (stable for a given chunk text and config).
"""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache

import tiktoken

UNIT_BREAK = re.compile(r"(?<=[.!?:;])\s+(?=[A-Z0-9“\"(\[•\-])|\n[ \t\xa0]*\n|\n(?=\S[^\n]*\|)")
ID_SEP = "::tc"


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


@lru_cache(maxsize=4)
def _encoding(name: str):
    return tiktoken.get_encoding(name)


def _units(text: str) -> list[tuple[int, int]]:
    """Sentence / paragraph / table-row units as (start, end) offsets within text."""
    pos, out = 0, []
    for m in UNIT_BREAK.finditer(text):
        if m.start() > pos:
            out.append((pos, m.start()))
        pos = m.end()
    if pos < len(text):
        out.append((pos, len(text)))
    return [(a, b) for a, b in out if text[a:b].strip()]


def split_text(text: str, cfg: dict) -> list[dict]:
    """Split one parent text into taxonomy chunk spans: [{start, end, text (whitespace-normalised), token_count}]."""
    t = cfg["taxonomy_chunking"]
    enc = _encoding(t["tokenizer"])
    tok = lambda s: len(enc.encode(s, allowed_special="all"))
    target, hard, minm = t["target_tokens"], t["hard_max_tokens"], t["min_tokens_merge"]
    us = []
    for a, b in _units(text):
        n = tok(text[a:b])
        if n <= hard:
            us.append((a, b, n)); continue
        ids = enc.encode(text[a:b], allowed_special="all")     # over-long unit: split at token boundaries
        # Window edges are the tokens' own character offsets: a token that starts inside a multi-byte character points
        # at that character's start, so a cut never lands mid-character (the decoded length of a slice would drift there).
        _, offs = enc.decode_with_offsets(ids)
        cur = a
        for k in range(0, len(ids), target):
            e = min(b, max(cur, a + offs[k + target])) if k + target < len(ids) else b
            if e > cur:
                us.append((cur, e, tok(text[cur:e]))); cur = e
    groups, cur, cur_t = [], [], 0
    for a, b, n in us:
        if cur and cur_t + n > target and (cur_t >= 0.5 * target or cur_t + n > hard):
            groups.append((cur, cur_t)); cur, cur_t = [], 0
        cur.append((a, b)); cur_t += n
    if cur:
        if groups and cur_t < minm and groups[-1][1] + cur_t <= hard:
            groups[-1] = (groups[-1][0] + cur, groups[-1][1] + cur_t)
        else:
            groups.append((cur, cur_t))
    out = []
    for g, _ in groups:
        a, b = g[0][0], g[-1][1]
        norm = " ".join(text[a:b].split())
        if norm:
            out.append({"start": a, "end": b, "text": norm, "token_count": tok(norm)})
    return out


def taxonomy_chunks(airweave_chunks, cfg: dict) -> list[dict]:
    """airweave_chunks: list of AirweaveChunk. Returns taxonomy chunk records with lineage to the parent chunk."""
    out = []
    for p in airweave_chunks:
        for j, s in enumerate(split_text(p.text, cfg)):
            out.append({"taxonomy_chunk_id": f"{p.chunk_id}{ID_SEP}{j:03d}", "chunk_id": p.chunk_id, "document_id": p.document_id,
                        "child_index": j, "char_start": s["start"], "char_end": s["end"], "token_count": s["token_count"],
                        "text": s["text"], "text_sha256": sha(s["text"])})
    return out
