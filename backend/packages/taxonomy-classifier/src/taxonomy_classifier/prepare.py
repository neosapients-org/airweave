"""Shared preparation: Airweave chunks -> taxonomy children -> embeddings + content-form tags + metadata frame."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .chunking import content_form
from .chunking.taxonomy_chunker import taxonomy_chunks
from .config import field_map


def children_frame(chunks, tc: list[dict], cfg: dict) -> pd.DataFrame:
    """One row per taxonomy child with the columns the algorithm uses:
    taxonomy_chunk_id, chunk_id, document_id, group, segment, doc_type, text."""
    f = field_map(cfg)
    by = {c.chunk_id: c for c in chunks}
    rows = []
    for t in tc:
        md = by[t["chunk_id"]].metadata
        rows.append({"taxonomy_chunk_id": t["taxonomy_chunk_id"], "chunk_id": t["chunk_id"], "document_id": t["document_id"],
                     "group": str(md.get(f["group"]) or t["document_id"]), "segment": str(md.get(f["segment"]) or "n/a"),
                     "doc_type": str(md.get(f["doc_type"]) or "n/a"), "text": t["text"]})
    return pd.DataFrame(rows, columns=["taxonomy_chunk_id", "chunk_id", "document_id", "group", "segment", "doc_type", "text"])


def prepare(chunks, cfg: dict, embedder, before_embedding=None) -> tuple[list[dict], pd.DataFrame, np.ndarray, pd.DataFrame]:
    """before_embedding(meta), if given, runs once the children exist and before they are sent to the (paid) embedder:
    a check that fails there costs nothing."""
    tc = taxonomy_chunks(chunks, cfg)
    meta = children_frame(chunks, tc, cfg)
    if before_embedding is not None:
        before_embedding(meta)
    X = embedder.embed(list(meta.text))
    tags = content_form.tag(list(meta.text), cfg)
    for t, (form, q) in zip(tc, zip(tags.content_form, tags.admin_quality)):
        t["content_form"], t["admin_quality"] = form, q
    return tc, meta, X, tags
