"""Rule-based content form / administrative tagging of taxonomy chunks (no LLM, no embeddings).

These tags are METADATA ONLY and never become semantic topics. The single effect on classification: chunks tagged
ADMINISTRATIVE (signature blocks, exhibit indexes, cover/form metadata, too little text) bypass semantic assignment.

The regular expressions are configuration (`content_form.patterns`), so the code carries no corpus-specific vocabulary.
The example config ships the patterns that were verified on the reference corpus (regulatory filings).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

ADMIN_FORMS = ("SIGNATURE", "EXHIBIT_INDEX", "FORM_OR_METADATA", "OTHER")
SENT = re.compile(r"[a-z0-9\)%][\.\?!](?:\s+|$)(?=[A-Z“\"(]|$)")
NUM = re.compile(r"^[\(\-–—$]*[\d,.]+%?\)?$")


def _patterns(cfg: dict) -> dict:
    p = cfg.get("content_form", {}).get("patterns", {})
    flags = re.I
    comp = lambda key, f=flags: re.compile(p[key], f) if p.get(key) else None
    return {"signature": comp("signature"), "exhibit": comp("exhibit", 0), "exhibit_row": comp("exhibit_row", 0),
            "cover": comp("cover"), "boilerplate": comp("boilerplate")}


def _hits(rx, text: pd.Series) -> pd.Series:
    return text.map(lambda t: len(rx.findall(t))) if rx is not None else pd.Series(0, index=text.index)


def tag(texts: list[str], cfg: dict) -> pd.DataFrame:
    """Returns one row per text: content_form, admin_quality (SUBSTANTIVE / BOILERPLATE / ADMINISTRATIVE), admin_reason."""
    rx = _patterns(cfg)
    text = pd.Series(texts, dtype=object).astype(str)
    f = pd.DataFrame(index=text.index)
    f["pipe_count"] = text.str.count(r"\|")
    toks = text.str.split()
    f["numeric_ratio"] = toks.map(lambda t: np.mean([bool(NUM.match(x)) for x in t]) if t else 0.0)
    f["alpha_tokens"] = text.map(lambda t: len(re.findall(r"[A-Za-z]{3,}", t)))
    f["sentence_count"] = text.map(lambda t: len(SENT.findall(t)))
    f["bullet_count"] = text.str.count("•")
    f["signature_hits"] = _hits(rx["signature"], text)
    f["exhibit_hits"] = _hits(rx["exhibit"], text)
    f["exhibit_rows"] = _hits(rx["exhibit_row"], text)
    f["cover_hits"] = _hits(rx["cover"], text)
    f["boilerplate_phrase_hits"] = _hits(rx["boilerplate"], text)
    form = np.full(len(f), "NARRATIVE", dtype=object)
    tbl = (f.pipe_count >= 12) & (f.sentence_count <= 2)
    mixed = (f.pipe_count >= 6) & ~tbl
    numeric_tbl = (f.pipe_count < 6) & (f.numeric_ratio >= 0.35) & (f.sentence_count <= 2)
    form[(f.bullet_count >= 4) & (f.pipe_count < 6)] = "LIST"
    form[mixed] = "MIXED_TABLE_AND_NARRATIVE"
    form[tbl | numeric_tbl] = "TABLE"
    form[(f.exhibit_hits >= 3) | ((f.exhibit_hits >= 1) & (f.exhibit_rows >= 3))] = "EXHIBIT_INDEX"
    form[f.cover_hits >= 3] = "FORM_OR_METADATA"
    form[f.signature_hits >= 3] = "SIGNATURE"
    form[(f.alpha_tokens < 15) & (f.pipe_count < 6)] = "OTHER"
    out = pd.DataFrame({"content_form": form})
    q = np.full(len(f), "SUBSTANTIVE", dtype=object)
    q[(f.boilerplate_phrase_hits >= 2).to_numpy()] = "BOILERPLATE"
    adm = np.isin(form, ADMIN_FORMS) | (f.alpha_tokens < 15).to_numpy()
    if not cfg.get("content_form", {}).get("administrative_bypass", True):
        adm[:] = False
    q[adm] = "ADMINISTRATIVE"
    out["admin_quality"] = q
    out["admin_reason"] = np.where(adm, np.where(f.alpha_tokens.to_numpy() < 15, "TOO_LITTLE_TEXT", form), "")
    return out
