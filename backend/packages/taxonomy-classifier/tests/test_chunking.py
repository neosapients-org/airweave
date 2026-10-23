import tiktoken

from taxonomy_classifier.chunking.content_form import tag
from taxonomy_classifier.chunking.taxonomy_chunker import split_text, taxonomy_chunks
from taxonomy_classifier.schemas import AirweaveChunk

ENC = tiktoken.get_encoding("cl100k_base")


def test_children_respect_token_limits_and_parent(cfg):
    text = " ".join(f"Sentence number {i} talks about deferred tax assets and the valuation allowance." for i in range(200))
    chunk = AirweaveChunk("AW_001", "DOC_1", text)
    tc = taxonomy_chunks([chunk], cfg)
    assert len(tc) > 5
    for j, t in enumerate(tc):
        assert t["taxonomy_chunk_id"] == f"AW_001::tc{j:03d}"
        assert t["chunk_id"] == "AW_001" and t["document_id"] == "DOC_1"
        assert t["token_count"] <= cfg["taxonomy_chunking"]["hard_max_tokens"]
        assert " ".join(text[t["char_start"]:t["char_end"]].split()) == t["text"]     # lineage offsets index the parent text
    assert all(abs(t["token_count"] - 300) <= 150 for t in tc[:-1])


def test_table_rows_not_split():
    cfg = {"taxonomy_chunking": {"tokenizer": "cl100k_base", "target_tokens": 40, "hard_max_tokens": 60, "min_tokens_merge": 12}}
    rows = "\n".join(f"Row {i} | {i * 10} | {i * 20} | total" for i in range(30))
    for s in split_text(rows, cfg):
        assert s["text"].startswith("Row ") and s["text"].endswith("total")


def test_chunking_is_deterministic(cfg):
    c = [AirweaveChunk("A", "D", "Alpha beta. " * 400)]
    assert taxonomy_chunks(c, cfg) == taxonomy_chunks(c, cfg)


def test_short_chunk_yields_one_child_and_empty_yields_none(cfg):
    assert len(taxonomy_chunks([AirweaveChunk("A", "D", "A short chunk about debt covenants.")], cfg)) == 1
    assert taxonomy_chunks([AirweaveChunk("B", "D", "   \n ")], cfg) == []


def test_content_form_administrative(cfg):
    from tests.conftest import SIGNATURE
    t = tag([SIGNATURE, "The company discussed deferred tax assets and the valuation allowance in detail. " * 3, "12 | 13"], cfg)
    assert list(t.admin_quality) == ["ADMINISTRATIVE", "SUBSTANTIVE", "ADMINISTRATIVE"]
    assert t.content_form[0] == "SIGNATURE"
