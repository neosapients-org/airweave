"""Input, cost and integrity guards: blank ids, multi-byte text, conservative spend bounds, the response cache,
artifact manifests, embedding responses, the wave's RRF constant and acceptance-model settings."""

import json
import re
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp
import tiktoken

from taxonomy_classifier.assignment.learned import LearnedAcceptanceModel
from taxonomy_classifier.assignment.rrf import ranks
from taxonomy_classifier.chunking.content_form import content_words, tag
from taxonomy_classifier.chunking.taxonomy_chunker import split_text
from taxonomy_classifier.embeddings.embedder import Embedder
from taxonomy_classifier.evolution.wave import _nearest
from taxonomy_classifier.llm.client import LLMClient
from taxonomy_classifier.llm.costs import CostLedger
from taxonomy_classifier.schemas import AirweaveChunk, InputError
from taxonomy_classifier.taxonomy import artifacts

from .conftest import FakeLLM

CHUNKING = {"taxonomy_chunking": {"tokenizer": "cl100k_base", "target_tokens": 70, "hard_max_tokens": 110,
                                  "min_tokens_merge": 25, "overlap_tokens": 0}}
CJK = "経済政策の影響について検討する必要があります" * 30          # no spaces and no sentence end: one over-long unit
EMOJI = "🙂🚀✨🌍🎉" * 120


# --------------------------------------------------------------------------- blank identifiers

@pytest.mark.parametrize("field", ["chunk_id", "document_id"])
def test_a_blank_identifier_is_refused(field):
    with pytest.raises(InputError, match=field):
        AirweaveChunk.from_dict({"chunk_id": "c1", "document_id": "d1", "text": "x", field: ""})


def test_an_empty_text_is_still_accepted():
    assert AirweaveChunk.from_dict({"chunk_id": "c1", "document_id": "d1", "text": ""}).text == ""


# --------------------------------------------------------------------------- multi-byte text

@pytest.mark.parametrize("text", [CJK, EMOJI], ids=["japanese", "emoji"])
def test_an_over_long_unit_of_multi_byte_text_is_split_without_losing_or_repeating_text(text):
    spans = split_text(text, CHUNKING)
    assert spans[0]["start"] == 0 and spans[-1]["end"] == len(text)
    assert all(a["end"] == b["start"] for a, b in zip(spans, spans[1:]))      # contiguous: nothing lost, nothing twice
    assert "".join(text[s["start"]:s["end"]] for s in spans) == text
    assert all(s["token_count"] <= CHUNKING["taxonomy_chunking"]["hard_max_tokens"] for s in spans)


@pytest.mark.parametrize("text", ["Payment is due within thirty days of the invoice date.",
                                  "Short note.", "a b c dd eee ffff 12345 under_score x9yz Page 3 of 9", ""])
def test_ascii_word_count_is_unchanged(text):
    assert content_words(text) == len(re.findall(r"[A-Za-z]{3,}", text))


def test_substantive_non_latin_text_is_not_administrative():
    arabic = "تلتزم الشركة بسداد الفواتير المستحقة خلال ثلاثين يوما من تاريخ الاستلام وفقا للشروط المتفق عليها في العقد المبرم بين الطرفين"
    chinese = "公司应在收到发票后三十天内支付所有未付款项，并按照合同约定的条款处理任何争议事项和相关费用"
    out = tag([arabic, chinese, "Signed."], {"content_form": {"patterns": {}, "administrative_bypass": True}})
    assert list(out.admin_quality) == ["SUBSTANTIVE", "SUBSTANTIVE", "ADMINISTRATIVE"]


# --------------------------------------------------------------------------- conservative spend bounds

def test_the_byte_bound_is_never_below_the_token_count():
    for enc in ("cl100k_base", "o200k_base"):
        e = tiktoken.get_encoding(enc)
        for text in (CJK, EMOJI, "plain english text " * 50):
            assert len(e.encode(text)) <= len(text.encode())


def test_a_reservation_holds_the_headroom_until_it_is_released(tmp_path):
    ledger = CostLedger(1.0, tmp_path / "cost_ledger.json")
    assert ledger.reserve(0.7)
    assert not ledger.reserve(0.4) and not ledger.can_spend(0.4)   # a concurrent call cannot use the same headroom
    ledger.add("llm", 0.2)
    ledger.release(0.7)
    assert ledger.reserve(0.8) and not ledger.reserve(0.01)


def _llm(tmp_path, cap=5.0, transport=None):
    cfg = {"llm": {"model": "m", "pricing": {"input": 0.75, "cached_input": 0.075, "output": 4.5},
                   "cache_dir": str(tmp_path / "llm_cache"), "corpus_description": "test",
                   "retry": {"max_attempts": 1, "backoff_s": 0, "retry_on_http": []}}}
    ledger = CostLedger(cap, tmp_path / "cost_ledger.json")
    return LLMClient(cfg, tmp_path / "run", ledger, "s", "cluster_semantic_review_v2", "low", 1500, transport or FakeLLM()), ledger


def test_the_llm_bound_counts_bytes_and_the_schema(tmp_path):
    cli, _ = _llm(tmp_path)
    user = CJK
    worst = cli._worst_case_usd(user)
    tokens_in = len(tiktoken.get_encoding("o200k_base").encode(cli.system + user + cli._schema_json))
    assert worst >= (tokens_in * 0.75 + 1500 * 4.5) / 1e6


def test_a_call_whose_bound_does_not_fit_the_cap_is_never_sent(tmp_path):
    sent = []
    cli, ledger = _llm(tmp_path, cap=0.001, transport=lambda body: sent.append(body))
    rec = cli.call(CJK, "c1")
    assert rec["status"] == "BUDGET_EXCEEDED" and sent == [] and ledger.data["total_usd"] == 0


def test_an_embedding_batch_is_reserved_by_its_bytes(tmp_path):
    sent = []
    cfg = {"embedding": {"model": "e", "dimensions": 4, "batch_size": 8, "cache_path": str(tmp_path / "emb.sqlite"),
                         "price_per_m_tokens_usd": 1.0}}
    ledger = CostLedger(len(CJK.encode()) / 1e6 - 1e-9, tmp_path / "cost_ledger.json")   # just under the batch's bound
    emb = Embedder(cfg, ledger, lambda batch: sent.append(batch))
    with pytest.raises(RuntimeError, match="cost cap"):
        emb.embed([CJK])
    assert sent == []


# --------------------------------------------------------------------------- the response cache

REVIEW_INPUT = "Most distinctive terms: invoice terms, payment"   # what the fake reviewer reads


def _cache_files(tmp_path):
    return sorted(p.name for p in (tmp_path / "llm_cache").iterdir())


def test_a_changed_schema_is_a_cache_miss(tmp_path):
    llm = FakeLLM()
    cli, _ = _llm(tmp_path, transport=llm)
    cli.call(REVIEW_INPUT, "c1")
    cli._schema_json += " "                                    # the same prompt version with an edited schema
    cli.call(REVIEW_INPUT, "c1")
    assert len(llm.calls) == 2 and len(_cache_files(tmp_path)) == 2


def test_a_torn_cache_entry_is_a_miss_and_is_rewritten_whole(tmp_path):
    llm = FakeLLM()
    cli, _ = _llm(tmp_path, transport=llm)
    first = cli.call(REVIEW_INPUT, "c1")
    entry = tmp_path / "llm_cache" / f"{first['input_hash']}.json"
    entry.write_text(entry.read_text()[:40])                  # a concurrent writer cut it short
    again = cli.call(REVIEW_INPUT, "c1")
    assert again["status"] == "OK" and not again["cache_hit"] and len(llm.calls) == 2
    assert json.loads(entry.read_text())["status"] == "OK"
    assert not [n for n in _cache_files(tmp_path) if n.endswith(".tmp")]


# --------------------------------------------------------------------------- artifacts and embedding responses

def test_an_optional_artifact_the_manifest_lists_must_exist(tmp_path):
    import hashlib

    d = tmp_path / "tax"; d.mkdir()
    listed = {}
    for name in artifacts.REQUIRED[1:]:                       # the required files are there and match
        (d / name).write_bytes(name.encode())
        listed[name] = hashlib.sha256(name.encode()).hexdigest()
    listed["category_centroids.npy"] = "0" * 64              # an optional file the manifest lists, missing
    (d / "final_taxonomy.json").write_text(json.dumps({"artifacts": listed}))
    with pytest.raises(FileNotFoundError, match="category_centroids.npy"):
        artifacts.load(d)


@pytest.mark.parametrize("vecs", [[np.ones(4)], [np.ones(4), np.ones(3)], [np.ones(4), np.full(4, np.nan)]])
def test_a_malformed_embedding_response_is_refused_before_anything_is_cached(tmp_path, vecs):
    cfg = {"embedding": {"model": "e", "dimensions": 4, "batch_size": 8, "cache_path": str(tmp_path / "emb.sqlite")}}
    emb = Embedder(cfg, None, lambda batch: (vecs, 4))
    with pytest.raises(RuntimeError, match="embedding"):
        emb.embed(["one text", "another text"])
    assert emb.cache.get("one text") is None and emb.cache.get("another text") is None


# --------------------------------------------------------------------------- the wave's RRF constant

def test_the_wave_ranks_candidates_with_the_taxonomys_rrf_constant():
    n = 10
    dense = np.linspace(1, 0, n); dense[[0, 3]] = dense[[3, 0]]                 # X = id 0 ranks 4th dense
    sparse = np.linspace(1, 0, n); sparse[[0, 9]] = sparse[[9, 0]]              # X ranks last sparse
    dense[[0, 1]], sparse[[3, 1]] = dense[[1, 0]], sparse[[1, 3]]
    rep = SimpleNamespace(centroids=np.diag(dense), sparse=sp.csr_matrix(np.diag(sparse)), ids=[f"C{i}" for i in range(n)])
    order = {}
    for k in (1, 60):
        work = SimpleNamespace(rep=rep, ids=rep.ids, final={"assignment": {"rrf_k": k}})
        got, _ = _nearest(work, np.ones(n), np.ones(n), n)
        r = 1 / (k + ranks(dense[None])[0]) + 1 / (k + ranks(sparse[None])[0])
        assert got == [rep.ids[j] for j in np.argsort(-r)]
        order[k] = got
    assert order[1] != order[60]                                                  # the constant does change the ranking


# --------------------------------------------------------------------------- acceptance-model settings

@pytest.mark.parametrize("change", [{"threshold": 1.5}, {"threshold": -0.1}, {"threshold": float("nan")},
                                    {"threshold": "high"}, {"max_labels_per_child": 0}, {"max_labels_per_child": 1.5}])
def test_an_acceptance_model_with_impossible_settings_is_refused_before_it_is_loaded(tmp_path, change):
    from taxonomy_classifier.assignment.learned import FEATURE_NAMES, FEATURE_SCHEMA_VERSION

    man = {"model_id": "m", "feature_schema_version": FEATURE_SCHEMA_VERSION, "feature_names": list(FEATURE_NAMES),
           "model_file": "absent.pkl", "model_sha256": "0" * 64, "threshold": 0.5, **change}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(man))
    with pytest.raises(ValueError):
        LearnedAcceptanceModel.load(path)
