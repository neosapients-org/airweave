# Airweave Taxonomy Classifier

Discovers a topic taxonomy from the chunks that Airweave already produces, freezes it, and tags every Airweave retrieval chunk with **semantic topics** and **broad families**. The tags are metadata: they can be written next to the chunk in Vespa and used for filtering, boosting, inspection or agent context. Airweave's retrieval is unchanged.

> Runtime decisions come only from Python code, the embeddings API, UMAP/HDBSCAN, BM25, RRF and (in the taxonomy build and novelty maintenance) logged LLM API calls. Nothing is decided interactively. No coding agent is part of the runtime.

> **Start here:** [docs/ENGINEER_HANDOFF.md](docs/ENGINEER_HANDOFF.md).
> - Entry point: `TaxonomyEngine` (`discover` / `load` / `classify` / `evolve`) with `LocalTaxonomyStore` ([docs/INTEGRATION.md](docs/INTEGRATION.md)).
> - Stable output contract: `topics[]`, `contract_version: "topics/1"`.
> - What is productionized vs research-only: [docs/PRODUCTION_STATUS.md](docs/PRODUCTION_STATUS.md).
> - Research summary: [docs/research/README.md](docs/research/README.md).

```python
from taxonomy_classifier import TaxonomyEngine, LocalTaxonomyStore
store = LocalTaxonomyStore("/var/lib/myapp/taxonomies")
engine = TaxonomyEngine("config/taxonomy.yaml", store=store)
engine.discover(build_chunks, taxonomy_id="my_corpus", domain="legal")       # any corpus; offline
result = TaxonomyEngine.load("my_corpus", "config/taxonomy.yaml", store).classify(new_chunks)
```

**Engine vs data.** The engine (`src/taxonomy_classifier/`) is domain-independent. A taxonomy is a versioned runtime artifact that `discover()` builds from any corpus and the store supplies at run time. The repo's `taxonomies/` folder only holds vector-free reference exports from the research phase (finance, healthcare, supply chain); production never needs it.

## What this repository does

| Responsibility | This repo | Airweave |
|---|---|---|
| Ingestion, source connectors | | ✔ |
| Retrieval chunking (`SemanticChunker`) | | ✔ |
| Embeddings for retrieval, Vespa storage, retrieval | | ✔ |
| ~300-token taxonomy analysis chunks (temporary) | ✔ | |
| Taxonomy discovery (UMAP + HDBSCAN, multi-seed stability) | ✔ | |
| Semantic review of candidate clusters (LLM, logged) | ✔ | |
| Broad-family consolidation (LLM, logged, validated) | ✔ | |
| Assigning categories to new chunks (dense + BM25 + RRF, **no LLM**) | ✔ | |
| Novelty maintenance (periodic, cluster-level LLM review) | ✔ | |
| Mapping labels back to Airweave chunk ids | ✔ | |

**This repo does NOT:** replace Airweave, perform retrieval, replace Vespa, change the Airweave chunk size, or require 300-token chunks in production retrieval. The ~300-token chunks exist only inside the classification workflow.

## Architecture

```
                         OFFLINE (occasional)  -- scripts/build_taxonomy.py
Airweave chunks
      ↓
300-token taxonomy chunks
      ↓
embeddings (canonical cache)
      ↓
UMAP + HDBSCAN  (5 seeds x 13 settings, stability + screens + de-duplication)
      ↓
LLM cluster review  (one logged call per candidate cluster)
      ↓
fine taxonomy
      ↓
broad families  (one logged LLM call + deterministic validation)
      ↓
calibration (leave-group-out) + frozen artifacts


                         ONLINE / INCREMENTAL  -- scripts/classify_airweave_chunks.py
New Airweave chunk
      ↓
300-token children
      ↓
Dense + BM25
      ↓
RRF  (k = 60, calibrated thresholds)
      ↓
semantic topic IDs
      ↓
aggregate to Airweave chunk


                         PERIODIC  -- scripts/run_novelty_maintenance.py
unresolved taxonomy chunks → cluster → compare to existing taxonomy → one LLM review per cluster
→ existing category (verified per chunk) OR proposed new category
```

Code map (`src/taxonomy_classifier/`):

| Package | Contents |
|---|---|
| `chunking/` | `taxonomy_chunker.py` (300-token children with lineage), `content_form.py` (administrative / table / narrative tags) |
| `embeddings/` | `cache.py` (canonical SQLite cache), `embedder.py` |
| `discovery/` | `clustering.py` (UMAP × HDBSCAN grid), `stability.py`, `candidate_clusters.py` (stability, screens, de-dup, hierarchy) |
| `llm/` | `client.py` (the only chat-API caller: versioned prompts, strict JSON schema, cache, call log, cost cap), `cluster_review.py`, `consolidation.py`, `novelty_review.py`, `prompts/` |
| `taxonomy/` | `builder.py` (categories from logged responses only), `artifacts.py` (save/load), `models.py` |
| `assignment/` | `dense.py`, `sparse.py` (BM25 with frozen vocabulary), `rrf.py`, `classifier.py` (calibration + multi-label assignment) |
| `novelty/` | `maintenance.py` |
| `integration/` | `airweave.py` (`classify_chunks`, `TaxonomyClassifier`), `lineage.py`, `aggregation.py` |
| `pipeline.py` | taxonomy build orchestration |

## How it connects to Airweave

Airweave already creates large retrieval chunks:

```
Airweave document  →  SemanticChunker  →  retrieval chunk AW_001   (entity_id "<entity>__chunk_<i>")
```

This repo receives `AW_001` and internally creates temporary analysis units:

```
AW_001
├── AW_001::tc000  (~300 tokens)
├── AW_001::tc001
├── AW_001::tc002
└── AW_001::tc003
```

Each child is classified, then the labels are aggregated back:

```
AW_001::tc000 → Deferred Taxes
AW_001::tc001 → Deferred Taxes + Fair Value
AW_001::tc002 → Debt
                    ↓
{"chunk_id": "AW_001", "taxonomy": {"topic_names": ["Deferred Taxes", "Fair Value", "Debt"], ...}}
```

Airweave continues retrieving `AW_001`. Why children? Airweave chunks are large (up to ~4k tokens) and usually cover several subjects; classifying ~300-token pieces gives multi-label topics with evidence (`supporting_taxonomy_chunks`) instead of one blurred label.

Full integration guide: [docs/AIRWEAVE_INTEGRATION.md](docs/AIRWEAVE_INTEGRATION.md).

## Installation

```bash
git clone <repo-url>
cd airweave-taxonomy-classifier
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env              # then put your OPENAI_API_KEY in .env
cp config/taxonomy.example.yaml config/taxonomy.yaml
```

Python ≥ 3.11 (verified on 3.12). Dependencies: numpy, pandas, scipy, scikit-learn, umap-learn, hdbscan, tiktoken, PyYAML. Nothing from the Airweave dependency tree is required.

## Environment variables

| Variable | Needed for |
|---|---|
| `OPENAI_API_KEY` | taxonomy build, novelty maintenance, and classification of chunks whose text is not yet in the embedding cache |

The key is read from the environment or `./.env` at call time and is never logged or written to any artifact.

## Input schema

JSONL, one Airweave retrieval chunk per line:

```json
{"chunk_id": "a1b2__chunk_3", "document_id": "a1b2", "text": "…chunk text…", "metadata": {"company": "…", "industry": "…", "source": "…"}}
```

| Field | Required | Airweave source |
|---|---|---|
| `chunk_id` | yes | chunk entity `entity_id` (`<original_entity_id>__chunk_<idx>`) |
| `document_id` | yes | `airweave_system_metadata.original_entity_id` |
| `text` | yes | chunk `textual_representation` |
| `metadata` | no | preserved verbatim in the output |

Three optional metadata keys feed the algorithm (`input.fields` in the config): `group` (default key `company`; used for diversity screens and leave-group-out calibration; e.g. company, tenant, source connection), `segment` (default `industry`) and `doc_type` (default `source`), both shown to the cluster reviewer. Missing keys fall back to `document_id` / `"n/a"`.

## Build a taxonomy

Run occasionally, offline, on a representative corpus (the reference build used ~1,100 Airweave chunks / 11k taxonomy chunks):

```bash
python scripts/build_taxonomy.py \
  --input airweave_chunks.jsonl \
  --config config/taxonomy.yaml \
  --output artifacts/taxonomy/
```

Pipeline: Airweave chunks → 300-token children → embeddings → UMAP/HDBSCAN → candidate clusters → LLM review → fine categories → broad families → calibration → frozen taxonomy. Also writes `build_enriched_chunks.jsonl` (the build corpus itself, enriched; disable with `--no-classify-input`).

Requirements on the corpus: at least 5 distinct `group` values (calibration folds), and topics that recur across ≥ 5 documents (`clustering.screens`). Set `llm.corpus_description` to a short phrase describing the corpus; it is inserted into the review prompts.

## Classify new Airweave chunks

Run for newly ingested documents. **0 LLM calls.**

```bash
python scripts/classify_airweave_chunks.py \
  --taxonomy artifacts/taxonomy/final_taxonomy.json \
  --input new_airweave_chunks.jsonl \
  --output enriched_chunks.jsonl \
  --config config/taxonomy.yaml \
  --unresolved-output unresolved_chunks.jsonl
```

Pipeline: new Airweave chunk → 300-token children → dense + BM25 → RRF → semantic topics → aggregate back to the original Airweave chunk. A category is accepted for a child when it is in the RRF top-5, outranks every reject (administrative/structural) prototype, clears its calibrated dense-cosine floor and clears the calibrated RRF threshold τ. Administrative children (signatures, exhibit indexes, cover pages, too little text) bypass assignment.

From Python:

```python
from taxonomy_classifier.integration.airweave import classify_chunks

enriched_chunks = classify_chunks(airweave_chunks, taxonomy="artifacts/taxonomy", config="config/taxonomy.yaml")
```

Use `TaxonomyClassifier(taxonomy, config).classify(chunks)` to keep the taxonomy loaded across batches; it also returns the `unresolved_children` novelty pool.

## Run novelty maintenance

Unresolved taxonomy chunks accumulate in `unresolved_chunks.jsonl`. Run periodically (e.g. weekly):

```bash
python scripts/run_novelty_maintenance.py \
  --taxonomy artifacts/taxonomy/final_taxonomy.json \
  --input unresolved_chunks.jsonl \
  --config config/taxonomy.yaml
```

Flow: unresolved chunks → UMAP + HDBSCAN (3 seeds) → stable residual clusters with ≥ 8 chunks from ≥ 2 documents → whole-taxonomy candidate search (dense + BM25 RRF) → **one LLM review per cluster** (never per chunk) → either *existing category* (each chunk joins it only if that category is in the chunk's own dense top-5) or *proposed new category* (needs coherence, GLOBAL/INDUSTRY scope and ≥ 2 source groups). Output: `novelty_residual_clusters.csv`, `novelty_mappings.csv`, `taxonomy_maintenance_proposal.json`. **The frozen taxonomy is not modified**; proposals are promoted by a rebuild.

## Output schema

One record per input Airweave chunk, same order, same `chunk_id`:

```json
{
  "chunk_id": "AW_001",
  "document_id": "DOC_01",
  "metadata": {"company": "…"},
  "taxonomy": {
    "taxonomy_version": "taxonomy-20261009",
    "status": "ASSIGNED",
    "topics": [
      {"category_id": "CAT021", "name": "Deferred Taxes", "broad_family_id": "BF04",
       "broad_family_name": "Tax & Accounting Estimates",
       "supporting_taxonomy_chunks": ["AW_001::tc000", "AW_001::tc001"], "support_share": 0.5, "sources": ["HYBRID_RRF"]}
    ],
    "topic_ids": ["CAT021"], "topic_names": ["Deferred Taxes"],
    "broad_family_ids": ["BF04"], "broad_family_names": ["Tax & Accounting Estimates"],
    "taxonomy_chunks": [{"taxonomy_chunk_id": "AW_001::tc000", "status": "ASSIGNED", "char_start": 0, "char_end": 1290, "topic_ids": ["CAT021"]}]
  },
  "content_metadata": {"content_form": "NARRATIVE", "content_form_counts": {"NARRATIVE": 3, "TABLE": 1}}
}
```

| `taxonomy.status` | Meaning |
|---|---|
| `ASSIGNED` | ≥ 1 topic and every substantive child is assigned |
| `PARTIALLY_ASSIGNED` | ≥ 1 topic, but some substantive children are unresolved (common for large multi-subject chunks) |
| `UNRESOLVED` | substantive text but no topic; its children go to the novelty pool |
| `ADMINISTRATIVE` | only administrative / non-semantic text |

Semantic topics never mix with content type: content form (table, narrative, list, signature, …) is under `content_metadata` only. A real example is in [examples/enriched_chunks.example.jsonl](examples/enriched_chunks.example.jsonl) (produced by the offline demo, so its names come from the stand-in reviewer).

## Write topics back into Airweave/Vespa

Add four fields to the Vespa document (the existing dense/sparse retrieval and ranking stay unchanged):

```
field topic_ids type array<string>        { indexing: attribute | summary  attribute: fast-search }
field topics type array<string>           { indexing: attribute | summary }
field broad_family_ids type array<string> { indexing: attribute | summary  attribute: fast-search }
field broad_families type array<string>   { indexing: attribute | summary }
```

and write `taxonomy.topic_ids / topic_names / broad_family_ids / broad_family_names` into them for the same chunk document. Uses: filtering (`topic_ids contains "CAT021"`), boosting, result inspection and agent context. These fields are separate from Airweave's existing document-level `doc_categories`. Details and code: [docs/AIRWEAVE_INTEGRATION.md](docs/AIRWEAVE_INTEGRATION.md).

## Runtime artifacts

`scripts/build_taxonomy.py --output DIR` writes:

| File | Needed for classification | Purpose |
|---|---|---|
| `final_taxonomy.json` | **yes** | families → categories, calibrated thresholds, artifact checksums, frozen chunking/content-form settings |
| `dense_prototypes.npy` | **yes** | dense category prototypes (+ reject prototypes) |
| `sparse_profiles.npz` | **yes** | BM25 category profiles |
| `bm25_vocabulary.json` | **yes** | frozen BM25 vocabulary + IDF |
| config (`config/taxonomy.yaml`) | **yes** | embedding model/dimensions and chunking must match the build (checked) |
| `category_centroids.npy`, `category_sparse_profiles.npz` | novelty only | whole-taxonomy search |
| `fine_categories.csv`, `broad_families.json`, `category_members.json`, `merge_recommendations.csv` | no | inspection |
| `semantic_reviews.csv`, `candidate_clusters.jsonl`, `llm_calls/`, `llm_calls.jsonl` | no | evidence for every decision |
| `prompt_versions/`, `run_manifest.json`, `embedding_stats.json`, `cost_ledger.json` | no | reproducibility record |
| `intermediate/` (`X.npy`, children, clustering runs) | no | debugging; can be deleted |

**For normal classification you need:** `final_taxonomy.json`, the dense category prototypes, the sparse/BM25 category profiles and vocabulary, and the config. **You do NOT need:** old clustering runs, historical plots, experiment reports, or Claude Code. Checksums in `final_taxonomy.json` are verified at load time.

## Cost model

| Operation | API calls | Reference figure |
|---|---|---|
| Build | embeddings for every unique child text; 1 LLM call per candidate cluster + 1–2 consolidation calls | ~1.1k Airweave chunks / 11.5k children / 199 LLM calls: **≈ $1.2** (gpt-5.4-mini, text-embedding-3-small) |
| Classification | embeddings for new child texts only (cache by text hash); **0 LLM calls** | ≈ $0.02 per 1M tokens of chunk text |
| Novelty maintenance | embeddings (usually cached) + 1 LLM call per recurring residual cluster | a few cents per run |

`llm.max_cost_usd` is a hard cap for each build / maintenance run; calls that could exceed it are refused (status `BUDGET_EXCEEDED`), never silently filled in. Classification roughly doubles embedding tokens relative to Airweave's own retrieval embeddings, because the chunk text is embedded again as ~300-token children.

## Reproducibility

- **Canonical embeddings.** The embeddings API is not bit-deterministic. Identical text is embedded once and its first stored vector is reused everywhere (`embeddings/cache.py`; test `tests/test_embedding_cache.py`). Without this, duplicate passages get different vectors and a rerun clusters differently.
- **LLM cache + call log.** Every call is stored with its prompt version, full payload, raw and parsed response, usage and cost (`llm_calls/`), and cached by input hash. Rerunning a build with the same caches reproduces the same taxonomy at $0.
- **Verified equivalence.** Packaging was checked against the authoritative reference run (60 filings, 20 companies) with the original v1 prompts and caches, at $0 (all 199 LLM calls were cache hits):
  - Taxonomy build: identical 127 categories, names, descriptions, members, parents and 10 broad families; identical calibration.
  - Classification of the 477 held-out Airweave chunks (4,878 children): identical statuses and labels for every child and every Airweave chunk, 0 LLM calls.
- **Prompts.** The packaged prompts are `*_v2`: the verified `*_v1` text with the corpus description made configurable (`llm.corpus_description`) and source-neutral wording ("documents", "source groups"). The decision rules, JSON schemas and payload structure are unchanged. A build with v2 prompts makes fresh LLM calls.
- **A cold rebuild** (empty caches) is not bit-identical: the chat API is not deterministic. In a rerun experiment with fixed embeddings but fresh LLM calls, clustering was 100 % identical and ≈ 95 % of categories were reproduced. Names and family groupings varied more (≈ 50 % exact-name agreement; 10 vs 14 families), and all of that variance came from the LLM layer. Freeze a taxonomy and classify against it; do not rebuild per document.

## Limitations

- Topic quality depends on the build corpus: categories exist only for subjects that recur across ≥ 5 documents. Rare subjects stay `UNRESOLVED` until novelty maintenance proposes them.
- Category names and families come from an LLM. Review `fine_categories.csv` and `merge_recommendations.csv` before publishing a taxonomy; overlapping/near-duplicate names can occur.
- Content-form patterns (administrative bypass) are configuration; the shipped patterns were tuned on regulatory filings. Adapt `content_form.patterns` for other document types.
- Calibration needs ≥ 5 source groups; with fewer, thresholds cannot be estimated leave-group-out.
- Large Airweave chunks are often `PARTIALLY_ASSIGNED` (some children have no confident topic); this is expected, not an error.
- Novelty maintenance only proposes. Promoting a proposal means a rebuild (or re-consolidation) and a new `taxonomy_version`.
- English-oriented BM25 tokenisation and stop words.

## Example

Offline (no API key, deterministic stand-ins for the APIs, $0):

```bash
python scripts/run_end_to_end_demo.py --offline
```

Real APIs on the example corpus (a few cents):

```bash
python scripts/run_end_to_end_demo.py
```

Worked example of the mapping:

```
Airweave chunk AW_012
"... deferred tax assets ... valuation allowance ... level 3 fair value inputs ..."
        ↓
AW_012::tc000 → Deferred Taxes
AW_012::tc001 → Deferred Taxes, Fair Value Measurement
        ↓
AW_012 enrichment:
  topics:   [Deferred Taxes, Fair Value Measurement]
  families: [Tax & Accounting Estimates, Fair Value, Securities & Derivatives]
  supporting_taxonomy_chunks: Deferred Taxes ← tc000, tc001; Fair Value Measurement ← tc001
```

(`tests/test_lineage.py::test_aggregation_worked_example` asserts exactly this aggregation.)

## Tests

```bash
pytest                 # all tests, no network, no API spend (fake transports)
pytest -m "not slow"   # unit tests only
```

Tests cover a full discover → store → load → classify → evolve run on a non-finance corpus (`tests/test_generic_corpus.py`), the `topics[]` contract on a neutral fixture taxonomy (`tests/fixtures/taxonomy.json`), chunking and lineage, the content-form bypass, RRF, assignment rules, the frozen BM25 round trip, the canonical-embedding duplicate-text rule, a synthetic end-to-end build, classification with the LLM disabled, novelty maintenance (one call per cluster), the CLIs, and a guard against corpus-specific terms and research-path dependencies in runtime code. `tests/reference_data/` validates the shipped reference exports in `taxonomies/`.
