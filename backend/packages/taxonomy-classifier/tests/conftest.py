"""Shared fixtures: synthetic corpus + fake embedding / LLM transports (no network, no API spend) + small config."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
FIXTURE_TAXONOMY = Path(__file__).resolve().parent / "fixtures" / "taxonomy.json"   # neutral, domain-free
sys.path.insert(0, str(ROOT / "examples"))
from offline_fakes import (BASIS, DIM, SIGNATURE, TOPICS, FakeEmbeddings, FakeLLM, fake_vector,  # noqa: E402,F401
                           synthetic_chunks, topic_text)


@pytest.fixture
def cfg(tmp_path):
    c = yaml.safe_load(open(ROOT / "config" / "taxonomy.example.yaml"))
    c = copy.deepcopy(c)
    c["embedding"].update({"model": "fake-embedding", "dimensions": DIM, "cache_path": str(tmp_path / "emb.sqlite")})
    c["llm"].update({"cache_dir": str(tmp_path / "llm_cache"), "corpus_description": "synthetic test documents"})
    c["clustering"]["hdbscan_settings"] = [[10, 5], [15, 5]]
    c["clustering"]["umap"]["n_neighbors"] = 10
    c["assignment"]["bm25"]["min_df"] = 2
    c["run"]["taxonomy_version"] = "test"
    return c


def write_jsonl(rows, p):
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return p
