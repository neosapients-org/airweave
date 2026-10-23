"""TaxonomyEngine: the one integration entry point. A thin facade over the existing build / classify / novelty code.

    engine = TaxonomyEngine(config="config/taxonomy.yaml", store=LocalTaxonomyStore("/data/taxonomies"))
    taxonomy = engine.discover(chunks, taxonomy_id="legal_contracts", domain="legal")   # offline; embeddings + LLM
    engine = TaxonomyEngine.load("legal_contracts", config=..., store=...)               # latest version
    result = engine.classify(chunks)            # ContractResult: records with topics[] + taxonomy_coverage; 0 LLM calls
    proposal = engine.evolve(result.unresolved_children, out_dir)   # novelty PROPOSALS only; the taxonomy is not modified
    wave = engine.run_wave(chunks, wave=2)                       # versioned evolution: may write taxonomy v+1 (vN unchanged)

Domain-independent: categories, names, hierarchy and thresholds all come from the taxonomy artifact and the config.
`taxonomy_id` and `domain` are free-form labels recorded in the artifact and on every output record.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from pathlib import Path

from .config import load_config
from .contract import ContractResult, to_contract
from .store import LocalTaxonomyStore
from .taxonomy import artifacts
from .taxonomy.models import Taxonomy


class TaxonomyEngine:
    def __init__(self, config: str | Path | dict, taxonomy: Taxonomy | str | Path | None = None,
                 store: LocalTaxonomyStore | None = None, embed_transport=None, llm_transport=None):
        """config: YAML path or dict (config/taxonomy.example.yaml). taxonomy: a loaded Taxonomy or a built taxonomy dir.
        embed_transport / llm_transport: optional injected API transports (tests, other providers)."""
        self.cfg = load_config(config) if not isinstance(config, dict) else config
        self.store = store
        self.embed_transport, self.llm_transport = embed_transport, llm_transport
        self.taxonomy = artifacts.load(taxonomy) if isinstance(taxonomy, (str, Path)) else taxonomy
        self._classifier = None

    @classmethod
    def load(cls, taxonomy_id: str, config, store: LocalTaxonomyStore, version: str | None = None, **kw) -> "TaxonomyEngine":
        """Engine bound to a stored taxonomy (latest version unless `version` is given)."""
        return cls(config, store.load(taxonomy_id, version), store=store, **kw)

    def discover(self, airweave_chunks, taxonomy_id: str | None = None, domain: str | None = None, version: str | None = None,
                 out_dir: str | Path | None = None, log=print) -> Taxonomy:
        """Build a new taxonomy from a corpus and bind the engine to it. Writes into the store (or out_dir)."""
        from .pipeline import build_taxonomy
        cfg = copy.deepcopy(self.cfg)
        run = cfg.setdefault("run", {})
        run["taxonomy_id"] = taxonomy_id or run.get("taxonomy_id") or artifacts.DEFAULT_TAXONOMY_ID
        run["domain"] = domain or run.get("domain")
        run["taxonomy_version"] = version or run.get("taxonomy_version") or f"taxonomy-{datetime.now(timezone.utc):%Y%m%d%H%M%S}"
        if out_dir is None:
            if self.store is None:
                raise ValueError("discover() needs out_dir or a store")
            out_dir = self.store.path(run["taxonomy_id"], run["taxonomy_version"])
            if Path(out_dir).exists():
                raise FileExistsError(f"{out_dir} already exists; taxonomy versions are immutable")
        build_taxonomy(airweave_chunks, cfg, out_dir, log=log, embed_transport=self.embed_transport,
                       llm_transport=self.llm_transport, classify_input=False)
        self.taxonomy, self._classifier = artifacts.load(out_dir), None
        return self.taxonomy

    def classify(self, airweave_chunks) -> ContractResult:
        """Airweave chunks -> one contract record per chunk (topics[], taxonomy_coverage, taxonomy_id/version)."""
        if self.taxonomy is None:
            raise ValueError("no taxonomy: call discover() or TaxonomyEngine.load() first")
        if self._classifier is None:
            from .integration.airweave import TaxonomyClassifier
            self._classifier = TaxonomyClassifier(self.taxonomy, self.cfg, self.embed_transport)
        res = self._classifier.classify(airweave_chunks)
        return ContractResult([to_contract(e, self.taxonomy) for e in res.enriched], res.unresolved_children, res.embedding_stats)

    def run_wave(self, airweave_chunks, wave: int, new_version: str | None = None, pool: list[dict] | None = None,
                 reviewer=None, structural_reviewer=None, log=print):
        """Versioned wave evolution: classify the wave against the bound taxonomy vN, accumulate unresolved children,
        decide recurring residual clusters (MAP_EXISTING / CREATE_NEW (PROBATIONARY) / STRUCTURAL_OR_ADMIN / KEEP_UNRESOLVED)
        and, if categories were added, write taxonomy vN+1 into the store and bind the engine to it. vN is never modified.

        reviewer / structural_reviewer: objects with review(ctx) (evolution.reviewers); default = logged LLM reviewers.
        Returns evolution.WaveResult (labels, decisions, carried-over pool, contract records, summary).
        """
        from .evolution import run_wave
        from .evolution.reviewers import LLMNoveltyReviewer, LLMStructuralReviewer
        from .llm.costs import CostLedger
        if self.taxonomy is None or self.store is None:
            raise ValueError("run_wave() needs a bound taxonomy and a store")
        new_version = new_version or f"{self.taxonomy.version}+w{wave}"
        self.store.path(self.taxonomy.taxonomy_id, new_version)   # both names are checked before any path is built from them
        # Every wave has a ledger, whichever reviewers it uses: the caller reads the wave's bill from it.
        run_dir = self.store.root / self.taxonomy.taxonomy_id / "_waves" / new_version
        run_dir.mkdir(parents=True, exist_ok=True)
        ledger = CostLedger(self.cfg["llm"]["max_cost_usd"], run_dir / "cost_ledger.json")
        if reviewer is None or structural_reviewer is None:
            reviewer = reviewer or LLMNoveltyReviewer(self.cfg, run_dir, ledger, self.llm_transport)
            structural_reviewer = structural_reviewer or LLMStructuralReviewer(self.cfg, run_dir, ledger, self.llm_transport)
        res = run_wave(self.taxonomy, self.cfg, airweave_chunks, self.store, wave, new_version, reviewer, structural_reviewer,
                       pool=pool, embed_transport=self.embed_transport, log=log, ledger=ledger)
        if res.new_taxonomy_path is not None:
            self.taxonomy, self._classifier = artifacts.load(res.new_taxonomy_path), None
        return res

    def evolve(self, unresolved_children: list[dict], out_dir: str | Path, log=print) -> dict:
        """Novelty maintenance over the accumulated unresolved pool -> mapping / new-category PROPOSALS.

        The bound taxonomy is NOT modified. Turning accepted proposals into the next taxonomy version (and probation /
        promotion of new categories) is not implemented in this package yet; see docs/PRODUCTION_STATUS.md.
        """
        from .novelty.maintenance import run_maintenance
        if self.taxonomy is None:
            raise ValueError("no taxonomy: call discover() or TaxonomyEngine.load() first")
        return run_maintenance(self.taxonomy, unresolved_children, self.cfg, out_dir, log=log,
                               embed_transport=self.embed_transport, llm_transport=self.llm_transport)
