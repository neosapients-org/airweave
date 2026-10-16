"""The classifier package behind an interface (A13 discover, A12 propose).

``EngineGateway`` is what the job runner calls. ``PackageGateway`` runs the real
``taxonomy_classifier`` package (the full install: UMAP, HDBSCAN, numba) in a working directory
and returns only what may leave the worker: the files the classifier needs to load a version,
plus a report. The package's LLM and embedding calls go through injectable transports, so the
tests run everything offline with fakes; **a real run calls OpenAI and needs the owner's
approval first**.

Nothing but the allow-listed files is returned. The package also writes its audit trail (the
prompts sent to the model, the cluster members, the candidate clusters), which contains
customer text; it stays in the working directory, which the runner deletes.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

from airweave.core.logging import logger
from airweave.domains.taxonomy.artifacts import ResolvedTaxonomy

PASSAGE_CHARS = 400
PASSAGES_PER_CATEGORY = 3


@dataclass
class BuildOutput:
    """What a job produces: files (name to bytes), the report and what it cost."""

    files: Dict[str, bytes]
    report: Dict[str, Any]
    cost_usd: float = 0.0
    categories: int = 0
    notes: List[str] = field(default_factory=list)


class SpendCapReached(RuntimeError):
    """The package's cost ledger refused a call: the job's cap would be exceeded."""


class EngineGateway(Protocol):
    """Blocking calls (the runner runs them on a thread)."""

    def discover(
        self,
        chunks: List[Dict[str, Any]],
        *,
        tenant_id: str,
        candidate_version: str,
        work_dir: Path,
        spend_cap_usd: float,
        corpus_description: Optional[str],
    ) -> BuildOutput:
        """Build a candidate first taxonomy version from a sample of chunks."""
        ...

    def propose(
        self,
        chunks: List[Dict[str, Any]],
        *,
        serving: ResolvedTaxonomy,
        candidate_version: str,
        work_dir: Path,
        spend_cap_usd: float,
    ) -> BuildOutput:
        """Run one evolution wave on the unresolved chunks against the serving version."""
        ...


def sha256_hex(data: bytes) -> str:
    """Hex digest."""
    return hashlib.sha256(data).hexdigest()


def passage(text: str) -> str:
    """A short example passage for the report (customer text; the report is an S3 result)."""
    text = " ".join(text.split())
    return text if len(text) <= PASSAGE_CHARS else text[: PASSAGE_CHARS - 1] + "…"


class PackageGateway:
    """Runs ``TaxonomyEngine.discover`` and ``TaxonomyEngine.run_wave``."""

    def __init__(
        self,
        build_config: Mapping[str, Any],
        *,
        embed_transport: Optional[Callable[..., Any]] = None,
        llm_transport: Optional[Callable[..., Any]] = None,
        reviewers: Optional[Tuple[Any, Any]] = None,
    ) -> None:
        """``build_config`` is the package's YAML as a dict; transports default to its own clients.

        ``reviewers`` (novelty, structural) replace the LLM reviewers of a wave (tests).
        """
        self._config = copy.deepcopy(dict(build_config))
        self._embed = embed_transport
        self._llm = llm_transport
        self._reviewers = reviewers

    # --------------------------------------------------------------- discover

    def discover(
        self,
        chunks: List[Dict[str, Any]],
        *,
        tenant_id: str,
        candidate_version: str,
        work_dir: Path,
        spend_cap_usd: float,
        corpus_description: Optional[str],
    ) -> BuildOutput:
        """Build the candidate first version in ``work_dir`` and collect what may be uploaded."""
        from taxonomy_classifier import (  # type: ignore[import-not-found]
            LocalTaxonomyStore,
            TaxonomyEngine,
        )

        cfg = self._job_config(work_dir, spend_cap_usd)
        if corpus_description:
            cfg["llm"]["corpus_description"] = corpus_description
        # No ``group`` key: the package then groups by document, so its leave-group-out
        # calibration has many groups even in a collection with a single source.
        cfg.setdefault("input", {})["fields"] = {"segment": "source", "doc_type": "entity_type"}
        store = LocalTaxonomyStore(work_dir / "store")
        engine = TaxonomyEngine(
            cfg, store=store, embed_transport=self._embed, llm_transport=self._llm
        )
        try:
            taxonomy = engine.discover(
                chunks,
                taxonomy_id=tenant_id,
                domain=None,
                version=candidate_version,
                log=lambda m: logger.info(f"[Taxonomy build] {m}"),
            )
        except RuntimeError as e:
            if "cost cap" in str(e):
                raise SpendCapReached(str(e)) from e
            raise
        folder = Path(taxonomy.path)
        files = self._allowed_files(folder, taxonomy.final)
        files["taxonomy.yaml"] = self._shipped_yaml(cfg)
        cost = _ledger_total(folder / "cost_ledger.json")
        report = self._report(
            kind="discover",
            candidate_version=candidate_version,
            final=taxonomy.final,
            by_id={c["chunk_id"]: c for c in chunks},
            cost=cost,
            sample={"chunks": len(chunks), "documents": len({c["document_id"] for c in chunks})},
        )
        return BuildOutput(files, report, cost, len(taxonomy.final["categories"]))

    # ---------------------------------------------------------------- propose

    def propose(
        self,
        chunks: List[Dict[str, Any]],
        *,
        serving: ResolvedTaxonomy,
        candidate_version: str,
        work_dir: Path,
        spend_cap_usd: float,
    ) -> BuildOutput:
        """One wave on the unresolved chunks; the candidate is serving plus the additions."""
        from taxonomy_classifier import (  # type: ignore[import-not-found]
            LocalTaxonomyStore,
            TaxonomyEngine,
        )
        from taxonomy_classifier.taxonomy import artifacts  # type: ignore[import-not-found]

        cfg = copy.deepcopy(serving.config)  # the serving version's own taxonomy.yaml
        job_cfg = self._job_config(work_dir, spend_cap_usd, base=cfg)
        base = artifacts.load(serving.local_path)
        store = LocalTaxonomyStore(work_dir / "store")
        engine = TaxonomyEngine(
            job_cfg,
            taxonomy=base,
            store=store,
            embed_transport=self._embed,
            llm_transport=self._llm,
        )
        reviewer, structural = self._reviewers or (None, None)
        try:
            wave = engine.run_wave(
                chunks,
                wave=1,
                new_version=candidate_version,
                reviewer=reviewer,
                structural_reviewer=structural,
                log=lambda m: None,
            )
        except RuntimeError as e:
            if "cost cap" in str(e):
                raise SpendCapReached(str(e)) from e
            raise
        decisions = [] if len(wave.decisions) == 0 else wave.decisions.to_dict("records")
        counts: Dict[str, int] = {}
        for d in decisions:
            counts[d["decision"]] = counts.get(d["decision"], 0) + 1
        if wave.new_taxonomy_path is None:
            report = self._wave_report(candidate_version, serving, None, wave, counts, chunks, 0.0)
            return BuildOutput({}, report, 0.0, 0, ["no category was added: nothing to propose"])
        folder = Path(wave.new_taxonomy_path)
        final = json.loads((folder / "final_taxonomy.json").read_text())
        files = self._allowed_files(folder, final)
        # the candidate keeps the serving version's own taxonomy.yaml (same embedding, same policy)
        files["taxonomy.yaml"] = (serving.local_path / "taxonomy.yaml").read_bytes()
        cost = _ledger_total(
            work_dir
            / "store"
            / base.taxonomy_id
            / "_waves"
            / candidate_version
            / "cost_ledger.json"
        )
        report = self._wave_report(candidate_version, serving, final, wave, counts, chunks, cost)
        return BuildOutput(files, report, cost, len(final["categories"]))

    # ---------------------------------------------------------------- helpers

    def _job_config(
        self, work_dir: Path, cap: float, base: Optional[Mapping[str, Any]] = None
    ) -> Dict[str, Any]:
        """The package's config with this job's paths and spend cap (local, never shipped)."""
        cfg = copy.deepcopy(dict(base if base is not None else self._config))
        cfg.setdefault("embedding", {})["cache_path"] = str(work_dir / "embeddings.sqlite")
        cfg.setdefault("llm", {}).update(
            {"cache_dir": str(work_dir / "llm_cache"), "max_cost_usd": float(cap)}
        )
        return cfg

    @staticmethod
    def _allowed_files(folder: Path, final: Mapping[str, Any]) -> Dict[str, bytes]:
        """``final_taxonomy.json`` and the artifacts it lists; nothing else leaves the worker."""
        names = ["final_taxonomy.json", *sorted((final.get("artifacts") or {}).keys())]
        return {name: (folder / name).read_bytes() for name in names}

    @staticmethod
    def _shipped_yaml(cfg: Mapping[str, Any]) -> bytes:
        """``taxonomy.yaml`` for the candidate: the config with local paths and caps removed.

        A first version has no model, so the policy is calibrated unless the config says otherwise.
        """
        import yaml  # type: ignore[import-untyped]

        shipped = copy.deepcopy(dict(cfg))
        shipped.setdefault("embedding", {})["cache_path"] = "cache/embeddings.sqlite"
        llm = shipped.setdefault("llm", {})
        llm["cache_dir"] = "cache/llm"
        llm.pop("max_cost_usd", None)
        assignment = shipped.setdefault("assignment", {})
        acceptance = assignment.setdefault("acceptance", {})
        acceptance.setdefault("policy", "calibrated")
        return str(yaml.safe_dump(shipped, sort_keys=True)).encode()

    def _report(
        self,
        *,
        kind: str,
        candidate_version: str,
        final: Mapping[str, Any],
        by_id: Mapping[str, Mapping[str, Any]],
        cost: float,
        sample: Mapping[str, Any],
    ) -> Dict[str, Any]:
        clusters = [self._category_entry(c, by_id) for c in final["categories"]]
        return {
            "kind": kind,
            "candidate_version": candidate_version,
            "classifier_version": final.get("classifier_version"),
            "sample": dict(sample),
            "categories": len(clusters),
            "clusters": clusters,
            "cost_usd": cost,
        }

    @staticmethod
    def _category_entry(
        category: Mapping[str, Any], by_id: Mapping[str, Mapping[str, Any]]
    ) -> Dict[str, Any]:
        examples: List[str] = []
        members = (category.get("created") or {}).get("member_taxonomy_chunk_ids") or []
        for tc in category.get("representative_chunk_ids") or members:
            chunk = by_id.get(str(tc).split("::", 1)[0])
            if chunk is not None and len(examples) < PASSAGES_PER_CATEGORY:
                examples.append(passage(chunk["text"]))
        return {
            "category_id": category["category_id"],
            "name": category.get("name"),
            "description": category.get("description"),
            "status": category.get("status"),
            "origin": category.get("origin", "BUILD"),
            "member_count": category.get("member_count"),
            "document_count": category.get("document_count"),
            "top_terms": category.get("top_terms"),
            "example_passages": examples,
        }

    def _wave_report(
        self,
        candidate_version: str,
        serving: ResolvedTaxonomy,
        final: Optional[Mapping[str, Any]],
        wave: Any,
        counts: Mapping[str, int],
        chunks: Sequence[Mapping[str, Any]],
        cost: float,
    ) -> Dict[str, Any]:
        by_id = {c["chunk_id"]: c for c in chunks}
        added = set(wave.summary.get("categories_added") or [])
        new = (
            [
                self._category_entry(c, by_id)
                for c in final["categories"]
                if c["category_id"] in added
            ]
            if final
            else []
        )
        for entry in new:
            entry["status"] = entry.get("status") or "PROBATIONARY"
        mapped = [
            {
                "cluster": d.get("residual_cluster_id"),
                "target_category_id": d.get("target_category_id"),
                "members_verified": d.get("members_verified"),
                "decided_by": d.get("decided_by"),
            }
            for d in ([] if len(wave.decisions) == 0 else wave.decisions.to_dict("records"))
            if d["decision"] == "MAP_EXISTING"
        ]
        kept = [
            {
                "cluster": d.get("residual_cluster_id"),
                "members": d.get("members"),
                "why": d.get("decided_by"),
            }
            for d in ([] if len(wave.decisions) == 0 else wave.decisions.to_dict("records"))
            if d["decision"] == "KEEP_UNRESOLVED"
        ]
        return {
            "kind": "propose",
            "candidate_version": candidate_version,
            "base_version": serving.taxonomy_version,
            "sample": {"chunks": len(chunks), "documents": len({c["document_id"] for c in chunks})},
            "decisions": dict(counts),
            "categories_added": new,
            "mappings_into_existing": mapped,
            "kept_unresolved": kept,
            "categories": len(final["categories"]) if final else len(serving.registry.categories),
            "summary": {
                k: v
                for k, v in dict(wave.summary).items()
                if isinstance(v, (int, float, str, list))
            },
            "cost_usd": cost,
        }


def _ledger_total(path: Path) -> float:
    try:
        return float(json.loads(path.read_text()).get("total_usd", 0.0))
    except (OSError, ValueError):
        return 0.0
