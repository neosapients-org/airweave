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
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

from airweave.core.logging import logger
from airweave.domains.taxonomy.artifacts import ResolvedTaxonomy, is_safe_name
from airweave.domains.taxonomy.jobs.housekeeping import tenant_cache_dir
from airweave.domains.taxonomy.jobs.sampling import TooFewDocuments

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


class CallLogDamaged(SpendCapReached):
    """A record of the run's call log cannot be read, so a refused call may be hidden in it."""


def refuse_if_capped(run_dir: Path, cause: Optional[BaseException] = None) -> None:
    """Fail the job, naming the cap, when the spend cap refused any of the run's model calls.

    The package records a refused call as ``BUDGET_EXCEEDED`` in the run's ``llm_calls.jsonl``
    and goes on: a refused cluster review drops that cluster, a refused novelty review keeps its
    chunks unresolved, and a refused consolidation fails later with another message. A
    candidate built with refused calls is missing decisions, so it is never returned.

    A record that cannot be read could be the refused one, so it fails the job too: the guard
    never passes on a log it could not read in full.

    Raises:
        SpendCapReached: When at least one call was refused (chained to ``cause`` if given).
        CallLogDamaged: When a record of the log cannot be read.
    """
    refused, damaged = _refused_calls(run_dir / "llm_calls.jsonl")
    if damaged:
        raise CallLogDamaged(
            f"{damaged} record(s) of the run's call log cannot be read, so it cannot be shown "
            "that the spend cap refused no call: start the job again"
        ) from cause
    if refused:
        raise SpendCapReached(
            f"the spend cap refused {refused} model call(s), so the candidate would be missing "
            "decisions: raise the job's spend cap and start it again"
        ) from cause


def _refused_calls(path: Path) -> Tuple[int, int]:
    """Calls in a run's ``llm_calls.jsonl`` the spend cap refused, and unreadable records."""
    if not path.exists():
        return 0, 0
    refused = damaged = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            damaged += 1
            continue
        if not isinstance(record, dict):
            damaged += 1
        elif record.get("status") == "BUDGET_EXCEEDED":
            refused += 1
    return refused, damaged


class EngineGateway(Protocol):
    """Blocking calls (the runner runs them on a thread)."""

    #: False when the worker has no classifier build config: discover cannot run, propose can.
    can_discover: bool

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
        build_config: Optional[Mapping[str, Any]],
        *,
        embed_transport: Optional[Callable[..., Any]] = None,
        llm_transport: Optional[Callable[..., Any]] = None,
        reviewers: Optional[Tuple[Any, Any]] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        """``build_config`` is the package's YAML as a dict; transports default to its own clients.

        ``build_config`` None: no discover (it needs that YAML); propose uses the serving version's.

        ``reviewers`` (novelty, structural) replace the LLM reviewers of a wave (tests).
        """
        self._config = copy.deepcopy(dict(build_config)) if build_config is not None else None
        self.can_discover = build_config is not None
        self._embed = embed_transport
        self._llm = llm_transport
        self._reviewers = reviewers
        # Empty = every job starts with empty caches (and pays again for a re-run). Set = one
        # embedding file and one LLM cache per tenant, kept between jobs: the same text is never
        # paid for twice and a re-run on the same documents reuses the earlier answers.
        self._cache_dir = Path(cache_dir) if cache_dir else None

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

        if self._config is None:
            raise RuntimeError("no classifier build config: discover cannot run on this worker")
        cfg = self._job_config(work_dir, spend_cap_usd, tenant_id=tenant_id)
        if corpus_description:
            cfg["llm"]["corpus_description"] = corpus_description
        store = LocalTaxonomyStore(work_dir / "store")
        engine = TaxonomyEngine(
            cfg, store=store, embed_transport=self._embed, llm_transport=self._llm
        )
        build_dir = work_dir / "store" / tenant_id / candidate_version  # where the package builds
        try:
            taxonomy = engine.discover(
                chunks,
                taxonomy_id=tenant_id,
                domain=None,
                version=candidate_version,
                log=lambda m: logger.info(f"[Taxonomy build] {m}"),
            )
        except ValueError as e:
            refuse_if_capped(build_dir, e)
            mapped = translate_package_error(e)
            if mapped is e:
                raise
            raise mapped from e
        except RuntimeError as e:
            refuse_if_capped(build_dir, e)
            if "cost cap" in str(e):
                raise SpendCapReached(str(e)) from e
            raise
        refuse_if_capped(build_dir)
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
        job_cfg = self._job_config(work_dir, spend_cap_usd, base=cfg, tenant_id=serving.tenant_id)
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
        wave_dir = work_dir / "store" / base.taxonomy_id / "_waves" / candidate_version
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
            refuse_if_capped(wave_dir, e)
            if "cost cap" in str(e):
                raise SpendCapReached(str(e)) from e
            raise
        refuse_if_capped(wave_dir)
        decisions = [] if len(wave.decisions) == 0 else wave.decisions.to_dict("records")
        counts: Dict[str, int] = {}
        for d in decisions:
            counts[d["decision"]] = counts.get(d["decision"], 0) + 1
        # The wave embeds, clusters and calls reviewers whether or not it adds a category, so its
        # bill is read from its ledger on both paths.
        cost = _ledger_total(wave_dir / "cost_ledger.json")
        if wave.new_taxonomy_path is None:
            report = self._wave_report(candidate_version, serving, None, wave, counts, chunks, cost)
            return BuildOutput({}, report, cost, 0, ["no category was added: nothing to propose"])
        folder = Path(wave.new_taxonomy_path)
        final = json.loads((folder / "final_taxonomy.json").read_text())
        files = self._allowed_files(folder, final)
        if place_new_topics_in_families(final, _load_centroids(folder)):
            files["final_taxonomy.json"] = json.dumps(final, indent=2).encode()
        # the candidate keeps the serving version's own taxonomy.yaml (same embedding, same policy)
        files["taxonomy.yaml"] = (serving.local_path / "taxonomy.yaml").read_bytes()
        report = self._wave_report(candidate_version, serving, final, wave, counts, chunks, cost)
        return BuildOutput(files, report, cost, len(final["categories"]))

    # ---------------------------------------------------------------- helpers

    def _job_config(
        self,
        work_dir: Path,
        cap: float,
        base: Optional[Mapping[str, Any]] = None,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """The package's config with this job's paths and spend cap (local, never shipped)."""
        cfg = copy.deepcopy(dict(base if base is not None else self._config or {}))
        root = work_dir
        if self._cache_dir is not None and tenant_id:
            root = tenant_cache_dir(self._cache_dir, tenant_id)
            root.mkdir(parents=True, exist_ok=True)
        # No ``group`` key: the package then groups by document, so its leave-group-out calibration
        # and a new topic's "at least two groups" rule count documents, not sources. Without this a
        # collection with a single source never gets a new topic (every cluster has one group).
        cfg.setdefault("input", {})["fields"] = {"segment": "source", "doc_type": "entity_type"}
        cfg.setdefault("embedding", {})["cache_path"] = str(root / "embeddings.sqlite")
        cfg.setdefault("llm", {}).update(
            {"cache_dir": str(root / "llm_cache"), "max_cost_usd": float(cap)}
        )
        return cfg

    @staticmethod
    def _allowed_files(folder: Path, final: Mapping[str, Any]) -> Dict[str, bytes]:
        """``final_taxonomy.json`` and the artifacts it lists; nothing else leaves the worker."""
        names = ["final_taxonomy.json", *sorted((final.get("artifacts") or {}).keys())]
        root = folder.resolve()
        files: Dict[str, bytes] = {}
        for name in names:
            # The names come from a file the package wrote; a name that is not a plain file name
            # (a path, "..", an absolute path) would read outside the folder.
            if not is_safe_name(name):
                raise ValueError(f"artifact name {name!r} is not a plain file name")
            path = (folder / name).resolve()
            if path.parent != root or not path.is_file():
                raise ValueError(f"artifact {name!r} is not a file of the build folder")
            files[name] = path.read_bytes()
        return files

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
        # A category a wave created before the package wrote its counts has them in its stats only;
        # Meridian refuses a report whose count is missing.
        stats = category.get("stats") or {}
        documents = category.get("document_count")
        return {
            "category_id": category["category_id"],
            "name": category.get("name"),
            "description": category.get("description"),
            "status": category.get("status"),
            "origin": category.get("origin", "BUILD"),
            "member_count": category.get("member_count"),
            "document_count": documents if documents is not None else stats.get("cat_documents"),
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


def _load_centroids(folder: Path) -> Any:
    import numpy as np

    return np.load(folder / "category_centroids.npy")


def place_new_topics_in_families(final: Dict[str, Any], centroids: Any) -> bool:
    """Give each topic a wave added, which the package leaves without a family, its nearest family.

    The nearest family is the one of the existing topic whose centroid is closest (cosine). The
    registry needs every topic in a family, and a reviewer can still move a topic later.

    Args:
        final: ``final_taxonomy.json`` as a dict; changed in place.
        centroids: One row per topic, in the order of ``final["categories"]``.

    Returns:
        True when something was changed.

    Raises:
        ValueError: When the rows do not match the topics, or no topic has a family.
    """
    import numpy as np

    categories = final["categories"]
    missing = [i for i, c in enumerate(categories) if not c.get("broad_family_id")]
    if not missing:
        return False
    if len(centroids) != len(categories):
        raise ValueError("the candidate's centroids do not match its topics")
    families = {f["broad_family_id"]: f for f in final.get("broad_families", [])}
    homed = [i for i, c in enumerate(categories) if c.get("broad_family_id") in families]
    if not homed:
        raise ValueError("no existing topic has a family to place a new topic in")
    matrix = np.asarray(centroids, dtype=np.float64)
    matrix = matrix / np.where(
        np.linalg.norm(matrix, axis=1, keepdims=True) == 0,
        1,
        np.linalg.norm(matrix, axis=1, keepdims=True),
    )
    for i in missing:
        nearest = homed[int(np.argmax(matrix[homed] @ matrix[i]))]
        family = categories[nearest]["broad_family_id"]
        categories[i]["broad_family_id"] = family
        categories[i]["broad_family"] = families[family]["name"]
    return True


def translate_package_error(error: Exception) -> Exception:
    """The package's "corpus too small" as the job's own "too little text".

    A calibration that has nothing to calibrate on shows the tab a sentence, not a class name.
    """
    if type(error).__name__ == "CorpusTooSmall":
        return TooFewDocuments(f"not enough text yet: {error}. Add more documents and try again")
    return error


class CostUnknown(RuntimeError):
    """A build finished but its cost ledger cannot be read: what it spent is not known."""


def _ledger_total(path: Path) -> float:
    """The ledger's total in USD.

    A missing or malformed ledger is an error, never zero: the calls were paid for, and a result
    that says USD 0 would hide that from the customer.
    """
    try:
        total = float(json.loads(path.read_text())["total_usd"])
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise CostUnknown(f"cost ledger {path.name} is missing or unreadable: {e}") from e
    if not math.isfinite(total) or total < 0:
        raise CostUnknown(f"cost ledger {path.name} holds an impossible total {total!r}")
    return total
