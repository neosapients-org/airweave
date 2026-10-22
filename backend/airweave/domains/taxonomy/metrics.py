"""Chunk counts per label status (contract C11: fail-open, counted and alerted).

One process-wide tally feeds two outputs:

- the Prometheus counter ``taxonomy_chunks_total{status}``, served by the worker's metrics
  endpoint (registered there only while ``TAXONOMY_ENABLED`` is on);
- the sync's summary log line, with a warning when ``ERROR`` passes 1% of the sync's chunks
  (``ERROR_SHARE_ALERT``), written when the sync ends.
"""

from __future__ import annotations

import threading
from collections import Counter
from typing import TYPE_CHECKING, Dict, Iterator, Mapping, Optional

from prometheus_client.core import CounterMetricFamily
from prometheus_client.registry import Collector

from airweave.domains.taxonomy.mapping import ERROR, STATUSES

if TYPE_CHECKING:
    from airweave.core.logging import ContextualLogger

ERROR_SHARE_ALERT = 0.01


class TaxonomyTally:
    """Thread-safe chunk counts per status: process totals and per sync job."""

    def __init__(self) -> None:
        """Start with empty counts."""
        self._lock = threading.Lock()
        self._totals: Counter = Counter()
        self._per_sync: Dict[str, Counter] = {}
        self._versions: Dict[str, set] = {}
        self._alerted: set = set()
        self._embed_fallbacks = 0

    def record(self, sync_job_id: str, counts: Mapping[str, int]) -> Counter:
        """Add one batch's counts; return the sync's counts so far."""
        with self._lock:
            self._totals.update(counts)
            sync = self._per_sync.setdefault(sync_job_id, Counter())
            sync.update(counts)
            return Counter(sync)

    def note_version(
        self,
        sync_job_id: str,
        version: str,
        generation: int,
        migration_id: Optional[str] = None,
        serving_version: Optional[str] = None,
    ) -> None:
        """Record which taxonomy version, generation and migration labelled the sync (section 8)."""
        with self._lock:
            self._versions.setdefault(sync_job_id, set()).add(
                (version, generation, migration_id or "", serving_version or "")
            )

    def pop_versions(self, sync_job_id: str) -> list:
        """Remove and return the sync's ``(version, generation, migration id, serving)``, sorted."""
        with self._lock:
            return sorted(self._versions.pop(sync_job_id, set()))

    def should_alert(self, sync_job_id: str, counts: Mapping[str, int]) -> bool:
        """True once per sync, when ``ERROR`` first passes 1% of its chunks."""
        total = sum(counts.values())
        if not total or counts.get(ERROR, 0) / total <= ERROR_SHARE_ALERT:
            return False
        with self._lock:
            if sync_job_id in self._alerted:
                return False
            self._alerted.add(sync_job_id)
            return True

    def pop_sync(self, sync_job_id: str) -> Optional[Counter]:
        """Remove and return a finished sync's counts (None if the step never ran for it)."""
        with self._lock:
            self._alerted.discard(sync_job_id)
            return self._per_sync.pop(sync_job_id, None)

    def note_embed_fallback(self, version: str) -> None:
        """Count a taxonomy version whose classifier uses the package's own embeddings client."""
        with self._lock:
            self._embed_fallbacks += 1

    def embed_fallbacks(self) -> int:
        """Taxonomy versions that fell back to the package's embeddings client."""
        with self._lock:
            return self._embed_fallbacks

    def totals(self) -> Counter:
        """Process totals per status."""
        with self._lock:
            return Counter(self._totals)


class TaxonomyChunksCollector(Collector):
    """Prometheus collector exposing the tally as ``taxonomy_chunks_total{status}``."""

    def __init__(self, tally: TaxonomyTally) -> None:
        """Wrap a tally."""
        self._tally = tally

    def collect(self) -> Iterator[CounterMetricFamily]:
        """Yield the counter family with one sample per status."""
        family = CounterMetricFamily(
            "taxonomy_chunks", "Chunks written per taxonomy label status", labels=["status"]
        )
        totals = self._tally.totals()
        for status in STATUSES:
            family.add_metric([status], float(totals.get(status, 0)))
        yield family
        fallback = CounterMetricFamily(
            "taxonomy_embed_fallback",
            "Taxonomy versions classified with the package's own embeddings client because "
            "their embedding model or dimensions differ from the deployment's",
        )
        fallback.add_metric([], float(self._tally.embed_fallbacks()))
        yield fallback


# The worker process's tally: the taxonomy step writes it, the worker's metrics endpoint and
# the sync's summary log read it.
TAXONOMY_TALLY = TaxonomyTally()


def format_counts(counts: Mapping[str, int]) -> str:
    """``ASSIGNED=12 ERROR=0 …`` in the contract's status order."""
    return " ".join(f"{s}={counts.get(s, 0)}" for s in STATUSES)


def _describe(version: str, generation: int, migration_id: str, serving: str) -> str:
    """``v (generation 3, migration m-1, serving v0)``; the migration parts only when present."""
    parts = [f"generation {generation}"]
    if migration_id:
        parts.append(f"migration {migration_id}")
    if serving and serving != version:
        parts.append(f"serving {serving}")
    return f"{version} ({', '.join(parts)})"


def log_sync_summary(
    sync_job_id: str, logger: "ContextualLogger", tally: TaxonomyTally = TAXONOMY_TALLY
) -> None:
    """Write the sync's label summary when it ends; nothing if the step never ran."""
    counts = tally.pop_sync(sync_job_id)
    versions = tally.pop_versions(sync_job_id)
    if not counts:
        return
    total = sum(counts.values())
    share = counts.get(ERROR, 0) / total if total else 0.0
    taxonomy = ", ".join(_describe(*v) for v in versions) or "none"
    message = (
        f"[Taxonomy] sync summary: {total} chunks, {format_counts(counts)}, "
        f"ERROR share {share:.2%}, taxonomy {taxonomy}"
    )
    if share > ERROR_SHARE_ALERT:
        logger.error(f"{message} (above {ERROR_SHARE_ALERT:.0%}: labelling is failing)")
    else:
        logger.info(message)
