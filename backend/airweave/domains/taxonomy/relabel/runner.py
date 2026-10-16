"""Runs re-label jobs (A9; P7 design section 5, contract 7).

A job scrolls the collection's chunks, classifies each chunk's **own text** through the same
classifier pool the sync uses, and writes label fields only. Every update is a Vespa test-and-set
on the chunk's content hash, so a sync that replaced the chunk meanwhile always wins: the update
is skipped and counted, never written onto new text.

The runner lives in the worker process (the taxonomy volume, the S3 role and the classifier
threads are there). The API only enqueues a row; ``sweep_once`` claims rows whose lease is free
or expired, which is also how a job resumes after a restart: the checkpoint is an optimisation,
a rescan is correct because chunks already at the target are skipped.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

from airweave.core.events.taxonomy import TaxonomyRelabelCompletedEvent
from airweave.domains.taxonomy.artifacts import (
    ResolvedTaxonomy,
    TaxonomyResolutionError,
    TaxonomyResolver,
)
from airweave.domains.taxonomy.classifier_pool import ClassifierPool
from airweave.domains.taxonomy.mapping import administrative_labels, map_record
from airweave.domains.taxonomy.relabel.chunk_store import (
    APPLIED,
    ChunkStore,
)
from airweave.domains.taxonomy.relabel.chunks import F_TEXT, SCAN_FIELDS, ChunkRow, label_fields
from airweave.domains.taxonomy.relabel.jobs import (
    CANCELLED,
    COMPLETED,
    FAILED,
    JobRepository,
    RelabelJob,
)
from airweave.domains.taxonomy.relabel.planning import CLASSIFY, STAMP, UNLABELLABLE, decide

logger = logging.getLogger(__name__)

MAX_FAILING_STEPS = 3  # consecutive classifier or store failures before the job gives up


class _Ended(Exception):
    """The job stopped (lease lost, cancelled, failed); nothing more to do for it."""


class _Abort(Exception):
    """The job fails with this message."""


class RelabelRunner:
    """Claims and runs jobs; ``run_forever`` is the worker's background loop."""

    def __init__(
        self,
        repo: JobRepository,
        store_factory: Callable[[], ChunkStore],
        resolver: TaxonomyResolver,
        pool: ClassifierPool,
        event_bus: Any,
        *,
        batch_size: int = 64,
        page_size: int = 100,
        write_concurrency: int = 8,
        lease_seconds: int = 120,
        classify_timeout: float = 60.0,
        usd_per_thousand: float = 0.006,
        resume_interval: float = 60.0,
        owner: Optional[str] = None,
    ) -> None:
        """Wire the runner; ``owner`` names this process's leases."""
        self._repo = repo
        self._store_factory = store_factory
        self._resolver = resolver
        self._pool = pool
        self._bus = event_bus
        self._batch = batch_size
        self._page = page_size
        self._concurrency = write_concurrency
        self._lease = lease_seconds
        self._timeout = classify_timeout
        self._rate = usd_per_thousand
        self._interval = resume_interval
        self._owner = owner or f"relabel-{uuid.uuid4()}"
        self._running: Dict[uuid.UUID, asyncio.Task] = {}

    # ------------------------------------------------------------------ loop

    async def run_forever(self, stop: asyncio.Event) -> None:
        """Sweep until ``stop`` is set; running jobs are left to resume elsewhere."""
        while not stop.is_set():
            try:
                await self.sweep_once()
            except Exception:
                logger.exception("[Relabel] sweep failed")
            try:
                await asyncio.wait_for(stop.wait(), self._interval)
            except asyncio.TimeoutError:
                pass
        for task in list(self._running.values()):
            task.cancel()
        await asyncio.gather(*self._running.values(), return_exceptions=True)

    async def sweep_once(self) -> List[uuid.UUID]:
        """Start every job free to run here and announce finished ones; returns the started ids."""
        started = []
        for job in await self._repo.resumable():
            task = self._running.get(job.id)
            if task is not None and not task.done():
                continue
            self._running[job.id] = asyncio.create_task(self.run_job(job.id))
            started.append(job.id)
        for job in await self._repo.unannounced():
            await self._announce(job)
        self._running = {k: t for k, t in self._running.items() if not t.done()}
        return started

    async def wait_idle(self) -> None:
        """Wait for the jobs started here (tests)."""
        await asyncio.gather(*self._running.values(), return_exceptions=True)

    # ------------------------------------------------------------------- job

    async def run_job(self, job_id: uuid.UUID) -> None:
        """Run one job to its end, or until its lease is lost."""
        job = await self._repo.claim(job_id, self._owner, self._lease)
        if job is None:
            return
        store: Optional[ChunkStore] = None
        try:
            taxonomy = await self._taxonomy(job)
            store = self._store_factory()
            await self._scan(job, taxonomy, store)
        except _Ended:
            return
        except _Abort as e:
            await self._end(job.id, FAILED, str(e))
        except TaxonomyResolutionError as e:
            await self._end(job.id, FAILED, f"taxonomy unusable: {e}")
        except asyncio.CancelledError:
            raise  # shutting down: the lease expires and another worker resumes from the checkpoint
        except Exception as e:
            logger.exception("[Relabel] job %s failed", job.id)
            await self._end(job.id, FAILED, f"{type(e).__name__}: {e}"[:500])
        finally:
            if store is not None:
                await store.close()

    async def _taxonomy(self, job: RelabelJob) -> ResolvedTaxonomy:
        taxonomy = await self._resolver.resolve(job.collection_readable_id, fresh=True)
        self._check_target(job, taxonomy)
        assert taxonomy is not None
        return taxonomy

    @staticmethod
    def _check_target(job: RelabelJob, taxonomy: Optional[ResolvedTaxonomy]) -> None:
        if taxonomy is None:
            raise _Abort("the collection has no active taxonomy file")
        if taxonomy.taxonomy_version != job.target_version:
            raise _Abort(
                f"the active file labels with {taxonomy.taxonomy_version!r}, "
                f"not the job's target {job.target_version!r}"
            )

    async def _scan(self, job: RelabelJob, taxonomy: ResolvedTaxonomy, store: ChunkStore) -> None:
        token = job.checkpoint
        counts = job.counts()
        failing = 0
        fields = (*SCAN_FIELDS, F_TEXT)
        while True:
            try:
                rows, nxt = await store.scroll(job.collection_id, fields, self._page, token)
            except (
                ValueError
            ):  # an unusable checkpoint: start over, chunks at the target are skipped
                token = None
                continue
            page = await self._page_work(job, taxonomy, store, rows, counts)
            failing = failing + 1 if page.failed_step else 0
            if failing >= MAX_FAILING_STEPS:
                await self._save(job, nxt, counts)
                raise _Abort(f"{failing} pages in a row failed: {page.reason}")
            saved = await self._save(job, nxt, counts)
            self._check_spend(job, counts)
            if saved.cancel_requested:
                await self._end(job.id, CANCELLED, None)
                raise _Ended
            token = nxt
            if nxt is None:
                await self._end(job.id, COMPLETED, None)
                return
            self._check_target(job, await self._resolver.resolve(job.collection_readable_id))

    async def _save(
        self, job: RelabelJob, token: Optional[str], counts: Dict[str, int]
    ) -> RelabelJob:
        saved = await self._repo.save_progress(
            job.id, self._owner, self._lease, checkpoint=token, **counts
        )
        if saved is None:
            raise _Ended  # another worker holds the lease now
        return saved

    def _check_spend(self, job: RelabelJob, counts: Dict[str, int]) -> None:
        classified = counts["chunks_relabelled"] + counts["chunks_failed"]
        spent = classified / 1000 * self._rate
        if spent > job.spend_cap_usd + 1e-9:
            raise _Abort(f"spend cap of USD {job.spend_cap_usd} reached ({spent:.4f} used)")

    # ------------------------------------------------------------------ page

    async def _page_work(
        self,
        job: RelabelJob,
        taxonomy: ResolvedTaxonomy,
        store: ChunkStore,
        rows: List[ChunkRow],
        counts: Dict[str, int],
    ) -> "_Page":
        page = _Page()
        to_classify: List[ChunkRow] = []
        writes: List[Tuple[ChunkRow, Dict[str, Any], str]] = []
        for row in rows:
            action = decide(row, job.target_version, job.mode, job.scope)
            if action == UNLABELLABLE:
                counts["chunks_unlabellable"] += 1
            elif action == STAMP:
                writes.append((row, {"taxonomy_version": job.target_version}, "chunks_stamped"))
            elif action == CLASSIFY:
                if row.own_text().strip():
                    to_classify.append(row)
                else:  # empty or whitespace own text (contract 10)
                    labels = label_fields(administrative_labels(), job.target_version)
                    writes.append((row, labels, "chunks_relabelled"))
        for start in range(0, len(to_classify), self._batch):
            batch = to_classify[start : start + self._batch]
            try:
                writes.extend(await self._classify(job, taxonomy, batch, counts))
            except Exception as e:
                logger.error(
                    "[Relabel] job %s: classifying %d chunks failed: %s", job.id, len(batch), e
                )
                counts["chunks_failed"] += len(batch)
                page.failed_step, page.reason = True, f"{type(e).__name__}: {e}"
        outcomes = await self._write(store, writes, counts)
        if outcomes.get("error") and not outcomes.get(APPLIED):
            page.failed_step, page.reason = True, outcomes["error"]
        return page

    async def _classify(
        self,
        job: RelabelJob,
        taxonomy: ResolvedTaxonomy,
        batch: List[ChunkRow],
        counts: Dict[str, int],
    ) -> List[Tuple[ChunkRow, Dict[str, Any], str]]:
        request = [
            {"chunk_id": _cid(r), "document_id": r.document_id, "text": r.own_text()} for r in batch
        ]
        records = {
            r["chunk_id"]: r for r in await self._pool.classify(taxonomy, request, self._timeout)
        }
        writes = []
        for row in batch:
            record = records.get(_cid(row))
            if record is None:
                counts["chunks_failed"] += 1
                continue
            start, end = row.own_range
            labels = map_record(record, taxonomy.registry, start, end)
            writes.append((row, label_fields(labels, job.target_version), "chunks_relabelled"))
        return writes

    async def _write(
        self,
        store: ChunkStore,
        writes: List[Tuple[ChunkRow, Dict[str, Any], str]],
        counts: Dict[str, int],
    ) -> Dict[str, Any]:
        gate = asyncio.Semaphore(self._concurrency)
        outcomes: Dict[str, Any] = {}

        async def one(row: ChunkRow, fields: Dict[str, Any], counter: str) -> None:
            if not row.content_hash:  # never write without the condition
                counts["chunks_failed"] += 1
                return
            async with gate:
                try:
                    result = await store.update(row, fields, row.content_hash)
                except Exception as e:
                    counts["chunks_failed"] += 1
                    outcomes["error"] = f"{type(e).__name__}: {e}"
                    return
            outcomes[result] = outcomes.get(result, 0) + 1
            counts[counter if result == APPLIED else "chunks_skipped_changed"] += 1

        await asyncio.gather(*(one(*w) for w in writes))
        return outcomes

    # ------------------------------------------------------------------- end

    async def _end(self, job_id: uuid.UUID, state: str, error: Optional[str]) -> None:
        done = await self._repo.finish(job_id, self._owner, state, error)
        if done is None:
            return
        logger.info("[Relabel] job %s %s %s", job_id, state, error or "")
        if state in (COMPLETED, FAILED):
            await self._announce(done)

    async def _announce(self, job: RelabelJob) -> None:
        """Publish ``taxonomy.relabel.completed``; a failure leaves it for the next sweep."""
        event = TaxonomyRelabelCompletedEvent(
            organization_id=job.organization_id,
            relabel_job_id=job.id,
            collection_id=job.collection_id,
            collection_readable_id=job.collection_readable_id,
            taxonomy_version=job.target_version,
            previous_taxonomy_version=job.previous_version,
            chunks_relabelled=job.relabelled_in_total,
            chunks_failed=job.chunks_failed,
            status="completed" if job.state == COMPLETED else "failed",
            error=job.error,
        )
        try:
            await self._bus.publish(event)
        except Exception as e:
            logger.error("[Relabel] could not publish the completion event of %s: %s", job.id, e)
            return
        await self._repo.mark_event_sent(job.id)


class _Page:
    def __init__(self) -> None:
        self.failed_step = False
        self.reason = ""


def _cid(row: ChunkRow) -> str:
    return f"{row.schema}/{row.doc_id}"
