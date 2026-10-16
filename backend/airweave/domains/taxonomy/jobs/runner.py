"""Runs discover (A13) and propose (A12) jobs.

Order of a run: take the lease; if the candidate folder already holds this job's report the job
is done (a replay); sample the chunk text from Vespa; claim the candidate folder (write once);
run the classifier package on a thread while a heartbeat renews the lease; write the allow-listed
files and then the report; delete the working directory (it holds customer text).

A lease that is lost stops the job quietly: another attempt owns it now. A cancel request is
honoured at each phase boundary and after the build, before anything is written to S3. The
package's own cost ledger enforces the spend cap during the build.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from airweave.domains.taxonomy.artifacts import (
    ResolvedTaxonomy,
    TaxonomyResolutionError,
    TaxonomyResolver,
)
from airweave.domains.taxonomy.jobs.candidates import (
    CandidateConflict,
    CandidateTaken,
    CandidateWriter,
)
from airweave.domains.taxonomy.jobs.gateway import BuildOutput, EngineGateway, SpendCapReached
from airweave.domains.taxonomy.jobs.records import (
    CANCELLED,
    COMPLETED,
    FAILED,
    KIND_DISCOVER,
    KIND_PROPOSE,
    TaxonomyJobRecord,
    TaxonomyJobRepository,
)
from airweave.domains.taxonomy.jobs.sampling import (
    TooFewDocuments,
    sample_documents,
    to_package_chunk,
)
from airweave.domains.taxonomy.relabel.chunk_store import ChunkStore
from airweave.domains.taxonomy.relabel.service import unresolved_page

logger = logging.getLogger(__name__)

BUSY = "busy"


class _Ended(Exception):
    """The job stopped (lease lost, cancelled, finished): nothing more to do."""


class TransientUploadError(RuntimeError):
    """The upload to S3 failed. The job stays running with its build kept; Temporal retries."""


class _Abort(Exception):
    """The job fails with this message."""


class TaxonomyJobRunner:
    """Claims and runs one job per call; the Temporal activity calls ``run_job``."""

    def __init__(
        self,
        repo: TaxonomyJobRepository,
        store_factory: Callable[[], ChunkStore],
        candidates: CandidateWriter,
        gateway: EngineGateway,
        resolver: TaxonomyResolver,
        *,
        work_dir: str | Path,
        lease_seconds: int = 300,
        max_chunks: int = 20000,
        default_sample_documents: int = 200,
        min_documents: int = 1,
        min_chunks: int = 0,
        owner: Optional[str] = None,
    ) -> None:
        """Wire the runner; ``owner`` names this process's leases."""
        self._repo = repo
        self._store_factory = store_factory
        self._candidates = candidates
        self._gateway = gateway
        self._resolver = resolver
        self._work = Path(work_dir)
        self._lease = lease_seconds
        self._max_chunks = max_chunks
        self._default_docs = default_sample_documents
        self._min_docs = min_documents
        self._min_chunks = min_chunks
        self._owner = owner or f"taxonomy-job-{uuid.uuid4()}"

    async def run_job(self, job_id: uuid.UUID) -> str:
        """Run the job; returns its final state, or ``busy`` if another worker holds the lease."""
        job = await self._repo.claim(job_id, self._owner, self._lease)
        if job is None:
            current = await self._repo.get(job_id)
            return current.state if current is not None and not current.active else BUSY
        keep_work = False
        try:
            await self._run(job)
        except _Ended:
            pass
        except _Abort as e:
            await self._end(job, FAILED, str(e), spend_used_usd=self._spent(job))
        except (TooFewDocuments, CandidateTaken, CandidateConflict, SpendCapReached) as e:
            await self._end(job, FAILED, str(e), spend_used_usd=self._spent(job))
        except TaxonomyResolutionError as e:
            await self._end(job, FAILED, f"taxonomy unusable: {e}", spend_used_usd=self._spent(job))
        except TransientUploadError:
            keep_work = True  # the build is paid for: the retry reuses it
            raise
        except asyncio.CancelledError:
            keep_work = True  # shutting down: the next attempt resumes and may reuse the build
            raise
        except Exception as e:
            logger.exception("[TaxonomyJob] job %s failed", job.id)
            await self._end(
                job, FAILED, f"{type(e).__name__}: {e}"[:500], spend_used_usd=self._spent(job)
            )
        finally:
            final = await self._repo.get(job.id)
            if not keep_work and final is not None and not final.active:
                shutil.rmtree(self._work / str(job.id), ignore_errors=True)
        final = await self._repo.get(job.id)
        return final.state if final is not None else FAILED

    def _spent(self, job: TaxonomyJobRecord) -> float:
        """What the package spent before the job failed: the cost ledgers under its work directory.

        A failed build loses its result but not its bill; without this the cost of a failed run is
        only an estimate. Costs the ledgers never saw (a call that died mid-way) are not counted.
        """
        total = 0.0
        for ledger in (self._work / str(job.id)).rglob("cost_ledger.json"):
            try:
                total += float(json.loads(ledger.read_text()).get("total_usd", 0.0))
            except (OSError, ValueError):
                continue
        return round(total, 6)

    # ------------------------------------------------------------------- run

    async def _run(self, job: TaxonomyJobRecord) -> None:
        done = await self._candidates.existing_report(job.tenant_id, job.candidate_version)
        if done is not None:
            if done.get("job_id") != str(job.id):
                raise _Abort(f"candidate version {job.candidate_version!r} belongs to another job")
            await self._complete(job, done, spend=float(done.get("cost_usd") or 0.0))
            return
        serving = await self._serving(job) if job.kind == KIND_PROPOSE else None
        await self._checkpoint(job, "sample")
        chunks = await self._sample(job)
        if not chunks:
            await self._complete(
                job, {"note": "no unresolved chunks: nothing to propose"}, 0.0, None
            )
            return
        await self._candidates.claim(job.tenant_id, job.candidate_version, str(job.id), job.kind)
        await self._checkpoint(job, "build")
        output = await self._build(job, chunks, serving)
        await self._checkpoint(job, "upload")
        try:
            report = await self._candidates.publish(
                job.tenant_id, job.candidate_version, str(job.id), output.files, output.report
            )
        except CandidateConflict:
            raise
        except Exception as e:
            raise TransientUploadError(f"upload failed: {type(e).__name__}: {e}") from e
        await self._complete(job, report, output.cost_usd)

    async def _serving(self, job: TaxonomyJobRecord) -> ResolvedTaxonomy:
        taxonomy = await self._resolver.resolve(job.collection_readable_id, fresh=True)
        if taxonomy is None:
            raise _Abort("the collection has no active taxonomy to evolve")
        if taxonomy.tenant_id != job.tenant_id:
            raise _Abort("the collection's taxonomy belongs to another tenant than the job's")
        if taxonomy.migration_id or (
            taxonomy.serving_version and taxonomy.serving_version != taxonomy.taxonomy_version
        ):
            raise _Abort("a migration is in progress: propose after it has ended")
        return taxonomy

    async def _sample(self, job: TaxonomyJobRecord) -> List[Dict[str, Any]]:
        store = self._store_factory()
        try:
            if job.kind == KIND_DISCOVER:
                size = int(job.params.get("sample_size") or self._default_docs)
                return await sample_documents(
                    store,
                    job.collection_id,
                    size,
                    self._max_chunks,
                    self._min_docs,
                    self._min_chunks,
                )
            chunks: List[Dict[str, Any]] = []
            cursor: Optional[str] = None
            while len(chunks) < self._max_chunks:
                items, cursor = await unresolved_page(store, job.collection_id, 500, cursor)
                chunks += [_unresolved_chunk(i) for i in items]
                if cursor is None:
                    break
            return chunks[: self._max_chunks]
        finally:
            await store.close()

    async def _build(
        self,
        job: TaxonomyJobRecord,
        chunks: List[Dict[str, Any]],
        serving: Optional[ResolvedTaxonomy],
    ) -> BuildOutput:
        work = self._work / str(job.id)
        built = _load_build(work)
        if built is not None:
            return built  # an earlier attempt finished the build: the same bytes, no second spend
        work.mkdir(parents=True, exist_ok=True)
        beat = _Heartbeat(self._repo, job.id, self._owner, self._lease)
        beat.start()
        try:
            output = await asyncio.to_thread(self._call_gateway, job, chunks, serving, work)
        finally:
            await beat.stop()
        if beat.lost:
            raise _Ended
        _save_build(work, output)
        if beat.cancelled:
            await self._end(job, CANCELLED, None)
            raise _Ended
        return output

    def _call_gateway(
        self,
        job: TaxonomyJobRecord,
        chunks: List[Dict[str, Any]],
        serving: Optional[ResolvedTaxonomy],
        work: Path,
    ) -> BuildOutput:
        if job.kind == KIND_DISCOVER:
            return self._gateway.discover(
                chunks,
                tenant_id=job.tenant_id,
                candidate_version=job.candidate_version,
                work_dir=work,
                spend_cap_usd=job.spend_cap_usd,
                corpus_description=job.params.get("corpus_description"),
            )
        assert serving is not None
        return self._gateway.propose(
            chunks,
            serving=serving,
            candidate_version=job.candidate_version,
            work_dir=work,
            spend_cap_usd=job.spend_cap_usd,
        )

    # ------------------------------------------------------------------- end

    async def _checkpoint(self, job: TaxonomyJobRecord, phase: str) -> None:
        saved = await self._repo.renew(job.id, self._owner, self._lease, phase=phase)
        if saved is None:
            raise _Ended
        if saved.cancel_requested:
            await self._end(job, CANCELLED, None)
            raise _Ended

    async def _complete(
        self,
        job: TaxonomyJobRecord,
        report: Dict[str, Any],
        spend: float,
        folder: Optional[str] = "",
    ) -> None:
        prefix = (
            self._candidates.folder(job.tenant_id, job.candidate_version)
            if folder == ""
            else folder
        )
        result = {
            "candidate_prefix": prefix,
            "candidate_version": job.candidate_version,
            "categories": report.get("categories"),
            "files": [f["file"] for f in report.get("files", [])],
            "cost_usd": spend,
            "note": report.get("note"),
        }
        await self._end(job, COMPLETED, None, result=result, spend_used_usd=spend)

    async def _end(
        self, job: TaxonomyJobRecord, state: str, error: Optional[str], **changes: Any
    ) -> None:
        done = await self._repo.finish(job.id, self._owner, state, error, **changes)
        if done is not None:
            logger.info("[TaxonomyJob] job %s (%s) %s %s", job.id, job.kind, state, error or "")


class _Heartbeat:
    """Renews the lease while the package runs on a thread; notes a lost lease or a cancel."""

    def __init__(self, repo: TaxonomyJobRepository, job_id: uuid.UUID, owner: str, lease: int):
        self._repo, self._job, self._owner, self._lease = repo, job_id, owner, lease
        self.lost = False
        self.cancelled = False
        self._task: Optional[asyncio.Task] = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(max(1.0, self._lease / 3))
            saved = await self._repo.renew(self._job, self._owner, self._lease, phase="build")
            if saved is None:
                self.lost = True
                return
            if saved.cancel_requested:
                self.cancelled = True


def _unresolved_chunk(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "chunk_id": item["chunk_id"],
        "document_id": item["document_id"],
        "text": item["text"],
        "metadata": {
            "source": item.get("source_name") or "n/a",
            "entity_type": item.get("entity_type") or "n/a",
        },
    }


def _save_build(work: Path, output: BuildOutput) -> None:
    out = work / "out"
    out.mkdir(parents=True, exist_ok=True)
    for name, data in output.files.items():
        (out / name).write_bytes(data)
    (work / "build.json").write_text(
        json.dumps(
            {
                "report": output.report,
                "cost_usd": output.cost_usd,
                "categories": output.categories,
                "notes": output.notes,
                "files": sorted(output.files),
            }
        )
    )
    (work / "built.ok").write_text("ok")


def _load_build(work: Path) -> Optional[BuildOutput]:
    if not (work / "built.ok").exists():
        return None
    meta = json.loads((work / "build.json").read_text())
    files = {name: (work / "out" / name).read_bytes() for name in meta["files"]}
    return BuildOutput(files, meta["report"], meta["cost_usd"], meta["categories"], meta["notes"])


__all__ = ["BUSY", "TaxonomyJobRunner", "to_package_chunk"]
