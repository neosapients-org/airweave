"""Topic labels on every chunk, written with the chunk (plan A7; contract C5, C11, C16, C20).

Runs after ``ChunkEmbedProcessor`` and before the write. Per batch:

1. ``TAXONOMY_ENABLED`` off: chunks pass through untouched (the fields stay absent, which
   readers treat as ``NOT_LABELLED``).
2. Resolve the collection's taxonomy from its ``readable_id``. No active file: ``NOT_LABELLED``.
3. Classify each chunk's **own text** (no header, no overlap) on a dedicated classifier thread.
4. Translate build ids to registry ids, shift piece offsets into document coordinates, write.
5. Any error or timeout fails open, visibly: the chunks are written with ``ERROR`` and no labels,
   logged with the sync job, and counted; the sync never stops because of labelling.

Only text chunks with positions (``CHUNKER_MODE=sentence_pack``) can carry labels: spans are
document offsets. Text chunks without positions get ``NOT_LABELLED``; code files are untouched.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from typing import TYPE_CHECKING, Any, Dict, List

from airweave.core.config import settings
from airweave.domains.taxonomy.artifacts import (
    ResolvedTaxonomy,
    TaxonomyResolutionError,
    TaxonomyResolver,
)
from airweave.domains.taxonomy.classifier_pool import ClassifierPool
from airweave.domains.taxonomy.mapping import (
    ERROR,
    NOT_LABELLED,
    ChunkLabels,
    administrative_labels,
    map_record,
)
from airweave.domains.taxonomy.metrics import TAXONOMY_TALLY, TaxonomyTally, format_counts
from airweave.platform.entities._base import BaseEntity, CodeFileEntity

if TYPE_CHECKING:
    from airweave.domains.sync_pipeline.contexts import SyncContext

_LABEL_FIELDS = (
    "topic_ids",
    "broad_family_ids",
    "taxonomy_spans",
    "taxonomy_version",
    "taxonomy_piece_count",
    "taxonomy_admin_pieces",
    "taxonomy_unresolved_pieces",
)


class TaxonomyProcessor:
    """Labels a batch of chunk entities in place; never raises."""

    def __init__(
        self,
        resolver: TaxonomyResolver,
        pool: ClassifierPool,
        timeout_seconds: float,
        tally: TaxonomyTally = TAXONOMY_TALLY,
    ) -> None:
        """Initialize with the taxonomy resolver, the classifier pool and the batch timeout."""
        self._resolver = resolver
        self._pool = pool
        self._timeout = timeout_seconds
        self._tally = tally

    def close(self) -> None:
        """Release the classifier threads and cache files (worker shutdown); blocking."""
        self._pool.close()

    async def process(
        self, chunks: List[BaseEntity], sync_context: "SyncContext"
    ) -> List[BaseEntity]:
        """Set the label fields on every text chunk of the batch; return the same list."""
        if not settings.TAXONOMY_ENABLED or not chunks:
            return chunks
        text_chunks = [c for c in chunks if not isinstance(c, CodeFileEntity)]
        if not text_chunks:
            return chunks
        counts: Counter = Counter()
        try:
            await self._label(text_chunks, sync_context, counts)
        except Exception as e:  # never fail the sync because of labelling (C11)
            sync_context.logger.error(
                f"[Taxonomy] labelling failed, {len(text_chunks)} chunks written as ERROR "
                f"(sync_job_id={sync_context.sync_job_id}): {type(e).__name__}: {e}",
                exc_info=True,
            )
            for chunk in text_chunks:
                _set_status(chunk, ERROR)
            counts = Counter({ERROR: len(text_chunks)})
        self._record(counts, sync_context)
        return chunks

    async def _label(
        self, chunks: List[BaseEntity], sync_context: "SyncContext", counts: Counter
    ) -> None:
        positioned = [c for c in chunks if _has_positions(c)]
        unpositioned = len(chunks) - len(positioned)
        if unpositioned:
            sync_context.logger.warning(
                f"[Taxonomy] {unpositioned} text chunks have no positions "
                f"(CHUNKER_MODE is not sentence_pack): written as NOT_LABELLED"
            )
            for chunk in chunks:
                if not _has_positions(chunk):
                    _set_status(chunk, NOT_LABELLED)
            counts[NOT_LABELLED] += unpositioned
        if not positioned:
            return

        readable_id = sync_context.collection.readable_id
        try:
            taxonomy = await self._resolver.resolve(readable_id)
        except TaxonomyResolutionError as e:
            sync_context.logger.error(
                f"[Taxonomy] collection {readable_id!r}: taxonomy unusable, "
                f"{len(positioned)} chunks written as ERROR "
                f"(sync_job_id={sync_context.sync_job_id}): {e}"
            )
            self._set_all(positioned, ERROR, counts)
            return
        if taxonomy is None:
            self._set_all(positioned, NOT_LABELLED, counts)
            return
        self._tally.note_version(
            str(sync_context.sync_job_id), taxonomy.taxonomy_version, taxonomy.generation
        )
        await self._classify(positioned, taxonomy, sync_context, counts)

    async def _classify(
        self,
        chunks: List[BaseEntity],
        taxonomy: ResolvedTaxonomy,
        sync_context: "SyncContext",
        counts: Counter,
    ) -> None:
        rows: List[Dict[str, Any]] = []
        to_classify: List[BaseEntity] = []
        for chunk in chunks:
            own = _own_text(chunk)
            if own.strip():
                rows.append(
                    {
                        "chunk_id": chunk.entity_id,
                        "document_id": _document_id(chunk),
                        "text": own,
                    }
                )
                to_classify.append(chunk)
            else:  # empty or whitespace-only own text (contract 10)
                _apply(chunk, administrative_labels(), taxonomy.taxonomy_version)
                counts[administrative_labels().taxonomy_status] += 1
        if not rows:
            return
        try:
            records = await self._pool.classify(taxonomy, rows, self._timeout)
        except asyncio.TimeoutError:
            sync_context.logger.error(
                f"[Taxonomy] classification timed out after {self._timeout}s, "
                f"{len(rows)} chunks written as ERROR (sync_job_id={sync_context.sync_job_id})"
            )
            self._set_all(to_classify, ERROR, counts)
            return
        except Exception as e:
            sync_context.logger.error(
                f"[Taxonomy] classification failed, {len(rows)} chunks written as ERROR "
                f"(sync_job_id={sync_context.sync_job_id}, version={taxonomy.taxonomy_version}): "
                f"{type(e).__name__}: {e}"
            )
            self._set_all(to_classify, ERROR, counts)
            return
        self._apply_records(to_classify, records, taxonomy, sync_context, counts)

    def _apply_records(
        self,
        chunks: List[BaseEntity],
        records: List[Dict[str, Any]],
        taxonomy: ResolvedTaxonomy,
        sync_context: "SyncContext",
        counts: Counter,
    ) -> None:
        by_id = {r["chunk_id"]: r for r in records}
        untranslated: set = set()
        for chunk in chunks:
            record = by_id.get(chunk.entity_id)
            if record is None:
                _set_status(chunk, ERROR)
                counts[ERROR] += 1
                continue
            own_start, end = _own_range(chunk)
            labels = map_record(record, taxonomy.registry, own_start, end)
            untranslated.update(labels.untranslated)
            _apply(chunk, labels, taxonomy.taxonomy_version)
            counts[labels.taxonomy_status] += 1
        if untranslated:
            sync_context.logger.error(
                f"[Taxonomy] build ids missing from registry_map.json of "
                f"{taxonomy.taxonomy_version} (labels dropped, pieces counted unresolved): "
                f"{sorted(untranslated)}"
            )

    @staticmethod
    def _set_all(chunks: List[BaseEntity], status: str, counts: Counter) -> None:
        for chunk in chunks:
            _set_status(chunk, status)
        counts[status] += len(chunks)

    def _record(self, counts: Counter, sync_context: "SyncContext") -> None:
        job = str(sync_context.sync_job_id)
        so_far = self._tally.record(job, counts)
        sync_context.logger.debug(f"[Taxonomy] batch: {format_counts(counts)}")
        if self._tally.should_alert(job, so_far):
            sync_context.logger.error(
                f"[Taxonomy] ERROR passed 1% of this sync's chunks so far "
                f"({format_counts(so_far)}, sync_job_id={job})"
            )


def _has_positions(chunk: BaseEntity) -> bool:
    return (
        chunk.chunk_char_start is not None
        and chunk.chunk_char_end is not None
        and chunk.chunk_own_start is not None
    )


def _own_range(chunk: BaseEntity) -> tuple[int, int]:
    """``[chunk_own_start, chunk_char_end)`` of a chunk with positions."""
    if chunk.chunk_own_start is None or chunk.chunk_char_end is None:
        raise ValueError(f"chunk {chunk.entity_id} has no positions")
    return chunk.chunk_own_start, chunk.chunk_char_end


def _own_text(chunk: BaseEntity) -> str:
    """The chunk's own text: no header, no overlap (contract 3.2, C16)."""
    own_start, _ = _own_range(chunk)
    body = (chunk.textual_representation or "")[chunk.chunk_body_offset or 0 :]
    return body[own_start - (chunk.chunk_char_start or 0) :]


def _document_id(chunk: BaseEntity) -> str:
    meta = chunk.airweave_system_metadata
    return (meta.original_entity_id if meta else None) or chunk.entity_id or ""


def _set_status(chunk: BaseEntity, status: str) -> None:
    """A status without labels (ERROR, NOT_LABELLED): every other label field stays unset."""
    for name in _LABEL_FIELDS:
        setattr(chunk, name, None)
    chunk.taxonomy_status = status


def _apply(chunk: BaseEntity, labels: ChunkLabels, version: str) -> None:
    chunk.taxonomy_status = labels.taxonomy_status
    chunk.topic_ids = dict(labels.topic_ids)
    chunk.broad_family_ids = list(labels.broad_family_ids)
    chunk.taxonomy_spans = labels.taxonomy_spans
    chunk.taxonomy_version = version
    chunk.taxonomy_piece_count = labels.taxonomy_piece_count
    chunk.taxonomy_admin_pieces = labels.taxonomy_admin_pieces
    chunk.taxonomy_unresolved_pieces = labels.taxonomy_unresolved_pieces
