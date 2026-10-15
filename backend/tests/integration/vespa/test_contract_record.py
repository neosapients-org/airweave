"""Feed the contract's chunk record to Vespa and read it back (Airweave A2, contract 4.4).

See conftest.py: needs a disposable Vespa, skipped otherwise.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from airweave.domains.search.types.filters import FilterCondition, FilterGroup
from airweave.domains.search.types.results import CHUNK_RESULT_FIELDS
from airweave.platform.destinations.vespa.transformer import EntityTransformer
from airweave.platform.entities._base import AirweaveSystemMetadata, Breadcrumb
from airweave.platform.entities.stub import PdfStubFileEntity


def _entity_from_record(record: dict, entity_id: str, with_fields: bool) -> PdfStubFileEntity:
    meta = record["airweave_system_metadata"]
    entity = PdfStubFileEntity(
        stub_id=entity_id,
        file_name=record["name"],
        breadcrumbs=[Breadcrumb(**b) for b in record["breadcrumbs"]],
        url="stub://acme",
        size=1,
        file_type="pdf",
        created_at=datetime(2026, 7, 2, 10, 15, tzinfo=timezone.utc),
        textual_representation=record["textual_representation"],
        airweave_system_metadata=AirweaveSystemMetadata(
            entity_type=meta["entity_type"],
            source_name=meta["source_name"],
            sync_id=UUID("5c0b9a6e-0000-4000-8000-000000000001"),
            sync_job_id=UUID("9e7d1c2a-0000-4000-8000-000000000002"),
            original_entity_id=entity_id,
            chunk_index=meta["chunk_index"],
            hash="h-1",
        ),
    )
    if with_fields:
        for name in CHUNK_RESULT_FIELDS:
            setattr(entity, name, record[name])
        entity.taxonomy_piece_count, entity.taxonomy_admin_pieces = 1, 0
        entity.taxonomy_unresolved_pieces = 0
    return entity


def _by_entity_id(entity_id: str) -> list:
    return [
        FilterGroup(
            conditions=[FilterCondition(field="entity_id", operator="equals", value=entity_id)]
        )
    ]


async def test_record_feeds_and_reads_back(
    airweave_neo_search_record, collection_id, feed, vector_db
):
    record = airweave_neo_search_record
    entity = _entity_from_record(record, "acme-q3__chunk_37", with_fields=True)
    feed([EntityTransformer(collection_id=UUID(collection_id)).transform(entity)])

    (result,) = await vector_db.filter_search(_by_entity_id(entity.entity_id), collection_id)
    dumped = result.model_dump(mode="json")
    for name in CHUNK_RESULT_FIELDS:
        assert dumped[name] == record[name], name
    assert dumped["textual_representation"] == record["textual_representation"]
    assert dumped["breadcrumbs"] == record["breadcrumbs"]


async def test_old_record_without_fields_still_reads(
    airweave_neo_search_record, collection_id, feed, vector_db
):
    entity = _entity_from_record(airweave_neo_search_record, "globex__chunk_0", with_fields=False)
    feed([EntityTransformer(collection_id=UUID(collection_id)).transform(entity)])

    (result,) = await vector_db.filter_search(_by_entity_id(entity.entity_id), collection_id)
    assert result.entity_id == "globex__chunk_0"
    assert not set(CHUNK_RESULT_FIELDS) & set(result.model_dump(mode="json"))
