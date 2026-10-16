"""Chunk position, page and label fields on entities and in the Vespa transformer (Airweave A2).

Contract airweave-neo/1, section 4.2. With the upstream chunk settings none of these fields is
set, and nothing about the Vespa document, the payload, the keyword input or the hash changes.
"""

import json
from uuid import UUID

import pytest

from airweave.domains.sync_pipeline.pipeline.hash_computer import HashComputer
from airweave.platform.destinations.vespa.transformer import CHUNK_LAYOUT_FIELDS, EntityTransformer
from airweave.platform.entities._base import AirweaveSystemMetadata, Breadcrumb
from airweave.platform.entities.stub import PdfStubFileEntity, SmallStubEntity

LABELS = {
    "chunk_char_start": 48210,
    "chunk_char_end": 49402,
    "chunk_own_start": 48391,
    "chunk_body_offset": 79,
    "page_from": 12,
    "page_to": 13,
    "topic_ids": {"acme:credit.loan_credit_quality": 1},
    "broad_family_ids": ["acme:credit"],
    "taxonomy_spans": '[{"c":"acme:credit.loan_credit_quality","s":48391,"e":49402,"p":[0,0]}]',
    "taxonomy_status": "ASSIGNED",
    "taxonomy_version": "acme-2026.10",
    "taxonomy_piece_count": 1,
    "taxonomy_admin_pieces": 0,
    "taxonomy_unresolved_pieces": 0,
}


def _pdf_entity(**chunk_fields) -> PdfStubFileEntity:
    entity = PdfStubFileEntity(
        stub_id="acme-q3",
        file_name="Q3 credit risk report.pdf",
        breadcrumbs=[Breadcrumb(entity_id="d41", name="Risk", entity_type="StubContainerEntity")],
        url="stub://file/pdf/acme-q3",
        size=1024,
        file_type="pdf",
        mime_type="application/pdf",
        textual_representation="Q3 credit risk report.pdf · stub · Risk\n…late payments rose…",
        airweave_system_metadata=AirweaveSystemMetadata(
            entity_type="PdfStubFileEntity",
            source_name="stub",
            sync_id=UUID("5c0b9a6e-0000-4000-8000-000000000001"),
            sync_job_id=UUID("9e7d1c2a-0000-4000-8000-000000000002"),
            hash="h-1",
            chunk_index=37,
            original_entity_id="acme-q3",
        ),
    )
    for name, value in chunk_fields.items():
        setattr(entity, name, value)
    return entity


class TestEntityFields:
    def test_fields_default_to_none(self):
        entity = _pdf_entity()
        assert all(getattr(entity, name) is None for name in CHUNK_LAYOUT_FIELDS)

    def test_fields_never_dumped(self):
        """exclude=True: no dump (keyword input, payload, raw-data store) ever carries them."""
        entity = _pdf_entity(**LABELS)
        for dump in (
            entity.model_dump(),
            entity.model_dump(mode="json"),
            entity.model_dump(mode="json", exclude={"airweave_system_metadata"}),
        ):
            assert not set(CHUNK_LAYOUT_FIELDS) & set(dump)

    def test_fields_survive_model_copy(self):
        """Chunk entities are deep copies of their parent; the copy keeps what was set."""
        copy = _pdf_entity(**LABELS).model_copy(deep=True)
        assert {name: getattr(copy, name) for name in CHUNK_LAYOUT_FIELDS} == LABELS

    async def test_hash_ignores_fields(self):
        """Labels on an entity never change its content hash."""
        plain = SmallStubEntity(
            stub_id="s1", title="Acme note", content="Globex invoice paid.", breadcrumbs=[]
        )
        labelled = plain.model_copy(deep=True)
        for name, value in LABELS.items():
            setattr(labelled, name, value)
        computer = HashComputer()
        assert await computer.compute_for_entity(plain) == await computer.compute_for_entity(
            labelled
        )


class TestTransformer:
    @pytest.fixture
    def transformer(self):
        return EntityTransformer(collection_id=UUID("12345678-1234-1234-1234-123456789abc"))

    def test_maps_every_field(self, transformer):
        doc = transformer.transform(_pdf_entity(**LABELS))
        assert {name: doc.fields[name] for name in CHUNK_LAYOUT_FIELDS} == LABELS

    def test_unset_fields_absent(self, transformer):
        doc = transformer.transform(_pdf_entity())
        assert not set(CHUNK_LAYOUT_FIELDS) & set(doc.fields)

    def test_document_unchanged_without_fields(self, transformer):
        """With the fields unset, the Vespa document is the same as before they existed."""
        with_labels = transformer.transform(_pdf_entity(**LABELS)).fields
        without = transformer.transform(_pdf_entity()).fields
        assert {k: v for k, v in with_labels.items() if k not in CHUNK_LAYOUT_FIELDS} == without

    def test_fields_never_in_payload(self, transformer):
        doc = transformer.transform(_pdf_entity(**LABELS))
        payload = json.loads(doc.fields.get("payload", "{}"))
        assert not set(CHUNK_LAYOUT_FIELDS) & set(payload)

    def test_every_schema_gets_fields(self, transformer):
        """Not only files: any entity type carries the fields (contract 4.2)."""
        entity = SmallStubEntity(
            stub_id="s1", title="Acme note", content="Globex invoice paid.", breadcrumbs=[]
        )
        entity.page_from, entity.taxonomy_status = 3, "NOT_LABELLED"
        doc = transformer.transform(entity)
        assert doc.schema_name == "base_entity"
        assert doc.fields["page_from"] == 3
        assert doc.fields["taxonomy_status"] == "NOT_LABELLED"

    def test_empty_labels_written(self, transformer):
        """A labelled chunk with no category writes empty topic and family sets."""
        doc = transformer.transform(_pdf_entity(topic_ids={}, broad_family_ids=[]))
        assert doc.fields["topic_ids"] == {}
        assert doc.fields["broad_family_ids"] == []
