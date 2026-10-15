"""Vespa document IDs must be scoped to their collection.

Regression coverage for cross-collection document overwrites. Source object ids
are only unique within their source, so two collections syncing the same
Confluence space / Drive folder emit identical ``entity_id`` values. When the
Vespa document ID omits the collection, both collections address one document
and the later sync silently takes ownership of it -- including of
``airweave_system_metadata_collection_id``. The first collection then returns
zero search results while its Postgres rows and its sync jobs both still report
success.

These assert through ``EntityTransformer.transform()`` rather than the private
id helper, so reverting the call site in ``transform()`` fails them.
"""

from uuid import UUID

import pytest

from airweave.platform.destinations.vespa.transformer import EntityTransformer
from airweave.platform.entities._airweave_field import AirweaveField
from airweave.platform.entities._base import BaseEntity

COLLECTION_A = UUID("12345678-1234-1234-1234-123456789abc")
COLLECTION_B = UUID("87654321-4321-4321-4321-cba987654321")

# A real colliding pair observed in dev: this one Confluence page id was claimed
# by twelve different syncs across twelve different collections, while Vespa held
# a single copy owned by whichever synced last.
ENTITY_ID = "1048577__chunk_0"


class FakePageEntity(BaseEntity):
    """Stand-in for a source entity, with the flags BaseEntity validation requires."""

    content_id: str = AirweaveField(
        ..., description="Source page id.", embeddable=False, is_entity_id=True
    )
    title: str = AirweaveField(..., description="Page title.", embeddable=True, is_name=True)


def _entity(entity_id: str = ENTITY_ID) -> BaseEntity:
    """Build a minimal entity carrying only what the id path needs.

    ``entity_id`` is set directly because the pipeline has already populated it
    from the flagged field (and appended any ``__chunk_N`` suffix) by the time
    the transformer runs.
    """
    return FakePageEntity(
        entity_id=entity_id,
        content_id=entity_id,
        title="Session Storage",
        breadcrumbs=[],
    )


@pytest.fixture
def transform_a():
    """Transform an entity as collection A."""
    return EntityTransformer(collection_id=COLLECTION_A).transform


@pytest.fixture
def transform_b():
    """Transform an entity as collection B."""
    return EntityTransformer(collection_id=COLLECTION_B).transform


def test_doc_id_is_prefixed_with_collection_id(transform_a):
    """The collection UUID is part of the document identity, not just a field."""
    doc = transform_a(_entity())

    assert doc.id.startswith(f"{COLLECTION_A}_")


def test_same_entity_in_two_collections_gets_distinct_doc_ids(transform_a, transform_b):
    """Two collections syncing the same source object must not collide.

    This is the defect itself: with an unscoped ``{entity_type}_{entity_id}``
    id these two are byte-identical, so the later writer overwrites the earlier
    one and the earlier collection's searches go empty.
    """
    doc_a = transform_a(_entity())
    doc_b = transform_b(_entity())

    assert doc_a.id != doc_b.id


def test_collection_scoping_does_not_disturb_the_filter_field(transform_a):
    """The filterable collection field still matches the id's collection.

    The id and the field are written from the same source, so a search filtered
    on the field resolves to documents this collection actually owns.
    """
    doc = transform_a(_entity())

    assert doc.fields["airweave_system_metadata_collection_id"] == str(COLLECTION_A)


def test_chunks_of_one_entity_stay_individually_addressable(transform_a):
    """Per-chunk documents must not collapse into each other."""
    chunk_0 = transform_a(_entity("1048577__chunk_0"))
    chunk_1 = transform_a(_entity("1048577__chunk_1"))

    assert chunk_0.id.endswith("__chunk_0")
    assert chunk_1.id.endswith("__chunk_1")
    assert chunk_0.id != chunk_1.id


def test_distinct_entities_within_a_collection_stay_distinct(transform_a):
    """Scoping by collection must not collapse different entities together."""
    first = transform_a(_entity("1048577"))
    second = transform_a(_entity("425986"))

    assert first.id != second.id


def test_no_collection_id_falls_back_to_unscoped_id():
    """A transformer with no collection emits no literal ``None`` prefix.

    Interpolating a missing collection would give every such document the same
    ``"None"`` prefix, reintroducing the collision this scoping prevents.
    """
    doc = EntityTransformer(collection_id=None).transform(_entity())

    assert "None" not in doc.id
    assert not doc.id.startswith("_")
