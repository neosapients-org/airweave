"""Chunk position, page and label fields on search results (Airweave A2, contract 4.4 and 5.2).

No Vespa here: hits are built as Vespa returns them and converted by VespaVectorDB.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from airweave.core.logging import logger
from airweave.domains.search.adapters.vector_db.filter_translator import FilterTranslator
from airweave.domains.search.adapters.vector_db.vespa_client import VespaVectorDB
from airweave.domains.search.types.results import CHUNK_RESULT_FIELDS


def record_as_vespa_hit(record: dict, weighted_set_as_list: bool = False) -> dict:
    """Render a contract search record as the Vespa hit it is read from (flattened metadata)."""
    fields = {k: v for k, v in record.items() if k not in ("relevance_score",)}
    meta = fields.pop("airweave_system_metadata")
    fields.update({f"airweave_system_metadata_{k}": v for k, v in meta.items()})
    for ts in ("created_at", "updated_at"):
        fields.pop(ts, None)
    fields["payload"] = json.dumps({"web_url": "https://drive.example.com/acme"})
    if weighted_set_as_list:
        fields["topic_ids"] = [{"item": k, "weight": v} for k, v in record["topic_ids"].items()]
    return {"relevance": record["relevance_score"], "fields": fields}


@pytest.fixture
def client() -> VespaVectorDB:
    ctx_logger = logger.with_context(request_id="test")
    return VespaVectorDB(
        app=MagicMock(), logger=ctx_logger, filter_translator=FilterTranslator(logger=ctx_logger)
    )


def _old_hit() -> dict:
    """A hit written before the new fields existed (or with the upstream chunk settings)."""
    return {
        "relevance": 0.5,
        "fields": {
            "entity_id": "g1__chunk_0",
            "name": "Globex memo.txt",
            "textual_representation": "Globex memo",
            "airweave_system_metadata_source_name": "google_drive",
            "airweave_system_metadata_entity_type": "GoogleDriveFileEntity",
            "airweave_system_metadata_original_entity_id": "g1",
            "airweave_system_metadata_chunk_index": 0,
            "payload": '{"web_url": "https://drive.example.com/globex"}',
        },
    }


class TestContractRecord:
    @pytest.mark.parametrize("as_list", [False, True])
    def test_record_reads_back_unchanged(self, client, as_list, airweave_neo_search_record):
        """The contract's 4.4 record, read from a Vespa hit, serializes to the same JSON.

        Timestamps are left out: Vespa stores epoch seconds and the reader returns them as
        naive local datetimes (upstream behaviour, unchanged here).
        """
        record = airweave_neo_search_record
        hit = record_as_vespa_hit(record, weighted_set_as_list=as_list)
        (result,) = client._convert_hits_to_results([hit]).results
        dumped = result.model_dump(mode="json")
        for key, value in record.items():
            if key in ("created_at", "updated_at"):
                continue
            if key == "airweave_system_metadata":
                assert {k: dumped[key][k] for k in value} == value
            else:
                assert dumped[key] == value, key

    def test_every_contract_field_returned(self, client, airweave_neo_search_record):
        hit = record_as_vespa_hit(airweave_neo_search_record)
        (result,) = client._convert_hits_to_results([hit]).results
        assert set(CHUNK_RESULT_FIELDS) <= set(result.model_dump(mode="json"))


class TestOldRecords:
    def test_old_record_still_reads(self, client):
        (result,) = client._convert_hits_to_results([_old_hit()]).results
        assert result.entity_id == "g1__chunk_0"
        assert all(getattr(result, name) is None for name in CHUNK_RESULT_FIELDS)

    def test_old_record_serializes_as_before(self, client):
        """Unset fields are left out, so the response for such a hit is the upstream one."""
        (result,) = client._convert_hits_to_results([_old_hit()]).results
        assert not set(CHUNK_RESULT_FIELDS) & set(result.model_dump(mode="json"))
        assert not set(CHUNK_RESULT_FIELDS) & set(result.model_dump())

    def test_partial_fields(self, client):
        """A chunk written with positions but not yet labelled carries only what it has."""
        hit = _old_hit()
        hit["fields"].update({"chunk_char_start": 0, "chunk_char_end": 9, "page_from": 1})
        (result,) = client._convert_hits_to_results([hit]).results
        dumped = result.model_dump(mode="json")
        assert dumped["chunk_char_start"] == 0 and dumped["page_from"] == 1
        assert "topic_ids" not in dumped and "taxonomy_status" not in dumped

    def test_empty_topic_set(self, client):
        hit = _old_hit()
        hit["fields"]["topic_ids"] = {}
        (result,) = client._convert_hits_to_results([hit]).results
        assert result.model_dump(mode="json")["topic_ids"] == {}
