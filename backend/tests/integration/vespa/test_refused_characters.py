"""Probe which code points Vespa refuses in a string field (Airweave A4/A11, contract 3.2, C19).

Each class is fed raw through the document API, bypassing every cleaning step. A refused class
that Vespa accepts, or a kept one it refuses, means the list in contract 3.2 must change.
See conftest.py: needs a disposable Vespa, skipped otherwise.
"""

from __future__ import annotations

import json
from urllib.parse import quote
from uuid import uuid4

import httpx
import pytest

from airweave.domains.sync_pipeline.processors.chunk_layout import replace_refused_characters

REFUSED = {
    "C0 NUL": "\x00",
    "C0 ETX": "\x03",
    "C0 VT": "\x0b",
    "C0 FF": "\x0c",
    "C0 US": "\x1f",
    "noncharacter FDD0": "\ufdd0",
    "noncharacter FDEF": "\ufdef",
    "noncharacter FFFE": "\ufffe",
    "noncharacter FFFF": "\uffff",
    "noncharacter 1FFFE": "\U0001fffe",
    "noncharacter 10FFFF": "\U0010ffff",
    "lone high surrogate": "\ud800",
    "lone low surrogate": "\udfff",
}
KEPT = {
    "tab": "\t",
    "line feed": "\n",
    "carriage return": "\r",
    "DEL": "\x7f",
    "C1 first (U+0080)": "\x80",
    "C1 NEL (U+0085)": "\x85",
    "C1 last (U+009F)": "\x9f",
    "emoji": "📈",
    "private use (page sentinel)": "\ue000",
}


def _feed_raw(base: str, collection_id: str, text: str) -> httpx.Response:
    doc_id = f"probe-{uuid4()}"
    url = f"{base}/document/v1/airweave/base_entity/docid/{quote(doc_id, safe='')}"
    fields = {
        "entity_id": doc_id,
        "name": "Acme probe",
        "airweave_system_metadata_collection_id": collection_id,
        "textual_representation": f"Acme {text} Globex",
    }
    # json.dumps escapes non-ASCII (lone surrogates included) as \uXXXX, so the raw code
    # point reaches Vespa's JSON parser instead of failing to encode here.
    body = json.dumps({"fields": fields}).encode("ascii")
    return httpx.post(url, content=body, headers={"Content-Type": "application/json"}, timeout=30)


@pytest.mark.parametrize("name", sorted(REFUSED))
def test_refused_class_is_refused_raw_and_accepted_cleaned(vespa_deployed, collection_id, name):
    ch = REFUSED[name]
    assert _feed_raw(vespa_deployed, collection_id, ch).status_code == 400, (
        f"Vespa accepted {name}: remove it from the refused list (contract 3.2)"
    )
    cleaned = replace_refused_characters(ch)
    assert cleaned == " "
    assert _feed_raw(vespa_deployed, collection_id, cleaned).status_code == 200


@pytest.mark.parametrize("name", sorted(KEPT))
def test_kept_class_is_accepted(vespa_deployed, collection_id, name):
    ch = KEPT[name]
    assert replace_refused_characters(ch) == ch
    assert _feed_raw(vespa_deployed, collection_id, ch).status_code == 200, (
        f"Vespa refused {name}: add it to the refused list (contract 3.2, C19)"
    )
