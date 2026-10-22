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
    "noncharacter FDDF": "\ufddf",
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


def _status(vespa: str, cid: str, cp: int) -> int:
    return _feed_raw(vespa, cid, chr(cp)).status_code


def test_every_c0_code_point_matches_the_list(vespa_deployed, collection_id):
    """All of U+0000 to U+001F, one by one: only tab, line feed and carriage return are kept."""
    for cp in range(0x20):
        expected = 200 if cp in (0x09, 0x0A, 0x0D) else 400
        assert _status(vespa_deployed, collection_id, cp) == expected, hex(cp)
        cleaned = replace_refused_characters(chr(cp))
        assert cleaned == (chr(cp) if expected == 200 else " "), hex(cp)


def test_every_c1_code_point_and_del_is_kept(vespa_deployed, collection_id):
    """U+007F and all of U+0080 to U+009F are accepted by Vespa and kept by the cleaner."""
    for cp in range(0x7F, 0xA0):
        assert _status(vespa_deployed, collection_id, cp) == 200, hex(cp)
        assert replace_refused_characters(chr(cp)) == chr(cp), hex(cp)


def test_noncharacter_block_refused_by_vespa_is_fdd0_to_fddf(vespa_deployed, collection_id):
    """Measured on Vespa 8: U+FDD0 to U+FDDF are refused, U+FDE0 to U+FDEF are accepted."""
    for cp in range(0xFDD0, 0xFDE0):
        assert _status(vespa_deployed, collection_id, cp) == 400, hex(cp)
    for cp in range(0xFDE0, 0xFDF0):
        assert _status(vespa_deployed, collection_id, cp) == 200, hex(cp)


def test_cleaner_also_replaces_fde0_to_fdef_which_vespa_would_accept(vespa_deployed, collection_id):
    """C19 lists all 32 Unicode noncharacters; 16 of them Vespa takes.

    Replacing them is a harmless superset (one space for one code point, so every offset holds).
    """
    for cp in range(0xFDD0, 0xFDF0):
        assert replace_refused_characters(chr(cp)) == " ", hex(cp)


def test_every_plane_noncharacter_pair_is_refused(vespa_deployed, collection_id):
    for plane in range(0x11):
        for low in (0xFFFE, 0xFFFF):
            cp = plane * 0x10000 + low
            assert _status(vespa_deployed, collection_id, cp) == 400, hex(cp)
            assert replace_refused_characters(chr(cp)) == " ", hex(cp)


def test_the_whole_surrogate_block_is_refused(vespa_deployed, collection_id):
    for cp in (0xD800, 0xDB7F, 0xDBFF, 0xDC00, 0xDFFF):
        assert _status(vespa_deployed, collection_id, cp) == 400, hex(cp)
        assert replace_refused_characters(chr(cp)) == " ", hex(cp)
