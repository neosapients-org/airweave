"""Lone surrogates in the Vespa field sanitizer (contract airweave-neo/1, C19; batch 1 item 18).

Vespa refuses a lone surrogate (and UTF-8 cannot even encode one), so the write would fail;
the sanitizer now removes them like the other refused characters. Emoji, which a Python str
holds as one code point, are kept.
"""

import json
from uuid import UUID

import pytest

from airweave.platform.destinations.vespa.transformer import (
    EntityTransformer,
    _sanitize_for_vespa,
)
from airweave.platform.entities._base import AirweaveSystemMetadata
from airweave.platform.entities.stub import SmallStubEntity


@pytest.mark.parametrize("ch", ["\ud800", "\udbff", "\udc00", "\udfff"])
def test_lone_surrogate_removed(ch):
    assert _sanitize_for_vespa(f"Acme{ch} report") == "Acme report"


def test_emoji_and_other_text_kept():
    text = "Globex 📈 Zürich 売上高 \ue000 tab\there"
    assert _sanitize_for_vespa(text) == text


def test_existing_classes_still_removed():
    assert _sanitize_for_vespa("a\x03b\ufdd0c\U0010ffffd") == "abcd"


def test_document_with_surrogate_in_name_is_encodable():
    """The transformed document can be serialised as UTF-8 JSON for the feed."""
    entity = SmallStubEntity(
        stub_id="s1", title="Initech\udc80 memo", content="Paid.", breadcrumbs=[]
    )
    entity.entity_id, entity.name = "s1", "Initech\udc80 memo"
    entity.textual_representation = "Initech\ud800 memo text"
    entity.airweave_system_metadata = AirweaveSystemMetadata(source_name="stub")
    doc = EntityTransformer(collection_id=UUID(int=1)).transform(entity)
    assert doc.fields["name"] == "Initech memo"
    assert doc.fields["textual_representation"] == "Initech memo text"
    json.dumps(doc.fields, ensure_ascii=False).encode("utf-8")
