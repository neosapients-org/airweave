"""Tests for SyncLifecycleEvent — focused on the failed-event payload.

We only test SyncLifecycleEvent.failed here, since the other factories
are mechanical and already exercised by callers. The failed factory has
extra semantics: it carries optional error_category so webhook
subscribers can distinguish classified user errors (NEEDS_REAUTH,
billing) from real outages without parsing the free-text error string.
"""

from uuid import uuid4

import pytest

from airweave.core.events.enums import SyncEventType
from airweave.core.events.sync import SyncLifecycleEvent
from airweave.core.shared_models import SourceConnectionErrorCategory


def _ids() -> dict:
    return {
        "organization_id": uuid4(),
        "sync_id": uuid4(),
        "sync_job_id": uuid4(),
        "collection_id": uuid4(),
        "source_connection_id": uuid4(),
    }


def test_failed_without_error_category_omits_field():
    """Backward-compat: callers that don't pass error_category get None."""
    event = SyncLifecycleEvent.failed(
        **_ids(),
        source_type="github",
        collection_name="my-collection",
        collection_readable_id="my-collection",
        error="boom",
    )

    assert event.event_type == SyncEventType.FAILED
    assert event.error == "boom"
    assert event.error_category is None


@pytest.mark.parametrize(
    "category, expected_value",
    [
        (
            SourceConnectionErrorCategory.OAUTH_CREDENTIALS_EXPIRED,
            "oauth_credentials_expired",
        ),
        (
            SourceConnectionErrorCategory.API_KEY_INVALID,
            "api_key_invalid",
        ),
        (
            SourceConnectionErrorCategory.AUTH_PROVIDER_ACCOUNT_GONE,
            "auth_provider_account_gone",
        ),
        (
            SourceConnectionErrorCategory.AUTH_PROVIDER_CREDENTIALS_INVALID,
            "auth_provider_credentials_invalid",
        ),
        (
            SourceConnectionErrorCategory.USAGE_LIMIT_EXCEEDED,
            "usage_limit_exceeded",
        ),
        (
            SourceConnectionErrorCategory.RATE_LIMITED,
            "rate_limited",
        ),
    ],
    ids=lambda v: v if isinstance(v, str) else v.name,
)
def test_failed_serializes_each_error_category(category, expected_value):
    """Each SourceConnectionErrorCategory serializes to its string value.

    External webhook consumers rely on stable string identifiers in the payload.
    """
    event = SyncLifecycleEvent.failed(
        **_ids(),
        source_type="github",
        collection_name="my-collection",
        collection_readable_id="my-collection",
        error="boom",
        error_category=category,
    )
    assert event.error_category == expected_value


def test_failed_event_serializes_to_dict_with_error_category():
    """Pydantic dump includes error_category — what Svix actually sends."""
    event = SyncLifecycleEvent.failed(
        **_ids(),
        source_type="github",
        collection_name="my-collection",
        collection_readable_id="my-collection",
        error="JWT expired",
        error_category=SourceConnectionErrorCategory.OAUTH_CREDENTIALS_EXPIRED,
    )
    payload = event.model_dump(mode="json")
    assert payload["error"] == "JWT expired"
    assert payload["error_category"] == "oauth_credentials_expired"
    assert payload["event_type"] == SyncEventType.FAILED.value
