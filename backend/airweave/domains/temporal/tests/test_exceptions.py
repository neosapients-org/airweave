"""Tests for Temporal domain exceptions."""

import pytest
from temporalio.exceptions import ApplicationError, ApplicationErrorCategory

from airweave.core.shared_models import SourceConnectionErrorCategory
from airweave.domains.temporal.exceptions import (
    CLASSIFIED_USER_ERROR_TYPE,
    OrphanedSyncError,
    classified_user_application_error,
)


@pytest.mark.unit
def test_orphaned_sync_error_init():
    err = OrphanedSyncError("sync-123")
    assert err.sync_id == "sync-123"
    assert err.reason == "Source connection not found"
    assert "Orphaned sync sync-123" in str(err)


@pytest.mark.unit
def test_orphaned_sync_error_custom_reason():
    err = OrphanedSyncError("sync-456", reason="Deleted during execution")
    assert err.sync_id == "sync-456"
    assert err.reason == "Deleted during execution"
    assert "Deleted during execution" in str(err)


# ---------------------------------------------------------------------------
# classified_user_application_error
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_classified_wrapper_sets_type_and_non_retryable():
    """The wrapper marks the error as CLASSIFIED_USER_ERROR_TYPE and.

    non_retryable so the workflow takes the no-op branch instead of
    retrying or counting the error toward temporal_workflow_failed.
    """
    original = RuntimeError("JWT expired")
    wrapped = classified_user_application_error(
        original, SourceConnectionErrorCategory.OAUTH_CREDENTIALS_EXPIRED
    )
    assert isinstance(wrapped, ApplicationError)
    assert wrapped.type == CLASSIFIED_USER_ERROR_TYPE
    assert wrapped.non_retryable is True
    assert wrapped.category == ApplicationErrorCategory.BENIGN


@pytest.mark.unit
def test_classified_wrapper_encodes_category_in_details():
    """The category value goes into details so consumers can inspect it.

    without parsing the message.
    """
    original = RuntimeError("over quota")
    wrapped = classified_user_application_error(
        original, SourceConnectionErrorCategory.USAGE_LIMIT_EXCEEDED
    )
    assert SourceConnectionErrorCategory.USAGE_LIMIT_EXCEEDED.value in wrapped.details


@pytest.mark.unit
@pytest.mark.parametrize(
    "category",
    [
        SourceConnectionErrorCategory.OAUTH_CREDENTIALS_EXPIRED,
        SourceConnectionErrorCategory.API_KEY_INVALID,
        SourceConnectionErrorCategory.AUTH_PROVIDER_ACCOUNT_GONE,
        SourceConnectionErrorCategory.AUTH_PROVIDER_CREDENTIALS_INVALID,
        SourceConnectionErrorCategory.USAGE_LIMIT_EXCEEDED,
        SourceConnectionErrorCategory.RATE_LIMITED,
    ],
)
def test_classified_wrapper_supports_every_category(category):
    """Every defined category produces a valid wrapped error — guards.

    against silently dropping a category if a new one is added.
    """
    wrapped = classified_user_application_error(RuntimeError("x"), category)
    assert wrapped.type == CLASSIFIED_USER_ERROR_TYPE
    assert category.value in wrapped.details


@pytest.mark.unit
def test_classified_wrapper_preserves_message():
    """The wrapped error stringifies to the original error's message."""
    original = ValueError("specific cause text")
    wrapped = classified_user_application_error(
        original, SourceConnectionErrorCategory.API_KEY_INVALID
    )
    assert "specific cause text" in str(wrapped)
