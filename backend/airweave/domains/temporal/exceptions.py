"""Domain exceptions for the Temporal module."""

from temporalio.exceptions import ApplicationError, ApplicationErrorCategory

from airweave.core.exceptions import AirweaveException, InvalidInputError
from airweave.core.shared_models import SourceConnectionErrorCategory

ORPHANED_SYNC_ERROR_TYPE = "OrphanedSyncError"
"""ApplicationError.type value used for orphaned sync detection across the
activity → workflow serialization boundary."""

CLASSIFIED_USER_ERROR_TYPE = "ClassifiedUserError"
"""ApplicationError.type value used for user-actionable sync failures
(expired credentials, usage limits, rate limits) across the activity →
workflow serialization boundary. The activity has already transitioned
the sync job to FAILED with the appropriate error_category, so the
workflow should complete normally rather than re-raise — this prevents
the Temporal workflow failure metric from spiking on customer errors."""


def classified_user_application_error(
    error: Exception,
    category: SourceConnectionErrorCategory,
) -> ApplicationError:
    """Wrap a user-actionable sync error for the activity → workflow boundary.

    Returns an ``ApplicationError`` carrying:
      - ``type=CLASSIFIED_USER_ERROR_TYPE`` so the workflow can detect it
        via the type field without importing this module
      - ``category=BENIGN`` so Temporal does not log it as a system failure
      - ``non_retryable=True`` because user-actionable errors won't resolve
        by retrying the same workflow attempt
      - ``details=[category_value, message]`` for diagnostics

    The activity has already performed the FAILED state transition,
    pause (if applicable), webhook emission, and analytics tracking
    before raising this. The workflow handles it by completing normally
    without an additional transition.
    """
    return ApplicationError(
        str(error),
        category.value,
        str(error),
        type=CLASSIFIED_USER_ERROR_TYPE,
        non_retryable=True,
        category=ApplicationErrorCategory.BENIGN,
    )


class OrphanedSyncError(AirweaveException):
    """Raised when a sync's source connection no longer exists.

    Domain code raises this. The activity boundary layer converts it to an
    explicit ``ApplicationError`` with ``type=ORPHANED_SYNC_ERROR_TYPE``,
    ``non_retryable=True``, and ``category=BENIGN`` so the workflow can
    detect it via the type field without importing this class.
    """

    def __init__(self, sync_id: str, reason: str = "Source connection not found"):
        """Initialize with sync ID and optional reason."""
        self.sync_id = sync_id
        self.reason = reason
        super().__init__(f"Orphaned sync {sync_id}: {reason}")


class InvalidCronExpressionError(InvalidInputError):
    """Raised when a cron expression fails validation.

    Extends InvalidInputError so the API layer's existing exception handler
    translates this to a 422 response automatically.
    """

    def __init__(self, cron_expression: str):
        """Initialize with the invalid cron expression."""
        self.cron_expression = cron_expression
        super().__init__(f"Invalid CRON expression: {cron_expression}")
