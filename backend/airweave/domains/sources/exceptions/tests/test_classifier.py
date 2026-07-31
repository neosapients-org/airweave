"""Tests for classifier.classify_error()."""

import pytest

from airweave.core.shared_models import SourceConnectionErrorCategory
from airweave.domains.auth_provider.exceptions import (
    AuthProviderAccountNotFoundError,
    AuthProviderAuthError,
)
from airweave.domains.sources.exceptions import SourceAuthError, SourceTokenRefreshError
from airweave.domains.sources.exceptions.classifier import classify_error
from airweave.domains.sources.token_providers.exceptions import (
    TokenCredentialsInvalidError,
    TokenExpiredError,
    TokenProviderAccountGoneError,
    TokenProviderServerError,
)
from airweave.domains.sources.token_providers.protocol import AuthProviderKind

# ---------------------------------------------------------------------------
# Legacy SourceAuthError / SourceTokenRefreshError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc, expected_category",
    [
        # SourceTokenRefreshError → OAUTH_CREDENTIALS_EXPIRED
        (
            SourceTokenRefreshError(
                "token expired",
                source_short_name="github",
                token_provider_kind=AuthProviderKind.OAUTH,
            ),
            SourceConnectionErrorCategory.OAUTH_CREDENTIALS_EXPIRED,
        ),
        # SourceAuthError + CREDENTIAL → API_KEY_INVALID
        (
            SourceAuthError(
                "invalid key",
                source_short_name="stripe",
                status_code=401,
                token_provider_kind=AuthProviderKind.CREDENTIAL,
            ),
            SourceConnectionErrorCategory.API_KEY_INVALID,
        ),
        # SourceAuthError + STATIC → API_KEY_INVALID
        (
            SourceAuthError(
                "bad token",
                source_short_name="openai",
                status_code=401,
                token_provider_kind=AuthProviderKind.STATIC,
            ),
            SourceConnectionErrorCategory.API_KEY_INVALID,
        ),
        # SourceAuthError + OAUTH → OAUTH_CREDENTIALS_EXPIRED
        (
            SourceAuthError(
                "oauth failed",
                source_short_name="gmail",
                status_code=401,
                token_provider_kind=AuthProviderKind.OAUTH,
            ),
            SourceConnectionErrorCategory.OAUTH_CREDENTIALS_EXPIRED,
        ),
        # SourceAuthError + AUTH_PROVIDER → AUTH_PROVIDER_CREDENTIALS_INVALID
        (
            SourceAuthError(
                "unauthorized",
                source_short_name="jira",
                status_code=401,
                token_provider_kind=AuthProviderKind.AUTH_PROVIDER,
            ),
            SourceConnectionErrorCategory.AUTH_PROVIDER_CREDENTIALS_INVALID,
        ),
    ],
    ids=[
        "token_refresh_error",
        "credential_api_key",
        "static_api_key",
        "oauth_expired",
        "auth_provider_invalid",
    ],
)
def test_classify_error(exc, expected_category):
    result = classify_error(exc)
    assert result.category == expected_category
    assert result.message is not None


# ---------------------------------------------------------------------------
# TokenProviderError hierarchy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc, expected_category",
    [
        # TokenCredentialsInvalidError + AUTH_PROVIDER → AUTH_PROVIDER_CREDENTIALS_INVALID
        (
            TokenCredentialsInvalidError(
                "Composio API key is invalid or revoked",
                source_short_name="jira",
                provider_kind=AuthProviderKind.AUTH_PROVIDER,
            ),
            SourceConnectionErrorCategory.AUTH_PROVIDER_CREDENTIALS_INVALID,
        ),
        # TokenCredentialsInvalidError + OAUTH → OAUTH_CREDENTIALS_EXPIRED
        (
            TokenCredentialsInvalidError(
                "refresh token revoked",
                source_short_name="gmail",
                provider_kind=AuthProviderKind.OAUTH,
            ),
            SourceConnectionErrorCategory.OAUTH_CREDENTIALS_EXPIRED,
        ),
        # TokenCredentialsInvalidError + CREDENTIAL → API_KEY_INVALID
        (
            TokenCredentialsInvalidError(
                "bad api key",
                source_short_name="stripe",
                provider_kind=AuthProviderKind.CREDENTIAL,
            ),
            SourceConnectionErrorCategory.API_KEY_INVALID,
        ),
        # TokenProviderAccountGoneError → AUTH_PROVIDER_ACCOUNT_GONE
        (
            TokenProviderAccountGoneError(
                "account deleted in composio",
                source_short_name="jira",
                provider_kind=AuthProviderKind.AUTH_PROVIDER,
                account_id="acc-123",
            ),
            SourceConnectionErrorCategory.AUTH_PROVIDER_ACCOUNT_GONE,
        ),
        # TokenExpiredError → OAUTH_CREDENTIALS_EXPIRED
        (
            TokenExpiredError(
                "JWT expired",
                source_short_name="github",
                provider_kind=AuthProviderKind.OAUTH,
            ),
            SourceConnectionErrorCategory.OAUTH_CREDENTIALS_EXPIRED,
        ),
    ],
    ids=[
        "token_creds_auth_provider",
        "token_creds_oauth",
        "token_creds_credential",
        "token_account_gone",
        "token_expired",
    ],
)
def test_classify_token_provider_error(exc, expected_category):
    result = classify_error(exc)
    assert result.category == expected_category
    assert result.message is not None


def test_token_provider_server_error_returns_empty():
    """Non-credential TokenProviderError subclasses are not classified."""
    result = classify_error(
        TokenProviderServerError(
            "upstream 500",
            source_short_name="jira",
            provider_kind=AuthProviderKind.AUTH_PROVIDER,
            status_code=500,
        )
    )
    assert result.category is None
    assert result.message is None


# ---------------------------------------------------------------------------
# Non-auth errors and unwrapping
# ---------------------------------------------------------------------------


def test_classify_error_non_auth_returns_empty():
    result = classify_error(RuntimeError("something broke"))
    assert result.category is None
    assert result.message is None


def test_classify_error_unwraps_chained_source_auth_error():
    """SourceValidationError wrapping SourceAuthError should classify the cause."""
    from airweave.domains.sources.exceptions import SourceValidationError

    cause = SourceAuthError(
        "Bad credentials",
        source_short_name="github",
        status_code=401,
        token_provider_kind=AuthProviderKind.CREDENTIAL,
    )
    wrapper = SourceValidationError(
        short_name="github",
        reason="credential validation failed",
    )
    wrapper.__cause__ = cause

    result = classify_error(wrapper)
    assert result.category == SourceConnectionErrorCategory.API_KEY_INVALID
    assert result.message is not None


def test_classify_error_unwraps_chained_token_provider_error():
    """SourceValidationError wrapping TokenCredentialsInvalidError should classify the cause."""
    from airweave.domains.sources.exceptions import SourceValidationError

    cause = TokenCredentialsInvalidError(
        "Auth provider credentials rejected for jira: Composio API key is invalid",
        source_short_name="jira",
        provider_kind=AuthProviderKind.AUTH_PROVIDER,
    )
    wrapper = SourceValidationError(
        short_name="jira",
        reason="credential validation failed",
    )
    wrapper.__cause__ = cause

    result = classify_error(wrapper)
    assert result.category == SourceConnectionErrorCategory.AUTH_PROVIDER_CREDENTIALS_INVALID
    assert result.message is not None


# ---------------------------------------------------------------------------
# AuthProviderError hierarchy (direct auth provider failures)
# ---------------------------------------------------------------------------


def test_auth_provider_auth_error():
    """AuthProviderAuthError (e.g. Composio API key invalid) → AUTH_PROVIDER_CREDENTIALS_INVALID."""
    result = classify_error(
        AuthProviderAuthError("Composio API key is invalid or revoked", provider_name="composio")
    )
    assert result.category == SourceConnectionErrorCategory.AUTH_PROVIDER_CREDENTIALS_INVALID
    assert "Composio API key is invalid or revoked" in result.message


def test_auth_provider_account_not_found():
    """AuthProviderAccountNotFoundError → AUTH_PROVIDER_ACCOUNT_GONE."""
    result = classify_error(
        AuthProviderAccountNotFoundError(
            "account not found", provider_name="composio", account_id="acc-123"
        )
    )
    assert result.category == SourceConnectionErrorCategory.AUTH_PROVIDER_ACCOUNT_GONE


def test_auth_provider_error_unwrapped_from_cause():
    """Generic exception wrapping AuthProviderAuthError should classify the cause."""
    cause = AuthProviderAuthError("API key revoked", provider_name="composio")
    wrapper = Exception("orchestrator failed")
    wrapper.__cause__ = cause

    result = classify_error(wrapper)
    assert result.category == SourceConnectionErrorCategory.AUTH_PROVIDER_CREDENTIALS_INVALID


# ---------------------------------------------------------------------------
# UsageLimitExceededError → USAGE_LIMIT_EXCEEDED
# ---------------------------------------------------------------------------


def test_usage_limit_exceeded_classified():
    """UsageLimitExceededError (entity cap hit) classifies as USAGE_LIMIT_EXCEEDED."""
    from airweave.domains.usage.exceptions import UsageLimitExceededError

    result = classify_error(
        UsageLimitExceededError(
            action_type="entities",
            limit=50000,
            current_usage=50103,
        )
    )
    assert result.category == SourceConnectionErrorCategory.USAGE_LIMIT_EXCEEDED
    assert result.message is not None
    assert "50000" in result.message


def test_usage_limit_exceeded_unwrapped_from_cause():
    """A wrapper exception with UsageLimitExceededError as __cause__ is classified."""
    from airweave.domains.usage.exceptions import UsageLimitExceededError

    cause = UsageLimitExceededError(action_type="entities", limit=100, current_usage=101)
    wrapper = RuntimeError("guard rail check failed")
    wrapper.__cause__ = cause

    result = classify_error(wrapper)
    assert result.category == SourceConnectionErrorCategory.USAGE_LIMIT_EXCEEDED


# ---------------------------------------------------------------------------
# Rate-limit signals → RATE_LIMITED
# ---------------------------------------------------------------------------


def test_source_rate_limit_error_classified():
    """The canonical 429 SourceRateLimitError classifies as RATE_LIMITED."""
    from airweave.domains.sources.exceptions import SourceRateLimitError

    result = classify_error(SourceRateLimitError(retry_after=60.0, source_short_name="hubspot"))
    assert result.category == SourceConnectionErrorCategory.RATE_LIMITED
    assert result.message is not None


def test_github_403_rate_limit_classified():
    """GitHub returns 403 for rate limits.

    SourceEntityForbiddenError carrying 'rate limit' in its message is
    classified as RATE_LIMITED.
    """
    from airweave.domains.sources.exceptions import SourceEntityForbiddenError

    result = classify_error(
        SourceEntityForbiddenError(
            "Forbidden (403): API rate limit exceeded for user ID 3160046",
            source_short_name="github",
        )
    )
    assert result.category == SourceConnectionErrorCategory.RATE_LIMITED


def test_github_403_abuse_detection_classified():
    """GitHub's other 403 rate-limit variant — 'secondary rate limit' / abuse.

    detection — is also classified as RATE_LIMITED.
    """
    from airweave.domains.sources.exceptions import SourceEntityForbiddenError

    result = classify_error(
        SourceEntityForbiddenError(
            "Forbidden (403): You have triggered an abuse detection mechanism",
            source_short_name="github",
        )
    )
    assert result.category == SourceConnectionErrorCategory.RATE_LIMITED


def test_403_without_rate_limit_message_not_classified():
    """A plain 403 (permission denied, not rate limit) is NOT classified —.

    those represent entity-level permission issues, not connection problems.
    """
    from airweave.domains.sources.exceptions import SourceEntityForbiddenError

    result = classify_error(
        SourceEntityForbiddenError(
            "Forbidden (403): You don't have access to this resource",
            source_short_name="github",
        )
    )
    assert result.category is None
    assert result.message is None
