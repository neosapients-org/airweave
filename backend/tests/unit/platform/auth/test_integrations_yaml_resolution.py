"""Unit tests for _resolve_integrations_yaml.

Resolution runs at import time, so an unresolvable path takes the whole app
down before it serves a request. These tests pin the behaviour for the shipped
environments and for a deployment environment with no dedicated YAML (staging).
"""

from unittest.mock import patch

import pytest

from airweave.platform.auth.settings import _resolve_integrations_yaml, parent_directory

DEV_YAML = "yaml/dev.integrations.yaml"


@pytest.mark.parametrize("environment", ["dev", "prd", "self-hosted"])
def test_uses_dedicated_file_when_one_ships(environment: str) -> None:
    """An environment with its own YAML resolves to that file."""
    resolved = _resolve_integrations_yaml(environment)

    assert resolved == parent_directory / f"yaml/{environment}.integrations.yaml"
    assert resolved.is_file()


def test_local_borrows_dev_file() -> None:
    """`local` has no YAML of its own and has always used dev's."""
    assert _resolve_integrations_yaml("local") == parent_directory / DEV_YAML


@pytest.mark.parametrize("environment", ["staging", "qa", "tenant-acme"])
def test_environment_without_a_file_falls_back_to_dev(environment: str) -> None:
    """An environment with no YAML resolves to dev's instead of a missing path.

    Regression guard: this used to build `yaml/<env>.integrations.yaml`
    unconditionally, so deploying with ENVIRONMENT=staging raised
    FileNotFoundError while importing the module.
    """
    resolved = _resolve_integrations_yaml(environment)

    assert resolved == parent_directory / DEV_YAML
    assert resolved.is_file()


def test_every_supported_environment_resolves_to_a_readable_file() -> None:
    """Whatever is returned must be loadable — that is the point of the helper."""
    for environment in ("local", "dev", "prd", "self-hosted", "staging"):
        assert _resolve_integrations_yaml(environment).is_file()


def test_fallback_is_warned_about() -> None:
    """Falling back is logged, so a typo'd ENVIRONMENT stays visible.

    The module logger sets propagate=False, so caplog cannot see it — assert on
    the logger itself instead.
    """
    with patch("airweave.platform.auth.settings.logger") as mock_logger:
        _resolve_integrations_yaml("staging")

    mock_logger.warning.assert_called_once()
    message = mock_logger.warning.call_args.args[0]
    assert "staging.integrations.yaml" in message
    assert "dev.integrations.yaml" in message


@pytest.mark.parametrize("environment", ["dev", "local", "prd", "self-hosted"])
def test_expected_environments_resolve_quietly(environment: str) -> None:
    """The environments we ship for must not warn."""
    with patch("airweave.platform.auth.settings.logger") as mock_logger:
        _resolve_integrations_yaml(environment)

    mock_logger.warning.assert_not_called()
