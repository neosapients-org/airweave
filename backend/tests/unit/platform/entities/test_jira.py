"""Unit tests for the Jira entity document rendering and source materialization.

These cover the fix that treats Jira issues as document FileEntities: the entity
renders a Markdown document and the source materializes it to a local file via the
FileService so it flows through the standard chunking/classification/search pipeline.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from airweave.platform.entities._base import Breadcrumb
from airweave.platform.entities.jira import JiraIssueEntity
from airweave.platform.sources.jira import JiraSource


def _project_breadcrumb() -> Breadcrumb:
    return Breadcrumb(entity_id="10000", name="Kanban", entity_type="JiraProjectEntity")


def _issue_api_dict(**overrides) -> dict:
    fields = {
        "summary": "Login fails",
        "description": "Users cannot log in after the latest release.",
        "status": {"name": "To Do"},
        "issuetype": {"name": "Bug"},
        "created": "2026-01-01T00:00:00.000+0000",
        "updated": "2026-01-02T00:00:00.000+0000",
    }
    fields.update(overrides.pop("fields", {}))
    data = {"id": "10001", "key": "KAN-6", "fields": fields}
    data.update(overrides)
    return data


# ------------------------------------------------------------------
# from_api()
# ------------------------------------------------------------------


def test_from_api_marks_issue_as_markdown_file_entity():
    """from_api produces a FileEntity with Markdown file metadata and no local_path yet."""
    entity = JiraIssueEntity.from_api(
        _issue_api_dict(),
        project_breadcrumb=_project_breadcrumb(),
        project_key="KAN",
        site_url="https://example.atlassian.net",
    )

    assert entity.file_type == "md"
    assert entity.mime_type == "text/markdown"
    assert entity.local_path is None  # set later by the source via FileService
    assert entity.url == "https://example.atlassian.net/browse/KAN-6"
    assert entity.issue_key == "KAN-6"
    assert entity.status == "To Do"
    assert entity.issue_type == "Bug"


# ------------------------------------------------------------------
# build_document_content()
# ------------------------------------------------------------------


def test_build_document_content_includes_metadata_and_description():
    """The rendered document carries heading, key metadata, and the description body."""
    entity = JiraIssueEntity.from_api(
        _issue_api_dict(),
        project_breadcrumb=_project_breadcrumb(),
        project_key="KAN",
        site_url="https://example.atlassian.net",
    )

    content = entity.build_document_content()

    assert content.startswith("# KAN-6: Login fails")
    assert "**Issue**: KAN-6" in content
    assert "**Project**: KAN" in content
    assert "**Type**: Bug" in content
    assert "**Status**: To Do" in content
    assert "## Description" in content
    assert "Users cannot log in after the latest release." in content


def test_build_document_content_omits_missing_optional_fields():
    """Optional fields that are absent are not rendered (no empty markdown lines)."""
    entity = JiraIssueEntity.from_api(
        _issue_api_dict(fields={"description": None, "status": None, "issuetype": None}),
        project_breadcrumb=_project_breadcrumb(),
        project_key="KAN",
    )

    content = entity.build_document_content()

    assert "**Status**" not in content
    assert "**Type**" not in content
    assert "## Description" not in content
    # Required context is still present.
    assert "**Issue**: KAN-6" in content
    assert "**Project**: KAN" in content


# ------------------------------------------------------------------
# _generate_issue_entities() materialization
# ------------------------------------------------------------------


def _bare_source() -> JiraSource:
    source = JiraSource(auth=AsyncMock(), logger=MagicMock(), http_client=AsyncMock())
    source.site_url = "https://example.atlassian.net"
    source.base_url = "https://api.atlassian.com/ex/jira/cloud-1"
    return source


def _project_entity():
    from airweave.platform.entities.jira import JiraProjectEntity

    return JiraProjectEntity(
        entity_id="10000",
        breadcrumbs=[],
        name="Kanban",
        project_id="10000",
        project_name="Kanban",
        project_key="KAN",
    )


@pytest.mark.asyncio
async def test_generate_issue_entities_materializes_local_path():
    """Each issue is written via FileService.save_bytes and yielded with a local_path."""
    source = _bare_source()
    source._post = AsyncMock(
        return_value={"issues": [_issue_api_dict()], "total": 1, "isLast": True}
    )

    async def save(entity, content, filename_with_extension, logger):
        assert filename_with_extension == "KAN-6.md"
        assert b"Login fails" in content
        entity.local_path = "/tmp/KAN-6.md"
        return entity

    files = MagicMock()
    files.save_bytes = AsyncMock(side_effect=save)

    results = [e async for e in source._generate_issue_entities(_project_entity(), files)]

    assert len(results) == 1
    assert results[0].local_path == "/tmp/KAN-6.md"
    files.save_bytes.assert_called_once()


@pytest.mark.asyncio
async def test_generate_issue_entities_skips_when_no_file_service():
    """Without a FileService the issue cannot be materialized, so it is skipped."""
    source = _bare_source()
    source._post = AsyncMock(
        return_value={"issues": [_issue_api_dict()], "total": 1, "isLast": True}
    )

    results = [e async for e in source._generate_issue_entities(_project_entity(), None)]

    assert results == []
