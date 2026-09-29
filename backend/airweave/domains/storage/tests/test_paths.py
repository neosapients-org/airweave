"""Unit tests for StoragePaths, in particular the upload path helpers."""

from airweave.domains.storage.paths import StoragePaths


class TestUploadPrefix:
    """upload_prefix / upload_file_path build the neo_file_upload storage layout."""

    def test_upload_prefix_scopes_by_org_and_collection(self):
        assert (
            StoragePaths.upload_prefix("org-1", "coll-1") == "uploads/org-1/coll-1"
        )

    def test_upload_file_path_appends_relative_path(self):
        assert (
            StoragePaths.upload_file_path("org-1", "coll-1", "folder/notes.md")
            == "uploads/org-1/coll-1/folder/notes.md"
        )


class TestSafeRelativePath:
    """safe_relative_path rejects traversal/absolute paths, preserves folders."""

    def test_simple_filename_is_unchanged(self):
        assert StoragePaths.safe_relative_path("notes.md") == "notes.md"

    def test_preserves_folder_structure(self):
        assert (
            StoragePaths.safe_relative_path("folder/subfolder/report.pdf")
            == "folder/subfolder/report.pdf"
        )

    def test_rejects_empty_path(self):
        assert StoragePaths.safe_relative_path("") is None

    def test_rejects_parent_traversal(self):
        assert StoragePaths.safe_relative_path("../../etc/passwd") is None

    def test_rejects_traversal_in_middle_segment(self):
        assert StoragePaths.safe_relative_path("folder/../../secret.md") is None

    def test_rejects_absolute_unix_path(self):
        assert StoragePaths.safe_relative_path("/etc/passwd") is None

    def test_rejects_windows_drive_letter(self):
        assert StoragePaths.safe_relative_path("C:\\Windows\\win.ini") is None

    def test_normalizes_backslashes_to_forward_slashes(self):
        assert (
            StoragePaths.safe_relative_path("folder\\notes.md") == "folder/notes.md"
        )

    def test_sanitizes_unsafe_characters_per_segment(self):
        result = StoragePaths.safe_relative_path("weird:name?.md")
        assert result is not None
        assert ":" not in result and "?" not in result

    def test_rejects_path_that_is_only_dot_segments(self):
        assert StoragePaths.safe_relative_path("./.") is None
