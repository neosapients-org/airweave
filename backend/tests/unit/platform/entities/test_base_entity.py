"""Unit tests for FileEntity document category fields."""

from airweave.platform.entities._base import FileEntity


class TestFileEntity:
    """Test FileEntity category metadata."""

    def test_doc_categories_accepts_list(self):
        entity = FileEntity.model_construct(
            entity_id="ent:001",
            name="report.pdf",
            url="https://example.com/report.pdf",
            size=123,
            file_type="pdf",
            doc_categories=["correspondence", "contract"],
        )

        assert entity.doc_categories == ["correspondence", "contract"]

    def test_doc_category_field_removed(self):
        assert "doc_category" not in FileEntity.model_fields
