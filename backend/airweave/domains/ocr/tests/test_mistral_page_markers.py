"""Page markers from Mistral OCR of PDFs, numbered across split parts (Airweave A4).

Fake OCR responses only; no Mistral call.
"""

from types import SimpleNamespace

from airweave.domains.converters.page_markers import strip_page_markers
from airweave.domains.ocr.mistral.converter import MistralOCR
from airweave.domains.ocr.mistral.models import BatchFileGroup, FileChunk, OcrResult
from airweave.domains.ocr.mistral.ocr_client import MistralOcrClient


def _group(path: str, n_parts: int) -> BatchFileGroup:
    group = BatchFileGroup(original_path=path, batch_index=0)
    for i in range(n_parts):
        group.chunks.append(
            FileChunk(
                original_path=path,
                chunk_path=f"/tmp/part{i}.pdf",
                batch_index=0,
                chunk_index=i,
                is_temp=n_parts > 1,
            )
        )
    return group


def _results(group: BatchFileGroup, pages_per_part, pdf: bool = True) -> dict:
    """OcrResults as MistralOcrClient builds them: joined markdown plus, for PDFs, pages."""
    out = {}
    for chunk, pages in zip(group.chunks, pages_per_part, strict=True):
        markdown = "\n\n".join(p for p in pages if p)
        out[(chunk.batch_index, chunk.chunk_index)] = OcrResult(
            chunk=chunk, markdown=markdown, pages=list(pages) if pdf else None
        )
    return out


class TestReassembly:
    def test_pages_numbered_across_split_parts(self):
        group = _group("/data/acme-annual.pdf", 2)
        results = _results(group, [["Acme overview", "", "Globex segment"], ["Initech outlook"]])
        text = MistralOCR._assemble_group(group, results)
        clean, starts = strip_page_markers(text)
        assert [p for _, p in starts] == [1, 3, 4]  # page 2 has no text, so no marker
        assert clean[starts[1][0] :].startswith("Globex segment")
        assert clean[starts[2][0] :].startswith("Initech outlook")

    def test_without_markers_the_text_is_upstreams(self):
        group = _group("/data/acme-annual.pdf", 2)
        pages = [["Acme overview", "", "Globex segment"], ["Initech outlook"]]
        results = _results(group, pages)
        upstream = "\n\n---\n\n".join("\n\n".join(p for p in part if p) for part in pages)
        assert strip_page_markers(MistralOCR._assemble_group(group, results))[0] == upstream

    def test_part_without_text_kept_as_upstream(self):
        """A split part whose pages are all blank still joins as an empty part."""
        group = _group("/data/acme-annual.pdf", 3)
        pages = [["Acme overview"], ["", ""], ["Initech outlook"]]
        upstream = "\n\n---\n\n".join("\n\n".join(p for p in part if p) for part in pages)
        text = MistralOCR._assemble_group(group, _results(group, pages))
        clean, starts = strip_page_markers(text)
        assert clean == upstream
        assert [p for _, p in starts] == [1, 4]

    def test_single_part(self):
        group = _group("/data/acme.pdf", 1)
        text = MistralOCR._assemble_group(group, _results(group, [["One", "Two"]]))
        assert [p for _, p in strip_page_markers(text)[1]] == [1, 2]

    def test_not_a_pdf_unchanged(self):
        group = _group("/data/acme.docx", 1)
        results = _results(group, [["Acme memo"]], pdf=False)
        assert MistralOCR._assemble_group(group, results) == "Acme memo"

    def test_failed_part_still_fails_the_file(self):
        group = _group("/data/acme.pdf", 2)
        results = _results(group, [["One"], ["Two"]])
        key = (0, 1)
        results[key] = OcrResult(chunk=results[key].chunk, markdown=None, error="boom")
        assert MistralOCR._assemble_group(group, results) is None


class TestClientPages:
    def test_page_markdowns_keep_response_order_and_empty_pages(self):
        resp = SimpleNamespace(
            pages=[
                SimpleNamespace(index=0, markdown="Acme"),
                SimpleNamespace(index=1, markdown=None),
                SimpleNamespace(index=2, markdown="Globex"),
            ]
        )
        assert MistralOcrClient._page_markdowns(resp) == ["Acme", "", "Globex"]

    def test_only_pdfs_get_pages(self):
        pdf = FileChunk(original_path="/d/a.PDF", chunk_path="/tmp/x.pdf", batch_index=0)
        docx = FileChunk(original_path="/d/a.docx", chunk_path="/d/a.docx", batch_index=0)
        assert MistralOcrClient._is_pdf(pdf) and not MistralOcrClient._is_pdf(docx)
