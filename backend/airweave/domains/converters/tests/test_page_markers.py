"""Page markers in converted text (Airweave A4, contract 3.1 and 3.2)."""

import random

import pytest

from airweave.domains.converters.page_markers import (
    PAGE_SENTINEL,
    escape_page_text,
    page_at,
    page_marker,
    strip_page_markers,
)
from airweave.domains.converters.text_extractors.pdf import (
    PageExtractionResult,
    PdfExtractionResult,
    text_to_markdown,
)


class TestMarkers:
    def test_marker_format(self):
        """A private-use sentinel around a lower-case label, one line (the fixture's form)."""
        assert page_marker(12) == "\ue000page 12\ue000\n"
        assert PAGE_SENTINEL == "\ue000"

    def test_strip_records_page_starts(self):
        text = "head\n" + page_marker(1) + "one\n" + page_marker(2) + "two"
        clean, starts = strip_page_markers(text)
        assert clean == "head\none\ntwo"
        assert starts == [(5, 1), (9, 2)]
        assert clean[5:].startswith("one") and clean[9:].startswith("two")

    def test_no_markers(self):
        assert strip_page_markers("Acme report") == ("Acme report", [])

    def test_lone_sentinel_is_text(self):
        """Only a full marker is removed; a private-use character on its own stays."""
        text = "Symbol \ue000 font glyph"
        assert strip_page_markers(text) == (text, [])

    def test_a_forged_marker_in_the_document_is_text_not_a_page_boundary(self):
        forged = "Invoice \ue000page 4\ue000 follows"
        text = page_marker(1) + escape_page_text(forged) + "\n\n" + page_marker(2) + "end"

        clean, starts = strip_page_markers(text)

        assert clean == forged + "\n\nend"
        assert starts == [(0, 1), (len(forged) + 2, 2)]

    def test_literal_sentinels_survive_a_round_trip_and_do_not_move_positions(self):
        page = "a\ue000b\ue000\ue000c\ue000"
        text = (
            page_marker(1) + escape_page_text(page) + page_marker(2) + escape_page_text("\ue000x")
        )

        clean, starts = strip_page_markers(text)

        assert clean == page + "\ue000x"
        assert starts == [(0, 1), (len(page), 2)]

    def test_the_escape_is_the_identity_without_a_sentinel(self):
        assert escape_page_text("plain text") == "plain text"

    def test_page_at(self):
        starts = [(5, 1), (9, 2), (20, 3)]
        assert page_at(0, starts) == 1  # text before the first marker: first page
        assert page_at(8, starts) == 1
        assert page_at(9, starts) == 2
        assert page_at(25, starts) == 3
        assert page_at(3, []) is None  # no markers, no pages

    def test_page_numbers_need_not_start_at_one(self):
        assert page_at(0, [(0, 7), (10, 8)]) == 7


def _page_text(rng: random.Random) -> str:
    lines = []
    for _ in range(rng.randint(2, 8)):
        kind = rng.choice(["title", "upper", "bullet", "prose", "num", "blank"])
        lines.append(
            {
                "title": "Quarterly Results Overview",
                "upper": "CONSOLIDATED BALANCE SHEETS",
                "bullet": "• Revenue grew at Acme and Globex",
                "prose": "Net sales increased 12% compared with the prior year period.",
                "num": "3. Segment information",
                "blank": "",
            }[kind]
        )
    return "\n".join(lines).strip() or "Initech"


class TestPdfMarkersThroughMarkdown:
    @pytest.mark.parametrize("seed", range(20))
    def test_stripping_gives_upstream_markdown(self, seed):
        """text_to_markdown leaves markers alone, so removing them gives upstream's text."""
        rng = random.Random(seed)  # noqa: S311 - test data, not security
        pages = [
            PageExtractionResult(page_num=i, text=_page_text(rng), needs_ocr=False)
            for i in range(rng.randint(1, 6))
        ]
        result = PdfExtractionResult(path="acme.pdf", pages=pages)
        upstream = text_to_markdown(result.full_text)
        clean, starts = strip_page_markers(text_to_markdown(result.full_text_with_page_markers))
        assert clean == upstream
        assert [p for _, p in starts] == [p.page_num + 1 for p in pages]

    def test_page_starts_point_at_page_text(self):
        pages = [
            PageExtractionResult(page_num=0, text="Acme revenue rose sharply.", needs_ocr=False),
            PageExtractionResult(page_num=1, text="Globex margins fell slightly.", needs_ocr=False),
        ]
        result = PdfExtractionResult(path="acme.pdf", pages=pages)
        clean, starts = strip_page_markers(text_to_markdown(result.full_text_with_page_markers))
        assert clean[starts[1][0] :].startswith("Globex")
        assert starts[0] == (0, 1)
