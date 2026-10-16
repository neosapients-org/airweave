"""SentencePackChunker: sentence packing with exact positions (Airweave A3, contract section 3)."""

import random
import re

import pytest
import tiktoken

from airweave.platform.chunkers.sentence_pack import SentencePackChunker, pack, sentences, units

ENC = tiktoken.get_encoding("cl100k_base")


def _tokens(text: str) -> int:
    return len(ENC.encode_ordinary(text))


def _check_contract_rules(text: str, chunks: list, target: int) -> None:
    """The position rules of contract 3.2 that hold for every document."""
    assert chunks[0]["start_index"] == 0
    assert chunks[-1]["end_index"] == len(text)
    for i, c in enumerate(chunks):
        assert text[c["start_index"] : c["end_index"]] == c["text"]
        assert _tokens(c["text"]) <= target, f"chunk {i} over target"
        expected_own = c["start_index"] if i == 0 else chunks[i - 1]["end_index"]
        assert c["own_start"] == expected_own
        assert c["own_start"] < c["end_index"], f"chunk {i} adds no new text"
        assert c["start_index"] <= c["own_start"]
    # own ranges tile the text: contiguous, no gaps, no overlap
    assert "".join(text[c["own_start"] : c["end_index"]] for c in chunks) == text


# ── The shared fixture ────────────────────────────────────────────────────────


class TestSharedFixture:
    @pytest.fixture(scope="class")
    def fixture(self, airweave_neo_chunk_fixture):
        return airweave_neo_chunk_fixture

    @pytest.fixture(scope="class")
    def chunks(self, fixture):
        s = fixture["settings"]
        chunker = SentencePackChunker(s["CHUNK_TARGET_TOKENS"], s["CHUNK_OVERLAP_TOKENS"])
        return chunker.chunk_text(fixture["document_text"])

    def test_same_ranges_as_fixture(self, fixture, chunks):
        got = [(c["start_index"], c["end_index"], c["own_start"]) for c in chunks]
        want = [
            (c["chunk_char_start"], c["chunk_char_end"], c["chunk_own_start"])
            for c in fixture["chunks"]
        ]
        assert got == want

    def test_same_bodies_and_token_counts(self, fixture, chunks):
        assert [c["text"] for c in chunks] == [c["_test"]["body"] for c in fixture["chunks"]]
        assert [c["token_count"] for c in chunks] == [
            c["_test"]["body_tokens"] for c in fixture["chunks"]
        ]

    def test_every_position_rule(self, fixture, chunks):
        _check_contract_rules(
            fixture["document_text"], chunks, fixture["settings"]["CHUNK_TARGET_TOKENS"]
        )

    def test_fixture_shows_overlap_and_emoji_shift(self, fixture, chunks):
        """The fixture covers overlap and code points that are not single UTF-16 units."""
        assert any(c["own_start"] > c["start_index"] for c in chunks)
        assert any(
            c["_test"]["utf16"]["chunk_char_end"] != c["chunk_char_end"]
            for c in fixture["chunks"]
        )

    def test_fixture_spans_lie_in_own_ranges(self, fixture, chunks):
        import json

        for c, fx in zip(chunks, fixture["chunks"], strict=True):
            for span in json.loads(fx["taxonomy_spans"]):
                assert c["own_start"] <= span["s"] < span["e"] <= c["end_index"]


# ── Parity with the reference implementation (make_chunk_fixture.py) ──────────

_REF_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")


def _reference_chunks(text: str, target: int, overlap: int) -> list:
    """The fixture script's units() and pack(), verbatim apart from parameters."""

    def ref_sentences(t):
        spans, start = [], 0
        for m in _REF_BOUNDARY.finditer(t):
            if m.end() - start >= 24:
                spans.append((start, m.end()))
                start = m.end()
        if start < len(t):
            spans.append((start, len(t)))
        return spans

    def ref_units(t):
        out = []
        for s, e in ref_sentences(t):
            n = len(ENC.encode_ordinary(t[s:e]))
            if n <= target:
                out.append((s, e, n))
                continue
            toks, p = ENC.encode_ordinary(t[s:e]), s
            for i in range(0, len(toks), target):
                piece = ENC.decode(toks[i : i + target])
                end = min(e, p + len(piece))
                out.append((p, end, len(toks[i : i + target])))
                p = end
        return out

    us = ref_units(text)
    chunks, i, n = [], 0, len(us)
    while i < n:
        j, tot = i, us[i][2]
        while j + 1 < n and tot + us[j + 1][2] <= target:
            j += 1
            tot += us[j][2]
        chunks.append((us[i][0], us[j][1]))
        if j == n - 1:
            break
        k, ov = j + 1, 0
        while k - 1 > i and ov + us[k - 1][2] <= overlap:
            k -= 1
            ov += us[k][2]
        while k <= j and ov + us[j + 1][2] > target:
            ov -= us[k][2]
            k += 1
        i = k
    return chunks


_WORDS = (
    "Acme Globex Initech revenue margin quarter loans allowance hedging guidance cash "
    "flow operating segment backlog impairment liquidity covenant dividend"
).split()


def _ascii_document(rng: random.Random, n_sentences: int) -> str:
    parts = []
    for _ in range(n_sentences):
        length = rng.choice([3, 8, 15, 30, 70, 400])  # some sentences pass any target
        words = " ".join(rng.choice(_WORDS) for _ in range(length))
        end = rng.choice([". ", "! ", "? ", ".\n", "\n\n", ". "])
        parts.append(words.capitalize() + end)
    return "".join(parts)


class TestReferenceParity:
    @pytest.mark.parametrize("seed", range(12))
    @pytest.mark.parametrize("target,overlap", [(40, 16), (300, 45), (600, 45), (120, 0)])
    def test_same_chunks_as_reference_on_ascii(self, seed, target, overlap):
        """Without multi-byte characters the port and the reference agree exactly."""
        text = _ascii_document(random.Random(seed), 60)
        chunker = SentencePackChunker(target, overlap)
        got = [(c["start_index"], c["end_index"]) for c in chunker.chunk_text(text)]
        assert got == _reference_chunks(text, target, overlap)

    @pytest.mark.parametrize("seed", range(12))
    @pytest.mark.parametrize("target,overlap", [(40, 16), (300, 45), (120, 0)])
    def test_rules_hold_on_generated_text(self, seed, target, overlap):
        text = _ascii_document(random.Random(seed), 60)
        _check_contract_rules(text, SentencePackChunker(target, overlap).chunk_text(text), target)


# ── Sentences longer than the target ──────────────────────────────────────────


class TestLongSentences:
    def test_long_sentence_split_into_target_windows(self):
        sentence = " ".join(["Globex quarterly revenue increased"] * 200) + "."
        chunks = SentencePackChunker(300, 45).chunk_text(sentence)
        assert len(chunks) > 1
        _check_contract_rules(sentence, chunks, 300)

    def test_windows_exact_around_multibyte_characters(self):
        """Window edges that fall inside an emoji or CJK character stay exact.

        The reference derives edges from the decoded window length, which counts replacement
        characters for a split character and drifts; the port uses token offsets instead.
        """
        sentence = ("売上高は前年同期比で増加しました📈" * 120) + "。"
        target = 40
        chunks = SentencePackChunker(target, 0).chunk_text(sentence)
        _check_contract_rules(sentence, chunks, target)
        assert "�" not in "".join(c["text"] for c in chunks)

    def test_reference_drifts_where_port_is_exact(self):
        """Documents the difference reported to the design session."""
        sentence = ("売上高は前年同期比で増加しました📈" * 120) + "。"
        ref = _reference_chunks(sentence, 40, 0)
        ref_text = "".join(sentence[s:e] for s, e in ref)
        port = SentencePackChunker(40, 0).chunk_text(sentence)
        assert "".join(c["text"] for c in port) == sentence
        assert ref_text != sentence or ref != [
            (c["start_index"], c["end_index"]) for c in port
        ]

    def test_units_cover_text_exactly(self):
        text = ("Acme 📈 Zürich 売上高 " * 300).strip() + ". Short tail sentence here."
        us = units(text, 50, ENC)
        assert us[0][0] == 0 and us[-1][1] == len(text)
        assert all(a[1] == b[0] for a, b in zip(us, us[1:], strict=False))
        assert all(s < e for s, e, _ in us)


# ── Packing and overlap ───────────────────────────────────────────────────────


class TestPacking:
    def test_no_overlap_when_zero(self):
        text = _ascii_document(random.Random(3), 40)
        chunks = SentencePackChunker(120, 0).chunk_text(text)
        assert all(c["own_start"] == c["start_index"] for c in chunks)

    def test_overlap_shrinks_so_every_chunk_adds_text(self):
        """Overlap plus the next sentence would pass the target: the overlap gives way."""
        us = [(0, 10, 20), (10, 20, 10), (20, 30, 35)]
        assert pack(us, 40, 16) == [(0, 20), (20, 30)]

    def test_overlap_counts_inside_target(self):
        us = [(0, 10, 10), (10, 20, 10), (20, 30, 10), (30, 40, 10), (40, 50, 10)]
        assert pack(us, 30, 10) == [(0, 30), (20, 50)]

    def test_sentence_rule_merges_tiny_pieces(self):
        assert sentences("Hi. Yo. This sentence is long enough. End") == [(0, 38), (38, 41)]

    def test_empty_text(self):
        assert SentencePackChunker(300, 45).chunk_text("") == []

    @pytest.mark.parametrize("target,overlap", [(300, 300), (300, 301), (300, -1)])
    def test_invalid_settings_rejected(self, target, overlap):
        with pytest.raises(ValueError):
            SentencePackChunker(target, overlap)

    async def test_chunk_batch_matches_chunk_text(self):
        chunker = SentencePackChunker(40, 16)
        texts = [_ascii_document(random.Random(s), 10) for s in range(3)]
        assert await chunker.chunk_batch(texts) == [chunker.chunk_text(t) for t in texts]
