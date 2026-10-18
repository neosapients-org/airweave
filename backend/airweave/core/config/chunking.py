"""How text is chunked for a deployment's ``CHUNKER_MODE`` and ``CHUNK_TARGET_TOKENS``.

Only the mode and the target are settings. Everything else follows from the mode, so a deployment
cannot pick a combination nobody tested (an overlap bigger than the target, a header that does not
fit the embedder, a version label that no longer matches the chunker's code):

- ``semantic`` (the default) is upstream's chunker, untouched: no overlap, no positions, no header.
- ``sentence_pack`` adds exact positions, pages and a one-line header before each chunk, with an
  overlap of ``OVERLAP_PERCENT`` of the target (45 tokens at a target of 300).

``SENTENCE_PACK_VERSION`` is part of every text entity's hash. It changes when the chunker's code
does, and then every text entity is chunked again on its next sync.
"""

from __future__ import annotations

from dataclasses import dataclass

# The embedder's hard limit: a chunk body plus its header must fit.
EMBEDDER_MAX_TOKENS = 8192
# A header longer than this is cut to it (it is embedded together with the body).
HEADER_MAX_TOKENS = 256
# sentence_pack overlaps consecutive chunks by this share of the target.
OVERLAP_PERCENT = 15
SEMANTIC_VERSION = "1"
SENTENCE_PACK_VERSION = "2"
SEMANTIC_DEFAULT_TARGET_TOKENS = 4096


@dataclass(frozen=True)
class ChunkPlan:
    """What the chunk processor does, decided once from the two settings."""

    mode: str
    target_tokens: int
    overlap_tokens: int
    header: bool
    header_max_tokens: int
    version: str

    @classmethod
    def for_mode(cls, mode: str, target_tokens: int) -> "ChunkPlan":
        """Build the plan for ``mode`` (``semantic`` or ``sentence_pack``) and a target."""
        if mode == "sentence_pack":
            return cls(
                mode=mode,
                target_tokens=target_tokens,
                overlap_tokens=target_tokens * OVERLAP_PERCENT // 100,
                header=True,
                header_max_tokens=HEADER_MAX_TOKENS,
                version=SENTENCE_PACK_VERSION,
            )
        return cls(
            mode=mode,
            target_tokens=target_tokens,
            overlap_tokens=0,
            header=False,
            header_max_tokens=HEADER_MAX_TOKENS,
            version=SEMANTIC_VERSION,
        )

    @classmethod
    def from_settings(cls, settings) -> "ChunkPlan":
        """Build the plan from the process settings."""
        return cls.for_mode(settings.CHUNKER_MODE, settings.CHUNK_TARGET_TOKENS)

    @property
    def positions(self) -> bool:
        """Whether chunks carry exact character ranges and pages (``sentence_pack`` only)."""
        return self.mode == "sentence_pack"

    @property
    def is_upstream(self) -> bool:
        """Whether this is exactly what upstream does (the entity hash then stays upstream's)."""
        return self == ChunkPlan.for_mode("semantic", SEMANTIC_DEFAULT_TARGET_TOKENS)
