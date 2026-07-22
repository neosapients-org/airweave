"""ClassifierProcessor — optional LLM-based document classification.

Runs between TextualRepresentationBuilder and ChunkEmbedProcessor when
CLASSIFICATION_ENABLED=true. Classifies each entity into 1-3 document
categories using an LLM (default: GPT-4o-mini), with optional vision OCR
for images that have poor text extraction from Docling.
"""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from openai import AsyncOpenAI

from airweave.core.config import settings
from airweave.core.logging import logger as default_logger

if TYPE_CHECKING:
    from airweave.platform.entities._base import BaseEntity, FileEntity

# ---------------------------------------------------------------------------
# Predefined categories
# ---------------------------------------------------------------------------

PREDEFINED_CATEGORIES: list[str] = [
    "identity_document",
    "financial_report",
    "investor_data",
    "portfolio_company",
    "deal_pipeline",
    "compliance_regulatory",
    "market_research",
    "contract",
    "correspondence",
    "technical_spec",
    "operational",
    "presentation",
    "spreadsheet",
    "reference",
    "travel",
    "personal",
    "other",
]

CLASSIFICATION_PROMPT = """\
Classify this document into 1 to 3 categories that best describe it.

Categories:
- identity_document: IDs, passports, PAN cards, driver's licenses, KYC documents
- financial_report: Balance sheets, P&L, annual reports, audits, tax filings
- investor_data: LP agreements, investor communications, fund performance, capital calls
- portfolio_company: Portfolio company info, board decks, company profiles
- deal_pipeline: Deal memos, term sheets, LOIs, due diligence, IC materials
- compliance_regulatory: Policies, regulatory filings, compliance checklists, audit reports
- market_research: Industry analysis, market reports, competitive intelligence
- contract: Agreements, NDAs, service contracts, employment contracts, leases
- correspondence: Emails, letters, memos, meeting notes, chat logs
- technical_spec: Technical documentation, API docs, architecture diagrams, specs
- operational: SOPs, process docs, HR documents, org charts, handbooks
- presentation: Slide decks, pitch decks, training presentations
- spreadsheet: Data tables, CSV data, Excel workbooks, financial models
- reference: Manuals, guides, FAQs, knowledge base articles, templates
- travel: Itineraries, trip plans, tour packages, booking confirmations, travel guides
- personal: Personal notes, journals, recipes, hobbies, non-work personal documents
- other: If none of the above fit

Rules:
- Return between 1 and 3 categories, ordered from most to least relevant.
- Only include a category if it clearly applies. Do not pad to 3.
- Use "other" only if no specific category fits.

Document info:
- Filename: {filename}
- MIME type: {mime_type}
- Content (first 1500 chars):
{content_snippet}

Respond with JSON: {{"doc_categories": ["<primary_category>", "<optional_second>", "<optional_third>"]}}
"""

VISION_OCR_PROMPT = (
    "Extract ALL text visible in this image. Return only the raw text content, "
    "preserving structure where possible. If no text is visible, return 'NO_TEXT_FOUND'."
)


def _normalize_categories(raw_categories: object, *, logger=None, name: str = "unknown") -> list[str]:
    """Normalize raw LLM output into up to 3 supported categories."""
    if isinstance(raw_categories, str):
        raw_categories = [raw_categories]
    if not isinstance(raw_categories, list):
        return ["other"]

    categories: list[str] = []
    for category in raw_categories:
        if not isinstance(category, str):
            continue
        if category in PREDEFINED_CATEGORIES:
            categories.append(category)
        elif category.startswith("other"):
            categories.append("other")
        else:
            if logger is not None:
                logger.warning(
                    f"[ClassifierProcessor] Unknown category '{category}' for {name}, "
                    "dropping it"
                )

        if len(categories) == 3:
            break

    return categories or ["other"]


def strip_category_prefix(text: str) -> str:
    """Remove an existing category prefix from textual representation."""
    if not text.startswith("[Category:") and not text.startswith("[Categories:"):
        return text

    _, sep, remainder = text.partition("\n")
    if not sep:
        return ""
    return remainder.lstrip("\n")


def prepend_categories_prefix(text: str, categories: list[str]) -> str:
    """Prepend a normalized category prefix to text."""
    categories_str = ", ".join(categories)
    return f"[Categories: {categories_str}]\n\n{text}"


async def classify_text(
    text: str,
    filename: str,
    mime_type: str,
    model: Optional[str] = None,
) -> list[str]:
    """Classify raw text into up to 3 document categories."""
    if not settings.OPENAI_API_KEY:
        return []

    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    prompt = CLASSIFICATION_PROMPT.format(
        filename=filename,
        mime_type=mime_type,
        content_snippet=text[:1500],
    )

    try:
        response = await client.chat.completions.create(
            model=model or settings.CLASSIFICATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=100,
        )
        raw = json.loads(response.choices[0].message.content or "{}")
    except Exception:
        return []

    return _normalize_categories(raw.get("doc_categories", []), name=filename)


class ClassifierProcessor:
    """Classifies entities by document categories using an LLM."""

    def __init__(self) -> None:
        self._client: Optional[AsyncOpenAI] = None
        self._logger = default_logger

    @property
    def enabled(self) -> bool:
        """Whether classification is enabled via settings."""
        return bool(settings.CLASSIFICATION_ENABLED)

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            if not settings.OPENAI_API_KEY:
                raise RuntimeError(
                    "CLASSIFICATION_ENABLED=true but OPENAI_API_KEY is not set. "
                    "Document classification requires an OpenAI API key."
                )
            self._client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        return self._client

    async def process(self, entities: List["BaseEntity"]) -> List["BaseEntity"]:
        """Classify all entities and set doc_categories."""
        if not self.enabled or not entities:
            return entities

        self._logger.info(f"[ClassifierProcessor] Classifying {len(entities)} entities")

        await self._vision_ocr_pass(entities)

        batch_size = settings.CLASSIFICATION_BATCH_SIZE
        for i in range(0, len(entities), batch_size):
            batch = entities[i : i + batch_size]
            await self._classify_batch(batch)

        classified = sum(1 for e in entities if getattr(e, "doc_categories", None))
        self._logger.info(
            f"[ClassifierProcessor] Classified {classified}/{len(entities)} entities"
        )

        return entities

    # -----------------------------------------------------------------------
    # Vision OCR
    # -----------------------------------------------------------------------

    async def _vision_ocr_pass(self, entities: List["BaseEntity"]) -> None:
        """Run vision OCR on ALL image entities.

        Images always use the vision model as their source of truth.
        Whatever Docling extracted (often raw base64 binary or empty) is
        discarded — the OCR result replaces textual_representation entirely
        so the chunker produces one clean, meaningful chunk instead of
        dozens of base64 garbage chunks.
        """
        from airweave.platform.entities._base import FileEntity

        image_entities = [
            e
            for e in entities
            if isinstance(e, FileEntity)
            and e.mime_type
            and e.mime_type.startswith("image/")
        ]

        if not image_entities:
            return

        self._logger.info(
            f"[ClassifierProcessor] Running vision OCR on {len(image_entities)} images"
        )

        tasks = [self._ocr_single(e) for e in image_entities]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for entity, result in zip(image_entities, results):
            if isinstance(result, Exception):
                self._logger.warning(
                    f"[ClassifierProcessor] OCR failed for {entity.entity_id}: {result}"
                )

    async def _ocr_single(self, entity: "FileEntity") -> None:
        """OCR a single image entity using the vision model."""
        image_data = None
        if entity.local_path and Path(entity.local_path).exists():
            image_data = Path(entity.local_path).read_bytes()

        if not image_data:
            self._logger.debug(
                f"[ClassifierProcessor] No image data for {entity.entity_id}, skipping OCR"
            )
            return

        b64 = base64.b64encode(image_data).decode()
        mime = entity.mime_type or "image/jpeg"

        client = self._get_client()
        response = await client.chat.completions.create(
            model=settings.CLASSIFICATION_VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VISION_OCR_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        },
                    ],
                }
            ],
            max_tokens=2000,
            temperature=0,
        )

        ocr_text = response.choices[0].message.content.strip()
        if ocr_text and ocr_text != "NO_TEXT_FOUND":
            entity.textual_representation = ocr_text
            self._logger.info(
                f"[ClassifierProcessor] OCR extracted {len(ocr_text)} chars "
                f"for {entity.name}"
            )

    # -----------------------------------------------------------------------
    # LLM Classification
    # -----------------------------------------------------------------------

    async def _classify_batch(self, batch: List["BaseEntity"]) -> None:
        """Classify a batch of entities concurrently."""
        tasks = [self._classify_single(e) for e in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for entity, result in zip(batch, results):
            if isinstance(result, Exception):
                self._logger.warning(
                    f"[ClassifierProcessor] Classification failed for "
                    f"{getattr(entity, 'entity_id', '?')}: {result}"
                )
                if hasattr(entity, "doc_categories"):
                    entity.doc_categories = ["other"]

    async def _classify_single(self, entity: "BaseEntity") -> None:
        """Classify a single entity using the configured LLM."""
        text = entity.textual_representation or ""
        snippet = text[:1500]
        name = getattr(entity, "name", "unknown")
        mime = getattr(entity, "mime_type", "unknown")

        prompt = CLASSIFICATION_PROMPT.format(
            filename=name,
            mime_type=mime,
            content_snippet=snippet,
        )

        client = self._get_client()
        response = await client.chat.completions.create(
            model=settings.CLASSIFICATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
            max_tokens=100,
        )

        try:
            result = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"LLM returned invalid JSON for entity '{name}': {e}"
            ) from e

        categories = _normalize_categories(
            result.get("doc_categories", []),
            logger=self._logger,
            name=name,
        )

        if hasattr(entity, "doc_categories"):
            entity.doc_categories = categories

        if entity.textual_representation:
            entity.textual_representation = prepend_categories_prefix(
                entity.textual_representation,
                categories,
            )
