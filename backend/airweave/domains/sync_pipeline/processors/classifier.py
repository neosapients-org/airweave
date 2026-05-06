"""ClassifierProcessor — Optional LLM-based document classification.

Runs between TextualRepresentationBuilder and ChunkEmbedProcessor when
CLASSIFICATION_ENABLED=true. Classifies each entity into a doc_category
using an LLM (default: GPT-4o-mini), with optional vision OCR for images
that have poor text extraction from Docling.

Configuration (environment variables):
    CLASSIFICATION_ENABLED: bool (default: false) — master switch
    CLASSIFICATION_MODEL: str (default: gpt-4o-mini) — model for classification
    CLASSIFICATION_VISION_MODEL: str (default: gpt-4o) — model for image OCR
    CLASSIFICATION_BATCH_SIZE: int (default: 25) — concurrent classifications per batch
    CLASSIFICATION_RATE_LIMIT_PER_HOUR: int (default: 1000) — max docs classified per org/hour
    OPENAI_API_KEY: str — required when classification is enabled

Rate limiting:
    A Redis counter keyed per org tracks how many documents have been classified
    in the current UTC hour. Once the hourly budget is exhausted, remaining entities
    in the sync pass through unclassified (doc_category stays None). The next sync
    run will attempt to classify them if budget has reset.

The processor sets `entity.doc_category` and prepends a `[Category: X]` prefix
to `textual_representation` so the category is baked into both the dense and
sparse embeddings stored in Vespa.
"""

from __future__ import annotations

import asyncio
import base64
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from openai import AsyncOpenAI

from airweave.core.config import settings
from airweave.core.logging import logger as default_logger
# from airweave.core.redis_client import redis_client  # re-enable with rate limiting

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
    "other",
]

CLASSIFICATION_PROMPT = """\
Classify this document into exactly ONE category.

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
- other: If none of the above fit. Use format "other:brief_description"

Document info:
- Filename: {filename}
- MIME type: {mime_type}
- Content (first 1500 chars):
{content_snippet}

Respond with JSON: {{"doc_category": "<category>"}}
"""

VISION_OCR_PROMPT = (
    "Extract ALL text visible in this image. Return only the raw text content, "
    "preserving structure where possible. If no text is visible, return 'NO_TEXT_FOUND'."
)

_RATE_LIMIT_KEY_PREFIX = "classification_rate_limit"


class ClassifierProcessor:
    """Classifies entities by document category using an LLM.

    Pipeline position: after TextualRepresentationBuilder, before ChunkEmbedProcessor.
    Only runs when ``settings.CLASSIFICATION_ENABLED`` is True.

    Rate limiting:
        Uses a Redis counter keyed to the current UTC hour per org.
        Key format: ``classification_rate_limit:{org_id}:{YYYYMMDDHH}``
        TTL is set to 2 hours so keys expire automatically.
    """

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

    # -----------------------------------------------------------------------
    # Rate limiting (Redis hourly window per org)
    # TODO: Enable rate limiting once we have a clearer picture of OpenAI costs
    #       at scale. Uncomment the methods below and re-enable the budget checks
    #       in process() to cap classifications at CLASSIFICATION_RATE_LIMIT_PER_HOUR
    #       documents per org per hour.
    # -----------------------------------------------------------------------

    # def _rate_limit_key(self, org_id: UUID) -> str:
    #     """Redis key for the current UTC hour's classification count for this org."""
    #     hour_bucket = datetime.now(timezone.utc).strftime("%Y%m%d%H")
    #     return f"{_RATE_LIMIT_KEY_PREFIX}:{org_id}:{hour_bucket}"

    # async def _get_hourly_budget_remaining(self, org_id: UUID) -> int:
    #     """Return how many more documents this org can classify this hour.
    #
    #     Returns the full limit if Redis is unavailable (fail-open).
    #     """
    #     limit = settings.CLASSIFICATION_RATE_LIMIT_PER_HOUR
    #     try:
    #         key = self._rate_limit_key(org_id)
    #         used_str = await redis_client.client.get(key)
    #         used = int(used_str) if used_str else 0
    #         return max(0, limit - used)
    #     except Exception as e:
    #         self._logger.warning(
    #             f"[ClassifierProcessor] Redis unavailable for rate limit check, "
    #             f"proceeding without limit: {e}"
    #         )
    #         return limit

    # async def _increment_hourly_counter(self, org_id: UUID, count: int) -> None:
    #     """Increment the hourly classification counter by ``count``.
    #
    #     Sets a 2-hour TTL so keys clean themselves up.
    #     """
    #     if count <= 0:
    #         return
    #     try:
    #         key = self._rate_limit_key(org_id)
    #         pipe = redis_client.client.pipeline()
    #         pipe.incrby(key, count)
    #         pipe.expire(key, 7200)  # 2-hour TTL
    #         await pipe.execute()
    #     except Exception as e:
    #         self._logger.warning(
    #             f"[ClassifierProcessor] Failed to increment rate limit counter: {e}"
    #         )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    async def process(
        self, entities: List["BaseEntity"]
    ) -> List["BaseEntity"]:
        """Classify all entities and set doc_category.

        All entities are classified regardless of count. No rate limiting is
        applied currently — all documents in the sync will be classified.

        Args:
            entities: Entities after text extraction, before chunking.

        Returns:
            The same entity list with ``doc_category`` set and
            ``textual_representation`` prefixed with the category tag.

        # TODO: Re-enable hourly rate limiting once OpenAI cost baseline is
        #       established. Add org_id: UUID to the signature, then use
        #       _get_hourly_budget_remaining(org_id) to cap classifications at
        #       CLASSIFICATION_RATE_LIMIT_PER_HOUR per org/hour and
        #       _increment_hourly_counter(org_id, len(to_classify)) to track usage.
        """
        if not self.enabled or not entities:
            return entities

        self._logger.info(f"[ClassifierProcessor] Classifying {len(entities)} entities")

        # Step 1: Vision OCR for image entities
        await self._vision_ocr_pass(entities)

        # Step 2: LLM classification in concurrent batches
        batch_size = settings.CLASSIFICATION_BATCH_SIZE
        for i in range(0, len(entities), batch_size):
            batch = entities[i : i + batch_size]
            await self._classify_batch(batch)

        classified = sum(1 for e in entities if getattr(e, "doc_category", None))
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
                # Default to "other" on failure so every entity has a category
                if hasattr(entity, "doc_category"):
                    entity.doc_category = "other"

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

        category = result.get("doc_category", "other")

        # Validate category
        if category not in PREDEFINED_CATEGORIES and not category.startswith("other:"):
            self._logger.warning(
                f"[ClassifierProcessor] Unknown category '{category}' for {name}, "
                f"defaulting to 'other'"
            )
            category = "other"

        # Set on entity (FileEntity has the field; BaseEntity doesn't)
        if hasattr(entity, "doc_category"):
            entity.doc_category = category

        # Prepend category to textual_representation for embedding boost
        if entity.textual_representation:
            entity.textual_representation = (
                f"[Category: {category}]\n\n{entity.textual_representation}"
            )

Runs between TextualRepresentationBuilder and ChunkEmbedProcessor when
CLASSIFICATION_ENABLED=true. Classifies each entity into a doc_category
using an LLM (default: GPT-4o-mini), with optional vision OCR for images
that have poor text extraction from Docling.

Configuration (environment variables):
    CLASSIFICATION_ENABLED: bool (default: false) — master switch
    CLASSIFICATION_MODEL: str (default: gpt-4o-mini) — model for classification
    CLASSIFICATION_VISION_MODEL: str (default: gpt-4o) — model for image OCR
    CLASSIFICATION_BATCH_SIZE: int (default: 25) — concurrent classifications per batch
    OPENAI_API_KEY: str — required when classification is enabled

The processor sets `entity.doc_category` and prepends a `[Category: X]` prefix
to `textual_representation` so the category is baked into both the dense and
sparse embeddings stored in Vespa.
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
    "other",
]

CLASSIFICATION_PROMPT = """\
Classify this document into exactly ONE category.

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
- other: If none of the above fit. Use format "other:brief_description"

Document info:
- Filename: {filename}
- MIME type: {mime_type}
- Content (first 1500 chars):
{content_snippet}

Respond with JSON: {{"doc_category": "<category>"}}
"""

VISION_OCR_PROMPT = (
    "Extract ALL text visible in this image. Return only the raw text content, "
    "preserving structure where possible. If no text is visible, return 'NO_TEXT_FOUND'."
)


class ClassifierProcessor:
    """Classifies entities by document category using an LLM.

    Pipeline position: after TextualRepresentationBuilder, before ChunkEmbedProcessor.
    Only runs when ``settings.CLASSIFICATION_ENABLED`` is True.
    """

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
        """Classify all entities and set doc_category.

        If classification is disabled, returns entities unchanged.

        Args:
            entities: Entities after text extraction, before chunking.

        Returns:
            The same entity list with ``doc_category`` set and
            ``textual_representation`` prefixed with the category tag.
        """
        if not self.enabled or not entities:
            return entities

        # Cost guardrail: limit how many entities we classify per invocation
        max_entities = settings.CLASSIFICATION_MAX_ENTITIES
        if len(entities) > max_entities:
            self._logger.warning(
                f"[ClassifierProcessor] Entity count ({len(entities)}) exceeds "
                f"CLASSIFICATION_MAX_ENTITIES ({max_entities}). "
                f"Only classifying first {max_entities}."
            )
            to_classify = entities[:max_entities]
            remainder = entities[max_entities:]
        else:
            to_classify = entities
            remainder = []

        self._logger.info(f"[ClassifierProcessor] Classifying {len(to_classify)} entities")

        # Step 1: Vision OCR for images with poor text extraction
        await self._vision_ocr_pass(to_classify)

        # Step 2: LLM classification in batches
        batch_size = settings.CLASSIFICATION_BATCH_SIZE
        for i in range(0, len(to_classify), batch_size):
            batch = to_classify[i : i + batch_size]
            await self._classify_batch(batch)

        classified = sum(1 for e in to_classify if getattr(e, "doc_category", None))
        self._logger.info(
            f"[ClassifierProcessor] Classified {classified}/{len(to_classify)} entities"
        )

        return to_classify + remainder

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
                # Default to "other" on failure so every entity has a category
                if hasattr(entity, "doc_category"):
                    entity.doc_category = "other"

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

        result = json.loads(response.choices[0].message.content)
        category = result.get("doc_category", "other")

        # Validate category
        if category not in PREDEFINED_CATEGORIES and not category.startswith("other:"):
            self._logger.warning(
                f"[ClassifierProcessor] Unknown category '{category}' for {name}, "
                f"defaulting to 'other'"
            )
            category = "other"

        # Set on entity (FileEntity has the field; BaseEntity doesn't)
        if hasattr(entity, "doc_category"):
            entity.doc_category = category

        # Prepend category to textual_representation for embedding boost
        if entity.textual_representation:
            entity.textual_representation = (
                f"[Category: {category}]\n\n{entity.textual_representation}"
            )
