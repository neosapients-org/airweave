# Multi-Category Classification & Admin Reclassification Design

**Date:** 2026-05-08  
**Status:** Approved

---

## Problem

The existing classifier assigns exactly **one** `doc_category` (a single string) to each document. This prevents documents that genuinely span multiple types (e.g. a financial report that is also a compliance filing) from being discoverable via both categories. Additionally, there is no mechanism to re-classify already-synced documents when new categories are added to `PREDEFINED_CATEGORIES`.

---

## Goals

1. Change `doc_category: Optional[str]` → `doc_categories: Optional[List[str]]` (1–3 categories per document).
2. Update the LLM prompt to request a JSON array of categories.
3. Add a `force_reclassify` flag to `ClassifierProcessor.process()`.
4. Add an admin endpoint `POST /admin/collections/{readable_id}/reclassify` that pages through all Vespa entities in a collection and re-classifies them in-place.
5. Update the Vespa schema field from `string` to `array<string>`.
6. Update search filters so `doc_category` (`DOC_CATEGORY`) maps to the new array field with appropriate operators.

---

## Section 1 — Data Model

### `BaseEntity` (`backend/airweave/platform/entities/_base.py`)

```python
# Before
doc_category: Optional[str] = Field(
    default=None,
    description="Document category assigned by LLM classification during sync.",
)

# After
doc_categories: Optional[List[str]] = Field(
    default=None,
    description="Document categories assigned by LLM classification (1–3 categories).",
)
```

The old `doc_category` field is removed. All references in the pipeline are updated to `doc_categories`.

### Vespa Schema

```xml
<!-- Before -->
<field name="doc_category" type="string" indexing="summary | attribute" />

<!-- After -->
<field name="doc_categories" type="array<string>" indexing="summary | attribute" />
```

`array<string>` supports element-membership queries (`contains`) natively in Vespa YQL.

### Vespa Transformer (`backend/airweave/platform/destinations/vespa/transformer.py`)

```python
# Before
if entity.doc_category:
    fields["doc_category"] = entity.doc_category

# After
if entity.doc_categories:
    fields["doc_categories"] = entity.doc_categories
```

---

## Section 2 — Classifier Changes

### LLM Prompt

Updated to request 1–3 categories as a JSON array:

```
Classify this document into 1 to 3 categories that apply.

...category descriptions...

Respond with JSON: {"doc_categories": ["<category1>", "<category2>"]}
```

### Validation

- Each returned category is validated against `PREDEFINED_CATEGORIES`.
- Unknown categories are dropped (not substituted with "other" individually) unless ALL are unknown, in which case `["other"]` is used.
- Result is deduplicated and capped at 3.

### `textual_representation` Prefix

```
[Categories: financial_report, compliance_regulatory]
```

### `force_reclassify` Parameter

`ClassifierProcessor.process()` gains:

```python
async def process(
    self,
    entities: List["BaseEntity"],
    force_reclassify: bool = False,
) -> List["BaseEntity"]:
```

When `force_reclassify=False` (default, normal syncs): skip entities that already have `doc_categories` set.  
When `force_reclassify=True` (reclassification endpoint): classify all entities regardless of existing value.

---

## Section 3 — Admin Reclassification Endpoint

### Route

```
POST /admin/collections/{readable_id}/reclassify
```

### Auth

Uses existing `_require_admin_permission(ctx, FeatureFlagEnum.API_KEY_ADMIN_SYNC)`.

### Behavior

1. Look up the collection by `readable_id`, bypassing org filtering.
2. Page through all entities in Vespa for the collection using scroll/search (page size configurable, default 100).
3. For each page, reconstruct minimal `FileEntity` objects from Vespa-stored fields (`name`, `mime_type`, `textual_representation`, `doc_categories`).
4. Call `ClassifierProcessor.process(entities, force_reclassify=True)`.
5. Issue Vespa partial updates: update only `doc_categories` and `textual_representation` fields (no re-embed).
6. Accumulate counts: total, reclassified, failed.

### Response

```json
{
  "collection": "my-collection",
  "total": 1500,
  "reclassified": 1487,
  "failed": 13
}
```

### Notes on Embeddings

This endpoint updates `doc_categories` as a filterable Vespa attribute immediately. The `[Categories: ...]` prefix in `textual_representation` is stored but embeddings are **not** recomputed. Category-based **filtering** works immediately after reclassification. The vector boost from the category prefix takes effect on the next full resync.

---

## Section 4 — Search Filter Changes

`DOC_CATEGORY` in `FilterableField` is retained (name unchanged for API compatibility) but its underlying Vespa field is now `doc_categories` (array).

`DOC_CATEGORY` is moved from `_TEXT_FIELDS` to a new `_ARRAY_FIELDS` frozenset. Allowed operators:

| Operator | Meaning |
|---|---|
| `contains` | document has this category in its list |
| `in` | document has any of these categories |
| `not_in` | document has none of these categories |

Ordering operators (`>`, `<`, etc.) and `equals`/`not_equals` are disallowed on array fields.

The Vespa query builder is updated to translate `contains` on `doc_categories` to a YQL `contains` expression on the array field.

---

## Out of Scope

- Re-computing embeddings during reclassification (requires a full resync).
- Automatic reclassification triggered on category list change.
- Per-category confidence scores.
