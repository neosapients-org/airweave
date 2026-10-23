"""The classifier package Airweave calls, checked when a process starts, not at its first sync.

``taxonomy_classifier`` (distribution ``airweave-taxonomy-classifier``) lives in
``backend/packages/taxonomy-classifier`` and is a path dependency in the lock file, so the image
always has it; every import of it is lazy. The check still guards a process installed some other
way: without it, ``TAXONOMY_ENABLED=true`` and no package would boot and serve, and fail only at
its first labelled chunk. These lists are the interface Airweave uses, by module and name.
"""

from __future__ import annotations

import importlib
from typing import Mapping, Sequence

#: What sync-time classification and version resolution call.
SYNC_INTERFACE: Mapping[str, Sequence[str]] = {
    "taxonomy_classifier.integration.airweave": ("TaxonomyClassifier",),
    "taxonomy_classifier.contract": ("to_contract",),
    "taxonomy_classifier.assignment.learned": ("FEATURE_SCHEMA_VERSION", "FEATURE_NAMES"),
}
#: What the discover and propose jobs call besides (the package's build extra).
JOBS_INTERFACE: Mapping[str, Sequence[str]] = {
    "taxonomy_classifier": ("TaxonomyEngine", "LocalTaxonomyStore"),
    "taxonomy_classifier.taxonomy.artifacts": ("load",),
}


class ClassifierPackageMissing(RuntimeError):
    """``TAXONOMY_ENABLED`` is true but the package, or a name Airweave calls, is missing."""


def require_classifier_package(*interfaces: Mapping[str, Sequence[str]]) -> None:
    """Import every module and name Airweave calls; raise naming each one that is missing.

    Args:
        *interfaces: The interfaces to check; none given means ``SYNC_INTERFACE``.

    Raises:
        ClassifierPackageMissing: When a module cannot be imported or lacks a name.
    """
    missing = []
    for interface in interfaces or (SYNC_INTERFACE,):
        for module_name, names in interface.items():
            try:
                module = importlib.import_module(module_name)
            except ImportError as e:
                missing.append(f"{module_name} ({e})")
                continue
            missing.extend(f"{module_name}.{n}" for n in names if not hasattr(module, n))
    if missing:
        raise ClassifierPackageMissing(
            "TAXONOMY_ENABLED is true but the taxonomy classifier package "
            "(airweave-taxonomy-classifier) is not installed or lacks: " + ", ".join(missing)
        )
