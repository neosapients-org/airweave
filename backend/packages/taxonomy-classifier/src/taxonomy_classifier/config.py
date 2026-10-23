"""Configuration loading and hashing.

Relative paths inside the config (caches, etc.) are resolved against the current working directory.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

PACKAGE_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = PACKAGE_DIR / "llm" / "prompts"


def resolve(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps({k: v for k, v in cfg.items() if not k.startswith("_")},
                                     sort_keys=True, default=str).encode()).hexdigest()


def load_config(path: str | Path) -> dict:
    path = resolve(path)
    cfg = yaml.safe_load(path.read_text())
    cfg["_config_path"] = str(path)
    cfg["_config_hash"] = config_hash(cfg)
    return cfg


def public(cfg: dict) -> dict:
    """Config without private (underscore) keys, safe to write into artifacts."""
    return copy.deepcopy({k: v for k, v in cfg.items() if not k.startswith("_")})


def field_map(cfg: dict) -> dict:
    """Which Airweave chunk metadata keys play the roles used by the algorithm (all optional).

    group:    source group used for diversity screens and leave-group-out calibration (e.g. company, tenant, source)
    segment:  coarser grouping shown to the reviewer (e.g. industry)
    doc_type: document type shown to the reviewer (e.g. form type, source connector)
    """
    f = cfg.get("input", {}).get("fields", {})
    return {"group": f.get("group", "company"), "segment": f.get("segment", "industry"), "doc_type": f.get("doc_type", "source")}
