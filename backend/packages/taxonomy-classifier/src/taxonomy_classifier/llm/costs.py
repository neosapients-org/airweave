"""Run-level API cost ledger with a hard cap (shared by embedding and LLM calls)."""

from __future__ import annotations

import json
import threading
from pathlib import Path


class CostLedger:
    def __init__(self, cap_usd: float, path: Path):
        self.cap, self.path = cap_usd, Path(path)
        self.lock = threading.RLock()
        self.data = json.load(open(self.path)) if self.path.exists() else {"cap_usd": cap_usd, "total_usd": 0.0, "stages": {}}
        self.data["cap_usd"] = cap_usd
        # Written at once, not at the first paid call: a run served entirely from the caches still has a ledger (USD 0),
        # so a missing ledger always means the run never got this far.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(self.data, open(self.path, "w"), indent=1)

    def can_spend(self, worst_case: float) -> bool:
        with self.lock:
            return self.data["total_usd"] + worst_case <= self.cap

    def add(self, stage: str, cost: float, usage: dict | None = None, latency: float = 0.0, calls: int = 1):
        with self.lock:
            s = self.data["stages"].setdefault(stage, {"calls": 0, "cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0,
                                                       "reasoning_tokens": 0, "latency_s": 0.0})
            s["calls"] += calls; s["cost_usd"] += cost; s["latency_s"] += latency
            if usage:
                s["input_tokens"] += usage.get("prompt_tokens", usage.get("total_tokens", 0))
                s["output_tokens"] += usage.get("completion_tokens", 0)
                s["reasoning_tokens"] += (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
            self.data["total_usd"] += cost
            json.dump(self.data, open(self.path, "w"), indent=1)
