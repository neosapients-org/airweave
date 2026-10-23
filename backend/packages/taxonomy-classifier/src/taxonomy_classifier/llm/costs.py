"""Run-level API cost ledger with a hard cap (shared by embedding and LLM calls)."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path


class CostLedger:
    def __init__(self, cap_usd: float, path: Path):
        self.cap, self.path = cap_usd, Path(path)
        self.lock = threading.RLock()
        self.data = json.load(open(self.path)) if self.path.exists() else {"cap_usd": cap_usd, "total_usd": 0.0, "stages": {}}
        self.data["cap_usd"] = cap_usd
        # Reservations still on file belong to calls of an earlier process that died before their cost was known: the
        # provider may have billed them, so they count as spent (at their worst case) rather than being forgotten.
        left = float(self.data.pop("in_flight_usd", 0.0) or 0.0)
        if left > 0:
            self._charge("unconfirmed", left)
        self.data["in_flight_usd"] = 0.0
        # Written at once, not at the first paid call: a run served entirely from the caches still has a ledger (USD 0),
        # so a missing ledger always means the run never got this far.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write()

    _reserved: float = 0.0

    def _write(self) -> None:
        """Replace the ledger file atomically, so an interrupted write never leaves it half written."""
        tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=1)
        os.replace(tmp, self.path)

    def _charge(self, stage: str, cost: float) -> None:
        s = self.data["stages"].setdefault(stage, {"calls": 0, "cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0,
                                                   "reasoning_tokens": 0, "latency_s": 0.0})
        s["calls"] += 1; s["cost_usd"] += cost
        self.data["total_usd"] += cost

    def can_spend(self, worst_case: float) -> bool:
        with self.lock:
            return self.data["total_usd"] + self._reserved + worst_case <= self.cap

    def reserve(self, worst_case: float) -> bool:
        """Hold a call's worst-case cost against the cap before the call is sent; False when the cap would be passed.

        Every successful reserve is followed by release() once the call's actual cost has been add()-ed (or the call
        failed), so concurrent calls can never together pass the cap on the strength of the same headroom. The held
        amount is on file too, so a process that dies mid-call leaves it to be counted as spent by the next one."""
        with self.lock:
            if self.data["total_usd"] + self._reserved + worst_case > self.cap:
                return False
            self._reserved += worst_case
            self.data["in_flight_usd"] = self._reserved
            self._write()
            return True

    def release(self, worst_case: float) -> None:
        with self.lock:
            self._reserved = max(0.0, self._reserved - worst_case)
            self.data["in_flight_usd"] = self._reserved
            self._write()

    def charge_unconfirmed(self, cost: float) -> None:
        """Count a call whose outcome is unknown (it timed out, so the provider may have billed it) at its worst case."""
        with self.lock:
            self._charge("unconfirmed", cost)
            self._write()

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
            self._write()
