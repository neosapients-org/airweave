"""The ONLY component that calls a chat LLM (used by taxonomy build and novelty maintenance, never by classification).

Every call:
  * uses a versioned prompt loaded from llm/prompts/<version>.system.txt / .user.txt / .schema.json
    ({corpus_description} in a system prompt is filled from config llm.corpus_description)
  * is cached by input hash = SHA256(model | prompt_version | system | user | reasoning_effort | max_tokens)
  * is written in full to <output>/llm_calls/<request_id>.json (system, user, settings, raw response, parsed response,
    usage, cost, input hash, timestamp) and appended to <output>/llm_calls.jsonl
  * is validated against the JSON schema; invalid -> retried once -> otherwise status FAILED (never filled in)
  * is refused when the run's hard cost cap could be exceeded (status BUDGET_EXCEEDED)
The API key is read from the environment (OPENAI_API_KEY) or ./.env at call time and never logged or stored.
`transport` can be injected (tests): a callable(body: dict) -> OpenAI chat-completions response dict.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from ..config import PROMPTS_DIR, resolve
from .costs import CostLedger
_LOCK = threading.RLock()


class LLMFailure(RuntimeError):
    pass


def api_key() -> str:
    k = os.environ.get("OPENAI_API_KEY")
    if k:
        return k
    p = Path.cwd() / ".env"                         # fallback for local runs: ./.env
    if p.exists():
        for line in p.read_text().splitlines():
            if line.startswith("OPENAI_API_KEY="):
                v = line.split("=", 1)[1].strip().strip('"').strip("'")
                if v:
                    return v
    raise LLMFailure("OPENAI_API_KEY is not set (environment or .env)")


def load_prompt(version: str, corpus_description: str | None = None) -> tuple[str, str, dict]:
    system = (PROMPTS_DIR / f"{version}.system.txt").read_text()
    user = (PROMPTS_DIR / f"{version}.user.txt").read_text()
    schema = json.loads((PROMPTS_DIR / f"{version}.schema.json").read_text())
    if "{corpus_description}" in system:
        if not corpus_description:
            raise LLMFailure(f"prompt {version} needs config llm.corpus_description")
        system = system.replace("{corpus_description}", corpus_description)
    return system, user, schema


def validate(obj, schema, path="$"):
    """Minimal JSON-schema validator for the subset used here (type, enum, required, properties, items,
    additionalProperties=false). Returns a list of error strings."""
    errs = []
    t = schema.get("type")
    types = t if isinstance(t, list) else [t] if t else []
    pymap = {"object": dict, "array": list, "string": str, "number": (int, float), "integer": int, "boolean": bool, "null": type(None)}
    if types and not any(isinstance(obj, pymap[x]) and not (x in ("number", "integer") and isinstance(obj, bool)) for x in types):
        return [f"{path}: expected {types}, got {type(obj).__name__}"]
    if "enum" in schema and obj not in schema["enum"]:
        errs.append(f"{path}: {obj!r} not in enum")
    if isinstance(obj, dict):
        for r in schema.get("required", []):
            if r not in obj:
                errs.append(f"{path}: missing {r}")
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            errs += [f"{path}: unexpected key {k}" for k in obj if k not in props]
        for k, v in obj.items():
            if k in props:
                errs += validate(v, props[k], f"{path}.{k}")
    if isinstance(obj, list) and "items" in schema:
        for i, v in enumerate(obj):
            errs += validate(v, schema["items"], f"{path}[{i}]")
    return errs


FRAMING_TOKENS = 100   # chat-format tokens around the messages (a few per message): generous on purpose


def _read_cache(path: Path) -> dict | None:
    """A cached response, or None when there is none or it cannot be read (a torn or corrupt entry is a miss)."""
    try:
        cached = json.loads(path.read_text())
    except (OSError, ValueError):                       # missing, unreadable, or torn by a concurrent writer
        return None
    return cached if isinstance(cached, dict) and cached.get("status") == "OK" else None


def _write_cache(path: Path, rec: dict) -> None:
    """Write a cache entry so that a reader never sees it half written: a temporary file, then an atomic rename."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(rec, indent=1))
    os.replace(tmp, path)


class LLMClient:
    def __init__(self, cfg: dict, run_dir: Path, ledger: CostLedger, stage: str, prompt_version: str,
                 reasoning_effort: str, max_output_tokens: int, transport=None):
        self.cfg, self.run_dir, self.ledger = cfg, Path(run_dir), ledger
        self.stage, self.version = stage, prompt_version
        self.model = cfg["llm"]["model"]
        self.effort, self.max_tokens = reasoning_effort, max_output_tokens
        self.system, self.user_template, self.schema = load_prompt(prompt_version, cfg["llm"].get("corpus_description"))
        self._schema_json = json.dumps(self.schema, sort_keys=True)   # part of the cache key and of the request's size
        self.cache_dir = resolve(cfg["llm"].get("cache_dir", "cache/llm"))   # a different dir => no reuse of other runs' responses
        self.transport = transport
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "llm_calls").mkdir(parents=True, exist_ok=True)
        p = cfg["llm"]["pricing"]
        self.price = (p["input"], p["cached_input"], p["output"])

    def render(self, **fields) -> str:
        return self.user_template.format(**fields)

    def _worst_case_usd(self, user: str) -> float:
        """The most the call can cost. A byte-level BPE token covers at least one UTF-8 byte, so the request's bytes
        (system, user and the JSON schema, which is sent too) bound its input tokens for any language, plus an
        allowance for the chat framing; output is bounded by max_completion_tokens."""
        in_tokens = len(self.system.encode()) + len(user.encode()) + len(self._schema_json.encode()) + FRAMING_TOKENS
        return (in_tokens * self.price[0] + self.max_tokens * self.price[2]) / 1e6

    def _cost(self, u):
        inp, out = u["prompt_tokens"], u["completion_tokens"]
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
        return ((inp - cached) * self.price[0] + cached * self.price[1] + out * self.price[2]) / 1e6

    def call(self, user: str, subject_id: str, extra_meta: dict | None = None) -> dict:
        """Returns the saved call record (with 'status' in OK / FAILED / BUDGET_EXCEEDED and 'parsed')."""
        input_hash = hashlib.sha256("\x1f".join([self.model, self.version, self.system, user, self.effort,
                                                  str(self.max_tokens), self._schema_json]).encode()).hexdigest()
        request_id = f"{self.stage}-{subject_id}-{input_hash[:12]}"
        cache_file = self.cache_dir / f"{input_hash}.json"
        base = {"request_id": request_id, "stage": self.stage, "prompt_version": self.version, "subject_id": subject_id,
                "model": self.model, "settings": {"reasoning_effort": self.effort, "max_completion_tokens": self.max_tokens,
                                                  "response_format": "json_schema(strict)", "temperature": "not set (API default)"},
                "system_prompt": self.system, "user_payload": user, "input_hash": input_hash, **(extra_meta or {})}
        cached = _read_cache(cache_file)
        if cached is not None:
            rec = {**base, **{k: cached[k] for k in ("raw_response", "parsed", "usage", "status", "attempts", "response_model",
                                                       "finish_reason", "original_timestamp", "validation_errors")},
                   "cost_usd": 0.0, "cache_hit": True, "timestamp": datetime.now(timezone.utc).isoformat(),
                   "original_cost_usd": cached.get("cost_usd", 0.0)}
            self._save(rec)
            return rec
        attempts, errors = [], []
        rec = None
        for attempt in range(2):                       # invalid_response_policy: retry once, then FAILED
            worst = self._worst_case_usd(user)
            if not self.ledger.reserve(worst):
                rec = {**base, "status": "BUDGET_EXCEEDED", "parsed": None, "raw_response": None, "usage": None,
                       "cost_usd": 0.0, "cache_hit": False, "attempts": attempts, "validation_errors": ["budget cap"]}
                break
            try:
                try:
                    resp, lat = self._post(user)
                except LLMFailure as e:                # transport failure after retries -> FAILED, never filled in
                    attempts.append({"attempt": attempt + 1, "finish_reason": None, "usage": None, "cost_usd": 0.0,
                                     "latency_s": 0.0, "validation_errors": [f"transport: {e}"], "raw_response": None})
                    errors = [f"transport: {e}"]
                    break
                u = resp["usage"]; c = self._cost(u)
                self.ledger.add(self.stage, c, u, lat)
            finally:
                self.ledger.release(worst)
            content = resp["choices"][0]["message"].get("content") or ""
            finish = resp["choices"][0].get("finish_reason")
            try:
                parsed = json.loads(content)
                errs = validate(parsed, self.schema)
            except json.JSONDecodeError as e:
                parsed, errs = None, [f"json decode: {e}"]
            attempts.append({"attempt": attempt + 1, "finish_reason": finish, "usage": u, "cost_usd": c, "latency_s": lat,
                             "validation_errors": errs, "raw_response": content})
            if not errs:
                rec = {**base, "status": "OK", "parsed": parsed, "raw_response": content, "usage": u, "finish_reason": finish,
                       "response_model": resp.get("model"), "cost_usd": sum(a["cost_usd"] for a in attempts), "cache_hit": False,
                       "attempts": attempts, "validation_errors": []}
                break
            errors = errs
        if rec is None:
            rec = {**base, "status": "FAILED", "parsed": None, "raw_response": attempts[-1]["raw_response"] if attempts else None,
                   "usage": attempts[-1]["usage"] if attempts else None, "finish_reason": attempts[-1]["finish_reason"] if attempts else None,
                   "response_model": None, "cost_usd": sum(a["cost_usd"] for a in attempts), "cache_hit": False,
                   "attempts": attempts, "validation_errors": errors}
        rec["timestamp"] = datetime.now(timezone.utc).isoformat()
        rec["original_timestamp"] = rec["timestamp"]
        if rec["status"] == "OK":                      # only successful responses are cached; failures are retried next run
            _write_cache(cache_file, rec)
        self._save(rec)
        return rec

    def _post(self, user):
        body = {"model": self.model, "reasoning_effort": self.effort, "max_completion_tokens": self.max_tokens,
                "messages": [{"role": "system", "content": self.system}, {"role": "user", "content": user}],
                "response_format": {"type": "json_schema", "json_schema": {"name": self.version, "strict": True, "schema": self.schema}}}
        if self.transport is not None:
            t = time.perf_counter()
            return self.transport(body), round(time.perf_counter() - t, 3)
        rc = self.cfg["llm"]["retry"]
        req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"})
        for i in range(rc["max_attempts"]):
            t = time.perf_counter()
            try:
                return json.load(urllib.request.urlopen(req, timeout=300)), round(time.perf_counter() - t, 3)
            except urllib.error.HTTPError as e:
                msg = e.read().decode()[:400]
                if e.code in rc["retry_on_http"] and i < rc["max_attempts"] - 1:
                    time.sleep(rc["backoff_s"] * 2 ** i); continue
                raise LLMFailure(f"HTTP {e.code}: {msg}")
            except (urllib.error.URLError, TimeoutError) as e:
                if i < rc["max_attempts"] - 1:
                    time.sleep(rc["backoff_s"] * 2 ** i); continue
                raise LLMFailure(str(e))

    def _save(self, rec):
        with _LOCK:
            json.dump(rec, open(self.run_dir / "llm_calls" / f"{rec['request_id']}.json", "w"), indent=1)
            slim = {k: rec.get(k) for k in ("request_id", "stage", "prompt_version", "subject_id", "model", "status", "cache_hit",
                                             "input_hash", "cost_usd", "timestamp")}
            slim["usage"] = rec.get("usage")
            with open(self.run_dir / "llm_calls.jsonl", "a") as f:
                f.write(json.dumps(slim) + "\n")
