"""Fakes for the discover and propose tests: S3, the package's gateway, a clock. Invented data."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from airweave.domains.storage.exceptions import StorageNotFoundError
from airweave.domains.taxonomy.jobs.gateway import BuildOutput
from airweave.domains.taxonomy.jobs.records import KIND_DISCOVER, TaxonomyJobRecord
from airweave.domains.taxonomy.relabel.tests.helpers import (  # noqa: F401
    COLLECTION,
    NEW,
    OLD,
    ORG,
    READABLE,
    FakeChunkStore,
    FakeResolver,
    chunk,
    taxonomy,
)

TENANT = "t-acme"
CANDIDATE = "acme-first"


class FakeS3:
    """In-memory bucket with S3's conditional put (``If-None-Match: *``)."""

    def __init__(self) -> None:
        self.objects: Dict[str, bytes] = {}
        self.writes: List[str] = []
        self.fail_on: Optional[str] = None

    async def write_file_once(self, path: str, content: bytes) -> bool:
        if self.fail_on and self.fail_on in path:
            raise RuntimeError("S3 unavailable")
        if path in self.objects:
            return False
        self.objects[path] = content
        self.writes.append(path)
        return True

    async def read_file(self, path: str) -> bytes:
        if path not in self.objects:
            raise StorageNotFoundError(path)
        return self.objects[path]

    async def delete(self, path: str) -> bool:
        return self.objects.pop(path, None) is not None

    def under(self, prefix: str) -> List[str]:
        return sorted(p for p in self.objects if p.startswith(prefix))


class FakeGateway:
    """The package behind its interface: returns fixed files and a report, records the calls."""

    def __init__(self, cost: float = 0.02, files: Optional[Dict[str, bytes]] = None) -> None:
        self.can_discover = True
        self.cost = cost
        self.files = files or {
            "final_taxonomy.json": b'{"categories": []}',
            "taxonomy.yaml": b"assignment: {}\n",
            "dense_prototypes.npy": b"npy",
        }
        self.calls: List[Dict[str, Any]] = []
        self.error: Optional[BaseException] = None
        self.delay = 0.0
        self.on_call: Any = None
        self.threads: List[int] = []

    def _run(self, kind: str, chunks, **kw) -> BuildOutput:
        self.threads.append(threading.get_ident())
        self.calls.append({"kind": kind, "chunks": chunks, **kw})
        if self.on_call:
            self.on_call()
        if self.delay:
            time.sleep(self.delay)
        if self.error:
            raise self.error
        report = {
            "kind": kind,
            "categories": 2,
            "clusters": [{"name": "Loans"}],
            "cost_usd": self.cost,
        }
        return BuildOutput(dict(self.files), report, self.cost, 2)

    def discover(self, chunks, **kw) -> BuildOutput:
        return self._run("discover", chunks, **kw)

    def propose(self, chunks, **kw) -> BuildOutput:
        return self._run("propose", chunks, **kw)


def job(kind: str = KIND_DISCOVER, **kw) -> TaxonomyJobRecord:
    base = {
        "id": uuid4(),
        "organization_id": ORG,
        "kind": kind,
        "collection_id": COLLECTION,
        "collection_readable_id": READABLE,
        "tenant_id": TENANT,
        "candidate_version": CANDIDATE,
        "idempotency_key": "k1",
        "spend_cap_usd": 1.0,
    }
    base.update(kw)
    return TaxonomyJobRecord(**base)


def docs(
    n: int = 14, chunks_each: int = 2, status: Optional[str] = None, version: Optional[str] = None
):
    """``n`` documents with ``chunks_each`` text chunks each, in the fake Vespa's format."""
    out = []
    for d in range(n):
        for c in range(chunks_each):
            out.append(
                chunk(
                    c,
                    doc=f"doc-{d:02d}",
                    body=f"Acme loan note {d} part {c}. Late payments rose.",
                    status=status,
                    version=version,
                )
            )
    return out


def work_dir(tmp_path: Path) -> Path:
    return tmp_path / "jobs"


UUID_ONE = UUID(int=1)
