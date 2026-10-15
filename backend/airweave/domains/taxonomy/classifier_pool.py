"""Dedicated classifier threads (plan A7; design review finding 4).

The classifier opens its SQLite embedding cache when it is created, and SQLite refuses a
connection used from another thread (``ProgrammingError``). ``asyncio.to_thread`` would run each
call on any thread of a shared pool, so calls would fail and fail-open would hide it. Instead the
pool keeps ``TAXONOMY_CLASSIFY_THREADS`` single-thread executors. Each thread creates and only
ever uses its own classifiers, cached by ``(taxonomy_version, manifest_sha256)`` in a small LRU,
with its own cache file ``<cache dir>/<pid>-<thread n>.sqlite``. When that file passes
``max_cache_mb`` the thread closes its classifiers, deletes the file and starts a new one
(``<pid>-<n>-g<generation>.sqlite``); ``close`` deletes the files too. A batch runs on one thread,
chosen round-robin; a timed-out batch still finishes on its thread, and the next batch for that
thread waits behind it.
"""

from __future__ import annotations

import asyncio
import copy
import itertools
import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from airweave.domains.embedders.protocols import DenseEmbedderProtocol
from airweave.domains.taxonomy.artifacts import ResolvedTaxonomy
from airweave.domains.taxonomy.embed_transport import AirweaveEmbedTransport, matches

# Loaded taxonomy versions kept per thread; memory is bounded by threads x this
CLASSIFIERS_PER_THREAD = 4

EmbedTransport = Callable[[List[str]], Any]

# SQLite writes these next to the main file
_SIDECARS = ("", "-wal", "-shm", "-journal")
# How long close() waits for each thread to close its connections before deleting the files
_CLOSE_WAIT_SECONDS = 5.0


class ClassifierPool:
    """Runs classifier calls on dedicated threads, one classifier set per thread."""

    def __init__(
        self,
        threads: int,
        cache_dir: str | Path,
        embed_transport: Optional[EmbedTransport] = None,
        classifiers_per_thread: int = CLASSIFIERS_PER_THREAD,
        dense_embedder: Optional[DenseEmbedderProtocol] = None,
        max_cache_mb: float = 512,
    ) -> None:
        """Initialize; threads start on first use.

        ``embed_transport`` replaces the classifier's own OpenAI embeddings call (tests use a
        fake). Otherwise a taxonomy whose embedding model and dimensions are the
        ``dense_embedder``'s embeds through it (Airweave's timeouts, retries and limits apply);
        any other taxonomy keeps the package's default client, which reads ``OPENAI_API_KEY``.
        ``max_cache_mb`` (0 = never) is the size at which a thread starts a new cache file.
        """
        if threads < 1:
            raise ValueError("threads must be at least 1")
        self._n = threads
        self._cache_dir = Path(cache_dir)
        self._transport = embed_transport
        self._dense = dense_embedder
        self._bridge = AirweaveEmbedTransport(dense_embedder) if dense_embedder else None
        self._lru_size = classifiers_per_thread
        self._max_cache_bytes = int(max_cache_mb * 1024 * 1024)
        self._executors: List[ThreadPoolExecutor] = []
        self._round_robin = itertools.count()
        self._local = threading.local()
        self._start_lock = threading.Lock()

    @property
    def threads(self) -> int:
        """Number of classifier threads."""
        return self._n

    def _ensure_started(self) -> None:
        with self._start_lock:
            if not self._executors:
                self._executors = [
                    ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"taxonomy-classify-{i}")
                    for i in range(self._n)
                ]

    async def classify(
        self, taxonomy: ResolvedTaxonomy, rows: List[Dict[str, Any]], timeout: float
    ) -> List[Dict[str, Any]]:
        """Classify one batch on the next thread; returns one contract record per row.

        Raises ``asyncio.TimeoutError`` after ``timeout`` seconds, and whatever the classifier
        raised otherwise.
        """
        self._ensure_started()
        index = next(self._round_robin) % self._n
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            self._executors[index], self._classify_on_thread, index, taxonomy, rows, loop, timeout
        )
        return await asyncio.wait_for(future, timeout)

    def _classify_on_thread(
        self,
        index: int,
        taxonomy: ResolvedTaxonomy,
        rows: List[Dict[str, Any]],
        loop: asyncio.AbstractEventLoop,
        timeout: float,
    ) -> List[Dict[str, Any]]:
        from taxonomy_classifier.contract import to_contract  # type: ignore[import-not-found]

        if self._bridge is not None:
            self._bridge.bind(loop, timeout)
        self._rotate_cache(index)
        classifier = self._classifier(index, taxonomy)
        result = classifier.classify(rows)
        return [to_contract(e, classifier.taxonomy) for e in result.enriched]

    def _classifier(self, index: int, taxonomy: ResolvedTaxonomy) -> Any:
        """This thread's classifier for the version, created here and used only here."""
        from taxonomy_classifier.integration.airweave import (  # type: ignore[import-not-found]
            TaxonomyClassifier,
        )

        cache: "OrderedDict[Any, Any]" = getattr(self._local, "classifiers", None) or OrderedDict()
        self._local.classifiers = cache
        key = taxonomy.cache_key
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        config = copy.deepcopy(taxonomy.config)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        config.setdefault("embedding", {})["cache_path"] = str(self._cache_path(index))
        classifier = TaxonomyClassifier(taxonomy.local_path, config, self._transport_for(config))
        cache[key] = classifier
        while len(cache) > self._lru_size:
            _, old = cache.popitem(last=False)
            _close(old)
        return classifier

    def _cache_path(self, index: int) -> Path:
        """This thread's current cache file; later generations carry a suffix."""
        generation = getattr(self._local, "generation", 0)
        suffix = f"-g{generation}" if generation else ""
        return self._cache_dir / f"{os.getpid()}-{index}{suffix}.sqlite"

    def _rotate_cache(self, index: int) -> None:
        """Start a new cache file once the thread's file is over the limit (on its thread).

        Every loaded classifier shares the file, so all of them are closed and reopened lazily
        on the new one; the full file is deleted. Vectors in it are recomputed on demand.
        """
        if self._max_cache_bytes <= 0:
            return
        path = self._cache_path(index)
        if _size(path) <= self._max_cache_bytes:
            return
        self._release_thread_state()
        _delete(path)
        self._local.generation = getattr(self._local, "generation", 0) + 1

    def _release_thread_state(self) -> None:
        """Close this thread's classifiers (run on the thread that owns them)."""
        cache = getattr(self._local, "classifiers", None)
        while cache:
            _, old = cache.popitem(last=False)
            _close(old)

    def _transport_for(self, config: Dict[str, Any]) -> Optional[EmbedTransport]:
        """The explicit transport, else Airweave's embedder if the model matches, else None."""
        if self._transport is not None:
            return self._transport
        if self._bridge is not None and matches(self._dense, config):
            return self._bridge
        return None

    def close(self) -> None:
        """Stop the threads and delete this process's cache files (the worker is shutting down).

        Each thread first closes its own classifiers, because SQLite connections close on the
        thread that opened them. Batches already queued finish first; a thread still busy
        with a timed-out batch is given a few seconds, then its files are deleted anyway.
        """
        executors, self._executors = self._executors, []
        closers = [executor.submit(self._release_thread_state) for executor in executors]
        for executor in executors:
            executor.shutdown(wait=False)  # queued closers still run
        wait(closers, timeout=_CLOSE_WAIT_SECONDS)
        for path in self._cache_dir.glob(f"{os.getpid()}-*.sqlite"):
            for suffix in _SIDECARS:
                _delete(Path(str(path) + suffix))


def _size(path: Path) -> int:
    """File size including SQLite's write-ahead log; 0 when absent."""
    total = 0
    for suffix in _SIDECARS[:2]:
        try:
            total += os.path.getsize(str(path) + suffix)
        except OSError:
            pass
    return total


def _delete(path: Path) -> None:
    """Remove a cache file and its SQLite sidecars; missing files are fine."""
    for suffix in _SIDECARS:
        try:
            os.remove(str(path) + suffix)
        except OSError:
            pass


def _close(classifier: Any) -> None:
    """Close an evicted classifier's cache connection (on its own thread)."""
    try:
        classifier.embedder.cache.close()
    except Exception:
        pass
