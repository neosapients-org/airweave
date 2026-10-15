"""Dedicated classifier threads (plan A7; design review finding 4).

The classifier opens its SQLite embedding cache when it is created, and SQLite refuses a
connection used from another thread (``ProgrammingError``). ``asyncio.to_thread`` would run each
call on any thread of a shared pool, so calls would fail and fail-open would hide it. Instead the
pool keeps ``TAXONOMY_CLASSIFY_THREADS`` single-thread executors. Each thread creates and only
ever uses its own classifiers, cached by ``(taxonomy_version, manifest_sha256)`` in a small LRU,
with its own cache file ``<cache dir>/<pid>-<thread n>.sqlite``. A batch runs on one thread,
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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from airweave.domains.embedders.protocols import DenseEmbedderProtocol
from airweave.domains.taxonomy.artifacts import ResolvedTaxonomy
from airweave.domains.taxonomy.embed_transport import AirweaveEmbedTransport, matches

# Loaded taxonomy versions kept per thread; memory is bounded by threads x this
CLASSIFIERS_PER_THREAD = 4

EmbedTransport = Callable[[List[str]], Any]


class ClassifierPool:
    """Runs classifier calls on dedicated threads, one classifier set per thread."""

    def __init__(
        self,
        threads: int,
        cache_dir: str | Path,
        embed_transport: Optional[EmbedTransport] = None,
        classifiers_per_thread: int = CLASSIFIERS_PER_THREAD,
        dense_embedder: Optional[DenseEmbedderProtocol] = None,
    ) -> None:
        """Initialize; threads start on first use.

        ``embed_transport`` replaces the classifier's own OpenAI embeddings call (tests use a
        fake). Otherwise a taxonomy whose embedding model and dimensions are the
        ``dense_embedder``'s embeds through it (Airweave's timeouts, retries and limits apply);
        any other taxonomy keeps the package's default client, which reads ``OPENAI_API_KEY``.
        """
        if threads < 1:
            raise ValueError("threads must be at least 1")
        self._n = threads
        self._cache_dir = Path(cache_dir)
        self._transport = embed_transport
        self._dense = dense_embedder
        self._bridge = AirweaveEmbedTransport(dense_embedder) if dense_embedder else None
        self._lru_size = classifiers_per_thread
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
        config.setdefault("embedding", {})["cache_path"] = str(
            self._cache_dir / f"{os.getpid()}-{index}.sqlite"
        )
        classifier = TaxonomyClassifier(taxonomy.local_path, config, self._transport_for(config))
        cache[key] = classifier
        while len(cache) > self._lru_size:
            _, old = cache.popitem(last=False)
            _close(old)
        return classifier

    def _transport_for(self, config: Dict[str, Any]) -> Optional[EmbedTransport]:
        """The explicit transport, else Airweave's embedder if the model matches, else None."""
        if self._transport is not None:
            return self._transport
        if self._bridge is not None and matches(self._dense, config):
            return self._bridge
        return None

    def close(self) -> None:
        """Stop the threads; queued batches are abandoned (the process is shutting down)."""
        for executor in self._executors:
            executor.shutdown(wait=False, cancel_futures=True)
        self._executors = []


def _close(classifier: Any) -> None:
    """Close an evicted classifier's cache connection (on its own thread)."""
    try:
        classifier.embedder.cache.close()
    except Exception:
        pass
