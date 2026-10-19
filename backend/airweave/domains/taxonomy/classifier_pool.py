"""Dedicated classifier threads (plan A7; design review finding 4).

The classifier opens its SQLite embedding cache when it is created, and SQLite refuses a
connection used from another thread (``ProgrammingError``). ``asyncio.to_thread`` would run each
call on any thread of a shared pool, so calls would fail and fail-open would hide it. Instead the
pool keeps ``TAXONOMY_CLASSIFY_THREADS`` single-thread executors. Each thread creates and only
ever uses its own classifiers, cached by ``(taxonomy_version, manifest_sha256)`` in a small LRU,
with its own cache file ``<cache dir>/<pid>-<thread n>.sqlite``. When that file passes
``max_cache_mb`` the thread closes its classifiers, deletes the file and starts a new one
(``<pid>-<n>-g<generation>.sqlite``); ``close`` deletes the pool's own files too, but only
once every thread has closed its connections. A batch runs on one thread, chosen round-robin. A
timed-out batch cannot be stopped (Python cannot stop a thread), but its embedding transport
refuses every call after the batch's deadline, so it stops spending; the next batch for that
thread waits behind it.
"""

from __future__ import annotations

import asyncio
import copy
import itertools
import logging
import os
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from airweave.domains.embedders.protocols import DenseEmbedderProtocol
from airweave.domains.taxonomy.artifacts import ResolvedTaxonomy
from airweave.domains.taxonomy.embed_transport import AirweaveEmbedTransport, matches
from airweave.domains.taxonomy.metrics import TAXONOMY_TALLY, TaxonomyTally

# Loaded taxonomy versions kept per thread; memory is bounded by threads x this
CLASSIFIERS_PER_THREAD = 4

logger = logging.getLogger(__name__)

# A cache file as the pool names it: <pid>-<thread>[-g<generation>].sqlite plus SQLite's sidecars;
# a second pool using the same directory at the same time names its files <pid>-s<slot>-<thread>...
# so two live pools never share a file.
_CACHE_FILE = re.compile(r"^(\d+)-(?:s\d+-)?\d+(?:-g\d+)?\.sqlite(?:-wal|-shm|-journal)?$")

# Cache directories pools in this process use now, with the slots they hold (a second pool must not
# take the directory's same-pid files for stale ones, nor share a first pool's files)
_OWNED_DIRS: Dict[str, set] = {}
_OWNED_LOCK = threading.Lock()

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
        tally: TaxonomyTally = TAXONOMY_TALLY,
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
        if max_cache_mb < 0:
            raise ValueError("max_cache_mb must be 0 (never) or more")
        self._slot = 0
        self._n = threads
        self._cache_dir = Path(cache_dir)
        self._transport = embed_transport
        self._dense = dense_embedder
        self._bridge = AirweaveEmbedTransport(dense_embedder) if dense_embedder else None
        self._lru_size = classifiers_per_thread
        self._max_cache_bytes = int(max_cache_mb * 1024 * 1024)
        self._tally = tally
        self._dir_key = str(self._cache_dir.resolve())
        self._claim_dir_and_remove_stale_files()
        self._fallback_noted: set = set()
        self._note_lock = threading.Lock()
        self._executors: List[ThreadPoolExecutor] = []
        self._round_robin = itertools.count()
        self._local = threading.local()
        self._start_lock = threading.Lock()

    def _claim_dir_and_remove_stale_files(self) -> None:
        """Delete cache files that belong to no live process (a killed worker leaves them).

        Files of another process that is still running are never touched. Files named for this
        process's pid are stale too unless a pool in this process already owns the directory:
        a restarted container often gets the pid of the one that died.
        """
        key = str(self._cache_dir.resolve())
        with _OWNED_LOCK:
            slots = _OWNED_DIRS.setdefault(key, set())
            first = not slots
            self._slot = next(s for s in itertools.count() if s not in slots)
            slots.add(self._slot)
        self._dir_key = key
        removed = 0
        try:
            names = os.listdir(self._cache_dir)
        except OSError:
            return
        for name in names:
            found = _CACHE_FILE.match(name)
            if not found:
                continue
            pid = int(found.group(1))
            if pid == os.getpid():
                if not first:
                    continue
            elif _pid_alive(pid):
                continue
            try:
                os.remove(self._cache_dir / name)
                removed += 1
            except OSError:
                pass
        if removed:
            logger.info("removed %d stale taxonomy cache files from %s", removed, self._cache_dir)

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
        # The caller's clock starts now, not when the queued batch reaches its thread.
        deadline = time.monotonic() + timeout
        future = loop.run_in_executor(
            self._executors[index],
            self._classify_on_thread,
            index,
            taxonomy,
            rows,
            loop,
            timeout,
            deadline,
        )
        return await asyncio.wait_for(future, timeout)

    def _classify_on_thread(
        self,
        index: int,
        taxonomy: ResolvedTaxonomy,
        rows: List[Dict[str, Any]],
        loop: asyncio.AbstractEventLoop,
        timeout: float,
        deadline: float,
    ) -> List[Dict[str, Any]]:
        from taxonomy_classifier.contract import to_contract  # type: ignore[import-not-found]

        if self._bridge is not None:
            self._bridge.bind(loop, timeout, deadline)
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
        classifier = TaxonomyClassifier(
            taxonomy.local_path, config, self._transport_for(taxonomy, config)
        )
        cache[key] = classifier
        while len(cache) > self._lru_size:
            _, old = cache.popitem(last=False)
            _close(old)
        return classifier

    def _cache_path(self, index: int) -> Path:
        """This thread's current cache file; later generations carry a suffix."""
        generation = getattr(self._local, "generation", 0)
        suffix = f"-g{generation}" if generation else ""
        slot = f"s{self._slot}-" if self._slot else ""
        return self._cache_dir / f"{os.getpid()}-{slot}{index}{suffix}.sqlite"

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

    def _transport_for(
        self, taxonomy: ResolvedTaxonomy, config: Dict[str, Any]
    ) -> Optional[EmbedTransport]:
        """The explicit transport, else Airweave's embedder if the model matches, else None.

        None means the package's own client; with an embedder configured that is a fallback,
        logged and counted once per taxonomy version.
        """
        if self._transport is not None:
            return self._transport
        if self._bridge is not None:
            if matches(self._dense, config):
                return self._bridge
            self._note_fallback(taxonomy, config)
        return None

    def _note_fallback(self, taxonomy: ResolvedTaxonomy, config: Dict[str, Any]) -> None:
        version = taxonomy.taxonomy_version
        with self._note_lock:
            if version in self._fallback_noted:
                return
            self._fallback_noted.add(version)
        wanted = config.get("embedding") or {}
        logger.warning(
            "taxonomy %s embeds with %s (%s dimensions) but the deployment's embedder is %s "
            "(%s dimensions): this version uses the package's own OpenAI client, outside "
            "Airweave's limits and cost control",
            version,
            wanted.get("model"),
            wanted.get("dimensions"),
            self._dense.model_name if self._dense else None,
            self._dense.dimensions if self._dense else None,
        )
        self._tally.note_embed_fallback(version)

    def close(self) -> None:
        """Stop the threads and delete this pool's cache files (the worker is shutting down).

        Each thread first closes its own classifiers, because SQLite connections close on the
        thread that opened them. Batches already queued finish first. A thread still busy with a
        timed-out batch is given a few seconds; if it has not closed by then its files are left
        in place (a live connection to a deleted database is worse than a file), and the next
        start removes them as stale. Another pool's files are never touched.
        """
        executors, self._executors = self._executors, []
        closers = [executor.submit(self._release_thread_state) for executor in executors]
        for executor in executors:
            executor.shutdown(wait=False)  # queued closers still run
        _, pending = wait(closers, timeout=_CLOSE_WAIT_SECONDS)
        if pending:
            logger.warning(
                "%d classifier thread(s) still busy at shutdown: their cache files are left in %s",
                len(pending),
                self._cache_dir,
            )
            # The slot stays claimed until the last of those threads has stopped: a pool
            # created meanwhile would take the same slot and, as the first in the directory,
            # delete the files this classifier still has open.
            remaining = len(pending)
            lock = threading.Lock()

            def _one_stopped(_future: Any) -> None:
                nonlocal remaining
                with lock:
                    remaining -= 1
                    last = remaining == 0
                if last:
                    self._release_slot()

            for future in pending:
                future.add_done_callback(_one_stopped)
            return
        try:
            self._delete_own_files()
        finally:
            self._release_slot()

    def _release_slot(self) -> None:
        """Let another pool in this process use this pool's slot of the directory."""
        with _OWNED_LOCK:
            slots = _OWNED_DIRS.get(self._dir_key, set())
            slots.discard(self._slot)
            if not slots:
                _OWNED_DIRS.pop(self._dir_key, None)

    def _delete_own_files(self) -> None:
        slot = f"s{self._slot}-" if self._slot else ""
        own = re.compile(rf"^{os.getpid()}-{slot}\d+(?:-g\d+)?\.sqlite$")
        try:
            names = sorted(os.listdir(self._cache_dir))
        except OSError:
            return  # the directory was never created: nothing of this pool's to delete
        for path in (self._cache_dir / n for n in names):
            if own.match(path.name):
                for suffix in _SIDECARS:
                    _delete(Path(str(path) + suffix))


def _pid_alive(pid: int) -> bool:
    """True if a process with this pid exists (one we may not signal still counts as alive)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


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
