"""Classifier embeddings through Airweave's own dense embedder (batch 3, item 9).

The classifier's embeddings call is a plain synchronous callable ``list[str] -> (vectors,
tokens)`` that runs on one of the pool's classifier threads. This one hands each batch to the
event loop that owns the dense embedder and waits for the answer, so the embedder's own client
timeout, retries, concurrency limit and error translation apply instead of the package's
urllib client. The only thing added here is a deadline for the batch, set by the pool per call.

It is used only when the taxonomy's embedding model and dimensions equal the embedder's: vectors
from two models are not comparable, so any other taxonomy keeps the package's client.
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from airweave.domains.embedders.protocols import DenseEmbedderProtocol


def matches(embedder: Optional[DenseEmbedderProtocol], config: Dict[str, Any]) -> bool:
    """True when the taxonomy's embedding model and dimensions are the embedder's."""
    if embedder is None:
        return False
    wanted = config.get("embedding") or {}
    return (
        wanted.get("model") == embedder.model_name
        and wanted.get("dimensions") == embedder.dimensions
    )


class AirweaveEmbedTransport:
    """Synchronous embeddings transport backed by an async ``DenseEmbedderProtocol``.

    ``bind`` sets the loop and the deadline for the calling thread; the pool binds before every
    batch, so one transport serves every classifier on the thread.
    """

    def __init__(self, embedder: DenseEmbedderProtocol) -> None:
        """Initialize with the deployment's dense embedder."""
        self._embedder = embedder
        self._local = threading.local()

    def bind(self, loop: asyncio.AbstractEventLoop, timeout: float) -> None:
        """Set the loop that owns the embedder and the deadline, for this thread."""
        self._local.loop = loop
        self._local.timeout = timeout

    def __call__(self, batch: List[str]) -> Tuple[List[np.ndarray], int]:
        """Embed one batch; raises whatever the embedder raised, or ``TimeoutError``."""
        loop = getattr(self._local, "loop", None)
        if loop is None:
            raise RuntimeError("embed transport used before bind() on this thread")
        future = asyncio.run_coroutine_threadsafe(self._embedder.embed_many(list(batch)), loop)
        try:
            embeddings = future.result(self._local.timeout)
        except FutureTimeout:
            future.cancel()
            raise TimeoutError("embedding batch exceeded its deadline") from None
        vectors = [np.asarray(e.vector, dtype=np.float32) for e in embeddings]
        return vectors, _count_tokens(batch)


def _count_tokens(batch: List[str]) -> int:
    """Tokens of the batch for the classifier's own stats (the embedder reports none)."""
    try:
        import tiktoken

        encoder = tiktoken.get_encoding("cl100k_base")
        return sum(len(encoder.encode(t, disallowed_special=())) for t in batch)
    except Exception:
        return sum(len(t) // 4 for t in batch)
