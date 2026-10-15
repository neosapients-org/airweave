"""The container builds the taxonomy step only when enabled, on Airweave's own embedder."""

from types import SimpleNamespace

from airweave.core.container.factory import _create_taxonomy_processor


def _settings(enabled: bool, bucket: str = ""):
    return SimpleNamespace(
        TAXONOMY_ENABLED=enabled,
        TAXONOMY_S3_BUCKET=bucket,
        TAXONOMY_S3_PREFIX="taxonomy/",
        TAXONOMY_LOCAL_DIR="/tmp/none",
        TAXONOMY_ACTIVE_CACHE_SECONDS=300,
        TAXONOMY_EMBED_CACHE_DIR="/tmp/none-cache",
        TAXONOMY_EMBED_CACHE_MAX_MB=7,
        TAXONOMY_CLASSIFY_THREADS=2,
        TAXONOMY_CLASSIFY_TIMEOUT_SECONDS=60,
    )


def test_off_builds_nothing():
    assert _create_taxonomy_processor(_settings(False), object()) is None


def test_on_hands_the_dense_embedder_to_the_pool():
    dense = object()
    processor = _create_taxonomy_processor(_settings(True), dense)
    assert processor._pool._dense is dense
    assert processor._pool._bridge is not None


def test_on_without_an_embedder_keeps_the_packages_client():
    processor = _create_taxonomy_processor(_settings(True))
    assert processor._pool._bridge is None


def test_cache_limit_comes_from_the_setting():
    processor = _create_taxonomy_processor(_settings(True))
    assert processor._pool._max_cache_bytes == 7 * 1024 * 1024


def test_processor_close_closes_the_pool():
    processor = _create_taxonomy_processor(_settings(True))
    closed = []
    processor._pool.close = lambda: closed.append(True)
    processor.close()
    assert closed == [True]
