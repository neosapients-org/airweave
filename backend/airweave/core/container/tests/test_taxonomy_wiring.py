"""The container builds the taxonomy step only when enabled, on Airweave's own embedder."""

from types import SimpleNamespace

from airweave.core.container.factory import _create_taxonomy_processor
from airweave.domains.taxonomy import limits


def _settings(enabled: bool, bucket: str = ""):
    return SimpleNamespace(
        TAXONOMY_ENABLED=enabled,
        TAXONOMY_S3_BUCKET=bucket,
        TAXONOMY_DATA_DIR="/tmp/none",
        TAXONOMY_CLASSIFY_THREADS=2,
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


def test_cache_limit_is_the_constant():
    processor = _create_taxonomy_processor(_settings(True))
    assert processor._pool._max_cache_bytes == limits.EMBED_CACHE_MAX_MB * 1024 * 1024


def test_the_data_folder_holds_the_versions_and_the_embed_caches():
    processor = _create_taxonomy_processor(_settings(True))
    assert str(processor.resolver._local_dir) == "/tmp/none/versions"
    assert str(processor._pool._cache_dir) == "/tmp/none/embed-cache"


def test_processor_close_closes_the_pool():
    processor = _create_taxonomy_processor(_settings(True))
    closed = []
    processor._pool.close = lambda: closed.append(True)
    processor.close()
    assert closed == [True]
