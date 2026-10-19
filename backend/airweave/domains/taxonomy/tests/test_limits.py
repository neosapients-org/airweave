"""The taxonomy's fixed numbers and folder layout."""

from airweave.domains.taxonomy import limits


def test_the_data_folder_holds_versions_embed_caches_and_job_scratch():
    assert limits.versions_dir("/data/airweave-taxonomy") == "/data/airweave-taxonomy/versions"
    assert (
        limits.embed_cache_dir("/data/airweave-taxonomy") == "/data/airweave-taxonomy/embed-cache"
    )
    assert limits.jobs_work_dir("/data/airweave-taxonomy") == "/data/airweave-taxonomy/jobs"


def test_the_numbers_are_sane():
    assert limits.ARTIFACT_MAX_MB <= limits.ARTIFACT_TOTAL_MAX_MB
    assert limits.RELABEL_USD_PER_THOUSAND_CHUNKS > 0
    for name in (
        "RELABEL_BATCH_SIZE",
        "RELABEL_PAGE_SIZE",
        "RELABEL_WRITE_CONCURRENCY",
        "RELABEL_LEASE_SECONDS",
        "RELABEL_RESUME_INTERVAL_SECONDS",
        "JOBS_LEASE_SECONDS",
        "JOBS_MAX_CHUNKS",
        "DISCOVER_SAMPLE_DOCUMENTS",
        "DISCOVER_MIN_DOCUMENTS",
        "JOBS_RECONCILE_SECONDS",
        "JOBS_DISPATCH_GRACE_SECONDS",
        "JOBS_MAX_AGE_HOURS",
        "ACTIVE_CACHE_SECONDS",
        "CLASSIFY_TIMEOUT_SECONDS",
    ):
        assert getattr(limits, name) >= 1, name


def test_a_retired_prefix_setting_is_refused_instead_of_orphaning_the_artifacts():
    import pytest

    from airweave.domains.taxonomy.artifacts import refuse_a_retired_prefix

    refuse_a_retired_prefix({})
    refuse_a_retired_prefix({"TAXONOMY_S3_PREFIX": "taxonomy"})
    refuse_a_retired_prefix({"TAXONOMY_S3_PREFIX": "taxonomy/"})
    with pytest.raises(RuntimeError, match="no longer a setting"):
        refuse_a_retired_prefix({"TAXONOMY_S3_PREFIX": "tax"})
