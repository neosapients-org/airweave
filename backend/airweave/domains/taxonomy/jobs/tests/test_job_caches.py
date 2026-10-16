"""The jobs' embedding and LLM caches: per job by default, per tenant when a folder is set."""

class TestJobCaches:
    def test_default_is_one_cache_per_job(self, tmp_path):
        from airweave.domains.taxonomy.jobs.gateway import PackageGateway

        cfg = PackageGateway({})._job_config(tmp_path / "job", 1.0, tenant_id="t1")
        assert cfg["embedding"]["cache_path"] == str(tmp_path / "job" / "embeddings.sqlite")
        assert cfg["llm"]["cache_dir"] == str(tmp_path / "job" / "llm_cache")

    def test_kept_caches_are_per_tenant_and_shared_between_jobs(self, tmp_path):
        from airweave.domains.taxonomy.jobs.gateway import PackageGateway

        gw = PackageGateway({}, cache_dir=str(tmp_path / "keep"))
        a1 = gw._job_config(tmp_path / "job1", 1.0, tenant_id="tenant-a")
        a2 = gw._job_config(tmp_path / "job2", 2.0, tenant_id="tenant-a")
        b = gw._job_config(tmp_path / "job3", 1.0, tenant_id="tenant-b")
        assert a1["embedding"]["cache_path"] == a2["embedding"]["cache_path"]
        assert a1["llm"]["cache_dir"] == a2["llm"]["cache_dir"]
        assert a1["embedding"]["cache_path"] != b["embedding"]["cache_path"]
        assert str(tmp_path / "keep") in a1["llm"]["cache_dir"]
        assert "tenant-a" not in a1["llm"]["cache_dir"]
        assert (a1["llm"]["max_cost_usd"], a2["llm"]["max_cost_usd"]) == (1.0, 2.0)

    def test_without_a_tenant_the_job_folder_is_used(self, tmp_path):
        from airweave.domains.taxonomy.jobs.gateway import PackageGateway

        gw = PackageGateway({}, cache_dir=str(tmp_path / "keep"))
        cfg = gw._job_config(tmp_path / "job", 1.0)
        assert cfg["llm"]["cache_dir"] == str(tmp_path / "job" / "llm_cache")
