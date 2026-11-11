"""Tenant binding of the active pointer (COS-725): the pointer's tenant must derive the collection's id."""

import json

import pytest

from airweave.domains.taxonomy.artifacts import (
    TaxonomyResolutionError,
    TaxonomyResolver,
    derive_collection_id,
)
from airweave.domains.taxonomy.tests.taxonomy_fixtures import publish
from airweave.domains.taxonomy.tests.test_artifacts import Clock, CountingBackend, _fake_build

A, B, WS_A, WS_B, VERSION = "t-acme", "t-other", "ws-a", "ws-b", "v1"


def test_derivation_matches_the_typescript_original():

    import hashlib

    want = "neo-" + hashlib.sha256(b"6:t-acme:4:ws-a").hexdigest()[:48]
    assert derive_collection_id(A, WS_A) == want
    assert derive_collection_id("a:b", "c") != derive_collection_id("a", "b:c")


def _resolver(tmp_path, s3, require=False):
    return TaxonomyResolver(CountingBackend(s3), "taxonomy/", tmp_path / "local", 300, Clock(),
                            require_collection_binding=require)


def _pointer(s3, readable_id, tenant, version, sha, workspace="__absent__"):
    body = {"collection_readable_id": readable_id, "tenant_id": tenant, "taxonomy_version": version,
            "version_prefix": f"taxonomy/tenants/{tenant}/versions/{version}/",
            "manifest_sha256": sha, "generation": 1}
    if workspace != "__absent__":
        body["workspace_id"] = workspace
    (s3 / "taxonomy" / "active" / f"{readable_id}.json").write_text(json.dumps(body))


@pytest.fixture
def s3(tmp_path):
    root = tmp_path / "s3"
    for tenant in (A, B):
        built, cfg = _fake_build(tmp_path / tenant)
        pub = publish(root, built, cfg, tenant, VERSION)
        setattr(root, "_x", None) if False else None
        (tmp_path / f"sha-{tenant}").write_text(pub.manifest_sha256)
    (root / "taxonomy" / "active").mkdir(parents=True, exist_ok=True)
    return root


def _sha(tmp_path, tenant):
    return (tmp_path / f"sha-{tenant}").read_text()


class TestBinding:
    async def test_t5_legitimate_taxonomy_resolves(self, tmp_path, s3):
        rid = derive_collection_id(A, WS_A)
        _pointer(s3, rid, A, VERSION, _sha(tmp_path, A), WS_A)
        t = await _resolver(tmp_path, s3, require=True).resolve(rid)
        assert t is not None and t.tenant_id == A

    async def test_t1_cross_tenant_pointer_is_refused(self, tmp_path, s3):
        rid = derive_collection_id(A, WS_A)           # tenant A's collection
        _pointer(s3, rid, B, VERSION, _sha(tmp_path, B), WS_B)   # points at tenant B's valid artifacts
        with pytest.raises(TaxonomyResolutionError, match="do not derive"):
            await _resolver(tmp_path, s3).resolve(rid)
        assert not (tmp_path / "local" / B).exists()   # nothing of B was downloaded

    async def test_t2_consistent_replacement_is_refused_even_when_hashes_agree(self, tmp_path, s3):
        rid = derive_collection_id(A, WS_A)
        _pointer(s3, rid, B, VERSION, _sha(tmp_path, B), WS_A)   # B's folder, B's manifest hash, A's workspace
        with pytest.raises(TaxonomyResolutionError):
            await _resolver(tmp_path, s3, require=True).resolve(rid)

    async def test_t3_identical_readable_ids_cannot_select_another_tenant(self, tmp_path, s3):
        # two (tenant, workspace) pairs never share a derived id, so a shared id can only name one tenant
        assert derive_collection_id(A, WS_A) != derive_collection_id(B, WS_A)
        rid = derive_collection_id(A, WS_A)
        _pointer(s3, rid, B, VERSION, _sha(tmp_path, B), WS_A)
        with pytest.raises(TaxonomyResolutionError):
            await _resolver(tmp_path, s3).resolve(rid)

    async def test_t4_missing_tenant_fails_closed(self, tmp_path, s3):
        rid = derive_collection_id(A, WS_A)
        body = {"collection_readable_id": rid, "taxonomy_version": VERSION, "manifest_sha256": _sha(tmp_path, A),
                "version_prefix": f"taxonomy/tenants/{A}/versions/{VERSION}/", "workspace_id": WS_A}
        (s3 / "taxonomy" / "active" / f"{rid}.json").write_text(json.dumps(body))
        with pytest.raises(TaxonomyResolutionError, match="tenant_id"):
            await _resolver(tmp_path, s3).resolve(rid)

    async def test_missing_workspace_is_refused_only_when_binding_is_required(self, tmp_path, s3):
        rid = "neo-legacy-1"
        _pointer(s3, rid, A, VERSION, _sha(tmp_path, A))
        assert (await _resolver(tmp_path, s3).resolve(rid)).tenant_id == A          # legacy pointer keeps working
        with pytest.raises(TaxonomyResolutionError, match="no workspace_id"):
            await _resolver(tmp_path, s3, require=True).resolve(rid)

    async def test_unsafe_workspace_and_stale_cache(self, tmp_path, s3):
        rid = derive_collection_id(A, WS_A)
        _pointer(s3, rid, A, VERSION, _sha(tmp_path, A), "../x")
        r = _resolver(tmp_path, s3)
        with pytest.raises(TaxonomyResolutionError, match="unsafe workspace_id"):
            await r.resolve(rid)
        _pointer(s3, rid, A, VERSION, _sha(tmp_path, A), WS_A)
        with pytest.raises(TaxonomyResolutionError):          # the refusal is cached until the TTL, as other failures are
            await r.resolve(rid)
        assert (await r.resolve(rid, fresh=True)).tenant_id == A


class TestMoreCases:
    async def test_wrong_workspace_is_refused_and_nothing_is_downloaded(self, tmp_path, s3):
        rid = derive_collection_id(A, WS_A)
        _pointer(s3, rid, A, VERSION, _sha(tmp_path, A), WS_B)       # right tenant, another workspace
        backend = CountingBackend(s3)
        r = TaxonomyResolver(backend, "taxonomy/", tmp_path / "local", 300, Clock())
        with pytest.raises(TaxonomyResolutionError, match="do not derive"):
            await r.resolve(rid)
        assert backend.reads == [f"taxonomy/active/{rid}.json"]      # only the pointer was read: no manifest, no artifact
        assert not (tmp_path / "local").exists()

    async def test_a_collection_id_that_was_not_derived_is_refused_when_the_pointer_claims_a_scope(self, tmp_path, s3):
        rid = "finance-data-ab123"                                   # a user-chosen readable_id
        _pointer(s3, rid, A, VERSION, _sha(tmp_path, A), WS_A)
        with pytest.raises(TaxonomyResolutionError, match="do not derive"):
            await _resolver(tmp_path, s3).resolve(rid)

    async def test_two_tenants_with_the_same_version_name_do_not_share_a_cache_entry(self, tmp_path, s3):
        ra, rb = derive_collection_id(A, WS_A), derive_collection_id(B, WS_B)
        _pointer(s3, ra, A, VERSION, _sha(tmp_path, A), WS_A)
        _pointer(s3, rb, B, VERSION, _sha(tmp_path, B), WS_B)
        r = _resolver(tmp_path, s3, require=True)
        ta, tb = await r.resolve(ra), await r.resolve(rb)
        assert (ta.tenant_id, tb.tenant_id) == (A, B) and ta.local_path != tb.local_path
        assert A in str(ta.local_path) and B in str(tb.local_path)

    async def test_binding_does_not_replace_the_integrity_checks(self, tmp_path, s3):
        rid = derive_collection_id(A, WS_A)
        _pointer(s3, rid, A, VERSION, _sha(tmp_path, A), WS_A)
        (s3 / f"taxonomy/tenants/{A}/versions/{VERSION}/dense_prototypes.npy").write_bytes(b"tampered")
        with pytest.raises(TaxonomyResolutionError):
            await _resolver(tmp_path, s3, require=True).resolve(rid)


class TestIdentifierShapes:
    """Malformed or unexpected identifier values are refused with the resolver's own error, whether or not strict binding is on."""

    @pytest.mark.parametrize("require", [False, True])
    @pytest.mark.parametrize("workspace", [5, True, ["ws-a"], {"id": "ws-a"}, "", "wörkspace", "日本", "ws a", "../x", "ws/a"])
    async def test_unexpected_workspace_values_are_refused(self, tmp_path, s3, workspace, require):
        rid = derive_collection_id(A, WS_A)
        _pointer(s3, rid, A, VERSION, _sha(tmp_path, A), workspace)
        with pytest.raises(TaxonomyResolutionError):
            await _resolver(tmp_path, s3, require=require).resolve(rid)

    @pytest.mark.parametrize("tenant", ["t-é", "租户", "", 7, None])
    async def test_unexpected_tenant_values_are_refused(self, tmp_path, s3, tenant):
        rid = derive_collection_id(A, WS_A)
        body = {"collection_readable_id": rid, "tenant_id": tenant, "taxonomy_version": VERSION, "manifest_sha256": _sha(tmp_path, A),
                "version_prefix": f"taxonomy/tenants/{A}/versions/{VERSION}/", "workspace_id": WS_A}
        (s3 / "taxonomy" / "active" / f"{rid}.json").write_text(json.dumps(body))
        with pytest.raises(TaxonomyResolutionError):
            await _resolver(tmp_path, s3).resolve(rid)

    @pytest.mark.parametrize("drop", ["manifest_sha256", "version_prefix", "taxonomy_version"])
    async def test_a_pointer_missing_required_fields_is_refused(self, tmp_path, s3, drop):
        rid = derive_collection_id(A, WS_A)
        body = {"collection_readable_id": rid, "tenant_id": A, "taxonomy_version": VERSION, "manifest_sha256": _sha(tmp_path, A),
                "version_prefix": f"taxonomy/tenants/{A}/versions/{VERSION}/", "workspace_id": WS_A}
        body.pop(drop)
        (s3 / "taxonomy" / "active" / f"{rid}.json").write_text(json.dumps(body))
        with pytest.raises(TaxonomyResolutionError):
            await _resolver(tmp_path, s3, require=True).resolve(rid)
