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


def _resolver(tmp_path, s3):
    return TaxonomyResolver(CountingBackend(s3), "taxonomy/", tmp_path / "local", 300, Clock())


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
        (tmp_path / f"sha-{tenant}").write_text(pub.manifest_sha256)
    (root / "taxonomy" / "active").mkdir(parents=True, exist_ok=True)
    return root


def _sha(tmp_path, tenant):
    return (tmp_path / f"sha-{tenant}").read_text()


class TestBinding:
    async def test_t5_legitimate_taxonomy_resolves(self, tmp_path, s3):
        rid = derive_collection_id(A, WS_A)
        _pointer(s3, rid, A, VERSION, _sha(tmp_path, A), WS_A)
        t = await _resolver(tmp_path, s3).resolve(rid)
        assert t is not None and t.tenant_id == A

    async def test_t1_cross_tenant_pointer_is_refused(self, tmp_path, s3):
        rid = derive_collection_id(A, WS_A)           # tenant A's collection
        _pointer(s3, rid, B, VERSION, _sha(tmp_path, B), WS_B)   # points at tenant B's valid artifacts
        backend = CountingBackend(s3)
        r = TaxonomyResolver(backend, "taxonomy/", tmp_path / "local", 300, Clock())
        with pytest.raises(TaxonomyResolutionError, match="do not derive"):
            await r.resolve(rid)
        # only the pointer was read: no manifest, no artifact, and no local artifact directory
        assert backend.reads == [f"taxonomy/active/{rid}.json"]
        assert not (tmp_path / "local").exists()

    async def test_t2_t3_consistent_replacement_is_refused(self, tmp_path, s3):
        # B's folder, B's manifest hash, A's workspace: internally consistent, but not A's collection
        assert derive_collection_id(A, WS_A) != derive_collection_id(B, WS_A)      # two tenants never share a derived id
        rid = derive_collection_id(A, WS_A)
        _pointer(s3, rid, B, VERSION, _sha(tmp_path, B), WS_A)
        with pytest.raises(TaxonomyResolutionError, match="do not derive"):
            await _resolver(tmp_path, s3).resolve(rid)

    @pytest.mark.parametrize("workspace", [WS_A, None, "__absent__"])
    async def test_t4_missing_tenant_fails_closed_before_anything_else_is_read(
        self, tmp_path, s3, workspace
    ):
        # whatever the pointer says about workspace_id, a pointer without a tenant is refused
        rid = derive_collection_id(A, WS_A)
        body = {
            "collection_readable_id": rid,
            "taxonomy_version": VERSION,
            "manifest_sha256": _sha(tmp_path, A),
            "version_prefix": f"taxonomy/tenants/{A}/versions/{VERSION}/",
        }
        if workspace != "__absent__":
            body["workspace_id"] = workspace
        (s3 / "taxonomy" / "active" / f"{rid}.json").write_text(json.dumps(body))
        backend = CountingBackend(s3)
        r = TaxonomyResolver(backend, "taxonomy/", tmp_path / "local", 300, Clock())
        with pytest.raises(TaxonomyResolutionError, match="tenant_id"):
            await r.resolve(rid)
        assert backend.reads == [f"taxonomy/active/{rid}.json"]
        assert not (tmp_path / "local").exists()

    @pytest.mark.parametrize("workspace", ["__absent__", None])
    async def test_a_missing_or_null_workspace_id_is_refused_before_anything_else_is_read(
        self, tmp_path, s3, workspace
    ):
        # An active pointer must carry a valid workspace_id: absent and explicit null are both invalid.
        rid = derive_collection_id(A, WS_A)
        _pointer(s3, rid, A, VERSION, _sha(tmp_path, A), workspace)
        backend = CountingBackend(s3)
        r = TaxonomyResolver(backend, "taxonomy/", tmp_path / "local", 300, Clock())
        with pytest.raises(TaxonomyResolutionError, match="unsafe workspace_id"):
            await r.resolve(rid)
        # only the pointer was read: no manifest, no artifact, and no local artifact directory
        assert backend.reads == [f"taxonomy/active/{rid}.json"]
        assert not (tmp_path / "local").exists()

    async def test_an_unsafe_workspace_is_refused(self, tmp_path, s3):
        rid = derive_collection_id(A, WS_A)
        _pointer(s3, rid, A, VERSION, _sha(tmp_path, A), "../x")
        with pytest.raises(TaxonomyResolutionError, match="unsafe workspace_id"):
            await _resolver(tmp_path, s3).resolve(rid)

    @pytest.mark.parametrize("recover", ["fresh", "ttl"])
    async def test_a_cached_refusal_lasts_until_the_ttl_or_a_fresh_read(
        self, tmp_path, s3, recover
    ):
        # The resolver caches every failure for the active-cache TTL (existing behaviour, not
        # specific to binding). The failure here is a binding mismatch, fixed afterwards.
        rid = derive_collection_id(A, WS_A)
        clock = Clock()
        r = TaxonomyResolver(CountingBackend(s3), "taxonomy/", tmp_path / "local", 300, clock)
        _pointer(s3, rid, A, VERSION, _sha(tmp_path, A), WS_B)
        with pytest.raises(TaxonomyResolutionError, match="do not derive"):
            await r.resolve(rid)
        _pointer(s3, rid, A, VERSION, _sha(tmp_path, A), WS_A)
        clock.now += 299
        with pytest.raises(TaxonomyResolutionError, match="do not derive"):  # still cached
            await r.resolve(rid)
        if recover == "fresh":
            assert (await r.resolve(rid, fresh=True)).tenant_id == A
        else:
            clock.now += 2
            assert (await r.resolve(rid)).tenant_id == A


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
        backend = CountingBackend(s3)
        r = TaxonomyResolver(
            backend, "taxonomy/", tmp_path / "local", 300, Clock()
        )
        ta, tb = await r.resolve(ra), await r.resolve(rb)
        assert (ta.tenant_id, tb.tenant_id) == (A, B) and ta.local_path != tb.local_path
        assert A in str(ta.local_path) and B in str(tb.local_path)
        # each collection's own pointer and its own tenant's manifest were read
        assert {f"taxonomy/active/{ra}.json", f"taxonomy/active/{rb}.json"} <= set(backend.reads)
        manifests = [k for k in backend.reads if k.endswith("manifest.json")]
        assert any(f"tenants/{A}/" in k for k in manifests)
        assert any(f"tenants/{B}/" in k for k in manifests)
        assert ta.manifest_sha256 != tb.manifest_sha256

    async def test_binding_does_not_replace_the_integrity_checks(self, tmp_path, s3):
        rid = derive_collection_id(A, WS_A)
        _pointer(s3, rid, A, VERSION, _sha(tmp_path, A), WS_A)
        (s3 / f"taxonomy/tenants/{A}/versions/{VERSION}/dense_prototypes.npy").write_bytes(b"tampered")
        with pytest.raises(TaxonomyResolutionError):
            await _resolver(tmp_path, s3).resolve(rid)


class TestIdentifierShapes:
    """Identifier values outside the resolver's path-safe contract are refused with its own error.

    Two groups, kept apart: malformed values (wrong type, empty, path-unsafe) and valid but
    unusual ones. The contract is ``is_safe_name`` (ASCII letters, digits, ``._+-``, starting
    with a letter or digit): a value outside it is refused even though it could be hashed,
    because tenant ids form storage paths. Production workspace ids are UUIDs or ObjectIds.
    """

    @pytest.mark.parametrize(
        "workspace", [5, True, ["ws-a"], {"id": "ws-a"}, "", "../x", "ws/a", ".hidden"]
    )
    async def test_malformed_or_path_unsafe_workspace_values_are_refused(
        self, tmp_path, s3, workspace
    ):
        rid = derive_collection_id(A, WS_A)
        _pointer(s3, rid, A, VERSION, _sha(tmp_path, A), workspace)
        with pytest.raises(TaxonomyResolutionError):
            await _resolver(tmp_path, s3).resolve(rid)

    @pytest.mark.parametrize("workspace", ["wörkspace", "日本", "ws a", " ws-a"])
    async def test_valid_but_unusual_workspace_values_outside_the_contract_are_refused(
        self, tmp_path, s3, workspace
    ):
        # the id could be derived: refusal is the contract, not a hashing failure
        rid = derive_collection_id(A, workspace)
        _pointer(s3, rid, A, VERSION, _sha(tmp_path, A), workspace)
        with pytest.raises(TaxonomyResolutionError, match="unsafe workspace_id"):
            await _resolver(tmp_path, s3).resolve(rid)

    @pytest.mark.parametrize(
        "workspace",
        ["123e4567-e89b-12d3-a456-426614174000", "507F1F77BCF86CD799439011", "ws.a+b_1-2"],
    )
    async def test_unusual_but_path_safe_workspace_values_resolve(
        self, tmp_path, s3, workspace
    ):
        # UUID, upper-case ObjectId, punctuation
        rid = derive_collection_id(A, workspace)
        _pointer(s3, rid, A, VERSION, _sha(tmp_path, A), workspace)
        assert (await _resolver(tmp_path, s3).resolve(rid)).tenant_id == A

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
            await _resolver(tmp_path, s3).resolve(rid)


class TestLegacyCollectionIds:
    async def test_a_neo_meridian_prefix_is_not_an_exemption_from_the_binding_check(self, tmp_path, s3):
        rid = f"neo-meridian-{A}"                                         # legacy per-tenant collection id
        _pointer(s3, rid, A, VERSION, _sha(tmp_path, A), WS_A)           # a pointer that claims a workspace for it
        with pytest.raises(TaxonomyResolutionError, match="do not derive"):
            await _resolver(tmp_path, s3).resolve(rid)


class TestBoundaryValidation:
    @pytest.mark.parametrize(
        "tenant, workspace", [(5, "w"), ("t", 5), (None, "w"), ("t", None), (b"t", "w")]
    )
    def test_derive_collection_id_rejects_non_string_ids_with_value_error(self, tenant, workspace):
        with pytest.raises(ValueError, match="strings"):
            derive_collection_id(tenant, workspace)
