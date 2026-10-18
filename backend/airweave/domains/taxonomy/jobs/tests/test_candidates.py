"""Write-once candidate folders."""

import json

import pytest

from airweave.domains.taxonomy.jobs.candidates import (
    CLAIM,
    REPORT,
    CandidateConflict,
    CandidateTaken,
    CandidateWriter,
)
from airweave.domains.taxonomy.jobs.tests.helpers import CANDIDATE, TENANT, FakeS3

FILES = {"final_taxonomy.json": b"{}", "taxonomy.yaml": b"a: 1\n"}
FOLDER = f"taxonomy/tenants/{TENANT}/candidates/{CANDIDATE}/"


def writer(s3=None):
    s3 = s3 or FakeS3()
    return CandidateWriter(s3, "taxonomy/"), s3


class TestPaths:
    def test_folder_is_under_candidates_never_versions(self):
        w, _ = writer()
        assert w.folder(TENANT, CANDIDATE) == FOLDER and "/versions/" not in FOLDER

    @pytest.mark.parametrize("bad", ["../x", "a/b", "", ".hidden", "a b"])
    def test_unsafe_names_are_refused(self, bad):
        w, _ = writer()
        with pytest.raises(ValueError):
            w.folder(TENANT, bad)
        with pytest.raises(ValueError):
            w.folder(bad, CANDIDATE)

    async def test_nothing_is_ever_written_outside_the_folder(self):
        w, s3 = writer()
        await w.claim(TENANT, CANDIDATE, "j1", "discover")
        await w.publish(TENANT, CANDIDATE, "j1", FILES, {"categories": 1})
        assert s3.objects and all(p.startswith(FOLDER) for p in s3.objects)

    async def test_file_names_cannot_escape(self):
        w, s3 = writer()
        with pytest.raises(ValueError):
            await w.publish(TENANT, CANDIDATE, "j1", {"../versions/x": b"1"}, {})
        assert s3.objects == {}


class TestWriteOnce:
    async def test_publishes_files_then_the_report_last(self):
        w, s3 = writer()
        await w.claim(TENANT, CANDIDATE, "j1", "discover")
        report = await w.publish(TENANT, CANDIDATE, "j1", FILES, {"categories": 1})
        assert s3.writes[0].endswith(CLAIM) and s3.writes[-1].endswith(REPORT)
        assert [f["file"] for f in report["files"]] == sorted(FILES)
        assert json.loads(s3.objects[FOLDER + REPORT])["job_id"] == "j1"

    async def test_a_second_job_for_the_same_candidate_is_refused_before_it_writes(self):
        w, s3 = writer()
        await w.claim(TENANT, CANDIDATE, "j1", "discover")
        before = dict(s3.objects)
        with pytest.raises(CandidateTaken):
            await w.claim(TENANT, CANDIDATE, "j2", "propose")
        assert s3.objects == before

    async def test_the_same_job_can_claim_again(self):
        w, _ = writer()
        await w.claim(TENANT, CANDIDATE, "j1", "discover")
        await w.claim(TENANT, CANDIDATE, "j1", "discover")

    async def test_an_existing_object_is_never_overwritten(self):
        w, s3 = writer()
        await w.claim(TENANT, CANDIDATE, "j1", "discover")
        s3.objects[FOLDER + "final_taxonomy.json"] = b"someone else's"
        with pytest.raises(CandidateConflict):
            await w.publish(TENANT, CANDIDATE, "j1", FILES, {})
        assert s3.objects[FOLDER + "final_taxonomy.json"] == b"someone else's"

    async def test_a_replay_with_the_same_bytes_completes_the_folder(self):
        w, s3 = writer()
        await w.claim(TENANT, CANDIDATE, "j1", "discover")
        s3.objects[FOLDER + "final_taxonomy.json"] = FILES["final_taxonomy.json"]  # half-written
        await w.publish(TENANT, CANDIDATE, "j1", FILES, {"categories": 1})
        assert FOLDER + REPORT in s3.objects and FOLDER + "taxonomy.yaml" in s3.objects

    async def test_report_marks_completion(self):
        w, s3 = writer()
        assert await w.existing_report(TENANT, CANDIDATE) is None
        await w.claim(TENANT, CANDIDATE, "j1", "discover")
        assert await w.existing_report(TENANT, CANDIDATE) is None  # claimed, not complete
        await w.publish(TENANT, CANDIDATE, "j1", FILES, {"categories": 1})
        assert (await w.existing_report(TENANT, CANDIDATE))["job_id"] == "j1"


async def test_the_real_storage_backends_have_a_conditional_write(tmp_path):
    from airweave.adapters.storage.filesystem import FilesystemBackend

    fs = FilesystemBackend(tmp_path)
    w = CandidateWriter(fs, "taxonomy/")
    await w.claim(TENANT, CANDIDATE, "j1", "discover")
    with pytest.raises(CandidateTaken):
        await w.claim(TENANT, CANDIDATE, "j2", "discover")
    await w.publish(TENANT, CANDIDATE, "j1", FILES, {})
    assert (tmp_path / FOLDER / "final_taxonomy.json").read_bytes() == b"{}"


async def test_s3_conditional_put_sends_if_none_match():
    from unittest.mock import AsyncMock, MagicMock

    from airweave.adapters.storage.aws_s3 import S3Backend

    backend = S3Backend("bkt", "us-east-1", prefix="")
    client = MagicMock()
    client.put_object = AsyncMock()
    backend._client = client
    assert await backend.write_file_once("a/b", b"x") is True
    assert client.put_object.await_args.kwargs["IfNoneMatch"] == "*"
    client.put_object = AsyncMock(side_effect=Exception("PreconditionFailed: 412"))
    assert await backend.write_file_once("a/b", b"x") is False
    client.put_object = AsyncMock(side_effect=Exception("AccessDenied"))
    with pytest.raises(Exception, match="Failed to write"):
        await backend.write_file_once("a/b", b"x")


class TestReleaseAndCheckpoint:
    async def test_release_deletes_only_the_callers_own_claim(self):
        w, s3 = writer()
        await w.claim(TENANT, CANDIDATE, "job-1", "discover")

        await w.release(TENANT, CANDIDATE, "job-2")  # not the owner
        assert any(p.endswith("_claim.json") for p in s3.objects)
        await w.release(TENANT, CANDIDATE, "job-1")
        assert s3.objects == {}

    async def test_release_without_a_claim_or_without_delete_is_harmless(self):
        w0, _ = writer()
        await w0.release(TENANT, CANDIDATE, "job-1")

        class NoDelete(FakeS3):
            delete = None

        w, _ = writer(NoDelete())
        await w.claim(TENANT, CANDIDATE, "job-1", "discover")
        await w.release(TENANT, CANDIDATE, "job-1")  # cannot delete: the claim only blocks a name

    async def test_publish_asks_the_checkpoint_before_every_object_and_the_report(self):
        w, s3 = writer()
        calls = []

        async def checkpoint():
            calls.append(len(s3.objects))

        await w.publish(TENANT, CANDIDATE, "job-1", {"a.json": b"1", "b.npy": b"2"}, {}, checkpoint)

        assert calls == [0, 1, 2]  # before a.json, before b.npy, before the report

    async def test_a_checkpoint_that_raises_stops_before_the_report(self):
        w, s3 = writer()

        class Cancelled(Exception):
            pass

        seen = []

        async def checkpoint():
            seen.append(1)
            if len(seen) == 2:
                raise Cancelled

        with pytest.raises(Cancelled):
            await w.publish(
                TENANT, CANDIDATE, "job-1", {"a.json": b"1", "b.npy": b"2"}, {}, checkpoint
            )

        assert not any(p.endswith("report.json") for p in s3.objects)
        assert not any(p.endswith("b.npy") for p in s3.objects)
