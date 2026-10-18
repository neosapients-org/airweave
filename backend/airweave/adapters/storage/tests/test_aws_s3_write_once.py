"""The conditional put: only a confirmed existing object is ``False``; a 409 race is retried."""

from unittest.mock import AsyncMock

import pytest

from airweave.adapters.storage.aws_s3 import S3Backend
from airweave.domains.storage.exceptions import StorageException


def backend(put_results, exists=False):
    s3 = S3Backend(bucket="b", region="us-east-1")
    client = AsyncMock()
    client.put_object = AsyncMock(side_effect=put_results)
    s3._get_client = AsyncMock(return_value=client)
    s3.exists = AsyncMock(return_value=exists)
    return s3, client


class TestWriteOnce:
    async def test_a_fresh_key_is_written(self):
        s3, client = backend([None])
        assert await s3.write_file_once("a/b", b"x") is True
        assert client.put_object.await_args.kwargs["IfNoneMatch"] == "*"

    async def test_precondition_failed_means_the_object_exists(self):
        s3, _ = backend([Exception("An error occurred (PreconditionFailed): 412")])
        assert await s3.write_file_once("a/b", b"x") is False

    async def test_a_409_race_is_retried_and_then_written(self):
        s3, client = backend([Exception("ConditionalRequestConflict 409"), None])
        assert await s3.write_file_once("a/b", b"x") is True
        assert client.put_object.await_count == 2

    async def test_a_409_is_false_only_when_the_object_is_then_found(self):
        s3, client = backend([Exception("ConditionalRequestConflict 409")], exists=True)
        assert await s3.write_file_once("a/b", b"x") is False
        assert client.put_object.await_count == 1

    async def test_a_409_that_never_clears_is_an_error_not_a_false(self):
        s3, _ = backend([Exception("ConditionalRequestConflict 409")] * 10)
        with pytest.raises(StorageException, match="kept conflicting"):
            await s3.write_file_once("a/b", b"x")

    async def test_any_other_failure_raises(self):
        s3, _ = backend([Exception("AccessDenied 403")])
        with pytest.raises(StorageException, match="AccessDenied"):
            await s3.write_file_once("a/b", b"x")
