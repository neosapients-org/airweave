"""Browse's queries carry search's access clause when a viewer's principals are given."""

from __future__ import annotations

from airweave.domains.search.adapters.vector_db.tests.test_vespa_client_name_substring import (
    _make_client,
)


def _yql(app) -> str:
    return app.query.call_args.kwargs["body"]["yql"]


class TestBrowseAccessClause:
    async def test_the_rows_and_the_count_carry_the_viewers_principals(self) -> None:
        client, app = _make_client()

        await client.filter_search([], "col-1", acl_principals=["user:ann", "group:fin"])
        rows = _yql(app)
        await client.count([], "col-1", acl_principals=["user:ann", "group:fin"])
        count = _yql(app)

        for yql in (rows, count):
            assert (
                "(access_is_public = true OR access_viewers contains 'user:ann' "
                "OR access_viewers contains 'group:fin')"
            ) in yql

    async def test_no_principals_means_no_access_clause_as_for_search(self) -> None:
        client, app = _make_client()

        await client.filter_search([], "col-1")
        assert "access_" not in _yql(app)
        await client.count([], "col-1")
        assert "access_" not in _yql(app)

    async def test_an_empty_principal_list_still_filters_to_public_rows(self) -> None:
        client, app = _make_client()

        await client.count([], "col-1", acl_principals=[])
        assert "(access_is_public = true)" in _yql(app)
