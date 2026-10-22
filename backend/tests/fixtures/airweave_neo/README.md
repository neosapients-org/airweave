# Shared fixtures of contract airweave-neo/1

- `chunk-positions.example.json`: a copy of the design folder's
  `0.t1_docs_implementation/fixtures/chunk-positions.example.json`, built by its
  `make_chunk_fixture.py`. neo-platform's tests run the same file. When the design session
  regenerates it, copy it again and update `_CHUNK_FIXTURE_SHA256` in `backend/conftest.py`.
- `search-record.example.json`: the chunk record of contract section 4.4, as returned by search.

Both are served to tests by the session fixtures `airweave_neo_chunk_fixture` and
`airweave_neo_search_record` (root `conftest.py`).
