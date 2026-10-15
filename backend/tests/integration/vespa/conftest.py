"""Vespa in Docker for the contract tests of airweave-neo/1 (Airweave A11).

These tests DEPLOY the fork's application package (``vespa/app``) to the Vespa they are given,
so point them only at a disposable Vespa, never at one another service uses. They are skipped
unless both variables are set, for example for a container started with
``docker run -d --name aw-neo-test-vespa -p 127.0.0.1:19072:19071 -p 127.0.0.1:19082:8081
vespaengine/vespa:8``:

    AIRWEAVE_NEO_TEST_VESPA=http://localhost:19082          # query and document API
    AIRWEAVE_NEO_TEST_VESPA_CONFIG=http://localhost:19072   # config server (deploys)

The schema is templated with 8-dimensional embeddings; documents carry tiny fake vectors, so no
embedding provider is called.
"""

from __future__ import annotations

import io
import os
import time
import zipfile
from pathlib import Path
from typing import Iterator, List
from urllib.parse import quote, urlparse
from uuid import uuid4

import httpx
import pytest

from airweave.core.logging import logger
from airweave.domains.search.adapters.vector_db.filter_translator import FilterTranslator
from airweave.domains.search.adapters.vector_db.vespa_client import VespaVectorDB
from airweave.platform.destinations.vespa.types import VespaDocument

VESPA_URL = os.environ.get("AIRWEAVE_NEO_TEST_VESPA")
CONFIG_URL = os.environ.get("AIRWEAVE_NEO_TEST_VESPA_CONFIG")
EMBEDDING_DIM = 8
APP_DIR = Path(__file__).resolve().parents[4] / "vespa" / "app"



def pytest_collection_modifyitems(config, items):  # noqa: D103
    here = str(Path(__file__).parent)
    skip = pytest.mark.skip(reason="set AIRWEAVE_NEO_TEST_VESPA and _CONFIG to a disposable Vespa")
    for item in items:
        if str(item.fspath).startswith(here):
            item.add_marker(pytest.mark.integration)
            if not (VESPA_URL and CONFIG_URL):
                item.add_marker(skip)


def _application_zip() -> bytes:
    """The application package with the schema placeholders filled in."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for path in APP_DIR.rglob("*"):
            if path.is_dir() or path.name.startswith("."):
                continue
            data = path.read_text()
            data = data.replace("{{EMBEDDING_DIM}}", str(EMBEDDING_DIM))
            data = data.replace("{{VERSION}}", "contract-test")
            zf.writestr(str(path.relative_to(APP_DIR)), data)
    return buf.getvalue()


@pytest.fixture(scope="session")
def vespa_deployed() -> str:
    """Deploy the application package and wait until documents can be fed."""
    deploy = httpx.post(
        f"{CONFIG_URL}/application/v2/tenant/default/prepareandactivate",
        content=_application_zip(),
        headers={"Content-Type": "application/zip"},
        timeout=120,
    )
    assert deploy.status_code == 200, deploy.text
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{VESPA_URL}/ApplicationStatus", timeout=5).status_code == 200:
                return VESPA_URL  # type: ignore[return-value]
        except httpx.HTTPError:
            pass
        time.sleep(3)
    raise AssertionError("Vespa did not come up after deploying the application package")


@pytest.fixture
def embedding_dim() -> int:
    """Dimensions of the deployed schema's dense embedding."""
    return EMBEDDING_DIM


@pytest.fixture
def collection_id(vespa_deployed) -> Iterator[str]:
    """A fresh collection id per test; its documents are deleted afterwards."""
    cid = str(uuid4())
    yield cid
    for schema in ("base_entity", "file_entity"):
        httpx.delete(
            f"{vespa_deployed}/document/v1/airweave/{schema}/docid",
            params={
                "selection": f'{schema}.airweave_system_metadata_collection_id=="{cid}"',
                "cluster": "airweave",
            },
            timeout=60,
        )


@pytest.fixture
def feed(vespa_deployed):
    """Feed VespaDocuments through the document API, then wait until they are visible."""

    def _feed(docs: List[VespaDocument]) -> None:
        for doc in docs:
            url = (
                f"{vespa_deployed}/document/v1/airweave/{doc.schema_name}/docid/"
                f"{quote(doc.id, safe='')}"
            )
            fields = dict(doc.fields)
            fields.setdefault("dense_embedding", {"values": [0.1] * EMBEDDING_DIM})
            for _ in range(30):  # the content node may still be starting after a deploy
                resp = httpx.post(url, json={"fields": fields}, timeout=30)
                if resp.status_code not in (503, 507):
                    break
                time.sleep(2)
            assert resp.status_code == 200, resp.text

    return _feed


@pytest.fixture
def vector_db(vespa_deployed) -> VespaVectorDB:
    """The search adapter the instant search endpoint uses, against the test Vespa."""
    from vespa.application import Vespa

    parsed = urlparse(vespa_deployed)
    ctx_logger = logger.with_context(request_id="contract-test")
    return VespaVectorDB(
        app=Vespa(url=f"{parsed.scheme}://{parsed.hostname}", port=parsed.port),
        logger=ctx_logger,
        filter_translator=FilterTranslator(logger=ctx_logger),
    )
