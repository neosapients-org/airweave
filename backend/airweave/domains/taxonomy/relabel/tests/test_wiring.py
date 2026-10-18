"""The event type, the container and the worker know about re-label jobs."""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from airweave.core.container.factory import _create_relabel
from airweave.core.events.enums import ALL_EVENT_TYPE_ENUMS, TaxonomyEventType
from airweave.core.events.taxonomy import TaxonomyRelabelCompletedEvent
from airweave.domains.taxonomy import limits
from airweave.domains.taxonomy.relabel.tests.helpers import COLLECTION, ORG


def test_the_event_type_is_known_to_the_bus_and_to_webhooks():
    assert TaxonomyEventType in ALL_EVENT_TYPE_ENUMS
    from airweave.domains.webhooks.types import EventType

    assert EventType("taxonomy.relabel.completed")
    event = TaxonomyRelabelCompletedEvent(
        organization_id=ORG,
        relabel_job_id=ORG,
        collection_id=COLLECTION,
        collection_readable_id="neo-acme",
        taxonomy_version="v2",
        status="completed",
    )
    assert event.event_type.value == "taxonomy.relabel.completed"
    assert set(event.model_dump(mode="json")) >= {
        "event_type",
        "relabel_job_id",
        "collection_id",
        "collection_readable_id",
        "taxonomy_version",
        "previous_taxonomy_version",
        "chunks_relabelled",
        "chunks_failed",
        "status",
        "error",
        "organization_id",
        "timestamp",
    }


def settings(**kw):
    base = {"TAXONOMY_SPEND_CAP_USD": 5.0}
    base.update(kw)
    return SimpleNamespace(**base)


def test_nothing_is_built_while_the_taxonomy_step_is_off():
    assert _create_relabel(settings(), None, object()) == (None, None)


def test_service_and_runner_share_the_syncs_resolver_and_pool():
    processor = SimpleNamespace(resolver=object(), pool=object())
    service, runner = _create_relabel(settings(), processor, object())
    assert runner._resolver is processor.resolver and runner._pool is processor.pool
    assert (runner._batch, runner._page, runner._concurrency) == (
        limits.RELABEL_BATCH_SIZE,
        limits.RELABEL_PAGE_SIZE,
        limits.RELABEL_WRITE_CONCURRENCY,
    )
    assert (runner._lease, runner._timeout, runner._interval) == (
        limits.RELABEL_LEASE_SECONDS,
        limits.CLASSIFY_TIMEOUT_SECONDS,
        limits.RELABEL_RESUME_INTERVAL_SECONDS,
    )
    assert service._cap == 5.0  # the one TAXONOMY_SPEND_CAP_USD ceiling


async def test_worker_starts_and_stops_the_runner():
    from airweave.core import container as container_mod
    from airweave.domains.temporal.worker import TemporalWorker
    from airweave.domains.temporal.worker.config import WorkerConfig

    ran = asyncio.Event()

    async def run_forever(stop):
        ran.set()
        await stop.wait()

    runner = SimpleNamespace(run_forever=run_forever)
    with (
        patch("temporalio.runtime.Runtime"),
        patch.object(
            container_mod, "container", SimpleNamespace(relabel_runner=runner), create=True
        ),
    ):
        worker = TemporalWorker(
            WorkerConfig(task_queue="q", metrics_port=8080, graceful_shutdown_timeout_seconds=30)
        )
        worker._start_relabel_runner()
        worker._start_relabel_runner()  # not started twice
        await asyncio.wait_for(ran.wait(), 2)
        assert worker._relabel_task is not None
        await worker._stop_relabel_runner()
        assert worker._relabel_task is None


async def test_worker_without_a_runner_does_nothing():
    from airweave.core import container as container_mod
    from airweave.domains.temporal.worker import TemporalWorker
    from airweave.domains.temporal.worker.config import WorkerConfig

    with (
        patch("temporalio.runtime.Runtime"),
        patch.object(container_mod, "container", SimpleNamespace(relabel_runner=None), create=True),
    ):
        worker = TemporalWorker(
            WorkerConfig(task_queue="q", metrics_port=8080, graceful_shutdown_timeout_seconds=30)
        )
        worker._start_relabel_runner()
        assert worker._relabel_task is None
        await worker._stop_relabel_runner()
