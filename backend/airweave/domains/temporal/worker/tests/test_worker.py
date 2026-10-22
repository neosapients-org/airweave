"""Tests for TemporalWorker — mock only Runtime (port-binding side effect).

Every other dependency (PrometheusConfig, TelemetryConfig, CollectorRegistry,
PrometheusWorkerMetrics, PrometheusMetricsRenderer, WorkerControlServer,
WorkerMetricsRegistry) is constructed for real so tests exercise actual wiring.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from temporalio.runtime import PrometheusConfig, TelemetryConfig

from airweave.domains.temporal.worker.config import WorkerConfig
from airweave.domains.temporal.worker.control_server import WorkerControlServer


def _make_config(**overrides):
    """Build a minimal WorkerConfig for testing."""
    defaults = dict(
        task_queue="test-queue",
        metrics_port=8080,
        graceful_shutdown_timeout_seconds=30,
    )
    defaults.update(overrides)
    return WorkerConfig(**defaults)


# ── __init__ tests ──────────────────────────────────────────────────


@patch("temporalio.runtime.Runtime")
def test_init_creates_runtime_with_correct_config(mock_runtime_cls):
    """__init__ builds a real TelemetryConfig(PrometheusConfig) and passes it to Runtime."""
    from airweave.domains.temporal.worker import TemporalWorker

    config = _make_config(sdk_metrics_port=9999)
    worker = TemporalWorker(config)

    # Runtime was called exactly once
    mock_runtime_cls.assert_called_once()
    _, kwargs = mock_runtime_cls.call_args

    # The telemetry arg is a real TelemetryConfig, not a mock
    telemetry = kwargs["telemetry"]
    assert isinstance(telemetry, TelemetryConfig)

    # Its metrics member is a real PrometheusConfig with the right address
    assert isinstance(telemetry.metrics, PrometheusConfig)
    assert telemetry.metrics.bind_address == "0.0.0.0:9999"

    # The worker holds the Runtime *instance*
    assert worker._runtime is mock_runtime_cls.return_value


@patch("temporalio.runtime.Runtime")
def test_init_default_sdk_metrics_port(mock_runtime_cls):
    """Default sdk_metrics_port (9090) wires through to PrometheusConfig."""
    from airweave.domains.temporal.worker import TemporalWorker

    config = _make_config()  # sdk_metrics_port defaults to 9090
    TemporalWorker(config)

    _, kwargs = mock_runtime_cls.call_args
    assert kwargs["telemetry"].metrics.bind_address == "0.0.0.0:9090"


@patch("temporalio.runtime.Runtime")
def test_init_constructs_real_control_server(mock_runtime_cls):
    """__init__ creates a real WorkerControlServer with real metrics objects."""
    from airweave.domains.temporal.worker import TemporalWorker

    config = _make_config()
    worker = TemporalWorker(config)

    assert isinstance(worker._control_server, WorkerControlServer)


# ── start() tests ───────────────────────────────────────────────────


@patch("airweave.domains.temporal.worker.Worker")
@patch("airweave.domains.temporal.worker.get_workflows", return_value=[])
@patch("airweave.domains.temporal.worker.create_activities", return_value=[])
@patch("temporalio.runtime.Runtime")
async def test_start_passes_runtime_to_get_client(
    mock_runtime_cls,
    _mock_activities,
    _mock_workflows,
    mock_worker_cls,
):
    """start() forwards self._runtime when calling get_client()."""
    from airweave.domains.temporal.worker import TemporalWorker

    config = _make_config(sdk_metrics_port=9999)
    worker = TemporalWorker(config)
    runtime_instance = worker._runtime

    # Make control_server.start() and Worker.run() no-ops
    worker._control_server.start = AsyncMock()
    mock_worker_cls.return_value.run = AsyncMock()

    with patch(
        "airweave.domains.temporal.client.get_client",
        new_callable=AsyncMock,
    ) as mock_get_client:
        mock_get_client.return_value = MagicMock()
        await worker.start()

        mock_get_client.assert_awaited_once()
        _, kwargs = mock_get_client.call_args
        assert kwargs["runtime"] is runtime_instance


# ── _get_sandbox_runner() tests ─────────────────────────────────────


@patch("temporalio.runtime.Runtime")
def test_get_sandbox_runner_default(mock_runtime_cls):
    """With disable_sandbox=False, returns a SandboxedWorkflowRunner."""
    from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner

    from airweave.domains.temporal.worker import TemporalWorker

    config = _make_config(disable_sandbox=False)
    worker = TemporalWorker(config)

    runner = worker._get_sandbox_runner()
    assert isinstance(runner, SandboxedWorkflowRunner)


@patch("temporalio.runtime.Runtime")
def test_get_sandbox_runner_disabled(mock_runtime_cls):
    """With disable_sandbox=True, returns an UnsandboxedWorkflowRunner."""
    from temporalio.worker import UnsandboxedWorkflowRunner

    from airweave.domains.temporal.worker import TemporalWorker

    config = _make_config(disable_sandbox=True)
    worker = TemporalWorker(config)

    runner = worker._get_sandbox_runner()
    assert isinstance(runner, UnsandboxedWorkflowRunner)


# ── stop() tests ────────────────────────────────────────────────────


@patch("airweave.domains.temporal.client.close", new_callable=AsyncMock)
@patch("temporalio.runtime.Runtime")
async def test_stop_shuts_down_worker_and_control_server(mock_runtime_cls, mock_client_close):
    """stop() shuts down the worker, stops the control server, and closes the client."""
    from airweave.domains.temporal.worker import TemporalWorker

    config = _make_config()
    worker = TemporalWorker(config)

    # Simulate a running worker
    mock_temporal_worker = AsyncMock()
    worker._worker = mock_temporal_worker
    worker._state.running = True
    worker._control_server.stop = AsyncMock()

    await worker.stop()

    mock_temporal_worker.shutdown.assert_awaited_once()
    worker._control_server.stop.assert_awaited_once()
    mock_client_close.assert_awaited_once()
    assert worker._state.running is False


@patch("airweave.domains.temporal.client.close", new_callable=AsyncMock)
@patch("temporalio.runtime.Runtime")
async def test_stop_skips_shutdown_when_not_running(mock_runtime_cls, mock_client_close):
    """stop() skips worker shutdown if no worker is active."""
    from airweave.domains.temporal.worker import TemporalWorker

    config = _make_config()
    worker = TemporalWorker(config)
    worker._control_server.stop = AsyncMock()

    # _worker is None, _state.running is False — should not raise
    await worker.stop()

    worker._control_server.stop.assert_awaited_once()
    mock_client_close.assert_awaited_once()


# ── taxonomy classifier pool (batch 3, item 14) ─────────────────────


def _with_container(monkeypatch, taxonomy_processor):
    import airweave.core.container as container_mod

    monkeypatch.setattr(
        container_mod,
        "container",
        MagicMock(taxonomy_processor=taxonomy_processor),
        raising=False,
    )


@patch("airweave.domains.temporal.client.close", new_callable=AsyncMock)
@patch("temporalio.runtime.Runtime")
async def test_stop_closes_the_classifier_pool_after_the_worker_drains(
    mock_runtime_cls, mock_client_close, monkeypatch
):
    """The pool is closed once the worker has shut down, so running batches finish first."""
    from airweave.domains.temporal.worker import TemporalWorker

    order = []
    processor = MagicMock()
    processor.close.side_effect = lambda: order.append("pool")
    _with_container(monkeypatch, processor)
    worker = TemporalWorker(_make_config())
    temporal_worker = AsyncMock()
    temporal_worker.shutdown.side_effect = lambda: order.append("worker")
    worker._worker = temporal_worker
    worker._state.running = True
    worker._control_server.stop = AsyncMock()

    await worker.stop()

    processor.close.assert_called_once_with()
    assert order == ["worker", "pool"]


@patch("airweave.domains.temporal.client.close", new_callable=AsyncMock)
@patch("temporalio.runtime.Runtime")
async def test_stop_without_the_taxonomy_step_is_unchanged(
    mock_runtime_cls, mock_client_close, monkeypatch
):
    from airweave.domains.temporal.worker import TemporalWorker

    _with_container(monkeypatch, None)
    worker = TemporalWorker(_make_config())
    worker._control_server.stop = AsyncMock()

    await worker.stop()

    mock_client_close.assert_awaited_once()


@patch("airweave.domains.temporal.client.close", new_callable=AsyncMock)
@patch("temporalio.runtime.Runtime")
async def test_a_failing_pool_close_does_not_stop_the_rest_of_shutdown(
    mock_runtime_cls, mock_client_close, monkeypatch
):
    from airweave.domains.temporal.worker import TemporalWorker

    processor = MagicMock()
    processor.close.side_effect = RuntimeError("disk gone")
    _with_container(monkeypatch, processor)
    worker = TemporalWorker(_make_config())
    worker._control_server.stop = AsyncMock()

    await worker.stop()

    worker._control_server.stop.assert_awaited_once()
    mock_client_close.assert_awaited_once()


@patch("airweave.domains.temporal.client.close", new_callable=AsyncMock)
@patch("temporalio.runtime.Runtime")
async def test_stop_before_the_container_exists(mock_runtime_cls, mock_client_close, monkeypatch):
    import airweave.core.container as container_mod
    from airweave.domains.temporal.worker import TemporalWorker

    monkeypatch.setattr(container_mod, "container", None, raising=False)
    worker = TemporalWorker(_make_config())
    worker._control_server.stop = AsyncMock()

    await worker.stop()

    mock_client_close.assert_awaited_once()
