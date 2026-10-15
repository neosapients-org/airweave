"""Dedicated classifier threads (plan A7 classifier_pool.py; design review finding 4).

The stand-in classifier behaves like the package's SQLite cache: it refuses to be used from any
thread but the one that created it. The real package and cache are exercised in
``test_classifier_integration.py``.
"""

import asyncio
import os
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from airweave.domains.taxonomy.artifacts import ResolvedTaxonomy
from airweave.domains.taxonomy.classifier_pool import ClassifierPool
from airweave.domains.taxonomy.mapping import RegistryMap


class WrongThread(Exception):
    pass


class StandInClassifier:
    created: list = []

    def __init__(self, taxonomy_path, config, transport=None):
        self.thread = threading.get_ident()
        self.config = config
        self.taxonomy = types.SimpleNamespace(path=taxonomy_path)
        self.embedder = types.SimpleNamespace(cache=types.SimpleNamespace(close=self._close))
        self.closed = False
        self.delay = float(config.get("_delay", 0))
        StandInClassifier.created.append(self)

    def _close(self):
        self.closed = True

    def classify(self, rows):
        if threading.get_ident() != self.thread:
            raise WrongThread("used from another thread")
        if self.delay:
            time.sleep(self.delay)
        return types.SimpleNamespace(
            enriched=[{"chunk_id": r["chunk_id"], "thread": self.thread} for r in rows])


@pytest.fixture
def stand_in(monkeypatch):
    StandInClassifier.created = []
    integ = types.ModuleType("taxonomy_classifier.integration.airweave")
    integ.TaxonomyClassifier = StandInClassifier
    contract = types.ModuleType("taxonomy_classifier.contract")
    contract.to_contract = lambda e, h: dict(e)
    for name, mod in (("taxonomy_classifier", types.ModuleType("taxonomy_classifier")),
                      ("taxonomy_classifier.integration", types.ModuleType("x")),
                      ("taxonomy_classifier.integration.airweave", integ),
                      ("taxonomy_classifier.contract", contract)):
        monkeypatch.setitem(sys.modules, name, mod)
    return StandInClassifier


def _taxonomy(version="acme-2026.10", sha="a" * 64, **config) -> ResolvedTaxonomy:
    return ResolvedTaxonomy("neo-acme", "t-acme", version, sha, 1, "calibrated", None,
                            Path("/tmp/none"), {"embedding": {}, **config},
                            RegistryMap(version, {}, {}))


def _rows(n, prefix="c"):
    return [{"chunk_id": f"{prefix}{i}", "document_id": "d", "text": "Acme"} for i in range(n)]


class TestThreads:
    async def test_each_classifier_only_used_on_its_thread(self, tmp_path, stand_in):
        pool = ClassifierPool(2, tmp_path)
        results = await asyncio.gather(*[pool.classify(_taxonomy(), _rows(3, f"b{i}_"), 10)
                                         for i in range(12)])
        assert all("thread" in r for batch in results for r in batch)
        assert len(stand_in.created) == 2  # one classifier per thread for the version
        assert len({c.thread for c in stand_in.created}) == 2
        pool.close()

    async def test_round_robin(self, tmp_path, stand_in):
        pool = ClassifierPool(3, tmp_path)
        threads = [(await pool.classify(_taxonomy(), _rows(1), 10))[0]["thread"]
                   for _ in range(6)]
        assert threads[:3] == threads[3:] and len(set(threads)) == 3
        pool.close()

    async def test_cache_file_per_process_and_thread(self, tmp_path, stand_in):
        pool = ClassifierPool(2, tmp_path / "cache")
        for _ in range(2):
            await pool.classify(_taxonomy(), _rows(1), 10)
        paths = sorted(c.config["embedding"]["cache_path"] for c in stand_in.created)
        assert paths == [str(tmp_path / "cache" / f"{os.getpid()}-{i}.sqlite") for i in (0, 1)]
        pool.close()

    async def test_versions_cached_with_lru(self, tmp_path, stand_in):
        pool = ClassifierPool(1, tmp_path, classifiers_per_thread=2)
        for v in ("v1", "v2", "v1", "v3"):
            await pool.classify(_taxonomy(version=v), _rows(1), 10)
        assert [c.taxonomy.path for c in stand_in.created] == [Path("/tmp/none")] * 3
        v1, v2, v3 = stand_in.created
        assert v2.closed and not v1.closed and not v3.closed  # v2 was least recently used
        pool.close()

    async def test_same_version_new_checksum_is_a_new_classifier(self, tmp_path, stand_in):
        pool = ClassifierPool(1, tmp_path)
        await pool.classify(_taxonomy(sha="a" * 64), _rows(1), 10)
        await pool.classify(_taxonomy(sha="b" * 64), _rows(1), 10)
        assert len(stand_in.created) == 2
        pool.close()


def _grow(path: str, mb: float) -> None:
    """Make the thread's cache file as big as the real one would become."""
    Path(path).write_bytes(b"x" * int(mb * 1024 * 1024))


class TestCacheRotation:
    async def test_file_over_the_limit_is_replaced_by_a_new_one(self, tmp_path, stand_in):
        pool = ClassifierPool(1, tmp_path / "c", max_cache_mb=1)
        await pool.classify(_taxonomy(), _rows(1), 10)
        first = stand_in.created[0]
        old = first.config["embedding"]["cache_path"]
        _grow(old, 1.5)
        await pool.classify(_taxonomy(), _rows(1), 10)
        second = stand_in.created[1]
        new = second.config["embedding"]["cache_path"]
        assert new != old and first.closed and not second.closed
        assert not Path(old).exists(), "the full file must be deleted, not left on the volume"
        assert os.path.basename(new).startswith(f"{os.getpid()}-0-")
        pool.close()

    async def test_file_under_the_limit_is_kept(self, tmp_path, stand_in):
        pool = ClassifierPool(1, tmp_path / "c", max_cache_mb=1)
        await pool.classify(_taxonomy(), _rows(1), 10)
        _grow(stand_in.created[0].config["embedding"]["cache_path"], 0.5)
        await pool.classify(_taxonomy(), _rows(1), 10)
        assert len(stand_in.created) == 1
        pool.close()

    async def test_every_loaded_version_moves_to_the_new_file(self, tmp_path, stand_in):
        pool = ClassifierPool(1, tmp_path / "c", max_cache_mb=1)
        for v in ("v1", "v2"):
            await pool.classify(_taxonomy(version=v), _rows(1), 10)
        _grow(stand_in.created[0].config["embedding"]["cache_path"], 2)
        await pool.classify(_taxonomy(version="v1"), _rows(1), 10)
        await pool.classify(_taxonomy(version="v2"), _rows(1), 10)
        paths = {c.config["embedding"]["cache_path"] for c in stand_in.created[2:]}
        assert len(stand_in.created) == 4 and len(paths) == 1
        assert all(c.closed for c in stand_in.created[:2])
        pool.close()

    async def test_threads_rotate_independently(self, tmp_path, stand_in):
        pool = ClassifierPool(2, tmp_path / "c", max_cache_mb=1)
        for _ in range(2):
            await pool.classify(_taxonomy(), _rows(1), 10)
        _grow(stand_in.created[0].config["embedding"]["cache_path"], 2)
        for _ in range(2):
            await pool.classify(_taxonomy(), _rows(1), 10)
        assert len(stand_in.created) == 3  # only thread 0 started over
        pool.close()

    async def test_sidecar_files_go_with_it(self, tmp_path, stand_in):
        pool = ClassifierPool(1, tmp_path / "c", max_cache_mb=1)
        await pool.classify(_taxonomy(), _rows(1), 10)
        old = stand_in.created[0].config["embedding"]["cache_path"]
        _grow(old, 1.5)
        for suffix in ("-wal", "-shm", "-journal"):
            Path(old + suffix).write_bytes(b"x")
        await pool.classify(_taxonomy(), _rows(1), 10)
        assert not [p for p in (tmp_path / "c").iterdir() if p.name.startswith(Path(old).name)]
        pool.close()

    async def test_zero_turns_rotation_off(self, tmp_path, stand_in):
        pool = ClassifierPool(1, tmp_path / "c", max_cache_mb=0)
        await pool.classify(_taxonomy(), _rows(1), 10)
        _grow(stand_in.created[0].config["embedding"]["cache_path"], 3)
        await pool.classify(_taxonomy(), _rows(1), 10)
        assert len(stand_in.created) == 1
        pool.close()

    async def test_close_deletes_this_processes_files_only(self, tmp_path, stand_in):
        cache = tmp_path / "c"
        pool = ClassifierPool(2, cache, max_cache_mb=1)
        for _ in range(2):
            await pool.classify(_taxonomy(), _rows(1), 10)
        mine = [Path(c.config["embedding"]["cache_path"]) for c in stand_in.created]
        for p in mine:
            p.write_bytes(b"x")
        theirs = cache / "999999-0.sqlite"  # another worker process on a shared directory
        theirs.write_bytes(b"x")
        pool.close()
        assert not any(p.exists() for p in mine) and theirs.exists()

    async def test_close_closes_the_connections_on_their_own_threads(self, tmp_path, stand_in):
        pool = ClassifierPool(2, tmp_path / "c")
        for _ in range(2):
            await pool.classify(_taxonomy(), _rows(1), 10)
        pool.close()
        assert all(c.closed for c in stand_in.created)

    async def test_close_is_safe_twice_and_before_use(self, tmp_path, stand_in):
        pool = ClassifierPool(2, tmp_path / "c")
        pool.close()
        pool.close()


@pytest.fixture
def dead_pid():
    """The pid of a process that has finished (and is not reused within the test)."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


@pytest.fixture
def live_pid():
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    yield proc.pid
    proc.kill()
    proc.wait()


def _touch(directory: Path, *names: str) -> list:
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / n for n in names]
    for p in paths:
        p.write_bytes(b"x")
    return paths


class TestStaleCacheFiles:
    """A killed worker leaves its files on the emptyDir; the next pool removes them."""

    def test_files_of_a_finished_process_are_deleted_at_start(self, tmp_path, dead_pid):
        stale = _touch(tmp_path, f"{dead_pid}-0.sqlite", f"{dead_pid}-1-g3.sqlite",
                       f"{dead_pid}-0.sqlite-wal", f"{dead_pid}-0.sqlite-shm",
                       f"{dead_pid}-0.sqlite-journal")
        ClassifierPool(2, tmp_path)
        assert not any(p.exists() for p in stale)

    def test_files_of_another_live_process_are_never_deleted(self, tmp_path, live_pid):
        theirs = _touch(tmp_path, f"{live_pid}-0.sqlite", f"{live_pid}-0-g2.sqlite",
                        f"{live_pid}-0.sqlite-wal")
        ClassifierPool(2, tmp_path)
        assert all(p.exists() for p in theirs)

    def test_leftovers_under_this_pid_are_stale_when_no_pool_owns_them(self, tmp_path):
        # a restarted container often gets the same pid as the one that was killed
        leftovers = _touch(tmp_path, f"{os.getpid()}-0.sqlite", f"{os.getpid()}-0-g4.sqlite")
        pool = ClassifierPool(2, tmp_path)
        assert not any(p.exists() for p in leftovers)
        pool.close()

    async def test_a_second_pool_never_deletes_the_first_pools_files(self, tmp_path, stand_in):
        first = ClassifierPool(1, tmp_path)
        await first.classify(_taxonomy(), _rows(1), 10)
        mine = Path(stand_in.created[0].config["embedding"]["cache_path"])
        mine.write_bytes(b"x")
        second = ClassifierPool(1, tmp_path)  # same process, same directory
        assert mine.exists()
        second.close()  # the pool that does not own the directory deletes nothing either
        assert mine.exists()
        first.close()
        assert not mine.exists()

    def test_a_pool_that_was_closed_releases_the_directory(self, tmp_path):
        ClassifierPool(1, tmp_path).close()
        later = _touch(tmp_path, f"{os.getpid()}-0.sqlite")
        ClassifierPool(1, tmp_path).close()
        assert not later[0].exists()

    def test_other_files_are_left_alone(self, tmp_path, dead_pid):
        keep = _touch(tmp_path, "notes.txt", "abc-1.sqlite", f"{dead_pid}.sqlite",
                      f"{dead_pid}-x.sqlite", f"x{dead_pid}-0.sqlite", f"{dead_pid}-0.db")
        ClassifierPool(1, tmp_path)
        assert all(p.exists() for p in keep)

    def test_a_missing_directory_is_fine(self, tmp_path):
        ClassifierPool(1, tmp_path / "does-not-exist-yet")

    def test_a_file_that_cannot_be_removed_does_not_stop_the_start(self, tmp_path, dead_pid,
                                                                    monkeypatch):
        _touch(tmp_path, f"{dead_pid}-0.sqlite")
        monkeypatch.setattr(os, "remove", lambda p: (_ for _ in ()).throw(PermissionError(p)))
        ClassifierPool(1, tmp_path)

    def test_the_deletion_is_logged(self, tmp_path, dead_pid, caplog):
        _touch(tmp_path, f"{dead_pid}-0.sqlite", f"{dead_pid}-1.sqlite")
        with caplog.at_level("INFO"):
            ClassifierPool(1, tmp_path)
        assert any("stale" in r.getMessage() and "2" in r.getMessage() for r in caplog.records)


class TestTimeout:
    async def test_timeout_raises_and_next_batch_waits_behind(self, tmp_path, stand_in):
        pool = ClassifierPool(1, tmp_path)
        slow = _taxonomy(_delay=0.6)
        start = time.monotonic()
        with pytest.raises(asyncio.TimeoutError):
            await pool.classify(slow, _rows(1), 0.1)
        result = await pool.classify(slow, _rows(1, "next"), 10)
        assert result[0]["chunk_id"] == "next0"
        assert time.monotonic() - start >= 1.1  # the timed-out batch finished first
        pool.close()

    def test_needs_a_thread(self, tmp_path):
        with pytest.raises(ValueError):
            ClassifierPool(0, tmp_path)
