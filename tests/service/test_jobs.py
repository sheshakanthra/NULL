"""Unit tests for the bounded, process-isolated job runner (service/jobs.py).

Deliberately independent of the audit pipeline: JobManager knows nothing
about backtests or audits, so these tests use small, fast, fully-controlled
workloads from tests/service/_support.py rather than real ~80s grid audits.
The endpoint-level wiring -- and the one real submit -> poll -> done
reproduction against the committed grid -- lives in tests/service/test_app.py.

Every submitted workload is a module-level function from _support.py, never
a lambda or a closure: since the post-W4 fix, JobManager runs jobs in a
separate OS process (ProcessPoolExecutor), and only plain, importable
functions (plus picklable arguments) survive that trip. Where a test needs
to control timing precisely, it passes a ``multiprocessing.Manager().Event()``
proxy (the ``mp_manager`` fixture below) -- a bare ``multiprocessing.Event()``
raises ``RuntimeError`` when pickled as a submit() argument (it can only be
shared with a process created directly as its child), and a
``threading.Event`` means nothing to a separate process at all.
"""

from __future__ import annotations

import multiprocessing
import time
from typing import Callable

import pytest

from service.jobs import Job, JobManager, JobQueueFullError
from tests.service._support import (
    block_until_released,
    burn_cpu_for_seconds,
    get_pid,
    raise_value_error,
    return_value,
)


def _wait_for(predicate: Callable[[], bool], timeout: float = 5.0, interval: float = 0.01) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval)
    pytest.fail(f"condition not met within {timeout}s")


@pytest.fixture
def make_manager():
    """A JobManager factory that shuts down every manager it creates at
    teardown -- each one owns a real ProcessPoolExecutor, i.e. real OS
    processes once submitted to, and a test session creating a dozen of
    these must not leak a dozen orphaned Python subprocesses."""
    created: list[JobManager] = []

    def _make(**kwargs: object) -> JobManager:
        manager = JobManager(**kwargs)  # type: ignore[arg-type]
        created.append(manager)
        return manager

    yield _make
    for manager in created:
        manager.shutdown(wait=False)


@pytest.fixture
def mp_manager():
    """A ``multiprocessing.Manager()`` for tests needing to signal a running
    job from the test process -- see the module docstring for why this,
    not a bare ``multiprocessing.Event()`` or a ``threading.Event()``."""
    manager = multiprocessing.Manager()
    yield manager
    manager.shutdown()


def test_a_submitted_job_runs_and_returns_its_result(make_manager) -> None:
    manager = make_manager(max_workers=1, max_queue_size=2)
    job_id = manager.submit(return_value, {"answer": 42})

    _wait_for(lambda: manager.get(job_id).status in ("done", "error"))  # type: ignore[union-attr]

    job = manager.get(job_id)
    assert job is not None
    assert job.status == "done"
    assert job.result == {"answer": 42}
    assert job.error is None


def test_an_unknown_job_id_returns_none(make_manager) -> None:
    manager = make_manager(max_workers=1, max_queue_size=2)
    assert manager.get("no-such-job") is None


def test_a_job_transitions_queued_then_running_then_done(make_manager, mp_manager) -> None:
    manager = make_manager(max_workers=1, max_queue_size=2)
    release = mp_manager.Event()

    # Occupy the sole worker first, so the job under test is guaranteed to
    # sit in "queued" until we release the blocker -- no race to observe it.
    blocker_id = manager.submit(block_until_released, release)
    _wait_for(lambda: manager.get(blocker_id).status == "running")  # type: ignore[union-attr]

    job_id = manager.submit(return_value, {"answer": 1})
    job = manager.get(job_id)
    assert job is not None
    assert job.status == "queued", "the worker is busy; this job must still be waiting"

    release.set()
    _wait_for(lambda: manager.get(blocker_id).status == "done")  # type: ignore[union-attr]
    _wait_for(lambda: manager.get(job_id).status == "done")  # type: ignore[union-attr]

    job = manager.get(job_id)
    assert job is not None
    assert job.status == "done"
    assert job.result == {"answer": 1}


def test_a_raising_job_is_reported_as_a_clean_error_not_a_traceback(make_manager) -> None:
    manager = make_manager(max_workers=1, max_queue_size=2)

    job_id = manager.submit(raise_value_error, "the audit could not find the cache")
    _wait_for(lambda: manager.get(job_id).status in ("done", "error"))  # type: ignore[union-attr]

    job = manager.get(job_id)
    assert job is not None
    assert job.status == "error"
    assert job.result is None
    assert job.error == "the audit could not find the cache"
    assert "Traceback" not in (job.error or "")


def test_submission_past_capacity_raises_job_queue_full_without_creating_a_job(
    make_manager, mp_manager
) -> None:
    """The concurrency cap: someone firing far more requests than the
    instance can hold gets a clean rejection at submission time, not a
    growing backlog and not a crash."""
    manager = make_manager(max_workers=1, max_queue_size=2)
    release = mp_manager.Event()

    # Occupy the sole worker first and confirm it, so every submission from
    # here on deterministically lands in the wait queue rather than racing a
    # worker that hasn't picked it up yet (queue.Queue.get() removes an item
    # the instant a worker claims it, before that item finishes running).
    running_id = manager.submit(block_until_released, release)
    _wait_for(lambda: manager.get(running_id).status == "running")  # type: ignore[union-attr]

    queued_ids = [manager.submit(block_until_released, release) for _ in range(manager.max_queue_size)]
    assert len(queued_ids) == manager.max_queue_size

    with pytest.raises(JobQueueFullError):
        manager.submit(block_until_released, release)

    # A firehose of further submissions must keep failing the same way --
    # not crash, not silently start dropping into an unbounded backlog.
    for _ in range(20):
        with pytest.raises(JobQueueFullError):
            manager.submit(block_until_released, release)

    # And the rejected submissions left no trace: no job_id was ever handed
    # back for any of them, so there is nothing orphaned in the job table
    # beyond the running job and the ones that genuinely filled the queue.
    release.set()
    for job_id in (running_id, *queued_ids):
        _wait_for(lambda job_id=job_id: manager.get(job_id).status == "done")  # type: ignore[union-attr]

    # Capacity freed up -- normal service resumes, it isn't melted.
    recovered_id = manager.submit(return_value, {"recovered": True})
    _wait_for(lambda: manager.get(recovered_id).status == "done")  # type: ignore[union-attr]
    job = manager.get(recovered_id)
    assert job is not None and job.result == {"recovered": True}


def test_a_job_slower_than_the_ttl_survives_running_and_stays_fetchable_when_polled(
    make_manager, mp_manager
) -> None:
    """Production incident: a real audit on Render's free-tier (slow) CPU
    outlived the in-memory job store and the client got 'job could not be
    found'. That specific 404 was traced to a process restart (the audit's
    CPU-bound work ran on a thread in this same process and plausibly
    starved /health), now fixed by running jobs in a separate process -- see
    the module docstring. This test pins the eviction contract that was
    already correct and stays correct: a job whose total runtime exceeds the
    TTL many times over, polled repeatedly during that run exactly like a
    real client would, must never be evicted while running, and must be
    fetchable the instant it finishes.
    """
    manager = make_manager(max_workers=1, max_queue_size=1, job_ttl_seconds=0.1)
    release = mp_manager.Event()

    job_id = manager.submit(block_until_released, release)
    _wait_for(lambda: manager.get(job_id).status == "running")  # type: ignore[union-attr]

    # Poll repeatedly during the run, each gap alone longer than the TTL --
    # exactly the shape of a client polling every few seconds against a job
    # that takes minutes.
    for _ in range(3):
        time.sleep(0.15)
        job = manager.get(job_id)
        assert job is not None, (
            "a running job must never be evicted, no matter how long it runs "
            "relative to the TTL"
        )
        assert job.status == "running"

    release.set()
    _wait_for(lambda: manager.get(job_id).status == "done")  # type: ignore[union-attr]

    # Fetchable immediately on completion -- not a race against the TTL.
    job = manager.get(job_id)
    assert job is not None
    assert job.status == "done"
    assert job.result == {"blocked": True}


def test_a_finished_job_stays_alive_while_the_client_keeps_polling_it(make_manager) -> None:
    """The grace-window refinement: the TTL clock resets on every successful
    get(), not just on finishing. A client polling a done job every couple
    of seconds (as the live page's poll loop does) must never have it
    evicted out from under a poll gap shorter than the TTL, even though the
    job has been sitting 'done' for far longer than the TTL in total."""
    manager = make_manager(max_workers=1, max_queue_size=1, job_ttl_seconds=0.1)
    job_id = manager.submit(return_value, {"ok": True})
    _wait_for(lambda: manager.get(job_id).status == "done")  # type: ignore[union-attr]

    for _ in range(5):
        time.sleep(0.06)  # less than the TTL between checks
        job = manager.get(job_id)
        assert job is not None, "polling more often than the TTL must keep the job alive"

    # Stop polling -- now it ages out normally.
    time.sleep(0.2)
    assert manager.get(job_id) is None


def test_only_finished_jobs_are_evicted_by_ttl(make_manager, mp_manager) -> None:
    manager = make_manager(max_workers=1, max_queue_size=2, job_ttl_seconds=0.05)
    release = mp_manager.Event()

    running_id = manager.submit(block_until_released, release)
    _wait_for(lambda: manager.get(running_id).status == "running")  # type: ignore[union-attr]

    time.sleep(0.2)  # well past the TTL, but running_id has no finished_at yet
    assert manager.get(running_id) is not None, "a running job must never be evicted"

    release.set()
    _wait_for(lambda: manager.get(running_id).status == "done")  # type: ignore[union-attr]

    time.sleep(0.2)  # now past the TTL since it finished
    # Eviction is lazy -- it happens on the next submit()/get() call, which
    # this next line itself triggers.
    assert manager.get(running_id) is None, "a long-finished job must be evicted"


def test_job_dataclass_defaults() -> None:
    job = Job(job_id="x")
    assert job.status == "queued"
    assert job.result is None
    assert job.error is None
    assert job.started_at is None
    assert job.finished_at is None


# ---------------------------------------------------------------------------
# Process isolation itself -- the actual fix for the production incident.
# ---------------------------------------------------------------------------


def test_a_cpu_bound_job_runs_in_a_separate_process_not_this_one(make_manager) -> None:
    """The direct claim: a job's own PID is not this test process's PID.
    Confirms the fix is real (jobs genuinely execute elsewhere), not just
    that the API still returns the right answer eventually."""
    import os

    manager = make_manager(max_workers=1, max_queue_size=1)
    job_id = manager.submit(get_pid)
    _wait_for(lambda: manager.get(job_id).status == "done")  # type: ignore[union-attr]

    job = manager.get(job_id)
    assert job is not None and job.result is not None
    worker_pid = job.result["pid"]
    assert worker_pid != os.getpid(), (
        "the job ran in this process -- it must run in a separate one, or a "
        "CPU-bound job can starve this process's own request handling again "
        "(the production incident this fix addresses)"
    )


def test_get_does_not_block_while_a_cpu_bound_job_is_running(make_manager) -> None:
    """A JobManager-level version of the /health proof (the full HTTP-level
    version, through the real FastAPI app, is
    tests/service/test_app.py::test_health_stays_responsive_while_a_job_is_running).
    get() itself was never meaningfully blocked by the old thread-based
    design either (CPython's GIL still time-slices cooperating threads), so
    this mainly documents the property rather than being the test that would
    have caught the incident -- the HTTP-level one is."""
    manager = make_manager(max_workers=1, max_queue_size=1)
    job_id = manager.submit(burn_cpu_for_seconds, 2.0)
    _wait_for(lambda: manager.get(job_id).status == "running")  # type: ignore[union-attr]

    start = time.monotonic()
    job = manager.get(job_id)
    elapsed = time.monotonic() - start

    assert job is not None
    assert job.status == "running"
    assert elapsed < 0.5, f"get() took {elapsed:.2f}s while a job was running"

    _wait_for(lambda: manager.get(job_id).status == "done", timeout=5)  # type: ignore[union-attr]
