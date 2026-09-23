"""Unit tests for the bounded in-process job runner (service/jobs.py).

Deliberately independent of the audit pipeline: JobManager knows nothing
about backtests or audits, so these tests use small, fast, fully-controlled
callables (gated on a threading.Event where precise timing matters) rather
than real ~80s grid audits. The endpoint-level wiring -- and the one real
submit -> poll -> done reproduction against the committed grid -- lives in
tests/service/test_app.py.
"""

from __future__ import annotations

import threading
import time

import pytest

from service.jobs import Job, JobManager, JobQueueFullError


def _wait_for(predicate, timeout: float = 5.0, interval: float = 0.01) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval)
    pytest.fail(f"condition not met within {timeout}s")


def test_a_submitted_job_runs_and_returns_its_result() -> None:
    manager = JobManager(max_workers=1, max_queue_size=2)
    job_id = manager.submit(lambda: {"answer": 42})

    _wait_for(lambda: manager.get(job_id).status in ("done", "error"))  # type: ignore[union-attr]

    job = manager.get(job_id)
    assert job is not None
    assert job.status == "done"
    assert job.result == {"answer": 42}
    assert job.error is None


def test_an_unknown_job_id_returns_none() -> None:
    manager = JobManager(max_workers=1, max_queue_size=2)
    assert manager.get("no-such-job") is None


def test_a_job_transitions_queued_then_running_then_done() -> None:
    manager = JobManager(max_workers=1, max_queue_size=2)
    release = threading.Event()

    def _blocker() -> dict[str, object]:
        release.wait(timeout=5)
        return {"done": True}

    # Occupy the sole worker first, so the job under test is guaranteed to
    # sit in "queued" until we release the blocker -- no race to observe it.
    blocker_id = manager.submit(_blocker)
    _wait_for(lambda: manager.get(blocker_id).status == "running")  # type: ignore[union-attr]

    job_id = manager.submit(lambda: {"answer": 1})
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


def test_a_raising_job_is_reported_as_a_clean_error_not_a_traceback() -> None:
    manager = JobManager(max_workers=1, max_queue_size=2)

    def _boom() -> dict[str, object]:
        raise ValueError("the audit could not find the cache")

    job_id = manager.submit(_boom)
    _wait_for(lambda: manager.get(job_id).status in ("done", "error"))  # type: ignore[union-attr]

    job = manager.get(job_id)
    assert job is not None
    assert job.status == "error"
    assert job.result is None
    assert job.error == "the audit could not find the cache"
    assert "Traceback" not in (job.error or "")


def test_submission_past_capacity_raises_job_queue_full_without_creating_a_job() -> None:
    """The concurrency cap: someone firing far more requests than the
    instance can hold gets a clean rejection at submission time, not a
    growing backlog and not a crash."""
    manager = JobManager(max_workers=1, max_queue_size=2)
    release = threading.Event()

    def _blocker() -> dict[str, object]:
        release.wait(timeout=5)
        return {}

    # Occupy the sole worker first and confirm it, so every submission from
    # here on deterministically lands in the wait queue rather than racing a
    # worker that hasn't picked it up yet (queue.Queue.get() removes an item
    # the instant a worker claims it, before that item finishes running).
    running_id = manager.submit(_blocker)
    _wait_for(lambda: manager.get(running_id).status == "running")  # type: ignore[union-attr]

    queued_ids = [manager.submit(_blocker) for _ in range(manager.max_queue_size)]
    assert len(queued_ids) == manager.max_queue_size

    with pytest.raises(JobQueueFullError):
        manager.submit(_blocker)

    # A firehose of further submissions must keep failing the same way --
    # not crash, not silently start dropping into an unbounded backlog.
    for _ in range(20):
        with pytest.raises(JobQueueFullError):
            manager.submit(_blocker)

    # And the rejected submissions left no trace: no job_id was ever handed
    # back for any of them, so there is nothing orphaned in the job table
    # beyond the running job and the ones that genuinely filled the queue.
    release.set()
    for job_id in (running_id, *queued_ids):
        _wait_for(lambda job_id=job_id: manager.get(job_id).status == "done")  # type: ignore[union-attr]

    # Capacity freed up -- normal service resumes, it isn't melted.
    recovered_id = manager.submit(lambda: {"recovered": True})
    _wait_for(lambda: manager.get(recovered_id).status == "done")  # type: ignore[union-attr]
    job = manager.get(recovered_id)
    assert job is not None and job.result == {"recovered": True}


def test_a_job_slower_than_the_ttl_survives_running_and_stays_fetchable_when_polled() -> None:
    """Production incident: a real audit on Render's free-tier (slow) CPU
    outlived the in-memory job store and the client got 'job could not be
    found'. Root cause was NOT a bug in this eviction logic -- a running job
    (finished_at is None) was never a candidate for eviction even before this
    test existed (test_only_finished_jobs_are_evicted_by_ttl already proved
    that). This test pins the exact production shape: a job whose total
    runtime exceeds the TTL many times over, polled repeatedly during that
    run exactly like a real client would, on a TTL short enough that any gap
    in the "never evict while running" guarantee would show up immediately.
    It must never be evicted while running, and must be fetchable the
    instant it finishes.
    """
    manager = JobManager(max_workers=1, max_queue_size=1, job_ttl_seconds=0.1)
    started = threading.Event()

    def _slow() -> dict[str, object]:
        started.set()
        time.sleep(0.5)  # 5x the TTL -- simulates a slow free-tier CPU
        return {"ok": True}

    job_id = manager.submit(_slow)
    assert started.wait(timeout=2), "job never started running"

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

    _wait_for(lambda: manager.get(job_id).status == "done")  # type: ignore[union-attr]

    # Fetchable immediately on completion -- not a race against the TTL.
    job = manager.get(job_id)
    assert job is not None
    assert job.status == "done"
    assert job.result == {"ok": True}


def test_a_finished_job_stays_alive_while_the_client_keeps_polling_it() -> None:
    """The grace-window refinement: the TTL clock resets on every successful
    get(), not just on finishing. A client polling a done job every couple
    of seconds (as the live page's checkWarmup-adjacent poll loop does) must
    never have it evicted out from under a poll gap shorter than the TTL,
    even though the job has been sitting 'done' for far longer than the TTL
    in total."""
    manager = JobManager(max_workers=1, max_queue_size=1, job_ttl_seconds=0.1)
    job_id = manager.submit(lambda: {"ok": True})
    _wait_for(lambda: manager.get(job_id).status == "done")  # type: ignore[union-attr]

    for _ in range(5):
        time.sleep(0.06)  # less than the TTL between checks
        job = manager.get(job_id)
        assert job is not None, "polling more often than the TTL must keep the job alive"

    # Stop polling -- now it ages out normally.
    time.sleep(0.2)
    assert manager.get(job_id) is None


def test_only_finished_jobs_are_evicted_by_ttl() -> None:
    manager = JobManager(max_workers=1, max_queue_size=2, job_ttl_seconds=0.05)
    release = threading.Event()

    def _blocker() -> dict[str, object]:
        release.wait(timeout=5)
        return {}

    running_id = manager.submit(_blocker)
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
