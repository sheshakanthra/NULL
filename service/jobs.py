"""In-process, bounded background job runner. BUILD.md's live-audit-service
phase, W2.

Render's free tier times out a request at ~30s; the real 108-variant RSI(2)
grid audit takes ~80s end to end through ``/audit/rsi2``. A synchronous
request would 502 in production. This turns it into submit -> poll -> result:
a caller enqueues a job and gets a ``job_id`` back immediately, a small fixed
pool of background threads runs jobs, and a separate endpoint polls for the
outcome.

Deliberately NOT Celery/Redis/any external queue. Render's free tier is one
instance; a job queue that needs a broker is infrastructure this demo service
has nowhere to run. **The tradeoff, stated rather than discovered the hard
way: jobs live in this process's memory only.** A restart -- a deploy, a
crash, the free tier's own idle-sleep -- loses every queued or in-flight job
without a trace. Acceptable for a demo service whose worst case is "submit it
again"; not acceptable for anything that must not lose a request, which this
is not.

Concurrency is bounded twice, not once:

  * ``MAX_CONCURRENT_JOBS`` worker threads run at most that many audits at
    once. Each one pins a CPU core for ~80s of mostly pure-Python
    per-day-per-symbol cost arithmetic that does not release the GIL, so more
    worker threads than cores buys queuing fairness, not real parallelism --
    picked conservatively (1) for a free-tier instance with limited, likely
    shared, vCPU.
  * The wait queue in front of them is a fixed-size ``queue.Queue``, not an
    unbounded list. Past ``MAX_QUEUE_SIZE`` waiting jobs, submission is
    refused outright (:class:`JobQueueFullError`) rather than accepted and
    left to pile up. That bound -- not the audit engine's own correctness --
    is what stops a burst of requests from melting a single instance; it is
    the real security surface this milestone exists to close.
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

__all__ = [
    "JOB_TTL_SECONDS",
    "MAX_CONCURRENT_JOBS",
    "MAX_QUEUE_SIZE",
    "Job",
    "JobManager",
    "JobQueueFullError",
    "JobStatus",
]

JobStatus = Literal["queued", "running", "done", "error"]

#: At most this many audits run at once. See the module docstring for why 1.
MAX_CONCURRENT_JOBS = 1

#: Waiting-room size in front of the workers. Total accepted capacity is
#: MAX_CONCURRENT_JOBS + MAX_QUEUE_SIZE; the next submission past that is a
#: 429, not a longer wait.
MAX_QUEUE_SIZE = 2

#: How long a FINISHED (done or error) job's result stays queryable before
#: it is evicted from memory. Only completed jobs age out this way -- a
#: queued or running job is never evicted out from under itself.
JOB_TTL_SECONDS = 30 * 60


class JobQueueFullError(Exception):
    """Submission refused: the wait queue is already at ``MAX_QUEUE_SIZE``.

    Raised by :meth:`JobManager.submit` before any job record is created --
    a rejected submission leaves no trace in the job table, so a caller
    polling a stale or guessed job_id gets a clean 404, not a job that was
    never actually accepted.
    """


@dataclass
class Job:
    job_id: str
    status: JobStatus = "queued"
    created_at: float = field(default_factory=time.monotonic)
    started_at: float | None = None
    finished_at: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


class JobManager:
    """Owns the job table and a fixed pool of worker threads.

    One instance per process: ``service/app.py`` constructs it at import
    time and every request shares it. Callers submit a zero-argument
    callable; ``JobManager`` knows nothing about audits specifically, which
    is what makes it unit-testable without running the real backtest/audit
    pipeline (see tests/service/test_jobs.py).
    """

    def __init__(
        self,
        max_workers: int = MAX_CONCURRENT_JOBS,
        max_queue_size: int = MAX_QUEUE_SIZE,
        job_ttl_seconds: float = JOB_TTL_SECONDS,
    ) -> None:
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self._job_ttl_seconds = job_ttl_seconds
        self._queue: queue.Queue[tuple[str, Callable[[], dict[str, Any]]]] = queue.Queue(
            maxsize=max_queue_size
        )
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._workers = [
            threading.Thread(target=self._worker_loop, daemon=True, name=f"audit-worker-{i}")
            for i in range(max_workers)
        ]
        for worker in self._workers:
            worker.start()

    def submit(self, fn: Callable[[], dict[str, Any]]) -> str:
        """Enqueue ``fn`` and return its job_id, or raise
        :class:`JobQueueFullError` if the wait queue is already full."""
        job_id = uuid.uuid4().hex
        job = Job(job_id=job_id)
        with self._lock:
            self._evict_expired_locked()
            self._jobs[job_id] = job
        try:
            self._queue.put_nowait((job_id, fn))
        except queue.Full:
            with self._lock:
                del self._jobs[job_id]
            raise JobQueueFullError(
                f"{self.max_workers + self.max_queue_size} audit(s) already running or "
                "queued, which is this instance's cap. Try again shortly."
            ) from None
        return job_id

    def get(self, job_id: str) -> Job | None:
        """The job's current state, or ``None`` if it never existed, was
        rejected at submission, or has aged out of the TTL window."""
        with self._lock:
            self._evict_expired_locked()
            return self._jobs.get(job_id)

    def _evict_expired_locked(self) -> None:
        """Drop completed jobs older than the TTL. Caller holds ``self._lock``."""
        cutoff = time.monotonic() - self._job_ttl_seconds
        expired = [
            job_id
            for job_id, job in self._jobs.items()
            if job.finished_at is not None and job.finished_at < cutoff
        ]
        for job_id in expired:
            del self._jobs[job_id]

    def _worker_loop(self) -> None:
        while True:
            job_id, fn = self._queue.get()
            job = self._set_running(job_id)
            if job is None:
                # Evicted between submit() and being picked up. Can't happen
                # within the TTL window in practice, but a vanished job is
                # not this loop's problem to raise on.
                self._queue.task_done()
                continue
            try:
                result = fn()
            except Exception as exc:  # noqa: BLE001 -- a worker must never die
                self._set_finished(job, status="error", error=str(exc))
            else:
                self._set_finished(job, status="done", result=result)
            finally:
                self._queue.task_done()

    def _set_running(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.status = "running"
                job.started_at = time.monotonic()
            return job

    def _set_finished(
        self,
        job: Job,
        *,
        status: Literal["done", "error"],
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            job.result = result
            job.error = error
            job.status = status
            job.finished_at = time.monotonic()
