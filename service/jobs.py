"""In-process, bounded background job runner. BUILD.md's live-audit-service
phase, W2; process-isolated execution added post-W4.

Render's free tier times out a request at ~30s; the real 108-variant RSI(2)
grid audit takes ~80s end to end through ``/audit/rsi2``. A synchronous
request would 502 in production. This turns it into submit -> poll -> result:
a caller enqueues a job and gets a ``job_id`` back immediately, a small fixed
pool of background threads supervises jobs, and a separate endpoint polls for
the outcome.

Deliberately NOT Celery/Redis/any external queue. Render's free tier is one
instance; a job queue that needs a broker is infrastructure this demo service
has nowhere to run. **The tradeoff, stated rather than discovered the hard
way: jobs live in this process's memory only.** A restart -- a deploy, a
crash, the free tier's own idle-sleep -- loses every queued or in-flight job
without a trace. Acceptable for a demo service whose worst case is "submit it
again"; not acceptable for anything that must not lose a request, which this
is not.

**Each job actually runs in a separate OS process, not a thread in this
one.** A production incident on Render's free tier traced to exactly the
opposite of that: a job's CPU-bound, GIL-holding pure-Python computation ran
on a *thread* in this same process, and on a slow/throttled free-tier CPU it
plausibly starved this process's own ability to answer ``/health`` promptly,
which Render's own health check read as "unresponsive" and restarted the
instance -- wiping the in-memory job table outright. A ``ProcessPoolExecutor``
worker is a genuinely separate process with its own interpreter and its own
GIL; however hard it spins, this process's event loop and threadpool (serving
``/health`` and every other request) are never waiting on the same GIL for
CPU time. The worker-thread supervisors below still exist and still do all
the queueing/capacity/TTL bookkeeping they always did -- they just block on
a ``Future`` from the process pool instead of running the work themselves,
which is a difference of one line (see ``_worker_loop``) with an entirely
different reliability property.

Concurrency is bounded twice, not once:

  * ``MAX_CONCURRENT_JOBS`` process-pool workers run at most that many
    audits at once -- picked conservatively (1) for a free-tier instance
    with limited, likely shared, vCPU; more processes than cores buys
    queuing fairness, not real parallelism.
  * The wait queue in front of them is a fixed-size ``queue.Queue``, not an
    unbounded list. Past ``MAX_QUEUE_SIZE`` waiting jobs, submission is
    refused outright (:class:`JobQueueFullError`) rather than accepted and
    left to pile up. That bound -- not the audit engine's own correctness --
    is what stops a burst of requests from melting a single instance; it is
    the real security surface this milestone exists to close.

**Picklability.** ``submit`` takes a plain, module-level function and its
arguments -- never a lambda or a closure -- because ``ProcessPoolExecutor``
pickles the callable and its arguments to hand them to the worker process.
A lambda can't be pickled at all; a closure over local state (a
``threading.Event``, a variable from the enclosing scope) wouldn't mean
anything in a separate process even if it could be. Every test in
tests/service/test_jobs.py submits functions from tests/service/_support.py
for exactly this reason.
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
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
#: it is evicted from memory, measured from whichever is later: when it
#: finished, or when it was last successfully fetched. Only completed jobs
#: age out this way -- a queued or running job is never evicted out from
#: under itself, however long it runs relative to this number (a slow
#: free-tier CPU taking minutes on a grid that's normally seconds does not
#: shorten this guarantee).
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
    #: Set by every successful get() that finds this job. Eviction measures
    #: the TTL from whichever is later, finished_at or this -- so a client
    #: that keeps polling, however slowly, keeps the job alive; see
    #: JobManager._evict_expired_locked.
    last_retrieved_at: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


class JobManager:
    """Owns the job table, a fixed pool of supervisor threads, and the
    process pool those supervisors actually run work in.

    One instance per process: ``service/app.py`` constructs it at import
    time and every request shares it. Callers submit a plain, module-level,
    picklable function plus its arguments; ``JobManager`` knows nothing
    about audits specifically, which is what makes it unit-testable without
    running the real backtest/audit pipeline (see tests/service/test_jobs.py
    and tests/service/_support.py).
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
        self._queue: queue.Queue[
            tuple[str, Callable[..., dict[str, Any]], tuple[Any, ...]]
        ] = queue.Queue(maxsize=max_queue_size)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        #: The actual work happens here, not on the supervisor threads below
        #: -- see the module docstring for why. max_workers processes, same
        #: cap as the thread pool it backs.
        self._process_pool = ProcessPoolExecutor(max_workers=max_workers)
        self._workers = [
            threading.Thread(target=self._worker_loop, daemon=True, name=f"audit-worker-{i}")
            for i in range(max_workers)
        ]
        for worker in self._workers:
            worker.start()

    def submit(self, fn: Callable[..., dict[str, Any]], *args: Any) -> str:
        """Enqueue ``fn(*args)`` and return its job_id, or raise
        :class:`JobQueueFullError` if the wait queue is already full.

        ``fn`` must be importable by reference (a module-level function) and
        every argument must be picklable -- both ``fn`` and ``args`` cross a
        process boundary to actually run. See the module docstring.
        """
        job_id = uuid.uuid4().hex
        job = Job(job_id=job_id)
        with self._lock:
            self._evict_expired_locked()
            self._jobs[job_id] = job
        try:
            self._queue.put_nowait((job_id, fn, args))
        except queue.Full:
            with self._lock:
                del self._jobs[job_id]
            raise JobQueueFullError(
                f"{self.max_workers + self.max_queue_size} audit(s) already running or "
                "queued, which is this instance's cap. Try again shortly."
            ) from None
        return job_id

    def shutdown(self, wait: bool = True) -> None:
        """Release the process pool. Not used by ``service/app.py``'s own
        process-lifetime singleton (there's nothing to release it *to*
        before the process itself exits) -- for tests, which construct many
        short-lived ``JobManager`` instances and must not leak a real OS
        process per instance across a whole test session."""
        self._process_pool.shutdown(wait=wait, cancel_futures=True)

    def get(self, job_id: str) -> Job | None:
        """The job's current state, or ``None`` if it never existed, was
        rejected at submission, or has aged out of the TTL window.

        Marks the job as retrieved *now* -- see ``Job.last_retrieved_at`` --
        so a client that keeps polling a slow job never loses it purely for
        having taken longer than ``job_ttl_seconds`` to finish. Done after
        eviction runs, so a job that just aged out this same call is
        correctly reported gone rather than resurrected by the fetch that
        found it missing.
        """
        with self._lock:
            self._evict_expired_locked()
            job = self._jobs.get(job_id)
            if job is not None:
                job.last_retrieved_at = time.monotonic()
            return job

    def _evict_expired_locked(self) -> None:
        """Drop completed jobs whose TTL has elapsed. Caller holds ``self._lock``.

        A job is eligible only once it has ``finished_at`` set -- a queued or
        running job (``finished_at is None``) is never a candidate, no matter
        how long it has been running or how short the TTL is; this is the
        one invariant that must hold regardless of how slow the machine
        running the job is. For an eligible job, the TTL clock resets on
        every successful ``get()`` (``last_retrieved_at``), so the window
        that matters is "how long since anyone last checked", not "how long
        since it finished" -- a client polling every few seconds keeps a
        finished job alive for as long as it keeps checking, and a job that
        finishes and is never polled again still gets the full TTL from
        ``finished_at`` as its grace period.
        """
        now = time.monotonic()
        expired = [
            job_id
            for job_id, job in self._jobs.items()
            if job.finished_at is not None
            and (now - max(job.finished_at, job.last_retrieved_at or job.finished_at))
            >= self._job_ttl_seconds
        ]
        for job_id in expired:
            del self._jobs[job_id]

    def _worker_loop(self) -> None:
        while True:
            job_id, fn, args = self._queue.get()
            job = self._set_running(job_id)
            if job is None:
                # Evicted between submit() and being picked up. Can't happen
                # within the TTL window in practice, but a vanished job is
                # not this loop's problem to raise on.
                self._queue.task_done()
                continue
            try:
                # The one line that matters: fn(*args) runs in a separate OS
                # process, not on this thread. This thread's only job now is
                # to block on that process's result -- releasing the GIL
                # while it waits -- so nothing CPU-bound ever runs where it
                # could compete with this process's own event loop for
                # answering /health. See the module docstring.
                result = self._process_pool.submit(fn, *args).result()
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
