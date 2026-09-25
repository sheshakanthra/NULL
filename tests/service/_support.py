"""Shared, picklable test workloads for tests/service/.

Since the post-W4 fix, ``JobManager`` runs every submitted function in a
separate OS process (``ProcessPoolExecutor``) -- see service/jobs.py's
module docstring. That means anything submitted to it in a test must be a
plain, module-level function (picklable by reference), never a lambda or a
closure defined inside a test function, and any synchronization needs a
cross-process primitive (``multiprocessing.Event``/``Value``), never
``threading.Event``, which means nothing to a separate process.

Not a ``test_*.py`` file on purpose -- pytest would try to collect it, and
it holds no tests of its own.
"""

from __future__ import annotations

import os
import time
from typing import Any


def get_pid() -> dict[str, Any]:
    """Returns the OS PID of whatever process actually ran this. Used to
    prove a job runs in a separate process from the one that submitted it --
    see service/jobs.py's module docstring for why that's the actual fix for
    the production incident, not just an implementation detail."""
    return {"pid": os.getpid()}


def return_value(value: dict[str, Any]) -> dict[str, Any]:
    """The simplest possible picklable job: hand back what you were given."""
    return value


def raise_value_error(message: str) -> dict[str, Any]:
    """A picklable job that fails -- for testing error propagation back
    across the process boundary."""
    raise ValueError(message)


def block_until_released(release: Any) -> dict[str, Any]:
    """Runs until ``release`` is set, or 10s pass. Lets a test
    deterministically hold a worker process occupied (e.g. to prove a
    second job stays "queued" while this one runs) without guessing at
    timing.

    ``release`` must be a ``multiprocessing.Manager().Event()`` proxy, not a
    bare ``multiprocessing.Event()`` -- a bare one can only be shared with a
    process created directly as its child (via inheritance at fork/spawn
    time), and raises ``RuntimeError: ... should only be shared between
    processes through inheritance`` if pickled as a submit() argument to a
    pool, which is exactly what every use here does. A Manager-backed proxy
    is built for exactly this: sharing across arbitrary, already-running
    pool workers.
    """
    release.wait(timeout=10)
    return {"blocked": True}


def burn_cpu_for_seconds(duration: float) -> dict[str, Any]:
    """A genuinely CPU-bound, GIL-holding busy-loop with no I/O and no
    voluntary yield -- stands in for the real audit's hot pure-Python cost
    loop (examples/rsi2_nifty/strategy.py's run_variant) in tests that need
    to prove a property about CPU contention without waiting ~80s for a
    real grid. At least as adversarial to a shared GIL as the real workload;
    since it now runs in its own process, it should be exactly as harmless
    to this process's own responsiveness."""
    deadline = time.monotonic() + duration
    iterations = 0
    while time.monotonic() < deadline:
        iterations += 1
    return {"iterations": iterations}
