"""Regression guard for the OOM production incident. docs/findings.md #11.

Render's free tier (512MB) OOM'd one minute after a deploy went live running
the committed 108-variant grid: peak worker-process RSS measured 1845MB
before any fix, dominated by every VariantResult in a 108-entry list
retaining its own full weight-change list (thousands of TargetWeight
objects for a high-turnover strategy, times 108). Fixed in
examples/rsi2_nifty/strategy.py (run_grid only keeps the best variant's
weights; a service/jobs.py using ProcessPoolExecutor's max_tasks_per_child=1
so nothing survives between jobs either) -- brought peak to ~880MB, and
production still crashed on the committed grid (502 then a wiped job
table -- the process restarted mid-run).

The remaining, then-dominant cost was ~297MB of null.contracts.Bar
Pydantic object overhead for the ~176,000 Bar objects the real NIFTY 50
universe loads. Bar itself is a frozen contract (CLAUDE.md invariant 5) and
null.verdict.engine's leakage checks genuinely need real Bar objects, so
that construction can't be skipped -- but the grid search's own internal
use (RSI, the weight schedule, the cost model) only ever read ``ts``,
``symbol``, ``close`` and ``adv_20`` off a Bar. run_grid now extracts those
into plain tuples/numpy arrays up front and drops its reference to the full
``bars`` tuple (and its caller, service/backtest/rsi2.py's run_backtest,
drops its own reference too) before the 108-variant loop runs, instead of
holding the full Bar tuple and the grid search's own working set at once.
That brought peak to 531.4MB.

That is still above the original 350MB target, and close enough to
Render's 512MB whole-container limit that the web process's own baseline
RSS (~150-160MB) on top of it may still be tight. This test is a
regression guard against the ~880MB pre-this-fix baseline, not a claim
that 350MB -- or even a safe margin under 512MB -- is met; see the session
record for production verification of this specific fix.
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import psutil
import pytest

#: Comfortably above the 531.4MB measured after the Bar-lifetime fix (see
#: the module docstring), comfortably below the ~880MB this fix improved on.
#: Catches a regression back toward "the full-universe Bar tuple stays
#: resident through the grid search", not a claim the original 350MB target
#: -- or a safe margin under Render's 512MB container limit -- is met.
PEAK_RSS_BUDGET_MB = 650.0

COMMITTED_HASH = "baff7b685ddcace89b71c6fb3d93182c992c8362b0c4d28889896bead3e498a9"


def _run_committed_grid_and_report_peak_rss() -> dict[str, Any]:
    """Runs INSIDE a real ProcessPoolExecutor worker -- the actual
    production execution path (service/jobs.py), not an approximation of
    it -- and reports its own peak RSS alongside the result."""
    proc = psutil.Process(os.getpid())
    samples: list[float] = [proc.memory_info().rss / (1024 * 1024)]
    stop = threading.Event()

    def sample() -> None:
        while not stop.is_set():
            samples.append(proc.memory_info().rss / (1024 * 1024))
            time.sleep(0.05)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()

    from service.backtest.rsi2 import build_grid_spec, run_rsi2_audit

    spec = build_grid_spec(
        periods=(2, 3, 4), entries=(5, 10, 15), exits=(50, 60, 70), holding_caps=(3, 5, 10, 15)
    )
    result = run_rsi2_audit(spec)

    stop.set()
    sampler.join(timeout=2)
    return {"peak_rss_mb": max(samples), "evidence_hash": result["verdict"]["evidence_hash"]}


def test_full_grid_worker_peak_rss_stays_under_the_regression_budget() -> None:
    """The committed 108-variant grid, run exactly as production runs it
    (a real ProcessPoolExecutor worker calling run_rsi2_audit), must not
    regress past PEAK_RSS_BUDGET_MB peak RSS. Slow (the real grid, same
    cost as every other committed-grid reproduction test in this suite) --
    that cost is the point; a synthetic stand-in would not measure the
    thing that actually OOM'd."""
    with ProcessPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_run_committed_grid_and_report_peak_rss)
        result = future.result(timeout=600)

    assert result["evidence_hash"] == COMMITTED_HASH, (
        "the memory fix must not change the audited result -- if this "
        "fails, the hash moved, which is a correctness regression, not a "
        "memory one"
    )
    assert result["peak_rss_mb"] < PEAK_RSS_BUDGET_MB, (
        f"worker peak RSS {result['peak_rss_mb']:.1f} MB exceeds the "
        f"{PEAK_RSS_BUDGET_MB} MB regression budget (pre-fix baseline was "
        "~1845 MB; see this module's docstring)"
    )
