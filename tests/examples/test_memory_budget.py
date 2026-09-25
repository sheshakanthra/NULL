"""Regression guard for the OOM production incident. docs/findings.md #11.

Render's free tier (512MB) OOM'd one minute after a deploy went live running
the committed 108-variant grid: peak worker-process RSS measured 1845MB
before this fix, dominated by every VariantResult in a 108-entry list
retaining its own full weight-change list (thousands of TargetWeight
objects for a high-turnover strategy, times 108). Fixed in
examples/rsi2_nifty/strategy.py (run_grid only keeps the best variant's
weights; a service/jobs.py using ProcessPoolExecutor's max_tasks_per_child=1
so nothing survives between jobs either) -- brought peak to ~880MB.

That is NOT the original 350MB target. The remaining, now-dominant cost is
~297MB of null.contracts.Bar Pydantic object overhead for the ~176,000 Bar
objects the real NIFTY 50 universe loads (measured directly: RSS jumps from
~150MB to ~445MB across exactly the null.data.ohlcv.load_bars call and
nothing else). Bar is a frozen contract (CLAUDE.md invariant 5); reducing
this further means either not materialising Bar objects for the grid
search's own use or some other restructuring of null/data/ohlcv.py's
loading path -- outside what this fix was authorised to change unilaterally.
This test is a regression guard against the 1845MB baseline, not a claim
that 350MB is met; see the session record for the open question.
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import psutil
import pytest

#: Comfortably above the ~880MB measured after the weights-retention and
#: panel-hoisting fixes (real machine variance observed: ~876-883MB across
#: repeated measurements), comfortably below the 1845MB pre-fix baseline.
#: Catches a regression back toward "no cap on what a variant retains", not
#: a claim the original 350MB target is met -- see the module docstring.
PEAK_RSS_BUDGET_MB = 1100.0

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
