"""W0 acceptance test: the service must reproduce the committed audit exactly.

BUILD.md's live-audit-service W0: ``POST /audit/demo`` runs the real,
committed ``examples/rsi2_nifty`` example through the real ``null audit``
engine and must return the same verdict already committed at
``examples/rsi2_nifty/audit_out/verdict.json`` -- same ``evidence_hash``,
same REJECT, same four failing gates. A service that returns anything else
is wiring the web layer to something other than NULL's own judgement.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from service.app import app, job_manager

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMITTED_VERDICT = (
    REPO_ROOT / "examples" / "rsi2_nifty" / "audit_out" / "verdict.json"
)

# Pinned per docs/findings.md #8: the artifact was regenerated after fixing
# null/cli.py's build_evidence() (it was feeding annualised trial Sharpes
# into a per-period-expecting deflated_sharpe_ratio). REJECT and all four
# failing gates held; only the DSR numbers and this hash moved. A literal pin
# here catches the committed artifact itself drifting silently, on top of the
# dynamic comparison below.
PINNED_HASH_PREFIX = "baff7b68"

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_audit_demo_reproduces_the_committed_verdict() -> None:
    committed = json.loads(COMMITTED_VERDICT.read_text(encoding="utf-8"))
    expected_hash = committed["evidence_hash"]
    assert expected_hash.startswith(PINNED_HASH_PREFIX)

    response = client.post("/audit/demo")
    assert response.status_code == 200

    body = response.json()
    assert body["evidence_hash"] == expected_hash
    assert body["verdict"]["evidence_hash"] == expected_hash
    assert body["verdict"]["result"] == "REJECT"

    failed = {g["name"] for g in body["verdict"]["gates"] if not g["passed"]}
    assert failed == {
        "beats_benchmark_net",
        "deflated_sharpe",
        "reality_check",
        "capacity",
    }
    assert len(failed) == 4

    passed = {g["name"] for g in body["verdict"]["gates"] if g["passed"]}
    assert passed == {"leakage_clean", "walkforward_consistency", "sensitivity_plateau"}

    # evidence.json's own recorded hash must agree too -- the service returns
    # both files, and they describe the same audit.
    assert body["evidence"]["evidence_hash"] == expected_hash


# ---------------------------------------------------------------------------
# W1/W2: POST /audit/rsi2 -- real input through a bounded backtester,
# submitted as a background job (Render's free tier times out a request at
# ~30s; the real 108-variant grid takes ~80s, so the endpoint no longer
# blocks -- see service/jobs.py). THE acceptance test that matters, per the
# phase brief: given the exact grid examples/rsi2_nifty/build_run.py used,
# the backtester must reproduce a run.json close enough that auditing it,
# through the async submit -> poll -> done path, yields the SAME verdict as
# the committed one. This is slow (the real grid against the full 50-name
# universe) -- that cost is the point; a fast test here would not be
# evidence of anything.
# ---------------------------------------------------------------------------

COMMITTED_GRID = {
    "periods": [2, 3, 4],
    "entries": [5, 10, 15],
    "exits": [50, 60, 70],
    "holding_caps": [3, 5, 10, 15],
}


def _poll_job(job_id: str, *, timeout: float = 180.0, interval: float = 0.5) -> dict[str, Any]:
    """Poll GET /audit/jobs/{job_id} until it leaves queued/running."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/audit/jobs/{job_id}")
        assert response.status_code == 200
        body = response.json()
        if body["status"] in ("done", "error"):
            return body
        time.sleep(interval)
    pytest.fail(f"job {job_id} did not finish within {timeout}s")


def test_audit_rsi2_reproduces_the_committed_verdict_for_the_committed_grid() -> None:
    committed = json.loads(COMMITTED_VERDICT.read_text(encoding="utf-8"))
    expected_hash = committed["evidence_hash"]

    submitted = client.post("/audit/rsi2", json=COMMITTED_GRID)
    assert submitted.status_code == 202
    submitted_body = submitted.json()
    assert submitted_body["status"] == "queued"
    job_id = submitted_body["job_id"]

    body = _poll_job(job_id)
    assert body["status"] == "done", body.get("error")
    assert body["n_trials"] == 108
    assert body["grid"] == COMMITTED_GRID

    assert body["verdict"]["evidence_hash"] == expected_hash
    assert body["verdict"]["result"] == "REJECT"
    assert body["verdict"]["generated_from"]["n_trials"] == 108

    failed = {g["name"] for g in body["verdict"]["gates"] if not g["passed"]}
    assert failed == {
        "beats_benchmark_net",
        "deflated_sharpe",
        "reality_check",
        "capacity",
    }
    passed = {g["name"] for g in body["verdict"]["gates"] if g["passed"]}
    assert passed == {"leakage_clean", "walkforward_consistency", "sensitivity_plateau"}

    assert body["evidence"]["evidence_hash"] == expected_hash
    assert body["evidence"]["deflated_sharpe"]["n_trials"] == 108


def test_audit_rsi2_rejects_a_grid_over_the_variant_cap() -> None:
    """Validation still happens synchronously at submission time -- a bad
    grid is a 422 with no job ever created, not a job that fails later."""
    response = client.post(
        "/audit/rsi2",
        json={
            "periods": [2, 3, 4, 5, 6],
            "entries": [5, 10, 15, 20, 25],
            "exits": [50, 60, 70, 80, 90],
            "holding_caps": [3, 5],
        },
    )
    assert response.status_code == 422
    assert "cap" in response.json()["detail"]


def test_audit_rsi2_rejects_a_period_outside_the_sane_band() -> None:
    response = client.post(
        "/audit/rsi2",
        json={"periods": [500], "entries": [5], "exits": [50], "holding_caps": [3]},
    )
    assert response.status_code == 422
    assert "band" in response.json()["detail"]


def test_audit_rsi2_rejects_an_empty_axis() -> None:
    response = client.post(
        "/audit/rsi2",
        json={"periods": [], "entries": [5], "exits": [50], "holding_caps": [3]},
    )
    # FastAPI's own Field(min_length=1) rejects this before it reaches
    # build_grid_spec -- still a 422, just from a different validator.
    assert response.status_code == 422


def test_audit_rsi2_limits_reports_the_enforced_bounds() -> None:
    response = client.get("/audit/rsi2/limits")
    assert response.status_code == 200
    body = response.json()
    assert body["max_grid_variants"] == 200
    assert body["period"]["min"] == 2


def test_unknown_job_id_is_a_clean_404() -> None:
    response = client.get("/audit/jobs/not-a-real-job-id")
    assert response.status_code == 404
    assert "not-a-real-job-id" in response.json()["detail"]


# ---------------------------------------------------------------------------
# W2: the job lifecycle itself, through the real endpoints. The sole worker
# is occupied deterministically (submitted and confirmed "running" directly
# on the shared job_manager) before each of these runs, so there is no race
# to observe "queued" or to trigger the 429 -- see service/jobs.py's own
# unit tests (tests/service/test_jobs.py) for JobManager tested in isolation.
# ---------------------------------------------------------------------------


def _occupy_the_worker() -> tuple[str, threading.Event]:
    release = threading.Event()
    job_id = job_manager.submit(lambda: (release.wait(timeout=10), {"blocked": True})[1])
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = job_manager.get(job_id)
        if job is not None and job.status == "running":
            return job_id, release
        time.sleep(0.005)
    pytest.fail("blocker job never reached 'running'")


def test_a_submitted_rsi2_job_transitions_queued_then_running_then_done() -> None:
    blocker_id, release = _occupy_the_worker()
    try:
        submitted = client.post(
            "/audit/rsi2",
            json={"periods": [2], "entries": [5], "exits": [50], "holding_caps": [3]},
        )
        assert submitted.status_code == 202
        job_id = submitted.json()["job_id"]

        # The sole worker is provably busy with the blocker, so this job
        # must still be waiting.
        queued = client.get(f"/audit/jobs/{job_id}")
        assert queued.status_code == 200
        assert queued.json()["status"] == "queued"
    finally:
        release.set()

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and job_manager.get(blocker_id).status != "done":  # type: ignore[union-attr]
        time.sleep(0.01)

    body = _poll_job(job_id, timeout=60)
    assert body["status"] == "done", body.get("error")
    assert body["n_trials"] == 1


def test_audit_rsi2_returns_429_when_the_instance_is_at_capacity() -> None:
    """The real security surface this milestone closes: a burst of requests
    past this instance's capacity is rejected outright, not queued without
    bound and not crashed into."""
    release = threading.Event()

    def _blocker() -> dict[str, Any]:
        release.wait(timeout=10)
        return {}

    capacity = job_manager.max_workers + job_manager.max_queue_size
    filler_ids = []
    try:
        first = job_manager.submit(_blocker)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and job_manager.get(first).status != "running":  # type: ignore[union-attr]
            time.sleep(0.005)
        filler_ids.append(first)
        filler_ids.extend(job_manager.submit(_blocker) for _ in range(capacity - 1))
        assert len(filler_ids) == capacity

        overflow = client.post("/audit/rsi2", json=COMMITTED_GRID)
        assert overflow.status_code == 429
        assert overflow.json()["detail"]
    finally:
        release.set()
        deadline = time.monotonic() + 10
        for job_id in filler_ids:
            while (
                time.monotonic() < deadline
                and job_manager.get(job_id).status != "done"  # type: ignore[union-attr]
            ):
                time.sleep(0.01)

    # Capacity freed up -- the instance is not melted, service resumes.
    recovered = client.post(
        "/audit/rsi2",
        json={"periods": [2], "entries": [5], "exits": [50], "holding_caps": [3]},
    )
    assert recovered.status_code == 202
    body = _poll_job(recovered.json()["job_id"], timeout=60)
    assert body["status"] == "done", body.get("error")
