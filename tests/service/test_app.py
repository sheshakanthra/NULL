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
from pathlib import Path

from fastapi.testclient import TestClient

from service.app import app

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMITTED_VERDICT = (
    REPO_ROOT / "examples" / "rsi2_nifty" / "audit_out" / "verdict.json"
)

# Pinned per the phase spec: "REJECT, 4 gates fail, same evidence_hash
# 3f40aa2e…". A literal pin here catches the committed artifact itself
# drifting silently, on top of the dynamic comparison below.
PINNED_HASH_PREFIX = "3f40aa2e"

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
