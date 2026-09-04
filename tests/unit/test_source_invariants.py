"""Source-text guards for the remaining CLAUDE.md hard invariants.

Invariant 1 (no LLM) has its own file, ``test_no_llm.py``, because CI names it
explicitly. These are the same mechanism applied to invariants 2, 3 and 4:

  2. No network calls in the audit path.
  3. No broker credentials, no order placement, no write access.
  4. Determinism -- no wall-clock in the audit path.

Same reasoning as the LLM grep: an import-graph check misses lazy imports and
string-built calls. Grep does not. Each family carries a negative control that
plants the violation in a temporary directory, for the same reason.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.unit.test_no_llm import scan

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "null"

# Invariant 2. Data fetch is a separate, cached, offline-first stage and does not
# live under null/ -- see the repository layout in BUILD.md section 1.
FORBIDDEN_NETWORK_PATTERNS: tuple[str, ...] = (
    r"\bimport\s+requests\b",
    r"\bimport\s+httpx\b",
    r"\bimport\s+aiohttp\b",
    r"\bimport\s+socket\b",
    r"\bimport\s+urllib\b",
    r"\bfrom\s+urllib\b",
    r"\bimport\s+yfinance\b",
    r"\bimport\s+ftplib\b",
    r"\burlopen\s*\(",
)

# Invariant 3. NULL reads. It has no broker credentials in any file, any
# environment, any test fixture.
FORBIDDEN_BROKER_PATTERNS: tuple[str, ...] = (
    r"\bplace_order\b",
    r"\bapi_secret\b",
    r"\baccess_token\b",
    r"\bkiteconnect\b",
    r"\bbroker_password\b",
)

# Invariant 4. Wall-clock in the audit path destroys determinism.
FORBIDDEN_CLOCK_PATTERNS: tuple[str, ...] = (
    r"datetime\.now\s*\(",
    r"datetime\.utcnow\s*\(",
    r"\btime\.time\s*\(",
    r"date\.today\s*\(",
    r"\brandom\.seed\s*\(\s*\)",
)

NETWORK_CONTROLS = ("import requests\n", "from urllib.request import urlopen\n")
BROKER_CONTROLS = ("kite.place_order(variety='regular')\n", "api_secret = 'x'\n")
CLOCK_CONTROLS = ("stamp = datetime.now()\n", "t0 = time.time()\n")

WHY = {
    "network": "`null audit` must run with the network off (CLAUDE.md invariant 2).",
    "broker": "NULL reads. It never places an order (invariant 3).",
    "clock": "Wall-clock in the audit path breaks determinism (invariant 4).",
}


@pytest.mark.parametrize("pattern", FORBIDDEN_NETWORK_PATTERNS)
def test_no_network_access_in_null_package(pattern: str) -> None:
    hits = scan(PACKAGE_ROOT, pattern)
    assert not hits, f"/{pattern}/ found inside null/. {WHY['network']}\n" + "\n".join(hits)


@pytest.mark.parametrize("pattern", FORBIDDEN_BROKER_PATTERNS)
def test_no_broker_credentials_or_order_placement(pattern: str) -> None:
    hits = scan(PACKAGE_ROOT, pattern)
    assert not hits, f"/{pattern}/ found inside null/. {WHY['broker']}\n" + "\n".join(hits)


@pytest.mark.parametrize("pattern", FORBIDDEN_CLOCK_PATTERNS)
def test_no_wall_clock_in_audit_path(pattern: str) -> None:
    hits = scan(PACKAGE_ROOT, pattern)
    assert not hits, f"/{pattern}/ found inside null/. {WHY['clock']}\n" + "\n".join(hits)


_CONTROL_CASES = [
    *[(c, FORBIDDEN_NETWORK_PATTERNS) for c in NETWORK_CONTROLS],
    *[(c, FORBIDDEN_BROKER_PATTERNS) for c in BROKER_CONTROLS],
    *[(c, FORBIDDEN_CLOCK_PATTERNS) for c in CLOCK_CONTROLS],
]


@pytest.mark.parametrize(("planted", "patterns"), _CONTROL_CASES)
def test_scanner_catches_a_planted_violation(
    planted: str, patterns: tuple[str, ...], tmp_path: Path
) -> None:
    """Negative control: prove each family fails when a violation is really there."""
    (tmp_path / "sneaky.py").write_text(planted, encoding="utf-8")
    caught = [p for p in patterns if scan(tmp_path, p)]
    assert caught, f"no pattern matched {planted!r} -- the scan has a hole"
