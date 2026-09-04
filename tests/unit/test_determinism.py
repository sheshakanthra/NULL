"""M0 determinism scaffold.

BUILD.md prime directive: same inputs -> byte-identical ``verdict.json``. Always.

This module proves the *contract layer* half of that claim: a fixed ``StrategyRun``
serialises to byte-identical bytes and a byte-identical SHA-256 across two
serialisations in the same session, across a JSON round-trip, and across two
separate interpreter processes started with different ``PYTHONHASHSEED`` values.

Full cross-MACHINE determinism (different OS, different BLAS, different numpy
build) is tested at M5. This is the scaffold it will be built on: the pinned
golden hash below is the value M5 will assert against on a second machine.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone

from null.contracts import Series, StrategyRun, TargetWeight, TrialRecord

# BUILD.md contracts are frozen after M0. If this constant has to change, a frozen
# contract changed shape, and that is a decision -- not a test fixup. See CLAUDE.md
# invariant 5.
GOLDEN_STRATEGY_RUN_SHA256 = "417134032b7702ece1ce5d294badf50c7917c3a74db3b9333c466fbe9c8b305f"

IST = timezone(timedelta(hours=5, minutes=30))


def _ts(day: int, hour: int = 15, minute: int = 30) -> datetime:
    """A fixed IST timestamp. No wall-clock anywhere in this module."""
    return datetime(2024, 1, day, hour, minute, tzinfo=IST)


def fixed_run() -> StrategyRun:
    """A completely fixed StrategyRun. Every value is a literal."""
    return StrategyRun(
        strategy_id="rsi2_nifty_meanrev",
        param_hash="a3f1c9d2e4b60718",
        n_trials=108,
        universe=("INFY", "RELIANCE", "TCS"),
        weights=(
            TargetWeight(ts=_ts(2), symbol="RELIANCE", weight=0.25),
            TargetWeight(ts=_ts(2), symbol="TCS", weight=-0.125),
            TargetWeight(ts=_ts(3), symbol="INFY", weight=0.5),
        ),
        trials=(
            TrialRecord(
                param_hash="a3f1c9d2e4b60718",
                sharpe=1.8,
                returns=Series(
                    ts=(_ts(2), _ts(3), _ts(4)),
                    values=(0.001, -0.002, 0.0035),
                ),
            ),
            TrialRecord(param_hash="0000111122223333", sharpe=0.42),
        ),
        decision_lag_bars=1,
        initial_capital=1_000_000.0,
    )


# ---------------------------------------------------------------------------
# in-session byte stability
# ---------------------------------------------------------------------------


def test_canonical_json_is_byte_identical_across_two_serialisations() -> None:
    first = fixed_run().canonical_json()
    second = fixed_run().canonical_json()
    assert isinstance(first, bytes)
    assert first == second


def test_content_hash_is_identical_across_two_serialisations() -> None:
    assert fixed_run().content_hash() == fixed_run().content_hash()


def test_content_hash_is_a_lowercase_sha256_hex_digest() -> None:
    digest = fixed_run().content_hash()
    assert len(digest) == 64
    assert digest == digest.lower()
    assert all(c in "0123456789abcdef" for c in digest)


def test_hash_survives_a_json_round_trip() -> None:
    run = fixed_run()
    revived = StrategyRun.model_validate_json(run.canonical_json())
    assert revived.content_hash() == run.content_hash()
    assert revived.canonical_json() == run.canonical_json()


# ---------------------------------------------------------------------------
# independence from caller-side ordering
# ---------------------------------------------------------------------------


def test_hash_is_independent_of_keyword_argument_order() -> None:
    """Field declaration order must drive the bytes, not kwarg order."""
    a = StrategyRun(
        strategy_id="s",
        param_hash="p",
        n_trials=3,
        universe=("A",),
        weights=(TargetWeight(ts=_ts(2), symbol="A", weight=1.0),),
        initial_capital=100.0,
    )
    b = StrategyRun(
        initial_capital=100.0,
        weights=(TargetWeight(weight=1.0, symbol="A", ts=_ts(2)),),
        universe=("A",),
        n_trials=3,
        param_hash="p",
        strategy_id="s",
    )
    assert a.canonical_json() == b.canonical_json()


def test_hash_is_independent_of_universe_input_order() -> None:
    """CLAUDE.md invariant 4: no reliance on set iteration order."""
    base = fixed_run()
    shuffled = StrategyRun.model_validate(
        {**base.model_dump(), "universe": ("TCS", "INFY", "RELIANCE")}
    )
    assert shuffled.content_hash() == base.content_hash()


def test_hash_is_independent_of_weight_input_order() -> None:
    run = fixed_run()
    reversed_weights = StrategyRun(
        strategy_id=run.strategy_id,
        param_hash=run.param_hash,
        n_trials=run.n_trials,
        universe=tuple(reversed(run.universe)),
        weights=tuple(reversed(run.weights)),
        trials=run.trials,
        decision_lag_bars=run.decision_lag_bars,
        initial_capital=run.initial_capital,
    )
    assert reversed_weights.content_hash() == run.content_hash()


def test_hash_is_independent_of_input_timezone_representation() -> None:
    """The same instant expressed in UTC must hash identically to its IST form."""
    ist_run = StrategyRun(
        strategy_id="s",
        param_hash="p",
        n_trials=1,
        universe=("A",),
        weights=(
            TargetWeight(
                ts=datetime(2024, 1, 2, 15, 30, tzinfo=IST), symbol="A", weight=1.0
            ),
        ),
        initial_capital=100.0,
    )
    utc_run = StrategyRun(
        strategy_id="s",
        param_hash="p",
        n_trials=1,
        universe=("A",),
        weights=(
            TargetWeight(
                ts=datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc),
                symbol="A",
                weight=1.0,
            ),
        ),
        initial_capital=100.0,
    )
    assert ist_run.content_hash() == utc_run.content_hash()


# ---------------------------------------------------------------------------
# float quantisation: sub-threshold noise must not move the hash
# ---------------------------------------------------------------------------


def test_sub_quantisation_float_noise_does_not_change_the_hash() -> None:
    """A last-ULP difference from a different BLAS must not flip the verdict hash."""
    assert 0.1 + 0.2 != 0.3  # the noise is real
    noisy = TargetWeight(ts=_ts(2), symbol="A", weight=0.1 + 0.2)
    exact = TargetWeight(ts=_ts(2), symbol="A", weight=0.3)
    assert noisy.content_hash() == exact.content_hash()  # and it is quantised away


def test_negative_zero_hashes_as_positive_zero() -> None:
    assert (
        TargetWeight(ts=_ts(2), symbol="A", weight=-0.0).content_hash()
        == TargetWeight(ts=_ts(2), symbol="A", weight=0.0).content_hash()
    )


# ---------------------------------------------------------------------------
# the hash must still be sensitive to things that matter
# ---------------------------------------------------------------------------


def test_hash_changes_when_n_trials_changes() -> None:
    """Declaring 108 trials and declaring 109 must not produce the same artifact."""
    base = fixed_run()
    lied = StrategyRun.model_validate({**base.model_dump(), "n_trials": 109})
    assert lied.content_hash() != base.content_hash()


def test_hash_changes_on_a_supra_quantisation_float_change() -> None:
    a = TargetWeight(ts=_ts(2), symbol="A", weight=0.30000000001)
    b = TargetWeight(ts=_ts(2), symbol="A", weight=0.3)
    assert a.content_hash() != b.content_hash()


# ---------------------------------------------------------------------------
# cross-process: two separate interpreter runs, different hash seeds
# ---------------------------------------------------------------------------

_CHILD = """
import sys
from null.contracts import StrategyRun
run = StrategyRun.model_validate_json(sys.stdin.buffer.read())
sys.stdout.write(run.content_hash())
"""


def _hash_in_subprocess(payload: bytes, hash_seed: str) -> str:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = hash_seed
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD],
        input=payload,
        capture_output=True,
        env=env,
        check=True,
    )
    return proc.stdout.decode()


def test_hash_identical_across_two_processes_with_different_hash_seeds() -> None:
    """Two runs, two interpreters, two PYTHONHASHSEEDs, one hash."""
    payload = fixed_run().canonical_json()
    assert _hash_in_subprocess(payload, "0") == _hash_in_subprocess(payload, "12345")


def test_subprocess_hash_matches_in_process_hash() -> None:
    run = fixed_run()
    assert _hash_in_subprocess(run.canonical_json(), "0") == run.content_hash()


# ---------------------------------------------------------------------------
# M5 scaffold: the pinned golden value
# ---------------------------------------------------------------------------


def test_fixed_run_matches_pinned_golden_hash() -> None:
    """The value M5 will assert against on a second machine.

    A failure here means either a frozen contract changed shape (CLAUDE.md
    invariant 5 -- stop and ask) or serialisation stopped being canonical.
    It is never a test to 'just update'.
    """
    assert fixed_run().content_hash() == GOLDEN_STRATEGY_RUN_SHA256
