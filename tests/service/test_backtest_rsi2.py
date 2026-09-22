"""Fast tests for the RSI(2) grid-spec validation and a small-grid backtest.

The load-bearing acceptance test -- that the bounded backtester reproduces
the committed 108-variant grid closely enough to yield the same verdict -- is
in tests/service/test_app.py, exercised through the real HTTP endpoint. It is
slow (the full grid against the 50-name universe takes real wall-clock time,
same as examples/rsi2_nifty/build_run.py does), so it isn't duplicated here.
This file covers what should stay fast: the validation surface itself, and a
determinism check on a grid small enough to run in seconds.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from service.backtest.rsi2 import (
    MAX_GRID_VARIANTS,
    MAX_HOLDING_CAP,
    MAX_PERIOD,
    MAX_THRESHOLD,
    MIN_HOLDING_CAP,
    MIN_PERIOD,
    MIN_THRESHOLD,
    GridSpecError,
    build_grid_spec,
    run_backtest,
)


def test_build_grid_spec_sorts_and_deduplicates() -> None:
    spec = build_grid_spec(
        periods=(4, 2, 3), entries=(15, 5, 10), exits=(70, 50, 60), holding_caps=(15, 3, 5, 10)
    )
    assert spec.periods == (2, 3, 4)
    assert spec.entries == (5, 10, 15)
    assert spec.exits == (50, 60, 70)
    assert spec.holding_caps == (3, 5, 10, 15)
    assert spec.n_variants == 108


@pytest.mark.parametrize(
    ("axis", "values"),
    [
        ("periods", ()),
        ("entries", ()),
        ("exits", ()),
        ("holding_caps", ()),
    ],
)
def test_build_grid_spec_rejects_an_empty_axis(axis: str, values: tuple[int, ...]) -> None:
    kwargs = dict(periods=(2,), entries=(5,), exits=(50,), holding_caps=(3,))
    kwargs[axis] = values
    with pytest.raises(GridSpecError, match="at least one value"):
        build_grid_spec(**kwargs)


def test_build_grid_spec_rejects_duplicate_values() -> None:
    with pytest.raises(GridSpecError, match="duplicate"):
        build_grid_spec(periods=(2, 2, 3), entries=(5,), exits=(50,), holding_caps=(3,))


@pytest.mark.parametrize(
    ("axis", "bad_value", "bound"),
    [
        ("periods", MIN_PERIOD - 1, "band"),
        ("periods", MAX_PERIOD + 1, "band"),
        ("entries", MIN_THRESHOLD - 1, "band"),
        ("entries", MAX_THRESHOLD + 1, "band"),
        ("exits", MIN_THRESHOLD - 1, "band"),
        ("exits", MAX_THRESHOLD + 1, "band"),
        ("holding_caps", MIN_HOLDING_CAP - 1, "band"),
        ("holding_caps", MAX_HOLDING_CAP + 1, "band"),
    ],
)
def test_build_grid_spec_rejects_out_of_band_values(
    axis: str, bad_value: int, bound: str
) -> None:
    kwargs = dict(periods=(2,), entries=(5,), exits=(50,), holding_caps=(3,))
    kwargs[axis] = (bad_value,)
    with pytest.raises(GridSpecError, match=bound):
        build_grid_spec(**kwargs)


def test_build_grid_spec_rejects_a_grid_over_the_cap() -> None:
    # 5 periods x 5 entries x 5 exits x 2 holding_caps = 250, over the 200 cap.
    with pytest.raises(GridSpecError, match=r"\bover the\b"):
        build_grid_spec(
            periods=(2, 3, 4, 5, 6),
            entries=(5, 10, 15, 20, 25),
            exits=(50, 60, 70, 80, 90),
            holding_caps=(3, 5),
        )


def test_build_grid_spec_accepts_a_grid_exactly_at_the_cap() -> None:
    # 5 x 5 x 4 x 2 = 200 exactly.
    spec = build_grid_spec(
        periods=(2, 3, 4, 5, 6),
        entries=(5, 10, 15, 20, 25),
        exits=(50, 60, 70, 80),
        holding_caps=(3, 5),
    )
    assert spec.n_variants == MAX_GRID_VARIANTS


def test_run_backtest_is_deterministic(tmp_path: Path) -> None:
    spec = build_grid_spec(periods=(2, 3), entries=(5,), exits=(50,), holding_caps=(3,))
    first = run_backtest(spec, tmp_path / "first")
    second = run_backtest(spec, tmp_path / "second")

    assert first.n_trials == second.n_trials == 2
    assert first.run_path.read_bytes() == second.run_path.read_bytes()
    assert (
        first.trials_parquet_path.read_bytes() == second.trials_parquet_path.read_bytes()
    )
    assert first.sensitivity_path.read_bytes() == second.sensitivity_path.read_bytes()


def test_run_backtest_n_trials_matches_the_actual_grid_size(tmp_path: Path) -> None:
    spec = build_grid_spec(periods=(2, 3, 4), entries=(5, 10), exits=(50,), holding_caps=(3,))
    artifacts = run_backtest(spec, tmp_path)
    assert artifacts.n_trials == spec.n_variants == 6
