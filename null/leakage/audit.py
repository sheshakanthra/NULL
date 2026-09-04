"""Aggregate the leakage checks into one verdict on whether the audit may proceed.

BUILD.md section 5 runs this **before any statistics**. A single fatal flag ends
the audit. The reason is stated plainly in the spec: do not compute a Sharpe on a
strategy that can see the future, because you will be tempted to believe it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from null.contracts import Bar, LeakageFlag, NonEmptyStr, NullModel, StrategyRun
from null.leakage.timestamps import (
    check_decision_lag,
    check_lookahead,
    check_timestamp_monotonicity,
)

__all__ = ["LeakageConfig", "LeakageReport", "audit_leakage"]

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "leakage_default.yaml"
)


class LeakageConfig(NullModel):
    max_directional_hit_rate: float
    min_observations: int
    max_abs_single_bar_return: float

    @classmethod
    def from_yaml(cls, path: Path = DEFAULT_CONFIG_PATH) -> LeakageConfig:
        raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(
            max_directional_hit_rate=raw["lookahead"]["max_directional_hit_rate"],
            min_observations=raw["lookahead"]["min_observations"],
            max_abs_single_bar_return=raw["corporate_actions"][
                "max_abs_single_bar_return"
            ],
        )


class LeakageReport(NullModel):
    flags: tuple[LeakageFlag, ...]
    checks_run: tuple[NonEmptyStr, ...]
    unchecked: tuple[NonEmptyStr, ...]
    """Checks from the section 5 table that could NOT run, and why.

    Named explicitly rather than omitted. A leakage audit that silently skips
    survivorship reads as a clean bill of health, which is worse than no audit."""

    @property
    def is_clean(self) -> bool:
        return not any(f.is_fatal for f in self.flags)

    @property
    def fatal(self) -> tuple[LeakageFlag, ...]:
        return tuple(f for f in self.flags if f.is_fatal)


def _check_corporate_actions(
    bars: tuple[Bar, ...], *, max_abs_return: float
) -> tuple[LeakageFlag, ...]:
    """Single-bar moves large enough to be an unadjusted split or bonus.

    Without a corporate-action calendar this can only ever be a warning: a genuine
    limit move and an unadjusted 1:5 split look identical from prices alone.
    """
    if len(bars) < 2:
        return ()
    closes = np.asarray([b.close for b in bars], dtype=np.float64)
    returns = closes[1:] / closes[:-1] - 1.0
    flags: list[LeakageFlag] = []
    for i, r in enumerate(returns):
        if abs(float(r)) > max_abs_return:
            flags.append(
                LeakageFlag(
                    kind="corporate_action",
                    severity="warning",
                    symbol=bars[i + 1].symbol,
                    ts=bars[i + 1].ts,
                    detail=(
                        f"{bars[i + 1].symbol} moved {float(r):+.1%} in a single bar on "
                        f"{bars[i + 1].ts.date().isoformat()}, beyond the "
                        f"{max_abs_return:.0%} threshold. This is the signature of an "
                        "unadjusted split, bonus or consolidation. No corporate-action "
                        "calendar is wired up, so this cannot be confirmed and is "
                        "reported as a warning rather than a fatal flag."
                    ),
                )
            )
    return tuple(flags)


def _check_declared_universe(run: StrategyRun) -> tuple[LeakageFlag, ...]:
    """Weights on symbols outside the declared universe.

    This is the half of the survivorship check that does not need point-in-time
    membership data. The contract already rejects it at construction, so reaching
    here means the run was built by some other path.
    """
    declared = set(run.universe)
    unknown = sorted({w.symbol for w in run.weights} - declared)
    return tuple(
        LeakageFlag(
            kind="survivorship",
            severity="fatal",
            symbol=symbol,
            detail=(
                f"{symbol} carries a target weight but is not in the declared "
                "universe, so its point-in-time index membership cannot be checked."
            ),
        )
        for symbol in unknown
    )


def audit_leakage(
    run: StrategyRun,
    bars: tuple[Bar, ...],
    *,
    config: LeakageConfig | None = None,
) -> LeakageReport:
    """Every section 5 check that can run, with the rest named explicitly."""
    cfg = config or LeakageConfig.from_yaml()

    flags: list[LeakageFlag] = []
    flags.extend(check_decision_lag(run))
    flags.extend(check_timestamp_monotonicity(run, bars))
    flags.extend(
        check_lookahead(
            run,
            bars,
            max_directional_hit_rate=cfg.max_directional_hit_rate,
            min_observations=cfg.min_observations,
        )
    )
    flags.extend(_check_declared_universe(run))
    flags.extend(
        _check_corporate_actions(bars, max_abs_return=cfg.max_abs_single_bar_return)
    )

    return LeakageReport(
        flags=tuple(flags),
        checks_run=(
            "decision_lag",
            "timestamp_monotonicity",
            "lookahead_signature",
            "declared_universe",
            "corporate_action_magnitude",
        ),
        unchecked=(
            "point_in_time_constituency: no point-in-time NIFTY membership source is "
            "wired up, so a symbol that was not an index constituent on a given date "
            "cannot be detected. Open M3 data decision.",
            "delisting_terminal_value: without a survivorship-free universe, silently "
            "dropped delisted names cannot be distinguished from names that were never "
            "held.",
            "corporate_action_calendar: large single-bar moves are flagged by magnitude "
            "only; without an action calendar an unadjusted split cannot be confirmed.",
            "nan_forward_fill: NULL receives realised weights, not indicator code, so a "
            "forward-fill across a data gap is not observable from this input.",
            "universe_rebalance_timing: requires index reconstitution effective dates, "
            "same missing source as point_in_time_constituency.",
        ),
    )
