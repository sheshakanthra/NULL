"""Signal-to-fill alignment checks. BUILD.md section 5.

Two of the six checks in the section 5 table live here:

  **Decision lag.** A signal computed on bar ``t`` close may not fill before bar
  ``t+1`` open. ``decision_lag_bars >= 1`` is enforced at the contract level, so a
  declared zero is unconstructible. That only stops the honest mistake. A strategy
  that declares a lag of 1 and then peeks anyway looks identical in the contract,
  and the only evidence left is that its weights predict the future.

  **Timestamp monotonicity.** No weight timestamp may precede the bar it depends on.

The lookahead detector is a statistical test and it is worth being explicit about
what that means: it cannot prove a strategy cheated, only that its realised
directional accuracy is not attainable without foresight. The threshold sits far
above anything real (see configs/leakage_default.yaml) so that a fatal flag is
near-certainly a data-handling bug rather than a punished good strategy.
"""

from __future__ import annotations

import numpy as np

from null.contracts import Bar, LeakageFlag, StrategyRun

__all__ = ["check_decision_lag", "check_lookahead", "check_timestamp_monotonicity"]


def check_decision_lag(run: StrategyRun) -> tuple[LeakageFlag, ...]:
    """The contract floors this at 1, so reaching here with 0 means it was bypassed."""
    if run.decision_lag_bars < 1:
        return (
            LeakageFlag(
                kind="decision_lag",
                severity="fatal",
                detail=(
                    f"decision_lag_bars is {run.decision_lag_bars}. A signal computed "
                    "on bar t close cannot fill before bar t+1 open, so a lag of zero "
                    "means every fill used information that did not exist yet."
                ),
            ),
        )
    return ()


def check_timestamp_monotonicity(
    run: StrategyRun, bars: tuple[Bar, ...]
) -> tuple[LeakageFlag, ...]:
    """Every weight must sit on a known bar, inside the sample."""
    if not bars:
        return ()
    known = {b.ts for b in bars}
    first, last = bars[0].ts, bars[-1].ts
    flags: list[LeakageFlag] = []
    for w in run.weights:
        if w.ts < first or w.ts > last:
            flags.append(
                LeakageFlag(
                    kind="timestamp_monotonicity",
                    severity="fatal",
                    symbol=w.symbol,
                    ts=w.ts,
                    detail=(
                        f"weight for {w.symbol} is timestamped {w.ts.isoformat()}, "
                        f"outside the bar range {first.isoformat()} to "
                        f"{last.isoformat()}. A decision cannot be made on a bar that "
                        "is not in the sample."
                    ),
                )
            )
        elif w.ts not in known:
            flags.append(
                LeakageFlag(
                    kind="timestamp_monotonicity",
                    severity="fatal",
                    symbol=w.symbol,
                    ts=w.ts,
                    detail=(
                        f"weight for {w.symbol} is timestamped {w.ts.isoformat()}, "
                        "which is not a bar close in the supplied series. The decision "
                        "time does not correspond to observable data."
                    ),
                )
            )
    return tuple(flags)


def check_lookahead(
    run: StrategyRun,
    bars: tuple[Bar, ...],
    *,
    max_directional_hit_rate: float,
    min_observations: int,
    severity: str = "fatal",
) -> tuple[LeakageFlag, ...]:
    """Flag directional accuracy that is not attainable without foresight.

    For each symbol, take the weight in force after the declared decision lag and
    compare its sign against the sign of the return it goes on to capture. A
    strategy with an honest lag has no systematic way to know that sign.
    """
    if len(bars) < 2:
        return ()

    by_symbol: dict[str, list[tuple[int, float]]] = {}
    index = {b.ts: i for i, b in enumerate(bars)}
    for w in run.weights:
        i = index.get(w.ts)
        if i is not None:
            by_symbol.setdefault(w.symbol, []).append((i, w.weight))

    closes = np.asarray([b.close for b in bars], dtype=np.float64)
    returns = closes[1:] / closes[:-1] - 1.0

    flags: list[LeakageFlag] = []
    for symbol, entries in sorted(by_symbol.items()):
        hits = 0
        total = 0
        for i, weight in sorted(entries):
            captured = i + run.decision_lag_bars - 1  # index into `returns`
            if weight == 0.0 or captured < 0 or captured >= returns.size:
                continue
            realised = float(returns[captured])
            if realised == 0.0:
                continue
            total += 1
            if (weight > 0.0) == (realised > 0.0):
                hits += 1

        if total < min_observations:
            continue
        hit_rate = hits / total
        if hit_rate > max_directional_hit_rate:
            flags.append(
                LeakageFlag(
                    kind="decision_lag",
                    severity="fatal" if severity == "fatal" else "warning",
                    symbol=symbol,
                    detail=(
                        f"{symbol} took a directional position that was correct on "
                        f"{hits} of {total} decisions ({hit_rate:.1%}), against a "
                        f"threshold of {max_directional_hit_rate:.0%}. A strategy with "
                        f"a declared decision lag of {run.decision_lag_bars} bar(s) has "
                        "no way to know the sign of the return it is about to capture. "
                        "This is the signature of a signal computed from data that had "
                        "not happened yet — most often an indicator shifted the wrong "
                        "way, or a fill priced off the same bar that generated it."
                    ),
                )
            )
    return tuple(flags)
