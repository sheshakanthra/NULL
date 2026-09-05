"""Parameter neighbourhood: plateau or spike. BUILD.md section 6.7.

Perturb each parameter by +/-1 and +/-2 steps, recompute the Sharpe, and compare
the neighbourhood against the peak.

**Gate: the mean Sharpe of the immediate neighbourhood must be at least 60% of the
peak.** A spike is curve-fitting -- the result depends on the exact parameter values
chosen and neighbouring values do not work, which means the search found a hole in
the noise rather than a property of the market. A plateau is weak evidence of
structure: not proof, but at least the result does not evaporate when a parameter
moves by one step.

"Immediate" means the +/-1 ring by default. The +/-2 ring is computed and reported
because the shape of the decay is informative -- a result that survives one step and
dies at two is a narrower plateau than one that holds at both -- but the gate reads
the immediate ring, per the spec.
"""

from __future__ import annotations

from itertools import product
from typing import Callable, Mapping, Sequence

import numpy as np

from null.contracts import ParamPoint, SensitivityResult

__all__ = [
    "DEFAULT_STEPS",
    "SharpeFn",
    "neighbourhood_offsets",
    "scan_neighbourhood",
]

#: Offsets probed per parameter. The spec asks for +/-1 and +/-2.
DEFAULT_STEPS = (-2, -1, 0, 1, 2)

#: Which offsets count as the "immediate" neighbourhood the gate reads.
IMMEDIATE = (-1, 1)

SharpeFn = Callable[[Mapping[str, int]], float]


def neighbourhood_offsets(
    param_names: Sequence[str],
    *,
    steps: Sequence[int] = DEFAULT_STEPS,
    one_at_a_time: bool = True,
) -> tuple[dict[str, int], ...]:
    """Offset combinations to probe, always including the all-zero peak.

    ``one_at_a_time`` walks each parameter independently, which is what section 6.7
    describes and what keeps the scan linear in the number of parameters. The full
    cartesian product is available but grows as ``len(steps) ** n_params`` and is
    rarely affordable past three parameters.
    """
    if not param_names:
        raise ValueError("need at least one parameter to scan")

    peak = {name: 0 for name in param_names}
    if not one_at_a_time:
        combos = [
            dict(zip(param_names, values))
            for values in product(steps, repeat=len(param_names))
        ]
        combos.sort(key=lambda d: tuple(d[name] for name in param_names))
        return tuple(combos)

    out: list[dict[str, int]] = [peak]
    for name in param_names:
        for step in steps:
            if step == 0:
                continue
            point = dict(peak)
            point[name] = step
            out.append(point)
    return tuple(out)


def scan_neighbourhood(
    param_names: Sequence[str],
    sharpe_of: SharpeFn,
    *,
    steps: Sequence[int] = DEFAULT_STEPS,
    one_at_a_time: bool = True,
    param_hash_of: Callable[[Mapping[str, int]], str] | None = None,
) -> SensitivityResult:
    """Build the surface and the ratio the gate reads.

    ``sharpe_of`` is called once per offset combination and must be a pure function
    of the offsets -- it is re-run for the peak rather than being handed a cached
    value, so that a caller whose evaluation is non-deterministic shows up as an
    inconsistent surface rather than silently biasing the ratio.
    """
    names = tuple(param_names)
    offsets = neighbourhood_offsets(names, steps=steps, one_at_a_time=one_at_a_time)

    points: list[ParamPoint] = []
    for combo in offsets:
        value = float(sharpe_of(combo))
        if not np.isfinite(value):
            raise ValueError(
                f"sharpe_of returned {value} at offsets {dict(combo)}. A "
                "non-finite Sharpe in the neighbourhood scan is missing evidence, "
                "not a zero."
            )
        label = (
            param_hash_of(combo)
            if param_hash_of is not None
            else "|".join(f"{n}{combo[n]:+d}" for n in names)
        )
        points.append(ParamPoint(param_hash=label, offsets=dict(combo), sharpe=value))

    peak_point = next(p for p in points if all(v == 0 for v in p.offsets.values()))
    peak = peak_point.sharpe

    immediate = [
        p.sharpe
        for p in points
        if any(p.offsets[n] in IMMEDIATE for n in names)
        and all(p.offsets[n] in (*IMMEDIATE, 0) for n in names)
        and not all(p.offsets[n] == 0 for n in names)
    ]
    mean_immediate = float(np.mean(immediate)) if immediate else peak

    # Ratio of neighbourhood to peak. A non-positive peak makes the ratio
    # meaningless rather than large: there is no plateau around a Sharpe of zero,
    # so it reports zero and the gate fails, which is the conservative direction.
    ratio = mean_immediate / peak if peak > 0.0 else 0.0
    ratio = max(0.0, min(ratio, 1.0)) if peak > 0.0 else 0.0

    return SensitivityResult(
        param_names=names,
        peak_sharpe=peak,
        neighborhood_mean_sharpe=mean_immediate,
        neighborhood_ratio=ratio,
        points=tuple(points),
    )
