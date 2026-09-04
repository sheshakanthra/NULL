"""NULL contracts -- BUILD.md section 2.

**These models are FROZEN after M0.** Every other module in this repository codes
against them and nothing else. If a milestone appears to need a contract change,
stop and ask (CLAUDE.md invariant 5). Do not widen a model to make a downstream
module easier.

Three decisions are baked in here and are worth understanding before you read on.

1. ``Series`` is a first-class frozen model of parallel timestamp/value tuples,
   not a ``pandas.Series``. A pandas object is mutable, its canonical byte
   representation depends on pandas internals, and neither is compatible with
   "same input -> byte-identical verdict.json". Compute code converts at the
   boundary via :meth:`Series.to_numpy` / :meth:`Series.to_pandas`.

2. Every float in a contract is quantised to 12 significant digits on validation,
   and non-finite floats are rejected outright. Different BLAS builds and numpy
   versions disagree in the last few bits of a float; one such bit would flip
   ``evidence_hash`` and break the prime directive. Twelve significant digits is
   far more precision than any reported metric needs and far less than the noise
   floor of a cross-machine numerical difference. Rejecting NaN and infinity is
   the same rule as invariant 6: missing evidence must fail loudly, never
   serialise into an artifact as a silent hole.

3. ``StrategyRun.n_trials`` is required and non-defaulted, and ``trials`` carries
   the per-variant record when the caller has it. Section 6.1 needs the variance
   across trial Sharpes to deflate one; ``n_trials`` alone cannot supply it.
   ``trials`` never substitutes for ``n_trials`` -- a caller may log a subset or
   none at all, but they must always declare the count.

A caller that passes ``n_trials=1`` after a 5,000-run grid search is committing
fraud, and the deflated-Sharpe gate is the only thing standing between them and
a lie.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Annotated, Literal, Self

import numpy as np
import numpy.typing as npt
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps import cost off the hot path
    import pandas as pd

__all__ = [
    "Bar",
    "Evidence",
    "FLOAT_SIGNIFICANT_DIGITS",
    "FoldResult",
    "GateResult",
    "IST",
    "LeakageFlag",
    "LeakageKind",
    "NullModel",
    "ParamPoint",
    "PerfMetrics",
    "RegressionResult",
    "SPEC_VERSION",
    "SensitivityResult",
    "Series",
    "StrategyRun",
    "TargetWeight",
    "TrialRecord",
    "Verdict",
]

#: Stamped onto every :class:`Verdict`. Bump only when a frozen contract changes,
#: which is a decision, not a refactor.
SPEC_VERSION = "0.1.0"

#: Indian Standard Time. All contract timestamps normalise to this offset so that
#: the same instant expressed in any zone produces the same bytes.
IST = timezone(timedelta(hours=5, minutes=30), name="IST")

#: See module docstring, decision 2.
FLOAT_SIGNIFICANT_DIGITS = 12

# ---------------------------------------------------------------------------
# canonical scalar types
# ---------------------------------------------------------------------------


def _canonical_float(value: float) -> float:
    """Quantise to a fixed significant-digit count and reject non-finite values."""
    if not math.isfinite(value):
        raise ValueError(
            "non-finite float in a contract field: missing or undefined evidence must "
            "fail the audit, not serialise into the artifact"
        )
    if value == 0.0:
        return 0.0  # collapse -0.0, which reprs differently but compares equal
    return float(f"{value:.{FLOAT_SIGNIFICANT_DIGITS}g}")


def _non_negative(value: float) -> float:
    if value < 0.0:
        raise ValueError(f"must be non-negative, got {value}")
    return value


def _positive(value: float) -> float:
    if value <= 0.0:
        raise ValueError(f"must be strictly positive, got {value}")
    return value


def _unit_interval(value: float) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"must lie in [0, 1], got {value}")
    return value


def _canonical_timestamp(value: datetime) -> datetime:
    """Require tz-awareness, then normalise the offset so bytes are stable."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(
            "timestamp must be timezone-aware; a naive datetime silently assumes a "
            "zone and that is how look-ahead bugs get in"
        )
    return value.astimezone(IST)


def _canonical_symbol(value: str) -> str:
    symbol = value.strip()
    if not symbol:
        raise ValueError("symbol must be a non-empty string")
    return symbol


def _non_empty_text(value: str) -> str:
    text = value.strip()
    if not text:
        raise ValueError("must be non-empty")
    return text


NullFloat = Annotated[float, AfterValidator(_canonical_float)]
NonNegativeFloat = Annotated[NullFloat, AfterValidator(_non_negative)]
PositiveFloat = Annotated[NullFloat, AfterValidator(_positive)]
Probability = Annotated[NullFloat, AfterValidator(_unit_interval)]
Timestamp = Annotated[datetime, AfterValidator(_canonical_timestamp)]
Symbol = Annotated[str, AfterValidator(_canonical_symbol)]
NonEmptyStr = Annotated[str, AfterValidator(_non_empty_text)]
Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

# ---------------------------------------------------------------------------
# base model
# ---------------------------------------------------------------------------


class NullModel(BaseModel):
    """Frozen, strict base for every contract.

    ``extra="forbid"`` matters as much as ``frozen``: a typo'd field name that
    silently lands in an ``extra`` bucket is a gate reading a number that was
    never set.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_assignment=True,
        validate_default=True,
        str_strip_whitespace=False,
    )

    def canonical_json(self) -> bytes:
        """The one true byte representation of this model.

        Keys are sorted, so field declaration order cannot leak into the hash;
        separators are tight, so formatting cannot; ``allow_nan=False`` is a
        second line of defence behind :func:`_canonical_float`.
        """
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")

    def content_hash(self) -> str:
        """SHA-256 of :meth:`canonical_json`, lowercase hex."""
        return hashlib.sha256(self.canonical_json()).hexdigest()

# ---------------------------------------------------------------------------
# market data
# ---------------------------------------------------------------------------


class Series(NullModel):
    """A timestamped float series. Strictly increasing in time, no gaps implied.

    Parallel tuples rather than a mapping: a mapping would put dict ordering
    between the caller and the hash, and CLAUDE.md invariant 4 forbids that.
    """

    ts: tuple[Timestamp, ...]
    values: tuple[NullFloat, ...]

    @model_validator(mode="after")
    def _check_shape(self) -> Self:
        if len(self.ts) != len(self.values):
            raise ValueError(
                f"ts and values must be the same length, got {len(self.ts)} and "
                f"{len(self.values)}"
            )
        for earlier, later in zip(self.ts, self.ts[1:]):
            if later <= earlier:
                raise ValueError(
                    f"timestamps must be strictly increasing, got {earlier} followed "
                    f"by {later}"
                )
        return self

    def __len__(self) -> int:
        return len(self.values)

    def to_numpy(self) -> npt.NDArray[np.float64]:
        """The compute boundary. Statistics operate on this, never on the model."""
        return np.asarray(self.values, dtype=np.float64)

    def to_pandas(self) -> pd.Series:
        import pandas as pd

        return pd.Series(
            self.to_numpy(),
            index=pd.DatetimeIndex(list(self.ts), name="ts"),
        )

    @classmethod
    def from_pandas(cls, series: pd.Series) -> Series:
        """Re-enter the contract layer from compute code.

        Validation applies on the way in, so a NaN produced mid-computation is
        caught here rather than surfacing in a report as a blank cell.
        """
        import pandas as pd

        index = series.index
        if not isinstance(index, pd.DatetimeIndex):
            raise TypeError(f"expected a DatetimeIndex, got {type(index).__name__}")
        if index.tz is None:
            raise ValueError("expected a timezone-aware DatetimeIndex")
        return cls(
            ts=tuple(ts.to_pydatetime() for ts in index),
            values=tuple(float(v) for v in series.to_numpy(dtype="float64")),
        )

    @classmethod
    def from_pairs(cls, pairs: tuple[tuple[datetime, float], ...]) -> Series:
        return cls(
            ts=tuple(ts for ts, _ in pairs),
            values=tuple(value for _, value in pairs),
        )


class Bar(NullModel):
    """One OHLCV bar. ``ts`` is the bar CLOSE time, tz-aware, IST."""

    ts: Timestamp
    symbol: Symbol
    open: PositiveFloat
    high: PositiveFloat
    low: PositiveFloat
    close: PositiveFloat
    volume: NonNegativeFloat
    adv_20: NonNegativeFloat | None = None
    """20d average daily value traded. Feeds the square-root impact model (M1)."""

    @model_validator(mode="after")
    def _check_ohlc(self) -> Self:
        if self.high < self.low:
            raise ValueError(f"high {self.high} below low {self.low}")
        if self.high < max(self.open, self.close):
            raise ValueError(
                f"high {self.high} below open/close ({self.open}/{self.close})"
            )
        if self.low > min(self.open, self.close):
            raise ValueError(
                f"low {self.low} above open/close ({self.open}/{self.close})"
            )
        return self

# ---------------------------------------------------------------------------
# strategy input
# ---------------------------------------------------------------------------


class TargetWeight(NullModel):
    """Canonical strategy output. NOT a trade list.

    ``weight`` is a signed fraction of equity: ``-0.5`` is 50% short. It is
    deliberately unbounded -- leverage is a thing a strategy may legitimately
    declare, and the capacity and drawdown gates are what judge it.
    """

    ts: Timestamp
    """Decision time."""
    symbol: Symbol
    weight: NullFloat


class TrialRecord(NullModel):
    """One variant tried on the way to the submitted strategy.

    ``returns`` is optional but valuable: supplied for every trial, it gives PBO
    (section 6.2) a real return matrix instead of a reconstructed one.
    """

    param_hash: NonEmptyStr
    sharpe: NullFloat
    returns: Series | None = None


def _canonical_universe(value: tuple[str, ...]) -> tuple[str, ...]:
    symbols = [symbol.strip() for symbol in value]
    if not symbols:
        raise ValueError("universe must contain at least one symbol")
    if any(not symbol for symbol in symbols):
        raise ValueError("universe contains an empty symbol")
    duplicates = sorted({s for s in symbols if symbols.count(s) > 1})
    if duplicates:
        raise ValueError(f"universe contains duplicate symbols: {duplicates}")
    return tuple(sorted(symbols))


def _canonical_weights(value: tuple[TargetWeight, ...]) -> tuple[TargetWeight, ...]:
    ordered = tuple(sorted(value, key=lambda w: (w.ts, w.symbol)))
    seen: set[tuple[datetime, str]] = set()
    for weight in ordered:
        key = (weight.ts, weight.symbol)
        if key in seen:
            raise ValueError(
                f"duplicate weight for {weight.symbol} at {weight.ts.isoformat()}: a "
                "strategy must state one target per symbol per decision time"
            )
        seen.add(key)
    return ordered


Universe = Annotated[tuple[Symbol, ...], AfterValidator(_canonical_universe)]
Weights = Annotated[tuple[TargetWeight, ...], AfterValidator(_canonical_weights)]


class StrategyRun(NullModel):
    """A strategy submitted for audit."""

    strategy_id: NonEmptyStr
    param_hash: NonEmptyStr
    """Hash of the exact param set used."""

    n_trials: int = Field(ge=1)
    """HOW MANY VARIANTS WERE TRIED TO GET HERE. Required, never defaulted.

    A strategy that will not declare how many variants were tried cannot be
    audited. Declaring 1 after a 5,000-run grid search is fraud; the
    deflated-Sharpe gate is what catches it.
    """

    universe: Universe
    """Canonically sorted and deduplicated so caller ordering cannot move the hash."""

    weights: Weights
    """Canonically ordered by (ts, symbol), one target per symbol per decision time."""

    trials: tuple[TrialRecord, ...] = ()
    """Per-variant evidence. May be empty or a subset; never exceeds ``n_trials``.

    When absent, section 6.1 must assume a variance across trial Sharpes, and the
    gate rationale has to say so out loud.
    """

    decision_lag_bars: int = Field(default=1, ge=1)
    """A signal computed on bar ``t`` close may not fill before bar ``t+1`` open."""

    initial_capital: PositiveFloat

    @model_validator(mode="after")
    def _check_coherence(self) -> Self:
        unknown = sorted({w.symbol for w in self.weights} - set(self.universe))
        if unknown:
            raise ValueError(
                f"weights reference symbols outside the declared universe: {unknown}. "
                "An undeclared symbol cannot be checked for point-in-time membership."
            )
        if len(self.trials) > self.n_trials:
            raise ValueError(
                f"{len(self.trials)} trial records supplied but only {self.n_trials} "
                "trials declared"
            )
        return self

# ---------------------------------------------------------------------------
# evidence components
# ---------------------------------------------------------------------------


class PerfMetrics(NullModel):
    """The metric panel from BUILD.md section 4."""

    cagr: NullFloat
    vol_annual: NonNegativeFloat
    sharpe: NullFloat
    sortino: NullFloat
    max_drawdown: NonNegativeFloat
    """Positive fraction: 0.35 means a 35% peak-to-trough decline."""
    calmar: NullFloat
    longest_underwater_days: int = Field(ge=0)
    hit_rate: Probability
    avg_win: NullFloat
    avg_loss: NullFloat
    turnover_annual: NonNegativeFloat
    """Annualised, two-sided."""
    time_in_market: Probability
    tail_ratio: NonNegativeFloat
    worst_5_days: tuple[NullFloat, ...] = Field(max_length=5)
    """Worst daily returns, ascending. Shorter than 5 only for a shorter sample."""
    n_obs: int = Field(ge=0)


class RegressionResult(NullModel):
    """Strategy excess returns regressed on benchmark excess returns.

    Section 4, rule 4: alpha with a t-stat below 2 is not alpha.
    """

    alpha_annual: NullFloat
    alpha_stderr: NonNegativeFloat
    alpha_tstat: NullFloat
    beta: NullFloat
    beta_tstat: NullFloat
    r_squared: Probability
    n_obs: int = Field(ge=0)


class FoldResult(NullModel):
    """One walk-forward fold, purged and embargoed (section 6.5)."""

    fold_index: int = Field(ge=0)
    train_start: Timestamp
    train_end: Timestamp
    test_start: Timestamp
    test_end: Timestamp
    purged_bars: int = Field(ge=0)
    embargo_bars: int = Field(ge=0)
    metrics: PerfMetrics
    net_return: NullFloat
    """Net of everything, over the test window."""

    @model_validator(mode="after")
    def _check_windows(self) -> Self:
        if self.train_end < self.train_start:
            raise ValueError("train window ends before it starts")
        if self.test_end < self.test_start:
            raise ValueError("test window ends before it starts")
        return self


class ParamPoint(NullModel):
    """One point on the parameter neighbourhood surface (section 6.7)."""

    param_hash: NonEmptyStr
    offsets: dict[str, int]
    """Step offset per parameter from the submitted point. All-zero is the peak."""
    sharpe: NullFloat


class SensitivityResult(NullModel):
    """A spike is curve-fitting. A plateau is (weak) evidence of structure."""

    param_names: tuple[NonEmptyStr, ...]
    peak_sharpe: NullFloat
    neighborhood_mean_sharpe: NullFloat
    neighborhood_ratio: NullFloat
    """Neighbourhood mean over peak. The gate wants this at 0.60 or better."""
    points: tuple[ParamPoint, ...]


LeakageKind = Literal[
    "decision_lag",
    "timestamp_monotonicity",
    "survivorship",
    "corporate_action",
    "nan_ffill",
    "universe_rebalance_timing",
]


class LeakageFlag(NullModel):
    """A finding from the leakage audit (section 5).

    A single ``fatal`` flag short-circuits the whole audit to REJECT. Do not
    compute a Sharpe on a strategy that can see the future -- you will be tempted
    to believe the number.
    """

    kind: LeakageKind
    severity: Literal["fatal", "warning"]
    symbol: Symbol | None = None
    ts: Timestamp | None = None
    detail: NonEmptyStr
    """Plain English, names the symbol and the date. Goes into the report."""

    @property
    def is_fatal(self) -> bool:
        return self.severity == "fatal"


class Evidence(NullModel):
    """Everything the gates consume. Produced by the audit pipeline.

    Gates are pure functions of this object: no I/O, no state, no globals.
    """

    equity_curve: Series
    benchmark_curve: Series
    net_returns: Series
    gross_returns: Series
    cost_breakdown: dict[str, NullFloat]
    """Charge component -> total currency amount. Keys sorted at serialisation."""
    turnover_annual: NonNegativeFloat
    time_in_market: Probability
    metrics: PerfMetrics
    benchmark_metrics: PerfMetrics
    alpha: RegressionResult
    deflated_sharpe: Probability
    """Probability the true Sharpe exceeds zero, after deflation (section 6.1)."""
    pbo: Probability
    reality_check_p: Probability
    mtrl_years: NonNegativeFloat
    walkforward: tuple[FoldResult, ...]
    regimes: dict[str, PerfMetrics]
    sensitivity: SensitivityResult
    leakage_flags: tuple[LeakageFlag, ...]

    @property
    def has_fatal_leakage(self) -> bool:
        return any(flag.is_fatal for flag in self.leakage_flags)

# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------


class GateResult(NullModel):
    """The outcome of one gate.

    ``rationale`` is written for a human reader and names the observed number,
    the threshold, and *why* it failed. These strings are the product.
    """

    name: NonEmptyStr
    passed: bool
    observed: NullFloat | str
    threshold: NullFloat | str
    rationale: NonEmptyStr


class Verdict(NullModel):
    """The artifact. Default REJECT.

    The PASS invariant is enforced here rather than only in the engine: a Verdict
    that claims PASS while carrying a failing gate, or carrying no gates at all,
    is unconstructible. Missing evidence never passes (CLAUDE.md invariant 6).
    """

    result: Literal["REJECT", "PASS"]
    gates: tuple[GateResult, ...]
    evidence_hash: Sha256Hex
    spec_version: NonEmptyStr
    generated_from: StrategyRun

    @model_validator(mode="after")
    def _check_default_reject(self) -> Self:
        if self.result != "PASS":
            return self
        if not self.gates:
            raise ValueError(
                "PASS with zero gates: a strategy that was never tested is rejected, "
                "not accepted"
            )
        failed = sorted(gate.name for gate in self.gates if not gate.passed)
        if failed:
            raise ValueError(
                f"PASS recorded while these gates failed: {failed}. Every gate must "
                "pass; the verdict is an AND, never a majority."
            )
        return self

def _assert_no_runtime_surprises() -> None:
    """Guard against a stdlib change silently breaking float canonicalisation."""
    if _canonical_float(0.1 + 0.2) != 0.3:  # pragma: no cover - environment sanity
        raise RuntimeError(
            "float quantisation is not behaving as specified; determinism is not safe"
        )


_assert_no_runtime_surprises()
