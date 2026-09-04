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
from typing import TYPE_CHECKING, Annotated, Self

import numpy as np
import numpy.typing as npt
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps import cost off the hot path
    import pandas as pd

__all__ = [
    "Bar",
    "FLOAT_SIGNIFICANT_DIGITS",
    "IST",
    "NullModel",
    "SPEC_VERSION",
    "Series",
    "StrategyRun",
    "TargetWeight",
    "TrialRecord",
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

def _assert_no_runtime_surprises() -> None:
    """Guard against a stdlib change silently breaking float canonicalisation."""
    if _canonical_float(0.1 + 0.2) != 0.3:  # pragma: no cover - environment sanity
        raise RuntimeError(
            "float quantisation is not behaving as specified; determinism is not safe"
        )


_assert_no_runtime_surprises()
