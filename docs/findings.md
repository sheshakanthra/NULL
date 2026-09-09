# Bugs NULL found in itself

Destined for the M7 README. NULL's claim is that unaudited output should not be
trusted; the honest form of that claim includes what auditing NULL turned up.

Each of these was a **green build hiding a real defect**. None was found by reading
the code.

---

## 1. A test passing on the sign of floating-point noise

`benchmark_clone` holds the index, so it tracks it exactly and the regression
residuals sit at the floating-point floor. Alpha and its standard error both
collapse toward zero, and the t-stat becomes `tiny/tiny` — numerically meaningless,
**with a sign that is noise**.

It came out at **−4.5867**, which rejected, and all nine acceptance tests went
green. Had the noise gone the other way it would have been **+4.5867** and
`benchmark_clone` would have **PASSED** — which BUILD.md §4 says means the harness
is broken. The bug would have shipped behind a green test, in the fixture whose
entire job is to detect a broken harness.

Fixed by treating a residual sum of squares below 1e-18 of total as *no evidence*:
t-stats zeroed, default REJECT applies. Verified across 40 seeds.

## 2. A one-bar look-ahead, in the module that detects look-ahead

Portfolio weights were aligned to returns as `held[1:]`, pairing the position in
force at the **end** of a period with the return earned **during** it.

It survived from M2 to the multi-symbol work because **every fixture exercising
that path held a constant weight**, and a constant weight is identical under both
alignments. No amount of regression testing against `benchmark_clone` could have
found it. An analytic two-symbol assertion — `w_A·r_A + w_B·r_B` computed by hand —
found it on the first run.

## 3. Look-ahead written into the fixture built to test for look-ahead

`overfit_grid`'s moving-average helper computed the position at bar `t` from prices
including `prices[t]`, then multiplied it by `noise[t]` — the very return that
produced that price.

It showed up as PBO 0.0 with the in-sample winner ranking in the **top 1%**
out-of-sample, because look-ahead is *persistent* rather than overfit. The fixture
was wrong, not the gate.

## 4. A dependency hole invisible to the test suite

`pyarrow` was an undeclared runtime dependency for **three commits**. `pandas`
cannot read or write parquet without an engine, and every committed cache is
parquet.

CI stayed green because **every test touching parquet was skipped for missing
data** — the TRI cache does not exist, so the tests that would have exercised the
dependency never ran. It surfaced only when the OHLCV loader added tests that
actually wrote a file, and then failed on all four matrix combinations at once.
Locally it had passed 352 tests throughout, because `pyarrow` happened to be
installed as a transitive dependency of something else.

## 5. A headline claim that was a coin flip standing on a boundary

Not a code defect — a **claim** defect, which is why it belongs here rather than in
a changelog. The M7 README's opening was: *selecting on gross Sharpe picks a
variant that loses money net of costs.* True at the configured rates. The
best-by-gross variant nets **−0.027**.

That number is a **sign test**, and it sits 0.027 from flipping. Scaling each
charge component by ±25% — a plausible error band for rates read off a published
schedule and never reconciled against a contract note — flips it in **4 of 25
cases**, every one of them in the direction of costs being *lower* than modelled.
STT alone, at 0.10% on both legs, is worth about 0.08 Sharpe on this turnover.

Nothing about the strategy changed. Nothing in the codebase was wrong. The claim
was simply never robust, and no test could have said so, because the claim was not
in the codebase to be tested — it was in a README, in prose, computed once at one
set of rates.

The same phenomenon stated so it does not balance on a boundary: the gross pick
**ranks 24th-to-93rd of 108** on the net-ranked grid across every case in the
sweep, was **never once** the best-by-net variant, and gives up **0.332 to 0.615**
net Sharpe. That is a bigger claim than the one it replaces, and it holds.

The fix was not to soften the README. It was to build the sweep
(`examples/rsi2_nifty/cost_robustness.py`), commit its output as an artifact, and
make `null/verdict/limitations.py` derive its cost-rate disclosure from that file
rather than from prose — so the next claim of this shape has to be measured before
it can be printed.

---

## The pattern

**Items 2 and 4 are the same shape.** In both, a test covering the defective path
*existed and passed*, and in both it was structurally incapable of discriminating:

| | the test that should have caught it | why it could not |
|---|---|---|
| **2** | constant-weight benchmark fixtures | a constant weight is identical under both alignments |
| **4** | the TRI parquet round-trip test | skipped, because the data it needed did not exist |

A test that cannot fail is not coverage. Both were fixed the same way — by making
the path exercisable **without the condition that made the test inert**:

- an analytic multi-weight fixture, whose expected values differ under the two
  alignments
- a synthetic parquet round-trip needing no external data, plus a CI job that
  installs with runtime dependencies alone and imports every module

The general rule this suggests: **when a test is skipped or its inputs are
degenerate, treat it as absent.** A skip is not a weaker pass; it is a hole with a
label on it. Item 1 is a sharper version of the same idea — the test ran, but its
outcome was decided by something that carried no information.

**Item 5 extends the rule past the test suite.** It was never covered by a test,
because it lived in prose rather than in code — but it is subject to the same
question the other four are: *could this have come out differently?* A finding
sitting 0.027 from its own sign boundary could, and one that could was never
established. The generalisation: **a result that has not been perturbed has not
been tested**, and it does not matter whether it lives in a test file or a README.
Perturbing it is cheap. Not perturbing it is how a coin flip gets printed as a
conclusion.

## Why this belongs in the report rather than a changelog

NULL exists to say that a number produced by an unaudited process should not be
believed. Four defects hidden behind passing tests, plus a headline claim that
turned out to rest on a boundary, are the strongest available evidence for that
claim — and the least comfortable. A tool that makes
this argument while concealing its own history of exactly this failure would be
making the argument dishonestly.
