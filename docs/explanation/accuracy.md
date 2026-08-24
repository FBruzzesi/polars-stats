---
icon: lucide/ruler
---

# Numerical accuracy

The maths runs on [`statrs`](https://docs.rs/statrs), so for the special-function families
(`Normal`, `LogNormal`, `Beta`, `Binomial`) accuracy is largely inherited: where `statrs` is good,
so is this library, and where `statrs` 0.19 has a known defect, this release inherits that too. The
inherited defects are tracked upstream and listed below with a regime and a magnitude, next to the
limits that are structural. This page states the posture, the tolerances, and every limit worth
knowing.

## How it is checked

`make audit` (`tools/accuracy_audit.py`) sweeps every distribution, every public method and every
parameter regime against an [`mpmath`](https://mpmath.org) oracle at 50 digits, including inputs
many decades past where `scipy` itself saturates. Each probe is *classified* rather than merely
measured, because a single relative-error number says nothing about the case that matters most: a
method that returns `-inf` or `0.0` where the true value is finite and representable is a different
kind of failure from one that is off in the last few digits.

The tolerance a method claims depends on what it is made of: `1e-12` for elementary closed forms,
`1e-10` for special-function methods, `1e-9` for log-scale results, and `1e-6` or integer equality
for a discrete `ppf` resolved by binary search.

One honesty note for this release: an unfiltered sweep does not run to completion, because the
`Beta.ppf` / `isf` extreme-tail probes land in the `statrs` non-termination band documented below.
Those two methods have to be skipped at the CLI (`--skip`) until the upstream fix lands; a skip is
recorded in the report with its reason rather than dropped.

## Use the log methods in the tails, where they are log methods

`cdf` and `sf` return `0.0` once the true value drops below ~`1e-308`, which is a `float64` range
limit and not something an algorithm can fix. For `Normal`, `LogNormal` and the closed-form
distributions (`Uniform`, `Exponential`, `Bernoulli`), `log_cdf` and `log_sf` stay finite far past
that and are the right methods for tail scoring. Likewise `isf(q)` rather than `ppf(1 - q)` on those
five: forming the complement quantises the tail mass to `1.1e-16` absolute before any inverse runs.

In this release that advice does **not** extend to `Beta` and `Binomial`. Their `log_cdf` / `log_sf`
are the linear value's logarithm, so they return `-inf` at exactly the point where `cdf` / `sf`
underflow, and their `isf` is the `ppf(1 - quantile)` composition, so it carries the complement's
quantisation (`1.1e-16 / q` relative, and for `q` below `1.1e-16` the complement is exactly `1.0`,
so `Beta.isf` returns the upper support bound `1.0` and `Binomial.isf` returns `n`). The regimes and
magnitudes are in the inherited limits below.

## Known limits

### Structural

* **Discrete `ppf` / `isf` at a step boundary.** `Binomial.ppf` is a binary search resolved to the
  cdf's own precision, so a quantile within `1e-12` *relative* of a cdf step may return the
  neighbouring support point. Parity with `scipy` is gated at integer equality rather than at a
  float tolerance.
* **A constant parameter and a column parameter can differ in the last bit.** `Uniform(-2.5, 7.5).cdf("x")`
  and `Uniform(pl.col("lo"), pl.col("hi")).cdf("x")` evaluate the same closed form, but polars folds the
  constant spelling over one row and the column spelling over `n`, and the two kernels do not round
  identically. It reaches the methods evaluated as polars expressions rather than in Rust: `Uniform`'s
  moments and value-keyed methods, and `Exponential`'s value-keyed methods. The difference is at or below
  `1e-15` relative, roughly 4x a double's ULP. Everything backed by `statrs` runs the same Rust body either
  way and stays bit-identical.
* **`float64` range, not algorithm.** Some quantities genuinely exceed the type: `LogNormal`'s
  variance overflows above `sigma ~ 18.8`, and `Exponential.pdf` lands in the subnormal range once
  `rate * x` passes ~708, where only one or two significant digits remain. `log_pdf` is exact well
  past that.

### Inherited from `statrs` 0.19, tracked upstream

The defects below live in the `statrs` 0.19 routines this release binds. Each is being reported
upstream: filed reports are linked, and the remaining links are added here as each one is filed. Per
the contract at the bottom of this page, a bullet comes out of this list when a fix lands.

One `statrs` 0.19 regression is already intercepted inside `polars-stats` and is deliberately *not*
listed: `Beta.log_pdf` for `x` within `1e-9` below `1`, where `statrs` returns `-inf` across the
whole band. `beta.rs::ln_pdf_value` computes that band directly and agrees with `statrs` 0.18 to
1 ulp, so the method behaves correctly here.

* **`Beta.ppf` / `isf` in the extreme lower tail.** The `statrs` inverse (AS 64) has an unguarded
  Newton step and an unbounded step-halving loop, which costs three regimes. Below `q ~ 1e-165` it
  **panics**, and a panic inside the plugin aborts the whole query, not just the row. For `q` in
  roughly `[1e-150, 1e-60]` at some shapes below 1 it **does not return** in reasonable time (over
  15 s for a *single* row at `(a, b) = (200, 2)`). And its convergence floor is absolute rather than
  relative, so results **saturate** to one constant across decades: `Beta(0.05, 0.05).ppf(q)` returns
  the same value from `q = 1e-40` all the way down to `1e-160`. `q >= ~1e-40` at ordinary shapes is
  well-behaved. `isf` composes `ppf(1 - q)`, so it shares all three regimes on top of the complement
  quantisation above.
* **`Beta` and `Binomial` `log_cdf` / `log_sf` are linear compositions.** They return `-inf` as soon
  as `cdf` / `sf` underflows: `Beta(200, 2).log_cdf(0.001)` is `-inf` against a true `-1376.25`, and
  `Binomial(100_000, 0.001).log_sf(5000)` is `-inf` against a true `-14791.39`. `scipy`'s `logcdf` /
  `logsf` are naive for this family too, so the two libraries agree while both are `-inf`. The fix is
  a log-space regularized incomplete beta, tracked upstream as
  [statrs PR #421](https://github.com/statrs-dev/statrs/pull/421).
* **`Beta` and `Binomial` at very large shapes.** `cdf` / `sf` run `statrs`' regularized
  incomplete-beta continued fraction, which stops at 141 terms. That cap is first exceeded somewhere
  between shapes of `1e4` and `1e5`, after which the fraction truncates silently: `Beta(s, s).cdf(0.5)`
  is exactly `0.5` by symmetry, and statrs returns `0.4912` at `s = 1e6`, `0.2129` at `1e7`, and
  `-1.147` at `1e8`, a probability outside `[0, 1]`. The audited range covers shapes to `1e3`.
* **`Beta.sf` saturates in the lower corner for shapes below 1.** `statrs` evaluates it as
  `beta_reg(b, a, 1 - x)`, passing the complement as the argument, which rounds to exactly `1` below
  `x ~ 1e-16`: `Beta(0.05, 0.05).sf(1e-16)` returns `1.0` where the true value is `0.9205`. Filed
  upstream: [statrs#432](https://github.com/statrs-dev/statrs/issues/432).
* **`Beta.pdf` and `log_pdf` disagree at a boundary whose shape is below 1.** The true density
  diverges there (`scipy` returns `inf`). `statrs`' `pdf` returns `inf` at small shapes but `0.0`
  once the opposite shape passes ~80 and it switches to evaluating `ln_pdf().exp()`:
  `Beta(0.5, 0.5).pdf(0.0)` is `inf` while `Beta(0.5, 100).pdf(0.0)` is `0.0`. `log_pdf` is `-inf`
  at both endpoints regardless of shape.
* **`LogNormal.pdf` underflows in a left-tail band.** It returns `0.0` across up to 49 decades where
  the density is finite and representable: `LogNormal(0, 5).pdf(1.38e-87)` is `0.0` against a true
  `2.11e-262`. `log_pdf` is exact there (`-602.55` at that point) and is the workaround.
* **`Binomial.entropy` returns `NaN` once any mass underflows.** The support sum takes
  `pmf * ln(pmf)` without the `x ln x -> 0` limit, so a single underflowed term poisons the whole
  sum: `Binomial(50, 1e-8)` and `Binomial(1000, 0.999)` both return `NaN` where the true entropy is
  finite.
* **`statrs` 0.19 widens two parameter-space branches** (an absolute `1e-9` epsilon on parameter
  comparisons; bounded effect, accepted as-is). `Binomial` with `p` within `1e-9` of `1` is treated
  as the degenerate `p = 1`: `Binomial(10^6, 1 - 1e-10).pmf(10^6)` is `1.0` against a true
  `0.99990`. `Beta` with *both* shapes within `1e-9` of `1` takes the uniform closed form, a
  relative error of at most ~`1e-9`.

## What to do if you find a defect

Two acceptable outcomes and no third: the algorithm gets fixed, or the caveat gets documented here
with a regime and a magnitude. A runtime warning is never the fix, because it cannot fire per-row
from inside the engine. See [Contributing > Numerical stability](../contributing.md#numerical-stability)
for the rules a new distribution has to satisfy.
