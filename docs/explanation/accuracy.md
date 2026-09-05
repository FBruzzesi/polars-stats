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
distributions (`Uniform`, `Exponential`, `Bernoulli`, `Geometric`, `DiscreteUniform`), `log_cdf` and
`log_sf` stay finite far past that and are the right methods for tail scoring. Likewise `isf(q)`
rather than `ppf(1 - q)` on those seven: forming the complement quantises the tail mass to `1.1e-16`
absolute before any inverse runs.

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
  neighbouring support point. `Geometric.ppf` / `isf` decide the same tie in the log domain, testing
  `k * log1p(-p)` against `log1p(-q)` rather than re-deriving `cdf(k)`. That is deliberate and it is
  the more accurate rule: across 1997 probes sitting on and either side of exact step boundaries it
  disagrees with exact rational arithmetic 357 times, against 490 for the alternative. The price is
  that `ppf` and `cdf` are not exact mutual inverses there, and the miss goes both ways, by at most
  one support point: at `p = 1e-8` with `q = sf(1)`, `isf(q)` is `2` where `1` is the answer. On a
  step the two roundings decide the last bit, so which side a given quantile falls on is also the
  platform's `exp` and `log1p` and not the rule alone: at `p = 0.1` with `q` one ulp above `cdf(10)`,
  `ppf(q)` is `10` on Apple's libm and `11` on glibc, because their `exp` puts `cdf(10)` itself an ulp
  apart. Neither libm is [correctly rounded](https://core-math.gitlabpages.inria.fr/), and a last-bit
  difference sits well inside
  [the error glibc documents](https://www.gnu.org/software/libc/manual/html_node/Errors-in-Math-Functions.html)
  for it. Only the one-support-point bound is portable, so it is the only thing pinned. `scipy`'s
  [`geom._ppf`](https://github.com/scipy/scipy/blob/main/scipy/stats/_discrete_distns.py) made the
  opposite trade (it tests against its own `_cdf`, so it is self-consistent and less accurate). Parity
  is gated at integer equality rather than at a float tolerance.

    `DiscreteUniform.ppf` / `isf` are closed forms rather than searches, and `ppf` carries scipy's
    own rounding, bit for bit: in a one-ulp quantile window above each cdf step edge, `q * N` rounds
    down across the integer boundary and the answer sits one support point below the exact rational
    one, so there `cdf(ppf(q)) < q`. Outside that window both inverses agree with exact rational
    arithmetic, checked on `N = 6, 8, 15, 101` and `1000`. **On** an exactly representable edge the
    two contracts diverge, and this library inverts its own `sf` rather than the exact rational:
    `isf` probes both neighbours of its candidate and keeps the smallest point whose survival
    quotient still satisfies the quantile, so `isf(sf(x)) == x` holds for 5/6, 15/15 and 101/101
    support points on `(1,6)`, `(-5,9)` and `(0,100)`, against 2/6, 7/15 and 58/101 under the
    rational rule. The one remaining miss is `sf`'s, not the inverse's: its reciprocal multiply
    lands an ulp below the exact `k / N`, so the point it came from genuinely no longer satisfies
    the survival contract at that quantile, and the portable bound stays one support point.
* **A `float` evaluation point cannot address a support narrower than one float step.**
  At `2**62` one float step is 1024 wide, so all 11 points of `{min, ..., min + 10}` denote the same
  `float64`. `cdf`, `sf` and their logs then answer for the whole support at once, because `value >= max`
  is true for each of them; `pmf` / `log_pmf` test membership rather than a count, so they are
  unaffected. Passing the point as a Python `int` keeps the arithmetic exact, since the count is
  subtracted in `Int64` inside the support: `cdf(min + 5)` on `DiscreteUniform(2**62, 2**62 + 10)`
  is `6/11` from an `int` and `1.0` from the equal `float`. It applies to any support whose width is
  below `2**-52` of its magnitude.
* **`DiscreteUniform.ppf` / `isf` return fewer distinct points than such a support has.** The same
  regime on the *output* side, and there is no `int` spelling to escape to: the candidate is formed as
  `min + ceil(q * N) - 1` in `Float64`, where `min + 1 - 1` is not `min` above `2**53`.
  `DiscreteUniform(2**60, 2**60 + 4).ppf(q)` answers `min` for every `q`, and
  `DiscreteUniform(2**53 + 2, 2**53 + 12)` resolves 3 of its 11 points. Both inverses are exact for
  bounds inside `±2**53`, which is where a support that a `float64` quantile can address at all lives.
* **A moment can be correctly rounded and still land outside the support.** `mean` and `median` form
  the midpoint as `min + (max - min) // 2` in `Int64` and round once, so they are exact for bounds
  inside `±2**53` and within one ulp everywhere else. Summing the two bounds in `Float64` instead
  would round each before they cancel, which costs up to `1.2e-13` relative for bounds straddling
  zero. For a support narrower than one float step near the `Int64` extremes the midpoint can still
  sit up to half a step outside `[min, max]` when compared in exact integers: it does so for about
  99% of random supports of width `< 20` drawn from `[2**62, 2**63)`, because the bounds themselves
  are not representable there. It never happens for a support wider than one float step, nor anywhere
  inside `±2**53`.
* **A constant parameter and a column parameter can differ in the last bit.** `Uniform(-2.5, 7.5).variance()`
  and `Uniform(pl.col("lo"), pl.col("hi")).variance()` evaluate the same closed form, but polars folds the
  constant spelling over one row and the column spelling over `n`, and the two kernels do not round
  identically. It reaches the methods evaluated as polars expressions rather than in Rust: `Uniform`'s
  moments, and `Geometric.std` / `.entropy`.
  The difference is at or below `1e-15` relative, roughly 4x a double's ULP. Everything backed by `statrs`
  runs the same Rust body either way and stays bit-identical.

    `Geometric`'s two are narrower than the rest: both divide by `p`, and only a `pl.repeat(p, n=pl.len())`
    parameter moves the last bit, because polars keeps that spelling scalar-backed and divides by it with a
    reciprocal multiply. A materialised `p` column matches the constant bit for bit.
* **`float64` range, not algorithm.** Some quantities genuinely exceed the type: `LogNormal`'s
  variance overflows above `sigma ~ 18.8`, and `Exponential.pdf` lands in the subnormal range once
  `rate * x` passes ~708, where only one or two significant digits remain. `log_pdf` is exact well
  past that.
* **`UInt64` range, for a discrete sample.** `Geometric.sample` draws a trial count, which averages
  `1 / p`, so a small enough `p` puts the draw past `u64::MAX`, where it saturates. A single draw
  does so with probability `exp(-u64::MAX * p)`: negligible at `p = 1e-18`, 16% at `1e-19` and 83%
  at `1e-20`. `mean()` and the other moments are `Float64` and stay exact far past that. No other
  sampler has a reachable limit: `Bernoulli` is `Boolean` and `Binomial`'s count is bounded by `n`.

### Inherited from `statrs` 0.19, tracked upstream

The defects below live in the `statrs` 0.19 routines this release binds. Each is being reported
upstream: filed reports are linked, and the remaining links are added here as each one is filed. Per
the contract at the bottom of this page, a bullet comes out of this list when a fix lands.

* **`Beta.ppf` / `isf` in the extreme lower tail.** The `statrs` inverse
  ([`inv_beta_reg`](https://github.com/statrs-dev/statrs/blob/v0.19.1/src/function/beta.rs#L269), an
  implementation of [AS 64](https://www.jstor.org/stable/2346798)) has an unguarded Newton step and an
  unbounded step-halving loop, which costs three regimes. Below `q ~ 1e-165` it **panics**, and a panic
  inside the plugin aborts the whole query, not just the row. For `q` in roughly `[1e-150, 1e-60]` at
  some shapes below 1 it **does not return** in reasonable time (over 15 s for a *single* row at
  `(a, b) = (200, 2)`). And its convergence floor is absolute rather than
  relative, so results **saturate** to one constant across decades: `Beta(0.05, 0.05).ppf(q)` returns
  the same value from `q = 1e-40` all the way down to `1e-160`. `q >= ~1e-40` at ordinary shapes is
  well-behaved. `isf` composes `ppf(1 - q)`, so it shares all three regimes on top of the complement
  quantisation above. Filed upstream:
  [statrs#435](https://github.com/statrs-dev/statrs/issues/435).
* **`Beta` and `Binomial` `log_cdf` / `log_sf` are linear compositions.** They return `-inf` as soon
  as `cdf` / `sf` underflows: `Beta(200, 2).log_cdf(0.001)` is `-inf` against a true `-1376.25`, and
  `Binomial(100_000, 0.001).log_sf(5000)` is `-inf` against a true `-14791.39`. `scipy`'s `logcdf` /
  `logsf` are naive for this family too, so the two libraries agree while both are `-inf`. The fix is
  a log-space regularized incomplete beta, tracked upstream as
  [statrs PR #421](https://github.com/statrs-dev/statrs/pull/421).
* **`Beta` and `Binomial` at very large shapes.** `cdf` / `sf` run `statrs`' regularized
  incomplete-beta continued fraction
  ([`beta_reg`](https://github.com/statrs-dev/statrs/blob/v0.19.1/src/function/beta.rs#L130)), which
  stops at 141 terms. That cap is first exceeded somewhere between shapes of `1e4` and `1e5`, after
  which the fraction truncates silently: `Beta(s, s).cdf(0.5)` is exactly `0.5` by symmetry, and statrs
  returns `0.4912` at `s = 1e6`, `0.2129` at `1e7`, and `-1.147` at `1e8`, a probability outside
  `[0, 1]`. The audited range covers shapes to `1e3`. Filed upstream:
  [statrs#434](https://github.com/statrs-dev/statrs/issues/434).
* **`LogNormal.pdf` underflows in a left-tail band.** It returns `0.0` across up to 49 decades where
  the density is finite and representable: `LogNormal(0, 5).pdf(1.38e-87)` is `0.0` against a true
  `2.11e-262`. `log_pdf` is exact there (`-602.55` at that point) and is the workaround.

## What to do if you find a defect

Two acceptable outcomes and no third: the algorithm gets fixed, or the caveat gets documented here
with a regime and a magnitude. A runtime warning is never the fix, because it cannot fire per-row
from inside the engine. See [Contributing > Numerical stability](../contributing.md#numerical-stability)
for the rules a new distribution has to satisfy.
