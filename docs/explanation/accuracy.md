---
icon: lucide/ruler
---

# Numerical accuracy

The maths runs on [`statrs`](https://docs.rs/statrs), so for the special-function families
(`Normal`, `LogNormal`, `Beta`, `Binomial`) accuracy is largely inherited: where `statrs` is good,
so is this library. What is *not* inherited is the composition around it, and that is where the
mistakes live. This page states the posture, the tolerances, and the limits worth knowing.

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

## Use the log methods in the tails

`cdf` and `sf` return `0.0` once the true value drops below ~`1e-308`, which is a `float64` range
limit and not something an algorithm can fix. `log_cdf` and `log_sf` stay finite far past that and
are the right methods for tail scoring. Likewise `isf(q)` rather than `ppf(1 - q)`: forming the
complement quantises the tail mass to `1.1e-16` absolute before any inverse runs.

## Known limits

* **Discrete `ppf` / `isf` at a step boundary.** `Binomial.ppf` is a binary search resolved to the
  cdf's own precision, so a quantile within `1e-12` *relative* of a cdf step may return the
  neighbouring support point. Parity with `scipy` is gated at integer equality rather than at a
  float tolerance.
* **`Beta` and `Binomial` at very large shapes.** `cdf` / `sf` run `statrs`' regularized
  incomplete-beta continued fraction, which stops at 141 terms. That cap is first exceeded somewhere
  between shapes of `1e4` and `1e5`, after which the fraction truncates silently: `Beta(s, s).cdf(0.5)`
  is exactly `0.5` by symmetry, and statrs returns `0.4912` at `s = 1e6`, `0.2129` at `1e7`, and
  `-1.147` at `1e8`, a probability outside `[0, 1]`. The audited range covers shapes to `1e3`.
* **`float64` range, not algorithm.** Some quantities genuinely exceed the type: `LogNormal`'s
  variance overflows above `sigma ~ 18.8`, and `Exponential.pdf` lands in the subnormal range once
  `rate * x` passes ~708, where only one or two significant digits remain. `log_pdf` is exact well
  past that.

## What to do if you find a defect

Two acceptable outcomes and no third: the algorithm gets fixed, or the caveat gets documented here
with a regime and a magnitude. A runtime warning is never the fix, because it cannot fire per-row
from inside the engine. See [Contributing > Numerical stability](../contributing.md#numerical-stability)
for the rules a new distribution has to satisfy.
