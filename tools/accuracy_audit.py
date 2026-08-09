"""Tail-accuracy audit of the shipped distributions against an `mpmath` oracle at 50 digits.

    make audit                                                        # full sweep + report
    uv run --group audit tools/accuracy_audit.py beta binomial        # a subset of distributions
    uv run --group audit tools/accuracy_audit.py --samples 200        # deeper random sampling

The test suite cannot find what this looks for, by construction: parity grids are finite and
"reasonable", `scipy` is not a valid oracle in the tails (its own `logcdf` / `logsf` are naive for
the incomplete-beta family, so parity *passes* while both libraries return `-inf`), and the property
suite asserts shape rather than accuracy. So the oracle here is `mpmath` at `dps = 50`, and every
`log_*` oracle takes its logarithm **inside** the high-precision context, which is what lets the
audit see a `-inf` that should have been `-1376`.

Each probe is classified rather than merely measured, because a single relative-error number is not
actionable and is meaningless in the case that matters most:

| Category | Meaning |
| --- | --- |
| `OK` | within the tolerance the method claims |
| `UNDERFLOW` | we saturate to `-inf` / `0.0` where the oracle is finite and representable |
| `DEGRADED` | both finite, relative error exceeds the claimed tolerance |
| `SIGN` / `NAN` | wrong sign, or `NaN` / null where the oracle is finite |
| `ORACLE_UNAVAILABLE` | mpmath did not converge; never silently a pass |

Probes are the a priori danger points, plus log-uniform random sampling spanning the representable
range of the input, plus the parameter sweep. The random spread is not decoration: it is the
mechanism that caught the `Gamma` `ppf` defect at `q ~ 1e-6`, a point no curated grid contained.
The RNG is seeded from a constant, so a finding can be re-probed.

Two self-tests calibrate the instrument and run on every invocation (`--no-controls` to skip):
a **positive control** that must be reported as a defect (if the tool cannot see one we already
know exists, every clean result it produces is worthless) and a **negative control** on a path that
was hardened and must therefore read `OK`.

Probes are constructed in units of the distribution's own scale (`mu + k * sigma`, `exp(mu + k *
sigma)`, `t / rate`), so the evaluation point is representable at full precision and the audit
measures the library's arithmetic rather than the rounding of a badly-chosen literal.
"""

# ruff: noqa: T201, S311, INP001

from __future__ import annotations

import argparse
import math
import random
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import mpmath as mp
import polars as pl

from polars_stats import Bernoulli, Beta, Binomial, Exponential, LogNormal, Normal, Uniform

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from polars_stats.distributions._base import _UnivariateDistribution

DPS = 50
"""Oracle working precision. Every reference value, and every logarithm of one, is computed here."""

mp.mp.dps = DPS

SMALLEST_SUBNORMAL = mp.mpf(2) ** -1074
"""Below this an oracle value has no `float64` representation, so returning `0.0` is correct."""

SMALLEST_NORMAL = mp.mpf(sys.float_info.min)
"""Below this `float64` degrades gradually, so a *relative* tolerance is not achievable; see
[`classify`]."""

LARGEST_FINITE = mp.mpf(sys.float_info.max)
"""Above this an oracle value has no finite `float64`, so returning `inf` is correct."""

SUBNORMAL_ULPS = 2
"""Slack allowed on a subnormal result, in units of the one spacing it has left."""

DEFAULT_SEED = 20260807
DEFAULT_SAMPLES = 40

CLOSED_FORM_RTOL = 1e-12
"""Elementary closed forms: arithmetic, `exp` / `log` of a well-scaled argument."""

SPECIAL_RTOL = 1e-10
"""Special-function methods: incomplete beta, `erfc_inv`, digamma."""

ERF_RTOL = 2e-10
"""`erfc`-backed methods. Measured, not aspirational: `statrs`' `erfc` holds `1.02e-10` relative at
the worst probe in the sweep, so a `1e-10` claim is the one thing that would be wrong here."""

LOG_RTOL = 1e-9
"""Log-scale methods, where a log-gamma prefactor sets the achievable number of digits."""

DISCRETE_LOG_RTOL = 1e-8
"""Discrete log-mass and support-sum entropy. `statrs`' `ln_pmf` composes `ln_binomial` with
`(n - k) * (1 - p).ln()`, measured at `5.0e-9`; the entropy accumulates `n + 1` such terms."""

DISCRETE_PPF_RTOL = 1e-6
"""Discrete `ppf`: a binary search resolved to the cdf's own precision at a step boundary."""

Category = Literal["OK", "UNDERFLOW", "DEGRADED", "SIGN", "NAN", "PANIC", "ORACLE_UNAVAILABLE"]
CATEGORIES: tuple[Category, ...] = ("OK", "PANIC", "UNDERFLOW", "SIGN", "NAN", "DEGRADED", "ORACLE_UNAVAILABLE")
"""`PANIC` is not in the audit's original category table: a probe that aborts the whole query is
materially worse than one that returns a wrong number, so it does not belong under `NAN`."""

Params = tuple[float, ...]
Point = tuple[float, str]
Oracle = Callable[[Params, float], mp.mpf]
SeededOracle = Callable[[Params, float, float], mp.mpf]
"""An inverse with no closed form, refined from the library's own answer; see [`solve_monotone`]."""
Points = Callable[[Params, random.Random, int], list[Point]]


class OracleUnavailableError(Exception):
    """The oracle cannot produce a trustworthy value for this probe.

    Raised rather than returned so an unavailable oracle can never be mistaken for a pass; the
    caller records the reason in the report.
    """


# Small mpmath helpers.


def ln(value: mp.mpf) -> mp.mpf:
    """Natural log with `ln(0) = -inf`, the limit every `log_*` method returns there."""
    return mp.ninf if value == 0 else mp.log(value)


def real(value: mp.mpf) -> mp.mpf:
    """Drop the negligible imaginary residue `betainc` or a root-finder can leave on a real value."""
    if not isinstance(value, mp.mpc):
        return value
    if abs(value.imag) > abs(value.real) * mp.mpf(10) ** -30:
        msg = f"oracle returned a genuinely complex value ({mp.nstr(value, 8)})"
        raise OracleUnavailableError(msg)
    return value.real


def solve_monotone(residual: Callable[[mp.mpf], mp.mpf], seed: float) -> mp.mpf:
    """Root of a strictly monotone `residual`, refined from `seed` by the secant method.

    The seed is the library's own answer, which makes this fast, and safe to seed that way: on a
    strictly monotone function the root is unique, so a converged root is the true one no matter
    how bad the seed was. Convergence is verified rather than assumed.
    """
    if not math.isfinite(seed):
        msg = f"no finite seed to refine (library returned {seed})"
        raise OracleUnavailableError(msg)
    try:
        root = real(mp.findroot(residual, mp.mpf(seed), solver="secant", tol=mp.mpf(10) ** -40))
        if abs(real(residual(root))) > mp.mpf(10) ** -20:
            msg = "secant converged to a point that is not a root"
            raise OracleUnavailableError(msg)  # noqa: TRY301
    except OracleUnavailableError:
        raise
    except Exception as exc:
        msg = f"secant did not converge ({type(exc).__name__})"
        raise OracleUnavailableError(msg) from exc
    return root


# Normal and LogNormal oracles.


def std_normal_quantile(q: mp.mpf) -> mp.mpf:
    """`sqrt(2) * erfinv(2q - 1)`, at enough working precision for a deep-tail `q`.

    At the audit's own 50 digits `2q - 1` rounds to exactly `-1` for `q` below ~1e-50 and `erfinv`
    returns `-inf`, so the working precision tracks the tail's depth. Widening precision is the
    documented response to an oracle that cannot resolve a probe.
    """
    if q <= 0:
        return mp.ninf
    if q >= 1:
        return mp.inf
    tail = min(q, 1 - q)
    extra = max(0, int(-mp.log10(tail)) + 10)
    with mp.workdps(DPS + extra):
        return mp.sqrt(2) * mp.erfinv(2 * q - 1)


def normal_z(params: Params, x: float) -> mp.mpf:
    """Standardised `(x - mu) / sigma` at oracle precision."""
    mu, sigma = params
    return (mp.mpf(x) - mp.mpf(mu)) / mp.mpf(sigma)


def normal_cdf(params: Params, x: float) -> mp.mpf:
    """`0.5 * erfc(-z / sqrt(2))`."""
    return mp.erfc(-normal_z(params, x) / mp.sqrt(2)) / 2


def normal_sf(params: Params, x: float) -> mp.mpf:
    """`0.5 * erfc(z / sqrt(2))`."""
    return mp.erfc(normal_z(params, x) / mp.sqrt(2)) / 2


def normal_pdf(params: Params, x: float) -> mp.mpf:
    """`exp(-z^2 / 2) / (sigma * sqrt(2 pi))`."""
    z = normal_z(params, x)
    return mp.e ** (-z * z / 2) / (mp.mpf(params[1]) * mp.sqrt(2 * mp.pi))


def normal_log_pdf(params: Params, x: float) -> mp.mpf:
    """`-z^2 / 2 - ln(sigma) - ln(2 pi) / 2`."""
    z = normal_z(params, x)
    return -z * z / 2 - mp.log(mp.mpf(params[1])) - mp.log(2 * mp.pi) / 2


def normal_ppf(params: Params, q: float) -> mp.mpf:
    """`mu + sigma * Phi^-1(q)`."""
    mu, sigma = params
    return mp.mpf(mu) + mp.mpf(sigma) * std_normal_quantile(mp.mpf(q))


def normal_isf(params: Params, q: float) -> mp.mpf:
    """`mu - sigma * Phi^-1(q)`, the exact inverse survival function.

    Deliberately *not* the library's `ppf(1 - q)` composition: the audit measures that composition
    against the true value, which is how the complement's rounding floor gets a number.
    """
    mu, sigma = params
    return mp.mpf(mu) - mp.mpf(sigma) * std_normal_quantile(mp.mpf(q))


def normal_entropy(params: Params, _x: float) -> mp.mpf:
    """`0.5 * ln(2 pi e sigma^2)`."""
    return mp.log(2 * mp.pi * mp.e * mp.mpf(params[1]) ** 2) / 2


def lognormal_z(params: Params, x: float) -> mp.mpf:
    """Standardised `(ln x - mu) / sigma`; the caller guards `x > 0`."""
    mu, sigma = params
    return (mp.log(mp.mpf(x)) - mp.mpf(mu)) / mp.mpf(sigma)


def lognormal_cdf(params: Params, x: float) -> mp.mpf:
    """`0` at/below the support, else the underlying normal's cdf at `ln x`."""
    return mp.mpf(0) if x <= 0 else mp.erfc(-lognormal_z(params, x) / mp.sqrt(2)) / 2


def lognormal_sf(params: Params, x: float) -> mp.mpf:
    """`1` at/below the support, else the underlying normal's sf at `ln x`."""
    return mp.mpf(1) if x <= 0 else mp.erfc(lognormal_z(params, x) / mp.sqrt(2)) / 2


def lognormal_pdf(params: Params, x: float) -> mp.mpf:
    """`exp(-z^2 / 2) / (x sigma sqrt(2 pi))`, `0` at/below the support."""
    if x <= 0:
        return mp.mpf(0)
    z = lognormal_z(params, x)
    return mp.e ** (-z * z / 2) / (mp.mpf(x) * mp.mpf(params[1]) * mp.sqrt(2 * mp.pi))


def lognormal_log_pdf(params: Params, x: float) -> mp.mpf:
    """`-z^2 / 2 - ln x - ln(sigma) - ln(2 pi) / 2`, `-inf` at/below the support."""
    if x <= 0:
        return mp.ninf
    z = lognormal_z(params, x)
    return -z * z / 2 - mp.log(mp.mpf(x)) - mp.log(mp.mpf(params[1])) - mp.log(2 * mp.pi) / 2


def lognormal_ppf(params: Params, q: float) -> mp.mpf:
    """`exp(mu + sigma Phi^-1(q))`."""
    mu, sigma = params
    return mp.e ** (mp.mpf(mu) + mp.mpf(sigma) * std_normal_quantile(mp.mpf(q)))


def lognormal_isf(params: Params, q: float) -> mp.mpf:
    """`exp(mu - sigma Phi^-1(q))`; see [`normal_isf`] for why the exact form is the oracle."""
    mu, sigma = params
    return mp.e ** (mp.mpf(mu) - mp.mpf(sigma) * std_normal_quantile(mp.mpf(q)))


# Exponential and Uniform oracles. These two are the elementary controls and should come back clean.


def exponential_cdf(params: Params, x: float) -> mp.mpf:
    """`-expm1(-rate x)` on the support, `0` below it."""
    return mp.mpf(0) if x < 0 else -mp.expm1(-mp.mpf(params[0]) * mp.mpf(x))


def exponential_sf(params: Params, x: float) -> mp.mpf:
    """`exp(-rate x)` on the support, `1` below it."""
    return mp.mpf(1) if x < 0 else mp.e ** (-mp.mpf(params[0]) * mp.mpf(x))


def exponential_pdf(params: Params, x: float) -> mp.mpf:
    """`rate exp(-rate x)` on the support, `0` below it."""
    return mp.mpf(0) if x < 0 else mp.mpf(params[0]) * mp.e ** (-mp.mpf(params[0]) * mp.mpf(x))


def exponential_log_pdf(params: Params, x: float) -> mp.mpf:
    """`ln(rate) - rate x` on the support, `-inf` below it."""
    return mp.ninf if x < 0 else mp.log(mp.mpf(params[0])) - mp.mpf(params[0]) * mp.mpf(x)


def exponential_log_sf(params: Params, x: float) -> mp.mpf:
    """`-rate x` on the support, `0` below it."""
    return mp.mpf(0) if x < 0 else -mp.mpf(params[0]) * mp.mpf(x)


def exponential_ppf(params: Params, q: float) -> mp.mpf:
    """`-log1p(-q) / rate`."""
    return mp.inf if q >= 1 else -mp.log1p(-mp.mpf(q)) / mp.mpf(params[0])


def exponential_isf(params: Params, q: float) -> mp.mpf:
    """`-ln(q) / rate`, the exact inverse survival function."""
    return -ln(mp.mpf(q)) / mp.mpf(params[0])


def uniform_span(params: Params) -> tuple[mp.mpf, mp.mpf, mp.mpf]:
    """`(lo, hi, width)` at oracle precision."""
    lo, hi = mp.mpf(params[0]), mp.mpf(params[1])
    return lo, hi, hi - lo


def uniform_cdf(params: Params, x: float) -> mp.mpf:
    """`(x - lo) / width`, clamped; `1` at/above `hi`, matching the closed-form hook."""
    lo, hi, width = uniform_span(params)
    if x < lo:
        return mp.mpf(0)
    return mp.mpf(1) if x >= hi else (mp.mpf(x) - lo) / width


def uniform_sf(params: Params, x: float) -> mp.mpf:
    """`(hi - x) / width`, clamped; `0` at/above `hi`."""
    lo, hi, width = uniform_span(params)
    if x < lo:
        return mp.mpf(1)
    return mp.mpf(0) if x >= hi else (hi - mp.mpf(x)) / width


def uniform_pdf(params: Params, x: float) -> mp.mpf:
    """`1 / width` on the closed support, `0` outside."""
    lo, hi, width = uniform_span(params)
    return 1 / width if lo <= x <= hi else mp.mpf(0)


def uniform_log_pdf(params: Params, x: float) -> mp.mpf:
    """`-ln(width)` on the closed support, `-inf` outside."""
    lo, hi, width = uniform_span(params)
    return -mp.log(width) if lo <= x <= hi else mp.ninf


def uniform_ppf(params: Params, q: float) -> mp.mpf:
    """`lo + q * width`."""
    lo, _, width = uniform_span(params)
    return lo + mp.mpf(q) * width


def uniform_isf(params: Params, q: float) -> mp.mpf:
    """`hi - q * width`, the exact inverse survival function."""
    _, hi, width = uniform_span(params)
    return hi - mp.mpf(q) * width


# Bernoulli oracles.


def bernoulli_pmf(params: Params, x: float) -> mp.mpf:
    """`1 - p` at `0`, `p` at `1`, `0` off the support."""
    p = mp.mpf(params[0])
    if x == 0:
        return 1 - p
    return p if x == 1 else mp.mpf(0)


def bernoulli_cdf(params: Params, x: float) -> mp.mpf:
    """`0` below `0`, `1 - p` on `[0, 1)`, `1` at/above `1`."""
    p = mp.mpf(params[0])
    if x < 0:
        return mp.mpf(0)
    return 1 - p if x < 1 else mp.mpf(1)


def bernoulli_sf(params: Params, x: float) -> mp.mpf:
    """`1` below `0`, `p` on `[0, 1)`, `0` at/above `1`."""
    p = mp.mpf(params[0])
    if x < 0:
        return mp.mpf(1)
    return p if x < 1 else mp.mpf(0)


def bernoulli_ppf(params: Params, q: float) -> mp.mpf:
    """Smallest support point with `cdf >= q`: `0` while `q + p <= 1`, else `1`.

    In exact rationals, not at 50 digits, and this is not fussiness. `q` and `p` are float64s, so
    they span 600 decades between them, and *any* spelling that forms `1 - p` or `1 - q` collapses
    to exactly `1` at one end of that range: `q <= 1 - p` reported a false defect at `p = 1e-300`,
    and `1 - q >= p` did the same at `p = 1` for a tiny `q`. Widening the working precision would
    also work, but only if the widening tracks both arguments; `Fraction` needs no such reasoning.

    Worth noting *why* this bit `ppf` and not `cdf`, whose oracle forms the same `1 - p`. A rounded
    linear value agrees with the correctly-rounded float64 the library returns, so the error cancels
    out of the comparison. A **discrete** inverse turns the same rounding into a jump between support
    points, which no tolerance absorbs. Oracles for discrete inverses need exact arithmetic.
    """
    return mp.mpf(0) if Fraction(q) + Fraction(params[0]) <= 1 else mp.mpf(1)


def bernoulli_isf(params: Params, q: float) -> mp.mpf:
    """Smallest support point with `sf <= q`: `1` while `q < p`, else `0`."""
    return mp.mpf(1) if mp.mpf(q) < mp.mpf(params[0]) else mp.mpf(0)


def bernoulli_entropy(params: Params, _x: float) -> mp.mpf:
    """`-p ln p - (1 - p) ln(1 - p)`, with the `0 ln 0 = 0` limit at the endpoints."""
    p = mp.mpf(params[0])
    if p in (0, 1):
        return mp.mpf(0)
    return -p * mp.log(p) - (1 - p) * mp.log1p(-p)


# Beta oracles.


def ln_beta_fn(a: mp.mpf, b: mp.mpf) -> mp.mpf:
    """`ln B(a, b)` via log-gamma, so a large shape does not overflow the Beta function itself."""
    return mp.loggamma(a) + mp.loggamma(b) - mp.loggamma(a + b)


def beta_cdf(params: Params, x: float) -> mp.mpf:
    """`I_x(a, b)`, clamped outside the support."""
    a, b = params
    if x <= 0:
        return mp.mpf(0)
    return mp.mpf(1) if x >= 1 else mp.betainc(a, b, 0, x, regularized=True)


def beta_sf(params: Params, x: float) -> mp.mpf:
    """`1 - I_x(a, b)`, via the reflection `I_{1-x}(b, a)`.

    **Not** `betainc(a, b, x, 1, regularized=True)`, which is the obvious spelling and is wrong in
    exactly the corner this audit exists to probe: at `(a, b) = (2, 200)`, `x = 1 - 1e-3` it returns
    `0.0` and at `(90, 80)`, `x = 1 - 1e-4` a *negative* `-1.8e-54`, against true values of `2.0e-598`
    and `3.6e-271`. mpmath integrates from `x` to `1` by differencing two near-equal quantities, so
    the whole result is cancellation. The reflection has no such subtraction, and is also ~1000x
    faster at large shapes.
    """
    a, b = params
    if x <= 0:
        return mp.mpf(1)
    if x >= 1:
        return mp.mpf(0)
    # `1 - x` is the argument, so the oracle needs enough digits to hold `x` *below* the leading 1:
    # at 50 digits `1 - 3.2e-57` is exactly `1` and the reflection returns a flat `1.0`, which then
    # reads as a library defect. Same widening as `std_normal_quantile`.
    with mp.workdps(DPS + max(0, int(-mp.log10(x)) + 10)):
        return mp.betainc(b, a, 0, 1 - mp.mpf(x), regularized=True)


def beta_log_pdf(params: Params, x: float) -> mp.mpf:
    """`(a - 1) ln x + (b - 1) ln(1 - x) - ln B(a, b)`, with the boundary limits."""
    a, b = mp.mpf(params[0]), mp.mpf(params[1])
    if x < 0 or x > 1:
        return mp.ninf
    if x == 0:
        return mp.inf if a < 1 else (-ln_beta_fn(a, b) if a == 1 else mp.ninf)
    if x == 1:
        return mp.inf if b < 1 else (-ln_beta_fn(a, b) if b == 1 else mp.ninf)
    return (a - 1) * mp.log(mp.mpf(x)) + (b - 1) * mp.log1p(-mp.mpf(x)) - ln_beta_fn(a, b)


def beta_pdf(params: Params, x: float) -> mp.mpf:
    """`exp` of [`beta_log_pdf`], which keeps the large-shape prefactor out of the linear domain."""
    log_density = beta_log_pdf(params, x)
    if log_density == mp.ninf:
        return mp.mpf(0)
    return mp.inf if log_density == mp.inf else mp.e**log_density


FALLBACK_LOG_SEED = math.log(1e-8)
"""Seed used when the library's own answer is saturated at a support bound and cannot be
transformed. The root of a strictly monotone function is unique, so any finite seed converges to
it; only the iteration count suffers."""


def ln_seed(x: float) -> float:
    """`ln(x)` as a secant seed, falling back when the value under test is not in the open support."""
    return math.log(x) if x > 0 else FALLBACK_LOG_SEED


def ln1p_seed(x: float) -> float:
    """`ln(1 - x)` as a secant seed, with the same fallback as [`ln_seed`]."""
    return math.log1p(-x) if x < 1 else FALLBACK_LOG_SEED


def beta_lower_root(params: Params, target: mp.mpf, seed: float) -> mp.mpf:
    """`x` with `ln I_x(a, b) = target`, solved in `ln x` so a deep left corner stays well-scaled."""
    a, b = params
    root = solve_monotone(lambda u: ln(mp.betainc(a, b, 0, mp.e**u, regularized=True)) - target, ln_seed(seed))
    return mp.e**root


def beta_upper_root(params: Params, target: mp.mpf, seed: float) -> mp.mpf:
    """`x` with `ln(1 - I_x(a, b)) = target`, solved in `y = ln(1 - x)` for the same reason.

    The residual is the reflection `I_{e^y}(b, a)`, never the `x`-to-`1` integral; see [`beta_sf`].
    """
    a, b = params
    root = solve_monotone(lambda y: ln(mp.betainc(b, a, 0, mp.e**y, regularized=True)) - target, ln1p_seed(seed))
    return cast("mp.mpf", 1 - mp.e**root)


def beta_ppf(params: Params, q: float, seed: float) -> mp.mpf:
    """`I^-1_q(a, b)`, refined from the library's own answer on whichever side of the median is small.

    Seeding from the value under test is safe on a strictly monotone cdf (the root is unique, so a
    converged root is the true quantile however bad the seed), and it is what makes a 50-digit
    inverse cheap enough to run over the whole sweep.
    """
    if q <= 0:
        return mp.mpf(0)
    if q >= 1:
        return mp.mpf(1)
    if mp.mpf(q) <= mp.mpf(0.5):
        return beta_lower_root(params, mp.log(mp.mpf(q)), seed)
    return beta_upper_root(params, mp.log1p(-mp.mpf(q)), seed)


def beta_isf(params: Params, q: float, seed: float) -> mp.mpf:
    """`I^-1_{1-q}(a, b)`, with the complement taken at oracle precision (see [`normal_isf`])."""
    if q <= 0:
        return mp.mpf(1)
    if q >= 1:
        return mp.mpf(0)
    if mp.mpf(q) <= mp.mpf(0.5):
        return beta_upper_root(params, mp.log(mp.mpf(q)), seed)
    return beta_lower_root(params, mp.log1p(-mp.mpf(q)), seed)


def beta_entropy(params: Params, _x: float) -> mp.mpf:
    """`ln B(a, b) - (a-1) psi(a) - (b-1) psi(b) + (a+b-2) psi(a+b)`."""
    a, b = mp.mpf(params[0]), mp.mpf(params[1])
    return ln_beta_fn(a, b) - (a - 1) * mp.digamma(a) - (b - 1) * mp.digamma(b) + (a + b - 2) * mp.digamma(a + b)


# Binomial oracles.


def binomial_pmf(params: Params, x: float) -> mp.mpf:
    """`C(n, k) p^k (1-p)^(n-k)` on the integer support, `0` off it."""
    n, p = int(params[0]), mp.mpf(params[1])
    if x < 0 or x > n or not x.is_integer():
        return mp.mpf(0)
    k = int(x)
    return mp.binomial(n, k) * p**k * (1 - p) ** (n - k)


def binomial_log_pmf(params: Params, x: float) -> mp.mpf:
    """`ln` of [`binomial_pmf`], taken at oracle precision."""
    return ln(binomial_pmf(params, x))


def binomial_cdf(params: Params, x: float) -> mp.mpf:
    """`P(X <= floor(x)) = I_{1-p}(n - k, k + 1)`."""
    n, p = int(params[0]), mp.mpf(params[1])
    if x < 0:
        return mp.mpf(0)
    k = min(math.floor(x), n)
    return mp.mpf(1) if k >= n else mp.betainc(n - k, k + 1, 0, 1 - p, regularized=True)


def binomial_sf(params: Params, x: float) -> mp.mpf:
    """`P(X > floor(x)) = I_p(k + 1, n - k)`."""
    n, p = int(params[0]), mp.mpf(params[1])
    if x < 0:
        return mp.mpf(1)
    k = min(math.floor(x), n)
    return mp.mpf(0) if k >= n else mp.betainc(k + 1, n - k, 0, p, regularized=True)


def binomial_ppf(params: Params, q: float) -> mp.mpf:
    """Smallest support point with `cdf(k) >= q`, by exact binary search over the oracle cdf."""
    n = int(params[0])
    if q <= 0:
        return mp.mpf(0)
    if q >= 1:
        return mp.mpf(n)
    lo, hi = 0, n
    target = mp.mpf(q)
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if binomial_cdf(params, float(mid)) >= target:
            hi = mid
        else:
            lo = mid + 1
    return mp.mpf(lo)


def binomial_isf(params: Params, q: float) -> mp.mpf:
    """Smallest support point with `sf(k) <= q`, the exact inverse survival function."""
    n = int(params[0])
    if q <= 0:
        return mp.mpf(n)
    if q >= 1:
        return mp.mpf(0)
    lo, hi = 0, n
    target = mp.mpf(q)
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if binomial_sf(params, float(mid)) <= target:
            hi = mid
        else:
            lo = mid + 1
    return mp.mpf(lo)


def binomial_entropy(params: Params, _x: float) -> mp.mpf:
    """`-sum_k pmf(k) ln pmf(k)` over the whole support, the sum statrs evaluates."""
    n = int(params[0])
    total = mp.mpf(0)
    for k in range(n + 1):
        mass = binomial_pmf(params, float(k))
        if mass > 0:
            total -= mass * mp.log(mass)
    return total


# Shared moment oracles, lifted into the standard signature.


def constant(value: Callable[[Params], mp.mpf]) -> Oracle:
    """Lift a parameter-only closed form into the `(params, x)` oracle signature."""

    def oracle(params: Params, _x: float) -> mp.mpf:
        return value(params)

    return oracle


def sqrt_of(oracle: Oracle) -> Oracle:
    """`sqrt` of another oracle, for `std` over `variance`."""

    def wrapped(params: Params, x: float) -> mp.mpf:
        return cast("mp.mpf", mp.sqrt(oracle(params, x)))

    return wrapped


def at_median(oracle: Oracle) -> Oracle:
    """An oracle evaluated at `q = 0.5`, for `median` over `ppf`."""

    def wrapped(params: Params, _x: float) -> mp.mpf:
        return oracle(params, 0.5)

    return wrapped


def at_median_seeded(oracle: SeededOracle) -> SeededOracle:
    """[`at_median`] for an inverse that refines the library's answer (Beta has no closed-form median)."""

    def wrapped(params: Params, _x: float, seed: float) -> mp.mpf:
        return oracle(params, 0.5, seed)

    return wrapped


# Probe generators, one per support shape.

SIGMA_GRID = (0.0, 0.5, 1.0, 2.0, 5.0, 8.3, 15.0, 26.0, 37.5, 40.0, 60.0, 120.0, 300.0)
"""Standardised offsets in units of the scale, straddling every `erfc` / naive-log threshold."""

QUANTILE_GRID = (0.0, 1e-300, 1e-100, 1e-16, 1e-8, 1e-4, 0.001, 0.1, 0.5, 0.9, 0.999, 1 - 1e-8, 1 - 1e-15, 1.0)
"""Quantile danger points, including both closed endpoints."""

SURVIVAL_GRID = (
    1e-300,
    1e-100,
    1e-16,
    1e-9,
    1e-6,
    1e-4,
    0.001,
    0.1,
    0.5,
    0.9,
    0.999,
    1 - 1e-6,
    1 - 1e-9,
)
"""`isf` probes, spanning the same decades as [`QUANTILE_GRID`].

This grid used to stop at `1e-9`, on the reasoning that `1 - q` has no `float64` resolution left
below it, so the audit could not usefully re-measure a documented limitation per point. That floor
was the defect's own shape used as a bound on the instrument: `isf` no longer forms `1 - q` at all,
and had the grid gone deeper it would have shown a limit that grew without end rather than one that
stopped at `1e-8`. **A sweep must not be bounded by what the current implementation is known to be
bad at.** The near-certain end still stops at `1 - 1e-9`, which is a real float64 limit rather than
an algorithmic one: past it the *input* has no resolution left."""


def symmetric_points(centre: float, scale: float, rng: random.Random, count: int) -> list[Point]:
    """Danger and log-uniform probes at `centre +- scale * k`, both tails."""
    points: list[Point] = []
    for k in SIGMA_GRID:
        points.extend(((centre + scale * k, "danger"), (centre - scale * k, "danger")))
    for _ in range(count):
        magnitude = 10.0 ** rng.uniform(-8.0, 2.5)
        points.extend(((centre + scale * magnitude, "random"), (centre - scale * magnitude, "random")))
    return points


def normal_points(params: Params, rng: random.Random, count: int) -> list[Point]:
    """Evaluation points spanning the representable tails of `Normal(mu, sigma)`."""
    return symmetric_points(params[0], params[1], rng, count)


def lognormal_points(params: Params, rng: random.Random, count: int) -> list[Point]:
    """`exp` of the underlying normal's probes, plus the support edges."""
    points: list[Point] = [(0.0, "danger"), (-1.0, "danger")]
    for value, origin in symmetric_points(params[0], params[1], rng, count):
        try:
            x = math.exp(value)
        except OverflowError:
            continue
        if x > 0 and math.isfinite(x):
            points.append((x, origin))
    return points


def exponential_points(params: Params, rng: random.Random, count: int) -> list[Point]:
    """Probes at `t / rate`, so the exponent `rate * x` sweeps the whole `exp` range."""
    rate = params[0]
    grid = (0.0, 1e-300, 1e-16, 1e-8, 0.01, 0.6931471805599453, 1.0, 10.0, 100.0, 700.0, 745.0, 800.0, 1e4)
    points: list[Point] = [(-1.0, "danger")]
    points.extend((t / rate, "danger") for t in grid)
    points.extend((10.0 ** rng.uniform(-16.0, 4.0) / rate, "random") for _ in range(count))
    return points


def uniform_points(params: Params, rng: random.Random, count: int) -> list[Point]:
    """Probes hugging both support edges, where the comparison chains live."""
    lo, hi = params
    width = hi - lo
    points: list[Point] = [(lo - width, "danger"), (lo, "danger"), (hi, "danger"), (hi + width, "danger")]
    for fraction in (1e-16, 1e-8, 1e-4, 0.5):
        points.extend(((lo + width * fraction, "danger"), (hi - width * fraction, "danger")))
    for _ in range(count):
        fraction = 10.0 ** rng.uniform(-16.0, -0.31)
        points.extend(((lo + width * fraction, "random"), (hi - width * fraction, "random")))
        points.append((lo + width * rng.random(), "random"))
    return points


def bernoulli_points(_params: Params, rng: random.Random, count: int) -> list[Point]:
    """The two support points, the gaps between them, and off-support values."""
    points: list[Point] = [(v, "danger") for v in (-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)]
    points.extend((rng.uniform(-1.0, 2.0), "random") for _ in range(count))
    return points


def unit_interval_points(_params: Params, rng: random.Random, count: int) -> list[Point]:
    """Beta evaluation points: both corners to the representable floor, plus the support edges."""
    danger = (-0.5, 0.0, 1e-300, 1e-100, 1e-16, 1e-8, 1e-4, 0.001, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99)
    points: list[Point] = [(v, "danger") for v in danger]
    points.extend(((1 - 1e-8, "danger"), (1 - 1e-15, "danger"), (1.0, "danger"), (1.5, "danger")))
    for _ in range(count):
        points.append((10.0 ** rng.uniform(-300.0, -0.31), "random"))
        points.append((1.0 - 10.0 ** rng.uniform(-15.0, -0.31), "random"))
    return points


def quantile_points(_params: Params, rng: random.Random, count: int) -> list[Point]:
    """`ppf` probes: both closed endpoints, both corners, and a log-uniform spread."""
    points: list[Point] = [(q, "danger") for q in QUANTILE_GRID]
    for _ in range(count):
        points.append((10.0 ** rng.uniform(-300.0, -0.31), "random"))
        points.append((1.0 - 10.0 ** rng.uniform(-15.0, -0.31), "random"))
        points.append((rng.random(), "random"))
    return points


def survival_points(_params: Params, rng: random.Random, count: int) -> list[Point]:
    """`isf` probes: the danger grid, plus log-uniform sampling over both ends.

    The small end spans 300 decades (see [`SURVIVAL_GRID`]); the near-certain end stops where
    `1 - q` stops resolving, because there the *input* has run out, not the algorithm.
    """
    points: list[Point] = [(q, "danger") for q in SURVIVAL_GRID]
    points.extend((10.0 ** rng.uniform(-300.0, -0.31), "random") for _ in range(count))
    points.extend((1.0 - 10.0 ** rng.uniform(-9.0, -0.31), "random") for _ in range(count))
    return points


def discrete_points(params: Params, rng: random.Random, count: int) -> list[Point]:
    """Binomial support probes: both ends, the mode, the step boundaries, and off-support values."""
    n, p = int(params[0]), params[1]
    mode = int(n * p)
    candidates = {-1.0, -0.5, 0.0, 0.5, 1.0, float(mode), float(max(mode - 1, 0)), float(mode + 1), float(n - 1)}
    candidates |= {float(n), float(n + 1), n / 2.0}
    points: list[Point] = [(v, "danger") for v in sorted(candidates)]
    points.extend((float(rng.randint(0, n)), "random") for _ in range(count))
    return points


def discrete_quantile_points(_params: Params, rng: random.Random, count: int) -> list[Point]:
    """Discrete `ppf` probes: the closed endpoints plus a spread over the whole unit interval."""
    points: list[Point] = [(q, "danger") for q in (0.0, 1e-300, 1e-16, 1e-8, 0.001, 0.1, 0.5, 0.9, 0.999, 1.0)]
    points.extend((rng.random(), "random") for _ in range(count))
    return points


# The registry itself.


@dataclass(frozen=True)
class MethodSpec:
    """One audited method: how to probe it, what to compare against, and what it claims."""

    name: str
    oracle: Oracle | SeededOracle
    tolerance: float
    points: Points | None = None
    """`None` marks a moment: no evaluation point, one probe per parameter set."""
    seeded_oracle: bool = False
    """The oracle is a [`SeededOracle`], refining the library's own answer."""
    max_first_param: float = math.inf
    """Skip parameter sets above this, for an oracle that is an `O(n)` sum (Binomial `entropy`)."""


@dataclass(frozen=True)
class DistributionSpec:
    """One audited distribution: its constructor, its parameter sweep, and its methods."""

    name: str
    build: Callable[[Sequence[pl.Expr]], _UnivariateDistribution]
    param_names: tuple[str, ...]
    param_dtypes: tuple[pl.DataType, ...]
    params: tuple[Params, ...]
    methods: tuple[MethodSpec, ...]


def moments(
    mean: Oracle, variance: Oracle, median: Oracle, entropy: Oracle, tolerance: float
) -> tuple[MethodSpec, ...]:
    """The five parameter-only methods every distribution exposes."""
    return (
        MethodSpec("mean", mean, tolerance),
        MethodSpec("variance", variance, tolerance),
        MethodSpec("std", sqrt_of(variance), tolerance),
        MethodSpec("median", median, tolerance),
        MethodSpec("entropy", entropy, tolerance),
    )


def override(specs: tuple[MethodSpec, ...], *replacements: MethodSpec) -> tuple[MethodSpec, ...]:
    """Replace entries of a [`moments`] tuple by name, keeping the order.

    Two distributions need it: Beta's `median` has no closed form (it inherits `ppf(0.5)`, so its
    oracle is the seeded inverse), and Binomial's `entropy` oracle is an `O(n)` support sum.
    """
    by_name = {spec.name: spec for spec in replacements}
    return tuple(by_name.get(spec.name, spec) for spec in specs)


def log_pair(value: Oracle, complement: Oracle) -> Oracle:
    """`ln(value)`, taken **inside** the 50-digit context, via `log1p(-complement)` when near `1`.

    The first half is the whole point of the audit: `ln` of the oracle value, never `ln` of a
    `float64` that has already rounded to zero. The second half is the oracle's own version of the
    same trap, and it bites at 50 digits exactly as it bites at 16: `1 - 2.5e-149` rounds to `1`,
    so a naive `ln` of the linear oracle collapses to `0` in precisely the regime where a library
    that carries the complement is *right*. Without this the audit reports its own precision loss
    as a library defect.
    """

    def wrapped(params: Params, x: float) -> mp.mpf:
        linear = value(params, x)
        if linear > mp.mpf(0.5):
            return mp.log1p(-complement(params, x))
        return ln(linear)

    return wrapped


def bernoulli_log_pmf(params: Params, x: float) -> mp.mpf:
    """`log1p(-p)` at `0`, `ln p` at `1`, `-inf` off the support.

    Spelled through `log1p` rather than `ln(1 - p)` for the same reason as [`log_pair`].
    """
    p = mp.mpf(params[0])
    if x == 0:
        return mp.log1p(-p)
    return ln(p) if x == 1 else mp.ninf


def build_registry() -> tuple[DistributionSpec, ...]:
    """Every (distribution, method) pair the audit covers, with its oracle and claimed tolerance.

    A distribution missing from here is not audited, in exactly the same way a distribution missing
    from `tests/property/_specs.py` is silently untested.
    """
    f64 = (pl.Float64(), pl.Float64())
    return (
        DistributionSpec(
            name="Normal",
            build=lambda p: Normal(mu=p[0], sigma=p[1]),
            param_names=("mu", "sigma"),
            param_dtypes=f64,
            params=((0.0, 1.0), (0.0, 1e-8), (0.0, 1e8), (5.0, 2.0), (-3.0, 0.5), (1e6, 1e3)),
            methods=(
                MethodSpec("pdf", normal_pdf, SPECIAL_RTOL, normal_points),
                MethodSpec("log_pdf", normal_log_pdf, LOG_RTOL, normal_points),
                MethodSpec("cdf", normal_cdf, ERF_RTOL, normal_points),
                MethodSpec("log_cdf", log_pair(normal_cdf, normal_sf), LOG_RTOL, normal_points),
                MethodSpec("sf", normal_sf, ERF_RTOL, normal_points),
                MethodSpec("log_sf", log_pair(normal_sf, normal_cdf), LOG_RTOL, normal_points),
                MethodSpec("ppf", normal_ppf, SPECIAL_RTOL, quantile_points),
                MethodSpec("isf", normal_isf, SPECIAL_RTOL, survival_points),
                *moments(
                    constant(lambda p: mp.mpf(p[0])),
                    constant(lambda p: mp.mpf(p[1]) ** 2),
                    constant(lambda p: mp.mpf(p[0])),
                    normal_entropy,
                    CLOSED_FORM_RTOL,
                ),
            ),
        ),
        DistributionSpec(
            name="LogNormal",
            build=lambda p: LogNormal(mu=p[0], sigma=p[1]),
            param_names=("mu", "sigma"),
            param_dtypes=f64,
            # `(0.0, 1e-4)` is here because the moments cancel there, not because the tails do: the
            # literal `exp(sigma^2) - 1` in `variance` was only good to `1.1e-08` relative at that
            # sigma, and no parameter set in this list was small enough to see it. A sweep that only
            # probes interesting *inputs* still misses defects that need an extreme *parameter*.
            params=((0.0, 1.0), (0.0, 0.1), (0.0, 1e-4), (0.0, 5.0), (0.0, 20.0), (3.0, 2.0), (-5.0, 0.25)),
            methods=(
                MethodSpec("pdf", lognormal_pdf, SPECIAL_RTOL, lognormal_points),
                MethodSpec("log_pdf", lognormal_log_pdf, LOG_RTOL, lognormal_points),
                MethodSpec("cdf", lognormal_cdf, ERF_RTOL, lognormal_points),
                MethodSpec("log_cdf", log_pair(lognormal_cdf, lognormal_sf), LOG_RTOL, lognormal_points),
                MethodSpec("sf", lognormal_sf, ERF_RTOL, lognormal_points),
                MethodSpec("log_sf", log_pair(lognormal_sf, lognormal_cdf), LOG_RTOL, lognormal_points),
                MethodSpec("ppf", lognormal_ppf, SPECIAL_RTOL, quantile_points),
                MethodSpec("isf", lognormal_isf, SPECIAL_RTOL, survival_points),
                *moments(
                    constant(lambda p: mp.e ** (mp.mpf(p[0]) + mp.mpf(p[1]) ** 2 / 2)),
                    constant(
                        lambda p: (mp.e ** mp.mpf(p[1]) ** 2 - 1) * mp.e ** (2 * mp.mpf(p[0]) + mp.mpf(p[1]) ** 2)
                    ),
                    constant(lambda p: mp.e ** mp.mpf(p[0])),
                    constant(lambda p: mp.mpf(p[0]) + mp.log(2 * mp.pi * mp.e * mp.mpf(p[1]) ** 2) / 2),
                    CLOSED_FORM_RTOL,
                ),
            ),
        ),
        DistributionSpec(
            name="Exponential",
            build=lambda p: Exponential(rate=p[0]),
            param_names=("rate",),
            param_dtypes=(pl.Float64(),),
            params=((1.0,), (1e-8,), (1e8,), (0.3,), (100.0,)),
            methods=(
                MethodSpec("pdf", exponential_pdf, CLOSED_FORM_RTOL, exponential_points),
                MethodSpec("log_pdf", exponential_log_pdf, CLOSED_FORM_RTOL, exponential_points),
                MethodSpec("cdf", exponential_cdf, CLOSED_FORM_RTOL, exponential_points),
                MethodSpec("log_cdf", log_pair(exponential_cdf, exponential_sf), LOG_RTOL, exponential_points),
                MethodSpec("sf", exponential_sf, CLOSED_FORM_RTOL, exponential_points),
                MethodSpec("log_sf", exponential_log_sf, CLOSED_FORM_RTOL, exponential_points),
                MethodSpec("ppf", exponential_ppf, CLOSED_FORM_RTOL, quantile_points),
                MethodSpec("isf", exponential_isf, CLOSED_FORM_RTOL, survival_points),
                *moments(
                    constant(lambda p: 1 / mp.mpf(p[0])),
                    constant(lambda p: cast("mp.mpf", 1 / mp.mpf(p[0]) ** 2)),
                    constant(lambda p: mp.log(2) / mp.mpf(p[0])),
                    constant(lambda p: 1 - mp.log(mp.mpf(p[0]))),
                    CLOSED_FORM_RTOL,
                ),
            ),
        ),
        DistributionSpec(
            name="Uniform",
            build=lambda p: Uniform(min=p[0], max=p[1]),
            param_names=("lo", "hi"),
            param_dtypes=f64,
            # `(-1.0, 0.0)` and `(-1e10, 1.0)` are here for `isf`, and only for `isf`: they are the
            # spans where the upper end of the answer sits at or near *zero*, so an absolutely
            # quantised quantile becomes a relatively wrong result. `Uniform(-1, 0).isf(1e-17)`
            # returned `0.0` against a true `-1e-17` and no span in the original list could see it.
            params=(
                (0.0, 1.0),
                (-1e8, 1e8),
                (0.0, 1e-8),
                (-3.0, 7.0),
                (1e6, 1e6 + 1.0),
                (-1.0, 0.0),
                (-1e10, 1.0),
            ),
            methods=(
                MethodSpec("pdf", uniform_pdf, CLOSED_FORM_RTOL, uniform_points),
                MethodSpec("log_pdf", uniform_log_pdf, CLOSED_FORM_RTOL, uniform_points),
                MethodSpec("cdf", uniform_cdf, CLOSED_FORM_RTOL, uniform_points),
                MethodSpec("log_cdf", log_pair(uniform_cdf, uniform_sf), LOG_RTOL, uniform_points),
                MethodSpec("sf", uniform_sf, CLOSED_FORM_RTOL, uniform_points),
                MethodSpec("log_sf", log_pair(uniform_sf, uniform_cdf), LOG_RTOL, uniform_points),
                MethodSpec("ppf", uniform_ppf, CLOSED_FORM_RTOL, quantile_points),
                MethodSpec("isf", uniform_isf, CLOSED_FORM_RTOL, survival_points),
                *moments(
                    constant(lambda p: (mp.mpf(p[0]) + mp.mpf(p[1])) / 2),
                    constant(lambda p: (mp.mpf(p[1]) - mp.mpf(p[0])) ** 2 / 12),
                    constant(lambda p: (mp.mpf(p[0]) + mp.mpf(p[1])) / 2),
                    constant(lambda p: mp.log(mp.mpf(p[1]) - mp.mpf(p[0]))),
                    CLOSED_FORM_RTOL,
                ),
            ),
        ),
        DistributionSpec(
            name="Bernoulli",
            build=lambda p: Bernoulli(p=p[0]),
            param_names=("p",),
            param_dtypes=(pl.Float64(),),
            # `1e-17` sits in the one gap the rest of this list leaves: small enough that `1 - p`
            # rounds to exactly `1.0` (which `1e-16` does not), yet large enough that the quantile
            # grid can get *below* it (which `1e-300` does not). Both conditions are needed to see
            # the `isf` / `ppf(1)` defect, which is why a list spanning 300 decades still missed it.
            params=(
                (0.5,),
                (0.001,),
                (0.999,),
                (1e-16,),
                (1e-17,),
                (1 - 1e-16,),
                (1e-300,),
                (0.0,),
                (1.0,),
            ),
            methods=(
                MethodSpec("pmf", bernoulli_pmf, CLOSED_FORM_RTOL, bernoulli_points),
                MethodSpec("log_pmf", bernoulli_log_pmf, LOG_RTOL, bernoulli_points),
                MethodSpec("cdf", bernoulli_cdf, CLOSED_FORM_RTOL, bernoulli_points),
                MethodSpec("log_cdf", log_pair(bernoulli_cdf, bernoulli_sf), LOG_RTOL, bernoulli_points),
                MethodSpec("sf", bernoulli_sf, CLOSED_FORM_RTOL, bernoulli_points),
                MethodSpec("log_sf", log_pair(bernoulli_sf, bernoulli_cdf), LOG_RTOL, bernoulli_points),
                MethodSpec("ppf", bernoulli_ppf, CLOSED_FORM_RTOL, quantile_points),
                MethodSpec("isf", bernoulli_isf, CLOSED_FORM_RTOL, survival_points),
                *moments(
                    constant(lambda p: mp.mpf(p[0])),
                    constant(lambda p: mp.mpf(p[0]) * (1 - mp.mpf(p[0]))),
                    at_median(bernoulli_ppf),
                    bernoulli_entropy,
                    CLOSED_FORM_RTOL,
                ),
            ),
        ),
        DistributionSpec(
            name="Beta",
            build=lambda p: Beta(a=p[0], b=p[1]),
            param_names=("a", "b"),
            param_dtypes=f64,
            params=(
                (2.0, 3.0),
                (200.0, 2.0),
                (2.0, 200.0),
                (80.0, 90.0),
                (0.05, 0.05),
                (0.5, 0.5),
                (1.0, 1.0),
                (1000.0, 1000.0),
                (0.1, 500.0),
            ),
            methods=(
                MethodSpec("pdf", beta_pdf, SPECIAL_RTOL, unit_interval_points),
                MethodSpec("log_pdf", beta_log_pdf, LOG_RTOL, unit_interval_points),
                MethodSpec("cdf", beta_cdf, SPECIAL_RTOL, unit_interval_points),
                MethodSpec("log_cdf", log_pair(beta_cdf, beta_sf), LOG_RTOL, unit_interval_points),
                MethodSpec("sf", beta_sf, SPECIAL_RTOL, unit_interval_points),
                MethodSpec("log_sf", log_pair(beta_sf, beta_cdf), LOG_RTOL, unit_interval_points),
                MethodSpec("ppf", beta_ppf, SPECIAL_RTOL, quantile_points, seeded_oracle=True),
                MethodSpec("isf", beta_isf, SPECIAL_RTOL, survival_points, seeded_oracle=True),
                *override(
                    moments(
                        constant(lambda p: mp.mpf(p[0]) / (mp.mpf(p[0]) + mp.mpf(p[1]))),
                        constant(
                            lambda p: (
                                mp.mpf(p[0])
                                * mp.mpf(p[1])
                                / ((mp.mpf(p[0]) + mp.mpf(p[1])) ** 2 * (mp.mpf(p[0]) + mp.mpf(p[1]) + 1))
                            )
                        ),
                        constant(lambda _p: mp.mpf(0)),
                        beta_entropy,
                        SPECIAL_RTOL,
                    ),
                    MethodSpec("median", at_median_seeded(beta_ppf), SPECIAL_RTOL, seeded_oracle=True),
                ),
            ),
        ),
        DistributionSpec(
            name="Binomial",
            build=lambda p: Binomial(n=p[0], p=p[1]),
            param_names=("n", "p"),
            param_dtypes=(pl.Int64(), pl.Float64()),
            # `n` tops out at 5000 because the *oracle* does: `betainc` at balanced huge shapes is
            # where mpmath's hypergeometric evaluation blows up (0.04 s at `n = 5000`, 2.9 s at
            # 10000, 10 s at 20000, no return in useful time at 100000), and an audit that cannot
            # finish is an audit nobody runs. Every qualitative regime is still covered: tiny `p`,
            # `p` near 1, both degenerate endpoints, and a large `n` with a small `p`.
            params=(
                (10.0, 0.5),
                (1000.0, 0.5),
                (5000.0, 0.001),
                (50.0, 1e-8),
                (1000.0, 0.999),
                (10.0, 0.0),
                (10.0, 1.0),
                (5000.0, 0.5),
            ),
            methods=(
                MethodSpec("pmf", binomial_pmf, SPECIAL_RTOL, discrete_points),
                MethodSpec("log_pmf", binomial_log_pmf, DISCRETE_LOG_RTOL, discrete_points),
                MethodSpec("cdf", binomial_cdf, SPECIAL_RTOL, discrete_points),
                MethodSpec("log_cdf", log_pair(binomial_cdf, binomial_sf), LOG_RTOL, discrete_points),
                MethodSpec("sf", binomial_sf, SPECIAL_RTOL, discrete_points),
                MethodSpec("log_sf", log_pair(binomial_sf, binomial_cdf), LOG_RTOL, discrete_points),
                MethodSpec("ppf", binomial_ppf, DISCRETE_PPF_RTOL, discrete_quantile_points),
                MethodSpec("isf", binomial_isf, DISCRETE_PPF_RTOL, survival_points),
                *override(
                    moments(
                        constant(lambda p: mp.mpf(p[0]) * mp.mpf(p[1])),
                        constant(lambda p: mp.mpf(p[0]) * mp.mpf(p[1]) * (1 - mp.mpf(p[1]))),
                        at_median(binomial_ppf),
                        binomial_entropy,
                        SPECIAL_RTOL,
                    ),
                    MethodSpec("entropy", binomial_entropy, DISCRETE_LOG_RTOL, max_first_param=1000.0),
                ),
            ),
        ),
    )


# Evaluation and classification.


@dataclass(frozen=True)
class Probe:
    """One evaluated (distribution, method, parameters, point) triple and its verdict."""

    distribution: str
    method: str
    params: Params
    x: float | None
    origin: str
    got: Observation
    expected: str
    category: Category
    rel_error: float
    excused: bool = False
    """`True` where the `OK` was earned by an absolute check, not by `rel_error`.

    Only ever set on an `OK` probe whose relative error is *worse* than the method's tolerance: a
    subnormal oracle, or an exactly-zero one. Those rows are correctly rounded and their relative
    number is meaningless, so [`summarise`] must not let them set the worst-error column.
    """

    def row(self) -> str:
        """One markdown table row."""
        point = "-" if self.x is None else f"`{self.x!r}`"
        error = "-" if math.isnan(self.rel_error) else f"{self.rel_error:.3g}"
        return (
            f"| `{self.category}` | `{self.distribution}{self.params}` | `{self.method}` | {point} "
            f"| {self.got.display()} | `{self.expected}` | {error} | {self.origin} |"
        )


def classify(got: float | None, expected: mp.mpf, tolerance: float) -> tuple[Category, float, bool]:  # noqa: C901, PLR0911
    """Bucket one probe; return its category, relative error, and whether it was excused absolutely.

    The order of the checks is the point: saturation is separated from mere inaccuracy, and both
    are separated from a value that is correctly saturated because the oracle itself has no
    `float64` representation.

    The third element is `True` only where an `OK` was earned by an absolute check *after* the
    relative one had already failed, so `rel_error` on that row describes nothing. See
    [`Probe.excused`].
    """
    if mp.isnan(expected):
        return "OK", math.nan, False
    if got is None or math.isnan(got):
        return "NAN", math.inf, False
    if math.isinf(got):
        if got > 0 and expected > LARGEST_FINITE:
            return "OK", 0.0, False
        if got < 0 and expected < -LARGEST_FINITE:
            return "OK", 0.0, False
        return ("UNDERFLOW" if got < 0 else "DEGRADED"), math.inf, False
    if mp.isinf(expected):
        return "DEGRADED", math.inf, False
    if got == 0.0:
        if expected == 0:
            return "OK", 0.0, False
        return ("OK", 0.0, False) if abs(expected) < SMALLEST_SUBNORMAL else ("UNDERFLOW", math.inf, False)
    if expected == 0:
        # An exactly-zero oracle admits no relative tolerance; every quantity audited here (a log,
        # an entropy, a probability) has a natural scale of 1, so the tolerance reads as absolute.
        within = abs(got) <= tolerance
        return ("OK" if within else "DEGRADED"), math.inf, within
    if (got > 0) != (expected > 0):
        return "SIGN", math.inf, False
    absolute = abs(mp.mpf(got) - expected)
    ratio = absolute / abs(expected)
    rel = float(ratio) if ratio < LARGEST_FINITE else math.inf
    if rel <= tolerance:
        return "OK", rel, False
    if abs(expected) < SMALLEST_NORMAL:
        # Gradual underflow: a subnormal has only the one spacing `2**-1074` left, so past the
        # relative check the correctly-rounded answer must still be allowed through absolutely.
        rounded = absolute <= SUBNORMAL_ULPS * SMALLEST_SUBNORMAL
        return ("OK" if rounded else "DEGRADED"), rel, rounded
    return "DEGRADED", rel, False


@dataclass(frozen=True)
class Observation:
    """What the library produced for one probe: a value, a null, or the error that aborted it."""

    value: float | None = None
    error: str | None = None

    def display(self) -> str:
        """The `Got` column of the report."""
        return f"**{self.error}**" if self.error is not None else repr(self.value)


def run_query(spec: DistributionSpec, method: MethodSpec, params: Params, xs: Sequence[float]) -> list[float | None]:
    """One polars query for the whole probe batch, with column-valued parameters.

    Column-valued rather than scalar: it is one query per (method, parameter set), and the two
    paths are bit-identical by construction (they share the per-method body), which
    `value_keyed_test.py::test_value_keyed_scalar_fast_path_matches_per_row` pins.
    """
    rows = max(len(xs), 1)
    columns = {
        name: pl.Series(name, [int(value) if dtype.is_integer() else value] * rows, dtype=dtype)
        for name, value, dtype in zip(spec.param_names, params, spec.param_dtypes, strict=True)
    }
    dist = spec.build([pl.col(name) for name in spec.param_names])
    if method.points is None:
        return pl.DataFrame(columns).select(result=getattr(dist, method.name)())["result"].to_list()
    frame = pl.DataFrame({**columns, "x": pl.Series("x", list(xs), dtype=pl.Float64())})
    return frame.select(result=getattr(dist, method.name)(pl.col("x")))["result"].to_list()


def evaluate(spec: DistributionSpec, method: MethodSpec, params: Params, xs: Sequence[float]) -> list[Observation]:
    """Run one method over `xs`, isolating a probe that aborts the query onto its own row.

    A `statrs` panic surfaces as an opaque `ComputeError` that kills the *whole batch*, so a single
    bad probe would otherwise cost the audit every other probe in its group and read as a crash of
    the instrument rather than a finding about the library. On failure the batch is retried one row
    at a time, which is slow and only ever runs on the error path.
    """
    try:
        return [Observation(value) for value in run_query(spec, method, params, xs)]
    except Exception as batch_error:  # noqa: BLE001
        if method.points is None:
            return [Observation(error=summarise_error(batch_error))]
    observations: list[Observation] = []
    for x in xs:
        try:
            observations.append(Observation(run_query(spec, method, params, [x])[0]))
        except Exception as row_error:  # noqa: BLE001, PERF203
            observations.append(Observation(error=summarise_error(row_error)))
    return observations


def summarise_error(error: Exception) -> str:
    """The first line of an exception, which is where the plugin panic message lands."""
    first = str(error).strip().splitlines()
    return f"{type(error).__name__}: {first[0]}" if first else type(error).__name__


def reference_value(method: MethodSpec, params: Params, x: float, got: float | None) -> mp.mpf:
    """The oracle value for one probe, feeding a seeded oracle the library's own answer.

    Coerced through [`real`]: `mpmath.betainc` returns an `mpc` with a negligible imaginary residue
    at some parameterisations, and a complex reference must become `ORACLE_UNAVAILABLE` rather than
    crash the sweep or silently compare as something else.
    """
    if not method.seeded_oracle:
        return real(method.oracle(params, x))  # type: ignore[call-arg]
    return real(method.oracle(params, x, math.nan if got is None else got))  # type: ignore[call-arg]


def audit_method(spec: DistributionSpec, method: MethodSpec, rng: random.Random, samples: int) -> list[Probe]:
    """Probe one method across the whole parameter sweep."""
    probes: list[Probe] = []
    for params in spec.params:
        if params[0] > method.max_first_param:
            continue
        moment = method.points is None
        points = [(math.nan, "moment")] if moment else method.points(params, rng, samples)  # type: ignore[misc]
        observed = evaluate(spec, method, params, [] if moment else [x for x, _ in points])
        for (x, origin), got in zip(points, observed, strict=True):
            point = None if moment else x
            if got.error is not None:
                probes.append(Probe(spec.name, method.name, params, point, origin, got, "-", "PANIC", math.nan))
                continue
            try:
                expected = reference_value(method, params, x, got.value)
            except OracleUnavailableError as exc:
                probes.append(
                    Probe(spec.name, method.name, params, point, origin, got, str(exc), "ORACLE_UNAVAILABLE", math.nan)
                )
                continue
            category, rel, excused = classify(got.value, expected, method.tolerance)
            probes.append(
                Probe(spec.name, method.name, params, point, origin, got, mp.nstr(expected, 17), category, rel, excused)
            )
    return probes


# Instrument controls.


@dataclass
class ControlResult:
    """One instrument self-test and what it observed."""

    name: str
    expected_category: str
    observed: Category
    detail: str
    passed: bool = field(init=False)

    def __post_init__(self) -> None:
        """Derive `passed` from the observed category, so a caller cannot set the two out of step."""
        self.passed = self.observed == self.expected_category


def run_controls() -> list[ControlResult]:
    """Calibrate the instrument before trusting a single clean row from it.

    **Positive control**: at `Beta(200, 2)`, `x = 0.001` the oracle is a large finite negative and
    `log_cdf` saturated to `-inf` before Port B, so the instrument must call a `-inf` there
    `UNDERFLOW`. It is checked against a deliberately saturated value rather than the live library
    output, so the control keeps calibrating the tool *after* the library is fixed at exactly this
    point; the live category, reported alongside, is what flips from `UNDERFLOW` to `OK`.

    **Negative control**: `Normal.log_sf` at 40 sigma. That path was hardened with `ln_erfc`, so a
    flag there means the oracle or the comparison is wrong, not the library.
    """
    beta_params, beta_x = (200.0, 2.0), 0.001
    beta_oracle = ln(beta_cdf(beta_params, beta_x))
    saturated, _, _ = classify(-math.inf, beta_oracle, LOG_RTOL)
    beta_live = pl.DataFrame({"x": [beta_x]}).select(r=Beta(a=200.0, b=2.0).log_cdf(pl.col("x")))["r"].item()
    live_category, _, _ = classify(beta_live, beta_oracle, LOG_RTOL)

    normal_params, normal_x = (0.0, 1.0), 40.0
    normal_got = pl.DataFrame({"x": [normal_x]}).select(r=Normal(mu=0.0, sigma=1.0).log_sf(pl.col("x")))["r"].item()
    normal_oracle = ln(normal_sf(normal_params, normal_x))
    normal_category, normal_rel, _ = classify(normal_got, normal_oracle, LOG_RTOL)

    return [
        ControlResult(
            "positive: a `-inf` at Beta(200, 2).log_cdf(0.001) must read as a defect",
            "UNDERFLOW",
            saturated,
            f"oracle {mp.nstr(beta_oracle, 17)}; live library value {beta_live!r} reads `{live_category}`",
        ),
        ControlResult(
            "negative: Normal(0, 1).log_sf(40) was hardened with ln_erfc",
            "OK",
            normal_category,
            f"got {normal_got!r}, oracle {mp.nstr(normal_oracle, 17)}, rel {normal_rel:.3g}",
        ),
    ]


# Reporting.


def worst_relative(probes: Iterable[Probe]) -> str:
    """The worst finite relative error in a group, or `-` if the group has none."""
    finite = [p.rel_error for p in probes if math.isfinite(p.rel_error)]
    return f"{max(finite):.3g}" if finite else "-"


def summarise(probes: Sequence[Probe]) -> list[str]:
    """The per-(distribution, method) summary table: category counts and worst relative error.

    The worst-error column counts only probes the *relative* check judged. A correctly-rounded
    subnormal keeps a huge relative number (a subnormal has one or two significant digits left), so
    letting it into this column reports `Exponential | log_cdf | 0.751` beside zero defect rows, and
    a reader scanning for trouble finds it in a method that is exact. Those probes get their own
    column instead, where the number means "excused, and here is how far off it looks".
    """
    lines = [
        "| Distribution | Method | Probes | Worst rel. error | Excused (worst) | " + " | ".join(CATEGORIES),
        "| --- | --- | --- | --- | --- | " + " | ".join("---" for _ in CATEGORIES),
    ]
    keys = dict.fromkeys((p.distribution, p.method) for p in probes)
    for distribution, method in keys:
        group = [p for p in probes if p.distribution == distribution and p.method == method]
        excused = [p for p in group if p.excused]
        judged = worst_relative(p for p in group if not p.excused)
        absolved = f"{len(excused)} ({worst_relative(excused)})" if excused else "-"
        counts = [str(sum(p.category == c for p in group)) for c in CATEGORIES]
        lines.append(f"| {distribution} | `{method}` | {len(group)} | {judged} | {absolved} | " + " | ".join(counts))
    return lines


def worst_per_group(probes: Sequence[Probe], limit: int) -> list[Probe]:
    """The `limit` worst defect rows per (distribution, method, category), so the report stays readable."""
    buckets: dict[tuple[str, str, str], list[Probe]] = {}
    for probe in probes:
        if probe.category != "OK":
            buckets.setdefault((probe.distribution, probe.method, probe.category), []).append(probe)
    selected: list[Probe] = []
    for group in buckets.values():
        group.sort(key=lambda p: (-p.rel_error if math.isfinite(p.rel_error) else -math.inf, str(p.x)))
        selected.extend(group[:limit])
    return selected


@dataclass(frozen=True)
class Run:
    """One audit invocation: what it covered, and what it deliberately did not."""

    probes: list[Probe]
    controls: list[ControlResult]
    skipped: dict[str, str]
    seed: int
    samples: int


def write_report(path: Path, run: Run) -> None:
    """Write the findings report: the controls, what was skipped, the summary, then every defect row."""
    probes = run.probes
    defects = [p for p in probes if p.category != "OK"]
    lines = [
        "# Accuracy audit findings",
        "",
        f"Generated by `make audit` (`tools/accuracy_audit.py`), oracle `mpmath` at {DPS} digits, "
        f"seed `{run.seed}`, `{run.samples}` random probes per (method, parameter set).",
        "",
        f"**{len(probes)} probes, {len(defects)} non-`OK`.**",
        "",
        "## Instrument controls",
        "",
        "| Control | Expected | Observed | Detail |",
        "| --- | --- | --- | --- |",
    ]
    lines.extend(f"| {c.name} | `{c.expected_category}` | `{c.observed}` | {c.detail} |" for c in run.controls)
    if run.skipped:
        lines.extend(["", "## Skipped, with reason", "", "| Method | Reason |", "| --- | --- |"])
        lines.extend(f"| `{name}` | {reason} |" for name, reason in sorted(run.skipped.items()))
    lines.extend(["", "## Summary", "", *summarise(probes), "", "## Defect rows", ""])
    if not defects:
        lines.append("None.")
    else:
        lines.extend(
            [
                "Worst 5 rows per (distribution, method, category).",
                "",
                "| Category | Distribution | Method | Point | Got | Oracle | Rel. error | Origin |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        lines.extend(p.row() for p in worst_per_group(defects, limit=5))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


KNOWN_HANGS: dict[str, str] = {}
"""Probes the sweep cannot run because the *library* does not return, keyed `<Distribution>.<method>`.

Empty, and it should stay that way. It held `Beta.ppf` and `Beta.isf` while `statrs`'
`ContinuousCDF::inverse_cdf` failed to terminate for a deep-tail quantile (>15 s for a *single* row
at `(a, b) = (200, 2)`, `q` in ~`[1e-150, 1e-60]`); `beta.rs::inverse_cdf` replaced it and the
entries came out. Anything listed here is recorded in the report with its reason rather than dropped,
because a silent cap reads as "covered everything". `--skip` overrides the default at the CLI.
"""


def main() -> int:
    """Run the audit and write the report.

    Exits non-zero if a control misbehaved, if the sweep produced no probes at all, or if any probe
    came back non-`OK`. A gate that always exits `0` is not a gate: the README accuracy notes claim
    this sweep is clean, and that claim is only worth something if a defective run can fail a job.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("distributions", nargs="*", help="restrict the sweep (default: all)")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES, help="random probes per method and params")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed, so a finding can be re-probed")
    parser.add_argument("--output", type=Path, default=Path("audit-findings.md"), help="report path")
    parser.add_argument("--no-controls", action="store_true", help="skip the instrument self-tests")
    parser.add_argument("--skip", nargs="*", default=None, help="`Dist.method` pairs to skip, replacing the defaults")
    args = parser.parse_args()

    controls: list[ControlResult] = []
    if not args.no_controls:
        controls = run_controls()
        for control in controls:
            status = "PASS" if control.passed else "FAIL"
            print(f"[{status}] control {control.name}: {control.observed} ({control.detail})")
        if not all(c.passed for c in controls):
            print("\nThe instrument is not calibrated; every clean result below would be worthless.")
            return 1

    registry = build_registry()
    if args.distributions:
        wanted = {name.lower() for name in args.distributions}
        registry = tuple(spec for spec in registry if spec.name.lower() in wanted)
        if not registry:
            print(f"no distribution matched {sorted(wanted)}")
            return 1

    skipped = KNOWN_HANGS if args.skip is None else dict.fromkeys(args.skip, "requested on the command line")
    rng = random.Random(args.seed)
    probes: list[Probe] = []
    for spec in registry:
        for method in spec.methods:
            key = f"{spec.name}.{method.name}"
            if key in skipped:
                print(f"{spec.name:<12} {method.name:<9} SKIPPED: {skipped[key]}")
                continue
            found = audit_method(spec, method, rng, args.samples)
            probes.extend(found)
            defects = sum(p.category != "OK" for p in found)
            flag = f"  <-- {defects} defect rows" if defects else ""
            print(f"{spec.name:<12} {method.name:<9} {len(found):>5} probes{flag}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    covered = {f"{spec.name}.{method.name}" for spec in registry for method in spec.methods}
    write_report(
        args.output, Run(probes, controls, {k: v for k, v in skipped.items() if k in covered}, args.seed, args.samples)
    )
    total = sum(p.category != "OK" for p in probes)
    print(f"\n{len(probes)} probes, {total} non-OK. Report written to {args.output}")
    if not probes:
        print("no probes ran, which is a broken sweep and not a clean one")
        return 1
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
