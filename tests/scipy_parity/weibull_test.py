from __future__ import annotations

import pytest
from scipy.stats import weibull_min as scipy_weibull_min

from polars_stats import Weibull
from tests.scipy_parity._harness import Case, assert_case_matches_scipy

# `shape` spans the three density regimes: diverging at `0` below `1`, the exponential at `1`, and a
# mode inside the support above it.
_PARAMS = [(1.0, 1.0), (0.5, 2.0), (1.5, 0.5), (3.0, 1.0), (10.0, 100.0)]
# Endpoints excluded: `ppf(0) = 0` and `ppf(1) = +inf` are asserted in `inverse_test.py`.
_QUANTILES = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]

# The value-keyed methods, `median` and `entropy` are elementary closed forms and hold the harness
# default `1e-12`. `mean`, `variance` and `std` go through the gamma function on both sides, which
# statrs' Lanczos approximation carries to ~`5e-15` relative; `1e-10` is the special-function claim.
_GAMMA_TOL = 1e-10
_CASES: list[Case[Weibull]] = [
    Case("pdf", "value", lambda w, x: w.pdf(x), "pdf"),
    Case("log_pdf", "value", lambda w, x: w.log_pdf(x), "logpdf"),
    Case("cdf", "value", lambda w, x: w.cdf(x), "cdf"),
    Case("log_cdf", "value", lambda w, x: w.log_cdf(x), "logcdf"),
    Case("sf", "value", lambda w, x: w.sf(x), "sf"),
    Case("log_sf", "value", lambda w, x: w.log_sf(x), "logsf"),
    Case("ppf", "quantile", lambda w, x: w.ppf(x), "ppf"),
    Case("isf", "quantile", lambda w, x: w.isf(x), "isf"),
    Case("mean", "scalar", lambda w, _: w.mean(), "mean", tol=_GAMMA_TOL),
    Case("variance", "scalar", lambda w, _: w.variance(), "var", tol=_GAMMA_TOL),
    Case("std", "scalar", lambda w, _: w.std(), "std", tol=_GAMMA_TOL),
    Case("median", "scalar", lambda w, _: w.median(), "median"),
    Case("entropy", "scalar", lambda w, _: w.entropy(), "entropy"),
]


_POWERS = (0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 20.0)
"""`(x / scale) ** shape`, from `cdf ~ 0.01` out to `sf = e ** -20`."""


def _value_grid(shape: float, scale: float) -> list[float]:
    """Below the support, then points at fixed powers so every method stays `O(10)` at every shape.

    A grid in multiples of `scale` would put `log_sf` at `-(5 ** 10)` for `shape = 10`, where the
    harness's absolute `1e-12` sits below one ulp. `0` itself is `support_test.py`'s.
    """
    return [-0.5 * scale, *(scale * t ** (1.0 / shape) for t in _POWERS)]


@pytest.mark.parametrize(("shape", "scale"), _PARAMS, ids=[f"shape={shape},scale={scale}" for shape, scale in _PARAMS])
@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.name)
def test_method_matches_scipy(case: Case[Weibull], shape: float, scale: float) -> None:
    """Every method matches `scipy.stats.weibull_min(c=shape, scale=scale)` across the parameter grid.

    Method-specific behaviour (null propagation, out-of-range quantiles, the support edge) is asserted
    in the shared contract files.
    """
    assert_case_matches_scipy(
        case,
        dist=Weibull(shape=shape, scale=scale),
        scipy_frozen=scipy_weibull_min(c=shape, scale=scale),
        value_grid=_value_grid(shape, scale),
        quantiles=_QUANTILES,
    )
