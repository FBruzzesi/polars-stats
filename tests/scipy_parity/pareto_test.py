from __future__ import annotations

import pytest
from scipy.stats import pareto as scipy_pareto

from polars_stats import Pareto
from tests.scipy_parity._harness import Case, assert_case_matches_scipy

# `shape` crosses both divergence thresholds, so the `+inf` moments are compared against scipy's own.
_PARAMS = [(1.0, 1.0), (1.5, 3.0), (0.5, 0.75), (2.0, 2.0), (100.0, 10.0)]
# Endpoints excluded: `ppf(0) = scale` and `ppf(1) = +inf` are asserted in `inverse_test.py`.
_QUANTILES = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]

# Elementary closed forms, so every case holds the harness default `1e-12`.
_CASES: list[Case[Pareto]] = [
    Case("pdf", "value", lambda p, x: p.pdf(x), "pdf"),
    Case("log_pdf", "value", lambda p, x: p.log_pdf(x), "logpdf"),
    Case("cdf", "value", lambda p, x: p.cdf(x), "cdf"),
    Case("log_cdf", "value", lambda p, x: p.log_cdf(x), "logcdf"),
    Case("sf", "value", lambda p, x: p.sf(x), "sf"),
    Case("log_sf", "value", lambda p, x: p.log_sf(x), "logsf"),
    Case("ppf", "quantile", lambda p, x: p.ppf(x), "ppf"),
    Case("isf", "quantile", lambda p, x: p.isf(x), "isf"),
    Case("mean", "scalar", lambda p, _: p.mean(), "mean"),
    Case("variance", "scalar", lambda p, _: p.variance(), "var"),
    Case("std", "scalar", lambda p, _: p.std(), "std"),
    Case("median", "scalar", lambda p, _: p.median(), "median"),
    Case("entropy", "scalar", lambda p, _: p.entropy(), "entropy"),
]


def _value_grid(scale: float) -> list[float]:
    """Below the support, on its edge, and out to `20 scale`.

    The nearest point above `scale` stays at `1.001 scale`: closer in, scipy's `1 - x ** -b` has lost the
    digits `precision_test.py` pins against the closed form.
    """
    return [scale * k for k in (0.5, 1.0, 1.001, 1.1, 1.5, 2.0, 5.0, 20.0)]


@pytest.mark.parametrize(("scale", "shape"), _PARAMS, ids=[f"scale={scale},shape={shape}" for scale, shape in _PARAMS])
@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.name)
def test_method_matches_scipy(case: Case[Pareto], scale: float, shape: float) -> None:
    """Every closed-form method matches `scipy.stats.pareto(b=shape, scale=scale)` across the parameter grid.

    Method-specific behaviour (null propagation, out-of-range quantiles, the support edge) is asserted
    in the shared contract files.
    """
    assert_case_matches_scipy(
        case,
        dist=Pareto(scale=scale, shape=shape),
        scipy_frozen=scipy_pareto(b=shape, scale=scale),
        value_grid=_value_grid(scale),
        quantiles=_QUANTILES,
    )
