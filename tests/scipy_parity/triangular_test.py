from __future__ import annotations

import pytest
from scipy.stats import triang as scipy_triang

from polars_stats import Triangular
from tests.scipy_parity._harness import Case, assert_case_matches_scipy

# Parameter and evaluation grids for the parity sweep. Owned by this test category and independent. `(1, 1, 4)` and
# `(1, 4, 4)` put the mode on each end, where the density at that end is its one-sided limit `2 / (max - min)`.
_PARAMS = [(0.0, 0.5, 1.0), (-2.0, 0.0, 3.0), (1.0, 1.0, 4.0), (1.0, 4.0, 4.0), (-5.0, -4.9, -1.0), (10.0, 17.0, 20.0)]
_QUANTILES = [0.0, 1e-12, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1 - 1e-12, 1.0]
# scipy's `isf` is its `ppf(1 - q)`, so at `q = 1e-12` the complement's rounding moves its answer by `7.8e-12` on
# `(0, 0.5, 1)`; this library solves on `q` itself, and `precision_test.py` pins that tail against the exact form.
_ISF_QUANTILES = [q for q in _QUANTILES if q != 1e-12]  # noqa: PLR2004

# Every method is an elementary closed form (arithmetic, `sqrt`, `log`, `log1p`), so it holds the harness default
# `1e-12`.
_CASES: list[Case[Triangular]] = [
    Case("pdf", "value", lambda d, x: d.pdf(x), "pdf"),
    Case("log_pdf", "value", lambda d, x: d.log_pdf(x), "logpdf"),
    Case("cdf", "value", lambda d, x: d.cdf(x), "cdf"),
    Case("log_cdf", "value", lambda d, x: d.log_cdf(x), "logcdf"),
    Case("sf", "value", lambda d, x: d.sf(x), "sf"),
    Case("log_sf", "value", lambda d, x: d.log_sf(x), "logsf"),
    Case("ppf", "quantile", lambda d, x: d.ppf(x), "ppf"),
    Case("isf", "quantile", lambda d, x: d.isf(x), "isf"),
    Case("mean", "scalar", lambda d, _: d.mean(), "mean"),
    Case("variance", "scalar", lambda d, _: d.variance(), "var"),
    Case("std", "scalar", lambda d, _: d.std(), "std"),
    Case("median", "scalar", lambda d, _: d.median(), "median"),
    Case("entropy", "scalar", lambda d, _: d.entropy(), "entropy"),
]


def _value_grid(lo: float, mode: float, hi: float) -> list[float]:
    """Below the support, both ends, both pieces, the mode, and above."""
    width = hi - lo
    return [
        lo - width,
        lo,
        lo + 0.1 * width,
        (lo + mode) / 2,
        mode,
        (mode + hi) / 2,
        hi - 0.1 * width,
        hi,
        hi + width,
    ]


@pytest.mark.parametrize(
    ("lo", "mode", "hi"), _PARAMS, ids=[f"min={lo},mode={mode},max={hi}" for lo, mode, hi in _PARAMS]
)
@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.name)
def test_method_matches_scipy(case: Case[Triangular], lo: float, mode: float, hi: float) -> None:
    """Every method matches `scipy.stats.triang(c=(mode - min) / (max - min), loc=min, scale=max - min)`.

    Method-specific behaviour (null propagation, out-of-range quantiles, support clamping) is asserted in the shared
    contract files.
    """
    width = hi - lo
    assert_case_matches_scipy(
        case,
        dist=Triangular(min=lo, mode=mode, max=hi),
        scipy_frozen=scipy_triang(c=(mode - lo) / width, loc=lo, scale=width),
        value_grid=_value_grid(lo, mode, hi),
        quantiles=_ISF_QUANTILES if case.name == "isf" else _QUANTILES,
    )
