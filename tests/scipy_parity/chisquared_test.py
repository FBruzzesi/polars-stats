from __future__ import annotations

from typing import Literal

import polars as pl
import pytest
from scipy.stats import chi2 as scipy_chi2

from polars_stats import ChiSquared
from tests._polars_compat import assert_series_equal
from tests.scipy_parity._harness import Case, assert_case_matches_scipy

# `dof` spans the density regimes (diverging at `0` below `2`, the exponential at `2`, an interior mode above it) and
# both sides of the incomplete gamma's Stirling switch at shape `10` (`dof = 20`) and its Temme switch at `100`. It
# starts at `1`: at `dof = 0.5` the density at `cdf = 0.001` is `1.9e8`, where the absolute tolerance is below one ulp.
_DOFS = [1.0, 1.5, 2.0, 5.0, 15.0, 80.0, 300.0]
# Endpoints excluded: `ppf(0) = 0` and `ppf(1) = +inf` are asserted in `inverse_test.py`.
_QUANTILES = [1e-10, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1 - 1e-10]

# Every method holds the default `1e-12` except `pdf`, the `exp` of the log density, as for `Gamma`.
_EXP_OF_LOG_DENSITY_TOL = 1e-11
_CASES: list[Case[ChiSquared]] = [
    Case("pdf", "value", lambda d, x: d.pdf(x), "pdf", tol=_EXP_OF_LOG_DENSITY_TOL),
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

_PROBABILITIES = (0.001, 0.05, 0.3, 0.5, 0.7, 0.95, 0.999)
"""The cdf levels the value grid sits at, so every method stays `O(10)` at every `dof`."""


def _value_grid(dof: float) -> list[float]:
    """Below the support, then the points at `_PROBABILITIES` and one at `sf = e ** -20`.

    `0` itself is `support_test.py`'s.
    """
    frozen = scipy_chi2(dof)
    return [-1.0, *(float(x) for x in frozen.ppf(_PROBABILITIES)), float(frozen.isf(2.0e-9))]


@pytest.mark.parametrize("dof", _DOFS, ids=lambda dof: f"dof={dof}")
@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.name)
def test_method_matches_scipy(case: Case[ChiSquared], dof: float) -> None:
    """Every method matches `scipy.stats.chi2(dof)` across the parameter grid.

    Method-specific behaviour (null propagation, out-of-range quantiles, the support edge) is asserted
    in the shared contract files.
    """
    assert_case_matches_scipy(
        case,
        dist=ChiSquared(dof=dof),
        scipy_frozen=scipy_chi2(dof),
        value_grid=_value_grid(dof),
        quantiles=_QUANTILES,
    )


_INVERSE_DOFS = [0.1, 1.0, 4.0, 200.0]
_INVERSE_QUANTILES = [1e-7, 1e-6, 1e-5, 0.9999, 1 - 1e-7]


@pytest.mark.parametrize("dof", _INVERSE_DOFS, ids=lambda dof: f"dof={dof}")
@pytest.mark.parametrize("method", ["ppf", "isf"])
def test_inverses_match_scipy_in_both_tails(dof: float, method: Literal["ppf", "isf"]) -> None:
    """`ppf` and `isf` hold `1e-12` relative at every quantile of the sweep.

    Relative, not the harness's absolute tolerance: the quantiles span `1e-140` to `320`.
    """
    frame = pl.DataFrame({"q": _INVERSE_QUANTILES})
    got = frame.select(r=getattr(ChiSquared(dof=dof), method)(pl.col("q")))["r"]
    expected = pl.Series(values=getattr(scipy_chi2(dof), method)(_INVERSE_QUANTILES))
    assert_series_equal(got, expected, rel_tol=1e-12, abs_tol=0.0, check_names=False)
