from __future__ import annotations

from typing import Literal

import polars as pl
import pytest
import scipy
from packaging.version import Version
from scipy.stats import t as scipy_t

from polars_stats import StudentsT
from tests._polars_compat import assert_series_equal
from tests.scipy_parity._harness import Case, assert_case_matches_scipy

# `dof` spans the moment regimes (no mean up to `1`, a divergent variance up to `2`), the closed-form quantiles at `1`
# and `2`, the solver between and beyond them, and the near-normal bulk at `1e3` and `1e6`.
_DOFS = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 30.0, 1e3, 1e6]
_LOC, _SCALE = -1.25, 2.5
# Endpoints excluded: `ppf(0) = -inf` and `ppf(1) = +inf` are asserted in `inverse_test.py`. The tails are the
# relative sweep's below: at `dof = 0.5` the quantile at `1e-10` is `1e19`, past any absolute tolerance.
_QUANTILES = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]

# Every method holds the default `1e-12` except `pdf`, the `exp` of the log density, as for `Gamma`, and the
# inverses: at `dof = 0.5` the quantile at `0.99` is `2570`, where the tail's few-ulp error, doubled by the `1 / dof`
# power the inverse takes, is `4.4e-12` absolute.
_EXP_OF_LOG_DENSITY_TOL = 1e-11
_INVERSE_TOL = 1e-11
_VALUE_KEYED_CASES: list[Case[StudentsT]] = [
    Case("pdf", "value", lambda d, x: d.pdf(x), "pdf", tol=_EXP_OF_LOG_DENSITY_TOL),
    Case("log_pdf", "value", lambda d, x: d.log_pdf(x), "logpdf"),
    Case("cdf", "value", lambda d, x: d.cdf(x), "cdf"),
    Case("log_cdf", "value", lambda d, x: d.log_cdf(x), "logcdf"),
    Case("sf", "value", lambda d, x: d.sf(x), "sf"),
    Case("log_sf", "value", lambda d, x: d.log_sf(x), "logsf"),
    Case("ppf", "quantile", lambda d, x: d.ppf(x), "ppf", tol=_INVERSE_TOL),
    Case("isf", "quantile", lambda d, x: d.isf(x), "isf", tol=_INVERSE_TOL),
    Case("median", "scalar", lambda d, _: d.median(), "median"),
    Case("entropy", "scalar", lambda d, _: d.entropy(), "entropy"),
]
_MEAN = Case[StudentsT]("mean", "scalar", lambda d, _: d.mean(), "mean")

_INVERSES_NEED_SCIPY_1_17 = pytest.mark.skipif(
    Version(scipy.__version__) < Version("1.17"),
    reason="scipy < 1.17's `t.ppf` / `t.isf` are up to `2e-11` relative off, `9e-12` at `t(1).ppf(1e-5)` against "
    "Cauchy's closed form, so the comparison would check scipy",
)
_SPREAD_CASES: list[Case[StudentsT]] = [
    Case("variance", "scalar", lambda d, _: d.variance(), "var"),
    Case("std", "scalar", lambda d, _: d.std(), "std"),
]

_PROBABILITIES = (1e-6, 0.001, 0.05, 0.3, 0.5, 0.7, 0.95, 0.999, 1 - 1e-6)
"""The cdf levels the value grid sits at, so every method stays `O(10)` at every `dof`."""


def _value_grid(dof: float) -> list[float]:
    return [float(x) for x in scipy_t(dof, loc=_LOC, scale=_SCALE).ppf(_PROBABILITIES)]


def _cases(dof: float) -> list[Case[StudentsT]]:
    """The value-keyed cases, plus the moments scipy has a value for: none up to `dof = 1`, see `moments_test.py`."""
    return [*_VALUE_KEYED_CASES, *([_MEAN, *_SPREAD_CASES] if dof > 1 else [])]


@pytest.mark.parametrize(
    ("case", "dof"),
    [
        pytest.param(
            case,
            dof,
            id=f"{case.name}-dof={dof}",
            marks=_INVERSES_NEED_SCIPY_1_17 if case.name in ("ppf", "isf") else (),
        )
        for dof in _DOFS
        for case in _cases(dof)
    ],
)
def test_method_matches_scipy(case: Case[StudentsT], dof: float) -> None:
    """Every method matches `scipy.stats.t(dof, loc, scale)` across the parameter grid.

    Method-specific behaviour (null propagation, out-of-range quantiles) is asserted in the shared contract files.
    """
    assert_case_matches_scipy(
        case,
        dist=StudentsT(dof=dof, loc=_LOC, scale=_SCALE),
        scipy_frozen=scipy_t(dof, loc=_LOC, scale=_SCALE),
        value_grid=_value_grid(dof),
        quantiles=_QUANTILES,
    )


_INVERSE_DOFS = [0.5, 1.0, 2.0, 3.0, 4.0, 4.5, 30.0, 1e6]
# Not deeper: scipy's own inverse leaves the tail there (`t(5).ppf(1e-300)` is `+inf`), which the audit measures.
_INVERSE_QUANTILES = [1e-10, 1e-7, 1e-5, 0.9999, 1 - 1e-7, 1 - 1e-10]


@_INVERSES_NEED_SCIPY_1_17
@pytest.mark.parametrize("dof", _INVERSE_DOFS, ids=lambda dof: f"dof={dof}")
@pytest.mark.parametrize("method", ["ppf", "isf"])
def test_inverses_match_scipy_in_both_tails(dof: float, method: Literal["ppf", "isf"]) -> None:
    """`ppf` and `isf` hold `1e-12` relative at every quantile of the sweep.

    Relative, not the harness's absolute tolerance: at `dof = 0.5` the quantiles reach `1e19`. The standard `loc` and
    `scale` leave the answer the solver's own.
    """
    frame = pl.DataFrame({"q": _INVERSE_QUANTILES})
    got = frame.select(r=getattr(StudentsT(dof=dof), method)(pl.col("q")))["r"]
    expected = pl.Series(values=getattr(scipy_t(dof), method)(_INVERSE_QUANTILES))
    assert_series_equal(got, expected, rel_tol=1e-12, abs_tol=0.0, check_names=False)


@_INVERSES_NEED_SCIPY_1_17
def test_inverses_on_a_dof_column_match_scipy() -> None:
    """A `dof` column builds each row's state on its own, so the closed forms at `1`, `2` and `4` and the solver's
    fallback bracket below `dof = 1` are reached per row, not only on the constant path.
    """
    dofs = [0.5, 1.0, 2.0, 4.0, 4.5, 0.5, 1.0, 2.0, 4.0, 4.5]
    qs = [1e-7, 0.01, 0.3, 0.7, 0.99, 1 - 1e-7, 0.45, 0.2, 1e-5, 0.9999]
    frame = pl.DataFrame({"dof": dofs, "q": qs})
    dist = StudentsT(dof=pl.col("dof"), loc=_LOC, scale=_SCALE)
    got = frame.select(ppf=dist.ppf(pl.col("q")), isf=dist.isf(pl.col("q")))
    frozen = scipy_t(dofs, loc=_LOC, scale=_SCALE)
    assert_series_equal(got["ppf"], pl.Series(values=frozen.ppf(qs)), rel_tol=1e-12, abs_tol=0.0, check_names=False)
    assert_series_equal(got["isf"], pl.Series(values=frozen.isf(qs)), rel_tol=1e-12, abs_tol=0.0, check_names=False)
