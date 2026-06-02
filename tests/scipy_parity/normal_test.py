from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
from scipy.stats import norm as scipy_norm

from polars_stats import Normal

if TYPE_CHECKING:
    from collections.abc import Callable

# Parameter and evaluation grids for the parity sweep. Owned by this test category and independent
# of the per-method functional tests under `tests/distributions/normal`.
_PARAMS = [(0.0, 1.0), (1.0, 2.0), (-3.0, 0.5), (10.0, 5.0), (0.0, 1e-3)]
_QUANTILES = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]

# Tolerance split, by upstream implementation rather than by aspiration:
#   * `statrs` evaluates pdf / ln_pdf / inverse_cdf and the moments to ~1e-13 against scipy,
#     so they hold the 1e-12 target the issue asks for;
#   * cdf / sf (and their logs) go through `erfc`, while scipy uses the Cephes `ndtr`; the two
#     agree only to ~1e-10. 1e-9 is the honest, comfortably-padded bound for that family. See the
#     PR description for the measured worst case.
_TOL_EXACT = 1e-12
_TOL_ERF = 1e-9


@dataclass(frozen=True)
class _Case:
    """One `Normal` method paired with the `scipy.stats.norm` attribute it must reproduce.

    `kind` selects the input grid: `"value"` evaluates over `value_grid`, `"quantile"` over
    `_QUANTILES`, `"scalar"` takes no input (moments / entropy). `tol` is the absolute tolerance.
    """

    name: str
    kind: str
    pl_fn: Callable[[Normal, pl.Expr], pl.Expr]
    scipy_attr: str
    tol: float


_CASES = [
    _Case("pdf", "value", lambda n, c: n.pdf(c), "pdf", _TOL_EXACT),
    _Case("log_pdf", "value", lambda n, c: n.log_pdf(c), "logpdf", _TOL_EXACT),
    _Case("cdf", "value", lambda n, c: n.cdf(c), "cdf", _TOL_ERF),
    _Case("log_cdf", "value", lambda n, c: n.log_cdf(c), "logcdf", _TOL_ERF),
    _Case("sf", "value", lambda n, c: n.sf(c), "sf", _TOL_ERF),
    _Case("log_sf", "value", lambda n, c: n.log_sf(c), "logsf", _TOL_ERF),
    _Case("ppf", "quantile", lambda n, c: n.ppf(c), "ppf", _TOL_EXACT),
    _Case("isf", "quantile", lambda n, c: n.isf(c), "isf", _TOL_EXACT),
    _Case("mean", "scalar", lambda n, _: n.mean(), "mean", _TOL_EXACT),
    _Case("variance", "scalar", lambda n, _: n.variance(), "var", _TOL_EXACT),
    _Case("std", "scalar", lambda n, _: n.std(), "std", _TOL_EXACT),
    _Case("median", "scalar", lambda n, _: n.median(), "median", _TOL_EXACT),
    _Case("entropy", "scalar", lambda n, _: n.entropy(), "entropy", _TOL_EXACT),
]


def _value_grid(mean: float, std: float) -> list[float]:
    """Evaluation points spanning both tails through the centre of a `(mean, std)` distribution."""
    return [mean - 3 * std, mean - std, mean - 0.25 * std, mean, mean + 0.25 * std, mean + std, mean + 3 * std]


@pytest.mark.parametrize(("mean", "std"), _PARAMS, ids=[f"mean={m},std={s}" for m, s in _PARAMS])
@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.name)
def test_method_matches_scipy(case: _Case, mean: float, std: float) -> None:
    """Every closed-form method matches `scipy.stats.norm` across the `(mean, std)` grid.

    Table-driven: adding a method is one row in `_CASES`. Method-specific behaviour (null
    propagation, out-of-range quantiles, infinite ppf endpoints) is asserted in the per-method files.
    """
    n = Normal(mean=mean, std_dev=std)
    frozen = scipy_norm(loc=mean, scale=std)

    if case.kind == "scalar":
        result = pl.DataFrame({"_": [0]}).select(r=case.pl_fn(n, pl.lit(0.0)))["r"].to_numpy()
        expected = np.asarray([getattr(frozen, case.scipy_attr)()], dtype=float)
    else:
        xs = _value_grid(mean, std) if case.kind == "value" else _QUANTILES
        result = pl.DataFrame({"x": xs}).select(r=case.pl_fn(n, pl.col("x")))["r"].to_numpy()
        expected = np.asarray(getattr(frozen, case.scipy_attr)(xs), dtype=float)

    np.testing.assert_allclose(result, expected, atol=case.tol, rtol=0)
