from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
from scipy.stats import uniform as scipy_uniform

from polars_stats import Uniform

if TYPE_CHECKING:
    from collections.abc import Callable

# Parameter and evaluation grids for the parity sweep. Owned by this test category and independent
# of the per-method functional tests under `tests/distributions/uniform`.
_PARAMS = [(0.0, 1.0), (-2.0, 3.0), (2.0, 5.0), (-5.0, -1.0), (0.0, 1e-3)]
_QUANTILES = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]


def _value_grid(mn: float, mx: float) -> list[float]:
    """Evaluation points for a `(min, max)` support: below, both endpoints, interior, above."""
    width = mx - mn
    return [mn - width, mn, mn + 0.25 * width, (mn + mx) / 2, mn + 0.75 * width, mx, mx + width]


@dataclass(frozen=True)
class _Case:
    """One `Uniform` method paired with the `scipy.stats.uniform` attribute it must reproduce.

    `kind` selects the input grid: `"value"` evaluates over `_value_grid`, `"quantile"` over
    `_QUANTILES`, `"scalar"` takes no input (moments / entropy).
    """

    name: str
    kind: str
    pl_fn: Callable[[Uniform, pl.Expr], pl.Expr]
    scipy_attr: str


_CASES = [
    _Case("pdf", "value", lambda u, c: u.pdf(c), "pdf"),
    _Case("log_pdf", "value", lambda u, c: u.log_pdf(c), "logpdf"),
    _Case("cdf", "value", lambda u, c: u.cdf(c), "cdf"),
    _Case("log_cdf", "value", lambda u, c: u.log_cdf(c), "logcdf"),
    _Case("sf", "value", lambda u, c: u.sf(c), "sf"),
    _Case("log_sf", "value", lambda u, c: u.log_sf(c), "logsf"),
    _Case("ppf", "quantile", lambda u, c: u.ppf(c), "ppf"),
    _Case("isf", "quantile", lambda u, c: u.isf(c), "isf"),
    _Case("mean", "scalar", lambda u, _: u.mean(), "mean"),
    _Case("variance", "scalar", lambda u, _: u.variance(), "var"),
    _Case("std", "scalar", lambda u, _: u.std(), "std"),
    _Case("median", "scalar", lambda u, _: u.median(), "median"),
    _Case("entropy", "scalar", lambda u, _: u.entropy(), "entropy"),
]


@pytest.mark.parametrize(("mn", "mx"), _PARAMS, ids=[f"min={mn},max={mx}" for mn, mx in _PARAMS])
@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.name)
def test_method_matches_scipy(case: _Case, mn: float, mx: float) -> None:
    """Every closed-form method matches `scipy.stats.uniform` across the `(min, max)` grid.

    Table-driven: adding a method is one row in `_CASES`. Method-specific behaviour (support
    clamping, null propagation, out-of-range quantiles) is asserted in the per-method test files.
    """
    u = Uniform(min=mn, max=mx)
    frozen = scipy_uniform(loc=mn, scale=mx - mn)

    if case.kind == "scalar":
        result = pl.DataFrame({"_": [0]}).select(r=case.pl_fn(u, pl.lit(0.0)))["r"].to_numpy()
        expected = np.asarray([getattr(frozen, case.scipy_attr)()], dtype=float)
    else:
        xs = _value_grid(mn, mx) if case.kind == "value" else _QUANTILES
        result = pl.DataFrame({"x": xs}).select(r=case.pl_fn(u, pl.col("x")))["r"].to_numpy()
        expected = np.asarray(getattr(frozen, case.scipy_attr)(xs), dtype=float)

    np.testing.assert_allclose(result, expected, atol=1e-12, rtol=0)
