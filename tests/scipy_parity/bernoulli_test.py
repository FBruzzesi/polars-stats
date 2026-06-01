from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
from scipy.stats import bernoulli as scipy_bernoulli

from polars_stats import Bernoulli

if TYPE_CHECKING:
    from collections.abc import Callable

# Parameter and evaluation grids for the parity sweep. Owned by this test category and independent
# of the per-method functional tests under `tests/distributions/bernoulli`.
_PROBS = [0.0, 0.25, 0.5, 0.75, 1.0]
# Interior quantiles only: scipy's discrete ppf/isf return the below-support sentinel -1 at the
# exact endpoints q in {0, 1}, which the Boolean-valued `ppf` here (range {False, True}) does not
# reproduce. The interior is where the two definitions agree.
_QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
# Bernoulli support is {0, 1}; the grid spans below, both support points, a non-integer, and above.
_VALUE_GRID = [-1.0, 0.0, 0.5, 1.0, 2.0]


@dataclass(frozen=True)
class _Case:
    """One `Bernoulli` method paired with the `scipy.stats.bernoulli` attribute it must reproduce.

    `kind` selects the input grid: `"value"` evaluates over `_VALUE_GRID`, `"quantile"` over the
    interior `_QUANTILES`, `"scalar"` takes no input (moments / entropy).
    """

    name: str
    kind: str
    pl_fn: Callable[[Bernoulli, pl.Expr], pl.Expr]
    scipy_attr: str


_CASES = [
    _Case("pmf", "value", lambda b, c: b.pmf(c), "pmf"),
    _Case("log_pmf", "value", lambda b, c: b.log_pmf(c), "logpmf"),
    _Case("cdf", "value", lambda b, c: b.cdf(c), "cdf"),
    _Case("log_cdf", "value", lambda b, c: b.log_cdf(c), "logcdf"),
    _Case("sf", "value", lambda b, c: b.sf(c), "sf"),
    _Case("log_sf", "value", lambda b, c: b.log_sf(c), "logsf"),
    _Case("ppf", "quantile", lambda b, c: b.ppf(c), "ppf"),
    _Case("isf", "quantile", lambda b, c: b.isf(c), "isf"),
    _Case("mean", "scalar", lambda b, _: b.mean(), "mean"),
    _Case("variance", "scalar", lambda b, _: b.variance(), "var"),
    _Case("std", "scalar", lambda b, _: b.std(), "std"),
    _Case("median", "scalar", lambda b, _: b.median(), "median"),
    _Case("entropy", "scalar", lambda b, _: b.entropy(), "entropy"),
]


@pytest.mark.parametrize("p", _PROBS, ids=[f"p={p}" for p in _PROBS])
@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.name)
def test_method_matches_scipy(case: _Case, p: float) -> None:
    """Every closed-form method matches `scipy.stats.bernoulli` across `_PROBS`.

    Table-driven: adding a method is one row in `_CASES`. `ppf`/`isf` use interior quantiles only,
    since scipy's discrete inverse returns the below-support sentinel -1 at `q` in `{0, 1}`, which
    the Boolean-valued `ppf` here does not reproduce. Method-specific behaviour (support handling,
    null propagation) is asserted in the per-method test files.
    """
    b = Bernoulli(p=p)
    frozen = scipy_bernoulli(p)

    if case.kind == "scalar":
        got = pl.DataFrame({"_": [0]}).select(r=case.pl_fn(b, pl.lit(0.0)))["r"].to_numpy()
        expected = np.asarray([getattr(frozen, case.scipy_attr)()], dtype=float)
    else:
        xs = _VALUE_GRID if case.kind == "value" else _QUANTILES
        got = pl.DataFrame({"x": xs}).select(r=case.pl_fn(b, pl.col("x")))["r"].to_numpy()
        expected = np.asarray(getattr(frozen, case.scipy_attr)(xs), dtype=float)

    # Cast `got` to float so Boolean outputs (ppf / isf / median) compare as 0.0 / 1.0.
    np.testing.assert_allclose(np.asarray(got, dtype=float), expected, atol=1e-12, rtol=0)
