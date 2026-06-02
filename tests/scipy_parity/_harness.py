from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Literal, TypeVar

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

# Default absolute tolerance for closed-form methods. `statrs` and the pure-polars closed forms agree
# with scipy to ~1e-13; 1e-12 is the padded bound. Methods routed through `erf`/`erfc` (e.g. the
# normal cdf/sf family) agree only to ~1e-10 and pass their own relaxed tolerance per `Case`.
DEFAULT_TOL = 1e-12

DistT = TypeVar("DistT")

Kind = Literal["value", "quantile", "scalar"]
"""Selects the evaluation grid for a `Case`:

* `"value"` evaluates `pl_fn` over the distribution's `value_grid` (pdf/pmf, cdf, sf, and their logs);
* `"quantile"` evaluates over `quantiles` (ppf, isf);
* `"scalar"` takes no input and compares a single number (mean, variance, std, median, entropy).
"""


@dataclass(frozen=True)
class Case(Generic[DistT]):
    """One polars-stats method paired with the `scipy.stats` attribute it must reproduce.

    Parametrising a parity test over a list of `Case` makes "add a method" one row rather than a new
    test. Per-distribution numeric grids stay in the test module; this type owns only the mapping from
    a method to its scipy oracle and the tolerance it must hold.

    Arguments:
        name: Method name, used only as the test id.
        kind: Which grid drives the comparison (see `Kind`).
        pl_fn: Builds the polars-stats expression from a distribution instance and an evaluation expr.
            For `kind="scalar"` the expr argument is unused (pass `pl.lit(0.0)`).
        scipy_attr: Attribute on the frozen scipy distribution producing the reference values.
        tol: Absolute tolerance for the comparison; defaults to `DEFAULT_TOL`.
    """

    name: str
    kind: Kind
    pl_fn: Callable[[DistT, pl.Expr], pl.Expr]
    scipy_attr: str
    tol: float = DEFAULT_TOL


def assert_case_matches_scipy(
    case: Case[DistT],
    *,
    dist: DistT,
    scipy_frozen: object,
    value_grid: Sequence[float],
    quantiles: Sequence[float],
) -> None:
    """Assert one `Case` reproduces its scipy oracle across the grid selected by `case.kind`.

    Boolean outputs (a discrete `ppf`/`isf`/`median`) are cast to `0.0`/`1.0` before comparison, so the
    same helper serves discrete and continuous distributions. `value_grid` / `quantiles` are owned by
    the calling test module because their support spans are distribution-specific.
    """
    if case.kind == "scalar":
        result = pl.DataFrame({"_": [0]}).select(r=case.pl_fn(dist, pl.lit(0.0)))["r"].to_numpy()
        expected = np.asarray([getattr(scipy_frozen, case.scipy_attr)()], dtype=float)
    else:
        xs = list(value_grid if case.kind == "value" else quantiles)
        result = pl.DataFrame({"x": xs}).select(r=case.pl_fn(dist, pl.col("x")))["r"].to_numpy()
        expected = np.asarray(getattr(scipy_frozen, case.scipy_attr)(xs), dtype=float)

    np.testing.assert_allclose(np.asarray(result, dtype=float), expected, atol=case.tol, rtol=0)
