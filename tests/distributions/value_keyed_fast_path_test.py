"""Validation contract of the constant-parameter value-keyed fast path (Normal, LogNormal, Binomial, Beta).

`pdf` / `pmf` / `cdf` / `sf` / `ppf` with all-scalar parameters route through a dedicated ``<name>_<method>_scalar``
plugin that validates and builds the `statrs` distribution once (via the shared `build_dist`), instead of rebuilding it
per row. Two properties of that routing are pinned here, both of which the bit-equality property test
(`tests/property/value_keyed_test.py`, valid params only) does not exercise:

* **Both paths agree on validation.** For an invalid parameterisation both the fast path and the
  general per-row path must raise the same `ComputeError`; for a non-finite-but-accepted one (a
  positive-infinite scale, which `statrs` allows: only `NaN` or a non-positive scale is rejected)
  both must produce identical output. The fast path cannot quietly accept or reject something the
  per-row path does not.
* **The fast path validates up front.** `build_dist` runs once before any value is touched, so
  invalid scalar parameters raise even on a zero-row frame; the per-row path validates inside its
  per-element closure, which never runs on an empty frame, so it returns empty. This divergence is
  intentional (validate-once, mirroring the sampler fast path) and pinned so it cannot regress
  silently.

A Python `None` parameter cannot reach either path: `coerce_param` rejects it as a `TypeError` at construction, covered
by each distribution's `construct_test.py`. So there is no "null scalar parameter" case to test here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from polars_stats import Beta, Binomial, LogNormal, Normal
from polars_stats.distributions._base import ContinuousDistribution
from tests._polars_compat import assert_series_equal

if TYPE_CHECKING:
    from collections.abc import Callable

    from polars_stats.distributions._base import _UnivariateDistribution

_NAN = float("nan")
_INF = float("inf")


def _density(dist: _UnivariateDistribution, value: pl.Expr) -> pl.Expr:
    """`pdf` (continuous) or `pmf` (discrete), the representative value-keyed method per family."""
    return dist.pdf(value) if isinstance(dist, ContinuousDistribution) else dist.pmf(value)  # type: ignore[attr-defined]


def _col(value: float, dtype: pl.DataType | None = None) -> pl.Expr:
    """A full-length constant column (`pl.repeat`), forcing the general per-row path for a scalar value."""
    return pl.repeat(value, n=pl.len(), dtype=dtype)


# Each case builds the same parameterisation two ways -- all-scalar (constant-parameter fast path)
# and all-column (general per-row path) -- and records whether evaluating a value-keyed method
# should raise. `int` `n` for binomial keeps its `Int64` column dtype.
# id -> (scalar factory, column factory, should_raise)
_CASES: dict[str, tuple[Callable[[], _UnivariateDistribution], Callable[[], _UnivariateDistribution], bool]] = {
    "normal mu=nan": (lambda: Normal(_NAN, 1.0), lambda: Normal(_col(_NAN), _col(1.0)), True),
    "normal std=0": (lambda: Normal(0.0, 0.0), lambda: Normal(_col(0.0), _col(0.0)), True),
    "normal std=-1": (lambda: Normal(0.0, -1.0), lambda: Normal(_col(0.0), _col(-1.0)), True),
    "normal std=nan": (lambda: Normal(0.0, _NAN), lambda: Normal(_col(0.0), _col(_NAN)), True),
    # A positive-infinite scale is accepted by statrs (only NaN / non-positive is rejected); both
    # paths must accept it identically, not merely both reject the bad cases.
    "normal std=inf (accepted)": (lambda: Normal(0.0, _INF), lambda: Normal(_col(0.0), _col(_INF)), False),
    "lognormal sigma=nan": (lambda: LogNormal(0.0, _NAN), lambda: LogNormal(_col(0.0), _col(_NAN)), True),
    "lognormal sigma=-1": (lambda: LogNormal(0.0, -1.0), lambda: LogNormal(_col(0.0), _col(-1.0)), True),
    "lognormal sigma=inf (accepted)": (lambda: LogNormal(0.0, _INF), lambda: LogNormal(_col(0.0), _col(_INF)), False),
    "binomial p=nan": (lambda: Binomial(5, _NAN), lambda: Binomial(_col(5, pl.Int64()), _col(_NAN)), True),
    "binomial p=1.5": (lambda: Binomial(5, 1.5), lambda: Binomial(_col(5, pl.Int64()), _col(1.5)), True),
    "binomial n=-1": (lambda: Binomial(-1, 0.5), lambda: Binomial(_col(-1, pl.Int64()), _col(0.5)), True),
    "beta a=nan": (lambda: Beta(_NAN, 1.0), lambda: Beta(_col(_NAN), _col(1.0)), True),
    "beta a=0": (lambda: Beta(0.0, 1.0), lambda: Beta(_col(0.0), _col(1.0)), True),
    "beta b=-1": (lambda: Beta(2.0, -1.0), lambda: Beta(_col(2.0), _col(-1.0)), True),
    # An infinite shape is *rejected* by statrs, unlike the Normal / LogNormal scale; both paths must
    # agree on that too.
    "beta a=inf (rejected)": (lambda: Beta(_INF, 1.0), lambda: Beta(_col(_INF), _col(1.0)), True),
}


@pytest.mark.parametrize(("scalar_mk", "column_mk", "should_raise"), _CASES.values(), ids=list(_CASES))
def test_scalar_and_column_paths_agree_on_validation(
    scalar_mk: Callable[[], _UnivariateDistribution],
    column_mk: Callable[[], _UnivariateDistribution],
    *,
    should_raise: bool,
) -> None:
    """The fast path and the per-row path agree on what is a valid parameterisation."""
    frame = pl.DataFrame({"x": [0.5, 1.0, 2.0]})
    scalar, column = scalar_mk(), column_mk()

    if should_raise:
        with pytest.raises(pl.exceptions.ComputeError):
            frame.select(r=_density(scalar, pl.col("x")))
        with pytest.raises(pl.exceptions.ComputeError):
            frame.select(r=_density(column, pl.col("x")))
    else:
        fast = frame.select(r=_density(scalar, pl.col("x")))["r"]
        per_row = frame.select(r=_density(column, pl.col("x")))["r"]
        assert_series_equal(fast, per_row, check_exact=True)


def test_scalar_fast_path_validates_on_empty_input() -> None:
    """The fast path raises on invalid scalar params even with no rows; the per-row path returns empty.

    `build_dist` runs once up front on the fast path, so an invalid scale is caught regardless of
    input length. The per-row path validates inside the per-element closure, which is never entered
    on a zero-row frame, so it produces an empty result instead. Pinned because it is the one
    intended observable difference between the two paths.
    """
    empty = pl.DataFrame({"x": []}, schema={"x": pl.Float64})

    with pytest.raises(pl.exceptions.ComputeError, match="sigma must be finite and strictly positive"):
        empty.select(r=Normal(0.0, -1.0).pdf(pl.col("x")))

    per_row = empty.select(r=Normal(_col(0.0), _col(-1.0)).pdf(pl.col("x")))
    assert per_row.height == 0
