"""Compatibility shims for ``polars`` across supported polars versions.

Polars renamed the tolerance keyword arguments of `assert_series_equal` from ``rtol``/``atol`` to
``rel_tol``/``abs_tol`` in v1.32.3. This project supports ``polars>=1.15.0``, so the test suite has to run
against both spellings.

The wrapper exposes the *latest* signature (``rel_tol``/``abs_tol``) explicitly, by remapping the two tolerance
arguments to whatever the installed polars version actually accepts; every other argument has a stable name and is
forwarded untouched.

`linear_space` backfills `polars.linear_space`, which is missing on older supported polars; the implementation here
reproduces its evenly spaced, inclusive-endpoint grid on every supported version.

`arr_explode` wraps `Series.arr.explode`: polars 1.36 added the `empty_as_null` flag and 1.42 deprecated its
default (a warning `filterwarnings = ["error"]` escalates), so newer polars needs the explicit kwarg while older
supported polars does not accept it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from packaging.version import Version
from polars.testing import assert_series_equal as _assert_series_equal

if TYPE_CHECKING:
    from polars import Series

__all__ = (
    "PARTITIONED_BROADCAST_AVAILABLE",
    "arr_explode",
    "assert_series_equal",
    "linear_space",
)

PL_VERSION = Version(pl.__version__)
_NEEDS_RENAMING = Version("1.32.3") > PL_VERSION
_REL_NAME = "rtol" if _NEEDS_RENAMING else "rel_tol"
_ABS_NAME = "atol" if _NEEDS_RENAMING else "abs_tol"
_EXPLODE_HAS_EMPTY_AS_NULL = Version("1.36.0") <= PL_VERSION

PARTITIONED_BROADCAST_AVAILABLE = Version("1.34.0") <= PL_VERSION
"""Whether polars handles a length-1 input *inside* `over` / `group_by().agg()` correctly.

`align_inputs` broadcasts a length-1 plugin input on every supported version, and a plain `select` /
`with_columns` is correct throughout. Partitioned contexts are not before polars v1.34, and the two
upstream defects are the reason the partition suites are gated rather than the broadcast itself:

* an expression whose plugin inputs are **all** length 1 panics under `over` / `group_by().agg()`
  (`impl error, should be a list at this point`). Broken through 1.33, fixed in 1.34;
* a length-1 *value* (`pl.col("x").max()`) beside length-n parameters makes `over` return results in
  group order instead of scattering back to row order. Broken through 1.32, fixed in 1.33.

Both are polars defects, not plugin defects: the same expressions are correct on 1.34 and above with
an unchanged plugin. Documented for users in `docs/reference/parameters-and-contracts.md`.
When the polars floor reaches 1.34 this constant and every gate on it can be deleted.
"""


def arr_explode(series: Series) -> Series:
    """Version-agnostic `Series.arr.explode` with empty-as-empty semantics.

    Passes `empty_as_null=False` where the kwarg exists (polars >= 1.36; from 1.42 omitting it warns
    about the default flipping, which the suite's `filterwarnings = ["error"]` escalates) and calls
    plain `explode` on older supported polars, which has no kwarg and no warning. Every call site
    explodes an `Array` column, whose rows are never empty, so the two paths return identical output.
    """
    return series.arr.explode(empty_as_null=False) if _EXPLODE_HAS_EMPTY_AS_NULL else series.arr.explode()


def linear_space(start: float, end: float, num_samples: int) -> Series:
    """Version-agnostic eager `polars.linear_space`.

    Returns ``num_samples`` evenly spaced points from ``start`` to ``end`` inclusive, matching
    `pl.linear_space(start, end, num_samples, eager=True)` but built from `pl.int_range` so it also
    works on polars versions that predate `linear_space`. ``num_samples`` must be at least 2.
    """
    if num_samples < 2:  # noqa: PLR2004  # pragma: no cover
        msg = f"num_samples must be >= 2, got {num_samples}"
        raise ValueError(msg)
    step = (end - start) / (num_samples - 1)
    return pl.int_range(0, num_samples, eager=True).cast(pl.Float64) * step + start


def assert_series_equal(  # noqa: PLR0913
    left: Series,
    right: Series,
    *,
    check_dtypes: bool = True,
    check_names: bool = True,
    check_order: bool = True,
    check_exact: bool = False,
    rel_tol: float = 1e-05,
    abs_tol: float = 1e-08,
    categorical_as_str: bool = False,
) -> None:
    """Version-agnostic `polars.testing.assert_series_equal`.

    Mirrors the latest polars signature. ``rel_tol``/``abs_tol`` are remapped to the
    spelling the installed polars version supports.
    """
    _assert_series_equal(
        left,
        right,
        check_dtypes=check_dtypes,
        check_names=check_names,
        check_order=check_order,
        check_exact=check_exact,
        categorical_as_str=categorical_as_str,
        **{_REL_NAME: rel_tol, _ABS_NAME: abs_tol},
    )
