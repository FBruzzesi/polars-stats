"""Compatibility shims for ``polars.testing`` across supported polars versions.

Polars renamed the tolerance keyword arguments of `assert_frame_equal` and `assert_series_equal` from
``rtol``/``atol`` to ``rel_tol``/``abs_tol`` in v1.32.3.
This project supports ``polars>=1.15.0``, so the test suite has to run against both spellings.

These wrappers expose the *latest* signature (``rel_tol``/``abs_tol``) explicitly, by remapping the two tolerance
arguments to whatever the installed polars version actually accepts; every other argument has a stable name and is
forwarded untouched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from packaging.version import Version
from polars.testing import assert_frame_equal as _assert_frame_equal
from polars.testing import assert_series_equal as _assert_series_equal

if TYPE_CHECKING:
    from polars import DataFrame, LazyFrame, Series

__all__ = ["assert_frame_equal", "assert_series_equal"]

_NEEDS_RENAMING = Version(pl.__version__) < Version("1.32.3")
_REL_NAME = "rtol" if _NEEDS_RENAMING else "rel_tol"
_ABS_NAME = "atol" if _NEEDS_RENAMING else "abs_tol"


def assert_frame_equal(  # noqa: PLR0913
    left: DataFrame | LazyFrame,
    right: DataFrame | LazyFrame,
    *,
    check_row_order: bool = True,
    check_column_order: bool = True,
    check_dtypes: bool = True,
    check_exact: bool = False,
    rel_tol: float = 1e-05,
    abs_tol: float = 1e-08,
    categorical_as_str: bool = False,
) -> None:
    """Version-agnostic `polars.testing.assert_frame_equal`.

    Mirrors the latest polars signature. ``rel_tol``/``abs_tol`` are remapped to the
    spelling the installed polars version supports.
    """
    _assert_frame_equal(
        left,
        right,
        check_row_order=check_row_order,
        check_column_order=check_column_order,
        check_dtypes=check_dtypes,
        check_exact=check_exact,
        categorical_as_str=categorical_as_str,
        **{_REL_NAME: rel_tol, _ABS_NAME: abs_tol},
    )


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
