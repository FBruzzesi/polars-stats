from __future__ import annotations

import polars as pl
import pytest

from polars_stats.distributions import _base
from polars_stats.distributions._base import row_index_expr

# `row_index_expr` has two implementations of one idea, chosen by `_LITERAL_LEN_IN_AGG`. From polars
# 1.35 a literal's length can be asked for inside a partition context; below it the frame-free lengths
# are resolved eagerly instead. Only one runs on any installed polars, so the gate is parametrised and
# every case asserts the same index on both arms, which is the agreement the gate assumes.

_FRAME = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
_FRAME_INDEX = [0, 1, 2]
_SERIES_ROWS = 8
_SERIES_INDEX = list(range(_SERIES_ROWS))


def _series_param() -> pl.Expr:
    return pl.lit(pl.Series([0.5] * _SERIES_ROWS))


# A bare column always has the frame's height and a length-1 input never sets the row count, so both
# leave the index at the frame's own length; a longer parameter sets it. `pl.col("x") * 2` is a
# candidate that is not a bare column and whose length needs a frame, which is what the eager arm
# defers to `.len()`.
_CASES: dict[str, tuple[tuple[pl.Expr, ...], list[int]]] = {
    "bare column": ((pl.col("x"),), _FRAME_INDEX),
    "length-1 literal": ((pl.lit(1.0),), _FRAME_INDEX),
    "series longer than the frame": ((_series_param(),), _SERIES_INDEX),
    "frame-dependent beside a series": ((pl.col("x") * 2, _series_param()), _SERIES_INDEX),
}

_GATE_ARMS = {"literal-len-in-agg": True, "eager-lengths": False}


@pytest.mark.parametrize("literal_len_in_agg", _GATE_ARMS.values(), ids=list(_GATE_ARMS))
@pytest.mark.parametrize(("params", "expected"), _CASES.values(), ids=list(_CASES))
def test_row_index_follows_the_calls_row_count(
    params: tuple[pl.Expr, ...],
    expected: list[int],
    *,
    literal_len_in_agg: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_base, "_LITERAL_LEN_IN_AGG", literal_len_in_agg)

    assert _FRAME.select(r=row_index_expr(params))["r"].to_list() == expected


_LENGTHS: dict[str, tuple[pl.Expr, int | None]] = {
    "length-1 literal": (pl.lit(1.0), 1),
    "series literal": (_series_param(), _SERIES_ROWS),
    "frame-dependent expr": (pl.col("x") * 2, None),
}


@pytest.mark.parametrize(("param", "expected"), _LENGTHS.values(), ids=list(_LENGTHS))
def test_frame_free_length_reports_only_what_it_can_resolve(param: pl.Expr, expected: int | None) -> None:
    assert _base._frame_free_length(param) == expected
