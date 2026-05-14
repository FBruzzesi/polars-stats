from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from polars_stats import Bernoulli

if TYPE_CHECKING:
    from collections.abc import Callable

    import polars as pl


@pytest.mark.parametrize("p", [0.0, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize(
    ("value", "expected_fn"),
    [
        (-0.5, lambda _p: 0.0),
        (0, lambda p: 1 - p),
        (0.5, lambda p: 1 - p),
        (1, lambda _p: 1.0),
        (1.5, lambda _p: 1.0),
    ],
)
def test_cdf_scalar_p(
    p: float,
    value: float,
    expected_fn: Callable[[float], float],
    unit_frame: pl.DataFrame,
) -> None:
    out = unit_frame.select(v=Bernoulli(p=p).cdf(value)).item(0, "v")
    assert out == pytest.approx(expected_fn(p))
