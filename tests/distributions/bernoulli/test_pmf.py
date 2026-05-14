from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from polars_stats import Bernoulli

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize("p", [0.0, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize(
    ("value", "expected_fn"),
    [
        (0, lambda p: 1 - p),
        (1, lambda p: p),
        (2, lambda _p: 0.0),
        (-1, lambda _p: 0.0),
        (0.5, lambda _p: 0.0),  # non-integer support points
    ],
)
def test_pmf_scalar_p(
    p: float,
    value: float,
    expected_fn: Callable[[float], float],
    unit_frame: pl.DataFrame,
) -> None:
    out = unit_frame.select(v=Bernoulli(p=p).pmf(value)).item(0, "v")
    assert out == pytest.approx(expected_fn(p))


def test_pmf_column_p() -> None:
    probs = [0.1, 0.4, 0.6, 0.9]
    df = pl.DataFrame({"p": probs})
    pmf_at_1 = df.select(v=Bernoulli(p=pl.col("p")).pmf(1))["v"].to_list()
    pmf_at_0 = df.select(v=Bernoulli(p=pl.col("p")).pmf(0))["v"].to_list()
    assert pmf_at_1 == pytest.approx(probs)
    assert pmf_at_0 == pytest.approx([1 - x for x in probs])
