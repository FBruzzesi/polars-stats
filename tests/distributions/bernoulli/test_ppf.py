from __future__ import annotations

import polars as pl
import pytest

from polars_stats import Bernoulli


@pytest.mark.parametrize(
    ("p", "quantile", "expected"),
    [
        (0.3, 0.0, False),
        (0.3, 0.5, False),  # 0.5 <= 1 - 0.3 = 0.7
        (0.3, 0.7, False),  # boundary: q <= 1 - p → 0
        (0.3, 0.8, True),
        (0.3, 1.0, True),
        (0.0, 0.5, False),
        (1.0, 0.5, True),
        (1.0, 0.0, False),  # 0 > 1 - 1 = 0 is False
    ],
)
def test_ppf_scalar(p: float, quantile: float, *, expected: bool, unit_frame: pl.DataFrame) -> None:
    out = unit_frame.select(v=Bernoulli(p=p).ppf(quantile)).item(0, "v")
    assert out is expected


def test_ppf_propagates_null() -> None:
    df = pl.DataFrame({"q": [0.1, None, 0.9]}, schema={"q": pl.Float64})
    out = df.select(v=Bernoulli(p=0.3).ppf(pl.col("q")))["v"].to_list()
    assert out[1] is None
    assert out[0] is False
    assert out[2] is True
