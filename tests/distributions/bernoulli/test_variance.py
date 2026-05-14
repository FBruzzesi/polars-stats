from __future__ import annotations

import polars as pl
import pytest

from polars_stats import Bernoulli


@pytest.mark.parametrize("p", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_variance_scalar(p: float, unit_frame: pl.DataFrame) -> None:
    out = unit_frame.select(v=Bernoulli(p=p).variance()).item(0, "v")
    assert out == pytest.approx(p * (1 - p))


def test_variance_column_p() -> None:
    probs = [0.1, 0.4, 0.6, 0.9]
    df = pl.DataFrame({"p": probs})
    out = df.select(v=Bernoulli(p=pl.col("p")).variance())["v"].to_list()
    assert out == pytest.approx([x * (1 - x) for x in probs])
