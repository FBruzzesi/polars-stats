from __future__ import annotations

import numpy as np
import polars as pl
from polars.testing import assert_series_equal

from polars_stats import Uniform


def test_entropy_is_log_width_column_bounds() -> None:
    df = pl.DataFrame({"lo": [0.0, -2.0], "hi": [1.0, 3.0]})
    got = df.select(r=Uniform(min=pl.col("lo"), max=pl.col("hi")).entropy())["r"]
    expected = pl.Series("r", [float(np.log(1.0)), float(np.log(5.0))], dtype=pl.Float64)
    assert_series_equal(got, expected)
