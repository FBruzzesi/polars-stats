from __future__ import annotations

import numpy as np
import polars as pl
from polars.testing import assert_series_equal

from polars_stats import Normal


def test_entropy_is_log_form_column_params() -> None:
    sigmas = [1.0, 0.5, 5.0]
    df = pl.DataFrame({"mu": [0.0, -3.0, 10.0], "sigma": sigmas})
    result = df.select(r=Normal(mean=pl.col("mu"), std_dev=pl.col("sigma")).entropy())["r"]
    # Differential entropy of a normal: 0.5 * log(2*pi*e*sigma^2), independent of the mean.
    expected = pl.Series("r", [0.5 * np.log(2.0 * np.pi * np.e * s**2) for s in sigmas], dtype=pl.Float64)
    assert_series_equal(result, expected)
