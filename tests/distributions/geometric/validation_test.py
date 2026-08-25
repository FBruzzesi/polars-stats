from __future__ import annotations

import polars as pl
import pytest

from polars_stats import Geometric


def test_invalid_scalar_p_raises() -> None:
    df = pl.DataFrame({"k": [1.0, 2.0]})
    with pytest.raises(pl.exceptions.ComputeError, match="p must be"):
        df.with_columns(pmf=Geometric(p=0.0).pmf("k"))


def test_invalid_column_p_raises() -> None:
    df = pl.DataFrame({"p": [0.5, 1.5], "k": [1.0, 1.0]})
    with pytest.raises(pl.exceptions.ComputeError, match="p must be"):
        df.with_columns(pmf=Geometric(p="p").pmf("k"))


def test_invalid_scalar_sample_raises() -> None:
    df = pl.DataFrame({"x": [1.0]})
    with pytest.raises(pl.exceptions.ComputeError, match="p must be"):
        df.with_columns(y=Geometric(p=1.5).sample(seed=0))
