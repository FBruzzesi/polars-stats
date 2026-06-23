from __future__ import annotations

import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st

from polars_stats import Exponential
from tests._polars_compat import assert_series_equal


@settings(deadline=None)
@given(
    rate=st.floats(min_value=1e-2, max_value=5.0),
    s=st.floats(min_value=0.0, max_value=15.0),
    t=st.floats(min_value=0.0, max_value=15.0),
)
def test_memoryless_property(rate: float, s: float, t: float) -> None:
    """The exponential's defining property: `P(X > s + t | X > s) = P(X > t)`.

    `P(X > s + t | X > s) = sf(s + t) / sf(s)`, which must equal `sf(t)` for every `s, t >= 0`. The
    parameter and offset ranges are bounded so `rate * (s + t)` stays well clear of float underflow,
    keeping both survival values strictly positive (no `0 / 0`).
    """
    dist = Exponential(rate=rate)
    df = pl.DataFrame({"_": [0]})

    conditional = df.select(r=dist.sf(s + t) / dist.sf(s))["r"]
    marginal = df.select(r=dist.sf(t))["r"]

    assert_series_equal(conditional, marginal, rel_tol=1e-9, abs_tol=1e-15, check_names=False)
