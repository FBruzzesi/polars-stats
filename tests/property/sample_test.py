from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from polars.testing import assert_series_equal

from tests.property._specs import ALL_SPECS

if TYPE_CHECKING:
    from tests.property._specs import DistSpec

_N_ROWS = 64
_SEED = 12345


@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: s.name)
@given(data=st.data())
def test_sample_seeded_is_reproducible(spec: DistSpec, data: st.DataObject) -> None:
    """`sample(seed=N)` returns identical draws across two calls in the same process."""
    params = data.draw(spec.params)
    dist = spec.make(params)
    frame = pl.DataFrame({"_": range(_N_ROWS)})

    first = frame.select(s=dist.sample(seed=_SEED))["s"]
    second = frame.select(s=dist.sample(seed=_SEED))["s"]

    assert_series_equal(first, second)
