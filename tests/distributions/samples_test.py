"""`samples`: `size` draws per row, as an `Array` column.

`tests/property/sample_test.py` owns the prefix stability (growing `size` extends each row without
disturbing what is already there), the `size=1`-equals-`sample` identity and the null-row layering.
What is left here is the shape of the output, the `size` argument's own contract, and that the draws
within one row actually differ.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from tests._polars_compat import arr_explode
from tests._registry import DRIVER_REGIMES, min_max

if TYPE_CHECKING:
    from tests._registry import DistSpec, Regime

_SEED = 42


@pytest.mark.parametrize("regime", DRIVER_REGIMES)
@pytest.mark.parametrize("size", [1, 4, 16])
def test_the_output_is_an_array_of_size_in_the_recorded_dtype(spec: DistSpec, regime: Regime, size: int) -> None:
    rows = 32
    got = pl.DataFrame({"i": range(rows)}).select(s=spec.build(regime).samples(size=size, seed=_SEED))
    assert got.height == rows
    assert got.schema["s"] == pl.Array(spec.sample_dtype, size)


@pytest.mark.parametrize("bad_size", [0, -1])
def test_a_non_positive_size_is_refused(spec: DistSpec, bad_size: int) -> None:
    """Rejected in Python at expression build time, so it never reaches the allocator."""
    with pytest.raises(ValueError, match="size must be a positive integer"):
        spec.build("scalar").samples(size=bad_size, seed=_SEED)


def test_the_draws_within_a_row_differ(spec: DistSpec) -> None:
    """Each element of a row's array is a fresh draw from that row's stream, not a repeat of the first.

    Read down the columns rather than across one row: a discrete distribution repeats values within
    a row by construction, but two *columns* being identical across 512 rows would mean the sampler
    advanced its stream once per row instead of once per element.
    """
    size, rows = 8, 512
    got = pl.DataFrame({"i": range(rows)}).select(s=spec.build("scalar").samples(size=size, seed=_SEED))["s"]
    columns = {tuple(got.arr.get(i).to_list()) for i in range(size)}
    assert len(columns) == size, f"{spec.name}: only {len(columns)} distinct columns out of {size}"


@pytest.mark.parametrize("regime", DRIVER_REGIMES)
def test_the_pooled_draws_lie_inside_the_support(spec: DistSpec, regime: Regime) -> None:
    """Every element of every array is within `spec.bounds`, as a single draw would be."""
    lo, hi = spec.bounds
    got = pl.DataFrame({"i": range(1_000)}).select(s=spec.build(regime).samples(size=8, seed=_SEED))["s"]
    low, high = min_max(arr_explode(got))
    assert low >= lo, f"{spec.name} drew {low} below {lo}"
    assert high <= hi, f"{spec.name} drew {high} above {hi}"
