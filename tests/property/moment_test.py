"""Bit-equality of the constant-parameter moment fast path against the per-row path.

The parameter-only moments (`mean`, `variance`, `std`, `entropy`) validate their parameters in Rust so an invalid
parameterisation raises consistently with the value-keyed methods.

With constant scalar parameters that validation runs **once** (the validating plugin is called on length-1 `pl.lit`
inputs); with column parameters it runs per row. Both return a length-n column and gate the same closed form, so for any
valid parameterisation the two paths must agree bit for bit, the moment counterpart of
`test_value_keyed_scalar_fast_path_matches_per_row`.

A divergence (a parameter-order swap in the length-1 lit args, a non-bit-identical width recompute on the scalar path,
or a length mismatch from a missing broadcast) must fail here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

from tests._polars_compat import assert_series_equal
from tests.property._specs import ALL_SPECS

if TYPE_CHECKING:
    from tests.property._specs import DistSpec

_N_ROWS = 64

# Every distribution exposes these as closed forms (or, for Binomial entropy, a Rust support sum)
# derived from the validated parameters; `median` is excluded as it routes through `ppf` for the
# discrete families (covered by the value-keyed suite) rather than the moment validator.
_MOMENTS = ("mean", "variance", "std", "entropy")


@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: s.name)
@pytest.mark.parametrize("moment", _MOMENTS)
@given(data=st.data())
def test_moment_scalar_fast_path_matches_per_row(spec: DistSpec, moment: str, data: st.DataObject) -> None:
    """Constant scalar parameters and the equivalent per-row columns evaluate each moment identically."""
    params = data.draw(spec.params)
    frame = pl.DataFrame({"_": range(_N_ROWS)})

    fast = frame.select(r=getattr(spec.make(params), moment)())["r"]
    per_row = frame.select(r=getattr(spec.make_columns(params), moment)())["r"]

    assert_series_equal(fast, per_row, check_exact=True)
    assert fast.len() == _N_ROWS  # the scalar fast path stays length-n (broadcast), not collapsed to 1
