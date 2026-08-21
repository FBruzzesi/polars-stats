"""Bit-equality of the constant-parameter moment fast path against the per-row path.

The parameter-only moments (`mean`, `variance`, `std`, `entropy`) validate their parameters in Rust so an invalid
parameterisation raises consistently with the value-keyed methods.

With constant scalar parameters that validation runs **once** (the validating plugin is called on length-1 `pl.lit`
inputs); with column parameters it runs per row. The two paths differ in *shape* and must agree in *value*: constant
parameters make the whole expression a scalar column, length 1 on its own and broadcast wherever it meets a longer
column, while column parameters give a length-n one. Gating the same closed form, they must agree on every row.

A divergence (a parameter-order swap in the length-1 lit args, a non-bit-identical width recompute on the scalar path,
or a scalar path that stopped being a scalar) must fail here. `fast_path_context_test.py` pins the same equality in the
contexts where the broadcast target is not the whole frame.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

from tests._polars_compat import assert_series_equal
from tests.property._specs import ALL_SPECS, ULP_ABS_TOL, ULP_REL_TOL, ULP_TOLERANT_MOMENT_SPECS

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

    fast = getattr(spec.make(params), moment)()
    per_row = getattr(spec.make_columns(params), moment)()

    # The scalar path is a scalar column: length 1 on its own, whatever the frame height.
    assert frame.select(r=fast).height == 1
    # Selected beside the per-row column, polars broadcasts it, and then every row must agree.
    both = frame.select(fast=fast, per_row=per_row)
    assert both.height == _N_ROWS
    exact = spec.name not in ULP_TOLERANT_MOMENT_SPECS
    assert_series_equal(
        both["fast"], both["per_row"], check_names=False, check_exact=exact, rel_tol=ULP_REL_TOL, abs_tol=ULP_ABS_TOL
    )
