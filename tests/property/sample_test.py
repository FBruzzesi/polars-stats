from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from packaging.version import Version
from polars.testing import assert_series_equal

from tests._polars_compat import PL_VERSION
from tests.property._specs import ALL_SPECS

if TYPE_CHECKING:
    from tests.property._specs import DistSpec

_N_ROWS = 64
_SEED = 12345

# A multi-chunk source is the stressor for engine invariance: enough physical chunks that the streaming
# engine cannot treat the input as one contiguous block.
_STREAMING_CHUNKS = 8
_STREAMING_CHUNK_ROWS = 256


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


@settings(max_examples=10)
@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: s.name)
@given(data=st.data())
@pytest.mark.skipif(Version("1.36.0") > PL_VERSION, reason="Arbitrary cut for when both engine's are available")
def test_sample_seeded_matches_across_engines(spec: DistSpec, data: st.DataObject) -> None:
    """`sample(seed=N)` is invariant to the execution engine (in-memory vs streaming).

    The per-row `(seed, row_index)` keying makes a draw depend only on its global position, so the
    streaming engine, which ingests the source in morsels and may split it across chunks, must produce
    the same column as the in-memory engine. The multi-chunk source is what makes this bite: if the
    injected `pl.int_range(0, pl.len())` row index were evaluated per chunk rather than globally, the
    index would repeat across chunks, sub-seeds would collide, and the two engines would diverge.

    `max_examples` is capped below the suite default: each example runs two full `collect`s, and engine
    invariance does not need a wide parameter sweep to falsify.
    """
    params = data.draw(spec.params)
    dist = spec.make(params)

    frame = pl.concat(
        [pl.DataFrame({"_": range(_STREAMING_CHUNK_ROWS)}) for _ in range(_STREAMING_CHUNKS)],
        rechunk=False,
    )
    assert frame.n_chunks() > 1  # the multi-chunk layout is the whole point; guard against a silent rechunk

    expr = dist.sample(seed=_SEED)
    in_memory = frame.lazy().select(s=expr).collect(engine="in-memory")["s"]
    streaming = frame.lazy().select(s=expr).collect(engine="streaming")["s"]

    assert_series_equal(in_memory, streaming)
