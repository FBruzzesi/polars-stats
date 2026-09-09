"""`is_discrete` / `is_continuous`: the two kinds partition the catalogue, and nothing else."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

import polars_stats as ps
from tests.property._specs import ALL_SPECS

if TYPE_CHECKING:
    from tests.property._specs import DistSpec


@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: s.name)
def test_every_distribution_answers_exactly_one_guard(spec: DistSpec) -> None:
    """The guards agree with the spec's own `continuous` flag, and never both hold."""
    dist = spec.make(spec.example)

    assert ps.is_continuous(dist) is spec.continuous
    assert ps.is_discrete(dist) is not spec.continuous


@pytest.mark.parametrize(
    "obj",
    [
        None,
        0.0,
        "Normal",
        pl.col("x"),
        # The class, not an instance: a guard takes an object, so an untyped caller can pass either.
        ps.Normal,
        ps.Binomial,
    ],
    ids=repr,
)
def test_non_distributions_answer_neither_guard(obj: object) -> None:
    assert ps.is_discrete(obj) is False
    assert ps.is_continuous(obj) is False
