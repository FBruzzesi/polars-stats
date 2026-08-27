"""`__repr__` contract: every distribution shows as a one-line constructor call."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

import polars_stats as ps
from tests.property._specs import ALL_SPECS

if TYPE_CHECKING:
    from tests.property._specs import DistSpec


@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: s.name)
def test_scalar_parameters_repr_as_their_values(spec: DistSpec) -> None:
    """`ClassName(name=value, ...)`, one line, one `name=value` per constant parameter."""
    dist = spec.make(spec.example)
    got = repr(dist)

    assert "\n" not in got
    assert got.startswith(f"{type(dist).__name__}(")
    assert got.endswith(")")

    assert dist._scalar_kwargs is not None
    for name, value in dist._scalar_kwargs.items():
        assert f"{name}={value}" in got


@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: s.name)
def test_column_parameters_repr_without_an_address(spec: DistSpec) -> None:
    """A column-valued parameter defers to the expression's own one-line form.

    `repr(pl.Expr)` carries the object's memory address, so the `0x` assertion keeps a
    column-parameterised distribution's repr deterministic.
    """
    dist = spec.make_columns(spec.example)
    got = repr(dist)

    assert "\n" not in got
    assert "0x" not in got
    assert got.startswith(f"{type(dist).__name__}(")
    assert got.endswith(")")


def test_documented_forms() -> None:
    """One exact string per kind of parameter a constructor accepts."""
    assert repr(ps.Normal(0.0, 1.0)) == "Normal(mu=0.0, sigma=1.0)"
    assert repr(ps.Normal()) == "Normal(mu=0.0, sigma=1.0)"
    assert repr(ps.Binomial(10, 0.5)) == "Binomial(n=10, p=0.5)"
    assert repr(ps.Normal(pl.col("m"), 1.0)) == 'Normal(mu=col("m"), sigma=1.0)'
    assert repr(ps.Normal("m", "s")) == 'Normal(mu=col("m"), sigma=col("s"))'
    assert repr(ps.Uniform(pl.Series("lo", [0.0, 1.0]), 2.0)) == "Uniform(min=Series[lo], max=2.0)"


def test_disagreeing_constructor_and_param_exprs_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    """A length mismatch between `__init__` and `_param_exprs` raises instead of mislabelling parameters."""
    dist = ps.Normal(0.0, 1.0)
    monkeypatch.setattr(type(dist), "_param_exprs", property(lambda self: (self._mu,)))

    with pytest.raises(ValueError, match="argument 2 is shorter"):
        repr(dist)


def test_no_value_equality_or_hash() -> None:
    """Instances keep identity semantics.

    Two instances parameterised by `pl.Expr` cannot be compared without evaluating them against a
    frame, so `__eq__` / `__hash__` are deliberately not defined.
    """
    left, right = ps.Normal(0.0, 1.0), ps.Normal(0.0, 1.0)

    assert left != right
    # Hashed by identity, so a set keeps both.
    assert len({left, right}) == len([left, right])
