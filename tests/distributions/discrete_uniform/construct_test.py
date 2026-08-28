from __future__ import annotations

import pytest

from polars_stats import DiscreteUniform


@pytest.mark.parametrize("bad_bound", [None, True, False, 1.5, [1, 2], (1,), {"max": 5}])
@pytest.mark.parametrize("name", ["min", "max"])
def test_construct_invalid_type_raises(name: str, bad_bound: object) -> None:
    kwargs: dict[str, object] = {"min": 0, "max": 5}
    kwargs[name] = bad_bound
    with pytest.raises(TypeError, match=f"{name} should be an int or IntoExprColumn"):
        DiscreteUniform(**kwargs)  # type: ignore[arg-type]


def test_construct_negative_and_large_bounds_are_accepted() -> None:
    # Signed bounds are the point of this distribution: validity is judged at evaluation, never at
    # construction, so a negative or huge bound constructs fine.
    DiscreteUniform(min=-10, max=-2)
    DiscreteUniform(min=-(2**62), max=2**62)
    DiscreteUniform(min=-(2**63), max=2**63 - 1)


@pytest.mark.parametrize(("name", "bound"), [("max", 2**63), ("min", -(2**63) - 1)])
def test_construct_bound_outside_int64_raises_at_construction(name: str, bound: int) -> None:
    """A bound polars cannot hold as an `Int64` literal is refused when the literal is built."""
    kwargs: dict[str, int] = {"min": 0, "max": 5}
    kwargs[name] = bound
    with pytest.raises(ValueError, match=f"{name} must be in "):
        DiscreteUniform(**kwargs)
