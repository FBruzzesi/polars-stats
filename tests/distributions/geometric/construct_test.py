from __future__ import annotations

import pytest

from polars_stats import Geometric


@pytest.mark.parametrize("bad_p", [None, True, False, 1, 0, [0.5, 0.5], (0.5,), {"p": 0.5}])
def test_construct_invalid_type_raises(bad_p: object) -> None:
    with pytest.raises(TypeError, match="p should be a float or IntoExprColumn"):
        Geometric(p=bad_p)  # type: ignore[arg-type]
