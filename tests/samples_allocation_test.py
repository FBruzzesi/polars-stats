"""An oversized `samples(size=...)` raises `ComputeError`; it never answers wrongly and never kills the interpreter.

Only the plugin can refuse an oversized `size`, since the buffer is `rows * size` and the row count is
a runtime fact. Each case asserts the wording of its own guard, so neither can stand in for the other.
Each runs in a subprocess because an abort would take pytest down with it and a returned frame would
look like a pass. Neither touches real memory: the first never allocates and the second asks for 32
PiB. The band above, where the allocator accepts what the machine cannot back, is an OS kill in any
library and is not tested.
"""

from __future__ import annotations

import subprocess
import sys

import polars as pl
import pytest

import polars_stats as ps

_ROWS = 4

_CASES = {
    "usize_overflow": (2**62, "a usize can address"),
    "allocator_refusal": (2**50, "cannot be allocated"),
}
"""Per guard: a `size` that trips it, and the fragment of the message only it emits.

`usize_overflow` reaches the checked multiply only while `_ROWS * 2**62` exceeds `usize::MAX`. Lower
`_ROWS` and the product fits, the allocator refuses instead, and the multiply goes unexercised.
"""

_REGIMES = ("scalar", "column")


def _outcome(frame: pl.DataFrame, expr: pl.Expr, expected: str) -> str:  # pragma: no cover
    """`"refused"` for this case's own guard, `"returned"`, or the exception name; an abort never returns.

    `PanicException` subclasses `BaseException`, so `Exception` alone would let a panic end the child
    with exit 1 and read as the abort this guards against.
    """
    try:
        frame.select(r=expr)
    except pl.exceptions.ComputeError as err:
        return "refused" if expected in str(err) else f"ComputeError({err})"
    except (Exception, pl.exceptions.PanicException) as err:  # noqa: BLE001
        return type(err).__name__
    return "returned"


def probe(size: int, expected: str) -> None:  # pragma: no cover
    """Ask both drivers for `size` draws over `_ROWS` rows, printing one `<regime> <outcome>` line each."""
    frame = pl.DataFrame({"mu": [0.0] * _ROWS})
    exprs = (
        ps.Normal(mu=0.0, sigma=1.0).samples(size=size),
        ps.Normal(mu=pl.col("mu"), sigma=1.0).samples(size=size),
    )
    for regime, expr in zip(_REGIMES, exprs, strict=True):
        print(regime, _outcome(frame, expr, expected), flush=True)  # noqa: T201


@pytest.mark.parametrize("case", _CASES, ids=str)
def test_an_oversized_size_is_refused_by_both_drivers(case: str) -> None:
    size, expected = _CASES[case]
    # Run this file as a script: `sys.path[0]` is then `tests/`, not the repo root, so `polars_stats`
    # resolves to the installed build and not to the `.so`-less source tree CI checks out.
    proc = subprocess.run(  # noqa: S603
        [sys.executable, __file__, str(size), expected],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    diagnosis = "aborted in the allocator" if proc.returncode < 0 else "failed before the driver"
    assert proc.returncode == 0, f"{case} {diagnosis} (exit {proc.returncode})\n{proc.stdout}{proc.stderr[-400:]}"

    outcomes = dict(line.split(maxsplit=1) for line in proc.stdout.splitlines() if line.startswith(_REGIMES))
    assert outcomes == dict.fromkeys(_REGIMES, "refused"), f"{case} reported {outcomes}"


if __name__ == "__main__":
    probe(int(sys.argv[1]), sys.argv[2])
