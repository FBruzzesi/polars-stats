"""An oversized `samples(size=...)` raises `ComputeError`; it never answers wrongly and never kills the interpreter.

`size` has no maximum: the draw buffer is `rows * size` values, and `rows` is only known once the
expression runs, so only the plugin can refuse it. Two refusals are probed: a product that does not fit
a `usize`, and one the allocator will not give. Both drivers are probed too, since constant and column
parameters reach different ones.

Each case runs in a subprocess because an abort would take pytest down with it and a returned frame
would look like a pass. Neither touches real memory: the first never allocates and the second asks for
32 PiB. The band above that, where the allocator accepts what the machine cannot back, is an OS kill
in any library (numpy included) and is not tested.
"""

from __future__ import annotations

import subprocess
import sys

import polars as pl
import pytest

import polars_stats as ps

_SIZES = {"overflow": 2**62, "refused": 2**50}

_REGIMES = ("scalar", "column")
_ROWS = 4


def _outcome(frame: pl.DataFrame, expr: pl.Expr) -> str:  # pragma: no cover
    """`"refused"` for the driver's own error, `"returned"`, or the exception name; an abort never returns.

    `PanicException` subclasses `BaseException`, so `Exception` alone would let a panic end the child
    with exit 1 and read as the abort this guards against.
    """
    try:
        frame.select(r=expr)
    except pl.exceptions.ComputeError as err:
        return "refused" if "lower size" in str(err) else type(err).__name__
    except (Exception, pl.exceptions.PanicException) as err:  # noqa: BLE001
        return type(err).__name__
    return "returned"


def probe(size: int) -> None:  # pragma: no cover
    """Ask both drivers for `size` draws over `_ROWS` rows, printing one `<regime> <outcome>` line each."""
    frame = pl.DataFrame({"mu": [0.0] * _ROWS})
    exprs = (
        ps.Normal(mu=0.0, sigma=1.0).samples(size=size),
        ps.Normal(mu=pl.col("mu"), sigma=1.0).samples(size=size),
    )
    for regime, expr in zip(_REGIMES, exprs, strict=True):
        print(regime, _outcome(frame, expr))  # noqa: T201


@pytest.mark.parametrize("case", _SIZES, ids=str)
def test_an_unallocatable_size_raises_in_both_drivers(case: str) -> None:
    # Run this file as a script: `sys.path[0]` is then `tests/`, not the repo root, so `polars_stats`
    # resolves to the installed build and not to the `.so`-less source tree CI checks out.
    proc = subprocess.run(  # noqa: S603
        [sys.executable, __file__, str(_SIZES[case])],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    diagnosis = "aborted in the allocator" if proc.returncode < 0 else "failed before the driver"
    assert proc.returncode == 0, f"{case} {diagnosis} (exit {proc.returncode})\n{proc.stdout}{proc.stderr[-400:]}"

    outcomes = dict(line.split() for line in proc.stdout.splitlines() if line.startswith(_REGIMES))
    assert outcomes == dict.fromkeys(_REGIMES, "refused"), f"{case} reported {outcomes}"


if __name__ == "__main__":
    probe(int(sys.argv[1]))
