"""No dtype crossing the plugin boundary may kill the interpreter.

`pyo3-polars` rebuilds the incoming `Series` inside this crate's polars build before our Rust runs. A
dtype that build lacks panics there, and the panic crosses the `extern "C"` entry point as an abort
nothing in Python can catch. The `dtype-full` feature on the `polars` dependency prevents it; this
module pins that feature.

Each case runs in a subprocess because an abort would take pytest down with it. The parent checks
that the child survived, reported all three shapes, and reached Rust in at least one of them:
without the last check a Python-side dtype guard could pass every shape vacuously.
"""

from __future__ import annotations

import subprocess
import sys
from decimal import Decimal
from typing import TYPE_CHECKING, Callable

import polars as pl
import pytest

import polars_stats as ps

if TYPE_CHECKING:
    from polars import Series

_SERIES: dict[str, Callable[[], Series]] = {
    "Int128": lambda: pl.Series("x", [1, 2], dtype=pl.Int128),
    "UInt128": lambda: pl.Series("x", [1, 2], dtype=pl.UInt128),
    "Float16": lambda: pl.Series("x", [1.0, 2.0], dtype=pl.Float16),
    "Decimal": lambda: pl.Series("x", [Decimal("0.50"), Decimal("1.00")], dtype=pl.Decimal(10, 2)),
    "Categorical": lambda: pl.Series("x", ["a", "b"], dtype=pl.Categorical),
    "Enum": lambda: pl.Series("x", ["a", "b"], dtype=pl.Enum(["a", "b"])),
    "Struct": lambda: pl.Series("x", [{"a": 1}, {"a": 2}], dtype=pl.Struct({"a": pl.Int64})),
    "Date": lambda: pl.Series("x", [1, 2], dtype=pl.Int32).cast(pl.Date),
    "Datetime": lambda: pl.Series("x", [1, 2], dtype=pl.Int64).cast(pl.Datetime("us")),
    "Duration": lambda: pl.Series("x", [1, 2], dtype=pl.Int64).cast(pl.Duration("us")),
    "Time": lambda: pl.Series("x", [1, 2], dtype=pl.Int64).cast(pl.Time),
    "Object": lambda: pl.Series("x", [object(), object()], dtype=pl.Object),
}
"""The first seven abort without `dtype-full`; the temporal dtypes and `Object` never did.

Deferred so `pl.Int128` and friends are only touched on a polars that has them; the parametrize
filter drops the rest.
"""

_SHAPES = ("value", "parameter", "sampler")
_REACHED_RUST = frozenset({"computed", "ComputeError"})


def _outcome(frame: pl.DataFrame, expr: pl.Expr) -> str:  # pragma: no cover
    """`"computed"` or the exception name; any catchable error passes, an abort never returns."""
    try:
        frame.select(r=expr)
    except Exception as err:  # noqa: BLE001
        return type(err).__name__
    return "computed"


def probe(dtype_name: str) -> None:  # pragma: no cover
    """Cross the boundary in all three shapes for one dtype, printing one `<shape> <outcome>` line each."""
    frame = _SERIES[dtype_name]().to_frame()
    exprs = (
        ps.Normal(0.0, 1.0).pdf(pl.col("x")),
        ps.Normal(mu=pl.col("x"), sigma=1.0).mean(),
        ps.Normal(mu=pl.col("x"), sigma=1.0).sample(seed=0),
    )
    for shape, expr in zip(_SHAPES, exprs, strict=True):
        print(shape, _outcome(frame, expr))  # noqa: T201


@pytest.mark.parametrize("dtype_name", [name for name in _SERIES if hasattr(pl, name)], ids=str)
def test_plugin_boundary_does_not_abort(dtype_name: str) -> None:
    # Run this file as a script: `sys.path[0]` is then `tests/`, not the repo root, so `polars_stats`
    # resolves to the installed build and not to the `.so`-less source tree CI checks out.
    proc = subprocess.run(  # noqa: S603
        [sys.executable, __file__, dtype_name],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    diagnosis = "aborted at the plugin boundary" if proc.returncode < 0 else "failed before the boundary"
    assert proc.returncode == 0, f"{dtype_name} {diagnosis} (exit {proc.returncode})\n{proc.stdout}{proc.stderr[-400:]}"

    outcomes = dict(line.split() for line in proc.stdout.splitlines() if line.startswith(_SHAPES))
    assert tuple(outcomes) == _SHAPES, f"{dtype_name} reported {tuple(outcomes)}, expected {_SHAPES}"
    assert _REACHED_RUST & set(outcomes.values()), f"{dtype_name} never reached Rust: {outcomes}"


if __name__ == "__main__":
    probe(sys.argv[1])
