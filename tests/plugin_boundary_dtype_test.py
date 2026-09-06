"""No dtype crossing the plugin boundary may kill the interpreter.

`pyo3-polars` rebuilds the incoming `Series` inside this crate's polars build before any of our Rust
runs; a dtype that build lacks panics there, and the panic crosses an `extern "C"` entry point and
becomes a non-unwinding abort that nothing in Python can catch. The `dtype-full` feature on the
`polars` dependency prevents it, and this module guards that feature.

Each case runs in a subprocess, because an abort would take pytest with it. The parent asserts the
child survived, reported all three shapes, and reached Rust in at least one of them: without the
last check a Python-side dtype guard could short-circuit every shape and pass vacuously.
"""

from __future__ import annotations

import subprocess
import sys
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import polars as pl
import pytest

import polars_stats as ps

if TYPE_CHECKING:
    from polars import Series

_REPO_ROOT = Path(__file__).parent.parent

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
"""The first seven abort without `dtype-full`; the temporal dtypes and `Object` are live on every build.

Constructors are deferred so a dtype the installed polars predates costs a skip, not a collection error.
"""

_SHAPES = ("value", "parameter", "sampler")
_REACHED_RUST = frozenset({"computed", "ComputeError"})


def _outcome(frame: pl.DataFrame, expr: pl.Expr) -> str:
    """`"computed"`, or the exception name: any catchable error is a pass, an abort is not catchable."""
    try:
        frame.select(r=expr)
    except Exception as err:  # noqa: BLE001
        return type(err).__name__
    return "computed"


def probe(dtype_name: str) -> None:
    """Cross the boundary in all three shapes for one dtype, printing each outcome. Runs in the child."""
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
    """The child survives all three crossings, and at least one of them reaches Rust."""
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-c", f"from {__name__} import probe; probe({dtype_name!r})"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    diagnosis = "aborted at the plugin boundary" if proc.returncode < 0 else "failed before the boundary"
    assert proc.returncode == 0, f"{dtype_name} {diagnosis} (exit {proc.returncode})\n{proc.stdout}{proc.stderr[-400:]}"

    outcomes = dict(line.split() for line in proc.stdout.splitlines())
    assert tuple(outcomes) == _SHAPES, f"{dtype_name} reported {tuple(outcomes)}, expected {_SHAPES}"
    assert _REACHED_RUST & set(outcomes.values()), f"{dtype_name} never reached Rust: {outcomes}"
