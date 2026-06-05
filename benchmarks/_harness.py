"""Shared comparison harness for the ``polars_stats`` vs ``scipy.stats`` sampling benchmarks.

Scope is deliberately narrow: the two sampling methods, `sample` (one variate per row) and `samples`
(``n_samples`` variates per row), against ``scipy.rvs``. These are the calls where `polars_stats` does real per-row
work, so they are the meaningful throughput and memory comparison.

`run.py` owns the `Comparison` registry (one entry per distribution) and the CLI; it calls `run_comparison` over a
`Sweep` of sizes. This module is *not* runnable; run ``uv run --group benchmarks benchmarks/run.py`` instead.

For each method and each side we report:

* **time**: median wall-clock over ``iterations`` runs, as ``p50 ± std`` (ms), warmup excluded;
* **peak memory**: peak resident-set growth (MiB) of one call, each measured in a fresh **isolated
  subprocess** where a background thread samples RSS, so it captures the native Rust/Arrow and NumPy
  allocations that `tracemalloc` would miss without the timing loop's allocator state skewing it.
"""

# ruff: noqa: T201
from __future__ import annotations

import gc
import json
import multiprocessing as mp
import platform
import threading
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl
import psutil  # type: ignore[import-untyped]
import scipy
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from scipy.stats._distn_infrastructure import rv_continuous_frozen, rv_discrete_frozen

    from polars_stats.distributions._base import _UnivariateDistribution

    # The frozen instance a spec passes (`norm(...)`, `binom(n, p)`, ...). Sampling-only, so we just call `rvs`;
    # typing against scipy's own frozen types rather than a hand-rolled Protocol keeps it honest against scipy-stubs.
    ScipyFrozen = rv_continuous_frozen | rv_discrete_frozen

    Distribution = _UnivariateDistribution
    """Any `polars_stats` distribution"""

    Method = Literal["sample", "samples"]
    Side = Literal["polars_stats", "scipy"]


OutputFormat = Literal["markdown", "json", "rich"]
"""How a report is emitted: a markdown file, a JSON file, or a rich table printed to the terminal."""

_MIB = 1024.0 * 1024.0
# Background memory sampler poll interval. Short enough to catch the transient peak of `samples`
# (`concat_arr` of `n_samples` columns before the final array), long enough not to peg a core.
_POLL_S = 5e-4


@dataclass(frozen=True)
class Comparison:
    """A distribution to benchmark: its `polars_stats` instance and the matching frozen scipy one.

    Arguments:
        name: Distribution id, used in the report header and as the CLI selector.
        dist: A `polars_stats` distribution instance with scalar parameters.
        scipy_frozen: The matching frozen `scipy.stats` distribution, reparameterised to scipy's
            convention by the caller (e.g. ``lognorm(s=sigma, scale=exp(mu))``).
    """

    name: str
    dist: Distribution
    scipy_frozen: ScipyFrozen


@dataclass(frozen=True)
class RunConfig:
    """One cell of the sweep: the sizes a single `sample` / `samples` measurement uses.

    Built only by `run_comparison` from an already-validated `Sweep`, so it carries no validation of its own.
    """

    rows: int = 1_000_000
    n_samples: int = 10
    iterations: int = 50
    seed: int = 0


@dataclass(frozen=True)
class Sweep:
    """A grid of sizes to benchmark in one report: each `rows`, crossed with each `n_samples` for `samples`.

    `sample` (one draw per row) ignores `n_samples`, so it is benchmarked once per `rows` value; `samples`
    is benchmarked over the full `rows` x `n_samples` product.
    """

    rows: tuple[int, ...] = (1_000_000,)
    n_samples: tuple[int, ...] = (10,)
    iterations: int = 50
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.rows or not self.n_samples:
            msg = "rows and n_samples must each contain at least one value"
            raise ValueError(msg)
        if min(*self.rows, *self.n_samples, self.iterations) < 1:
            msg = (
                "every rows / n_samples value and iterations must be >= 1; "
                f"got rows={self.rows}, n_samples={self.n_samples}, iterations={self.iterations}"
            )
            raise ValueError(msg)


@dataclass(frozen=True)
class Measurement:
    """One contender's cost for one method: median +/- std runtime (ms) and peak memory growth (MiB)."""

    p50_ms: float
    std_ms: float
    peak_mib: float


@dataclass(frozen=True)
class Result:
    method: str
    rows: int
    n_samples: int | None
    """None for `sample` method: one draw per row, so n_samples does not apply"""
    polars_stats: Measurement
    scipy: Measurement
    matches: bool

    @property
    def speedup(self) -> float:
        """Median scipy time over median polars_stats time: > 1 means polars_stats is faster."""
        return self.scipy.p50_ms / self.polars_stats.p50_ms if self.polars_stats.p50_ms > 0 else float("nan")


def _time(fn: Callable[[], object], *, iterations: int) -> tuple[float, float]:
    """Median and standard deviation of wall-clock time in ms over `iterations` runs, after one warmup.

    Median as the centre (robust to the occasional GC pause or scheduler hiccup); std as the reported
    spread, measured in its own loop so the memory sampler never perturbs the timing.
    """
    fn()  # warmup: pays first-touch allocation and lazy-import costs once, outside the timed loop
    samples = np.empty(iterations, dtype=np.float64)
    for i in range(iterations):
        start = time.perf_counter()
        fn()
        samples[i] = time.perf_counter() - start
    return float(np.median(samples)) * 1_000.0, float(np.std(samples)) * 1_000.0


def _peak_memory(fn: Callable[[], object]) -> float:
    """Peak resident-set growth in MiB during a single `fn()` call, over a gc-collected baseline.

    A daemon thread polls process RSS while `fn` runs; the result is held alive until the final reading
    so the output buffer counts toward the peak. RSS (not `tracemalloc`) so native polars/Arrow and
    NumPy allocations are included. Approximate by nature: the allocator may not release pages between
    contenders, so treat these as same-machine relative numbers, not absolute footprints.
    """
    proc = psutil.Process()
    gc.collect()
    baseline = proc.memory_info().rss
    peak = baseline
    stop = threading.Event()

    def poll() -> None:
        nonlocal peak
        while not stop.is_set():
            peak = max(peak, proc.memory_info().rss)
            time.sleep(_POLL_S)

    sampler = threading.Thread(target=poll, daemon=True)
    sampler.start()
    try:
        result = fn()
        peak = max(peak, proc.memory_info().rss)
    finally:
        stop.set()
        sampler.join()
    del result
    gc.collect()
    return max(0.0, (peak - baseline) / _MIB)  # type: ignore[no-any-return]


def _length_frame(rows: int) -> pl.LazyFrame:
    """A `rows`-long frame; the sampler expressions key off `pl.len()`, not its contents."""
    return pl.LazyFrame({"_": pl.Boolean()}).clear(rows)


def _build_fn(comp: Comparison, method: Method, config: RunConfig, side: Side) -> Callable[[], object]:
    """The single source of the timed/measured call, so timing, memory, and the shape check never drift.

    Returns a zero-arg closure. The polars side builds its length frame up front (outside the closure)
    so frame construction is not part of what is timed or charged to peak memory.
    """
    rows, n, seed = config.rows, config.n_samples, config.seed
    match side:
        case "scipy":
            frozen = comp.scipy_frozen
            size: int | tuple[int, int] = rows if method == "sample" else (rows, n)
            return lambda: frozen.rvs(size=size, random_state=seed)
        case "polars_stats":
            dist, lf = comp.dist, _length_frame(rows)
            expr = dist.sample(seed=seed) if method == "sample" else dist.samples(n, seed=seed)
            return lambda: lf.select(s=expr).collect(engine="streaming")
        case _:
            msg = "Unreachable path"
            raise AssertionError(msg)


def _shape_ok(method: Method, config: RunConfig, polars_out: object, scipy_out: object) -> bool:
    """Shape-only correctness gate: independent RNGs mean values cannot match, but shapes must."""
    assert isinstance(polars_out, pl.DataFrame)  # noqa: S101  # help the type checker
    assert isinstance(scipy_out, np.ndarray)  # noqa: S101  # help the type checker
    if method == "sample":
        return polars_out.height == config.rows and scipy_out.shape == (config.rows,)
    dtype = polars_out.to_series().dtype
    return (
        polars_out.height == config.rows
        and isinstance(dtype, pl.Array)
        and getattr(dtype, "size", None) == config.n_samples
        and scipy_out.shape == (config.rows, config.n_samples)
    )


def _mem_entry(queue: object, comp: Comparison, method: Method, config: RunConfig, side: Side) -> None:
    """Subprocess entrypoint: measure peak memory of one isolated call and post it back."""
    try:
        result = _peak_memory(_build_fn(comp, method, config, side))
        queue.put(("ok", result))  # type: ignore[attr-defined]
    except BaseException as exc:  # noqa: BLE001 - surface any child failure to the parent as a message
        queue.put(("err", repr(exc)))  # type: ignore[attr-defined]


def _peak_memory_isolated(comp: Comparison, method: Method, config: RunConfig, side: Side) -> float:
    """Peak memory of one call, measured in a fresh spawned process.

    Isolation is the point: an in-process measurement after the timing loop is unreliable because each
    library's allocator retains freed pages differently (scipy would read ~0 while polars re-allocates).
    A fresh process per call sidesteps that, at the cost of one interpreter start each.
    """
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_mem_entry, args=(queue, comp, method, config, side))
    proc.start()
    try:
        status, payload = queue.get(timeout=600)
    except Exception as exc:
        proc.terminate()
        proc.join()
        msg = f"memory subprocess for {comp.name}.{method} ({side}) did not report back"
        raise RuntimeError(msg) from exc
    proc.join()
    if status != "ok":
        msg = f"memory subprocess for {comp.name}.{method} ({side}) failed: {payload}"
        raise RuntimeError(msg)
    return float(payload)


def _measure(comp: Comparison, method: Method, config: RunConfig) -> Result:
    sides: tuple[Side, ...] = ("polars_stats", "scipy")
    fns = {side: _build_fn(comp, method, config, side) for side in sides}
    matches = _shape_ok(method, config, fns["polars_stats"](), fns["scipy"]())
    measurements = {
        side: Measurement(
            *_time(fns[side], iterations=config.iterations),
            peak_mib=_peak_memory_isolated(comp, method, config, side),
        )
        for side in sides
    }
    return Result(
        method=method,
        rows=config.rows,
        n_samples=config.n_samples if method == "samples" else None,
        polars_stats=measurements["polars_stats"],
        scipy=measurements["scipy"],
        matches=matches,
    )


def run_comparison(comp: Comparison, sweep: Sweep) -> list[Result]:
    """Benchmark `sample` (per `rows`) and `samples` (per `rows` x `n_samples`) across the sweep grid."""
    results = [
        _measure(comp, "sample", RunConfig(rows=rows, n_samples=1, iterations=sweep.iterations, seed=sweep.seed))
        for rows in sweep.rows
    ]
    results.extend(
        _measure(comp, "samples", RunConfig(rows=rows, n_samples=n, iterations=sweep.iterations, seed=sweep.seed))
        for rows in sweep.rows
        for n in sweep.n_samples
    )
    return results


def emit(
    comp: Comparison, results: Sequence[Result], sweep: Sweep, *, fmt: OutputFormat, output_dir: Path
) -> Path | None:
    """Render `results` in `fmt`. Markdown/JSON are written to ``output_dir/<name>.<ext>``; rich prints.

    Returns the written path for the file formats, or `None` for the terminal (`rich`) format.
    """
    if fmt == "rich":
        _render_rich(comp, results, sweep)
        return None

    renderer, ext = (_render_markdown, "md") if fmt == "markdown" else (_render_json, "json")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{comp.name}.{ext}"
    path.write_text(renderer(comp, results, sweep), encoding="utf-8")
    print(f"wrote {path}")
    return path


def _environment() -> dict[str, str]:
    """Version + platform stamp, so a saved report carries the context it was measured in."""
    return {
        "python": f"{platform.python_implementation()} {platform.python_version()}",
        "platform": platform.platform(),
        "polars": pl.__version__,
        "scipy": scipy.__version__,
        "numpy": np.__version__,
    }


def _header_lines(sweep: Sweep) -> list[str]:
    env = _environment()
    rows = ", ".join(f"{r:_}" for r in sweep.rows)
    draws = ", ".join(str(n) for n in sweep.n_samples)
    return [
        f"- rows: {rows}; `samples` draws per row: {draws}",
        f"- {sweep.iterations} iterations; time p50 +/- std (ms), memory peak RSS (MiB)",
        f"- {env['python']} on {env['platform']}",
        f"- polars {env['polars']}, scipy {env['scipy']}, numpy {env['numpy']}",
    ]


def _render_markdown(comp: Comparison, results: Sequence[Result], sweep: Sweep) -> str:
    """A markdown document (heading, environment stamp, table), ready to paste into the README / docs."""
    lines = [
        f"## {comp.name}: polars_stats vs scipy.stats",
        "",
        *_header_lines(sweep),
        "",
        "| method | rows | n_samples | polars_stats (ms) | scipy (ms) | speedup | polars_stats (MiB) | scipy (MiB) | match |",  # noqa: E501
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for r in results:
        flag = "ok" if r.matches else "**MISMATCH**"
        n_samples = "-" if r.n_samples is None else str(r.n_samples)
        lines.append(
            f"| `{r.method}` | {r.rows:_} | {n_samples} "
            f"| {r.polars_stats.p50_ms:.3f} +/- {r.polars_stats.std_ms:.3f} "
            f"| {r.scipy.p50_ms:.3f} +/- {r.scipy.std_ms:.3f} "
            f"| {r.speedup:.2f}x | {r.polars_stats.peak_mib:.1f} | {r.scipy.peak_mib:.1f} | {flag} |"
        )
    return "\n".join(lines) + "\n"


def _render_json(comp: Comparison, results: Sequence[Result], sweep: Sweep) -> str:
    """Machine-readable report: environment, sweep, and one object per (method, rows, n_samples) cell."""
    payload = {
        "distribution": comp.name,
        "environment": _environment(),
        "sweep": asdict(sweep),
        "results": [
            {
                "method": r.method,
                "rows": r.rows,
                "n_samples": r.n_samples,
                "polars_stats": asdict(r.polars_stats),
                "scipy": asdict(r.scipy),
                "speedup": r.speedup,
                "matches": r.matches,
            }
            for r in results
        ],
    }
    return json.dumps(payload, indent=2) + "\n"


def _render_rich(comp: Comparison, results: Sequence[Result], sweep: Sweep) -> None:
    """Print a coloured table to the terminal (speedup green when polars_stats wins, red otherwise)."""
    env = _environment()
    table = Table(
        title=f"{comp.name}: polars_stats vs scipy.stats",
        caption=(
            f"{sweep.iterations} iters | time p50 +/- std (ms), peak RSS (MiB) "
            f"| polars {env['polars']}, scipy {env['scipy']}, numpy {env['numpy']}"
        ),
    )
    table.add_column("method", style="bold cyan")
    table.add_column("rows", justify="right")
    table.add_column("n_samples", justify="right")
    for header in ("polars_stats (ms)", "scipy (ms)", "speedup", "polars_stats (MiB)", "scipy (MiB)"):
        table.add_column(header, justify="right")
    table.add_column("match", justify="center")
    for r in results:
        speedup_style = "green" if r.speedup >= 1.0 else "red"
        match = "[green]ok[/]" if r.matches else "[red]MISMATCH[/]"
        table.add_row(
            r.method,
            f"{r.rows:_}",
            "-" if r.n_samples is None else str(r.n_samples),
            f"{r.polars_stats.p50_ms:.3f} ± {r.polars_stats.std_ms:.3f}",
            f"{r.scipy.p50_ms:.3f} ± {r.scipy.std_ms:.3f}",
            f"[{speedup_style}]{r.speedup:.2f}x[/]",
            f"{r.polars_stats.peak_mib:.1f}",
            f"{r.scipy.peak_mib:.1f}",
            match,
        )
    Console().print(table)
