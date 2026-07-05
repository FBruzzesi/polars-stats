"""Shared comparison harness for the ``polars_stats`` vs ``scipy.stats`` benchmarks.

Two method families are compared:

* **sampling**: `sample` (one variate per row) and `samples` (``n_samples`` variates per row),
  against ``scipy.rvs``. Independent RNGs mean values cannot match, so correctness is a shape-only
  gate.
* **value-keyed**: `density` / `log_density` (`pdf` / `pmf` and `log_pdf` / `log_pmf` by family),
  `cdf`, `log_cdf`, `sf`, `log_sf` and `ppf`, against the matching frozen-scipy method. Both sides
  evaluate the *same* deterministic inputs (the distribution's own seeded draws; open-interval
  quantiles for `ppf`), so here the `match` column is a real ``np.allclose`` value check, not just
  a shape check.

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

from polars_stats.distributions._base import ContinuousDistribution

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from scipy.stats._distn_infrastructure import rv_continuous_frozen, rv_discrete_frozen

    from polars_stats.distributions._base import _UnivariateDistribution

    # The frozen instance a spec passes (`norm(...)`, `binom(n, p)`, ...). Typing against scipy's own frozen
    # types rather than a hand-rolled Protocol keeps it honest against scipy-stubs.
    ScipyFrozen = rv_continuous_frozen | rv_discrete_frozen

    Distribution = _UnivariateDistribution
    """Any `polars_stats` distribution"""

    Side = Literal["polars_stats", "scipy"]


Method = Literal[
    "sample",
    "samples",
    "density",
    "log_density",
    "cdf",
    "log_cdf",
    "sf",
    "log_sf",
    "ppf",
    "mean",
    "variance",
    "std",
    "entropy",
]
"""A benchmarkable method. `density` / `log_density` resolve to `pdf` / `log_pdf` (continuous) or
`pmf` / `log_pmf` (discrete) per distribution; `log_cdf` / `log_sf` call scipy's `logcdf` / `logsf`.

`mean` / `variance` / `std` / `entropy` are the parameter-only moments: they take no value column, so
polars_stats returns a length-`rows` column of the (constant, for scalar params) moment while scipy
returns a single scalar. The comparison is therefore inherently O(n) vs O(1); it exists to track the
polars_stats cost (time and peak memory) across a change, not to "beat" scipy on a scalar reduction.
"""

ALL_METHODS: tuple[Method, ...] = (
    "sample",
    "samples",
    "density",
    "log_density",
    "cdf",
    "log_cdf",
    "sf",
    "log_sf",
    "ppf",
    "mean",
    "variance",
    "std",
    "entropy",
)

_VALUE_METHODS: frozenset[Method] = frozenset({"density", "log_density", "cdf", "log_cdf", "sf", "log_sf", "ppf"})

_MOMENT_METHODS: frozenset[Method] = frozenset({"mean", "variance", "std", "entropy"})
"""Parameter-only moments: no value column, one expression evaluated over the length frame."""

OutputFormat = Literal["markdown", "json", "rich"]
"""How a report is emitted: a markdown file, a JSON file, or a rich table printed to the terminal."""

# Value-keyed match tolerances: statrs and scipy implement the same special functions with
# different algorithms, so agreement is to high relative precision, not bit-equality. The
# scipy-parity test suite owns the tight per-method bounds; this is a sanity gate.
_MATCH_RTOL = 1e-5
_MATCH_ATOL = 1e-12

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

    Every method except `samples` works one value per row and ignores `n_samples`, so it is
    benchmarked once per `rows` value; `samples` is benchmarked over the full `rows` x `n_samples`
    product.
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


def _density_name(dist: Distribution) -> Literal["pdf", "pmf"]:
    """The family-specific density method `density` resolves to (both as report label and call)."""
    return "pdf" if isinstance(dist, ContinuousDistribution) else "pmf"


def _report_name(dist: Distribution, method: Method) -> str:
    """The report label: the `polars_stats` method actually called (`density` -> `pdf` / `pmf`, ...)."""
    if method == "density":
        return _density_name(dist)
    if method == "log_density":
        return f"log_{_density_name(dist)}"
    return method


def _scipy_name(dist: Distribution, method: Method) -> str:
    """The frozen-scipy attribute for a value-keyed method (`log_cdf` -> `logcdf`, `log_density` -> `logpdf`)."""
    return _report_name(dist, method).replace("log_", "log")


def _eval_inputs(comp: Comparison, config: RunConfig, method: Method) -> np.ndarray:
    """Deterministic evaluation inputs for a value-keyed method, shared by both sides.

    Derived only from `(comp, config)` so the two sides (and the isolated memory subprocesses)
    regenerate the identical array. `ppf` gets uniform quantiles from `[0, 1)`; the other methods
    get the distribution's own seeded draws, which cover the support with realistic density.
    """
    if method == "ppf":
        return np.random.default_rng(config.seed).uniform(0.0, 1.0, size=config.rows)
    return np.asarray(comp.scipy_frozen.rvs(size=config.rows, random_state=config.seed), dtype=np.float64)


def _value_expr(dist: Distribution, method: Method) -> pl.Expr:
    """The `polars_stats` expression for a value-keyed method, evaluated on the shared `x` column."""
    if method not in _VALUE_METHODS:
        msg = f"not a value-keyed method: {method}"
        raise AssertionError(msg)
    # `_report_name` resolves `density` / `log_density` to the family-specific `pdf` / `log_pmf` / ...;
    # the other tokens are the method names themselves.
    return getattr(dist, _report_name(dist, method))(pl.col("x"))  # type: ignore[no-any-return]


def _build_fn(comp: Comparison, method: Method, config: RunConfig, side: Side) -> Callable[[], object]:
    """The single source of the timed/measured call, so timing, memory, and the match check never drift.

    Returns a zero-arg closure. Input construction (the length frame for samplers, the shared
    evaluation column for value-keyed methods) happens up front, outside the closure, so it is not
    part of what is timed or charged to peak memory.
    """
    rows, n, seed = config.rows, config.n_samples, config.seed
    match side:
        case "scipy":
            frozen = comp.scipy_frozen
            if method in _VALUE_METHODS:
                values = _eval_inputs(comp, config, method)
                fn = getattr(frozen, _scipy_name(comp.dist, method))
                return lambda: fn(values)
            if method in _MOMENT_METHODS:
                # scipy's variance is `var`; `mean` / `std` / `entropy` share the name. Each is one
                # scalar (O(1)), the baseline the length-n polars_stats moment is measured against.
                # The bound method is itself the zero-arg callable the timing loop invokes.
                return getattr(frozen, "var" if method == "variance" else method)  # type: ignore[no-any-return]
            size: int | tuple[int, int] = rows if method == "sample" else (rows, n)
            return lambda: frozen.rvs(size=size, random_state=seed)
        case "polars_stats":
            dist = comp.dist
            if method in _VALUE_METHODS:
                lf = pl.LazyFrame({"x": _eval_inputs(comp, config, method)})
                expr = _value_expr(dist, method)
            elif method in _MOMENT_METHODS:
                lf = _length_frame(rows)
                expr = getattr(dist, method)()
            else:
                lf = _length_frame(rows)
                expr = dist.sample(seed=seed) if method == "sample" else dist.samples(n, seed=seed)
            return lambda: lf.select(s=expr).collect(engine="streaming")
        case _:
            msg = "Unreachable path"
            raise AssertionError(msg)


def _matches(method: Method, config: RunConfig, polars_out: object, scipy_out: object) -> bool:
    """Correctness gate.

    * Samplers: shape-only (independent RNGs cannot match values).
    * Value-keyed methods evaluate the same inputs on both sides, so values must agree to `np.allclose` within
        the loose `_MATCH_RTOL`/`_MATCH_ATOL` (the scipy-parity suite owns the tight bounds).
    """
    assert isinstance(polars_out, pl.DataFrame)  # noqa: S101  # help the type checker
    if method in _MOMENT_METHODS:
        # polars_stats returns a length-`rows` column of the (constant) moment; scipy returns one
        # scalar. Gate on the column length and on every entry matching the scipy scalar.
        ours = polars_out.to_series().cast(pl.Float64).to_numpy()
        theirs = np.asarray(scipy_out, dtype=np.float64)
        return polars_out.height == config.rows and np.allclose(
            ours, theirs, rtol=_MATCH_RTOL, atol=_MATCH_ATOL, equal_nan=True
        )
    assert isinstance(scipy_out, np.ndarray)  # noqa: S101  # help the type checker
    if method == "samples":
        dtype = polars_out.to_series().dtype
        return (
            polars_out.height == config.rows
            and isinstance(dtype, pl.Array)
            and getattr(dtype, "size", None) == config.n_samples
            and scipy_out.shape == (config.rows, config.n_samples)
        )
    if polars_out.height != config.rows or scipy_out.shape != (config.rows,):
        return False
    if method == "sample":
        return True
    ours = polars_out.to_series().cast(pl.Float64).to_numpy()
    theirs = np.asarray(scipy_out, dtype=np.float64)
    return np.allclose(ours, theirs, rtol=_MATCH_RTOL, atol=_MATCH_ATOL, equal_nan=True)


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
    matches = _matches(method, config, fns["polars_stats"](), fns["scipy"]())
    measurements = {
        side: Measurement(
            *_time(fns[side], iterations=config.iterations),
            peak_mib=_peak_memory_isolated(comp, method, config, side),
        )
        for side in sides
    }
    return Result(
        # `density` / `log_density` are the sweep tokens; the report shows the method actually called.
        method=_report_name(comp.dist, method),
        rows=config.rows,
        n_samples=config.n_samples if method == "samples" else None,
        polars_stats=measurements["polars_stats"],
        scipy=measurements["scipy"],
        matches=matches,
    )


def run_comparison(comp: Comparison, sweep: Sweep, methods: Sequence[Method] = ALL_METHODS) -> list[Result]:
    """Benchmark each requested method across the sweep grid.

    Every method is measured once per `rows` value; `samples` additionally crosses `rows` with each
    `n_samples` width.
    """
    results: list[Result] = []
    for method in methods:
        widths = sweep.n_samples if method == "samples" else (1,)
        results.extend(
            _measure(comp, method, RunConfig(rows=rows, n_samples=n, iterations=sweep.iterations, seed=sweep.seed))
            for rows in sweep.rows
            for n in widths
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
