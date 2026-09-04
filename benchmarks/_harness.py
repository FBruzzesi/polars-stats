"""Comparison harness for the ``polars_stats`` vs ``scipy.stats`` benchmarks.

Every cell is measured in one of three parameter `Regime`s. They are different code paths on both sides and
**must never be cross-compared**: a column cell does strictly more work per row by design.

The regime is part of `Case.id` and a column of every report so the two cannot be diffed by accident.

This module is not runnable; run ``uv run --group benchmarks benchmarks/run.py`` instead.
"""

# ruff: noqa: T201
# pyright: reportUnknownParameterType=false
# `ScipyFrozen` aliases scipy-stubs' generic frozen types, whose own arguments are partly Unknown.
# `[tool.pyright] include` keeps benchmarks/ out of `make typing`, but an editor still checks it.
from __future__ import annotations

import gc
import json
import multiprocessing as mp
import pickle
import platform
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Annotated, Literal, NamedTuple, Protocol, TypedDict, TypeVar, cast, get_args

import numpy as np
import polars as pl
import psutil  # type: ignore[import-untyped]
import scipy
from cyclopts import Parameter
from rich.console import Console
from rich.table import Table

from polars_stats import _internal
from polars_stats.distributions._base import ContinuousDistribution

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence
    from pathlib import Path
    from typing import NoReturn, TypeAlias

    import numpy.typing as npt
    from scipy.stats._distn_infrastructure import rv_continuous_frozen, rv_discrete_frozen
    from typing_extensions import TypeIs

    from polars_stats.distributions._base import _UnivariateDistribution

    ScipyFrozen: TypeAlias = rv_continuous_frozen | rv_discrete_frozen
    Distribution: TypeAlias = _UnivariateDistribution
    Array: TypeAlias = "npt.NDArray[np.float64] | npt.NDArray[np.int64]"

    Outcome: TypeAlias = "pl.DataFrame | Array | np.float64 | float"
    """What one contender's call returns: a collected frame, an array, or a single moment."""

    Call: TypeAlias = Callable[[], "Outcome"]
    DistFactory: TypeAlias = Callable[["Params"], tuple[Distribution, ScipyFrozen]]
    MemoryReply: TypeAlias = "_PeakMiB | _ChildFailure"


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
    "isf",
    "mean",
    "variance",
    "std",
    "entropy",
]

MethodKind = Literal["sampler", "value", "moment"]
"""Which family a method belongs to: how a contender builds its call, and how `matches` is gated.

`sampler` is gated on shape alone, since independent RNGs cannot agree on values. `value` evaluates a
shared input column on both sides, so values must agree. `moment` takes no input column, so its
output height follows the regime.
"""

ValueInput = Literal["support", "quantile"]
DensityToken = Literal["pdf", "pmf"]
OutputFormat = Literal["markdown", "json", "rich"]
Regime = Literal["scalar", "column", "broadcast"]
"""How a distribution's parameters are supplied: Python numbers, `pl.col`, or `pl.lit`.

Three different code paths on both sides, never cross-compared. `README.md` has the table.
"""

ALL_REGIMES: tuple[Regime, ...] = get_args(Regime)

_DENSITY = "{d}"
"""Stands in for the family-specific density token in a `MethodSpec` attribute name."""

_VALUE_COLUMN = "x"
"""Frame column holding the shared evaluation inputs. A parameter may not take this name."""

_PARAM_STREAM = 0x9E37
"""Second seed word for the parameter draws, keeping them off the evaluation-input stream."""

REFERENCE_CONTENDER = "polars_stats"
"""The contender every ratio is taken against, and the one `matches` compares the others to."""

SCHEMA_VERSION = 3
"""`json` report schema. 3 nested the per-contender numbers under `measurements` and stamps
`methods`; 2 added the regime axis, per-cell iteration counts and nullable peak memory."""

# statrs and scipy implement the same special functions with different algorithms, so agreement is to
# high relative precision, not bit-equality. The scipy-parity suite owns the tight per-method bounds.
_MATCH_RTOL = 1e-5
_MATCH_ATOL = 1e-12

_MIB = 1024.0 * 1024.0

_POLL_INTERVAL_S = 5e-4
"""Short enough to catch the transient peak of `samples`, long enough not to peg a core."""

_CHILD_REPLY_TIMEOUT_S = 600.0
_CHILD_JOIN_TIMEOUT_S = 30.0

_Axis = TypeVar("_Axis", bound="int | str")


def _unreachable(kind: str) -> NoReturn:
    msg = f"unhandled MethodSpec kind: {kind!r}"
    raise AssertionError(msg)


@dataclass(frozen=True)
class MethodSpec:
    """One row of the method metadata table: what to call on either side, and how to gate it.

    Both attribute names are `{d}`-templated for the density family, resolved against the
    distribution's own `pdf` / `pmf` token. `plugin_attr` doubles as the report label.
    """

    kind: MethodKind
    plugin_attr: str
    scipy_attr: str
    value_input: ValueInput | None = None
    multi_draw: bool = False


METHOD_SPECS: Mapping[Method, MethodSpec] = {
    "sample": MethodSpec(kind="sampler", plugin_attr="sample", scipy_attr="rvs"),
    "samples": MethodSpec(kind="sampler", plugin_attr="samples", scipy_attr="rvs", multi_draw=True),
    "density": MethodSpec(kind="value", plugin_attr=_DENSITY, scipy_attr=_DENSITY, value_input="support"),
    "log_density": MethodSpec(
        kind="value", plugin_attr=f"log_{_DENSITY}", scipy_attr=f"log{_DENSITY}", value_input="support"
    ),
    "cdf": MethodSpec(kind="value", plugin_attr="cdf", scipy_attr="cdf", value_input="support"),
    "log_cdf": MethodSpec(kind="value", plugin_attr="log_cdf", scipy_attr="logcdf", value_input="support"),
    "sf": MethodSpec(kind="value", plugin_attr="sf", scipy_attr="sf", value_input="support"),
    "log_sf": MethodSpec(kind="value", plugin_attr="log_sf", scipy_attr="logsf", value_input="support"),
    "ppf": MethodSpec(kind="value", plugin_attr="ppf", scipy_attr="ppf", value_input="quantile"),
    "isf": MethodSpec(kind="value", plugin_attr="isf", scipy_attr="isf", value_input="quantile"),
    "mean": MethodSpec(kind="moment", plugin_attr="mean", scipy_attr="mean"),
    "variance": MethodSpec(kind="moment", plugin_attr="variance", scipy_attr="var"),
    "std": MethodSpec(kind="moment", plugin_attr="std", scipy_attr="std"),
    "entropy": MethodSpec(kind="moment", plugin_attr="entropy", scipy_attr="entropy"),
}
"""What a method is and what to call on either side. Insertion order is report order."""

ALL_METHODS: tuple[Method, ...] = tuple(METHOD_SPECS)

if set(METHOD_SPECS) != set(get_args(Method)):
    # A token without a row would only fail deep inside a sweep.
    _msg = (
        "METHOD_SPECS and Method are out of step: "
        f"{sorted(set(get_args(Method)) - set(METHOD_SPECS))} lack a row, "
        f"{sorted(set(METHOD_SPECS) - set(get_args(Method)))} lack a token"
    )
    raise RuntimeError(_msg)


@dataclass(frozen=True)
class ParamSpec:
    """How to realise one parameter: a fixed value, plus the inclusive domain `column` draws from.

    `[low, high]` must be chosen so that *every* draw is a valid parameterisation, including any joint
    constraint: an ordered pair like `min < max` is expressed as two non-overlapping domains rather
    than as a rejection step, so realisation never has to retry. Holding `scalar` fixed is what makes
    `broadcast` an honest comparison against `scalar`: same numbers, different path.
    """

    scalar: float | int
    low: float
    high: float
    integer: bool = False

    def __post_init__(self) -> None:
        if self.low > self.high:
            msg = f"domain is empty: low={self.low} > high={self.high}"
            raise ValueError(msg)
        if not self.low <= self.scalar <= self.high:
            msg = f"scalar={self.scalar} lies outside its own domain [{self.low}, {self.high}]"
            raise ValueError(msg)

    def draw(self, rng: np.random.Generator, rows: int) -> Array:
        """One `column`-regime array of `rows` valid values."""
        if self.integer:
            return rng.integers(int(self.low), int(self.high) + 1, size=rows, dtype=np.int64)
        return rng.uniform(self.low, self.high, size=rows).astype(np.float64, copy=False)

    @property
    def fixed(self) -> Array:
        """The length-1 array the `scalar` and `broadcast` regimes share."""
        if self.integer:
            return np.array([self.scalar], dtype=np.int64)
        return np.array([self.scalar], dtype=np.float64)


def _only_value(values: Array) -> float | int:
    return int(values[0]) if np.issubdtype(values.dtype, np.integer) else float(values[0])


@dataclass(frozen=True)
class Params:
    """One regime's realised parameters, in the two forms a `DistFactory` needs.

    `plugin` and `scipy` are the same numbers viewed differently, so a factory spells each parameter
    once per side and stays regime-agnostic.
    """

    regime: Regime
    values: Mapping[str, Array]
    """Per parameter: length `rows` under `column`, length 1 otherwise."""

    def plugin(self, name: str) -> float | int | pl.Expr:
        """The parameter as `polars_stats` should receive it, which is what selects the code path.

        A Python number routes to the constant-parameter fast path; `pl.lit` gives a length-1 input
        the plugin broadcasts; `pl.col` gives the per-row path.
        """
        if self.regime == "column":
            return pl.col(name)
        value = _only_value(self.values[name])
        return pl.lit(value) if self.regime == "broadcast" else value

    def plugin_int(self, name: str) -> int | pl.Expr:
        """`plugin` narrowed for the constructors that require an `int`.

        The check is on the realised dtype, so a `ParamSpec` missing `integer=True` raises here in
        every regime rather than deep inside the plugin.
        """
        if not np.issubdtype(self.values[name].dtype, np.integer):
            msg = f"{name!r} realised as a float; declare it as ParamSpec(..., integer=True)"
            raise TypeError(msg)
        value = self.plugin(name)
        return value if isinstance(value, (int, pl.Expr)) else int(value)

    def scipy(self, name: str) -> float | int | Array:
        """The parameter as frozen scipy should receive it: a scalar, or the array to broadcast."""
        return _only_value(self.values[name]) if self.regime == "scalar" else self.values[name]

    @property
    def columns(self) -> dict[str, Array]:
        """The parameter columns the frame must carry, which is only the `column` regime's."""
        return dict(self.values) if self.regime == "column" else {}


@dataclass(frozen=True)
class Comparison:
    """A distribution to benchmark: one `ParamSpec` per parameter, and a factory building both sides.

    `build` turns realised `Params` into `(polars_stats instance, frozen scipy)`, applying scipy's
    reparameterisation (e.g. ``lognorm(s=sigma, scale=exp(mu))``). It must be a module-level function:
    a `Comparison` is pickled into the memory subprocess, so a lambda or closure would not survive.
    """

    name: str
    params: Mapping[str, ParamSpec]
    build: DistFactory
    density_token: DensityToken = field(init=False)
    """`pdf` or `pmf`, resolved once here rather than per cell: resolving it builds the pair."""

    def __post_init__(self) -> None:
        if _VALUE_COLUMN in self.params:
            msg = f"{self.name}: a parameter may not be named {_VALUE_COLUMN!r}"
            raise ValueError(msg)
        try:
            pickle.dumps(self.build)
        except (AttributeError, TypeError, pickle.PicklingError) as exc:
            msg = f"{self.name}: build must be picklable to reach the memory subprocess ({exc})"
            raise TypeError(msg) from exc

        dist, _ = self.build(_draw_params(self, regime="scalar", rows=1, seed=0))
        token: DensityToken = "pdf" if isinstance(dist, ContinuousDistribution) else "pmf"
        object.__setattr__(self, "density_token", token)

    def label(self, method: Method) -> str:
        """The report label for `method`: the `polars_stats` attribute actually called."""
        return METHOD_SPECS[method].plugin_attr.format(d=self.density_token)


@dataclass(frozen=True)
class Budget:
    """How hard to measure one cell: whichever limit binds first.

    A cell whose *single* call already costs more than `max_seconds` is measured exactly once, since
    `min_iterations` of those would overshoot the budget several times over.
    """

    max_iterations: Annotated[int, Parameter(help="Upper bound on timed runs per cell.")] = 50
    max_seconds: Annotated[float, Parameter(help="Wall-clock budget for one cell's timed loop.")] = 30.0
    min_iterations: Annotated[int, Parameter(help="Lower bound on timed runs per cell.")] = 3

    def __post_init__(self) -> None:
        if self.min_iterations < 1 or self.max_iterations < self.min_iterations:
            msg = (
                "need 1 <= min_iterations <= max_iterations; "
                f"got min_iterations={self.min_iterations}, max_iterations={self.max_iterations}"
            )
            raise ValueError(msg)
        if not np.isfinite(self.max_seconds) or self.max_seconds <= 0:
            msg = f"max_seconds must be finite and > 0, got {self.max_seconds}"
            raise ValueError(msg)


@dataclass(frozen=True)
class Case:
    """One measurable cell: which distribution, which method, which regime, at which sizes.

    Frozen, hashable and picklable on purpose. Picklable is load-bearing: `_isolated_peak_mib` ships
    a case to a spawned subprocess, which rebuilds the call there from the case plus a contender
    *name*. A built closure could not cross that boundary.

    `n_samples` is the draws-per-row width for the one method that has one, and `None` for the rest.
    One `seed` covers the samplers, the evaluation inputs and the parameter draws.
    """

    distribution: str
    method: Method
    regime: Regime
    rows: int
    n_samples: int | None
    seed: int

    @property
    def spec(self) -> MethodSpec:
        return METHOD_SPECS[self.method]

    @property
    def draws_per_row(self) -> int:
        """The draws-per-row width, for the one method that has one."""
        if self.n_samples is None:
            msg = f"{self.method} has no draws axis, so no width"
            raise AssertionError(msg)
        return self.n_samples

    @property
    def id(self) -> str:
        """A stable label for this cell, used in report diffs and subprocess error messages."""
        draws = "" if self.n_samples is None else f",draws={self.n_samples}"
        return f"{self.distribution}.{self.method}[{self.regime},rows={self.rows}{draws},seed={self.seed}]"


def _distinct(axis: str, values: tuple[_Axis, ...]) -> tuple[_Axis, ...]:
    if len(set(values)) != len(values):
        msg = f"{axis} contains a repeated value, which would measure the same cell twice: {list(values)}"
        raise ValueError(msg)
    return values


_MULTI = Parameter(consume_multiple=True, negative_iterable="")
"""`--rows 1 2 3`, not `--rows 1 --rows 2`, and no `--empty-rows` counterpart."""


@dataclass(frozen=True)
class Sweep:
    """A grid to benchmark in one report: regimes x methods x `rows`, crossed with `n_samples`.

    Every method except `samples` works one value per row and ignores `n_samples`, so it is
    benchmarked once per `rows` value; `samples` runs the full `rows` x `n_samples` product.

    This is also the CLI's option surface: `run.py` flattens it, so the defaults here are the
    defaults a bare `benchmarks/run.py` uses.
    """

    rows: Annotated[tuple[int, ...], _MULTI, Parameter(help="Row counts to sweep over.")] = (1_000_000,)
    n_samples: Annotated[tuple[int, ...], _MULTI, Parameter(help="Draws-per-row widths for `samples`.")] = (10,)
    regimes: Annotated[tuple[Regime, ...], _MULTI, Parameter(help="Parameter regimes to compare.")] = ALL_REGIMES
    methods: Annotated[tuple[Method, ...], _MULTI, Parameter(help="Methods to compare.")] = ALL_METHODS
    budget: Annotated[Budget, Parameter(name="*")] = field(default_factory=Budget)
    seed: Annotated[int, Parameter(help="Seed for the samplers, evaluation inputs and parameter draws.")] = 0

    def __post_init__(self) -> None:
        if not self.rows or not self.n_samples or not self.regimes or not self.methods:
            msg = "rows, n_samples, regimes and methods must each contain at least one value"
            raise ValueError(msg)
        if any(value < 1 for value in (*self.rows, *self.n_samples)):
            msg = f"every rows / n_samples value must be >= 1; got rows={self.rows}, n_samples={self.n_samples}"
            raise ValueError(msg)
        for axis, values in (
            ("rows", self.rows),
            ("n_samples", self.n_samples),
            ("regimes", self.regimes),
            ("methods", self.methods),
        ):
            _distinct(axis, values)

    def cases(self, distribution: str) -> Iterator[Case]:
        """Expand this sweep into one `Case` per cell, in report order.

        Regime is the outermost axis, so each regime's block is contiguous and internally ordered
        exactly as a single-regime run would be.
        """
        for regime in self.regimes:
            for method in self.methods:
                widths: tuple[int | None, ...] = self.n_samples if METHOD_SPECS[method].multi_draw else (None,)
                for rows in self.rows:
                    for width in widths:
                        yield Case(
                            distribution=distribution,
                            method=method,
                            regime=regime,
                            rows=rows,
                            n_samples=width,
                            seed=self.seed,
                        )


class Timing(NamedTuple):
    """Median and sample standard deviation (ms) of a timed loop, and how many runs it managed."""

    p50_ms: float
    std_ms: float
    iterations: int


class MeasurementPayload(TypedDict):
    """One contender's numbers as they appear in a `json` report."""

    p50_ms: float
    std_ms: float | None
    iterations: int
    peak_mib: float | None


@dataclass(frozen=True)
class Measurement:
    """One contender's cost for one cell: median +/- std runtime (ms) over `iterations` runs.

    `std_ms` is `nan` for a single-iteration cell, which the reports render as `-` rather than a
    flattering zero. `peak_mib` is `None` unless peak memory was requested, since measuring it costs
    a subprocess.
    """

    p50_ms: float
    std_ms: float
    iterations: int
    peak_mib: float | None = None

    @property
    def payload(self) -> MeasurementPayload:
        """This measurement as a `json` report object, with `nan` nulled out to keep it parseable."""
        payload = cast("MeasurementPayload", asdict(self))
        if np.isnan(self.std_ms):
            payload["std_ms"] = None
        return payload


@dataclass(frozen=True)
class Result:
    """One case's outcome: a `Measurement` per contender, plus the correctness verdict.

    Keyed by contender name rather than holding a fixed pair of fields, so a ratio is always taken
    explicitly against a named reference.
    """

    case: Case
    label: str
    """The report label: the `polars_stats` method actually called (`density` -> `pdf` / `pmf`)."""
    measurements: Mapping[str, Measurement]
    matches: bool

    def ratio(self, contender: str, *, reference: str = REFERENCE_CONTENDER) -> float:
        """`contender`'s median time over `reference`'s: > 1 means the reference is faster."""
        base = self.measurements[reference].p50_ms
        return self.measurements[contender].p50_ms / base if base > 0 else float("nan")


class Contender(Protocol):
    """Builds the zero-argument call that one competitor runs for one case.

    A builder, not a prepared call: input construction happens when the builder runs, so it is
    outside the returned closure and charged to neither time nor peak memory. Registered by name in
    `CONTENDERS`, because the name is what crosses into the memory subprocess, not the closure.
    """

    def __call__(self, comp: Comparison, case: Case, /) -> Call: ...


def _time_call(call: Call, budget: Budget) -> tuple[Outcome, Timing]:
    """Time `call` under `budget`, returning the warmup's output for the correctness gate.

    The warmup is excluded from the reported numbers, and its duration plans the iteration count. A
    call that already costs more than the whole budget is measured exactly once rather than
    `min_iterations` times.
    """
    start = time.perf_counter()
    outcome = call()
    warmup_s = time.perf_counter() - start

    if warmup_s > budget.max_seconds:
        planned = 1
    else:
        affordable = budget.max_iterations if warmup_s <= 0.0 else int(budget.max_seconds / warmup_s)
        planned = max(budget.min_iterations, min(budget.max_iterations, affordable))

    durations_s = np.empty(planned, dtype=np.float64)
    deadline = time.perf_counter() + budget.max_seconds
    measured = 0
    for index in range(planned):
        call_start = time.perf_counter()
        call()
        durations_s[index] = time.perf_counter() - call_start
        measured = index + 1
        if measured >= budget.min_iterations and time.perf_counter() >= deadline:
            break

    timed_s = durations_s[:measured]
    std_s = float(np.std(timed_s, ddof=1)) if measured > 1 else float("nan")
    return outcome, Timing(float(np.median(timed_s)) * 1_000.0, std_s * 1_000.0, measured)


def _peak_mib(call: Call) -> float:
    """Peak resident-set growth in MiB during a single `call()`, over a gc-collected baseline.

    The result is held alive until the final reading so the output buffer counts toward the peak.
    Same-machine relative numbers, not absolute footprints: the allocator may not release pages
    between contenders.
    """
    proc = psutil.Process()
    gc.collect()
    baseline: int = proc.memory_info().rss
    peak: int = baseline
    stop = threading.Event()

    def poll() -> None:
        nonlocal peak
        while not stop.is_set():
            peak = max(peak, proc.memory_info().rss)
            time.sleep(_POLL_INTERVAL_S)

    sampler = threading.Thread(target=poll, daemon=True)
    sampler.start()
    try:
        result = call()
    finally:
        stop.set()
        sampler.join()
    peak = max(peak, proc.memory_info().rss)
    del result
    gc.collect()
    return max(0.0, (peak - baseline) / _MIB)


def _sized_frame(rows: int) -> pl.LazyFrame:
    """A `rows`-long frame; the sampler expressions key off `pl.len()`, not its contents."""
    return pl.LazyFrame({"_": pl.Boolean()}).clear(rows)


def _draw_params(comp: Comparison, *, regime: Regime, rows: int, seed: int) -> Params:
    """Draw a case's parameters, deterministically from `(regime, rows, seed)` alone.

    Determinism is required, not merely nice: both contenders and each isolated memory subprocess
    call this independently and must agree on the numbers, or the two sides would be measured on
    different distributions. The draws take their own seed stream so that widening a parameter domain
    cannot shift the evaluation inputs.
    """
    rng = np.random.default_rng([seed, _PARAM_STREAM])
    values = {name: spec.draw(rng, rows) if regime == "column" else spec.fixed for name, spec in comp.params.items()}
    return Params(regime=regime, values=values)


def _evaluation_points(case: Case, frozen: ScipyFrozen) -> Array:
    """Deterministic evaluation inputs for a value-keyed method, shared by both sides.

    The inverses get uniform quantiles from ``[0, 1)``; every other method gets the distribution's
    own seeded draws. Feeding an inverse the support draws instead puts almost every row on the
    null-outside-``[0, 1]`` path, which measures the guard rather than the algorithm.

    These take the plain int seed rather than `_scipy_generator`: this is input construction, outside
    every timed closure, so only its reproducibility matters. The length is asserted because scipy
    requires the evaluation point to broadcast against `column` parameters.
    """
    match case.spec.value_input:
        case "quantile":
            values: Array = np.random.default_rng(case.seed).uniform(0.0, 1.0, size=case.rows)
        case "support":
            values = np.asarray(frozen.rvs(size=case.rows, random_state=case.seed), dtype=np.float64)
        case _:
            msg = f"{case.method} is not a value-keyed method, so it has no evaluation inputs"
            raise AssertionError(msg)
    if values.shape != (case.rows,):
        msg = f"{case.id}: evaluation inputs have shape {values.shape}, expected {(case.rows,)}"
        raise AssertionError(msg)
    return values


def _input_frame(params: Params, case: Case, points: Array | None) -> pl.LazyFrame:
    """The frame a `polars_stats` call selects from.

    Under `column` the parameter columns set the call's row count. Otherwise it comes from the frame,
    which a moment needs only one row of: with length-1 parameters the `select` returns height 1
    whatever it is given, so building a `rows`-long one would allocate a column nothing reads.
    """
    columns: dict[str, Array] = params.columns
    if points is not None:
        columns[_VALUE_COLUMN] = points
    if columns:
        return pl.LazyFrame(columns)
    return _sized_frame(1 if case.spec.kind == "moment" else case.rows)


def _polars_stats_call(comp: Comparison, case: Case) -> Call:
    """One lazy `select` on the streaming engine, the path users hit at scale."""
    params = _draw_params(comp, regime=case.regime, rows=case.rows, seed=case.seed)
    dist, frozen = comp.build(params)
    spec = case.spec
    attr = spec.plugin_attr.format(d=comp.density_token)
    points = _evaluation_points(case, frozen) if spec.kind == "value" else None
    lf = _input_frame(params, case, points)
    match spec.kind:
        case "value":
            expr = getattr(dist, attr)(pl.col(_VALUE_COLUMN))
        case "moment":
            expr = getattr(dist, attr)()
        case "sampler":
            expr = dist.samples(case.draws_per_row, seed=case.seed) if spec.multi_draw else dist.sample(seed=case.seed)
        case _:
            _unreachable(spec.kind)
    return lambda: lf.select(s=expr).collect(engine="streaming")


def _scipy_generator(seed: int) -> np.random.Generator:
    """The RNG scipy's samplers are handed, which is what keeps the sampler cells comparable.

    Passing `random_state=<int>` instead makes scipy build a legacy `RandomState` per call and draw
    from MT19937, while our Rust side draws from `Pcg64Mcg` (`src/rng.rs`). That is a 1.1x to 2.4x
    handicap depending on the distribution, and it is an artefact of the seed's *type*, not of the
    API a scipy user writes. A `Generator` puts both sides on the same PCG family.

    Built per call, not once per cell, so scipy pays the same construct-from-seed cost our side does
    and every timed iteration still draws identical values. Construction is ~3 us.
    """
    return np.random.default_rng(seed)


def _scipy_call(comp: Comparison, case: Case) -> Call:
    """The frozen distribution's matching method on NumPy arrays."""
    params = _draw_params(comp, regime=case.regime, rows=case.rows, seed=case.seed)
    _, frozen = comp.build(params)
    spec = case.spec
    attr = spec.scipy_attr.format(d=comp.density_token)
    match spec.kind:
        case "value":
            points = _evaluation_points(case, frozen)
            evaluate = cast("Callable[[Array], Outcome]", getattr(frozen, attr))
            return lambda: evaluate(points)
        case "moment":
            return cast("Call", getattr(frozen, attr))
        case "sampler":
            seed = case.seed
            if not spec.multi_draw:
                rows = case.rows
                return lambda: frozen.rvs(size=rows, random_state=_scipy_generator(seed))
            if case.regime == "column":
                # `(rows, draws)` cannot broadcast against length-`rows` parameters, only
                # `(draws, rows)` can. The transpose back is a view, so scipy never materialises the
                # row-major layout the polars_stats side returns.
                transposed = (case.draws_per_row, case.rows)
                return lambda: frozen.rvs(size=transposed, random_state=_scipy_generator(seed)).T
            size = (case.rows, case.draws_per_row)
            return lambda: frozen.rvs(size=size, random_state=_scipy_generator(seed))
        case _:
            _unreachable(spec.kind)


CONTENDERS: Mapping[str, Contender] = {
    REFERENCE_CONTENDER: _polars_stats_call,
    "scipy": _scipy_call,
}
"""The registered contenders. Insertion order is column order, reference first."""


CHALLENGERS: tuple[str, ...] = tuple(name for name in CONTENDERS if name != REFERENCE_CONTENDER)
"""The contenders a report shows a ratio for, in registration order."""


def _is_array(outcome: Outcome) -> TypeIs[Array]:
    return isinstance(outcome, np.ndarray)


def _outputs_agree(case: Case, reference_out: Outcome, other_out: Outcome) -> bool:
    """Correctness gate for one reference-vs-challenger output pair.

    * Samplers: shape-only, since independent RNGs cannot match values.
    * Value-keyed methods evaluate the same inputs on both sides, so values must agree to
      `np.allclose` within the loose `_MATCH_RTOL` / `_MATCH_ATOL`. The scipy-parity suite owns the
      tight bounds.
    * Moments: height follows the regime. With length-1 parameters both sides return one value; with
      parameter columns both return one value per row.

    Shaped around a polars frame against a NumPy array. A contender returning a frame rather than an
    array would need this widened.
    """
    if not isinstance(reference_out, pl.DataFrame):
        msg = f"{case.id}: reference contender returned {type(reference_out).__name__}, expected a DataFrame"
        raise TypeError(msg)
    spec = case.spec
    if spec.kind == "moment":
        expected = case.rows if case.regime == "column" else 1
        allowed = ((case.rows,),) if case.regime == "column" else ((), (1,))
        if reference_out.height != expected or np.shape(other_out) not in allowed:
            return False
        return _values_agree(reference_out, other_out)
    if not _is_array(other_out):
        msg = f"{case.id}: challenger returned {type(other_out).__name__}, expected an array"
        raise TypeError(msg)
    if spec.multi_draw:
        dtype = reference_out.to_series().dtype
        return (
            reference_out.height == case.rows
            and isinstance(dtype, pl.Array)
            and dtype.size == case.draws_per_row
            and other_out.shape == (case.rows, case.draws_per_row)
        )
    if reference_out.height != case.rows or other_out.shape != (case.rows,):
        return False
    return True if spec.kind == "sampler" else _values_agree(reference_out, other_out)


def _values_agree(reference_out: pl.DataFrame, other_out: Outcome) -> bool:
    """Whether the two sides agree to the harness's loose sanity tolerances.

    A null on our side fails the gate outright. `to_numpy` renders it as `nan`, which `equal_nan`
    would otherwise read as agreeing with a genuine scipy `nan`, quietly turning a regression into
    an `ok`.
    """
    reference = reference_out.to_series().cast(pl.Float64)
    if reference.null_count():
        return False
    return np.allclose(
        reference.to_numpy(),
        np.asarray(other_out, dtype=np.float64),
        rtol=_MATCH_RTOL,
        atol=_MATCH_ATOL,
        equal_nan=True,
    )


def _check_gate_can_reject() -> None:
    """Prove `_outputs_agree` both accepts a real match and rejects each way it can be wrong.

    The gate is the harness's only correctness signal, and its failure mode is silent: a broken gate
    reports `ok` for every cell forever. Nothing else in the repo exercises it, so it is checked here
    on the same terms as the `METHOD_SPECS` table above -- at import, not mid-sweep. The positive
    case is part of the check on purpose, since a gate stuck on `False` would satisfy the rest.
    """
    value = Case("_check", "cdf", "scalar", 3, None, 0)
    ok = pl.DataFrame({"s": [0.25, 0.5, 0.75]})
    draws = Case("_check", "samples", "scalar", 2, 3, 0)
    wide = pl.DataFrame({"s": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]}, schema={"s": pl.Array(pl.Float64, 3)})
    expectations = (
        ("matching values", True, value, ok, np.array([0.25, 0.5, 0.75])),
        ("value past rtol", False, value, ok, np.array([0.25, 0.5, 0.7501])),
        ("short reference", False, value, pl.DataFrame({"s": [0.25, 0.5]}), np.array([0.25, 0.5, 0.75])),
        ("null reference", False, value, pl.DataFrame({"s": [0.25, None, 0.75]}), np.array([0.25, 0.5, 0.75])),
        ("matching draws", True, draws, wide, np.zeros((2, 3))),
        ("wrong draw width", False, draws, wide, np.zeros((2, 2))),
    )
    for label, expected, case, reference, other in expectations:
        if _outputs_agree(case, reference, other) is not expected:
            msg = f"_outputs_agree is broken: {label!r} should be {expected}"
            raise RuntimeError(msg)


_check_gate_can_reject()


class _PeakMiB(NamedTuple):
    value: float


class _ChildFailure(NamedTuple):
    detail: str


def _peak_memory_worker(queue: mp.Queue[MemoryReply], comp: Comparison, case: Case, contender: str) -> None:
    """Subprocess entrypoint: rebuild `contender`'s call from the registry and post back its peak."""
    try:
        queue.put(_PeakMiB(_peak_mib(CONTENDERS[contender](comp, case))))
    except BaseException as exc:  # noqa: BLE001 - surface any child failure to the parent as a message
        queue.put(_ChildFailure(repr(exc)))


def _isolated_peak_mib(comp: Comparison, case: Case, contender: str) -> float:
    """Peak memory of one call, measured in a fresh spawned process."""
    ctx = mp.get_context("spawn")
    queue: mp.Queue[MemoryReply] = ctx.Queue()
    proc = ctx.Process(target=_peak_memory_worker, args=(queue, comp, case, contender))
    proc.start()
    try:
        reply = queue.get(timeout=_CHILD_REPLY_TIMEOUT_S)
    except BaseException:
        proc.terminate()
        raise
    finally:
        proc.join(timeout=_CHILD_JOIN_TIMEOUT_S)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=_CHILD_JOIN_TIMEOUT_S)
    if isinstance(reply, _PeakMiB):
        return reply.value
    msg = f"memory subprocess for {case.id} ({contender}) failed: {reply.detail}"
    raise RuntimeError(msg)


def _measure(comp: Comparison, case: Case, *, budget: Budget, memory: bool) -> Result:
    """Time every registered contender on one case, gate correctness, and optionally profile memory.

    The correctness gate reads each timed loop's own warmup output, so no contender is called an
    extra time to produce it.
    """
    outcomes: dict[str, Outcome] = {}
    timings: dict[str, Timing] = {}
    for name, build in CONTENDERS.items():
        outcomes[name], timings[name] = _time_call(build(comp, case), budget)
    reference_out = outcomes[REFERENCE_CONTENDER]
    verdicts = [
        _outputs_agree(case, reference_out, out) for name, out in outcomes.items() if name != REFERENCE_CONTENDER
    ]
    peaks = {name: _isolated_peak_mib(comp, case, name) for name in CONTENDERS} if memory else {}
    return Result(
        case=case,
        label=comp.label(case.method),
        measurements={name: Measurement(*timing, peak_mib=peaks.get(name)) for name, timing in timings.items()},
        matches=bool(verdicts) and all(verdicts),
    )


def measure_cases(comp: Comparison, sweep: Sweep, *, memory: bool = False) -> Iterator[Result]:
    """Yield one `Result` per cell of the sweep grid, in report order.

    A cell that raises is reported and skipped rather than discarding the cells already measured;
    the caller can therefore still emit a partial report after an interrupt.
    """
    cases = list(sweep.cases(comp.name))
    try:
        for index, case in enumerate(cases, start=1):
            print(f"  [{index}/{len(cases)}] {case.id}", end="\r", flush=True)
            try:
                yield _measure(comp, case, budget=sweep.budget, memory=memory)
            except Exception as exc:  # noqa: BLE001 - one bad cell must not abandon the sweep
                print(f"\nSKIPPED {case.id}: {exc!r}")
    finally:
        print(" " * 100, end="\r")


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


def _memory_measured(results: Sequence[Result]) -> bool:
    return any(m.peak_mib is not None for r in results for m in r.measurements.values())


class ReportRow(NamedTuple):
    """One report row's values, before either renderer styles them.

    The two file renderers and the terminal renderer differ only in presentation, so the cell
    *contents* are computed once here.
    """

    regime: str
    label: str
    rows: str
    n_samples: str
    iterations: str
    times: tuple[str, ...]
    ratios: tuple[float, ...]
    peaks: tuple[str, ...]
    matches: bool

    @property
    def head(self) -> tuple[str, ...]:
        return (self.regime, self.label, self.rows, self.n_samples, self.iterations)


def _report_headers(*, memory: bool) -> list[str]:
    """The report's column headers, shared by every renderer."""
    contenders, challengers = tuple(CONTENDERS), CHALLENGERS
    return [
        "regime",
        "method",
        "rows",
        "n_samples",
        f"iters ({'/'.join(contenders)})",
        *(f"{name} (ms)" for name in contenders),
        *(f"{name} / {REFERENCE_CONTENDER}" for name in challengers),
        *(f"{name} (MiB)" for name in contenders if memory),
        "match",
    ]


def _report_row(result: Result, *, memory: bool, plus_minus: str) -> ReportRow:
    contenders = tuple(CONTENDERS)

    def timing(name: str) -> str:
        m = result.measurements[name]
        spread = "-" if np.isnan(m.std_ms) else f"{m.std_ms:.3f}"
        return f"{m.p50_ms:.3f} {plus_minus} {spread}"

    case = result.case
    return ReportRow(
        regime=case.regime,
        label=result.label,
        rows=f"{case.rows:_}",
        n_samples="-" if case.n_samples is None else str(case.n_samples),
        iterations="/".join(str(result.measurements[name].iterations) for name in contenders),
        times=tuple(timing(name) for name in contenders),
        ratios=tuple(result.ratio(name) for name in CHALLENGERS),
        peaks=tuple(f"{result.measurements[name].peak_mib:.1f}" for name in contenders if memory),
        matches=result.matches,
    )


BuildProfile = Literal["release", "debug", "unknown"]


def build_profile() -> BuildProfile:
    """Whether the installed extension was compiled with optimisations.

    `unknown` means the extension predates the `__debug_build__` flag, so it cannot vouch for
    itself; the guard treats that the same as `debug` rather than assuming the good case.
    """
    if (flag := getattr(_internal, "__debug_build__", None)) is None:
        return "unknown"
    return "debug" if flag else "release"


def require_release_build() -> None:
    """Refuse to measure an unoptimised build, which is the one failure the numbers cannot show.

    A debug `maturin develop` runs the Rust math unoptimised and would make `polars_stats` look far
    slower than scipy's optimised C. The resulting table looks entirely normal, so this is checked
    rather than documented.
    """
    if (profile := build_profile()) == "release":
        return
    detail = (
        "the installed polars_stats was built with `maturin develop` (debug profile)"
        if profile == "debug"
        else "the installed polars_stats predates the build-profile flag, so it cannot report its profile"
    )
    msg = (
        f"refusing to benchmark: {detail}. A debug build runs the Rust math unoptimised, which would "
        "make polars_stats look far slower than scipy and invalidate every number. "
        "Run `make install-release` first."
    )
    raise RuntimeError(msg)


class Environment(TypedDict):
    """The version and platform stamp a saved report carries."""

    python: str
    platform: str
    polars: str
    scipy: str
    numpy: str
    polars_stats: str
    build: BuildProfile


def _environment() -> Environment:
    return {
        "python": f"{platform.python_implementation()} {platform.python_version()}",
        "platform": platform.platform(),
        "polars": pl.__version__,
        "scipy": scipy.__version__,
        "numpy": np.__version__,
        "polars_stats": _internal.__version__,
        "build": build_profile(),
    }


def _header_lines(sweep: Sweep, results: Sequence[Result]) -> list[str]:
    env = _environment()
    budget = sweep.budget
    memory = "peak RSS (MiB), isolated subprocess" if _memory_measured(results) else "not measured (pass --memory)"
    return [
        f"- rows: {', '.join(f'{r:_}' for r in sweep.rows)}; `samples` draws per row: "
        f"{', '.join(str(n) for n in sweep.n_samples)}",
        f"- methods: {', '.join(sweep.methods)}",
        (
            f"- regimes: {', '.join(sweep.regimes)}. **Never compare across regimes**: a column cell"
            " does strictly more work per row by design"
        ),
        (
            f"- budget per cell: <= {budget.max_iterations} iterations, <= {budget.max_seconds:g}s,"
            f" >= {budget.min_iterations}; `iters` is the count actually measured, per contender"
        ),
        f"- time p50 +/- std (ms); memory: {memory}",
        f"- {env['python']} on {env['platform']}",
        f"- polars_stats {env['polars_stats']} ({env['build']} build), polars {env['polars']},"
        f" scipy {env['scipy']}, numpy {env['numpy']}",
    ]


def _render_markdown(comp: Comparison, results: Sequence[Result], sweep: Sweep) -> str:
    """A markdown document (heading, environment stamp, table), ready to paste into the docs."""
    memory = _memory_measured(results)
    headers = _report_headers(memory=memory)
    aligns = ["---", "---", *("---:" for _ in headers[2:-1]), ":---:"]
    lines = [
        f"## {comp.name}: polars_stats vs scipy.stats",
        "",
        *_header_lines(sweep, results),
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(aligns) + " |",
    ]
    for result in results:
        row = _report_row(result, memory=memory, plus_minus="+/-")
        cells = [
            row.regime,
            f"`{row.label}`",
            *row.head[2:],
            *row.times,
            *(f"{ratio:.2f}x" for ratio in row.ratios),
            *row.peaks,
            "ok" if row.matches else "**MISMATCH**",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


class SweepPayload(TypedDict):
    """The sweep configuration as it appears in a `json` report."""

    rows: list[int]
    n_samples: list[int]
    regimes: list[str]
    methods: list[str]
    budget: dict[str, float]
    seed: int


class ResultPayload(TypedDict):
    """One cell as it appears in a `json` report."""

    regime: Regime
    method: str
    rows: int
    n_samples: int | None
    measurements: dict[str, MeasurementPayload]
    ratio: float
    matches: bool


class ReportPayload(TypedDict):
    """A whole `json` report."""

    schema: int
    distribution: str
    environment: Environment
    sweep: SweepPayload
    results: list[ResultPayload]


def _render_json(comp: Comparison, results: Sequence[Result], sweep: Sweep) -> str:
    """Machine-readable report: schema, environment, sweep, and one object per cell."""
    challenger = CHALLENGERS[0]
    payload: ReportPayload = {
        "schema": SCHEMA_VERSION,
        "distribution": comp.name,
        "environment": _environment(),
        "sweep": {
            "rows": list(sweep.rows),
            "n_samples": list(sweep.n_samples),
            "regimes": list(sweep.regimes),
            "methods": list(sweep.methods),
            "budget": asdict(sweep.budget),
            "seed": sweep.seed,
        },
        "results": [
            {
                "regime": result.case.regime,
                "method": result.label,
                "rows": result.case.rows,
                "n_samples": result.case.n_samples,
                "measurements": {name: m.payload for name, m in result.measurements.items()},
                "ratio": result.ratio(challenger),
                "matches": result.matches,
            }
            for result in results
        ],
    }
    return json.dumps(payload, indent=2, allow_nan=False) + "\n"


def _render_rich(comp: Comparison, results: Sequence[Result], sweep: Sweep) -> None:
    """Print a coloured table to the terminal (ratio green when polars_stats wins, red otherwise)."""
    env = _environment()
    memory = _memory_measured(results)
    budget = sweep.budget
    table = Table(
        title=f"{comp.name}: polars_stats vs scipy.stats",
        caption=(
            f"<= {budget.max_iterations} iters / <= {budget.max_seconds:g}s per cell "
            f"| time p50 +/- std (ms){', peak RSS (MiB)' if memory else ''} "
            f"| regimes are separate paths, never cross-compared "
            f"| polars_stats {env['polars_stats']} ({env['build']}), polars {env['polars']},"
            f" scipy {env['scipy']}, numpy {env['numpy']}"
        ),
    )
    headers = _report_headers(memory=memory)
    table.add_column(headers[0], style="magenta")
    table.add_column(headers[1], style="bold cyan")
    for header in headers[2:-1]:
        table.add_column(header, justify="right")
    table.add_column(headers[-1], justify="center")
    for result in results:
        row = _report_row(result, memory=memory, plus_minus="±")
        table.add_row(
            *row.head,
            *row.times,
            *(f"[{'green' if ratio >= 1.0 else 'red'}]{ratio:.2f}x[/]" for ratio in row.ratios),
            *row.peaks,
            "[green]ok[/]" if row.matches else "[red]MISMATCH[/]",
        )
    Console().print(table)
