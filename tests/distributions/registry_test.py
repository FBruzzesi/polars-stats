"""Every exported distribution has a row in `tests/_registry.py`, and every row names an export.

The registry-driven suites iterate `ALL_SPECS`, so a missing row means they never see the
distribution: the run passes, the diff looks complete, and no contract those suites exist to pin is
checked. `docs/contributing.md` names that as the worst failure mode in the project. This module is
what turns it into a red test.

These assertions are structural. A row with a wrong `eval_range` or a wrong `bounds` passes every one
of them; what they catch is a row that is missing, names no export, or contradicts itself.
"""

from __future__ import annotations

import inspect
import math
from typing import TYPE_CHECKING, get_args

import polars as pl
import pytest

import polars_stats
from polars_stats.distributions._base import _UnivariateDistribution
from tests._registry import ALL_SPECS, CONTRACT_FRAME, DEGENERATE, MOMENTS, ULP_TOLERANT_MOMENTS, Regime

if TYPE_CHECKING:
    from tests._registry import DistSpec


def _exported_distributions() -> dict[str, type[_UnivariateDistribution]]:
    """The concrete distribution classes in `polars_stats.__all__`, keyed by their plugin prefix.

    Abstract bases (`ContinuousDistribution`, `DiscreteDistribution`) and non-class exports are
    excluded: they carry no plugin prefix and nothing in the registry describes them.
    """
    found: dict[str, type[_UnivariateDistribution]] = {}
    for name in polars_stats.__all__:
        obj = getattr(polars_stats, name)
        if isinstance(obj, type) and issubclass(obj, _UnivariateDistribution) and not inspect.isabstract(obj):
            found[obj._distribution_name] = obj
    return found


EXPORTED = _exported_distributions()
SPEC_NAMES = frozenset(spec.name for spec in ALL_SPECS)

_REGIMES: tuple[Regime, ...] = get_args(Regime)
"""Read off `Regime` rather than restated, so a seventh spelling cannot be added uncovered."""


def test_every_exported_distribution_has_a_spec() -> None:
    missing = sorted(set(EXPORTED) - SPEC_NAMES)
    assert not missing, f"exported but absent from ALL_SPECS: {missing}"


def test_every_spec_names_an_exported_distribution() -> None:
    stale = sorted(SPEC_NAMES - set(EXPORTED))
    assert not stale, f"in ALL_SPECS but not exported: {stale}"


def test_spec_names_are_unique() -> None:
    names = [spec.name for spec in ALL_SPECS]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    assert not duplicates, f"duplicate spec names: {duplicates}"


def test_spec_names_its_own_class_and_plugin_prefix(spec: DistSpec) -> None:
    """Ties the registry id to the Rust plugin namespace, so a rename cannot move only one of them."""
    assert spec.cls is EXPORTED[spec.name], f"{spec.name}: row carries {spec.cls.__name__}"
    assert spec.cls._distribution_name == spec.name, f"{spec.name}: plugin prefix is {spec.cls._distribution_name!r}"


@pytest.mark.parametrize("regime", _REGIMES)
def test_spec_builds_every_parameter_regime(spec: DistSpec, regime: Regime) -> None:
    """A row whose `series` spelling raised would otherwise fail deep inside whichever suite reached
    it first, with a traceback about that suite rather than about the row."""
    spec.build(regime, mask=pl.lit(value=False))  # must not raise


def test_spec_parameter_count_matches_its_example(spec: DistSpec) -> None:
    """`build` zips parameters and values strictly, so a short tuple raises inside whichever test
    reaches it first rather than here."""
    arity = len(spec.parameters)
    assert len(spec.example) == arity, f"{spec.name}: example has {len(spec.example)} values for {arity} parameters"
    wrong = [case.label for case in spec.invalid if len(case.params) != arity]
    assert not wrong, f"{spec.name}: invalid rows with the wrong arity: {wrong}"


def test_spec_parameter_columns_are_distinct_and_in_the_contract_frame(spec: DistSpec) -> None:
    """Distinctness is what makes the output-naming contract discriminating: with two parameters
    sharing a column, an expression that inherited the *second* input's name would still pass."""
    columns = [p.column for p in spec.parameters]
    assert len(set(columns)) == len(columns), f"{spec.name}: parameters share a column: {columns}"
    missing = [c for c in columns if c not in CONTRACT_FRAME.columns]
    assert not missing, f"{spec.name}: columns absent from the contract frame: {missing}"


def test_spec_parameter_dtypes_match_the_contract_frame(spec: DistSpec) -> None:
    """A row that tagged `Binomial.n` as a float parameter would build a `Float64` column the plugin
    refuses; failing here keeps the report on the row rather than on the plugin."""
    wrong = [
        (p.name, p.column, CONTRACT_FRAME.schema[p.column])
        for p in spec.parameters
        if p.column in CONTRACT_FRAME.schema and CONTRACT_FRAME.schema[p.column] != p.dtype
    ]
    assert not wrong, f"{spec.name}: parameter dtype disagrees with its contract column: {wrong}"


def test_spec_family_fields_match_its_family(spec: DistSpec) -> None:
    """`tests/property/pdf_test.py` selects on `continuous` and then reads the matching field, so a
    row tagged one way while carrying the other family's field skips the mass property silently."""
    if spec.continuous:
        assert spec.integration_bounds is not None, f"{spec.name}: continuous row has no integration_bounds"
        assert spec.support is None, f"{spec.name}: continuous row carries a discrete `support`"
    else:
        assert spec.support is not None, f"{spec.name}: discrete row has no support"
        assert spec.integration_bounds is None, f"{spec.name}: discrete row carries continuous integration_bounds"


def test_spec_family_matches_the_class(spec: DistSpec) -> None:
    """A wrongly tagged row sends the density assertions at `pdf` on a distribution that has only
    `pmf`; nothing else checks the tag against the class it claims."""
    assert polars_stats.is_continuous(spec.build("scalar")) is spec.continuous


def test_spec_bounds_are_ordered_and_contain_on_support(spec: DistSpec) -> None:
    """`on_support` is inside `bounds` and off the endpoints where the answer saturates.

    A row whose `on_support` sat at the upper bound (`Bernoulli`'s `1.0`, once) evaluates where the cdf
    is `1` whatever the parameters are, so the null contract passes while asserting nothing. The lower
    bound is the same trap for a continuous row, where the cdf is `0`; for a discrete one the lowest
    mass point *is* the lower bound and carries real mass, so only the upper end is excluded.

    Whether every parameter reaches the formula at that point, no structural check can see.
    """
    lo, hi = spec.bounds
    assert lo < hi, f"{spec.name}: bounds are not ordered: {spec.bounds}"
    floor = lo if spec.continuous else math.nextafter(lo, -math.inf)
    assert floor < spec.on_support_point < hi, (
        f"{spec.name}: on_support {spec.on_support_point} saturates within {spec.bounds}"
    )


def test_spec_invalid_table_is_not_empty_and_names_its_parameter(spec: DistSpec) -> None:
    """Every distribution has at least one finite out-of-domain parameterisation, and each names a rule.

    An emptied table would remove a distribution from the validation contract with no signal. The
    `must be` fragment is what stops a match against polars' `the plugin failed with message:`
    preamble, which every one-letter parameter name would otherwise satisfy.
    """
    assert spec.invalid, f"{spec.name}: no invalid parameterisations recorded"
    unnamed = [case.label for case in spec.invalid if spec.blamed(case) is None]
    assert not unnamed, f"{spec.name}: invalid rows whose match names no parameter: {unnamed}"


def test_the_ulp_tolerant_table_names_real_specs_and_real_moments() -> None:
    """Every relaxed `(spec, moment)` pair exists, so a rename cannot silently re-tighten or loosen one.

    The table is the one place the suite gives up bit-exactness between the two parameter routings.
    A typo there is invisible: the pair simply never matches, and the comparison goes back to exact
    (or, for a mistyped moment, stays relaxed for a moment that no longer exists).
    """
    stale = sorted(set(ULP_TOLERANT_MOMENTS) - SPEC_NAMES)
    assert not stale, f"ULP_TOLERANT_MOMENTS names specs that do not exist: {stale}"
    unknown = sorted({m for moments in ULP_TOLERANT_MOMENTS.values() for m in moments} - set(MOMENTS))
    assert not unknown, f"ULP_TOLERANT_MOMENTS names moments that do not exist: {unknown}"


def test_the_degenerate_table_names_real_specs() -> None:
    """`DEGENERATE` is keyed by `DistributionName`, so a typo is a type error; dropping a row from
    `ALL_SPECS` while leaving its degenerate case behind is not, and this catches that."""
    named = {name for name, _ in DEGENERATE.values()}
    assert named <= SPEC_NAMES, f"DEGENERATE names specs that do not exist: {sorted(named - SPEC_NAMES)}"


def test_spec_windows_are_ordered_and_finite_at_its_example(spec: DistSpec) -> None:
    """The three window callables are the only fields nothing calls before `tests/property/` does.

    A swapped index or a wrong arity there fails as a hypothesis falsifying example inside a
    trapezoidal mass integral, which names the property rather than the row that broke it.
    """
    windows = {"eval_range": spec.eval_range(spec.example)}
    if spec.integration_bounds is not None:
        windows["integration_bounds"] = spec.integration_bounds(spec.example)

    malformed = [
        (field, window)
        for field, window in windows.items()
        if not all(map(math.isfinite, window)) or window[0] >= window[1]
    ]
    assert not malformed, f"{spec.name}: window is not a finite ordered pair: {malformed}"

    if spec.support is not None:
        assert spec.support(spec.example), f"{spec.name}: support at its example is empty"


def test_recorded_sample_dtype_matches_the_plugin(spec: DistSpec, dist: _UnivariateDistribution) -> None:
    """A recorded dtype that drifted from the plugin would make every dtype assertion in
    `sample_test.py` / `samples_test.py` agree with the registry instead of with the code."""
    got = pl.DataFrame({"_": range(4)}).select(z=dist.sample(seed=0)).schema["z"]
    assert got == spec.sample_dtype, f"{spec.name}: sample returns {got}, registry records {spec.sample_dtype}"
