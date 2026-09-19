# Test layout convention

**One registry row per distribution, one file per contract.** A new distribution adds a row to
[`_registry.py`](_registry.py) and inherits every shared contract; it writes new test *files* only for
facts that are genuinely its own.

Tests split by what they assert, into four places.

## 1. The registry: `tests/_registry.py`

One frozen `DistSpec` per distribution, holding its class, whether it is `continuous`, its parameters
(each with a constructor keyword, a column in the shared contract frame, and a type), a `hypothesis`
strategy, one `example` parameterisation, the evaluation window, the support `bounds`, a
discriminating `on_support_point`, the recorded `sample` dtype, the finite out-of-domain `invalid`
table, and the family's mass field (`integration_bounds` for a continuous row, `support` for a
discrete one).

[`conftest.py`](conftest.py) delivers it: a test that requests the `spec` fixture runs once per
distribution, and `dist` hangs off it. `contract_frame` and `single_row_frame` are standalone
fixtures over the registry's own `CONTRACT_FRAME`.

Everything else is **derived**, not recorded, so the row stays a table: the six parameter spellings
come from `parameters` (`spec.build("scalar" | "column" | "masked" | "literal" | "series" | "name")`),
the density methods from the family, the off-support points from `bounds`, and the non-finite invalid
sweep from the float parameters.

[`distributions/registry_test.py`](distributions/registry_test.py) is the guard: it cross-checks
`polars_stats.__all__` against the rows, so a distribution exported without a row fails a test rather
than being silently skipped.

Four sparse per-distribution tables sit outside the row, because most distributions have no entry
and a field that is empty more often than not is a bespoke fact wearing a row's clothes. The split is
by reader: a table more than one module reads lives in `_registry.py` beside the rows, and one only
its own test file reads stays in that file.

| Table | Lives in | What it records |
| --- | --- | --- |
| `DEGENERATE` | `_registry.py` | parameterisations where the mass collapses onto one point |
| `ULP_TOLERANT_MOMENTS` | `_registry.py` | the `(spec, moment)` pairs that are not bit-exact across routings |
| `UNDEFINED_MOMENTS` | `_registry.py` | the `(spec, moment)` pairs with no value, pinned to null on every valid row |
| `_DENSITY_AT_ENDPOINT` | `support_test.py` | the density *at* a finite support endpoint |

None is invisible. All four are keyed by `DistributionName`, so a mistyped key is a type error; the
three in `_registry.py` are additionally tied back to `ALL_SPECS` by `registry_test.py`, and
`_DENSITY_AT_ENDPOINT` is asserted present whenever the support has a finite endpoint.

## 2. Behavioural contracts: `tests/distributions/`

Flat, one file per contract. Most are parametrised over the registry; `identities_test.py` and
`precision_test.py` are the two bespoke-fact files and are not. These assert behaviour a numeric
match against SciPy cannot see:

| File | Contract |
| --- | --- |
| `validation_test.py` | an invalid parameter raises and names its rule; a null one nulls that row only; a null or `NaN` evaluation point; the numeric and integer dtype gates |
| `support_test.py` | outside the support the answers are saturated constants, and a finite endpoint of a continuous support is already saturated |
| `inverse_test.py` | `ppf(0)` / `ppf(1)` are the support bounds, and a discrete inverse is integer-valued |
| `moments_test.py` | `std` is the square root of `variance`, the mean is inside the support, the variance is non-negative, or both are null where `UNDEFINED_MOMENTS` records them; a divergent `Pareto` moment is `+inf` on both routings |
| `construct_test.py` | the constructor refuses a wrong scalar *type* and defers every *value* to evaluation |
| `naming_test.py` | output names follow polars' first-input rule; a `str` argument means `pl.col(name)` |
| `fast_path_test.py` | the constant-parameter paths refuse and accept exactly what the per-row paths do |
| `sample_test.py`, `samples_test.py` | shape, dtype, per-row independence, support, and a CLT-tolerance mean check |
| `broadcast_test.py` | length-1 and over-long inputs, in `select` and in partition contexts |
| `identities_test.py` | algebraic reductions between distributions, and a distribution's own symmetry |
| `precision_test.py` | numerical-regime facts with no shared contract and no scipy oracle |

`repr_test.py`, `sampler_args_test.py` and `type_guard_test.py` are registry-parametrised too, on a
single subject each.
`row_index_test.py` and `std_representable_range_test.py` sit outside the registry axis, and
[`plugin_boundary_dtype_test.py`](plugin_boundary_dtype_test.py) and
[`samples_allocation_test.py`](samples_allocation_test.py) run a subprocess per case because the abort
each guards against would take pytest down with it. A new distribution adds nothing to those two.

**Where a new distribution's bespoke facts go:** an algebraic identity to `identities_test.py`, a
numerical-regime fact to `precision_test.py`, anything else to the file that owns its subject. A fact
true of exactly one distribution stays a bespoke test; it never becomes a registry field.

## 3. SciPy parity: `tests/scipy_parity/<name>_test.py`

One table-driven module per distribution. Each lists its methods as `Case` rows and runs them through
the shared harness in [`_harness.py`](scipy_parity/_harness.py), which compares every method against
the matching `scipy.stats` attribute across a parameter grid.

Adding a method is one row; the comparison mechanics (grid selection, Boolean→float casting, tolerance)
are not re-implemented.

Tolerances follow the *upstream implementation*, not aspiration: closed-form methods hold the default
`1e-12`; methods routed through `erf`/`erfc` or a binary-search `inverse_cdf` pass a relaxed per-`Case`
tolerance (e.g. the normal cdf/sf family at `1e-9`). Document any exception inline on the `Case`.

This suite is also *why* most per-distribution value assertions are unnecessary: parity pins the scalar
path against scipy, `property/value_keyed_test.py` pins the column path against the scalar one bit for
bit, and the two compose.

## 4. Property tests: `tests/property/`

Distribution-agnostic invariants under [`hypothesis`](https://hypothesis.readthedocs.io), parametrised
over the same registry (with `@pytest.mark.parametrize("spec", ALL_SPECS)` rather than the fixture,
since `@given` cannot draw from a fixture):

* `0 <= cdf(x) <= 1` and `cdf` non-decreasing in `x`;
* `cdf(ppf(q)) ~= q` for `q` in the open unit interval (continuous);
* `pdf(x) >= 0` (continuous) / `pmf(x) >= 0` (discrete);
* trapezoidal integral of `pdf` over the (truncated) support `~= 1` (continuous);
* sum of `pmf` over the finite support `~= 1` (discrete);
* the constant-parameter path equals the per-row path bit for bit, for values, moments and draws.

The capped profile in [`conftest.py`](property/conftest.py) (`max_examples`, `deadline=None`) keeps the
suite under the CI budget and non-flaky; raise `max_examples` locally when investigating a failure.
