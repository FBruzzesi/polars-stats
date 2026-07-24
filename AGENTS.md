# AGENTS.md

Guidance for AI agents working in this repository. The documents under [docs/](docs/) are canonical for everything they
cover; this file adds only what is agent-specific and points at the rest.

## Project in one paragraph

`polars-stats` is a Polars expression plugin that exposes `scipy.stats`-style distributions natively inside Polars
expressions, with column-valued parameters. Rust + statrs handle the math, Python provides the user-facing distribution
classes.

## Where to look first

| You want to know | Read |
|---|---|
| Build / test / lint commands, the dependency stack, and the step-by-step for adding a distribution | [Contributing](./docs/contributing.md) |
| What the system is and how it is wired | [Architecture](./docs/explanation/architecture.md) |
| Why a decision was made, or what is still open | [Design notes](./docs/explanation/design.md) |
| Public method surface and base-class defaults | [polars_stats/distributions/_base.py](polars_stats/distributions/_base.py) |
| Canonical pattern, closed-form continuous | [_uniform.py](polars_stats/distributions/_uniform.py) / [uniform.rs](src/distributions/uniform.rs) |
| Canonical pattern, statrs-backed continuous | [_normal.py](polars_stats/distributions/_normal.py) / [normal.rs](src/distributions/normal.rs) |
| Canonical pattern, discrete | [_bernoulli.py](polars_stats/distributions/_bernoulli.py) / [bernoulli.rs](src/distributions/bernoulli.rs) |
| Canonical test layout | [tests/distributions/bernoulli/](tests/distributions/bernoulli/) |

## Conventions

Code, prose, test, and git conventions live in [Contributing > Conventions](./docs/contributing.md#conventions);
follow them, do not re-derive them here. On top of those:

* **Priorities, in order: correctness, maintainability, ergonomics, performance.** Performance last is deliberate:
  the polars engine does the heavy lifting on large frames, so a clear formula beats a fast one. Only reject a
  choice on performance grounds when it is clearly suboptimal (an `O(n)` draw per row, a per-draw rebuild), not to
  shave constants.
* Comments never narrate what the code does, and never reference tasks, fixes, or callers (that is PR-description
  material). Docstrings that pin a contract or invariant (null propagation, seeding, fast-path bit-equality) are the
  house style; everything else defaults to no comment.

## How to add a new distribution

The steps (Rust plugin surface, Python subclass shape, test files) are in
[Contributing > Adding a distribution](./docs/contributing.md#adding-a-distribution).

What matters specifically for an agent:

* **Work from the map, not the tree.** The architecture is already documented; read the contributing steps, the
  canonical pair for your family (closed-form `Uniform` vs statrs-backed `Normal`, plus `Bernoulli` for discrete),
  and the matching `tests/distributions/<canonical>/` rather than reverse-engineering the codebase. When you genuinely
  must explore further, fan the reads out in parallel. Resolve the choices the issue flags (a validator arity, a scipy
  reparam, an out-of-regime convention) up front from the issue file, not mid-implementation. In a repo this
  invariant-dense, understanding the fast-path / null / parity contracts beats debugging the property suite that
  encodes them.
* **The shared test registries are the trap.** Each needs one new entry per distribution; miss one and that suite
  silently skips your distribution, with the run still green. All five:
  [tests/property/_specs.py](tests/property/_specs.py) (`ALL_SPECS`, drives the whole property suite),
  [output_name_test.py](tests/distributions/output_name_test.py),
  [value_arg_str_test.py](tests/distributions/value_arg_str_test.py),
  [value_keyed_fast_path_test.py](tests/distributions/value_keyed_fast_path_test.py) and
  [moment_fast_path_test.py](tests/distributions/moment_fast_path_test.py) (scalar-vs-column validation contracts of
  the two fast paths, including invalid-parameter cases).
* **Write the scipy-parity test first**, then the implementation, then iterate until parity passes within tolerance.

## Common pitfalls

* **Row-alignment of scalar params.** Scalars must be coerced via `pl.repeat(value, n=pl.len(), dtype=pl.Float64())`,
   not `pl.lit(value)`. The reason: `is_elementwise=True` plugins under `over` / `group_by` expect length to track the
   partition. A bare `pl.lit` breaks this. This applies to the general (per-row) plugins; the sampler's
   constant-parameter fast path sidesteps it by passing the scalars in `kwargs` rather than as inputs.

* **The constant-parameter sampler fast path must match the per-row path bit for bit.** `<name>_sample_scalar` and the
   general sampler share `(root_seed, row_index)` seeding and the same draw, so for one seed they must agree. The fast
   path is the *only* place a distribution parameter rides in `kwargs`: a wrong field name or argument order silently
   feeds the wrong constant. `test_sample_scalar_fast_path_matches_per_row` (via the `make` vs `make_columns` spec
   builders) is what catches that; keep both builders in sync.

* **Method value arguments accept column names.** The base coerces a value / quantile arg with `as_expr`: a `str` means
   `pl.col(name)`, a `pl.Expr` passes through, a scalar or `pl.Series` becomes a literal. So `dist.pdf("x")` reads column
   `x`. Do not wrap a bare string in `pl.lit` yourself: `pl.lit("x")` is the string literal `"x"`, which the numeric
   plugins cast to all-null.

* **`is_elementwise=True` matters.** Without it, the plugin is treated as an aggregation under `over`.
   Every elementwise method (pdf, pmf, cdf, sf, ppf, log_*, sample) must pass it.

* **Null propagation.** A row with any null input must produce a null output. `try_binary_elementwise` /
   `try_ternary_elementwise` propagate nulls automatically; raw `.into_iter()` over chunks does not.

* **Invalid params on a row raise, they never silently null.** Map the `statrs` error through `PolarsError::InvalidOperation`
   and `?`-propagate (surfaces as `ComputeError`), consistent across distributions. Factor this into a
   `build_dist(...) -> PolarsResult<Dist>` helper, as every `.rs` file does.
   Null is reserved for null *inputs*, not invalid parameters.

* **Sample dtype is per distribution.** Bernoulli returns `Boolean`. Other discretes return `UInt64`. Continuous returns
   `Float64`. Do not normalise to `Float64` for "consistency".

* **Override the `_x` hook, not the public method.** The composing defaults (`_sf` = `1 - _cdf`,
   `_log_cdf` = `_cdf().log()`, ...) call the other hooks, so overriding public `cdf` instead of `_cdf` would be
   silently bypassed by `_sf`/`_log_cdf`. Always override `_cdf`, `_sf`, `_log_pdf`, etc.

* **Don't use the base-class `_log_pdf` / `_log_pmf` defaults if statrs has a native `ln_pdf` / `ln_pmf`.**
   The default is `_pdf(x).log()`, which underflows in the tails. Override the hook to bind the native log version.

* **`_log_cdf` / `_log_sf` have no native shortcut, and their defaults underflow.** Unlike `ln_pdf`, statrs
   exposes no `ln_cdf` / `ln_sf` (its `ContinuousCDF` / `DiscreteCDF` are `cdf` / `sf` / `inverse_cdf` only) and
   Polars has no `erf`, so there is no one-line bind. The base defaults `_cdf().log()` / `_sf().log()` return
   `-inf` once `cdf` / `sf` round to `0` in the tail (Normal: past ~38 sigma), the exact regime `log_sf` exists
   to serve. For a distribution used in tail work, override with either: a `log1p` / exact closed form
   (`Exponential._log_sf` = `-rate * x`, `Uniform._log_sf`), or a special-function port bound through the
   `value_keyed` machinery (`normal.rs` `ln_erfc` -> `normal_ln_cdf` / `normal_ln_sf`, wired exactly like
   `ln_pdf`). A related distribution can reuse an existing port: `LogNormal` composes the underlying normal's
   `ln_erfc` form with `ln(x)` (statrs `LogNormal` hides its `location` / `scale`, so build the underlying
   `Normal` from `(mu, sigma)` rather than reading them back off the `LogNormal`). Add a scipy-parity case
   probing *beyond* the underflow threshold (e.g. ~40 sigma for Normal, not merely `sf ~ 1e-300` where the
   naive path is still finite), the regime the finite grids miss. A statrs-backed family with no elementary
   tail can port its special function to log space instead: keep the underflow-prone prefactor in logs and
   reuse statrs's own series / continued-fraction evaluation verbatim; note scipy's `logcdf` / `logsf` may be
   naive there too, so the parity oracle past scipy's underflow is an independent series. The Gamma / Beta
   ports (`ln_gamma_lr` / `ln_gamma_ur`, `ln_beta_reg`) are the reference implementations; they live on the
   pending Gamma branch until it merges. Binomial still carries the naive default, documented.

* **`statrs` `inverse_cdf` is binary-search for several distributions** (Chi, Bernoulli, Binomial, Poisson,
   Geometric, NegBinom, Hypergeom, DiscreteUniform). Document the tolerance loosening (`1e-6`) in the relevant
   test. When that accuracy is not enough, a dedicated log-space inverse is the pattern (the pending Gamma
   branch's `inv_gamma_lr` / `inv_gamma_ur`, ~1e-14 relative in both tails, reusable for ChiSquared and friends).

* **scipy reparam.** Several distributions need a parameter mapping at the docstring level (Exp, Gamma, Pareto,
   LogNormal, Weibull, Uniform, Binomial, DiscreteUniform). Spell out the scipy equivalent so test writers know what to
   call.

## What not to do

The hard rules (no speculative refactors, no backwards-compatibility shims, no `DistKind` dispatch macro, no
destructive git operations, no `--no-verify`) are in [Contributing > Conventions](./docs/contributing.md#conventions).
Additionally, for agents:

* **Do not create planning, decision, or analysis documents** unless the user asks. Work from conversation context.
* **Do not add features beyond the issue scope.** If you find work that needs doing, propose it; do not silently extend
   the PR.
* **Do not add comment blocks narrating your changes.** New docstrings follow the house style above: contracts and
   invariants only.

## Running things

The Makefile targets and their rationale are in
[Contributing > Day-to-day commands](./docs/contributing.md#day-to-day-commands). Agent notes:

* Bernoulli is the smoke test for the whole pipeline, so after any Rust change a fast loop is:

    ```terminal
    make install-release && uv run pytest tests/distributions/bernoulli/ -x
    ```

* Pre-commit hooks are run by `prek` (locally via `make lint`, and in CI); do not bypass them with `--no-verify`
  unless asked.
