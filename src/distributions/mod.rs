pub mod bernoulli;
pub mod beta;
pub mod binomial;
pub mod exponential;
pub mod lognormal;
pub mod normal;
pub mod uniform;

use polars::prelude::arity::{try_binary_elementwise, try_unary_elementwise, unary_elementwise};
use polars::prelude::*;

/// Shared driver for the constant-parameter value-keyed fast paths.
///
/// The constant-parameter counterpart of each distribution's `value_keyed` helper: when every
/// distribution parameter is a Python scalar, the caller validates them and builds the
/// distribution **once**, and only the evaluation-point column crosses FFI. This maps `f` over
/// that single column, where `f` is the same per-method body the per-row path uses (a named
/// function in each distribution file), so the two paths cannot drift and output is bit-identical.
///
/// Null contract: `value` is the only nullable input this driver sees. The distribution
/// parameters are not passed here at all; the caller validates them once (via `build_dist`,
/// which raises on an invalid or non-finite parameterisation before this runs) and bakes them
/// into `f` through a pre-built `dist`. So a null `value` propagates to null, and `f` returning
/// `None` nulls the row on the method's own terms (e.g. `ppf` outside `[0, 1]`), matching the
/// per-row path element for element.
///
/// `NaN` contract: a `NaN` evaluation point short-circuits to `NaN` (scipy semantics) before `f`
/// runs, for every method including `ppf`. The short-circuit is central (here and in
/// [`value_keyed_per_row!`]) rather than per body, because two bodies genuinely need it and none
/// may be forgotten: the regularized incomplete beta behind `Beta` `cdf`/`sf` panics on `NaN`
/// (aborting the whole query), and binomial's support mapping saturates (`NaN.floor() as u64` is
/// `0`), returning a confident `P(X <= 0)`. The Python-side `propagate_null_and_nan` guard cannot
/// substitute: polars evaluates every `when`/`then`/`otherwise` branch over the full column, so a
/// plugin runs on the `NaN` rows even though the guard discards their output. Pinned by
/// `tests/distributions/plugin_nan_test.py`.
pub(crate) fn value_keyed_scalar<F>(value: &Series, f: F) -> PolarsResult<Series>
where
    F: Fn(f64) -> Option<f64>,
{
    let value = value.cast(&DataType::Float64)?;
    let value_ca = value.f64()?;
    let name = value_ca.name().clone();

    let ca: Float64Chunked = unary_elementwise(value_ca, |opt| {
        opt.and_then(|v| if v.is_nan() { Some(f64::NAN) } else { f(v) })
    });
    Ok(ca.with_name(name).into_series())
}

/// Generates a distribution's per-row `value_keyed` helper: the column-parameter counterpart of
/// [`value_keyed_scalar`], building and validating the distribution **once per row** over
/// `(value, p1, p2)` and applying the per-method body `f`.
///
/// The general (non-fast-path) side of the value-keyed methods. Where the scalar fast path builds
/// the distribution once per call, this rebuilds it per row because at least one parameter is a
/// column. The three things that vary between distributions are the macro's inputs:
///
/// * the distribution type, used in the `F: Fn(&$dist, f64) -> Option<f64>` bound;
/// * each parameter's `<cast dtype> => <accessor>` pair (binomial's `n` is `Int64`/`.i64()`, the
///   continuous scales are `Float64`/`.f64()`); the evaluation point is always cast to `Float64`;
/// * `build`: the distribution constructor (`build_dist`), validating per row and raising on an
///   invalid parameterisation (`?` is available inside the closure).
///
/// `f` is the same named per-method body the scalar fast path applies, so the two paths share one
/// body and cannot drift. A `NaN` evaluation point short-circuits to `NaN` after the distribution
/// is built, so an invalid parameterisation still raises on a `NaN` row; see [`value_keyed_scalar`]
/// for why the guard lives in the shared drivers. Only the 2-parameter (ternary) arm exists today,
/// since every current distribution takes two parameters; a 1-parameter distribution would add a
/// `try_binary_elementwise` arm here. Call sites are the distribution modules
/// (`polars::prelude::*` in scope).
macro_rules! value_keyed_per_row {
    (
        $(#[$meta:meta])*
        fn $name:ident(&$dist:ty);
        params = ($p1_dt:expr => $p1_acc:ident, $p2_dt:expr => $p2_acc:ident);
        build = $build:path;
    ) => {
        $(#[$meta])*
        fn $name<F>(inputs: &[Series], f: F) -> PolarsResult<Series>
        where
            F: Fn(&$dist, f64) -> Option<f64>,
        {
            let value = inputs[0].cast(&DataType::Float64)?;
            let value_ca = value.f64()?;
            let p1 = inputs[1].cast(&$p1_dt)?;
            let p1_ca = p1.$p1_acc()?;
            let p2 = inputs[2].cast(&$p2_dt)?;
            let p2_ca = p2.$p2_acc()?;
            let name = inputs[0].name().clone();

            let ca: Float64Chunked = polars::prelude::arity::try_ternary_elementwise(
                value_ca,
                p1_ca,
                p2_ca,
                |value_opt, p1_opt, p2_opt| -> PolarsResult<Option<f64>> {
                    match (value_opt, p1_opt, p2_opt) {
                        (Some(v), Some(a), Some(b)) => {
                            // Build first so an invalid parameterisation raises on a `NaN` row.
                            let dist = $build(a, b)?;
                            Ok(if v.is_nan() { Some(f64::NAN) } else { f(&dist, v) })
                        },
                        _ => Ok(None),
                    }
                },
            )?;

            Ok(ca.with_name(name).into_series())
        }
    };
}
pub(crate) use value_keyed_per_row;

/// Shared driver for the two-parameter validation plugins: validate `(a, b)` per row by
/// constructing the distribution inside `validate`, and emit the `Float64` that closure returns.
///
/// These plugins exist so the closed-form moments and (for Uniform / Bernoulli) value-keyed
/// methods, which are pure Polars expressions, still report an invalid parameterisation through the
/// same Rust `build_dist` rather than silently producing a garbage result. The constant-parameter
/// fast path calls the same plugin on length-1 `pl.lit` inputs, so it is built once instead of per
/// row.
///
/// `validate` validates and derives in one step: it `?`-propagates the `InvalidOperation` out of
/// `build_dist` and returns the value to emit, either a parameter itself (`sigma`) or a quantity
/// derived from both (Uniform's `max - min`).
///
/// The two parameter dtypes are independent, so a mixed `(i64, f64)` parameterisation (Binomial)
/// fits, as in [`ternary_param_rows`](crate::rng::ternary_param_rows): the caller does the cast and
/// the accessor (`.f64()` / `.i64()`), which fixes `A` and `B`.
///
/// Null contract: any null input nulls the row without calling `validate`, matching the samplers.
///
/// Polars resolves a plugin expression's output name from its first input and ignores the name set
/// here, so the frame column follows `inputs[0]` whichever input is passed (pinned by
/// `tests/distributions/output_name_test.py`). `name` labels the returned `Series` only; callers
/// pass the input whose quantity they return.
///
/// Keep `validate` a generic `F: Fn`: it monomorphises into the row loop, where a `&dyn Fn` or a
/// `fn` pointer would cost an indirect call per row.
pub(crate) fn validate_params_binary<A, B, F>(
    a: &ChunkedArray<A>,
    b: &ChunkedArray<B>,
    name: PlSmallStr,
    validate: F,
) -> PolarsResult<Series>
where
    A: PolarsNumericType,
    B: PolarsNumericType,
    F: Fn(A::Native, B::Native) -> PolarsResult<f64>,
{
    let ca: Float64Chunked =
        try_binary_elementwise(a, b, |a_opt, b_opt| -> PolarsResult<Option<f64>> {
            match (a_opt, b_opt) {
                (Some(a), Some(b)) => Ok(Some(validate(a, b)?)),
                _ => Ok(None),
            }
        })?;

    Ok(ca.with_name(name).into_series())
}

/// Single-parameter counterpart of [`validate_params_binary`], same contracts
/// (`bernoulli_proba`, `exponential_rate`).
pub(crate) fn validate_params_unary<A, F>(
    a: &ChunkedArray<A>,
    name: PlSmallStr,
    validate: F,
) -> PolarsResult<Series>
where
    A: PolarsNumericType,
    F: Fn(A::Native) -> PolarsResult<f64>,
{
    let ca: Float64Chunked = try_unary_elementwise(a, |a_opt| -> PolarsResult<Option<f64>> {
        match a_opt {
            Some(a) => Ok(Some(validate(a)?)),
            None => Ok(None),
        }
    })?;

    Ok(ca.with_name(name).into_series())
}
