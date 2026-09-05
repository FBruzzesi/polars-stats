pub mod bernoulli;
pub mod beta;
pub mod binomial;
pub mod discrete_uniform;
pub mod exponential;
pub mod geometric;
pub mod lognormal;
pub mod normal;
pub mod uniform;

use std::borrow::Cow;

use polars::prelude::arity::{
    try_binary_elementwise, try_ternary_elementwise, try_unary_elementwise, unary_elementwise,
};
use polars::prelude::*;

/// Broadcast every length-1 input up to the call's row count, and reject lengths that cannot align.
///
/// The row count is the one input length that is not 1, so length-1 inputs never set it: an
/// all-length-1 call stays length 1, and a 0-row frame beside a `pl.lit` stays empty.
pub(crate) fn align_inputs(inputs: &[Series]) -> PolarsResult<Cow<'_, [Series]>> {
    let mut anchor: Option<&Series> = None;
    let mut needs_broadcast = false;

    for input in inputs {
        if input.len() == 1 {
            needs_broadcast = true;
        } else if let Some(anchor) = anchor {
            polars_ensure!(
                input.len() == anchor.len(),
                ShapeMismatch:
                "inputs have incompatible lengths: '{}' has length {} but '{}' has length {}. \
                 Every input must share one length, or be length 1 to broadcast",
                anchor.name(), anchor.len(), input.name(), input.len()
            );
        } else {
            anchor = Some(input);
        }
    }

    // Borrow when there is nothing to expand: no anchor means every input is already length 1, and
    // `!needs_broadcast` means they already agree, which is every call whose parameters are columns.
    let row_count = match anchor {
        Some(anchor) if needs_broadcast => anchor.len(),
        _ => return Ok(Cow::Borrowed(inputs)),
    };

    Ok(Cow::Owned(
        inputs
            .iter()
            .map(|input| {
                if input.len() == 1 {
                    input.new_from_index(0, row_count)
                } else {
                    input.clone()
                }
            })
            .collect(),
    ))
}

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
/// [`value_keyed_per_row`]) rather than per body, because two bodies genuinely need it and none
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

/// Shared driver for the column-parameter value-keyed per-row paths.
///
/// The column-parameter counterpart of [`value_keyed_scalar`]: at least one distribution parameter
/// is a column, so `build` validates and constructs once per row instead of once per call. `f` is
/// the same named per-method body the fast path applies (`cdf_value`, `ppf_value`, ...), so the two
/// paths cannot drift and agree bit for bit.
///
/// The caller does the cast and the accessor (`.f64()` / `.u64()`), which fixes `A` and `B`, so a
/// mixed `(u64, f64)` parameterisation (Binomial's `UInt64` `n` beside its `Float64` `p`) fits, as in
/// [`ternary_param_rows`](crate::rng::ternary_param_rows). `S` needs no trait bound: it is whatever
/// `build` returns.
///
/// Null contract: any null among `(value, p1, p2)` nulls the row without calling `build`, matching
/// the samplers.
///
/// `NaN` contract: as in [`value_keyed_scalar`], except that the short-circuit runs after `build`,
/// so an invalid parameterisation still raises on a `NaN` row.
///
/// Keep `build` and `f` generic `Fn`s: they monomorphise into the row loop, where a `&dyn Fn` or a
/// `fn` pointer would cost an indirect call per row.
pub(crate) fn value_keyed_per_row<A, B, S, Build, F>(
    value: &Float64Chunked,
    p1: &ChunkedArray<A>,
    p2: &ChunkedArray<B>,
    name: PlSmallStr,
    build: Build,
    f: F,
) -> PolarsResult<Series>
where
    A: PolarsNumericType,
    B: PolarsNumericType,
    Build: Fn(A::Native, B::Native) -> PolarsResult<S>,
    F: Fn(&S, f64) -> Option<f64>,
{
    let ca: Float64Chunked = try_ternary_elementwise(
        value,
        p1,
        p2,
        |value_opt, p1_opt, p2_opt| -> PolarsResult<Option<f64>> {
            match (value_opt, p1_opt, p2_opt) {
                (Some(value), Some(p1), Some(p2)) => {
                    // Build before the `NaN` short-circuit, so an invalid parameterisation still
                    // raises on a `NaN` row. Pinned by `tests/distributions/plugin_nan_test.py`.
                    let dist = build(p1, p2)?;
                    Ok(if value.is_nan() {
                        Some(f64::NAN)
                    } else {
                        f(&dist, value)
                    })
                },
                _ => Ok(None),
            }
        },
    )?;

    Ok(ca.with_name(name).into_series())
}

/// Shared driver for the two-parameter validation plugins: `validate` builds the distribution per
/// row and returns the `Float64` to emit, either a parameter itself (`sigma`) or a quantity derived
/// from both (Uniform's `max - min`), `?`-propagating the `InvalidOperation` out of `build_dist`.
/// Any null input nulls the row without calling `validate`, matching the samplers.
///
/// These plugins are what let the closed-form moments, and the value-keyed methods still assembled
/// in Polars (`Exponential`, `Geometric`, `Uniform`), report an invalid parameterisation through the
/// same Rust `build_dist`. The constant-parameter fast path calls the same plugin on length-1
/// `pl.lit` inputs, so it is built once instead of per row.
///
/// The two parameter dtypes are independent, so a mixed `(u64, f64)` parameterisation (Binomial)
/// fits, as in [`ternary_param_rows`](crate::rng::ternary_param_rows): the caller does the cast and
/// the accessor (`.f64()` / `.u64()`), which fixes `A` and `B`.
///
/// Takes no output name: polars resolves a plugin expression's output name from its first input, so
/// the frame column follows `inputs[0]` (pinned by `tests/distributions/output_name_test.py`).
///
/// Keep `validate` a generic `F: Fn`: it monomorphises into the row loop, where a `&dyn Fn` or a
/// `fn` pointer would cost an indirect call per row.
pub(crate) fn validate_params_binary<A, B, F>(
    a: &ChunkedArray<A>,
    b: &ChunkedArray<B>,
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

    Ok(ca.into_series())
}

/// Single-parameter counterpart of [`validate_params_binary`], same contracts
/// (`bernoulli_proba`, `exponential_rate`).
pub(crate) fn validate_params_unary<A, F>(a: &ChunkedArray<A>, validate: F) -> PolarsResult<Series>
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

    Ok(ca.into_series())
}
