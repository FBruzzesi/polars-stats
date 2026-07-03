#![allow(clippy::unused_unit)]
use polars::prelude::arity::{try_binary_elementwise, try_ternary_elementwise};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distributions::Distribution as RandDistribution;
use statrs::distribution::{Beta, Continuous, ContinuousCDF};
use statrs::statistics::Distribution as StatrsDistribution;

use crate::distributions::{param_validator, value_keyed_per_row, value_keyed_scalar_plugins};
use crate::rng::{
    sample_scalar_plugin, samples_f64_output, samples_per_row, ternary_param_rows, SampleKwargs,
    SamplesKwargs,
};

/// Construct a `statrs::Beta`, mapping the invalid-parameter case to a `ComputeError`.
///
/// `statrs::Beta::new(shape_a, shape_b)` rejects a `NaN`, infinite, or non-positive shape.
/// We surface that as `InvalidOperation` so an invalid shape fails the whole evaluation
/// rather than silently nulling the row.
/// Validation lives here so every method that builds a distribution reports an invalid shape identically.
fn build_dist(a: f64, b: f64) -> PolarsResult<Beta> {
    Beta::new(a, b).map_err(|e| {
        PolarsError::InvalidOperation(
            format!("a and b must be finite and strictly positive, got a={a}, b={b}: {e}").into(),
        )
    })
}

param_validator! {
    /// Validate the `(a, b)` parameterisation and return the validated `b`.
    ///
    /// `inputs[0]` is `a`, `inputs[1]` is `b`. Mirrors `normal_std_dev` / `binomial_params`: the
    /// closed-form moments (`mean = a / (a + b)`, `variance = a * b / ((a + b)^2 * (a + b + 1))`)
    /// are computed from Polars expressions and gated on this single FFI round-trip, so they report
    /// an invalid parameterisation identically to the value-keyed methods that build the
    /// distribution directly. `null` in either input propagates; a `NaN`, infinite, or non-positive
    /// shape raises `InvalidOperation` via [`build_dist`].
    fn beta_params;
    params = (a: DataType::Float64 => f64, b: DataType::Float64 => f64);
    build = build_dist;
    returns = b;
    output_name = inputs[1];
}

value_keyed_per_row! {
    /// Apply a value-keyed function `f(dist, value)` element-wise over `(value, a, b)`.
    ///
    /// `inputs[0]` is the evaluation point, `inputs[1]` is `a`, `inputs[2]` is `b`. `null` in any
    /// input propagates to `null`; an invalid shape raises via [`build_dist`]. `f` returns an
    /// `Option` so a method can null a row on its own terms (e.g. `ppf` outside `[0, 1]`). Shared
    /// by `pdf`, `ln_pdf`, `cdf`, `sf`, `ppf`.
    fn value_keyed(&Beta);
    params = (DataType::Float64 => f64, DataType::Float64 => f64);
    build = build_dist;
}

/// Apply a parameter-keyed moment `f(dist)` element-wise over `(a, b)`.
///
/// `inputs[0]` is `a`, `inputs[1]` is `b`. `null` in either propagates; an invalid shape raises
/// via [`build_dist`]. Only `entropy` routes through here (the one moment without an elementary
/// closed form); `mean` and `variance` are closed forms computed in Polars and gated on
/// [`beta_params`].
fn params_keyed<F>(inputs: &[Series], f: F) -> PolarsResult<Series>
where
    F: Fn(&Beta) -> f64,
{
    let a = inputs[0].cast(&DataType::Float64)?;
    let a_ca = a.f64()?;
    let b = inputs[1].cast(&DataType::Float64)?;
    let b_ca = b.f64()?;
    let name = inputs[1].name().clone();

    let ca: Float64Chunked =
        try_binary_elementwise(a_ca, b_ca, |a_opt, b_opt| -> PolarsResult<Option<f64>> {
            match (a_opt, b_opt) {
                (Some(a), Some(b)) => {
                    let dist = build_dist(a, b)?;
                    Ok(Some(f(&dist)))
                },
                _ => Ok(None),
            }
        })?;

    Ok(ca.with_name(name).into_series())
}

/// Element-wise Beta sampler.
///
/// `inputs[0]` carries `a`, `inputs[1]` `b`, and `inputs[2]` a per-row index used to derive a
/// per-row sub-seed, so the function is genuinely element-wise: chunking and threading cannot
/// change the output. With `seed=None`, a fresh root seed is drawn once per call.
///
/// Per-row validation:
///   * `null` (in any input) propagates;
///   * a `NaN`, infinite, or non-positive shape raises `InvalidOperation`.
///
/// The draw keeps `statrs` (two `O(1)`-amortised Gamma draws, normalised); routing it through
/// `rand_distr` would buy nothing, since that is already the algorithm class it uses (unlike the
/// binomial draw, see DESIGN.md). Returns a `Float64` series.
#[polars_expr(output_type=Float64)]
fn beta_sample(inputs: &[Series], kwargs: SampleKwargs) -> PolarsResult<Series> {
    let a = inputs[0].cast(&DataType::Float64)?;
    let a_ca = a.f64()?;
    let b = inputs[1].cast(&DataType::Float64)?;
    let b_ca = b.f64()?;
    let index = inputs[2].cast(&DataType::UInt64)?;
    let index_ca = index.u64()?;
    let name = inputs[0].name().clone();

    let rngs = kwargs.row_rngs();

    let ca: Float64Chunked = try_ternary_elementwise(
        a_ca,
        b_ca,
        index_ca,
        |a_opt, b_opt, i_opt| -> PolarsResult<Option<f64>> {
            match (a_opt, b_opt, i_opt) {
                (Some(a), Some(b), Some(i)) => {
                    let dist = build_dist(a, b)?;
                    let mut rng = rngs.rng(i);
                    let draw: f64 = RandDistribution::sample(&dist, &mut rng);
                    Ok(Some(draw))
                },
                _ => Ok(None),
            }
        },
    )?;

    Ok(ca.with_name(name).into_series())
}

sample_scalar_plugin! {
    /// Static parameters for the constant-parameter sampler fast path: when both `a` and `b` are
    /// Python scalars, they travel here as kwargs instead of as two full-length `pl.repeat`
    /// columns re-validated on every row.
    struct BetaScalarKwargs { a: f64, b: f64 }

    /// Constant-parameter Beta sampler: [`beta_sample`] for the all-scalar case. The distribution
    /// is validated and built once, and only the per-row index travels as an input; seeding and
    /// the draw are unchanged, so output matches `beta_sample` for the same `(seed, index, a, b)`.
    fn beta_sample_scalar(output_type = Float64, physical = Float64Type);

    samples = beta_samples_scalar as BetaSamplesScalarKwargs -> samples_f64_output;

    build = |kw| build_dist(kw.a, kw.b)?;
    draw = |dist, rng| RandDistribution::sample(&dist, rng);
}

/// Element-wise multi-draw Beta sampler: `size` draws per row in one call.
///
/// The column-parameter counterpart of [`beta_samples_scalar`]: the distribution is built once per
/// row instead of once per draw. Row `i`'s draws come from the one stream seeded `(seed, i)`, so
/// output is bit-identical to the scalar path for the same parameters (the seeding is positional,
/// parameters never enter it, so equal-parameter rows still draw independently). Null/error
/// contract follows [`beta_sample`] per row; a null row yields a null array element. Returns
/// `Array(Float64, size)`.
#[polars_expr(output_type_func_with_kwargs=samples_f64_output)]
fn beta_samples(inputs: &[Series], kwargs: SamplesKwargs) -> PolarsResult<Series> {
    let a = inputs[0].cast(&DataType::Float64)?;
    let b = inputs[1].cast(&DataType::Float64)?;
    let index = inputs[2].cast(&DataType::UInt64)?;
    let name = inputs[0].name().clone();

    let rows = ternary_param_rows(a.f64()?, b.f64()?, index.u64()?, build_dist);

    samples_per_row::<Float64Type, _, _, _, _>(
        name,
        kwargs.seed,
        kwargs.size,
        rows,
        RandDistribution::sample,
    )
}

// Per-method bodies, named so the per-row plugins and the constant-parameter `*_scalar` twins
// share one definition each and cannot drift (the property test pinning their bit-equality only
// samples parameterisations; sharing the body makes it structural).

fn pdf_value(dist: &Beta, v: f64) -> Option<f64> {
    Some(dist.pdf(v))
}

fn ln_pdf_value(dist: &Beta, v: f64) -> Option<f64> {
    Some(dist.ln_pdf(v))
}

fn cdf_value(dist: &Beta, v: f64) -> Option<f64> {
    Some(dist.cdf(v))
}

fn sf_value(dist: &Beta, v: f64) -> Option<f64> {
    Some(dist.sf(v))
}

/// A quantile outside `[0, 1]` yields `null` (statrs' `inverse_cdf` panics there, so the guard is
/// load-bearing); the endpoints map to the support bounds (`ppf(0) = 0`, `ppf(1) = 1`), matching
/// `scipy.stats.beta.ppf`.
fn ppf_value(dist: &Beta, q: f64) -> Option<f64> {
    if !(0.0..=1.0).contains(&q) {
        None
    } else {
        Some(dist.inverse_cdf(q))
    }
}

/// Element-wise pdf via `statrs` `Continuous::pdf` (Beta function); `0` outside `[0, 1]`, and
/// divergent (`inf`) at a boundary whose shape is `< 1`. See [`value_keyed`] for the null/error
/// contract.
#[polars_expr(output_type=Float64)]
fn beta_pdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, pdf_value)
}

/// Element-wise log-pdf via native `Continuous::ln_pdf` (more accurate than `pdf().ln()`);
/// `-inf` outside `[0, 1]`.
#[polars_expr(output_type=Float64)]
fn beta_ln_pdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, ln_pdf_value)
}

/// Element-wise cdf via `statrs` `ContinuousCDF::cdf` (regularized incomplete beta); `0` below the
/// support, `1` at/above `1`.
#[polars_expr(output_type=Float64)]
fn beta_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, cdf_value)
}

/// Element-wise survival function via native `ContinuousCDF::sf` (accurate in the upper tail);
/// `1` below the support, `0` at/above `1`.
#[polars_expr(output_type=Float64)]
fn beta_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, sf_value)
}

/// Element-wise ppf via the closed-form `ContinuousCDF::inverse_cdf` (inverse regularized
/// incomplete beta). See [`ppf_value`] for the endpoint and out-of-range contract.
#[polars_expr(output_type=Float64)]
fn beta_ppf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, ppf_value)
}

value_keyed_scalar_plugins! {
    /// Static parameters for the constant-parameter value-keyed fast paths (`<method>_scalar`).
    ///
    /// Like [`BetaScalarKwargs`] minus the sampler `seed`: when both parameters are Python
    /// scalars, the Python layer routes them here as kwargs instead of expanding each into a
    /// full-length column re-validated on every row. The distribution is validated and built once;
    /// only the evaluation-point column travels as an input.
    struct BetaParamsKwargs { a: f64, b: f64 }

    build = |kw| build_dist(kw.a, kw.b)?;

    methods {
        /// Constant-parameter pdf; same body as [`beta_pdf`] via [`pdf_value`], dist built once.
        fn beta_pdf_scalar => pdf_value;

        /// Constant-parameter log-pdf; same body as [`beta_ln_pdf`] via [`ln_pdf_value`], dist built once.
        fn beta_ln_pdf_scalar => ln_pdf_value;

        /// Constant-parameter cdf; same body as [`beta_cdf`] via [`cdf_value`], dist built once.
        fn beta_cdf_scalar => cdf_value;

        /// Constant-parameter sf; same body as [`beta_sf`] via [`sf_value`], dist built once.
        fn beta_sf_scalar => sf_value;

        /// Constant-parameter ppf; same body as [`beta_ppf`] via [`ppf_value`], dist built once.
        fn beta_ppf_scalar => ppf_value;
    }
}

/// Element-wise differential entropy (in nats) via `statrs` `Distribution::entropy`:
/// `ln B(a, b) - (a - 1) psi(a) - (b - 1) psi(b) + (a + b - 2) psi(a + b)`.
///
/// Kept in Rust, unlike `mean` / `variance`: the beta entropy needs log-Beta and digamma, which
/// have no elementary closed form, so there is no numerically equivalent Polars expression to move
/// it to (the same reasoning as `binomial_entropy`).
#[polars_expr(output_type=Float64)]
fn beta_entropy(inputs: &[Series]) -> PolarsResult<Series> {
    params_keyed(inputs, |dist| dist.entropy().unwrap())
}
