#![allow(clippy::unused_unit)]
use polars::prelude::arity::{try_binary_elementwise, try_ternary_elementwise};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distributions::Distribution as RandDistribution;
use serde::Deserialize;
use statrs::distribution::{Continuous, ContinuousCDF, Normal};

use crate::distributions::value_keyed_scalar;
use crate::rng::{sample_scalar_plugin, SampleKwargs};

/// Construct a `statrs::Normal`, mapping the invalid-parameter case to a `ComputeError`.
///
/// `statrs::Normal::new` rejects a non-finite `mean`, a `NaN` `std_dev`, or `std_dev <= 0`.
/// We surface that as `InvalidOperation` so an invalid scale fails the whole evaluation,
/// rather than silently nulling the row.
/// Validation lives here so every method that builds a distribution reports an invalid scale identically.
fn build_dist(mean: f64, std_dev: f64) -> PolarsResult<Normal> {
    Normal::new(mean, std_dev).map_err(|e| {
        PolarsError::InvalidOperation(
            format!(
                "std_dev must be finite and strictly positive, got mean={mean}, std_dev={std_dev}: {e}"
            )
            .into(),
        )
    })
}

/// Validate the `(mean, std_dev)` parameterisation and return the validated `std_dev`.
///
/// `inputs[0]` is `mean`, `inputs[1]` is `std_dev`. Mirrors `uniform_range`: the closed-form moments
/// (`mean`, `variance`, `median`, `entropy`) all derive from this single FFI round-trip, so they
/// report an invalid parameterisation identically to the value-keyed methods that build the
/// distribution directly. `null` in either input propagates; a `NaN` mean or a non-positive / `NaN`
/// `std_dev` raises `InvalidOperation` via [`build_dist`].
#[polars_expr(output_type=Float64)]
fn normal_std_dev(inputs: &[Series]) -> PolarsResult<Series> {
    let mean = inputs[0].cast(&DataType::Float64)?;
    let mean_ca = mean.f64()?;
    let std_dev = inputs[1].cast(&DataType::Float64)?;
    let std_dev_ca = std_dev.f64()?;
    let name = inputs[1].name().clone();

    let ca: Float64Chunked = try_binary_elementwise(
        mean_ca,
        std_dev_ca,
        |mean_opt, std_opt| -> PolarsResult<Option<f64>> {
            match (mean_opt, std_opt) {
                (Some(m), Some(s)) => {
                    build_dist(m, s)?;
                    Ok(Some(s))
                },
                _ => Ok(None),
            }
        },
    )?;

    Ok(ca.with_name(name).into_series())
}

/// Apply a value-keyed function `f(dist, value)` element-wise over `(value, mean, std_dev)`.
///
/// `inputs[0]` is the evaluation point, `inputs[1]` is `mean`, `inputs[2]` is `std_dev`. `null` in
/// any input propagates to `null`; an invalid `std_dev` raises via [`build_dist`]. `f` returns an
/// `Option` so a method can null a row on its own terms (e.g. `ppf` outside `[0, 1]`). Shared by
/// `pdf`, `ln_pdf`, `cdf`, `sf`, `ppf`.
fn value_keyed<F>(inputs: &[Series], f: F) -> PolarsResult<Series>
where
    F: Fn(&Normal, f64) -> Option<f64>,
{
    let value = inputs[0].cast(&DataType::Float64)?;
    let value_ca = value.f64()?;
    let mean = inputs[1].cast(&DataType::Float64)?;
    let mean_ca = mean.f64()?;
    let std_dev = inputs[2].cast(&DataType::Float64)?;
    let std_dev_ca = std_dev.f64()?;
    let name = inputs[0].name().clone();

    let ca: Float64Chunked = try_ternary_elementwise(
        value_ca,
        mean_ca,
        std_dev_ca,
        |value_opt, mean_opt, std_opt| -> PolarsResult<Option<f64>> {
            match (value_opt, mean_opt, std_opt) {
                (Some(v), Some(m), Some(s)) => {
                    let dist = build_dist(m, s)?;
                    Ok(f(&dist, v))
                },
                _ => Ok(None),
            }
        },
    )?;

    Ok(ca.with_name(name).into_series())
}

/// Element-wise Normal sampler.
///
/// `inputs[0]` carries `mean`, `inputs[1]` `std_dev`, and `inputs[2]` a per-row index used to derive
/// a per-row sub-seed, so the function is genuinely element-wise: chunking and threading cannot
/// change the output. With `seed=None`, a fresh root seed is drawn once per call.
///
/// Per-row validation:
///   * `null` (in any input) propagates;
///   * a `NaN` `mean`, or a non-positive / `NaN` `std_dev`, raises `InvalidOperation`.
///
/// Returns a `Float64` series.
#[polars_expr(output_type=Float64)]
fn normal_sample(inputs: &[Series], kwargs: SampleKwargs) -> PolarsResult<Series> {
    let mean = inputs[0].cast(&DataType::Float64)?;
    let mean_ca = mean.f64()?;
    let std_dev = inputs[1].cast(&DataType::Float64)?;
    let std_dev_ca = std_dev.f64()?;
    let index = inputs[2].cast(&DataType::UInt64)?;
    let index_ca = index.u64()?;
    let name = inputs[0].name().clone();

    let rngs = kwargs.row_rngs();

    let ca: Float64Chunked = try_ternary_elementwise(
        mean_ca,
        std_dev_ca,
        index_ca,
        |mean_opt, std_opt, i_opt| -> PolarsResult<Option<f64>> {
            match (mean_opt, std_opt, i_opt) {
                (Some(m), Some(s), Some(i)) => {
                    let dist = build_dist(m, s)?;
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
    /// Static parameters for the constant-parameter sampler fast path: when both `mean` and
    /// `std_dev` are Python scalars, they travel here as kwargs instead of as two full-length
    /// `pl.repeat` columns re-validated on every row.
    struct NormalScalarKwargs { mean: f64, std_dev: f64 }

    /// Constant-parameter Normal sampler: [`normal_sample`] for the all-scalar case. The
    /// distribution is validated and built once, and only the per-row index travels as an input;
    /// seeding and the draw are unchanged, so output matches `normal_sample` for the same
    /// `(seed, index, mean, std_dev)`.
    fn normal_sample_scalar(output_type = Float64, physical = Float64Type);
    build = |kw| build_dist(kw.mean, kw.std_dev)?;
    draw = |dist, rng| RandDistribution::sample(&dist, rng);
}

// Per-method bodies, named so the per-row plugins and the constant-parameter `*_scalar` twins
// share one definition each and cannot drift (the property test pinning their bit-equality only
// samples parameterisations; sharing the body makes it structural).

fn pdf_value(dist: &Normal, v: f64) -> Option<f64> {
    Some(dist.pdf(v))
}

fn ln_pdf_value(dist: &Normal, v: f64) -> Option<f64> {
    Some(dist.ln_pdf(v))
}

fn cdf_value(dist: &Normal, v: f64) -> Option<f64> {
    Some(dist.cdf(v))
}

fn sf_value(dist: &Normal, v: f64) -> Option<f64> {
    Some(dist.sf(v))
}

/// A quantile outside `[0, 1]` yields `null`; the closed endpoints map to the infinite tails
/// (`ppf(0) = -inf`, `ppf(1) = +inf`), matching `scipy.stats.norm.ppf`.
fn ppf_value(dist: &Normal, q: f64) -> Option<f64> {
    if !(0.0..=1.0).contains(&q) {
        None
    } else if q == 0.0 {
        Some(f64::NEG_INFINITY)
    } else if q == 1.0 {
        Some(f64::INFINITY)
    } else {
        Some(dist.inverse_cdf(q))
    }
}

/// Element-wise pdf via `statrs` `Continuous::pdf`. See [`value_keyed`] for the null/error contract.
#[polars_expr(output_type=Float64)]
fn normal_pdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, pdf_value)
}

/// Element-wise log-pdf via native `Continuous::ln_pdf` (more accurate than `pdf().ln()`).
#[polars_expr(output_type=Float64)]
fn normal_ln_pdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, ln_pdf_value)
}

/// Element-wise cdf via `statrs` `ContinuousCDF::cdf`.
#[polars_expr(output_type=Float64)]
fn normal_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, cdf_value)
}

/// Element-wise survival function via native `ContinuousCDF::sf` (accurate in the upper tail).
#[polars_expr(output_type=Float64)]
fn normal_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, sf_value)
}

/// Element-wise ppf (inverse cdf) via the closed-form `ContinuousCDF::inverse_cdf`.
/// See [`ppf_value`] for the endpoint and out-of-range contract.
#[polars_expr(output_type=Float64)]
fn normal_ppf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, ppf_value)
}

/// Static parameters for the constant-parameter value-keyed fast paths (`<method>_scalar`).
///
/// Like [`NormalScalarKwargs`] minus the sampler `seed`: when both parameters are Python scalars,
/// the Python layer routes them here as kwargs instead of expanding each into a full-length
/// column re-validated on every row. The distribution is validated and built once; only the
/// evaluation-point column travels as an input.
#[derive(Deserialize)]
struct NormalParamsKwargs {
    mean: f64,
    std_dev: f64,
}

/// Constant-parameter pdf; same body as [`normal_pdf`] via [`pdf_value`], dist built once.
#[polars_expr(output_type=Float64)]
fn normal_pdf_scalar(inputs: &[Series], kwargs: NormalParamsKwargs) -> PolarsResult<Series> {
    let dist = build_dist(kwargs.mean, kwargs.std_dev)?;
    value_keyed_scalar(&inputs[0], |v| pdf_value(&dist, v))
}

/// Constant-parameter log-pdf; same body as [`normal_ln_pdf`] via [`ln_pdf_value`], dist built once.
#[polars_expr(output_type=Float64)]
fn normal_ln_pdf_scalar(inputs: &[Series], kwargs: NormalParamsKwargs) -> PolarsResult<Series> {
    let dist = build_dist(kwargs.mean, kwargs.std_dev)?;
    value_keyed_scalar(&inputs[0], |v| ln_pdf_value(&dist, v))
}

/// Constant-parameter cdf; same body as [`normal_cdf`] via [`cdf_value`], dist built once.
#[polars_expr(output_type=Float64)]
fn normal_cdf_scalar(inputs: &[Series], kwargs: NormalParamsKwargs) -> PolarsResult<Series> {
    let dist = build_dist(kwargs.mean, kwargs.std_dev)?;
    value_keyed_scalar(&inputs[0], |v| cdf_value(&dist, v))
}

/// Constant-parameter sf; same body as [`normal_sf`] via [`sf_value`], dist built once.
#[polars_expr(output_type=Float64)]
fn normal_sf_scalar(inputs: &[Series], kwargs: NormalParamsKwargs) -> PolarsResult<Series> {
    let dist = build_dist(kwargs.mean, kwargs.std_dev)?;
    value_keyed_scalar(&inputs[0], |v| sf_value(&dist, v))
}

/// Constant-parameter ppf; same body as [`normal_ppf`] via [`ppf_value`], dist built once.
#[polars_expr(output_type=Float64)]
fn normal_ppf_scalar(inputs: &[Series], kwargs: NormalParamsKwargs) -> PolarsResult<Series> {
    let dist = build_dist(kwargs.mean, kwargs.std_dev)?;
    value_keyed_scalar(&inputs[0], |q| ppf_value(&dist, q))
}
