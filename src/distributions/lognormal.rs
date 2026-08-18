#![allow(clippy::unused_unit)]
use polars::prelude::arity::try_ternary_elementwise;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distr::Distribution as RandDistribution;
use statrs::distribution::{Continuous, ContinuousCDF, LogNormal, Normal};

use crate::distributions::{
    normal, validate_params_binary, value_keyed_per_row, value_keyed_scalar,
};
use crate::rng::{
    sample_scalar_plugin, samples_f64_output, samples_per_row, ternary_param_rows, SampleKwargs,
    SamplesKwargs,
};

/// Construct a `statrs::LogNormal`, mapping the invalid-parameter case to a `ComputeError`.
///
/// `statrs::LogNormal::new` rejects a `NaN` `mu`, a `NaN` `sigma`, or `sigma <= 0`. That surfaces as
/// `InvalidOperation`, so an invalid parameterisation fails the whole evaluation rather than silently
/// nulling the row.
fn build_dist(mu: f64, sigma: f64) -> PolarsResult<LogNormal> {
    LogNormal::new(mu, sigma).map_err(|e| {
        PolarsError::InvalidOperation(
            format!(
                "sigma must be finite and strictly positive (mu must be finite), got mu={mu}, sigma={sigma}: {e}"
            )
            .into(),
        )
    })
}

/// The underlying `statrs::Normal` (mean `mu`, std-dev `sigma`) a built `LogNormal` wraps,
/// recovered through the `location()` / `scale()` accessors. Backs the stable `log_cdf` / `log_sf`
/// / `isf`, which reuse the normal's `ln_erfc` forms on `ln(x)`.
///
/// Infallible: any `(location, scale)` a built `LogNormal` carries is a valid `Normal`
/// parameterisation, so validation stays with [`build_dist`] alone.
fn underlying_normal(dist: &LogNormal) -> Normal {
    Normal::new(dist.location(), dist.scale())
        .expect("Normal::new accepts every (location, scale) a built LogNormal carries")
}

value_keyed_per_row! {
    /// Apply a value-keyed `f(dist, value)` element-wise over `(value, mu, sigma)`; shared by every
    /// Rust-bound value-keyed method. `null` propagates; an invalid parameterisation raises
    /// via [`build_dist`]; `f` may return `None` to null a row on its own terms.
    fn value_keyed(&LogNormal);
    params = (DataType::Float64 => f64, DataType::Float64 => f64);
    build = build_dist;
}

/// Validate the `(mu, sigma)` parameterisation and return the validated `sigma`.
///
/// `inputs[0]` is `mu`, `inputs[1]` is `sigma`. The Python closed-form moments all derive from
/// this single FFI round-trip, so they raise on an invalid parameterisation exactly like the
/// value-keyed methods. `null` in either input propagates; invalid raises via [`build_dist`].
#[polars_expr(output_type=Float64)]
fn lognormal_sigma(inputs: &[Series]) -> PolarsResult<Series> {
    let mu = inputs[0].cast(&DataType::Float64)?;
    let sigma = inputs[1].cast(&DataType::Float64)?;
    let name = inputs[1].name().clone();

    validate_params_binary(mu.f64()?, sigma.f64()?, name, |mu, sigma| {
        build_dist(mu, sigma)?;
        Ok(sigma)
    })
}

/// Element-wise LogNormal sampler over `(mu, sigma, row_index)`, returning `Float64`.
///
/// Per row, `null` propagates and an invalid parameterisation raises via [`build_dist`]. Seeding
/// and chunk-invariance follow [`SampleKwargs::row_rngs`].
#[polars_expr(output_type=Float64)]
fn lognormal_sample(inputs: &[Series], kwargs: SampleKwargs) -> PolarsResult<Series> {
    let mu = inputs[0].cast(&DataType::Float64)?;
    let mu_ca = mu.f64()?;
    let sigma = inputs[1].cast(&DataType::Float64)?;
    let sigma_ca = sigma.f64()?;
    let index = inputs[2].cast(&DataType::UInt64)?;
    let index_ca = index.u64()?;
    let name = inputs[0].name().clone();

    let rngs = kwargs.row_rngs()?;

    let ca: Float64Chunked = try_ternary_elementwise(
        mu_ca,
        sigma_ca,
        index_ca,
        |mu_opt, sigma_opt, i_opt| -> PolarsResult<Option<f64>> {
            match (mu_opt, sigma_opt, i_opt) {
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
    struct LogNormalScalarKwargs { mu: f64, sigma: f64 }

    /// Constant-parameter fast path for [`lognormal_sample`].
    fn lognormal_sample_scalar(output_type = Float64, physical = Float64Type);

    samples = lognormal_samples_scalar as LogNormalSamplesScalarKwargs -> samples_f64_output;

    build = |kw| build_dist(kw.mu, kw.sigma)?;
    draw = |dist, rng| RandDistribution::sample(&dist, rng);
}

/// Element-wise multi-draw LogNormal sampler: `size` draws per row in one call, the distribution
/// built once per row. Returns `Array(Float64, size)`.
///
/// Seeding and the null/error contract follow [`samples_per_row`] and [`lognormal_sample`].
#[polars_expr(output_type_func_with_kwargs=samples_f64_output)]
fn lognormal_samples(inputs: &[Series], kwargs: SamplesKwargs) -> PolarsResult<Series> {
    let mu = inputs[0].cast(&DataType::Float64)?;
    let sigma = inputs[1].cast(&DataType::Float64)?;
    let index = inputs[2].cast(&DataType::UInt64)?;
    let name = inputs[0].name().clone();

    let rows = ternary_param_rows(mu.f64()?, sigma.f64()?, index.u64()?, build_dist);

    samples_per_row::<Float64Type, _, _, _, _>(
        name,
        kwargs.seed,
        kwargs.size,
        rows,
        RandDistribution::sample,
    )
}

// Per-method bodies, shared by the per-row plugins and their `*_scalar` twins.

fn pdf_value(dist: &LogNormal, v: f64) -> Option<f64> {
    Some(dist.pdf(v))
}

fn ln_pdf_value(dist: &LogNormal, v: f64) -> Option<f64> {
    Some(dist.ln_pdf(v))
}

fn cdf_value(dist: &LogNormal, v: f64) -> Option<f64> {
    Some(dist.cdf(v))
}

fn sf_value(dist: &LogNormal, v: f64) -> Option<f64> {
    Some(dist.sf(v))
}

/// Stable log-cdf: the underlying normal's [`normal::ln_cdf_value`] at `ln(v)`; `-inf` for `v <= 0`
/// (`cdf = 0` outside the support). See [`underlying_normal`] for the recovery.
fn ln_cdf_value(dist: &LogNormal, v: f64) -> Option<f64> {
    if v <= 0.0 {
        Some(f64::NEG_INFINITY)
    } else {
        normal::ln_cdf_value(&underlying_normal(dist), v.ln())
    }
}

/// Stable log-sf: the underlying normal's [`normal::ln_sf_value`] at `ln(v)`; `0` for `v <= 0`
/// (`sf = 1` outside the support).
fn ln_sf_value(dist: &LogNormal, v: f64) -> Option<f64> {
    if v <= 0.0 {
        Some(0.0)
    } else {
        normal::ln_sf_value(&underlying_normal(dist), v.ln())
    }
}

/// A quantile outside `[0, 1]` yields `null` (guarding the `statrs` panic). The closed endpoints
/// map to the support boundaries (`ppf(0) = 0`, `ppf(1) = +inf`), matching `scipy.stats.lognorm.ppf`.
fn ppf_value(dist: &LogNormal, q: f64) -> Option<f64> {
    if !(0.0..=1.0).contains(&q) {
        None
    } else {
        Some(dist.inverse_cdf(q))
    }
}

/// Inverse survival function, the underlying normal's [`normal::isf_value`] exponentiated.
///
/// Not `ppf(1 - q)`: see [`normal::isf_value`] for why the complement is unrecoverable for a tiny
/// `q`. Composing through `exp` turns the normal's *absolute* error at the quantile into a
/// *relative* one here, so a large `sigma` amplifies it.
///
/// The endpoints follow from the normal's: `isf(0) = exp(+inf) = +inf` and `isf(1) = exp(-inf) = 0`,
/// which are the support boundaries `ppf` maps in the other order.
fn isf_value(dist: &LogNormal, q: f64) -> Option<f64> {
    normal::isf_value(&underlying_normal(dist), q).map(f64::exp)
}

/// Element-wise pdf via `statrs` `Continuous::pdf`; `0` for `value <= 0` (outside the support).
/// See [`value_keyed`] for the null/error contract.
#[polars_expr(output_type=Float64)]
fn lognormal_pdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, pdf_value)
}

/// Element-wise log-pdf via native `Continuous::ln_pdf` (more accurate than `pdf().ln()`);
/// `-inf` for `value <= 0`.
#[polars_expr(output_type=Float64)]
fn lognormal_ln_pdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, ln_pdf_value)
}

/// Element-wise cdf via `statrs` `ContinuousCDF::cdf`; `0` for `value <= 0`.
#[polars_expr(output_type=Float64)]
fn lognormal_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, cdf_value)
}

/// Element-wise survival function via native `ContinuousCDF::sf` (accurate in the upper tail);
/// `1` for `value <= 0`.
#[polars_expr(output_type=Float64)]
fn lognormal_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, sf_value)
}

/// Element-wise log-cdf via the underlying normal's stable `ln_erfc` form (finite in the far-left
/// tail, unlike `cdf().ln()`); `-inf` for `value <= 0`.
#[polars_expr(output_type=Float64)]
fn lognormal_ln_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, ln_cdf_value)
}

/// Element-wise log-sf via the underlying normal's stable `ln_erfc` form (finite in the far-right
/// tail, unlike `sf().ln()`); `0` for `value <= 0`.
#[polars_expr(output_type=Float64)]
fn lognormal_ln_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, ln_sf_value)
}

/// Element-wise ppf (inverse cdf) via the closed-form `ContinuousCDF::inverse_cdf`.
/// See [`ppf_value`] for the endpoint and out-of-range contract.
#[polars_expr(output_type=Float64)]
fn lognormal_ppf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, ppf_value)
}

/// Element-wise isf via the underlying normal's symmetry form, not `ppf(1 - q)`.
/// See [`isf_value`] for why, and for the endpoint and out-of-range contract.
#[polars_expr(output_type=Float64)]
fn lognormal_isf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, isf_value)
}

/// Constant parameters for LogNormal's value-keyed fast paths, deserialised once per call.
#[derive(serde::Deserialize)]
struct LogNormalParamsKwargs {
    mu: f64,
    sigma: f64,
}

impl LogNormalParamsKwargs {
    /// Validate the parameters and build the distribution **once per call**, outside the row loop.
    ///
    /// A swapped or misnamed field compiles, runs, and returns wrong numbers. Pinned by
    /// `tests/property/value_keyed_test.py`, which asserts exact scalar-vs-per-row equality.
    fn build(&self) -> PolarsResult<LogNormal> {
        build_dist(self.mu, self.sigma)
    }
}

/// Constant-parameter fast path for [`lognormal_pdf`].
#[polars_expr(output_type=Float64)]
fn lognormal_pdf_scalar(inputs: &[Series], kwargs: LogNormalParamsKwargs) -> PolarsResult<Series> {
    let dist = kwargs.build()?;
    value_keyed_scalar(&inputs[0], |v| pdf_value(&dist, v))
}

/// Constant-parameter fast path for [`lognormal_ln_pdf`].
#[polars_expr(output_type=Float64)]
fn lognormal_ln_pdf_scalar(
    inputs: &[Series],
    kwargs: LogNormalParamsKwargs,
) -> PolarsResult<Series> {
    let dist = kwargs.build()?;
    value_keyed_scalar(&inputs[0], |v| ln_pdf_value(&dist, v))
}

/// Constant-parameter fast path for [`lognormal_cdf`].
#[polars_expr(output_type=Float64)]
fn lognormal_cdf_scalar(inputs: &[Series], kwargs: LogNormalParamsKwargs) -> PolarsResult<Series> {
    let dist = kwargs.build()?;
    value_keyed_scalar(&inputs[0], |v| cdf_value(&dist, v))
}

/// Constant-parameter fast path for [`lognormal_sf`].
#[polars_expr(output_type=Float64)]
fn lognormal_sf_scalar(inputs: &[Series], kwargs: LogNormalParamsKwargs) -> PolarsResult<Series> {
    let dist = kwargs.build()?;
    value_keyed_scalar(&inputs[0], |v| sf_value(&dist, v))
}

/// Constant-parameter fast path for [`lognormal_ppf`].
#[polars_expr(output_type=Float64)]
fn lognormal_ppf_scalar(inputs: &[Series], kwargs: LogNormalParamsKwargs) -> PolarsResult<Series> {
    let dist = kwargs.build()?;
    value_keyed_scalar(&inputs[0], |v| ppf_value(&dist, v))
}

/// Constant-parameter fast path for [`lognormal_ln_cdf`].
#[polars_expr(output_type=Float64)]
fn lognormal_ln_cdf_scalar(
    inputs: &[Series],
    kwargs: LogNormalParamsKwargs,
) -> PolarsResult<Series> {
    let dist = kwargs.build()?;
    value_keyed_scalar(&inputs[0], |v| ln_cdf_value(&dist, v))
}

/// Constant-parameter fast path for [`lognormal_ln_sf`].
#[polars_expr(output_type=Float64)]
fn lognormal_ln_sf_scalar(
    inputs: &[Series],
    kwargs: LogNormalParamsKwargs,
) -> PolarsResult<Series> {
    let dist = kwargs.build()?;
    value_keyed_scalar(&inputs[0], |v| ln_sf_value(&dist, v))
}

/// Constant-parameter fast path for [`lognormal_isf`].
#[polars_expr(output_type=Float64)]
fn lognormal_isf_scalar(inputs: &[Series], kwargs: LogNormalParamsKwargs) -> PolarsResult<Series> {
    let dist = kwargs.build()?;
    value_keyed_scalar(&inputs[0], |v| isf_value(&dist, v))
}
