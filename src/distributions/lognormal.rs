#![allow(clippy::unused_unit)]
use polars::prelude::arity::try_ternary_elementwise;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distributions::Distribution as RandDistribution;
use statrs::distribution::{Continuous, ContinuousCDF, LogNormal, Normal};

use crate::distributions::{
    normal, param_validator, value_keyed_per_row, value_keyed_scalar_plugins,
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

/// Construct the *underlying* `statrs::Normal` (mean `mu`, std-dev `sigma`) for the stable
/// `log_cdf` / `log_sf`, which reuse the normal's `ln_erfc` form on `ln(x)`.
///
/// `statrs::LogNormal` hides its `(location, scale)` (no accessors), so the value functions cannot
/// recover `(mu, sigma)` from a built `LogNormal`; they receive this underlying `Normal` instead.
/// Validation is delegated to [`build_dist`], so an invalid parameterisation reports the identical
/// error (down to the statrs suffix) as every other method.
fn build_underlying_normal(mu: f64, sigma: f64) -> PolarsResult<Normal> {
    build_dist(mu, sigma)?;
    Ok(Normal::new(mu, sigma)
        .expect("Normal::new accepts every (mu, sigma) LogNormal::new accepts"))
}

value_keyed_per_row! {
    /// Apply a value-keyed `f(dist, value)` element-wise over `(value, mu, sigma)`; shared by
    /// `pdf`, `ln_pdf`, `cdf`, `sf`, `ppf`. `null` propagates; an invalid parameterisation raises
    /// via [`build_dist`]; `f` may return `None` to null a row on its own terms.
    fn value_keyed(&LogNormal);
    params = (DataType::Float64 => f64, DataType::Float64 => f64);
    build = build_dist;
}

value_keyed_per_row! {
    /// Apply a value-keyed function `f(&Normal, value)` over `(value, mu, sigma)`, building the
    /// *underlying* normal (see [`build_underlying_normal`]) rather than the `LogNormal`.
    ///
    /// Backs the stable `log_cdf` / `log_sf` only: they compute the underlying normal's log-cdf / log-sf
    /// at `ln(value)`, which `statrs::LogNormal`'s hidden parameters would not let them reach via
    /// [`value_keyed`]. Null / invalid-parameter contract is identical to [`value_keyed`].
    fn value_keyed_norm(&Normal);
    params = (DataType::Float64 => f64, DataType::Float64 => f64);
    build = build_underlying_normal;
}

param_validator! {
    /// Validate the `(mu, sigma)` parameterisation and return the validated `sigma`.
    ///
    /// `inputs[0]` is `mu`, `inputs[1]` is `sigma`. The Python closed-form moments all derive from
    /// this single FFI round-trip, so they raise on an invalid parameterisation exactly like the
    /// value-keyed methods. `null` in either input propagates; invalid raises via [`build_dist`].
    fn lognormal_sigma;
    params = (mu: DataType::Float64 => f64, sigma: DataType::Float64 => f64);
    build = build_dist;
    returns = sigma;
    output_name = inputs[1];
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

    let rngs = kwargs.row_rngs();

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
/// (`cdf = 0` outside the support). `norm` is the underlying normal built by [`build_underlying_normal`].
fn ln_cdf_value(norm: &Normal, v: f64) -> Option<f64> {
    if v <= 0.0 {
        Some(f64::NEG_INFINITY)
    } else {
        normal::ln_cdf_value(norm, v.ln())
    }
}

/// Stable log-sf: the underlying normal's [`normal::ln_sf_value`] at `ln(v)`; `0` for `v <= 0`
/// (`sf = 1` outside the support).
fn ln_sf_value(norm: &Normal, v: f64) -> Option<f64> {
    if v <= 0.0 {
        Some(0.0)
    } else {
        normal::ln_sf_value(norm, v.ln())
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
    value_keyed_norm(inputs, ln_cdf_value)
}

/// Element-wise log-sf via the underlying normal's stable `ln_erfc` form (finite in the far-right
/// tail, unlike `sf().ln()`); `0` for `value <= 0`.
#[polars_expr(output_type=Float64)]
fn lognormal_ln_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_norm(inputs, ln_sf_value)
}

/// Element-wise ppf (inverse cdf) via the closed-form `ContinuousCDF::inverse_cdf`.
/// See [`ppf_value`] for the endpoint and out-of-range contract.
#[polars_expr(output_type=Float64)]
fn lognormal_ppf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, ppf_value)
}

value_keyed_scalar_plugins! {
    struct LogNormalParamsKwargs { mu: f64, sigma: f64 }

    build = |kw| build_dist(kw.mu, kw.sigma)?;

    methods {
        fn lognormal_pdf_scalar => pdf_value;
        fn lognormal_ln_pdf_scalar => ln_pdf_value;
        fn lognormal_cdf_scalar => cdf_value;
        fn lognormal_sf_scalar => sf_value;
        fn lognormal_ppf_scalar => ppf_value;
    }
}

value_keyed_scalar_plugins! {
    /// Constant-parameter fast path for the stable `log_cdf` / `log_sf`, building the *underlying*
    /// normal once (see [`build_underlying_normal`]) rather than the `LogNormal`.
    ///
    /// The scalar twin of [`value_keyed_norm`]: same field names as [`LogNormalParamsKwargs`]
    /// (`mu`, `sigma`, so the Python layer routes either method's scalar params here unchanged), but
    /// `build` returns the underlying `Normal` the `ln_erfc` bodies need.
    struct LogNormalLogKwargs { mu: f64, sigma: f64 }

    build = |kw| build_underlying_normal(kw.mu, kw.sigma)?;

    methods {
        fn lognormal_ln_cdf_scalar => ln_cdf_value;
        fn lognormal_ln_sf_scalar => ln_sf_value;
    }
}
