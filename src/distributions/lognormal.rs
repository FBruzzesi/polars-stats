#![allow(clippy::unused_unit)]
use polars::prelude::arity::{try_binary_elementwise, try_ternary_elementwise};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distributions::Distribution as RandDistribution;
use statrs::distribution::{Continuous, ContinuousCDF, LogNormal};

use crate::rng::SampleKwargs;

/// Construct a `statrs::LogNormal`, mapping the invalid-parameter case to a `ComputeError`.
///
/// `statrs::LogNormal::new` rejects a `NaN` `mu`, a `NaN` `sigma`, or `sigma <= 0`. We surface that
/// as `InvalidOperation` so an invalid parameterisation fails the whole evaluation, rather than
/// silently nulling the row. Validation lives here so every method that builds a distribution
/// reports it identically.
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

/// Apply a value-keyed function `f(dist, value)` element-wise over `(value, mu, sigma)`.
///
/// `inputs[0]` is the evaluation point, `inputs[1]` is `mu`, `inputs[2]` is `sigma`. `null` in any
/// input propagates to `null`; an invalid parameterisation raises via [`build_dist`]. `f` returns an
/// `Option` so a method can null a row on its own terms (e.g. `ppf` outside `[0, 1]`). Shared by
/// `pdf`, `ln_pdf`, `cdf`, `sf`, `ppf`.
fn value_keyed<F>(inputs: &[Series], f: F) -> PolarsResult<Series>
where
    F: Fn(&LogNormal, f64) -> Option<f64>,
{
    let value = inputs[0].cast(&DataType::Float64)?;
    let value_ca = value.f64()?;
    let mu = inputs[1].cast(&DataType::Float64)?;
    let mu_ca = mu.f64()?;
    let sigma = inputs[2].cast(&DataType::Float64)?;
    let sigma_ca = sigma.f64()?;
    let name = inputs[0].name().clone();

    let ca: Float64Chunked = try_ternary_elementwise(
        value_ca,
        mu_ca,
        sigma_ca,
        |value_opt, mu_opt, sigma_opt| -> PolarsResult<Option<f64>> {
            match (value_opt, mu_opt, sigma_opt) {
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

/// Validate the `(mu, sigma)` parameterisation and return the validated `sigma`.
///
/// `inputs[0]` is `mu`, `inputs[1]` is `sigma`. Mirrors `normal_std_dev` / `uniform_range`: the
/// closed-form moments (`mean`, `variance`, `median`, `entropy`) are computed in Python and all
/// derive from this single FFI round-trip, so they report an invalid parameterisation identically to
/// the value-keyed methods that build the distribution directly. `null` in either input propagates;
/// a `NaN` `mu` or a non-positive / `NaN` `sigma` raises `InvalidOperation` via [`build_dist`].
#[polars_expr(output_type=Float64)]
fn lognormal_sigma(inputs: &[Series]) -> PolarsResult<Series> {
    let mu = inputs[0].cast(&DataType::Float64)?;
    let mu_ca = mu.f64()?;
    let sigma = inputs[1].cast(&DataType::Float64)?;
    let sigma_ca = sigma.f64()?;
    let name = inputs[1].name().clone();

    let ca: Float64Chunked = try_binary_elementwise(
        mu_ca,
        sigma_ca,
        |mu_opt, sigma_opt| -> PolarsResult<Option<f64>> {
            match (mu_opt, sigma_opt) {
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

/// Element-wise LogNormal sampler.
///
/// `inputs[0]` carries `mu`, `inputs[1]` `sigma`, and `inputs[2]` a per-row index used to derive a
/// per-row sub-seed, so the function is genuinely element-wise: chunking and threading cannot change
/// the output. With `seed=None`, a fresh root seed is drawn once per call.
///
/// Per-row validation:
///   * `null` (in any input) propagates;
///   * a `NaN` `mu`, or a non-positive / `NaN` `sigma`, raises `InvalidOperation`.
///
/// Returns a `Float64` series.
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

/// Element-wise pdf via `statrs` `Continuous::pdf`; `0` for `value <= 0` (outside the support).
/// See [`value_keyed`] for the null/error contract.
#[polars_expr(output_type=Float64)]
fn lognormal_pdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, |dist, v| Some(dist.pdf(v)))
}

/// Element-wise log-pdf via native `Continuous::ln_pdf` (more accurate than `pdf().ln()`);
/// `-inf` for `value <= 0`.
#[polars_expr(output_type=Float64)]
fn lognormal_ln_pdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, |dist, v| Some(dist.ln_pdf(v)))
}

/// Element-wise cdf via `statrs` `ContinuousCDF::cdf`; `0` for `value <= 0`.
#[polars_expr(output_type=Float64)]
fn lognormal_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, |dist, v| Some(dist.cdf(v)))
}

/// Element-wise survival function via native `ContinuousCDF::sf` (accurate in the upper tail);
/// `1` for `value <= 0`.
#[polars_expr(output_type=Float64)]
fn lognormal_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, |dist, v| Some(dist.sf(v)))
}

/// Element-wise ppf (inverse cdf) via the closed-form `ContinuousCDF::inverse_cdf`.
///
/// A quantile outside `[0, 1]` yields `null` (guarding the `statrs` panic). The closed endpoints map
/// to the support boundaries (`ppf(0) = 0`, `ppf(1) = +inf`), matching `scipy.stats.lognorm.ppf`.
#[polars_expr(output_type=Float64)]
fn lognormal_ppf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, |dist, q| {
        if !(0.0..=1.0).contains(&q) {
            None
        } else {
            Some(dist.inverse_cdf(q))
        }
    })
}
