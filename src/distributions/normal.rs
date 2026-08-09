#![allow(clippy::unused_unit)]
use std::f64::consts::{LN_2, SQRT_2};

use polars::prelude::arity::try_ternary_elementwise;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distributions::Distribution as RandDistribution;
use statrs::distribution::{Continuous, ContinuousCDF, Normal};
use statrs::function::erf;
use statrs::statistics::Distribution as StatrsDistribution;

use crate::distributions::{param_validator, value_keyed_per_row, value_keyed_scalar_plugins};
use crate::rng::{
    sample_scalar_plugin, samples_f64_output, samples_per_row, ternary_param_rows, SampleKwargs,
    SamplesKwargs,
};

/// Construct a `statrs::Normal`, mapping the invalid-parameter case to a `ComputeError`.
///
/// `statrs::Normal::new` rejects a non-finite `mu`, a `NaN` `sigma`, or `sigma <= 0`. That surfaces as
/// `InvalidOperation`, so an invalid scale fails the whole evaluation rather than silently nulling the row.
fn build_dist(mu: f64, sigma: f64) -> PolarsResult<Normal> {
    Normal::new(mu, sigma).map_err(|e| {
        PolarsError::InvalidOperation(
            format!("sigma must be finite and strictly positive, got mu={mu}, sigma={sigma}: {e}")
                .into(),
        )
    })
}

param_validator! {
    /// Validate the `(mu, sigma)` parameterisation and return the validated `sigma`.
    ///
    /// `inputs[0]` is `mu`, `inputs[1]` is `sigma`. The Python closed-form moments all derive from
    /// this single FFI round-trip, so they raise on an invalid parameterisation exactly like the
    /// value-keyed methods. `null` in either input propagates; invalid raises via [`build_dist`].
    fn normal_sigma;
    params = (mu: DataType::Float64 => f64, sigma: DataType::Float64 => f64);
    build = build_dist;
    returns = sigma;
    output_name = inputs[1];
}

value_keyed_per_row! {
    /// Apply a value-keyed `f(dist, value)` element-wise over `(value, mu, sigma)`; shared by
    /// `pdf`, `ln_pdf`, `cdf`, `sf`, `ppf`. `null` propagates; an invalid parameterisation raises
    /// via [`build_dist`]; `f` may return `None` to null a row on its own terms.
    fn value_keyed(&Normal);
    params = (DataType::Float64 => f64, DataType::Float64 => f64);
    build = build_dist;
}

/// Element-wise Normal sampler over `(mu, sigma, row_index)`, returning `Float64`.
///
/// Per row, `null` propagates and an invalid parameterisation raises via [`build_dist`]. Seeding
/// and chunk-invariance follow [`SampleKwargs::row_rngs`].
#[polars_expr(output_type=Float64)]
fn normal_sample(inputs: &[Series], kwargs: SampleKwargs) -> PolarsResult<Series> {
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
    struct NormalScalarKwargs { mu: f64, sigma: f64 }

    /// Constant-parameter fast path for [`normal_sample`].
    fn normal_sample_scalar(output_type = Float64, physical = Float64Type);

    samples = normal_samples_scalar as NormalSamplesScalarKwargs -> samples_f64_output;

    build = |kw| build_dist(kw.mu, kw.sigma)?;
    draw = |dist, rng| RandDistribution::sample(&dist, rng);
}

/// Element-wise multi-draw Normal sampler: `size` draws per row in one call, the distribution
/// built once per row. Returns `Array(Float64, size)`.
///
/// Seeding and the null/error contract follow [`samples_per_row`] and [`normal_sample`].
#[polars_expr(output_type_func_with_kwargs=samples_f64_output)]
fn normal_samples(inputs: &[Series], kwargs: SamplesKwargs) -> PolarsResult<Series> {
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

/// Point past which `erfc(t)` is close enough to underflowing (`erfc ~ 1e-274` at `t = 25`,
/// underflow near `t = 26.6`) that we switch `ln_erfc` to the asymptotic series.
const LN_ERFC_ASYMPTOTIC_MIN: f64 = 25.0;

/// Natural log of the complementary error function, stable in the right tail.
///
/// `erfc(t).ln()` returns `-inf` once `erfc(t)` rounds to `0` (`t` above ~26.6), the regime `log_cdf`
/// / `log_sf` exist to serve. Past [`LN_ERFC_ASYMPTOTIC_MIN`] we switch to the asymptotic expansion
/// `erfc(t) = exp(-t^2) / (t * sqrt(pi)) * (1 - 1/(2t^2) + 3/(4t^4) - 15/(8t^6) + ...)`, whose log is a
/// large finite negative number; below it, statrs `erfc` holds full relative precision so the direct
/// form is exact. The two branches agree to ~1e-13 at the crossover.
fn ln_erfc(t: f64) -> f64 {
    if t < LN_ERFC_ASYMPTOTIC_MIN {
        erf::erfc(t).ln()
    } else {
        let t2 = t * t;
        let u = 0.5 / t2;
        let series = 1.0 - u * (1.0 - 3.0 * u * (1.0 - 5.0 * u * (1.0 - 7.0 * u)));
        -t2 - t.ln() - 0.5 * statrs::consts::LN_PI + series.ln()
    }
}

/// `ln(0.5 * erfc(s))` with full relative precision on both sides: [`ln_erfc`] for the small half
/// (`s >= 0`), `ln_1p` for the near-one half (`s < 0`, where `erfc(s) = 2 - erfc(-s)` rounds to `2`
/// and the direct log collapses to `0` instead of the true tiny negative, e.g. `-7.6e-24` at
/// 10 sigma; scipy's `log_ndtr` takes the same branch).
fn ln_half_erfc(s: f64) -> f64 {
    if s >= 0.0 {
        -LN_2 + ln_erfc(s)
    } else {
        (-0.5 * erf::erfc(-s)).ln_1p()
    }
}

/// Standardise `x` into the `erfc` argument `t = (x - mu) / (sigma * sqrt(2))`, so that
/// `cdf(x) = 0.5 * erfc(-t)` and `sf(x) = 0.5 * erfc(t)` (statrs' own `cdf` / `sf` definition).
fn erfc_arg(dist: &Normal, x: f64) -> f64 {
    let mu = dist.mean().expect("Normal always has a mean");
    let sigma = dist.std_dev().expect("Normal always has a std_dev");
    (x - mu) / (sigma * SQRT_2)
}

/// Native log-cdf via `ln(0.5 * erfc(-t))`: finite in the left tail (no `cdf().ln()` underflow) and
/// full relative precision in the right one (see [`ln_half_erfc`]).
///
/// `pub(crate)` so `LogNormal` reuses it on the underlying normal at `ln(x)` (log-normal log-cdf is
/// the underlying normal's log-cdf composed with `ln`).
pub(crate) fn ln_cdf_value(dist: &Normal, v: f64) -> Option<f64> {
    Some(ln_half_erfc(-erfc_arg(dist, v)))
}

/// Native log-sf via `ln(0.5 * erfc(t))`: finite in the right tail (no `sf().ln()` underflow) and
/// full relative precision in the left one (see [`ln_half_erfc`]).
///
/// `pub(crate)` for the same reason as [`ln_cdf_value`].
pub(crate) fn ln_sf_value(dist: &Normal, v: f64) -> Option<f64> {
    Some(ln_half_erfc(erfc_arg(dist, v)))
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

/// Inverse survival function, `mu + sigma * sqrt(2) * erfc_inv(2q)`, solved on `q` rather than on
/// its complement.
///
/// Deliberately not `ppf(1 - q)`, the base-class default: that composes `statrs`' `inverse_cdf`
/// into `erfc_inv(2 - 2q)`, whose argument resolves to `2.2e-16` absolute, so the tail mass is
/// quantised before the inverse runs. The symmetry `z_(1-q) = -z_q` puts the sign on the scale
/// instead, leaving the exact power-of-two `2q` as the only thing the inverse sees.
///
/// Contract mirrors [`ppf_value`]: `null` outside `[0, 1]`, closed endpoints to the infinite tails
/// (`isf(0) = +inf`, `isf(1) = -inf`, the reverse of `ppf`). `pub(crate)` so `LogNormal` composes it.
pub(crate) fn isf_value(dist: &Normal, q: f64) -> Option<f64> {
    if !(0.0..=1.0).contains(&q) {
        None
    } else if q == 0.0 {
        Some(f64::INFINITY)
    } else if q == 1.0 {
        Some(f64::NEG_INFINITY)
    } else {
        let mu = dist.mean().expect("Normal always has a mean");
        let sigma = dist.std_dev().expect("Normal always has a std_dev");
        Some(mu + sigma * SQRT_2 * erf::erfc_inv(2.0 * q))
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

/// Element-wise log-cdf via the stable [`ln_erfc`] form (finite in the left tail, unlike `cdf().ln()`).
#[polars_expr(output_type=Float64)]
fn normal_ln_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, ln_cdf_value)
}

/// Element-wise log-sf via the stable [`ln_erfc`] form (finite in the right tail, unlike `sf().ln()`).
#[polars_expr(output_type=Float64)]
fn normal_ln_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, ln_sf_value)
}

/// Element-wise ppf (inverse cdf) via the closed-form `ContinuousCDF::inverse_cdf`.
/// See [`ppf_value`] for the endpoint and out-of-range contract.
#[polars_expr(output_type=Float64)]
fn normal_ppf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, ppf_value)
}

/// Element-wise isf (inverse survival function) via the symmetry form, not `ppf(1 - q)`.
/// See [`isf_value`] for why, and for the endpoint and out-of-range contract.
#[polars_expr(output_type=Float64)]
fn normal_isf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, isf_value)
}

value_keyed_scalar_plugins! {
    struct NormalParamsKwargs { mu: f64, sigma: f64 }

    build = |kw| build_dist(kw.mu, kw.sigma)?;

    methods {
        fn normal_pdf_scalar => pdf_value;
        fn normal_ln_pdf_scalar => ln_pdf_value;
        fn normal_cdf_scalar => cdf_value;
        fn normal_sf_scalar => sf_value;
        fn normal_ln_cdf_scalar => ln_cdf_value;
        fn normal_ln_sf_scalar => ln_sf_value;
        fn normal_ppf_scalar => ppf_value;
        fn normal_isf_scalar => isf_value;
    }
}
