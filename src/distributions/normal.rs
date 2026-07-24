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
/// `statrs::Normal::new` rejects a non-finite `mu`, a `NaN` `sigma`, or `sigma <= 0`.
/// We surface that as `InvalidOperation` so an invalid scale fails the whole evaluation,
/// rather than silently nulling the row.
/// Validation lives here so every method that builds a distribution reports an invalid scale identically.
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
    /// `inputs[0]` is `mu`, `inputs[1]` is `sigma`. Mirrors `uniform_range`: the closed-form
    /// moments (`mean`, `variance`, `median`, `entropy`) all derive from this single FFI
    /// round-trip, so they report an invalid parameterisation identically to the value-keyed
    /// methods that build the distribution directly. `null` in either input propagates; a `NaN`
    /// mu or a non-positive / `NaN` `sigma` raises `InvalidOperation` via [`build_dist`].
    fn normal_sigma;
    params = (mu: DataType::Float64 => f64, sigma: DataType::Float64 => f64);
    build = build_dist;
    returns = sigma;
    output_name = inputs[1];
}

value_keyed_per_row! {
    /// Apply a value-keyed function `f(dist, value)` element-wise over `(value, mu, sigma)`.
    ///
    /// `inputs[0]` is the evaluation point, `inputs[1]` is `mu`, `inputs[2]` is `sigma`. `null`
    /// in any input propagates to `null`; an invalid `sigma` raises via [`build_dist`]. `f`
    /// returns an `Option` so a method can null a row on its own terms (e.g. `ppf` outside
    /// `[0, 1]`). Shared by `pdf`, `ln_pdf`, `cdf`, `sf`, `ppf`.
    fn value_keyed(&Normal);
    params = (DataType::Float64 => f64, DataType::Float64 => f64);
    build = build_dist;
}

/// Element-wise Normal sampler.
///
/// `inputs[0]` carries `mu`, `inputs[1]` `sigma`, and `inputs[2]` a per-row index used to derive
/// a per-row sub-seed, so the function is genuinely element-wise: chunking and threading cannot
/// change the output. With `seed=None`, a fresh root seed is drawn once per call.
///
/// Per-row validation:
///   * `null` (in any input) propagates;
///   * a `NaN` `mu`, or a non-positive / `NaN` `sigma`, raises `InvalidOperation`.
///
/// Returns a `Float64` series.
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
    /// Static parameters for the constant-parameter sampler fast path: when both `mu` and
    /// `sigma` are Python scalars, they travel here as kwargs instead of as two full-length
    /// `pl.repeat` columns re-validated on every row.
    struct NormalScalarKwargs { mu: f64, sigma: f64 }

    /// Constant-parameter Normal sampler: [`normal_sample`] for the all-scalar case. The
    /// distribution is validated and built once, and only the per-row index travels as an input;
    /// seeding and the draw are unchanged, so output matches `normal_sample` for the same
    /// `(seed, index, mu, sigma)`.
    fn normal_sample_scalar(output_type = Float64, physical = Float64Type);

    samples = normal_samples_scalar as NormalSamplesScalarKwargs -> samples_f64_output;

    build = |kw| build_dist(kw.mu, kw.sigma)?;
    draw = |dist, rng| RandDistribution::sample(&dist, rng);
}

/// Element-wise multi-draw Normal sampler: `size` draws per row in one call.
///
/// The column-parameter counterpart of [`normal_samples_scalar`], replacing `samples`' former
/// construction of `k` [`normal_sample`] calls glued by `concat_arr`: the distribution is built
/// once per row instead of once per draw. Row `i`'s draws come from the one stream seeded
/// `(seed, i)`, so output is bit-identical to the scalar path for the same parameters (the
/// seeding is positional, parameters never enter it, so equal-parameter rows still draw
/// independently). Null/error contract follows [`normal_sample`] per row; a null row yields a
/// null array element. Returns `Array(Float64, size)`.
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

/// Point past which `erfc(t)` is close enough to underflowing (`erfc ~ 1e-274` at `t = 25`,
/// underflow near `t = 26.6`) that we switch `ln_erfc` to the asymptotic series.
const LN_ERFC_ASYMPTOTIC_MIN: f64 = 25.0;

/// Natural log of the complementary error function, stable in the right tail.
///
/// `erfc(t).ln()` returns `-inf` once `erfc(t)` rounds to `0` (`t` above ~26.6), the regime `log_cdf`
/// / `log_sf` exist to serve. Past [`LN_ERFC_ASYMPTOTIC_MIN`] we switch to the asymptotic expansion
/// `erfc(t) = exp(-t^2) / (t * sqrt(pi)) * (1 - 1/(2t^2) + 3/(4t^4) - 15/(8t^6) + ...)`, whose log is a
/// large finite negative number; below it, statrs `erfc` holds full relative precision so the direct
/// form is exact. The two branches agree to ~1e-13 at the crossover, so the join is seamless.
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

value_keyed_scalar_plugins! {
    /// Static parameters for the constant-parameter value-keyed fast paths (`<method>_scalar`).
    ///
    /// Like [`NormalScalarKwargs`] minus the sampler `seed`: when both parameters are Python
    /// scalars, the Python layer routes them here as kwargs instead of expanding each into a
    /// full-length column re-validated on every row. The distribution is validated and built once;
    /// only the evaluation-point column travels as an input.
    struct NormalParamsKwargs { mu: f64, sigma: f64 }

    build = |kw| build_dist(kw.mu, kw.sigma)?;

    methods {
        /// Constant-parameter pdf; same body as [`normal_pdf`] via [`pdf_value`], dist built once.
        fn normal_pdf_scalar => pdf_value;

        /// Constant-parameter log-pdf; same body as [`normal_ln_pdf`] via [`ln_pdf_value`], dist built once.
        fn normal_ln_pdf_scalar => ln_pdf_value;

        /// Constant-parameter cdf; same body as [`normal_cdf`] via [`cdf_value`], dist built once.
        fn normal_cdf_scalar => cdf_value;

        /// Constant-parameter sf; same body as [`normal_sf`] via [`sf_value`], dist built once.
        fn normal_sf_scalar => sf_value;

        /// Constant-parameter log-cdf; same body as [`normal_ln_cdf`] via [`ln_cdf_value`], dist built once.
        fn normal_ln_cdf_scalar => ln_cdf_value;

        /// Constant-parameter log-sf; same body as [`normal_ln_sf`] via [`ln_sf_value`], dist built once.
        fn normal_ln_sf_scalar => ln_sf_value;

        /// Constant-parameter ppf; same body as [`normal_ppf`] via [`ppf_value`], dist built once.
        fn normal_ppf_scalar => ppf_value;
    }
}
