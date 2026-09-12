use std::f64::consts::{LN_2, SQRT_2};

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distr::Distribution as RandDistribution;
use statrs::distribution::{Continuous, ContinuousCDF, Normal};
use statrs::function::erf;
use statrs::statistics::Distribution as StatrsDistribution;

use crate::distributions::{
    coerce_f64, validated_pair, value_keyed_scalar, value_keyed_ternary, ParamDomain,
};
use crate::rng::{
    sample_by_index, sample_per_row_ternary, samples_by_index, samples_f64_output,
    samples_per_row_ternary, SampleKwargs, SampleScalarKwargs, SamplesKwargs, SamplesScalarKwargs,
};

const MU: ParamDomain = ParamDomain::finite("mu");
const SIGMA: ParamDomain = ParamDomain::positive("sigma");

fn check_params(mu: &Float64Chunked, sigma: &Float64Chunked) -> PolarsResult<()> {
    MU.check_column(mu)?;
    SIGMA.check_column(sigma)
}

fn build_dist(mu: f64, sigma: f64) -> PolarsResult<Normal> {
    Normal::new(mu, sigma).map_err(|e| polars_err!(ComputeError: "{e}"))
}

/// Constant parameters, deserialised once per call. Every `_scalar` twin builds through
/// [`Self::build`], so it cannot rebuild per row.
#[derive(serde::Deserialize)]
struct NormalParams {
    mu: f64,
    sigma: f64,
}

impl NormalParams {
    fn build(&self) -> PolarsResult<Normal> {
        MU.check(self.mu)?;
        SIGMA.check(self.sigma)?;
        build_dist(self.mu, self.sigma)
    }

    fn value_keyed<Body>(&self, value: &Series, body: Body) -> PolarsResult<Series>
    where
        Body: Fn(&Normal, f64) -> Option<f64>,
    {
        value_keyed_scalar(value, &self.build()?, body)
    }
}

fn value_keyed<Body>(inputs: &[Series], body: Body) -> PolarsResult<Series>
where
    Body: Fn(&Normal, f64) -> Option<f64>,
{
    value_keyed_ternary(
        inputs,
        coerce_f64,
        coerce_f64,
        check_params,
        build_dist,
        body,
    )
}

#[polars_expr(output_type=Float64)]
fn normal_sigma(inputs: &[Series]) -> PolarsResult<Series> {
    validated_pair(inputs, coerce_f64, check_params)
}

#[inline]
fn draw(dist: &Normal, rng: &mut impl rand::Rng) -> f64 {
    RandDistribution::sample(dist, rng)
}

#[polars_expr(output_type=Float64)]
fn normal_sample(inputs: &[Series], kwargs: SampleKwargs) -> PolarsResult<Series> {
    sample_per_row_ternary(
        inputs,
        kwargs,
        coerce_f64,
        coerce_f64,
        check_params,
        build_dist,
        draw,
    )
}

#[polars_expr(output_type=Float64)]
fn normal_sample_scalar(
    inputs: &[Series],
    kwargs: SampleScalarKwargs<NormalParams>,
) -> PolarsResult<Series> {
    let dist = kwargs.params.build()?;
    sample_by_index(&inputs[0], kwargs.seed, |rng| draw(&dist, rng))
}

#[polars_expr(output_type_func_with_kwargs=samples_f64_output)]
fn normal_samples_scalar(
    inputs: &[Series],
    kwargs: SamplesScalarKwargs<NormalParams>,
) -> PolarsResult<Series> {
    let dist = kwargs.params.build()?;
    samples_by_index(&inputs[0], kwargs.seed, kwargs.size, |rng| draw(&dist, rng))
}

#[polars_expr(output_type_func_with_kwargs=samples_f64_output)]
fn normal_samples(inputs: &[Series], kwargs: SamplesKwargs) -> PolarsResult<Series> {
    samples_per_row_ternary(
        inputs,
        kwargs,
        coerce_f64,
        coerce_f64,
        check_params,
        build_dist,
        draw,
    )
}

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

/// Past this `erfc(t)` is close to underflowing (`~1e-274` at `t = 25`, `0` near `t = 26.6`), so
/// [`ln_erfc`] switches to the asymptotic series. The two branches agree to `~1e-13` here.
const LN_ERFC_ASYMPTOTIC_MIN: f64 = 25.0;

/// `ln(erfc(t))`, finite in the right tail where `erfc(t).ln()` is `-inf`: the direct log below
/// [`LN_ERFC_ASYMPTOTIC_MIN`], the log of
/// `exp(-t^2) / (t sqrt(pi)) * (1 - 1/(2t^2) + 3/(4t^4) - 15/(8t^6) + ...)` above it.
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

/// `ln(0.5 * erfc(s))` with full relative precision on both sides: [`ln_erfc`] for `s >= 0`,
/// `ln_1p(-0.5 * erfc(-s))` for `s < 0`, where `erfc(s)` rounds to `2` and the direct log collapses
/// to `0` instead of the true tiny negative (`-7.6e-24` at 10 sigma).
fn ln_half_erfc(s: f64) -> f64 {
    if s >= 0.0 {
        -LN_2 + ln_erfc(s)
    } else {
        (-0.5 * erf::erfc(-s)).ln_1p()
    }
}

fn mu_sigma(dist: &Normal) -> (f64, f64) {
    let mu = dist.mean().expect("Normal always has a mean");
    let sigma = dist.std_dev().expect("Normal always has a std_dev");
    (mu, sigma)
}

/// `t = (x - mu) / (sigma sqrt 2)`, so that `cdf(x) = 0.5 erfc(-t)` and `sf(x) = 0.5 erfc(t)`.
fn erfc_arg(dist: &Normal, x: f64) -> f64 {
    let (mu, sigma) = mu_sigma(dist);
    (x - mu) / (sigma * SQRT_2)
}

pub(crate) fn ln_cdf_value(dist: &Normal, v: f64) -> Option<f64> {
    Some(ln_half_erfc(-erfc_arg(dist, v)))
}

pub(crate) fn ln_sf_value(dist: &Normal, v: f64) -> Option<f64> {
    Some(ln_half_erfc(erfc_arg(dist, v)))
}

/// Null outside `[0, 1]`; the closed endpoints map to the infinite tails.
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

/// `mu + sigma sqrt(2) erfc_inv(2q)`, solved on `q` itself. `ppf(1 - q)` would hand `erfc_inv`
/// the argument `2 - 2q`, which quantises the tail mass to `2.2e-16` before the inverse runs; the
/// symmetry `z_(1-q) = -z_q` puts the sign on the scale instead. Endpoints reverse `ppf`'s.
pub(crate) fn isf_value(dist: &Normal, q: f64) -> Option<f64> {
    if !(0.0..=1.0).contains(&q) {
        None
    } else if q == 0.0 {
        Some(f64::INFINITY)
    } else if q == 1.0 {
        Some(f64::NEG_INFINITY)
    } else {
        let (mu, sigma) = mu_sigma(dist);
        Some(mu + sigma * SQRT_2 * erf::erfc_inv(2.0 * q))
    }
}

#[polars_expr(output_type=Float64)]
fn normal_pdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, pdf_value)
}

#[polars_expr(output_type=Float64)]
fn normal_ln_pdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, ln_pdf_value)
}

#[polars_expr(output_type=Float64)]
fn normal_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, cdf_value)
}

#[polars_expr(output_type=Float64)]
fn normal_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, sf_value)
}

#[polars_expr(output_type=Float64)]
fn normal_ln_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, ln_cdf_value)
}

#[polars_expr(output_type=Float64)]
fn normal_ln_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, ln_sf_value)
}

#[polars_expr(output_type=Float64)]
fn normal_ppf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, ppf_value)
}

#[polars_expr(output_type=Float64)]
fn normal_isf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, isf_value)
}

#[polars_expr(output_type=Float64)]
fn normal_pdf_scalar(inputs: &[Series], kwargs: NormalParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], pdf_value)
}

#[polars_expr(output_type=Float64)]
fn normal_ln_pdf_scalar(inputs: &[Series], kwargs: NormalParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], ln_pdf_value)
}

#[polars_expr(output_type=Float64)]
fn normal_cdf_scalar(inputs: &[Series], kwargs: NormalParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], cdf_value)
}

#[polars_expr(output_type=Float64)]
fn normal_sf_scalar(inputs: &[Series], kwargs: NormalParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], sf_value)
}

#[polars_expr(output_type=Float64)]
fn normal_ln_cdf_scalar(inputs: &[Series], kwargs: NormalParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], ln_cdf_value)
}

#[polars_expr(output_type=Float64)]
fn normal_ln_sf_scalar(inputs: &[Series], kwargs: NormalParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], ln_sf_value)
}

#[polars_expr(output_type=Float64)]
fn normal_ppf_scalar(inputs: &[Series], kwargs: NormalParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], ppf_value)
}

#[polars_expr(output_type=Float64)]
fn normal_isf_scalar(inputs: &[Series], kwargs: NormalParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], isf_value)
}
