use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distr::Distribution as RandDistribution;
use statrs::distribution::{Continuous, ContinuousCDF, LogNormal, Normal};

use crate::distributions::{
    coerce_f64, normal, validated_pair, value_keyed_scalar, value_keyed_ternary, ParamDomain,
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

fn build_dist(mu: f64, sigma: f64) -> PolarsResult<LogNormal> {
    LogNormal::new(mu, sigma).map_err(|e| polars_err!(ComputeError: "{e}"))
}

/// Constant parameters, deserialised once per call. Every `_scalar` twin builds through
/// [`Self::build`], so it cannot rebuild per row.
#[derive(serde::Deserialize)]
struct LogNormalParams {
    mu: f64,
    sigma: f64,
}

impl LogNormalParams {
    fn build(&self) -> PolarsResult<LogNormal> {
        MU.check(self.mu)?;
        SIGMA.check(self.sigma)?;
        build_dist(self.mu, self.sigma)
    }

    fn value_keyed<Body>(&self, value: &Series, body: Body) -> PolarsResult<Series>
    where
        Body: Fn(&LogNormal, f64) -> Option<f64>,
    {
        value_keyed_scalar(value, &self.build()?, body)
    }
}

fn value_keyed<Body>(inputs: &[Series], body: Body) -> PolarsResult<Series>
where
    Body: Fn(&LogNormal, f64) -> Option<f64>,
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

/// The `Normal(mu, sigma)` under a built `LogNormal`; infallible because the parameters passed
/// [`build_dist`].
fn underlying_normal(dist: &LogNormal) -> Normal {
    Normal::new(dist.location(), dist.scale())
        .expect("Normal::new accepts every (location, scale) a built LogNormal carries")
}

#[polars_expr(output_type=Float64)]
fn lognormal_sigma(inputs: &[Series]) -> PolarsResult<Series> {
    validated_pair(inputs, coerce_f64, check_params)
}

#[inline]
fn draw(dist: &LogNormal, rng: &mut impl rand::Rng) -> f64 {
    RandDistribution::sample(dist, rng)
}

#[polars_expr(output_type=Float64)]
fn lognormal_sample(inputs: &[Series], kwargs: SampleKwargs) -> PolarsResult<Series> {
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
fn lognormal_sample_scalar(
    inputs: &[Series],
    kwargs: SampleScalarKwargs<LogNormalParams>,
) -> PolarsResult<Series> {
    let dist = kwargs.params.build()?;
    sample_by_index(&inputs[0], kwargs.seed, |rng| draw(&dist, rng))
}

#[polars_expr(output_type_func_with_kwargs=samples_f64_output)]
fn lognormal_samples_scalar(
    inputs: &[Series],
    kwargs: SamplesScalarKwargs<LogNormalParams>,
) -> PolarsResult<Series> {
    let dist = kwargs.params.build()?;
    samples_by_index(&inputs[0], kwargs.seed, kwargs.size, |rng| draw(&dist, rng))
}

#[polars_expr(output_type_func_with_kwargs=samples_f64_output)]
fn lognormal_samples(inputs: &[Series], kwargs: SamplesKwargs) -> PolarsResult<Series> {
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

/// The normal's stable log-cdf at `ln(v)`; `-inf` off the support (`v <= 0`).
fn ln_cdf_value(dist: &LogNormal, v: f64) -> Option<f64> {
    if v <= 0.0 {
        Some(f64::NEG_INFINITY)
    } else {
        normal::ln_cdf_value(&underlying_normal(dist), v.ln())
    }
}

/// The normal's stable log-sf at `ln(v)`; `0` off the support (`v <= 0`).
fn ln_sf_value(dist: &LogNormal, v: f64) -> Option<f64> {
    if v <= 0.0 {
        Some(0.0)
    } else {
        normal::ln_sf_value(&underlying_normal(dist), v.ln())
    }
}

/// Null outside `[0, 1]`; `statrs` panics there. The endpoints map to `0` and `+inf`.
fn ppf_value(dist: &LogNormal, q: f64) -> Option<f64> {
    if !(0.0..=1.0).contains(&q) {
        None
    } else {
        Some(dist.inverse_cdf(q))
    }
}

/// The normal's [`normal::isf_value`] exponentiated, so the tail is solved on `q` itself rather than
/// through `ppf(1 - q)`. `exp` turns the normal's absolute error at the quantile into a relative one
/// here, so a large `sigma` amplifies it.
fn isf_value(dist: &LogNormal, q: f64) -> Option<f64> {
    normal::isf_value(&underlying_normal(dist), q).map(f64::exp)
}

#[polars_expr(output_type=Float64)]
fn lognormal_pdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, pdf_value)
}

#[polars_expr(output_type=Float64)]
fn lognormal_ln_pdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, ln_pdf_value)
}

#[polars_expr(output_type=Float64)]
fn lognormal_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, cdf_value)
}

#[polars_expr(output_type=Float64)]
fn lognormal_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, sf_value)
}

#[polars_expr(output_type=Float64)]
fn lognormal_ln_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, ln_cdf_value)
}

#[polars_expr(output_type=Float64)]
fn lognormal_ln_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, ln_sf_value)
}

#[polars_expr(output_type=Float64)]
fn lognormal_ppf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, ppf_value)
}

#[polars_expr(output_type=Float64)]
fn lognormal_isf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, isf_value)
}

#[polars_expr(output_type=Float64)]
fn lognormal_pdf_scalar(inputs: &[Series], kwargs: LogNormalParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], pdf_value)
}

#[polars_expr(output_type=Float64)]
fn lognormal_ln_pdf_scalar(inputs: &[Series], kwargs: LogNormalParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], ln_pdf_value)
}

#[polars_expr(output_type=Float64)]
fn lognormal_cdf_scalar(inputs: &[Series], kwargs: LogNormalParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], cdf_value)
}

#[polars_expr(output_type=Float64)]
fn lognormal_sf_scalar(inputs: &[Series], kwargs: LogNormalParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], sf_value)
}

#[polars_expr(output_type=Float64)]
fn lognormal_ln_cdf_scalar(inputs: &[Series], kwargs: LogNormalParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], ln_cdf_value)
}

#[polars_expr(output_type=Float64)]
fn lognormal_ln_sf_scalar(inputs: &[Series], kwargs: LogNormalParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], ln_sf_value)
}

#[polars_expr(output_type=Float64)]
fn lognormal_ppf_scalar(inputs: &[Series], kwargs: LogNormalParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], ppf_value)
}

#[polars_expr(output_type=Float64)]
fn lognormal_isf_scalar(inputs: &[Series], kwargs: LogNormalParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], isf_value)
}
