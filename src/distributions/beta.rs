use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distr::Distribution as RandDistribution;
use statrs::distribution::{Beta, Continuous, ContinuousCDF};
use statrs::statistics::Distribution as StatrsDistribution;

use crate::distributions::{
    coerce_f64, param_keyed, validated_pair, value_keyed_scalar, value_keyed_ternary, ParamDomain,
};
use crate::rng::{
    sample_by_index, sample_per_row_ternary, samples_by_index, samples_f64_output,
    samples_per_row_ternary, SampleKwargs, SampleScalarKwargs, SamplesKwargs, SamplesScalarKwargs,
};

const A: ParamDomain = ParamDomain::positive("a");
const B: ParamDomain = ParamDomain::positive("b");

fn check_params(a: &Float64Chunked, b: &Float64Chunked) -> PolarsResult<()> {
    A.check_column(a)?;
    B.check_column(b)
}

fn build_dist(a: f64, b: f64) -> PolarsResult<Beta> {
    Beta::new(a, b).map_err(|e| polars_err!(ComputeError: "{e}"))
}

/// Constant parameters, deserialised once per call. Every `_scalar` twin builds through
/// [`Self::build`], so it cannot rebuild per row.
#[derive(serde::Deserialize)]
struct BetaParams {
    a: f64,
    b: f64,
}

impl BetaParams {
    fn build(&self) -> PolarsResult<Beta> {
        A.check(self.a)?;
        B.check(self.b)?;
        build_dist(self.a, self.b)
    }

    fn value_keyed<Body>(&self, value: &Series, body: Body) -> PolarsResult<Series>
    where
        Body: Fn(&Beta, f64) -> Option<f64>,
    {
        value_keyed_scalar(value, &self.build()?, body)
    }
}

fn value_keyed<Body>(inputs: &[Series], body: Body) -> PolarsResult<Series>
where
    Body: Fn(&Beta, f64) -> Option<f64>,
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
fn beta_params(inputs: &[Series]) -> PolarsResult<Series> {
    validated_pair(inputs, coerce_f64, check_params)
}

#[inline]
fn draw(dist: &Beta, rng: &mut impl rand::Rng) -> f64 {
    RandDistribution::sample(dist, rng)
}

#[polars_expr(output_type=Float64)]
fn beta_sample(inputs: &[Series], kwargs: SampleKwargs) -> PolarsResult<Series> {
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
fn beta_sample_scalar(
    inputs: &[Series],
    kwargs: SampleScalarKwargs<BetaParams>,
) -> PolarsResult<Series> {
    let dist = kwargs.params.build()?;
    sample_by_index(&inputs[0], kwargs.seed, |rng| draw(&dist, rng))
}

#[polars_expr(output_type_func_with_kwargs=samples_f64_output)]
fn beta_samples_scalar(
    inputs: &[Series],
    kwargs: SamplesScalarKwargs<BetaParams>,
) -> PolarsResult<Series> {
    let dist = kwargs.params.build()?;
    samples_by_index(&inputs[0], kwargs.seed, kwargs.size, |rng| draw(&dist, rng))
}

#[polars_expr(output_type_func_with_kwargs=samples_f64_output)]
fn beta_samples(inputs: &[Series], kwargs: SamplesKwargs) -> PolarsResult<Series> {
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

fn pdf_value(dist: &Beta, v: f64) -> Option<f64> {
    Some(dist.pdf(v))
}

fn ln_pdf_value(dist: &Beta, v: f64) -> Option<f64> {
    Some(dist.ln_pdf(v))
}

// The regularized incomplete beta behind `cdf` / `sf` panics on a `NaN` point; the drivers'
// short-circuit is what keeps it out.

fn cdf_value(dist: &Beta, v: f64) -> Option<f64> {
    Some(dist.cdf(v))
}

fn sf_value(dist: &Beta, v: f64) -> Option<f64> {
    Some(dist.sf(v))
}

/// Null outside `[0, 1]`; `statrs` panics there. The endpoints map to the support bounds.
fn ppf_value(dist: &Beta, q: f64) -> Option<f64> {
    if !(0.0..=1.0).contains(&q) {
        None
    } else {
        Some(dist.inverse_cdf(q))
    }
}

fn isf_value(dist: &Beta, q: f64) -> Option<f64> {
    ppf_value(dist, 1.0 - q)
}

#[polars_expr(output_type=Float64)]
fn beta_pdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, pdf_value)
}

#[polars_expr(output_type=Float64)]
fn beta_ln_pdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, ln_pdf_value)
}

#[polars_expr(output_type=Float64)]
fn beta_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, cdf_value)
}

#[polars_expr(output_type=Float64)]
fn beta_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, sf_value)
}

#[polars_expr(output_type=Float64)]
fn beta_ppf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, ppf_value)
}

#[polars_expr(output_type=Float64)]
fn beta_isf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, isf_value)
}

#[polars_expr(output_type=Float64)]
fn beta_pdf_scalar(inputs: &[Series], kwargs: BetaParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], pdf_value)
}

#[polars_expr(output_type=Float64)]
fn beta_ln_pdf_scalar(inputs: &[Series], kwargs: BetaParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], ln_pdf_value)
}

#[polars_expr(output_type=Float64)]
fn beta_cdf_scalar(inputs: &[Series], kwargs: BetaParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], cdf_value)
}

#[polars_expr(output_type=Float64)]
fn beta_sf_scalar(inputs: &[Series], kwargs: BetaParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], sf_value)
}

#[polars_expr(output_type=Float64)]
fn beta_ppf_scalar(inputs: &[Series], kwargs: BetaParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], ppf_value)
}

#[polars_expr(output_type=Float64)]
fn beta_isf_scalar(inputs: &[Series], kwargs: BetaParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], isf_value)
}

/// Differential entropy (nats), `ln B(a, b) - (a - 1) psi(a) - (b - 1) psi(b) + (a + b - 2) psi(a + b)`:
/// log-Beta and digamma have no Polars expression to move to.
#[polars_expr(output_type=Float64)]
fn beta_entropy(inputs: &[Series]) -> PolarsResult<Series> {
    param_keyed(inputs, coerce_f64, coerce_f64, check_params, |a, b| {
        Ok(build_dist(a, b)?
            .entropy()
            .expect("Beta::entropy is Some for every valid (a, b)"))
    })
}
