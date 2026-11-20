//! `ChiSquared(dof)` is `Gamma(dof / 2, 1 / 2)` and runs `gamma.rs`'s bodies at that shape and rate, so every
//! value-keyed method and `entropy` agree with it to the last bit.

use polars::prelude::arity::unary_elementwise;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distr::Distribution as RandDistribution;
use statrs::distribution::ChiSquared;

use crate::distributions::gamma::{
    self, cdf_value, isf_value, ln_cdf_value, ln_pdf_value, ln_sf_value, pdf_value, ppf_value,
    sf_value, ScaledGamma,
};
use crate::distributions::{
    coerce_f64, validated_param, value_keyed_derived_binary, value_keyed_scalar, ParamDomain,
};
use crate::rng::{
    sample_by_index, sample_per_row_binary, samples_by_index, samples_f64_output,
    samples_per_row_binary, SampleKwargs, SampleScalarKwargs, SamplesKwargs, SamplesScalarKwargs,
};

const DOF: ParamDomain = ParamDomain::positive("dof");

const GAMMA_RATE: f64 = 0.5;

fn as_gamma(dof: f64) -> ScaledGamma {
    ScaledGamma::new(dof / 2.0, GAMMA_RATE)
}

fn build_dist(dof: f64) -> PolarsResult<ChiSquared> {
    ChiSquared::new(dof).map_err(|e| polars_err!(ComputeError: "{e}"))
}

/// Constant parameter, deserialised once per call; every `_scalar` twin checks it once here.
#[derive(serde::Deserialize)]
struct ChiSquaredParams {
    dof: f64,
}

impl ChiSquaredParams {
    fn build(&self) -> PolarsResult<ChiSquared> {
        DOF.check(self.dof)?;
        build_dist(self.dof)
    }

    fn value_keyed(
        &self,
        value: &Series,
        select: impl Fn(&ScaledGamma, f64) -> Option<f64>,
    ) -> PolarsResult<Series> {
        DOF.check(self.dof)?;
        value_keyed_scalar(value, &as_gamma(self.dof), select)
    }
}

#[polars_expr(output_type=Float64)]
fn chisquared_dof(inputs: &[Series]) -> PolarsResult<Series> {
    validated_param(inputs, &DOF)
}

#[polars_expr(output_type=Float64)]
fn chisquared_pdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_binary(inputs, &DOF, as_gamma, pdf_value)
}

#[polars_expr(output_type=Float64)]
fn chisquared_ln_pdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_binary(inputs, &DOF, as_gamma, ln_pdf_value)
}

#[polars_expr(output_type=Float64)]
fn chisquared_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_binary(inputs, &DOF, as_gamma, cdf_value)
}

#[polars_expr(output_type=Float64)]
fn chisquared_ln_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_binary(inputs, &DOF, as_gamma, ln_cdf_value)
}

#[polars_expr(output_type=Float64)]
fn chisquared_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_binary(inputs, &DOF, as_gamma, sf_value)
}

#[polars_expr(output_type=Float64)]
fn chisquared_ln_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_binary(inputs, &DOF, as_gamma, ln_sf_value)
}

#[polars_expr(output_type=Float64)]
fn chisquared_ppf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_binary(inputs, &DOF, as_gamma, ppf_value)
}

#[polars_expr(output_type=Float64)]
fn chisquared_isf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_binary(inputs, &DOF, as_gamma, isf_value)
}

#[polars_expr(output_type=Float64)]
fn chisquared_pdf_scalar(inputs: &[Series], kwargs: ChiSquaredParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], pdf_value)
}

#[polars_expr(output_type=Float64)]
fn chisquared_ln_pdf_scalar(inputs: &[Series], kwargs: ChiSquaredParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], ln_pdf_value)
}

#[polars_expr(output_type=Float64)]
fn chisquared_cdf_scalar(inputs: &[Series], kwargs: ChiSquaredParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], cdf_value)
}

#[polars_expr(output_type=Float64)]
fn chisquared_ln_cdf_scalar(inputs: &[Series], kwargs: ChiSquaredParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], ln_cdf_value)
}

#[polars_expr(output_type=Float64)]
fn chisquared_sf_scalar(inputs: &[Series], kwargs: ChiSquaredParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], sf_value)
}

#[polars_expr(output_type=Float64)]
fn chisquared_ln_sf_scalar(inputs: &[Series], kwargs: ChiSquaredParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], ln_sf_value)
}

#[polars_expr(output_type=Float64)]
fn chisquared_ppf_scalar(inputs: &[Series], kwargs: ChiSquaredParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], ppf_value)
}

#[polars_expr(output_type=Float64)]
fn chisquared_isf_scalar(inputs: &[Series], kwargs: ChiSquaredParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], isf_value)
}

/// Differential entropy (nats): digamma has no Polars expression to move to.
#[polars_expr(output_type=Float64)]
fn chisquared_entropy(inputs: &[Series]) -> PolarsResult<Series> {
    let dof = coerce_f64(&inputs[0])?;
    DOF.check_column(&dof)?;
    let out: Float64Chunked = unary_elementwise(&dof, |row| {
        row.map(|dof| gamma::entropy(dof / 2.0, GAMMA_RATE))
    });
    Ok(out.into_series())
}

/// statrs draws through its `Gamma(dof / 2, 1 / 2)`, so a seed gives `Gamma`'s draws.
#[inline]
fn draw(dist: &ChiSquared, rng: &mut impl rand::Rng) -> f64 {
    RandDistribution::sample(dist, rng)
}

#[polars_expr(output_type=Float64)]
fn chisquared_sample(inputs: &[Series], kwargs: SampleKwargs) -> PolarsResult<Series> {
    sample_per_row_binary(inputs, kwargs, &DOF, build_dist, draw)
}

#[polars_expr(output_type=Float64)]
fn chisquared_sample_scalar(
    inputs: &[Series],
    kwargs: SampleScalarKwargs<ChiSquaredParams>,
) -> PolarsResult<Series> {
    let dist = kwargs.params.build()?;
    sample_by_index(&inputs[0], kwargs.seed, |rng| draw(&dist, rng))
}

#[polars_expr(output_type_func_with_kwargs=samples_f64_output)]
fn chisquared_samples_scalar(
    inputs: &[Series],
    kwargs: SamplesScalarKwargs<ChiSquaredParams>,
) -> PolarsResult<Series> {
    let dist = kwargs.params.build()?;
    samples_by_index(&inputs[0], kwargs.seed, kwargs.size, |rng| draw(&dist, rng))
}

#[polars_expr(output_type_func_with_kwargs=samples_f64_output)]
fn chisquared_samples(inputs: &[Series], kwargs: SamplesKwargs) -> PolarsResult<Series> {
    samples_per_row_binary(inputs, kwargs, &DOF, build_dist, draw)
}
