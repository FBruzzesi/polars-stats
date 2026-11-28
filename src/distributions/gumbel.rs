use std::f64::consts::LN_2;

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distr::{Distribution as RandDistribution, Open01};

use crate::distributions::{
    coerce_f64, expm1, ln_abs_expm1, on_the_line, on_unit_interval, standardise, unstandardise,
    validated_pair, value_keyed_derived_ternary, value_keyed_scalar, ParamDomain,
};
use crate::rng::{
    sample_by_index, sample_per_row_ternary, samples_by_index, samples_f64_output,
    samples_per_row_ternary, SampleKwargs, SampleScalarKwargs, SamplesKwargs, SamplesScalarKwargs,
};

const LOC: ParamDomain = ParamDomain::finite("loc");
const SCALE: ParamDomain = ParamDomain::positive("scale");

fn check_params(loc: &Float64Chunked, scale: &Float64Chunked) -> PolarsResult<()> {
    LOC.check_column(loc)?;
    SCALE.check_column(scale)
}

/// Constant parameters, deserialised once per call, and the sampler's per-row state. Every `_scalar` twin checks
/// them once here.
#[derive(serde::Deserialize)]
struct GumbelParams {
    loc: f64,
    scale: f64,
}

impl GumbelParams {
    fn check(&self) -> PolarsResult<()> {
        LOC.check(self.loc)?;
        SCALE.check(self.scale)
    }

    fn value_keyed<Branches>(
        &self,
        value: &Series,
        derive: impl Fn(f64, f64) -> Branches,
        select: impl Fn(&Branches, f64) -> Option<f64>,
    ) -> PolarsResult<Series> {
        self.check()?;
        value_keyed_scalar(value, &derive(self.loc, self.scale), select)
    }
}

/// Unchecked: the sampler drivers check the parameter columns before building any row.
fn build_params(loc: f64, scale: f64) -> PolarsResult<GumbelParams> {
    Ok(GumbelParams { loc, scale })
}

/// The standardised point `z` and `w = exp(-z)`, the `Exponential(1)` variate with `z = -ln w`.
#[inline]
fn z_and_w(loc: f64, scale: f64, value: f64) -> (f64, f64) {
    let z = standardise(loc, scale, value);
    (z, (-z).exp())
}

/// `-(z + w) - ln scale`, or `-inf` once `w` overflows: the answer is then below `-f64::MAX`, and at `z = -inf` the
/// sum would read `inf - inf`.
#[inline]
fn ln_density(z: f64, w: f64, ln_scale: f64) -> f64 {
    if w.is_finite() {
        -(z + w) - ln_scale
    } else {
        f64::NEG_INFINITY
    }
}

fn derive_ln_pdf(loc: f64, scale: f64) -> impl Fn(f64) -> f64 {
    let ln_scale = scale.ln();
    move |value: f64| {
        let (z, w) = z_and_w(loc, scale, value);
        ln_density(z, w, ln_scale)
    }
}

/// `w exp(-w) / scale` while that mass is a normal number; past it, the exponentiated [`ln_density`], whose
/// rounding costs up to a few hundred ulp. On the left tail both forms also inherit about `w / 2` ulp from rounding
/// `w` itself.
fn derive_pdf(loc: f64, scale: f64) -> impl Fn(f64) -> f64 {
    let ln_scale = scale.ln();
    move |value: f64| {
        let (z, w) = z_and_w(loc, scale, value);
        let mass = w * (-w).exp();
        if mass >= f64::MIN_POSITIVE {
            mass / scale
        } else {
            ln_density(z, w, ln_scale).exp()
        }
    }
}

fn derive_cdf(loc: f64, scale: f64) -> impl Fn(f64) -> f64 {
    move |value: f64| {
        let (_, w) = z_and_w(loc, scale, value);
        (-w).exp()
    }
}

fn derive_ln_cdf(loc: f64, scale: f64) -> impl Fn(f64) -> f64 {
    move |value: f64| {
        let (_, w) = z_and_w(loc, scale, value);
        -w
    }
}

/// `-expm1(-w)` below `w = ln 2`, where `1 - exp(-w)` cancels; the difference above, where nothing cancels and
/// [`expm1`]'s `sinh(w / 2)` would overflow past `w ~ 1420`.
fn derive_sf(loc: f64, scale: f64) -> impl Fn(f64) -> f64 {
    move |value: f64| {
        let (_, w) = z_and_w(loc, scale, value);
        if w < LN_2 {
            -expm1(-w)
        } else {
            1.0 - (-w).exp()
        }
    }
}

/// `ln(1 - exp(-w))`. Below `w = 2^-52` it rounds to `ln w = -z`, which stays finite past the underflow of `w` at
/// `z ~ 745`; up to `ln 2` it is [`ln_abs_expm1`]; above, `ln_1p(-exp(-w))`, with nothing to cancel.
fn derive_ln_sf(loc: f64, scale: f64) -> impl Fn(f64) -> f64 {
    move |value: f64| {
        let (z, w) = z_and_w(loc, scale, value);
        if w < f64::EPSILON {
            -z
        } else if w < LN_2 {
            ln_abs_expm1(-w)
        } else {
            (-(-w).exp()).ln_1p()
        }
    }
}

/// `z = -ln w` at `w = -ln q`. `ln(-0.0)` is `-inf`, so `q = 1` and `q = -0.0` land on the support ends.
fn derive_ppf(loc: f64, scale: f64) -> impl Fn(f64) -> f64 {
    move |q: f64| {
        let w = -q.ln();
        unstandardise(loc, scale, -w.ln())
    }
}

/// `z = -ln w` at `w = -ln(1 - q)`, read through `ln_1p` so a small `q` keeps its digits rather than being
/// quantised by `1 - q`.
fn derive_isf(loc: f64, scale: f64) -> impl Fn(f64) -> f64 {
    move |q: f64| {
        let w = -(-q).ln_1p();
        unstandardise(loc, scale, -w.ln())
    }
}

#[polars_expr(output_type=Float64)]
fn gumbel_pdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, derive_pdf, on_the_line)
}

#[polars_expr(output_type=Float64)]
fn gumbel_ln_pdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, derive_ln_pdf, on_the_line)
}

#[polars_expr(output_type=Float64)]
fn gumbel_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, derive_cdf, on_the_line)
}

#[polars_expr(output_type=Float64)]
fn gumbel_ln_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, derive_ln_cdf, on_the_line)
}

#[polars_expr(output_type=Float64)]
fn gumbel_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, derive_sf, on_the_line)
}

#[polars_expr(output_type=Float64)]
fn gumbel_ln_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, derive_ln_sf, on_the_line)
}

#[polars_expr(output_type=Float64)]
fn gumbel_ppf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, derive_ppf, on_unit_interval)
}

#[polars_expr(output_type=Float64)]
fn gumbel_isf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, derive_isf, on_unit_interval)
}

#[polars_expr(output_type=Float64)]
fn gumbel_pdf_scalar(inputs: &[Series], kwargs: GumbelParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_pdf, on_the_line)
}

#[polars_expr(output_type=Float64)]
fn gumbel_ln_pdf_scalar(inputs: &[Series], kwargs: GumbelParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_ln_pdf, on_the_line)
}

#[polars_expr(output_type=Float64)]
fn gumbel_cdf_scalar(inputs: &[Series], kwargs: GumbelParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_cdf, on_the_line)
}

#[polars_expr(output_type=Float64)]
fn gumbel_ln_cdf_scalar(inputs: &[Series], kwargs: GumbelParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_ln_cdf, on_the_line)
}

#[polars_expr(output_type=Float64)]
fn gumbel_sf_scalar(inputs: &[Series], kwargs: GumbelParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_sf, on_the_line)
}

#[polars_expr(output_type=Float64)]
fn gumbel_ln_sf_scalar(inputs: &[Series], kwargs: GumbelParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_ln_sf, on_the_line)
}

#[polars_expr(output_type=Float64)]
fn gumbel_ppf_scalar(inputs: &[Series], kwargs: GumbelParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_ppf, on_unit_interval)
}

#[polars_expr(output_type=Float64)]
fn gumbel_isf_scalar(inputs: &[Series], kwargs: GumbelParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_isf, on_unit_interval)
}

/// The validated `scale`, which the Python moments gate on.
#[polars_expr(output_type=Float64)]
fn gumbel_scale(inputs: &[Series]) -> PolarsResult<Series> {
    validated_pair(inputs, coerce_f64, check_params)
}

/// The inverse transform `z = -ln w` at `w = -ln u`, with `u` from [`Open01`]. Its grid `(k + 1/2) 2^-52` holds
/// neither `0` nor `1`, so every draw is finite and within `[-ln(53 ln 2), 53 ln 2] ~ [-3.6, 36.7]` scales of
/// `loc`. statrs' sampler takes `ln(-u)` and is `NaN` for every `u`.
#[inline]
fn draw(params: &GumbelParams, rng: &mut impl rand::Rng) -> f64 {
    let u: f64 = RandDistribution::sample(&Open01, rng);
    let w = -u.ln();
    unstandardise(params.loc, params.scale, -w.ln())
}

#[polars_expr(output_type=Float64)]
fn gumbel_sample(inputs: &[Series], kwargs: SampleKwargs) -> PolarsResult<Series> {
    sample_per_row_ternary(
        inputs,
        kwargs,
        coerce_f64,
        coerce_f64,
        check_params,
        build_params,
        draw,
    )
}

#[polars_expr(output_type=Float64)]
fn gumbel_sample_scalar(
    inputs: &[Series],
    kwargs: SampleScalarKwargs<GumbelParams>,
) -> PolarsResult<Series> {
    let params = kwargs.params;
    params.check()?;
    sample_by_index(&inputs[0], kwargs.seed, |rng| draw(&params, rng))
}

#[polars_expr(output_type_func_with_kwargs=samples_f64_output)]
fn gumbel_samples_scalar(
    inputs: &[Series],
    kwargs: SamplesScalarKwargs<GumbelParams>,
) -> PolarsResult<Series> {
    let params = kwargs.params;
    params.check()?;
    samples_by_index(&inputs[0], kwargs.seed, kwargs.size, |rng| {
        draw(&params, rng)
    })
}

#[polars_expr(output_type_func_with_kwargs=samples_f64_output)]
fn gumbel_samples(inputs: &[Series], kwargs: SamplesKwargs) -> PolarsResult<Series> {
    samples_per_row_ternary(
        inputs,
        kwargs,
        coerce_f64,
        coerce_f64,
        check_params,
        build_params,
        draw,
    )
}
