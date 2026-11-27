use std::f64::consts::LN_2;
use std::hint::select_unpredictable;

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distr::Distribution as RandDistribution;
use statrs::distribution::Laplace;

use crate::distributions::{
    coerce_f64, on_the_line, on_unit_interval, validated_pair, value_keyed_derived_ternary,
    value_keyed_scalar, ParamDomain,
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

fn build_dist(loc: f64, scale: f64) -> PolarsResult<Laplace> {
    Laplace::new(loc, scale).map_err(|e| polars_err!(ComputeError: "{e}"))
}

/// Constant parameters, deserialised once per call. Every `_scalar` twin checks them once here.
#[derive(serde::Deserialize)]
struct LaplaceParams {
    loc: f64,
    scale: f64,
}

impl LaplaceParams {
    fn check(&self) -> PolarsResult<()> {
        LOC.check(self.loc)?;
        SCALE.check(self.scale)
    }

    fn build(&self) -> PolarsResult<Laplace> {
        self.check()?;
        build_dist(self.loc, self.scale)
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

/// `(value - loc) / scale`, or `value / scale - loc / scale` where the distance overflows (`value` and `loc` of
/// opposite signs near `f64::MAX`). An infinite `value` keeps the direct form: the split one reads `inf - inf` once
/// `loc / scale` overflows.
#[inline]
fn standardise(loc: f64, scale: f64, value: f64) -> f64 {
    let distance = value - loc;
    if distance.is_finite() || value.is_infinite() {
        distance / scale
    } else {
        value / scale - loc / scale
    }
}

/// `loc + scale z`, the inverse of [`standardise`], split the same way where `scale z` overflows; an infinite `z`
/// keeps the direct form.
#[inline]
fn unstandardise(loc: f64, scale: f64, z: f64) -> f64 {
    let offset = scale * z;
    if offset.is_finite() || z.is_infinite() {
        loc + offset
    } else {
        scale * (loc / scale + z)
    }
}

/// `ln(2 scale)` in one rounding where `ln 2 + ln(scale)` cancels near `scale = 1/2`; the sum only where `2 scale`
/// overflows.
fn ln_two_scale(scale: f64) -> f64 {
    let two_scale = 2.0 * scale;
    if two_scale.is_finite() {
        two_scale.ln()
    } else {
        LN_2 + scale.ln()
    }
}

/// `-|z| - ln(2 scale)`, finite past the underflow of `exp(-|z|)`.
fn derive_ln_pdf(loc: f64, scale: f64) -> impl Fn(f64) -> f64 {
    let ln_two_scale = ln_two_scale(scale);
    move |value: f64| -(standardise(loc, scale, value).abs() + ln_two_scale)
}

/// `exp(-|z|) / (2 scale)` while the mass is a normal number and `2 scale` finite, which keeps an ulp or two;
/// `exp(-(|z| + ln(2 scale)))` past either, where exponentiating the rounded sum costs up to a few hundred ulp.
fn derive_pdf(loc: f64, scale: f64) -> impl Fn(f64) -> f64 {
    let two_scale = 2.0 * scale;
    move |value: f64| {
        let distance = standardise(loc, scale, value).abs();
        let mass = (-distance).exp();
        if mass >= f64::MIN_POSITIVE && two_scale.is_finite() {
            mass / two_scale
        } else {
            (-(distance + ln_two_scale(scale))).exp()
        }
    }
}

/// The tail mass `exp(-|z|) / 2` below the median and its complement above, so the small side is never a
/// difference.
#[inline]
fn standard_cdf(z: f64) -> f64 {
    let tail = (-z.abs()).exp() / 2.0;
    if z <= 0.0 {
        tail
    } else {
        1.0 - tail
    }
}

/// `z - ln 2` below the median, which no underflow reaches, and `ln_1p(-exp(-z) / 2)` above, where `ln(cdf)` would
/// round the answer's own magnitude away.
#[inline]
fn standard_ln_cdf(z: f64) -> f64 {
    if z <= 0.0 {
        z - LN_2
    } else {
        (-(-z).exp() / 2.0).ln_1p()
    }
}

fn derive_cdf(loc: f64, scale: f64) -> impl Fn(f64) -> f64 {
    move |value: f64| standard_cdf(standardise(loc, scale, value))
}

fn derive_sf(loc: f64, scale: f64) -> impl Fn(f64) -> f64 {
    move |value: f64| standard_cdf(-standardise(loc, scale, value))
}

fn derive_ln_cdf(loc: f64, scale: f64) -> impl Fn(f64) -> f64 {
    move |value: f64| standard_ln_cdf(standardise(loc, scale, value))
}

fn derive_ln_sf(loc: f64, scale: f64) -> impl Fn(f64) -> f64 {
    move |value: f64| standard_ln_cdf(-standardise(loc, scale, value))
}

/// `ln(2 q)` up to the median and `-ln(2 (1 - q))` above, where `1 - q` is exact (Sterbenz). `ln(0) = -inf` gives
/// the closed endpoints, `-0.0` included. Selects rather than branches: on an unsorted quantile column a branch on the
/// median mispredicts half the time.
fn standard_quantile(q: f64) -> f64 {
    let upper = q > 0.5;
    let ln_tail = (2.0 * select_unpredictable(upper, 1.0 - q, q)).ln();
    select_unpredictable(upper, -ln_tail, ln_tail)
}

fn derive_ppf(loc: f64, scale: f64) -> impl Fn(f64) -> f64 {
    move |q: f64| unstandardise(loc, scale, standard_quantile(q))
}

/// `isf(q) = 2 loc - ppf(q)`, solved on `q` itself rather than on `1 - q`.
fn derive_isf(loc: f64, scale: f64) -> impl Fn(f64) -> f64 {
    move |q: f64| unstandardise(loc, scale, -standard_quantile(q))
}

#[polars_expr(output_type=Float64)]
fn laplace_pdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, derive_pdf, on_the_line)
}

#[polars_expr(output_type=Float64)]
fn laplace_ln_pdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, derive_ln_pdf, on_the_line)
}

#[polars_expr(output_type=Float64)]
fn laplace_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, derive_cdf, on_the_line)
}

#[polars_expr(output_type=Float64)]
fn laplace_ln_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, derive_ln_cdf, on_the_line)
}

#[polars_expr(output_type=Float64)]
fn laplace_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, derive_sf, on_the_line)
}

#[polars_expr(output_type=Float64)]
fn laplace_ln_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, derive_ln_sf, on_the_line)
}

#[polars_expr(output_type=Float64)]
fn laplace_ppf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, derive_ppf, on_unit_interval)
}

#[polars_expr(output_type=Float64)]
fn laplace_isf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, derive_isf, on_unit_interval)
}

#[polars_expr(output_type=Float64)]
fn laplace_pdf_scalar(inputs: &[Series], kwargs: LaplaceParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_pdf, on_the_line)
}

#[polars_expr(output_type=Float64)]
fn laplace_ln_pdf_scalar(inputs: &[Series], kwargs: LaplaceParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_ln_pdf, on_the_line)
}

#[polars_expr(output_type=Float64)]
fn laplace_cdf_scalar(inputs: &[Series], kwargs: LaplaceParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_cdf, on_the_line)
}

#[polars_expr(output_type=Float64)]
fn laplace_ln_cdf_scalar(inputs: &[Series], kwargs: LaplaceParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_ln_cdf, on_the_line)
}

#[polars_expr(output_type=Float64)]
fn laplace_sf_scalar(inputs: &[Series], kwargs: LaplaceParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_sf, on_the_line)
}

#[polars_expr(output_type=Float64)]
fn laplace_ln_sf_scalar(inputs: &[Series], kwargs: LaplaceParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_ln_sf, on_the_line)
}

#[polars_expr(output_type=Float64)]
fn laplace_ppf_scalar(inputs: &[Series], kwargs: LaplaceParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_ppf, on_unit_interval)
}

#[polars_expr(output_type=Float64)]
fn laplace_isf_scalar(inputs: &[Series], kwargs: LaplaceParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_isf, on_unit_interval)
}

/// The validated `scale`, which the Python moments gate on.
#[polars_expr(output_type=Float64)]
fn laplace_scale(inputs: &[Series]) -> PolarsResult<Series> {
    validated_pair(inputs, coerce_f64, check_params)
}

/// statrs' inverse transform on one uniform in `[-0.5, 0.5)` on a `2^-52` grid. Its `-0.5` endpoint, probability
/// `2^-52`, draws `-inf`; every other draw lies within `51 ln 2 ~ 35.4` scales of `loc`.
#[inline]
fn draw(dist: &Laplace, rng: &mut impl rand::Rng) -> f64 {
    RandDistribution::sample(dist, rng)
}

#[polars_expr(output_type=Float64)]
fn laplace_sample(inputs: &[Series], kwargs: SampleKwargs) -> PolarsResult<Series> {
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
fn laplace_sample_scalar(
    inputs: &[Series],
    kwargs: SampleScalarKwargs<LaplaceParams>,
) -> PolarsResult<Series> {
    let dist = kwargs.params.build()?;
    sample_by_index(&inputs[0], kwargs.seed, |rng| draw(&dist, rng))
}

#[polars_expr(output_type_func_with_kwargs=samples_f64_output)]
fn laplace_samples_scalar(
    inputs: &[Series],
    kwargs: SamplesScalarKwargs<LaplaceParams>,
) -> PolarsResult<Series> {
    let dist = kwargs.params.build()?;
    samples_by_index(&inputs[0], kwargs.seed, kwargs.size, |rng| draw(&dist, rng))
}

#[polars_expr(output_type_func_with_kwargs=samples_f64_output)]
fn laplace_samples(inputs: &[Series], kwargs: SamplesKwargs) -> PolarsResult<Series> {
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
