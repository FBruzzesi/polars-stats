//! `ln(X / scale)` of a `Pareto(scale, shape)` variate is `Exponential(shape)`. The cumulative forms
//! and the inverses are `exponential.rs`'s read at that log ratio; the densities carry the Jacobian
//! `1 / x` as well.

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distr::Distribution as RandDistribution;
use statrs::distribution::Pareto;

use crate::distributions::{
    coerce_f64, exponential, on_unit_interval, validated_pair, value_keyed_derived_ternary,
    value_keyed_scalar, ParamDomain, Sides,
};
use crate::rng::{
    sample_by_index, sample_per_row_ternary, samples_by_index, samples_f64_output,
    samples_per_row_ternary, SampleKwargs, SampleScalarKwargs, SamplesKwargs, SamplesScalarKwargs,
};

const SCALE: ParamDomain = ParamDomain::positive("scale");
const SHAPE: ParamDomain = ParamDomain::positive("shape");

fn check_params(scale: &Float64Chunked, shape: &Float64Chunked) -> PolarsResult<()> {
    SCALE.check_column(scale)?;
    SHAPE.check_column(shape)
}

fn build_dist(scale: f64, shape: f64) -> PolarsResult<Pareto> {
    Pareto::new(scale, shape).map_err(|e| polars_err!(ComputeError: "{e}"))
}

/// Constant parameters, deserialised once per call. Every `_scalar` twin checks them once here.
#[derive(serde::Deserialize)]
struct ParetoParams {
    scale: f64,
    shape: f64,
}

impl ParetoParams {
    fn check(&self) -> PolarsResult<()> {
        SCALE.check(self.scale)?;
        SHAPE.check(self.shape)
    }

    fn build(&self) -> PolarsResult<Pareto> {
        self.check()?;
        build_dist(self.scale, self.shape)
    }

    fn value_keyed<Branches>(
        &self,
        value: &Series,
        derive: impl Fn(f64, f64) -> Branches,
        select: impl Fn(&Branches, f64) -> Option<f64>,
    ) -> PolarsResult<Series> {
        self.check()?;
        value_keyed_scalar(value, &derive(self.scale, self.shape), select)
    }
}

/// `ln(x / scale)` for `x >= scale`, as `ln_1p` of the excess `(x - scale) / scale`, which Sterbenz
/// makes exact up to `x = 2 scale`: the literal `(x / scale).ln()` rounds the ratio first and is off
/// by `1.5e-2` relative at `x = scale (1 + 1e-14)`. Once the excess overflows, under a tiny `scale`
/// beside an ordinary `x`, the difference of logs takes over.
fn log_ratio(scale: f64, x: f64) -> f64 {
    let excess = (x - scale) / scale;
    if excess.is_finite() {
        excess.ln_1p()
    } else {
        x.ln() - scale.ln()
    }
}

/// The exponential's closed form read at [`log_ratio`], its below-support constant kept.
fn at_log_ratio(
    scale: f64,
    exponential_form: Sides<impl Fn(f64) -> f64>,
) -> Sides<impl Fn(f64) -> f64> {
    Sides {
        floor: scale,
        below_support: exponential_form.below_support,
        on_support: move |x: f64| (exponential_form.on_support)(log_ratio(scale, x)),
    }
}

/// `scale * exp(t)` as `(scale * exp(t / 2)) * exp(t / 2)`: `exp(t)` alone overflows past
/// `t ~ 709.8`, where the answer under a small `scale` is still ordinary; the half exponent holds to
/// `t ~ 1419`, past which `scale * exp(t)` has left `f64` for every normal `scale`.
fn scale_exp(scale: f64, t: f64) -> f64 {
    let half = (t / 2.0).exp();
    (scale * half) * half
}

/// `(shape / x) (scale / x)^shape` on the support, `0` below, as `(shape * half / x) * half` with
/// `half = exp(-shape ln(x / scale) / 2)`. The literal power forms `scale^shape` and `x^(shape + 1)`,
/// `0 / 0` at `scale = 1e-4, shape = 100`; a single `exp(-shape t)` rounds into the subnormals before
/// `shape / x` restores the magnitude. `half` is applied before the division because `shape / x`
/// alone overflows where the density is ordinary (`4.5e305` at `scale = 1e-300, shape = 1e10`), and
/// `shape * half` cannot, `half` never exceeding `1` on the support.
fn derive_pdf(scale: f64, shape: f64) -> Sides<impl Fn(f64) -> f64> {
    Sides {
        floor: scale,
        below_support: 0.0,
        on_support: move |x: f64| {
            let half = (-shape * log_ratio(scale, x) / 2.0).exp();
            (shape * half / x) * half
        },
    }
}

/// `ln(shape) - shape ln(x / scale) - ln(x)`: the exponential's `ln(rate) - rate t` less `ln(x)`.
/// Not `shape ln(scale) - (shape + 1) ln(x)`, whose two large terms cancel to an absolute error of
/// `shape * eps * ln(x)`: `1.4e-5` relative at `shape = 1e12`.
fn derive_ln_pdf(scale: f64, shape: f64) -> Sides<impl Fn(f64) -> f64> {
    let exponential = exponential::derive_ln_pdf(shape);
    Sides {
        floor: scale,
        below_support: f64::NEG_INFINITY,
        on_support: move |x: f64| (exponential.on_support)(log_ratio(scale, x)) - x.ln(),
    }
}

fn derive_cdf(scale: f64, shape: f64) -> Sides<impl Fn(f64) -> f64> {
    at_log_ratio(scale, exponential::derive_cdf(shape))
}

fn derive_ln_cdf(scale: f64, shape: f64) -> Sides<impl Fn(f64) -> f64> {
    at_log_ratio(scale, exponential::derive_ln_cdf(shape))
}

fn derive_sf(scale: f64, shape: f64) -> Sides<impl Fn(f64) -> f64> {
    at_log_ratio(scale, exponential::derive_sf(shape))
}

fn derive_ln_sf(scale: f64, shape: f64) -> Sides<impl Fn(f64) -> f64> {
    at_log_ratio(scale, exponential::derive_ln_sf(shape))
}

/// `scale (1 - q)^(-1 / shape)` as `scale exp(t)`, `t = -ln_1p(-q) / shape` the exponential
/// quantile: the power would round `1 - q` first, quantising a small `q` to `1.1e-16` absolute.
fn derive_ppf(scale: f64, shape: f64) -> impl Fn(f64) -> f64 {
    let exponential = exponential::derive_ppf(shape);
    move |quantile: f64| scale_exp(scale, exponential(quantile))
}

/// `scale q^(-1 / shape)` as `scale exp(-ln(q) / shape)`, solved on `q` itself.
fn derive_isf(scale: f64, shape: f64) -> impl Fn(f64) -> f64 {
    let exponential = exponential::derive_isf(shape);
    move |quantile: f64| scale_exp(scale, exponential(quantile))
}

#[polars_expr(output_type=Float64)]
fn pareto_pdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, derive_pdf, Sides::at)
}

#[polars_expr(output_type=Float64)]
fn pareto_ln_pdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, derive_ln_pdf, Sides::at)
}

#[polars_expr(output_type=Float64)]
fn pareto_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, derive_cdf, Sides::at)
}

#[polars_expr(output_type=Float64)]
fn pareto_ln_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, derive_ln_cdf, Sides::at)
}

#[polars_expr(output_type=Float64)]
fn pareto_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, derive_sf, Sides::at)
}

#[polars_expr(output_type=Float64)]
fn pareto_ln_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, derive_ln_sf, Sides::at)
}

#[polars_expr(output_type=Float64)]
fn pareto_ppf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, derive_ppf, on_unit_interval)
}

#[polars_expr(output_type=Float64)]
fn pareto_isf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, derive_isf, on_unit_interval)
}

#[polars_expr(output_type=Float64)]
fn pareto_pdf_scalar(inputs: &[Series], kwargs: ParetoParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_pdf, Sides::at)
}

#[polars_expr(output_type=Float64)]
fn pareto_ln_pdf_scalar(inputs: &[Series], kwargs: ParetoParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_ln_pdf, Sides::at)
}

#[polars_expr(output_type=Float64)]
fn pareto_cdf_scalar(inputs: &[Series], kwargs: ParetoParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_cdf, Sides::at)
}

#[polars_expr(output_type=Float64)]
fn pareto_ln_cdf_scalar(inputs: &[Series], kwargs: ParetoParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_ln_cdf, Sides::at)
}

#[polars_expr(output_type=Float64)]
fn pareto_sf_scalar(inputs: &[Series], kwargs: ParetoParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_sf, Sides::at)
}

#[polars_expr(output_type=Float64)]
fn pareto_ln_sf_scalar(inputs: &[Series], kwargs: ParetoParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_ln_sf, Sides::at)
}

#[polars_expr(output_type=Float64)]
fn pareto_ppf_scalar(inputs: &[Series], kwargs: ParetoParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_ppf, on_unit_interval)
}

#[polars_expr(output_type=Float64)]
fn pareto_isf_scalar(inputs: &[Series], kwargs: ParetoParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_isf, on_unit_interval)
}

/// The validated `shape`, which the Python moments gate on.
#[polars_expr(output_type=Float64)]
fn pareto_shape(inputs: &[Series]) -> PolarsResult<Series> {
    validated_pair(inputs, coerce_f64, check_params)
}

#[inline]
fn draw(dist: &Pareto, rng: &mut impl rand::Rng) -> f64 {
    RandDistribution::sample(dist, rng)
}

#[polars_expr(output_type=Float64)]
fn pareto_sample(inputs: &[Series], kwargs: SampleKwargs) -> PolarsResult<Series> {
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
fn pareto_sample_scalar(
    inputs: &[Series],
    kwargs: SampleScalarKwargs<ParetoParams>,
) -> PolarsResult<Series> {
    let dist = kwargs.params.build()?;
    sample_by_index(&inputs[0], kwargs.seed, |rng| draw(&dist, rng))
}

#[polars_expr(output_type_func_with_kwargs=samples_f64_output)]
fn pareto_samples_scalar(
    inputs: &[Series],
    kwargs: SamplesScalarKwargs<ParetoParams>,
) -> PolarsResult<Series> {
    let dist = kwargs.params.build()?;
    samples_by_index(&inputs[0], kwargs.seed, kwargs.size, |rng| draw(&dist, rng))
}

#[polars_expr(output_type_func_with_kwargs=samples_f64_output)]
fn pareto_samples(inputs: &[Series], kwargs: SamplesKwargs) -> PolarsResult<Series> {
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
