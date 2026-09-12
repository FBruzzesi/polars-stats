use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distr::Distribution as RandDistribution;
use statrs::distribution::Exp;

use crate::distributions::{
    expm1, on_unit_interval, validated_param, value_keyed_derived_binary, value_keyed_scalar,
    ParamDomain, Sides,
};
use crate::rng::{
    sample_by_index, sample_per_row_binary, samples_by_index, samples_f64_output,
    samples_per_row_binary, SampleKwargs, SampleScalarKwargs, SamplesKwargs, SamplesScalarKwargs,
};

const RATE: ParamDomain = ParamDomain::positive("rate");

fn build_dist(rate: f64) -> PolarsResult<Exp> {
    Exp::new(rate).map_err(|e| polars_err!(ComputeError: "{e}"))
}

/// Constant parameter, deserialised once per call; every `_scalar` twin checks it once here.
#[derive(serde::Deserialize)]
struct ExponentialParams {
    rate: f64,
}

impl ExponentialParams {
    fn build(&self) -> PolarsResult<Exp> {
        RATE.check(self.rate)?;
        build_dist(self.rate)
    }

    fn value_keyed<Branches>(
        &self,
        value: &Series,
        derive: impl Fn(f64) -> Branches,
        select: impl Fn(&Branches, f64) -> Option<f64>,
    ) -> PolarsResult<Series> {
        RATE.check(self.rate)?;
        value_keyed_scalar(value, &derive(self.rate), select)
    }
}

#[polars_expr(output_type=Float64)]
fn exponential_rate(inputs: &[Series]) -> PolarsResult<Series> {
    validated_param(inputs, &RATE)
}

/// Crossover of [`derive_cdf`] in units of `t = rate * x`: above it the plain `1 - exp(-t)` is
/// already exact, and the [`expm1`] identity below it would overflow past `t ~ 1420`. The two
/// branches agree to `1e-16` here.
const CDF_SINH_MAX: f64 = 1.0;

/// `rate * exp(-rate * x)` on `x >= 0`, `0` below, as `(rate * exp(-t / 2)) * exp(-t / 2)`: written
/// literally, `exp(-t)` rounds into the subnormal range before the scale is applied and the final
/// multiply magnifies what was thrown away.
fn derive_pdf(rate: f64) -> Sides<impl Fn(f64) -> f64, 0> {
    Sides {
        below_support: 0.0,
        on_support: move |x: f64| {
            let half_exp = (-rate * x / 2.0).exp();
            (rate * half_exp) * half_exp
        },
    }
}

/// `ln(rate) - rate * x` on `x >= 0`, `-inf` below; `ln(rate)` is hoisted out of the row loop.
fn derive_ln_pdf(rate: f64) -> Sides<impl Fn(f64) -> f64, 0> {
    Sides {
        below_support: f64::NEG_INFINITY,
        on_support: {
            let ln_rate = rate.ln();
            move |x: f64| ln_rate - rate * x
        },
    }
}

/// `1 - exp(-rate * x)` on `x >= 0`, `0` below. `1 - exp(-t)` cancels to `0` below `t ~ 1.1e-16`,
/// so the small side reads `-expm1(-t)`.
fn derive_cdf(rate: f64) -> Sides<impl Fn(f64) -> f64, 0> {
    Sides {
        below_support: 0.0,
        on_support: move |x: f64| {
            let t = rate * x;
            if t < CDF_SINH_MAX {
                -expm1(-t)
            } else {
                1.0 - (-t).exp()
            }
        },
    }
}

/// `ln(cdf)` where `t < 1`, `ln_1p(-sf)` above; `-inf` below the support. As `cdf -> 1` its log
/// rounds to a tiny inaccurate value, so that side goes through `ln_1p` of the small `sf`; the
/// predicate says where `ln(cdf)` is well conditioned and coincides with [`CDF_SINH_MAX`], which is
/// what lets the left arm inline [`derive_cdf`]'s `expm1` branch.
fn derive_ln_cdf(rate: f64) -> Sides<impl Fn(f64) -> f64, 0> {
    Sides {
        below_support: f64::NEG_INFINITY,
        on_support: move |x: f64| {
            let t = rate * x;
            if t < 1.0 {
                (-expm1(-t)).ln()
            } else {
                (-(-t).exp()).ln_1p()
            }
        },
    }
}

/// `exp(-rate * x)` on `x >= 0`, `1` below; never `1 - cdf`, which quantises the upper tail to the
/// `1.1e-16` spacing of `1.0`.
fn derive_sf(rate: f64) -> Sides<impl Fn(f64) -> f64, 0> {
    Sides {
        below_support: 1.0,
        on_support: move |x: f64| (-rate * x).exp(),
    }
}

fn derive_ln_sf(rate: f64) -> Sides<impl Fn(f64) -> f64, 0> {
    Sides {
        below_support: 0.0,
        on_support: move |x: f64| -rate * x,
    }
}

/// `-ln_1p(-q) / rate`; `ln(1 - q)` rounds `1 - q` to exactly `1` below `q ~ 1.1e-16`. The rate is
/// divided by, never reciprocated: `x * (1 / rate)` rounds twice, and at a subnormal rate the
/// reciprocal reaches `inf` where the division stays finite.
fn derive_ppf(rate: f64) -> impl Fn(f64) -> f64 {
    move |quantile: f64| (-((-quantile).ln_1p())) / rate
}

/// `-ln(q) / rate`, solved on `q` itself rather than as `ppf(1 - q)`.
fn derive_isf(rate: f64) -> impl Fn(f64) -> f64 {
    move |quantile: f64| (-quantile.ln()) / rate
}

#[polars_expr(output_type=Float64)]
fn exponential_pdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_binary(inputs, &RATE, derive_pdf, Sides::at)
}

#[polars_expr(output_type=Float64)]
fn exponential_ln_pdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_binary(inputs, &RATE, derive_ln_pdf, Sides::at)
}

#[polars_expr(output_type=Float64)]
fn exponential_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_binary(inputs, &RATE, derive_cdf, Sides::at)
}

#[polars_expr(output_type=Float64)]
fn exponential_ln_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_binary(inputs, &RATE, derive_ln_cdf, Sides::at)
}

#[polars_expr(output_type=Float64)]
fn exponential_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_binary(inputs, &RATE, derive_sf, Sides::at)
}

#[polars_expr(output_type=Float64)]
fn exponential_ln_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_binary(inputs, &RATE, derive_ln_sf, Sides::at)
}

#[polars_expr(output_type=Float64)]
fn exponential_ppf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_binary(inputs, &RATE, derive_ppf, on_unit_interval)
}

#[polars_expr(output_type=Float64)]
fn exponential_isf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_binary(inputs, &RATE, derive_isf, on_unit_interval)
}

#[polars_expr(output_type=Float64)]
fn exponential_pdf_scalar(inputs: &[Series], kwargs: ExponentialParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_pdf, Sides::at)
}

#[polars_expr(output_type=Float64)]
fn exponential_ln_pdf_scalar(inputs: &[Series], kwargs: ExponentialParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_ln_pdf, Sides::at)
}

#[polars_expr(output_type=Float64)]
fn exponential_cdf_scalar(inputs: &[Series], kwargs: ExponentialParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_cdf, Sides::at)
}

#[polars_expr(output_type=Float64)]
fn exponential_ln_cdf_scalar(inputs: &[Series], kwargs: ExponentialParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_ln_cdf, Sides::at)
}

#[polars_expr(output_type=Float64)]
fn exponential_sf_scalar(inputs: &[Series], kwargs: ExponentialParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_sf, Sides::at)
}

#[polars_expr(output_type=Float64)]
fn exponential_ln_sf_scalar(inputs: &[Series], kwargs: ExponentialParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_ln_sf, Sides::at)
}

#[polars_expr(output_type=Float64)]
fn exponential_ppf_scalar(inputs: &[Series], kwargs: ExponentialParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_ppf, on_unit_interval)
}

#[polars_expr(output_type=Float64)]
fn exponential_isf_scalar(inputs: &[Series], kwargs: ExponentialParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_isf, on_unit_interval)
}

#[inline]
fn draw(dist: &Exp, rng: &mut impl rand::Rng) -> f64 {
    RandDistribution::sample(dist, rng)
}

#[polars_expr(output_type=Float64)]
fn exponential_sample(inputs: &[Series], kwargs: SampleKwargs) -> PolarsResult<Series> {
    sample_per_row_binary(inputs, kwargs, &RATE, build_dist, draw)
}

#[polars_expr(output_type=Float64)]
fn exponential_sample_scalar(
    inputs: &[Series],
    kwargs: SampleScalarKwargs<ExponentialParams>,
) -> PolarsResult<Series> {
    let dist = kwargs.params.build()?;
    sample_by_index(&inputs[0], kwargs.seed, |rng| draw(&dist, rng))
}

#[polars_expr(output_type_func_with_kwargs=samples_f64_output)]
fn exponential_samples_scalar(
    inputs: &[Series],
    kwargs: SamplesScalarKwargs<ExponentialParams>,
) -> PolarsResult<Series> {
    let dist = kwargs.params.build()?;
    samples_by_index(&inputs[0], kwargs.seed, kwargs.size, |rng| draw(&dist, rng))
}

#[polars_expr(output_type_func_with_kwargs=samples_f64_output)]
fn exponential_samples(inputs: &[Series], kwargs: SamplesKwargs) -> PolarsResult<Series> {
    samples_per_row_binary(inputs, kwargs, &RATE, build_dist, draw)
}
