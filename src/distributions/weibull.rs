//! `(X / scale)^shape` of a `Weibull(shape, scale)` variate is `Exponential(1)`. The cumulative
//! forms and the inverses are `exponential.rs`'s read at that power; the densities carry the
//! Jacobian `shape (x / scale)^(shape - 1) / scale` as well.

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distr::Distribution as RandDistribution;
use statrs::distribution::Weibull;
use statrs::function::gamma::{gamma, ln_gamma};

use crate::distributions::{
    coerce_f64, expm1, exponential, on_unit_interval, param_keyed, scale_exp, validated_pair,
    value_keyed_derived_ternary, value_keyed_scalar, ParamDomain, Sides,
};
use crate::rng::{
    sample_by_index, sample_per_row_ternary, samples_by_index, samples_f64_output,
    samples_per_row_ternary, SampleKwargs, SampleScalarKwargs, SamplesKwargs, SamplesScalarKwargs,
};

const SHAPE: ParamDomain = ParamDomain::positive("shape");
const SCALE: ParamDomain = ParamDomain::positive("scale");

fn check_params(shape: &Float64Chunked, scale: &Float64Chunked) -> PolarsResult<()> {
    SHAPE.check_column(shape)?;
    SCALE.check_column(scale)
}

fn build_dist(shape: f64, scale: f64) -> PolarsResult<Weibull> {
    Weibull::new(shape, scale).map_err(|e| polars_err!(ComputeError: "{e}"))
}

/// Constant parameters, deserialised once per call. Every `_scalar` twin checks them once here.
#[derive(serde::Deserialize)]
struct WeibullParams {
    shape: f64,
    scale: f64,
}

impl WeibullParams {
    fn check(&self) -> PolarsResult<()> {
        SHAPE.check(self.shape)?;
        SCALE.check(self.scale)
    }

    fn build(&self) -> PolarsResult<Weibull> {
        self.check()?;
        build_dist(self.shape, self.scale)
    }

    fn value_keyed<Branches>(
        &self,
        value: &Series,
        derive: impl Fn(f64, f64) -> Branches,
        select: impl Fn(&Branches, f64) -> Option<f64>,
    ) -> PolarsResult<Series> {
        self.check()?;
        value_keyed_scalar(value, &derive(self.shape, self.scale), select)
    }
}

/// `ln(x / scale)` for `x >= 0`. Within a factor of two of `scale` it is `ln_1p` of the excess
/// `(x - scale) / scale`, which Sterbenz makes exact: the literal `(x / scale).ln()` rounds the
/// ratio first, and `shape` magnifies that rounding into `shape * 1.1e-16` relative on the power.
/// Further out the ratio's own log is exact enough; where the ratio leaves the normal range (a tiny
/// `x` under a large `scale`, or `x = 0`) the difference of logs takes over.
fn log_ratio(scale: f64, x: f64) -> f64 {
    let ratio = x / scale;
    if (0.5..=2.0).contains(&ratio) {
        ((x - scale) / scale).ln_1p()
    } else if ratio.is_normal() {
        ratio.ln()
    } else {
        x.ln() - scale.ln()
    }
}

/// The exponential's closed form read at the power `(x / scale)^shape`, as `exp(shape ln(x / scale))`.
fn at_power(
    shape: f64,
    scale: f64,
    exponential_form: Sides<impl Fn(f64) -> f64>,
) -> Sides<impl Fn(f64) -> f64> {
    exponential_form.read_at(0.0, move |x| (shape * log_ratio(scale, x)).exp())
}

/// The density at `x = 0`, where `(shape - 1) ln(x / scale)` is `0 * -inf`: `1 / scale` at
/// `shape = 1`, infinite below it, `0` above it.
fn pdf_at_origin(shape: f64, scale: f64) -> f64 {
    if shape == 1.0 {
        1.0 / scale
    } else if shape < 1.0 {
        f64::INFINITY
    } else {
        0.0
    }
}

fn ln_pdf_at_origin(shape: f64, scale: f64) -> f64 {
    if shape == 1.0 {
        -scale.ln()
    } else {
        pdf_at_origin(shape, scale).ln()
    }
}

/// `(shape / scale) (x / scale)^(shape - 1) exp(-t)` on the support, `t` the power, `0` below, as
/// `(shape * half / scale) * half` with `half = exp(((shape - 1) ln(x / scale) - t) / 2)`: the
/// power's log folded into the exponent keeps a subnormal `t` from costing digits, the half
/// exponent stays normal down to `exp(-1420)`, and `shape / scale` alone overflows where the
/// density is ordinary. An infinite power is the one place the density is `0` before the formula.
fn derive_pdf(shape: f64, scale: f64) -> Sides<impl Fn(f64) -> f64> {
    Sides {
        floor: 0.0,
        below_support: 0.0,
        on_support: move |x: f64| {
            if x == 0.0 {
                return pdf_at_origin(shape, scale);
            }
            let ln_ratio = log_ratio(scale, x);
            let t = (shape * ln_ratio).exp();
            if t.is_infinite() {
                return 0.0;
            }
            let half = (((shape - 1.0) * ln_ratio - t) / 2.0).exp();
            (shape * half / scale) * half
        },
    }
}

/// `ln(shape) - ln(scale) + (shape - 1) ln(x / scale) - t` on the support, `-inf` below. The two
/// logs are kept apart: `ln(shape / scale)` over- and underflows where neither does.
fn derive_ln_pdf(shape: f64, scale: f64) -> Sides<impl Fn(f64) -> f64> {
    Sides {
        floor: 0.0,
        below_support: f64::NEG_INFINITY,
        on_support: {
            let ln_shape_over_scale = shape.ln() - scale.ln();
            move |x: f64| {
                if x == 0.0 {
                    return ln_pdf_at_origin(shape, scale);
                }
                let ln_ratio = log_ratio(scale, x);
                let t = (shape * ln_ratio).exp();
                if t.is_infinite() {
                    return f64::NEG_INFINITY;
                }
                ln_shape_over_scale + (shape - 1.0) * ln_ratio - t
            }
        },
    }
}

fn derive_cdf(shape: f64, scale: f64) -> Sides<impl Fn(f64) -> f64> {
    at_power(shape, scale, exponential::derive_cdf(1.0))
}

/// Below this `ln(t)`, `t < 2^-53` and `ln(1 - exp(-t)) = ln(t) - t / 2 + O(t^2)` rounds to `ln(t)`
/// itself, so the power need not be formed: it underflows long before `ln(t)` runs out.
const LN_CDF_IS_LN_POWER_BELOW: f64 = -36.7368005696771;

/// The exponential's `ln_cdf` at the power; deep in the left tail, the log of the power itself.
fn derive_ln_cdf(shape: f64, scale: f64) -> Sides<impl Fn(f64) -> f64> {
    let exponential = exponential::derive_ln_cdf(1.0);
    Sides {
        floor: 0.0,
        below_support: f64::NEG_INFINITY,
        on_support: move |x: f64| {
            let ln_t = shape * log_ratio(scale, x);
            if ln_t < LN_CDF_IS_LN_POWER_BELOW {
                ln_t
            } else {
                (exponential.on_support)(ln_t.exp())
            }
        },
    }
}

fn derive_sf(shape: f64, scale: f64) -> Sides<impl Fn(f64) -> f64> {
    at_power(shape, scale, exponential::derive_sf(1.0))
}

fn derive_ln_sf(shape: f64, scale: f64) -> Sides<impl Fn(f64) -> f64> {
    at_power(shape, scale, exponential::derive_ln_sf(1.0))
}

/// `scale (-ln(1 - q))^(1 / shape)` as `scale exp(ln(t) / shape)`, `t = -ln_1p(-q)` the unit
/// exponential quantile: the root is taken on the log scale so a tiny `scale` beside a large
/// exponent survives, and `1 - q` is never formed.
fn derive_ppf(shape: f64, scale: f64) -> impl Fn(f64) -> f64 {
    let exponential = exponential::derive_ppf(1.0);
    move |quantile: f64| scale_exp(scale, exponential(quantile).ln() / shape)
}

/// `scale (-ln q)^(1 / shape)`, solved on `q` itself.
fn derive_isf(shape: f64, scale: f64) -> impl Fn(f64) -> f64 {
    let exponential = exponential::derive_isf(1.0);
    move |quantile: f64| scale_exp(scale, exponential(quantile).ln() / shape)
}

#[polars_expr(output_type=Float64)]
fn weibull_pdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, derive_pdf, Sides::at)
}

#[polars_expr(output_type=Float64)]
fn weibull_ln_pdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, derive_ln_pdf, Sides::at)
}

#[polars_expr(output_type=Float64)]
fn weibull_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, derive_cdf, Sides::at)
}

#[polars_expr(output_type=Float64)]
fn weibull_ln_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, derive_ln_cdf, Sides::at)
}

#[polars_expr(output_type=Float64)]
fn weibull_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, derive_sf, Sides::at)
}

#[polars_expr(output_type=Float64)]
fn weibull_ln_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, derive_ln_sf, Sides::at)
}

#[polars_expr(output_type=Float64)]
fn weibull_ppf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, derive_ppf, on_unit_interval)
}

#[polars_expr(output_type=Float64)]
fn weibull_isf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, derive_isf, on_unit_interval)
}

#[polars_expr(output_type=Float64)]
fn weibull_pdf_scalar(inputs: &[Series], kwargs: WeibullParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_pdf, Sides::at)
}

#[polars_expr(output_type=Float64)]
fn weibull_ln_pdf_scalar(inputs: &[Series], kwargs: WeibullParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_ln_pdf, Sides::at)
}

#[polars_expr(output_type=Float64)]
fn weibull_cdf_scalar(inputs: &[Series], kwargs: WeibullParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_cdf, Sides::at)
}

#[polars_expr(output_type=Float64)]
fn weibull_ln_cdf_scalar(inputs: &[Series], kwargs: WeibullParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_ln_cdf, Sides::at)
}

#[polars_expr(output_type=Float64)]
fn weibull_sf_scalar(inputs: &[Series], kwargs: WeibullParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_sf, Sides::at)
}

#[polars_expr(output_type=Float64)]
fn weibull_ln_sf_scalar(inputs: &[Series], kwargs: WeibullParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_ln_sf, Sides::at)
}

#[polars_expr(output_type=Float64)]
fn weibull_ppf_scalar(inputs: &[Series], kwargs: WeibullParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_ppf, on_unit_interval)
}

#[polars_expr(output_type=Float64)]
fn weibull_isf_scalar(inputs: &[Series], kwargs: WeibullParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_isf, on_unit_interval)
}

/// The validated `scale`, which the Python moments gate on.
#[polars_expr(output_type=Float64)]
fn weibull_scale(inputs: &[Series]) -> PolarsResult<Series> {
    validated_pair(inputs, coerce_f64, check_params)
}

/// The parameter-keyed moments, one driver.
fn moment(inputs: &[Series], body: impl Fn(f64, f64) -> f64) -> PolarsResult<Series> {
    param_keyed(
        inputs,
        coerce_f64,
        coerce_f64,
        check_params,
        |shape, scale| Ok(body(shape, scale)),
    )
}

/// Below this `a = 1 / shape` the two log-gammas of [`second_moment_terms`] cancel to fewer digits
/// than the series keeps: at the crossover the difference has lost two of statrs' sixteen and the
/// series truncation sits at `1e-17` relative.
const LN_GAMMA_RATIO_SERIES_BELOW: f64 = 0.125;

/// `(-1)^n zeta(n) (2^n - 2) / n` for `n = 2..=27`: the Taylor coefficients at `a = 0` of
/// `ln Gamma(1 + 2a) - 2 ln Gamma(1 + a)`, whose linear terms cancel exactly.
const LN_GAMMA_RATIO_SERIES: [f64; 26] = [
    1.6449340668482264,
    -2.4041138063191885,
    3.7881313179889835,
    -6.22156653086022,
    10.512544973839308,
    -18.150286992874612,
    31.87945605928473,
    -56.780475593477995,
    102.301645578063,
    -186.0919190803662,
    341.2506231957703,
    -630.0773094089744,
    1170.2145262106094,
    -2184.466816943389,
    4095.9375942242555,
    -7710.058882793788,
    14563.500037382837,
    -27594.0526552217,
    52428.75001498929,
    -99864.33334285778,
    190650.13636970092,
    -364722.0434821298,
    699050.6250024727,
    -1342177.240001583,
    2581110.11538563,
    -4971026.925926577,
];

/// `(ln Gamma(1 + 2a), 1 - Gamma(1 + a)^2 / Gamma(1 + 2a))` for `a = 1 / shape`: the log of the
/// second raw moment over `scale^2`, and the variance's fraction of it. The fraction is
/// `1 - exp(-D)` with `D = ln Gamma(1 + 2a) - 2 ln Gamma(1 + a)`, which as a difference cancels to
/// `pi^2 a^2 / 6` from two terms of order `a`, so below [`LN_GAMMA_RATIO_SERIES_BELOW`] `D` is the
/// series instead. `1 - exp(-D)` goes through [`expm1`] where `D` is small and directly above `1`,
/// where the `sinh` identity would overflow past `D ~ 1420`.
fn second_moment_terms(a: f64) -> (f64, f64) {
    let ln_second_moment = ln_gamma(1.0 + 2.0 * a);
    let ln_gamma_ratio = if a < LN_GAMMA_RATIO_SERIES_BELOW {
        let tail = LN_GAMMA_RATIO_SERIES
            .iter()
            .rev()
            .fold(0.0, |acc, coefficient| acc * a + coefficient);
        a * a * tail
    } else {
        ln_second_moment - 2.0 * ln_gamma(1.0 + a)
    };
    let variance_fraction = if ln_gamma_ratio < 1.0 {
        -expm1(-ln_gamma_ratio)
    } else {
        1.0 - (-ln_gamma_ratio).exp()
    };
    (ln_second_moment, variance_fraction)
}

/// `scale Gamma(1 + 1 / shape)`.
#[polars_expr(output_type=Float64)]
fn weibull_mean(inputs: &[Series]) -> PolarsResult<Series> {
    moment(inputs, |shape, scale| scale * gamma(1.0 + 1.0 / shape))
}

/// `scale^2 (Gamma(1 + 2a) - Gamma(1 + a)^2)` as `scale^2 Gamma(1 + 2a)` times the variance
/// fraction of [`second_moment_terms`]: the literal difference cancels at large `shape`, where the
/// answer is `pi^2 scale^2 / (6 shape^2)`.
#[polars_expr(output_type=Float64)]
fn weibull_variance(inputs: &[Series]) -> PolarsResult<Series> {
    moment(inputs, |shape, scale| {
        let (ln_second_moment, variance_fraction) = second_moment_terms(1.0 / shape);
        scale * scale * ln_second_moment.exp() * variance_fraction
    })
}

/// `scale sqrt(Gamma(1 + 2a) - Gamma(1 + a)^2)` as `scale exp(ln Gamma(1 + 2a) / 2)` times the root
/// of the variance fraction, so it is finite wherever the standard deviation is rather than
/// wherever its square is.
#[polars_expr(output_type=Float64)]
fn weibull_std(inputs: &[Series]) -> PolarsResult<Series> {
    moment(inputs, |shape, scale| {
        let (ln_second_moment, variance_fraction) = second_moment_terms(1.0 / shape);
        scale * (ln_second_moment / 2.0).exp() * variance_fraction.sqrt()
    })
}

#[inline]
fn draw(dist: &Weibull, rng: &mut impl rand::Rng) -> f64 {
    RandDistribution::sample(dist, rng)
}

#[polars_expr(output_type=Float64)]
fn weibull_sample(inputs: &[Series], kwargs: SampleKwargs) -> PolarsResult<Series> {
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
fn weibull_sample_scalar(
    inputs: &[Series],
    kwargs: SampleScalarKwargs<WeibullParams>,
) -> PolarsResult<Series> {
    let dist = kwargs.params.build()?;
    sample_by_index(&inputs[0], kwargs.seed, |rng| draw(&dist, rng))
}

#[polars_expr(output_type_func_with_kwargs=samples_f64_output)]
fn weibull_samples_scalar(
    inputs: &[Series],
    kwargs: SamplesScalarKwargs<WeibullParams>,
) -> PolarsResult<Series> {
    let dist = kwargs.params.build()?;
    samples_by_index(&inputs[0], kwargs.seed, kwargs.size, |rng| draw(&dist, rng))
}

#[polars_expr(output_type_func_with_kwargs=samples_f64_output)]
fn weibull_samples(inputs: &[Series], kwargs: SamplesKwargs) -> PolarsResult<Series> {
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
