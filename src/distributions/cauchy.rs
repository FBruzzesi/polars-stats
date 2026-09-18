use std::f64::consts::PI;

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distr::Distribution as RandDistribution;
use statrs::consts::LN_PI;
use statrs::distribution::Cauchy;

use crate::distributions::{
    coerce_f64, on_unit_interval, validated_pair, value_keyed_derived_ternary, value_keyed_scalar,
    ParamDomain,
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

fn build_dist(loc: f64, scale: f64) -> PolarsResult<Cauchy> {
    Cauchy::new(loc, scale).map_err(|e| polars_err!(ComputeError: "{e}"))
}

/// Constant parameters, deserialised once per call. Every `_scalar` twin checks them once here.
#[derive(serde::Deserialize)]
struct CauchyParams {
    loc: f64,
    scale: f64,
}

impl CauchyParams {
    fn check(&self) -> PolarsResult<()> {
        LOC.check(self.loc)?;
        SCALE.check(self.scale)
    }

    fn build(&self) -> PolarsResult<Cauchy> {
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

/// The `select` of a closed form whose support is the whole line, so there is no branch to pick.
#[inline]
fn on_the_line<Arm: Fn(f64) -> f64>(arm: &Arm, value: f64) -> Option<f64> {
    Some(arm(value))
}

/// `1 / (pi scale (1 + z^2))`, `z = (value - loc) / scale`, spelled so that only a density outside
/// `f64` range is lost. Two intermediates leave the range before the answer does:
///
/// * past `|z| ~ 1.3e154` the square is `inf`, so that arm reads `scale / (pi d^2)` off the raw
///   distance. Dividing the peak by `z` twice would be `inf / inf` where a subnormal `scale` has
///   overflowed the peak and the standardised point too, and that `NaN` is the sentinel reserved
///   for a `NaN` evaluation point;
/// * below `scale ~ 1.77e-309` the peak `1 / (pi scale)` is itself unrepresentable, which would make
///   every point inside `|z| ~ 1.3e154` read `inf` however small its density. The last arm divides
///   the large factor out first, so only the unrepresentable neighbourhood of the mode saturates.
fn derive_pdf(loc: f64, scale: f64) -> impl Fn(f64) -> f64 {
    // Divided twice rather than by the product: `pi scale` overflows above `scale ~ 5.7e307`, which
    // would take the peak to `0` and the whole density with it.
    let peak = 1.0 / PI / scale;
    move |value: f64| {
        let distance = value - loc;
        let z = distance / scale;
        let z_squared = z * z;
        if !z_squared.is_finite() {
            scale / distance / distance / PI
        } else if peak.is_finite() {
            peak / (1.0 + z_squared)
        } else {
            1.0 / PI / (1.0 + z_squared) / scale
        }
    }
}

/// `-(ln(pi scale) + ln(1 + z^2))`: `ln_1p` keeps the second-order term of a tiny `z`, and
/// `ln(scale) - 2 ln|d| - ln(pi)` takes over once the square has overflowed. Both spellings sum the
/// logs rather than logging a product, so neither `pi scale` nor `z` has to be representable.
fn derive_ln_pdf(loc: f64, scale: f64) -> impl Fn(f64) -> f64 {
    let ln_scale = scale.ln();
    let ln_peak = LN_PI + ln_scale;
    move |value: f64| {
        let distance = value - loc;
        let z = distance / scale;
        let z_squared = z * z;
        if z_squared.is_finite() {
            -(ln_peak + z_squared.ln_1p())
        } else {
            ln_scale - 2.0 * distance.abs().ln() - LN_PI
        }
    }
}

/// `cdf = 1/2 + atan(z) / pi`, as `atan2(scale, loc - value) / pi`. The literal form cancels in the
/// lower tail (absolute error `~1e-17` on a value `~1 / (pi |z|)`, so relative error grows with
/// `|z|`); the two-argument arctangent has no subtraction. It also takes the distance and the scale
/// unreduced, since `atan2(k y, k x)` is `atan2(y, x)` for any `k > 0` and `scale` is checked
/// positive: standardising first would lose the whole tail to `inf` once `|d| / scale` overflows,
/// which needs only `scale = 1e-300` and `|d| > 1.8e8`.
fn derive_cdf(loc: f64, scale: f64) -> impl Fn(f64) -> f64 {
    move |value: f64| scale.atan2(loc - value) / PI
}

/// The mirror of [`derive_cdf`]: `atan2(scale, value - loc) / pi`, exact in the upper tail.
fn derive_sf(loc: f64, scale: f64) -> impl Fn(f64) -> f64 {
    move |value: f64| scale.atan2(value - loc) / PI
}

/// `ln(cdf)` at the signed distance `distance = value - loc`. The outer tail
/// `min(cdf, sf) = atan(scale / |d|) / pi` is read as `ln(tail)` below the median and as
/// `ln_1p(-tail)` above it, where `ln(cdf)` would round the answer's own magnitude away. `ln_sf` is
/// the same function at `-distance`, so the two share one body; `distance == 0` and `-0.0` both take
/// the `ln_1p` arm, which is the `ln(0.5)` the other arm would also give.
///
/// The tail mass itself leaves `f64` range once `scale / |d|` does, near `|d| / scale ~ 2e323`, and
/// its log is still an ordinary number there (`-747` at `scale = 1e-300`, `d = -1e24`). Below the
/// smallest normal the small-angle limit `atan(t) = t` is exact to far beyond `f64`, so that arm
/// reads `ln(scale) - ln|d| - ln(pi)` and never logs a rounded-away tail.
fn ln_lower_tail(scale: f64, distance: f64) -> f64 {
    let tail = scale.atan2(distance.abs()) / PI;
    if distance >= 0.0 {
        (-tail).ln_1p()
    } else if tail >= f64::MIN_POSITIVE {
        tail.ln()
    } else {
        scale.ln() - distance.abs().ln() - LN_PI
    }
}

fn derive_ln_cdf(loc: f64, scale: f64) -> impl Fn(f64) -> f64 {
    move |value: f64| ln_lower_tail(scale, value - loc)
}

fn derive_ln_sf(loc: f64, scale: f64) -> impl Fn(f64) -> f64 {
    move |value: f64| ln_lower_tail(scale, loc - value)
}

/// The standard quantile `tan(pi (q - 1/2))`, spelled per third of the unit interval so every
/// tangent argument keeps the quantile's own relative precision: `-cot(pi q)` below `1/4`,
/// `cot(pi (1 - q))` above `3/4` (Sterbenz makes `1 - q` exact there), the tangent itself between.
/// The single form loses digits in both tails: `q - 1/2` drops the low bits of a small `q` and the
/// tangent magnifies them by `1 / (pi q)^2`, to `~1e-9` relative at `q = 1e-8`. `q.abs()` keeps a
/// `-0.0` quantile on the `-inf` side; the closed endpoints fall out of `1 / tan(0)`.
fn standard_quantile(q: f64) -> f64 {
    if q < 0.25 {
        -1.0 / (PI * q.abs()).tan()
    } else if q > 0.75 {
        1.0 / (PI * (1.0 - q)).tan()
    } else {
        (PI * (q - 0.5)).tan()
    }
}

fn derive_ppf(loc: f64, scale: f64) -> impl Fn(f64) -> f64 {
    move |q: f64| loc + scale * standard_quantile(q)
}

/// `isf(q) = ppf(1 - q)` through the symmetry `2 loc - ppf(q)`, solved on `q` itself.
fn derive_isf(loc: f64, scale: f64) -> impl Fn(f64) -> f64 {
    move |q: f64| loc - scale * standard_quantile(q)
}

#[polars_expr(output_type=Float64)]
fn cauchy_pdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, derive_pdf, on_the_line)
}

#[polars_expr(output_type=Float64)]
fn cauchy_ln_pdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, derive_ln_pdf, on_the_line)
}

#[polars_expr(output_type=Float64)]
fn cauchy_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, derive_cdf, on_the_line)
}

#[polars_expr(output_type=Float64)]
fn cauchy_ln_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, derive_ln_cdf, on_the_line)
}

#[polars_expr(output_type=Float64)]
fn cauchy_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, derive_sf, on_the_line)
}

#[polars_expr(output_type=Float64)]
fn cauchy_ln_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, derive_ln_sf, on_the_line)
}

#[polars_expr(output_type=Float64)]
fn cauchy_ppf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, derive_ppf, on_unit_interval)
}

#[polars_expr(output_type=Float64)]
fn cauchy_isf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, derive_isf, on_unit_interval)
}

#[polars_expr(output_type=Float64)]
fn cauchy_pdf_scalar(inputs: &[Series], kwargs: CauchyParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_pdf, on_the_line)
}

#[polars_expr(output_type=Float64)]
fn cauchy_ln_pdf_scalar(inputs: &[Series], kwargs: CauchyParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_ln_pdf, on_the_line)
}

#[polars_expr(output_type=Float64)]
fn cauchy_cdf_scalar(inputs: &[Series], kwargs: CauchyParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_cdf, on_the_line)
}

#[polars_expr(output_type=Float64)]
fn cauchy_ln_cdf_scalar(inputs: &[Series], kwargs: CauchyParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_ln_cdf, on_the_line)
}

#[polars_expr(output_type=Float64)]
fn cauchy_sf_scalar(inputs: &[Series], kwargs: CauchyParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_sf, on_the_line)
}

#[polars_expr(output_type=Float64)]
fn cauchy_ln_sf_scalar(inputs: &[Series], kwargs: CauchyParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_ln_sf, on_the_line)
}

#[polars_expr(output_type=Float64)]
fn cauchy_ppf_scalar(inputs: &[Series], kwargs: CauchyParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_ppf, on_unit_interval)
}

#[polars_expr(output_type=Float64)]
fn cauchy_isf_scalar(inputs: &[Series], kwargs: CauchyParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_isf, on_unit_interval)
}

/// The validated `scale`, which the Python moments gate on.
#[polars_expr(output_type=Float64)]
fn cauchy_scale(inputs: &[Series]) -> PolarsResult<Series> {
    validated_pair(inputs, coerce_f64, check_params)
}

#[inline]
fn draw(dist: &Cauchy, rng: &mut impl rand::Rng) -> f64 {
    RandDistribution::sample(dist, rng)
}

#[polars_expr(output_type=Float64)]
fn cauchy_sample(inputs: &[Series], kwargs: SampleKwargs) -> PolarsResult<Series> {
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
fn cauchy_sample_scalar(
    inputs: &[Series],
    kwargs: SampleScalarKwargs<CauchyParams>,
) -> PolarsResult<Series> {
    let dist = kwargs.params.build()?;
    sample_by_index(&inputs[0], kwargs.seed, |rng| draw(&dist, rng))
}

#[polars_expr(output_type_func_with_kwargs=samples_f64_output)]
fn cauchy_samples_scalar(
    inputs: &[Series],
    kwargs: SamplesScalarKwargs<CauchyParams>,
) -> PolarsResult<Series> {
    let dist = kwargs.params.build()?;
    samples_by_index(&inputs[0], kwargs.seed, kwargs.size, |rng| draw(&dist, rng))
}

#[polars_expr(output_type_func_with_kwargs=samples_f64_output)]
fn cauchy_samples(inputs: &[Series], kwargs: SamplesKwargs) -> PolarsResult<Series> {
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
