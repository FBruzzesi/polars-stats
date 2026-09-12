use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distr::{Distribution, StandardUniform};

use crate::distributions::{
    coerce_f64, on_unit_interval, param_keyed, value_keyed_derived_ternary, value_keyed_scalar,
    PairDomain, ParamDomain,
};
use crate::rng::{
    sample_by_index, sample_per_row_ternary, samples_by_index, samples_f64_output,
    samples_per_row_ternary, SampleKwargs, SampleScalarKwargs, SamplesKwargs, SamplesScalarKwargs,
};

const MIN: ParamDomain = ParamDomain::finite("min");
const MAX: ParamDomain = ParamDomain::finite("max");

/// `max - min` overflows to `inf` for `min=-1e308, max=1e308`, and every derived quantity with it.
const BOUNDS: PairDomain<f64> = PairDomain {
    names: ("min", "max"),
    rule: "strictly greater than min, and max - min must be finite",
    accepts: |min, max| max > min && (max - min).is_finite(),
};

/// The samplers' per-row state: the bounds themselves, checked.
fn checked_bounds(min: f64, max: f64) -> PolarsResult<(f64, f64)> {
    MIN.check(min)?;
    MAX.check(max)?;
    BOUNDS.check(min, max)?;
    Ok((min, max))
}

fn check_params(min: &Float64Chunked, max: &Float64Chunked) -> PolarsResult<()> {
    MIN.check_column(min)?;
    MAX.check_column(max)?;
    BOUNDS.check_columns(min, max)
}

/// Constant parameters, deserialised once per call; every `_scalar` twin checks them once here.
#[derive(serde::Deserialize)]
struct UniformParams {
    min: f64,
    max: f64,
}

impl UniformParams {
    fn value_keyed<Branches>(
        &self,
        value: &Series,
        derive: impl Fn(f64, f64) -> Branches,
        select: impl Fn(&Branches, f64) -> Option<f64>,
    ) -> PolarsResult<Series> {
        checked_bounds(self.min, self.max)?;
        value_keyed_scalar(value, &derive(self.min, self.max), select)
    }
}

/// Where both inverses switch which bound they interpolate from.
const MEDIAN_QUANTILE: f64 = 0.5;

/// `min + range / 2` rather than `(min + max) / 2`: `min + max` can overflow where `range` is
/// already known finite.
fn midpoint(min: f64, range: f64) -> f64 {
    min + range / 2.0
}

/// `pdf` / `log_pdf`: the answer on the closed support `[min, max]`, and off it.
struct Density {
    min: f64,
    max: f64,
    off_support: f64,
    on_support: f64,
}

impl Density {
    fn pdf(min: f64, max: f64) -> Self {
        Density {
            min,
            max,
            off_support: 0.0,
            on_support: 1.0 / (max - min),
        }
    }

    /// `-ln(range)` from the width rather than `ln(pdf)`, so it stays exact where the density has
    /// under- or overflowed.
    fn ln_pdf(min: f64, max: f64) -> Self {
        Density {
            min,
            max,
            off_support: f64::NEG_INFINITY,
            on_support: -(max - min).ln(),
        }
    }

    /// Closed at both ends (scipy's convention), so `pdf(max) == 1 / range`; the one asymmetry with
    /// the sampler, which draws on `[min, max)`.
    fn at(&self, value: f64) -> Option<f64> {
        Some(if value < self.min || value > self.max {
            self.off_support
        } else {
            self.on_support
        })
    }
}

/// `cdf` / `log_cdf` / `sf` / `log_sf`: the three places a point lands. These saturate from `max`
/// up, so `max` takes `at_or_above_max`.
struct Regions<Arm> {
    min: f64,
    max: f64,
    below_min: f64,
    at_or_above_max: f64,
    interior: Arm,
}

impl<Arm: Fn(f64) -> f64> Regions<Arm> {
    fn at(&self, value: f64) -> Option<f64> {
        Some(if value < self.min {
            self.below_min
        } else if value >= self.max {
            self.at_or_above_max
        } else {
            (self.interior)(value)
        })
    }
}

fn derive_cdf(min: f64, max: f64) -> Regions<impl Fn(f64) -> f64> {
    Regions {
        min,
        max,
        below_min: 0.0,
        at_or_above_max: 1.0,
        interior: {
            let range = max - min;
            move |value: f64| (value - min) / range
        },
    }
}

/// `ln(cdf)` below the midpoint, `ln_1p(-sf)` above it. Approaching `max` the cdf ratio rounds to
/// exactly `1` and its log to `0` where the truth is a small negative, so the near-certain half reads
/// the survival fraction through `ln_1p`.
fn derive_ln_cdf(min: f64, max: f64) -> Regions<impl Fn(f64) -> f64> {
    Regions {
        min,
        max,
        below_min: f64::NEG_INFINITY,
        at_or_above_max: 0.0,
        interior: {
            let range = max - min;
            let midpoint = midpoint(min, range);
            move |value: f64| {
                if value > midpoint {
                    (-((max - value) / range)).ln_1p()
                } else {
                    ((value - min) / range).ln()
                }
            }
        },
    }
}

/// `(max - value) / range`; never `1 - cdf`, which quantises the upper tail to the `1.1e-16` spacing
/// of `1.0`.
fn derive_sf(min: f64, max: f64) -> Regions<impl Fn(f64) -> f64> {
    Regions {
        min,
        max,
        below_min: 1.0,
        at_or_above_max: 0.0,
        interior: {
            let range = max - min;
            move |value: f64| (max - value) / range
        },
    }
}

/// The mirror of [`derive_ln_cdf`]: `ln_1p` on the lower half, where the survival function is the
/// near-certain one.
fn derive_ln_sf(min: f64, max: f64) -> Regions<impl Fn(f64) -> f64> {
    Regions {
        min,
        max,
        below_min: 0.0,
        at_or_above_max: f64::NEG_INFINITY,
        interior: {
            let range = max - min;
            let midpoint = midpoint(min, range);
            move |value: f64| {
                if value > midpoint {
                    ((max - value) / range).ln()
                } else {
                    (-((value - min) / range)).ln_1p()
                }
            }
        },
    }
}

/// `ppf` (`ascending`) and `isf`, interpolating from whichever bound the answer is nearest:
/// `min + q * range` is a difference of nearly equal numbers once the result lands near `max`, so
/// above [`MEDIAN_QUANTILE`] the answer anchors to the far bound through `1 - q`, which Sterbenz
/// makes exact there. Mirroring rather than `ppf(1 - q)`: below `q ~ 1.1e-16` that complement rounds
/// to `1.0`.
fn derive_inverse(min: f64, max: f64, ascending: bool) -> impl Fn(f64) -> f64 {
    let range = max - min;
    let (at_zero, at_one, step) = if ascending {
        (min, max, range)
    } else {
        (max, min, -range)
    };
    move |quantile: f64| {
        if quantile <= MEDIAN_QUANTILE {
            at_zero + quantile * step
        } else {
            at_one - (1.0 - quantile) * step
        }
    }
}

fn derive_ppf(min: f64, max: f64) -> impl Fn(f64) -> f64 {
    derive_inverse(min, max, true)
}

fn derive_isf(min: f64, max: f64) -> impl Fn(f64) -> f64 {
    derive_inverse(min, max, false)
}

#[polars_expr(output_type=Float64)]
fn uniform_pdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, Density::pdf, Density::at)
}

#[polars_expr(output_type=Float64)]
fn uniform_ln_pdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, Density::ln_pdf, Density::at)
}

#[polars_expr(output_type=Float64)]
fn uniform_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, derive_cdf, Regions::at)
}

#[polars_expr(output_type=Float64)]
fn uniform_ln_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, derive_ln_cdf, Regions::at)
}

#[polars_expr(output_type=Float64)]
fn uniform_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, derive_sf, Regions::at)
}

#[polars_expr(output_type=Float64)]
fn uniform_ln_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, derive_ln_sf, Regions::at)
}

#[polars_expr(output_type=Float64)]
fn uniform_ppf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, derive_ppf, on_unit_interval)
}

#[polars_expr(output_type=Float64)]
fn uniform_isf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, derive_isf, on_unit_interval)
}

#[polars_expr(output_type=Float64)]
fn uniform_pdf_scalar(inputs: &[Series], kwargs: UniformParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], Density::pdf, Density::at)
}

#[polars_expr(output_type=Float64)]
fn uniform_ln_pdf_scalar(inputs: &[Series], kwargs: UniformParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], Density::ln_pdf, Density::at)
}

#[polars_expr(output_type=Float64)]
fn uniform_cdf_scalar(inputs: &[Series], kwargs: UniformParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_cdf, Regions::at)
}

#[polars_expr(output_type=Float64)]
fn uniform_ln_cdf_scalar(inputs: &[Series], kwargs: UniformParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_ln_cdf, Regions::at)
}

#[polars_expr(output_type=Float64)]
fn uniform_sf_scalar(inputs: &[Series], kwargs: UniformParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_sf, Regions::at)
}

#[polars_expr(output_type=Float64)]
fn uniform_ln_sf_scalar(inputs: &[Series], kwargs: UniformParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_ln_sf, Regions::at)
}

#[polars_expr(output_type=Float64)]
fn uniform_ppf_scalar(inputs: &[Series], kwargs: UniformParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_ppf, on_unit_interval)
}

#[polars_expr(output_type=Float64)]
fn uniform_isf_scalar(inputs: &[Series], kwargs: UniformParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_isf, on_unit_interval)
}

/// One `[lo, hi)` draw, `lo + (hi - lo) * U[0, 1)`, bypassing `statrs`' `sample`, which rebuilds a
/// `rand::distr::Uniform` sampler on every call. The multiply-add can round onto `hi` when `u` is
/// close to `1`, so the result is nudged back below `hi`; `lo < hi` guarantees `hi.next_down() >= lo`.
#[inline]
fn draw(&(lo, hi): &(f64, f64), rng: &mut impl rand::Rng) -> f64 {
    let u: f64 = StandardUniform.sample(rng);
    let x = lo + (hi - lo) * u;
    if x < hi {
        x
    } else {
        hi.next_down()
    }
}

#[polars_expr(output_type=Float64)]
fn uniform_sample(inputs: &[Series], kwargs: SampleKwargs) -> PolarsResult<Series> {
    sample_per_row_ternary(
        inputs,
        kwargs,
        coerce_f64,
        coerce_f64,
        check_params,
        checked_bounds,
        draw,
    )
}

#[polars_expr(output_type=Float64)]
fn uniform_sample_scalar(
    inputs: &[Series],
    kwargs: SampleScalarKwargs<UniformParams>,
) -> PolarsResult<Series> {
    let bounds = checked_bounds(kwargs.params.min, kwargs.params.max)?;
    sample_by_index(&inputs[0], kwargs.seed, |rng| draw(&bounds, rng))
}

#[polars_expr(output_type_func_with_kwargs=samples_f64_output)]
fn uniform_samples_scalar(
    inputs: &[Series],
    kwargs: SamplesScalarKwargs<UniformParams>,
) -> PolarsResult<Series> {
    let bounds = checked_bounds(kwargs.params.min, kwargs.params.max)?;
    samples_by_index(&inputs[0], kwargs.seed, kwargs.size, |rng| {
        draw(&bounds, rng)
    })
}

#[polars_expr(output_type_func_with_kwargs=samples_f64_output)]
fn uniform_samples(inputs: &[Series], kwargs: SamplesKwargs) -> PolarsResult<Series> {
    samples_per_row_ternary(
        inputs,
        kwargs,
        coerce_f64,
        coerce_f64,
        check_params,
        checked_bounds,
        draw,
    )
}

/// The support width `max - min`, which the Python moments derive from.
#[polars_expr(output_type=Float64)]
fn uniform_range(inputs: &[Series]) -> PolarsResult<Series> {
    param_keyed(inputs, coerce_f64, coerce_f64, check_params, |lo, hi| {
        Ok(hi - lo)
    })
}
