use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distr::{Distribution, StandardUniform};
use statrs::distribution::Uniform;

use crate::distributions::{
    align_inputs, validate_params_binary, value_keyed_derived_pair_per_row, value_keyed_scalar,
    Domain,
};
use crate::rng::{
    sample_by_index, sample_per_row_ternary, samples_by_index, samples_f64_output, samples_per_row,
    ternary_param_rows, SampleKwargs, SampleScalarKwargs, SamplesKwargs, SamplesScalarKwargs,
};

fn build_dist(min: f64, max: f64) -> PolarsResult<(f64, f64)> {
    // `statrs` accepts any finite `min < max`, but a support wider than `f64::MAX` (e.g.
    // `min=-1e308, max=1e308`) makes `max - min` overflow to `inf`, and with it every derived
    // quantity: `range`, the moments, and the draw itself would all silently emit `inf` instead
    // of erroring. Reject it here so all uniform plugins report it as an invalid
    // parameterisation.
    if !(max - min).is_finite() {
        return Err(PolarsError::InvalidOperation(
            format!("max - min must be finite, got min={min:e}, max={max:e}").into(),
        ));
    }
    Uniform::new(min, max).map_err(|e| {
        PolarsError::InvalidOperation(
            format!("max must be strictly greater than min, got min={min}, max={max}: {e}").into(),
        )
    })?;
    Ok((min, max))
}

/// Uniform's constant bounds, deserialised once per call.
#[derive(serde::Deserialize)]
struct UniformParamsKwargs {
    min: f64,
    max: f64,
}

impl UniformParamsKwargs {
    /// Constant-bounds twin of [`value_keyed_derived_pair_per_row`]: validates and derives once per
    /// call, then maps `select` over the evaluation-point column.
    ///
    /// The Python side routes here only once both bounds are Python scalars, so `derive` always sees
    /// `Some` and the bound-only terms it hoists (`1 / range`, `-ln(range)`) are computed once
    /// instead of per row.
    fn value_keyed<Branches>(
        &self,
        value: &Series,
        derive: impl Fn(Option<f64>, Option<f64>) -> Branches,
        select: impl Fn(&Branches, f64) -> Option<f64>,
    ) -> PolarsResult<Series> {
        build_dist(self.min, self.max)?;
        let branches = derive(Some(self.min), Some(self.max));
        value_keyed_scalar(value, |v| select(&branches, v))
    }
}

/// Where both inverses switch which bound they interpolate from; see [`derive_inverse`].
const MEDIAN_QUANTILE: f64 = 0.5;

/// A validated `(min, max)` pair and its width, which every branch table below reads.
#[derive(Clone, Copy)]
struct Span {
    min: f64,
    max: f64,
    range: f64,
}

/// `None` when either bound is null, leaving the `interior` / `on_support` slots below `None` on
/// exactly the rows whose answer needs both bounds.
fn span(min: Option<f64>, max: Option<f64>) -> Option<Span> {
    let (min, max) = (min?, max?);
    Some(Span {
        min,
        max,
        range: max - min,
    })
}

impl Span {
    /// Where the two log methods swap conditioning, as `min + range / 2` rather than
    /// `(min + max) / 2`.
    ///
    /// The two round differently on a span that straddles zero, and `min + max` can overflow where
    /// `range / 2` cannot: `range` is already known finite.
    fn midpoint(self) -> f64 {
        self.min + self.range / 2.0
    }
}

/// `pdf` / `log_pdf`: the answer on the closed support `[min, max]`, and off it.
///
/// The bounds are `Option`s and the off-support answer a plain `f64`, so a slot answers whenever the
/// bound that places the point is known, whatever the other one is.
struct Density {
    min: Option<f64>,
    max: Option<f64>,
    off_support: f64,
    on_support: Option<f64>,
}

impl Density {
    /// `1 / range` on the support, `0` off it.
    fn pdf(min: Option<f64>, max: Option<f64>) -> Self {
        Density {
            min,
            max,
            off_support: 0.0,
            on_support: span(min, max).map(|s| 1.0 / s.range),
        }
    }

    /// `-ln(range)` on the support, `-inf` off it. From the width rather than as `ln(pdf)`, so it
    /// stays exact where the density itself has underflowed or overflowed.
    fn ln_pdf(min: Option<f64>, max: Option<f64>) -> Self {
        Density {
            min,
            max,
            off_support: f64::NEG_INFINITY,
            on_support: span(min, max).map(|s| -s.range.ln()),
        }
    }

    /// The support is closed at both ends, so `max` is on it (`pdf(max) == 1 / range`) and only
    /// `value > max` is outside. scipy's convention, and the one asymmetry with `uniform_sample`,
    /// which draws on the half-open `[min, max)`.
    ///
    /// A `NaN` point never reaches here: both drivers short-circuit it.
    fn at(&self, value: f64) -> Option<f64> {
        if self.min.is_some_and(|min| value < min) || self.max.is_some_and(|max| value > max) {
            return Some(self.off_support);
        }
        self.on_support
    }
}

/// `cdf` / `log_cdf` / `sf` / `log_sf`: the three places an evaluation point lands, and the answer
/// at each.
///
/// These saturate from `max` up, so `max` takes `at_or_above_max` rather than the interior.
struct Regions<Arm> {
    min: Option<f64>,
    max: Option<f64>,
    below_min: f64,
    at_or_above_max: f64,
    interior: Option<Arm>,
}

impl<Arm: Fn(f64) -> f64> Regions<Arm> {
    fn at(&self, value: f64) -> Option<f64> {
        if self.min.is_some_and(|min| value < min) {
            return Some(self.below_min);
        }
        if self.max.is_some_and(|max| value >= max) {
            return Some(self.at_or_above_max);
        }
        self.interior.as_ref().map(|arm| arm(value))
    }
}

/// `(value - min) / range` on the support, clamped to `0` below and `1` from `max` up.
fn derive_cdf(min: Option<f64>, max: Option<f64>) -> Regions<impl Fn(f64) -> f64> {
    Regions {
        min,
        max,
        below_min: 0.0,
        at_or_above_max: 1.0,
        interior: span(min, max).map(|s| move |value: f64| (value - s.min) / s.range),
    }
}

/// `ln(cdf)` below the midpoint, `ln_1p(-sf)` above it; `-inf` below `min`, `0` from `max` up.
///
/// Approaching `max` the cdf ratio rounds to exactly `1` and its log to `0`, where the truth is a
/// small negative, so the near-certain half reads the *survival* fraction through `ln_1p` instead.
/// The other half is well conditioned and takes the plain log, the same branch `normal.rs`'s
/// `ln_half_erfc` takes.
fn derive_ln_cdf(min: Option<f64>, max: Option<f64>) -> Regions<impl Fn(f64) -> f64> {
    Regions {
        min,
        max,
        below_min: f64::NEG_INFINITY,
        at_or_above_max: 0.0,
        interior: span(min, max).map(|s| {
            let midpoint = s.midpoint();
            move |value: f64| {
                if value > midpoint {
                    (-((s.max - value) / s.range)).ln_1p()
                } else {
                    ((value - s.min) / s.range).ln()
                }
            }
        }),
    }
}

/// `(max - value) / range` on the support, `1` below `min` and `0` from `max` up.
///
/// The closed form, never `1 - cdf`: the complement quantises the upper tail to the `1.1e-16`
/// spacing of `1.0` and reaches `0.0` below that.
fn derive_sf(min: Option<f64>, max: Option<f64>) -> Regions<impl Fn(f64) -> f64> {
    Regions {
        min,
        max,
        below_min: 1.0,
        at_or_above_max: 0.0,
        interior: span(min, max).map(|s| move |value: f64| (s.max - value) / s.range),
    }
}

/// The mirror of [`derive_ln_cdf`]: `0` below `min`, `-inf` from `max` up, and the `ln_1p` branch on
/// the **lower** half, which is where the survival function is the near-certain one.
fn derive_ln_sf(min: Option<f64>, max: Option<f64>) -> Regions<impl Fn(f64) -> f64> {
    Regions {
        min,
        max,
        below_min: 0.0,
        at_or_above_max: f64::NEG_INFINITY,
        interior: span(min, max).map(|s| {
            let midpoint = s.midpoint();
            move |value: f64| {
                if value > midpoint {
                    ((s.max - value) / s.range).ln()
                } else {
                    (-((value - s.min) / s.range)).ln_1p()
                }
            }
        }),
    }
}

/// `ppf` and `isf`, interpolating from whichever bound the answer is nearest.
///
/// Both inverses are one multiply-add, and both lose the answer when they interpolate from the
/// *far* bound: `min + quantile * range` is a difference of nearly equal numbers once the result
/// lands near `max`. Anchoring to the near bound makes the addition well conditioned, and the
/// `1 - quantile` it needs is formed only above [`MEDIAN_QUANTILE`], where Sterbenz makes it
/// exact.
///
/// `ascending` is `true` for `ppf`, whose quantile `0` sits at `min`, and `false` for `isf`, whose
/// sits at `max`. Mirroring rather than `ppf(1 - q)`: below `q ~ 1.1e-16` that complement rounds to
/// `1.0` and the whole tail collapses onto one bound.
fn derive_inverse(
    min: Option<f64>,
    max: Option<f64>,
    ascending: bool,
) -> Domain<impl Fn(f64) -> f64> {
    Domain {
        inside: span(min, max).map(move |s| {
            let (at_zero, at_one, step) = if ascending {
                (s.min, s.max, s.range)
            } else {
                (s.max, s.min, -s.range)
            };
            move |quantile: f64| {
                if quantile <= MEDIAN_QUANTILE {
                    at_zero + quantile * step
                } else {
                    at_one - (1.0 - quantile) * step
                }
            }
        }),
    }
}

/// `min + quantile * range`, from `max` down above the median; null outside `[0, 1]`.
fn derive_ppf(min: Option<f64>, max: Option<f64>) -> Domain<impl Fn(f64) -> f64> {
    derive_inverse(min, max, true)
}

/// `max - quantile * range`, from `min` up above the median; null outside `[0, 1]`.
fn derive_isf(min: Option<f64>, max: Option<f64>) -> Domain<impl Fn(f64) -> f64> {
    derive_inverse(min, max, false)
}

/// Element-wise pdf; see [`Density::at`] for why the support is closed at `max`.
/// See [`value_keyed_derived_pair_per_row`] for the null/error contract.
#[polars_expr(output_type=Float64)]
fn uniform_pdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_pair_per_row(inputs, build_dist, Density::pdf, Density::at)
}

/// Element-wise log-pdf; see [`Density::ln_pdf`].
#[polars_expr(output_type=Float64)]
fn uniform_ln_pdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_pair_per_row(inputs, build_dist, Density::ln_pdf, Density::at)
}

/// Element-wise cdf `P(X <= value)`; see [`derive_cdf`].
#[polars_expr(output_type=Float64)]
fn uniform_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_pair_per_row(inputs, build_dist, derive_cdf, Regions::at)
}

/// Element-wise log-cdf; see [`derive_ln_cdf`].
#[polars_expr(output_type=Float64)]
fn uniform_ln_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_pair_per_row(inputs, build_dist, derive_ln_cdf, Regions::at)
}

/// Element-wise survival function `P(X > value)`; see [`derive_sf`] for why it is not `1 - cdf`.
#[polars_expr(output_type=Float64)]
fn uniform_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_pair_per_row(inputs, build_dist, derive_sf, Regions::at)
}

/// Element-wise log-sf; see [`derive_ln_sf`].
#[polars_expr(output_type=Float64)]
fn uniform_ln_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_pair_per_row(inputs, build_dist, derive_ln_sf, Regions::at)
}

/// Element-wise ppf (inverse cdf); see [`derive_inverse`].
#[polars_expr(output_type=Float64)]
fn uniform_ppf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_pair_per_row(inputs, build_dist, derive_ppf, Domain::at)
}

/// Element-wise inverse survival function; see [`derive_inverse`] for why it never forms a
/// complement.
#[polars_expr(output_type=Float64)]
fn uniform_isf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_pair_per_row(inputs, build_dist, derive_isf, Domain::at)
}

/// Constant-bounds fast path for [`uniform_pdf`].
#[polars_expr(output_type=Float64)]
fn uniform_pdf_scalar(inputs: &[Series], kwargs: UniformParamsKwargs) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], Density::pdf, Density::at)
}

/// Constant-bounds fast path for [`uniform_ln_pdf`].
#[polars_expr(output_type=Float64)]
fn uniform_ln_pdf_scalar(inputs: &[Series], kwargs: UniformParamsKwargs) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], Density::ln_pdf, Density::at)
}

/// Constant-bounds fast path for [`uniform_cdf`].
#[polars_expr(output_type=Float64)]
fn uniform_cdf_scalar(inputs: &[Series], kwargs: UniformParamsKwargs) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_cdf, Regions::at)
}

/// Constant-bounds fast path for [`uniform_ln_cdf`].
#[polars_expr(output_type=Float64)]
fn uniform_ln_cdf_scalar(inputs: &[Series], kwargs: UniformParamsKwargs) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_ln_cdf, Regions::at)
}

/// Constant-bounds fast path for [`uniform_sf`].
#[polars_expr(output_type=Float64)]
fn uniform_sf_scalar(inputs: &[Series], kwargs: UniformParamsKwargs) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_sf, Regions::at)
}

/// Constant-bounds fast path for [`uniform_ln_sf`].
#[polars_expr(output_type=Float64)]
fn uniform_ln_sf_scalar(inputs: &[Series], kwargs: UniformParamsKwargs) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_ln_sf, Regions::at)
}

/// Constant-bounds fast path for [`uniform_ppf`].
#[polars_expr(output_type=Float64)]
fn uniform_ppf_scalar(inputs: &[Series], kwargs: UniformParamsKwargs) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_ppf, Domain::at)
}

/// Constant-bounds fast path for [`uniform_isf`].
#[polars_expr(output_type=Float64)]
fn uniform_isf_scalar(inputs: &[Series], kwargs: UniformParamsKwargs) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], derive_isf, Domain::at)
}

/// One half-open `[lo, hi)` draw: `lo + (hi - lo) * U[0, 1)`, matching scipy's
/// `loc + scale * U[0, 1)`.
///
/// Uniform's counterpart of the other distributions' `fn draw`: every Uniform sampler draws through
/// here, per-row and fast path alike, so their bit-equality is structural rather than only sampled
/// by `sample_test.py`.
///
/// This deliberately bypasses `statrs`' `Distribution::sample`, which rebuilds a
/// `rand::distr::Uniform` float sampler (scale/bias/rejection-zone setup) on *every* call: that
/// fixed per-draw cost dwarfs the single multiply-add here, enough to leave the sampler slower
/// than scipy.
///
/// `u < 1` does not survive the multiply-add's rounding: `lo + (hi - lo) * u` can land exactly on
/// `hi` (or one ulp above) when `u` is close to 1, so the result is nudged back to the largest
/// float below `hi` to keep the documented half-open contract. `lo < hi` guarantees
/// `hi.next_down() >= lo`.
#[inline]
fn draw_half_open(lo: f64, hi: f64, rng: &mut impl rand::Rng) -> f64 {
    let u: f64 = StandardUniform.sample(rng);
    let x = lo + (hi - lo) * u;
    if x < hi {
        x
    } else {
        hi.next_down()
    }
}

/// Element-wise continuous Uniform sampler over `[min, max)`, taking `(min, max, row_index)` and
/// returning `Float64`.
///
/// Per row, `null` propagates and an invalid parameterisation (`max <= min`, non-finite bounds, or
/// a width overflowing `f64`) raises via [`build_dist`]. Seeding and chunk-invariance follow
/// [`sample_per_row_ternary`].
#[polars_expr(output_type=Float64)]
fn uniform_sample(inputs: &[Series], kwargs: SampleKwargs) -> PolarsResult<Series> {
    let inputs = align_inputs(inputs)?;
    let min = inputs[0].cast(&DataType::Float64)?;
    let max = inputs[1].cast(&DataType::Float64)?;
    let index = inputs[2].cast(&DataType::UInt64)?;
    let name = inputs[0].name().clone();

    sample_per_row_ternary(
        name,
        min.f64()?,
        max.f64()?,
        index.u64()?,
        kwargs.seed,
        build_dist,
        |&(lo, hi), rng| draw_half_open(lo, hi, rng),
    )
}

/// Constant-bounds fast path for [`uniform_sample`].
#[polars_expr(output_type=Float64)]
fn uniform_sample_scalar(
    inputs: &[Series],
    kwargs: SampleScalarKwargs<UniformParamsKwargs>,
) -> PolarsResult<Series> {
    let (lo, hi) = build_dist(kwargs.params.min, kwargs.params.max)?;
    let name = inputs[0].name().clone();

    sample_by_index(name, &inputs[0], kwargs.seed, |rng| {
        draw_half_open(lo, hi, rng)
    })
}

/// Constant-bounds multi-draw fast path: the `samples` twin of [`uniform_sample_scalar`].
///
/// `size` consecutive draws from each row's stream, so `samples(size=1)` matches `sample` bit for
/// bit and the bounds are validated once per call. Returns `Array(Float64, size)`.
#[polars_expr(output_type_func_with_kwargs=samples_f64_output)]
fn uniform_samples_scalar(
    inputs: &[Series],
    kwargs: SamplesScalarKwargs<UniformParamsKwargs>,
) -> PolarsResult<Series> {
    let (lo, hi) = build_dist(kwargs.params.min, kwargs.params.max)?;
    let name = inputs[0].name().clone();

    samples_by_index(name, &inputs[0], kwargs.seed, kwargs.size, |rng| {
        draw_half_open(lo, hi, rng)
    })
}

/// Element-wise multi-draw Uniform sampler over `[min, max)`: `size` draws per row in one call,
/// the bounds validated once per row. Returns `Array(Float64, size)`.
///
/// Seeding and the null/error contract follow [`samples_per_row`] and [`uniform_sample`]; the
/// draw is the shared [`draw_half_open`].
#[polars_expr(output_type_func_with_kwargs=samples_f64_output)]
fn uniform_samples(inputs: &[Series], kwargs: SamplesKwargs) -> PolarsResult<Series> {
    let inputs = align_inputs(inputs)?;
    let min = inputs[0].cast(&DataType::Float64)?;
    let max = inputs[1].cast(&DataType::Float64)?;
    let index = inputs[2].cast(&DataType::UInt64)?;
    let name = inputs[0].name().clone();

    let rows = ternary_param_rows(min.f64()?, max.f64()?, index.u64()?, build_dist);

    samples_per_row(name, rows, kwargs.seed, kwargs.size, |&(lo, hi), rng| {
        draw_half_open(lo, hi, rng)
    })
}

/// Element-wise support width `max - min`, validating the parameterisation.
///
/// `inputs[0]` is the lower bound, `inputs[1]` the upper bound. `null` in either propagates;
/// `max <= min`, non-finite bounds, or a width overflowing `f64` raise `InvalidOperation`
/// (surfaces as a `ComputeError`).
///
/// Every closed-form Python method derives from this width, so routing it through Rust is what
/// lets them report an invalid parameterisation consistently with `uniform_sample`, instead of
/// silently producing a negative or infinite result.
#[polars_expr(output_type=Float64)]
fn uniform_range(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = align_inputs(inputs)?;
    let min = inputs[0].cast(&DataType::Float64)?;
    let max = inputs[1].cast(&DataType::Float64)?;

    validate_params_binary(min.f64()?, max.f64()?, |lo, hi| {
        build_dist(lo, hi).map(|(lo, hi)| hi - lo)
    })
}
