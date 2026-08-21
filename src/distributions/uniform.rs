#![allow(clippy::unused_unit)]
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distr::{Distribution, StandardUniform};
use statrs::distribution::Uniform;

use crate::distributions::{align_inputs, validate_params_binary};
use crate::rng::{
    sample_by_index, sample_per_row_ternary, samples_by_index, samples_f64_output, samples_per_row,
    ternary_param_rows, SampleKwargs, SampleScalarKwargs, SamplesKwargs, SamplesScalarKwargs,
};

fn build_dist(min: f64, max: f64) -> PolarsResult<Uniform> {
    // `statrs` accepts any finite `min < max`, but a support wider than `f64::MAX` (e.g.
    // `min=-1e308, max=1e308`) makes `max - min` overflow to `inf`, and with it every derived
    // quantity: `range`, the moments, and the draw itself would all silently emit `inf` instead
    // of erroring. Reject it here so all uniform plugins report it as an invalid
    // parameterisation.
    if !(max - min).is_finite() {
        // Scientific notation: this only fires for enormous bounds, whose plain `Display` is a
        // 300-digit decimal expansion.
        return Err(PolarsError::InvalidOperation(
            format!("max - min must be finite, got min={min:e}, max={max:e}").into(),
        ));
    }
    Uniform::new(min, max).map_err(|e| {
        PolarsError::InvalidOperation(
            format!("max must be strictly greater than min, got min={min}, max={max}: {e}").into(),
        )
    })
}

/// Uniform's constant bounds, deserialised once per call.
///
/// The one place this file spells `(min, max)`: both samplers validate through
/// [`Self::validated_bounds`], so the order cannot drift between them.
#[derive(serde::Deserialize)]
struct UniformParamsKwargs {
    min: f64,
    max: f64,
}

impl UniformParamsKwargs {
    /// Validate the constant bounds once per call and return them raw.
    ///
    /// The built distribution is discarded on purpose: [`draw_half_open`] wants the `(lo, hi)` pair,
    /// not a `statrs::Uniform`.
    fn validated_bounds(&self) -> PolarsResult<(f64, f64)> {
        build_dist(self.min, self.max)?;
        Ok((self.min, self.max))
    }
}

/// One half-open `[lo, hi)` draw: `lo + (hi - lo) * U[0, 1)`, matching scipy's
/// `loc + scale * U[0, 1)`.
///
/// Uniform's counterpart of the other distributions' `fn draw`: every Uniform sampler draws through
/// here, per-row and fast path alike, so their bit-equality is structural rather than only sampled
/// by `sample_test.py`.
///
/// This deliberately bypasses `statrs`' `Distribution::sample`, which rebuilds a
/// `rand::distributions::Uniform` float sampler (scale/bias/rejection-zone setup) on *every*
/// call: that fixed per-draw cost dwarfs the single multiply-add here and is what made the
/// sampler slower than scipy.
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
///
/// The built distribution is discarded on purpose: the row state is the raw `(min, max)` pair
/// [`draw_half_open`] draws from.
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
        |lo, hi| {
            build_dist(lo, hi)?;
            Ok((lo, hi))
        },
        |&(lo, hi), rng| draw_half_open(lo, hi, rng),
    )
}

/// Constant-bounds fast path for [`uniform_sample`]. The built distribution is intentionally
/// unused; the draw is [`draw_half_open`] over the raw bounds.
#[polars_expr(output_type=Float64)]
fn uniform_sample_scalar(
    inputs: &[Series],
    kwargs: SampleScalarKwargs<UniformParamsKwargs>,
) -> PolarsResult<Series> {
    let (lo, hi) = kwargs.params.validated_bounds()?;
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
    let (lo, hi) = kwargs.params.validated_bounds()?;
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

    // The built distribution is validated then dropped; the row state keeps the raw bounds, as
    // [`draw_half_open`] needs (mirrors the `uniform_sample_scalar` `build`).
    let rows = ternary_param_rows(min.f64()?, max.f64()?, index.u64()?, |lo, hi| {
        build_dist(lo, hi)?;
        Ok((lo, hi))
    });

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
        build_dist(lo, hi)?;
        Ok(hi - lo)
    })
}
