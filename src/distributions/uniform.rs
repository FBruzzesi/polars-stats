#![allow(clippy::unused_unit)]
use polars::prelude::arity::{try_binary_elementwise, try_ternary_elementwise};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distributions::{Distribution, Standard};
use serde::Deserialize;
use statrs::distribution::Uniform;

use crate::rng::{
    sample_scalar_plugin, samples_by_index, samples_f64_output, samples_per_row, SampleKwargs,
    SamplesKwargs,
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

/// One half-open `[lo, hi)` draw: `lo + (hi - lo) * U[0, 1)`, matching scipy's
/// `loc + scale * U[0, 1)`.
///
/// Shared by [`uniform_sample`] and [`uniform_sample_scalar`] so the two paths cannot drift; the
/// property test pinning their bit-equality only samples parameterisations, this makes it
/// structural.
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
    let u: f64 = Standard.sample(rng);
    let x = lo + (hi - lo) * u;
    if x < hi {
        x
    } else {
        hi.next_down()
    }
}

/// Element-wise continuous Uniform sampler over `[min, max)`.
///
/// `inputs[0]` carries the lower bound, `inputs[1]` the upper bound, and `inputs[2]` a per-row
/// index used to derive a per-row sub-seed, so the function is genuinely elementwise: chunking
/// and threading cannot change the output. With `seed=None`, a fresh root seed is drawn once per
/// call.
///
/// Per-row validation:
///   * `null` (in any input) propagates;
///   * `max <= min`, non-finite bounds, or a width overflowing `f64` raise `InvalidOperation`
///     (surfaces as a `ComputeError`),
///     consistent with how every distribution reports an invalid parameterisation.
///
/// Returns a `Float64` series.
#[polars_expr(output_type=Float64)]
fn uniform_sample(inputs: &[Series], kwargs: SampleKwargs) -> PolarsResult<Series> {
    let min = inputs[0].cast(&DataType::Float64)?;
    let min_ca = min.f64()?;
    let max = inputs[1].cast(&DataType::Float64)?;
    let max_ca = max.f64()?;
    let index = inputs[2].cast(&DataType::UInt64)?;
    let index_ca = index.u64()?;
    let name = inputs[0].name().clone();

    let rngs = kwargs.row_rngs();

    let ca: Float64Chunked = try_ternary_elementwise(
        min_ca,
        max_ca,
        index_ca,
        |min_opt, max_opt, i_opt| -> PolarsResult<Option<f64>> {
            match (min_opt, max_opt, i_opt) {
                (Some(lo), Some(hi), Some(i)) => {
                    // Validate the parameterisation (finite bounds and width, `max > min`) with
                    // the same error contract as `uniform_range`; the built distribution is
                    // intentionally unused for the draw (see `draw_half_open`).
                    build_dist(lo, hi)?;
                    let mut rng = rngs.rng(i);
                    Ok(Some(draw_half_open(lo, hi, &mut rng)))
                },
                _ => Ok(None),
            }
        },
    )?;

    Ok(ca.with_name(name).into_series())
}

sample_scalar_plugin! {
    /// Static parameters for the constant-bounds sampler fast path: when both bounds are Python
    /// scalars, they travel here as kwargs instead of as two full-length `pl.repeat` columns
    /// re-validated on every row.
    struct UniformScalarKwargs { min: f64, max: f64 }

    /// Constant-bounds Uniform sampler over `[min, max)`: [`uniform_sample`] for the all-scalar
    /// case. The bounds are validated once (same error contract as `uniform_range`; the built
    /// distribution is intentionally unused, see [`draw_half_open`]) and only the per-row index
    /// travels as an input; seeding and the draw are unchanged, so output matches
    /// `uniform_sample` for the same `(seed, index, min, max)`.
    fn uniform_sample_scalar(output_type = Float64, physical = Float64Type);
    build = |kw| { build_dist(kw.min, kw.max)?; (kw.min, kw.max) };
    draw = |(lo, hi), rng| draw_half_open(lo, hi, rng);
}

/// Static parameters for the constant-bounds multi-draw fast path.
///
/// Like [`UniformScalarKwargs`] plus the draw count `size` (the shared [`SamplesKwargs`] shape
/// with the scalar bounds alongside).
#[derive(Deserialize)]
struct UniformSamplesScalarKwargs {
    seed: Option<u64>,
    size: usize,
    min: f64,
    max: f64,
}

/// Constant-bounds multi-draw Uniform sampler: `size` draws per row in one call.
///
/// Row `i`'s draws are consecutive values from the one stream seeded `(seed, i)`, the same
/// stream [`uniform_sample_scalar`] takes its single draw from, so `samples(size=1)` matches
/// `sample` bit for bit and growing `size` extends each row's array. Returns
/// `Array(Float64, size)`.
#[polars_expr(output_type_func_with_kwargs=samples_f64_output)]
fn uniform_samples_scalar(
    inputs: &[Series],
    kwargs: UniformSamplesScalarKwargs,
) -> PolarsResult<Series> {
    build_dist(kwargs.min, kwargs.max)?;
    let (lo, hi) = (kwargs.min, kwargs.max);
    let name = inputs[0].name().clone();

    samples_by_index::<Float64Type, _, _>(name, &inputs[0], kwargs.seed, kwargs.size, |rng| {
        draw_half_open(lo, hi, rng)
    })
}

/// Element-wise multi-draw Uniform sampler over `[min, max)`: `size` draws per row in one
/// call.
///
/// The column-parameter counterpart of [`uniform_samples_scalar`], replacing `samples`' former
/// construction of `k` [`uniform_sample`] calls glued by `concat_arr`: the bounds are validated
/// once per row instead of once per draw. Row `i`'s draws come from the one stream seeded
/// `(seed, i)` via the shared [`draw_half_open`], so output is bit-identical to the scalar path
/// for the same bounds (the seeding is positional, parameters never enter it, so equal-bound
/// rows still draw independently). Null/error contract follows [`uniform_sample`] per row; a
/// null row yields a null array element. Returns `Array(Float64, size)`.
#[polars_expr(output_type_func_with_kwargs=samples_f64_output)]
fn uniform_samples(inputs: &[Series], kwargs: SamplesKwargs) -> PolarsResult<Series> {
    let min = inputs[0].cast(&DataType::Float64)?;
    let min_ca = min.f64()?;
    let max = inputs[1].cast(&DataType::Float64)?;
    let max_ca = max.f64()?;
    let index = inputs[2].cast(&DataType::UInt64)?;
    let index_ca = index.u64()?;
    let name = inputs[0].name().clone();

    let rows =
        min_ca
            .iter()
            .zip(max_ca.iter())
            .zip(index_ca.iter())
            .map(
                |((min_opt, max_opt), i_opt)| match (min_opt, max_opt, i_opt) {
                    (Some(lo), Some(hi), Some(i)) => {
                        build_dist(lo, hi)?;
                        Ok(Some((i, (lo, hi))))
                    },
                    _ => Ok(None),
                },
            );

    samples_per_row::<Float64Type, _, _, _, _>(
        name,
        kwargs.seed,
        kwargs.size,
        rows,
        |&(lo, hi), rng| draw_half_open(lo, hi, rng),
    )
}

/// Element-wise support width `max - min`, validating the parameterisation.
///
/// `inputs[0]` is the lower bound, `inputs[1]` the upper bound. `null` in either propagates;
/// `max <= min`, non-finite bounds, or a width overflowing `f64` raise `InvalidOperation`
/// (surfaces as a `ComputeError`).
///
/// Every closed-form Python method derives from this width, so routing it through Rust is what lets
/// them report an invalid parameterisation consistently with `uniform_sample`, instead of silently
/// producing a negative or infinite result.
#[polars_expr(output_type=Float64)]
fn uniform_range(inputs: &[Series]) -> PolarsResult<Series> {
    let min = inputs[0].cast(&DataType::Float64)?;
    let min_ca = min.f64()?;
    let max = inputs[1].cast(&DataType::Float64)?;
    let max_ca = max.f64()?;
    let name = inputs[0].name().clone();

    let ca: Float64Chunked = try_binary_elementwise(
        min_ca,
        max_ca,
        |min_opt, max_opt| -> PolarsResult<Option<f64>> {
            match (min_opt, max_opt) {
                (Some(lo), Some(hi)) => {
                    build_dist(lo, hi)?;
                    Ok(Some(hi - lo))
                },
                _ => Ok(None),
            }
        },
    )?;

    Ok(ca.with_name(name).into_series())
}
