#![allow(clippy::unused_unit)]
use polars::prelude::arity::try_ternary_elementwise;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distributions::{Distribution, Standard};
use statrs::distribution::Uniform;

use crate::distributions::param_validator;
use crate::rng::{
    sample_scalar_plugin, samples_f64_output, samples_per_row, ternary_param_rows, SampleKwargs,
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

/// Element-wise continuous Uniform sampler over `[min, max)`, taking `(min, max, row_index)` and
/// returning `Float64`.
///
/// Per row, `null` propagates and an invalid parameterisation (`max <= min`, non-finite bounds, or
/// a width overflowing `f64`) raises via [`build_dist`]. Seeding and chunk-invariance follow
/// [`SampleKwargs::row_rngs`].
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
    struct UniformScalarKwargs { min: f64, max: f64 }

    /// Constant-bounds fast path for [`uniform_sample`]. The built distribution is intentionally
    /// unused; the draw is [`draw_half_open`] over the raw bounds.
    fn uniform_sample_scalar(output_type = Float64, physical = Float64Type);

    samples = uniform_samples_scalar as UniformSamplesScalarKwargs -> samples_f64_output;

    build = |kw| { build_dist(kw.min, kw.max)?; (kw.min, kw.max) };
    draw = |(lo, hi), rng| draw_half_open(lo, hi, rng);
}

/// Element-wise multi-draw Uniform sampler over `[min, max)`: `size` draws per row in one call,
/// the bounds validated once per row. Returns `Array(Float64, size)`.
///
/// Seeding and the null/error contract follow [`samples_per_row`] and [`uniform_sample`]; the
/// draw is the shared [`draw_half_open`].
#[polars_expr(output_type_func_with_kwargs=samples_f64_output)]
fn uniform_samples(inputs: &[Series], kwargs: SamplesKwargs) -> PolarsResult<Series> {
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

    samples_per_row::<Float64Type, _, _, _, _>(
        name,
        kwargs.seed,
        kwargs.size,
        rows,
        |&(lo, hi), rng| draw_half_open(lo, hi, rng),
    )
}

param_validator! {
    /// Element-wise support width `max - min`, validating the parameterisation.
    ///
    /// `inputs[0]` is the lower bound, `inputs[1]` the upper bound. `null` in either propagates;
    /// `max <= min`, non-finite bounds, or a width overflowing `f64` raise `InvalidOperation`
    /// (surfaces as a `ComputeError`).
    ///
    /// Every closed-form Python method derives from this width, so routing it through Rust is what
    /// lets them report an invalid parameterisation consistently with `uniform_sample`, instead of
    /// silently producing a negative or infinite result.
    fn uniform_range;
    params = (min: DataType::Float64 => f64, max: DataType::Float64 => f64);
    build = build_dist;
    returns = max - min;
    output_name = inputs[0];
}
