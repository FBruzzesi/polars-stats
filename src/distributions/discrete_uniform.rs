use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distr::Distribution;
use statrs::distribution::DiscreteUniform;

use crate::distributions::{align_inputs, validate_params_binary};
use crate::rng::{
    sample_by_index, sample_per_row_ternary, samples_by_index, samples_i64_output, samples_per_row,
    ternary_param_rows, SampleKwargs, SampleScalarKwargs, SamplesKwargs, SamplesScalarKwargs,
};

/// Construct a `statrs::DiscreteUniform`, mapping an invalid parameterisation to `InvalidOperation`.
///
/// `statrs` only rejects `max < min` (`min == max` is the legitimate one-point mass), so the one
/// extra guard here is the support count: `max - min + 1` overflows `i64` for a span wider than
/// `i64::MAX - 1`, and every closed form divides by that count. The width is computed in `i128`
/// so the check itself cannot wrap.
fn build_dist(min: i64, max: i64) -> PolarsResult<DiscreteUniform> {
    let n = max as i128 - min as i128 + 1;
    if n > i64::MAX as i128 {
        return Err(PolarsError::InvalidOperation(
            format!("support width max - min + 1 must fit in i64, got min={min}, max={max}").into(),
        ));
    }
    DiscreteUniform::new(min, max).map_err(|e| {
        PolarsError::InvalidOperation(
            format!("max must be greater than or equal to min, got min={min}, max={max}: {e}")
                .into(),
        )
    })
}

/// Widen a bound column to the `Int64` both distribution types take.
///
/// The signed counterpart of binomial's `coerce_n`: a float dtype is refused rather than cast
/// (a bare cast would silently truncate `2.7` to `2`), and the strict cast to `Int64` is what
/// reports values outside the range. A `Null`-dtype column widens to all-null so null-in-null-out
/// holds; the dtype gate stays column-level, so a *float* column is refused even when every value
/// in it is null.
fn coerce_bound(bound: &Series) -> PolarsResult<Int64Chunked> {
    let dtype = bound.dtype();
    if dtype == &DataType::Null {
        return Ok(bound.cast(&DataType::Int64)?.i64()?.clone());
    }
    if !dtype.is_integer() {
        let msg = format!(
            "bounds must be integer columns, got {dtype}: cast it explicitly, e.g. \
             `pl.col(\"max\").cast(pl.Int64)`. The bounds are inclusive integers, so casting it \
             here would silently truncate a fractional value"
        );
        return Err(PolarsError::InvalidOperation(msg.into()));
    }
    let cast = bound.strict_cast(&DataType::Int64).map_err(|e| {
        PolarsError::InvalidOperation(
            format!("bounds must be integers that fit in i64: {e}").into(),
        )
    })?;
    Ok(cast.i64()?.clone())
}

/// DiscreteUniform's constant bounds, deserialised once per call.
#[derive(serde::Deserialize)]
struct DiscreteUniformParamsKwargs {
    min: i64,
    max: i64,
}

impl DiscreteUniformParamsKwargs {
    fn build(&self) -> PolarsResult<DiscreteUniform> {
        build_dist(self.min, self.max)
    }
}

/// Element-wise validation of the `(min, max)` parameterisation: returns the support count
/// `N = max - min + 1` as `Float64`, raising if `max < min` or the width overflows `i64`.
/// `null` propagates.
///
/// Every closed-form Python method divides by this count, so routing it through Rust is what lets
/// them report an invalid parameterisation consistently with `discreteuniform_sample`, instead of
/// silently dividing by zero or a negative.
///
/// Callers that need one count *per row* rather than a broadcast scalar append a third input, whose
/// height [`align_inputs`] broadcasts the bounds up to; the closure never reads it. `_ppf` and `_isf`
/// need that, because their step-boundary correction divides by the count and compares against the
/// quantile with no slack: a length-1 count makes polars see a broadcast-scalar division, which the
/// in-memory engine evaluates as a reciprocal multiply, one ulp from the true division and exactly at
/// the steps where that flips the comparison to the wrong side of a support point.
#[polars_expr(output_type=Float64)]
fn discreteuniform_range(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = align_inputs(inputs)?;
    let min = coerce_bound(&inputs[0])?;
    let max = coerce_bound(&inputs[1])?;

    validate_params_binary(&min, &max, |lo, hi| {
        build_dist(lo, hi)?;
        Ok((hi as i128 - lo as i128 + 1) as f64)
    })
}

/// One DiscreteUniform draw from a `&mut` per-row RNG already seeded from `(root_seed, index)`.
///
/// Every sampler draws through here, per-row and fast path alike, so their bit-equality is
/// structural rather than only sampled by `sample_test.py`.
#[inline]
fn draw(dist: &DiscreteUniform, rng: &mut impl rand::Rng) -> i64 {
    <DiscreteUniform as Distribution<i64>>::sample(dist, rng)
}

/// Element-wise DiscreteUniform sampler over `(min, max, row_index)`, returning `Int64`.
///
/// Per row, `null` propagates and an invalid parameterisation raises via [`build_dist`]. Seeding
/// and chunk-invariance follow [`sample_per_row_ternary`].
#[polars_expr(output_type=Int64)]
fn discreteuniform_sample(inputs: &[Series], kwargs: SampleKwargs) -> PolarsResult<Series> {
    let inputs = align_inputs(inputs)?;
    let min = coerce_bound(&inputs[0])?;
    let max = coerce_bound(&inputs[1])?;
    let index = inputs[2].cast(&DataType::UInt64)?;
    let name = inputs[0].name().clone();

    sample_per_row_ternary(
        name,
        &min,
        &max,
        index.u64()?,
        kwargs.seed,
        build_dist,
        draw,
    )
}

/// Constant-parameter fast path for [`discreteuniform_sample`].
#[polars_expr(output_type=Int64)]
fn discreteuniform_sample_scalar(
    inputs: &[Series],
    kwargs: SampleScalarKwargs<DiscreteUniformParamsKwargs>,
) -> PolarsResult<Series> {
    let dist = kwargs.params.build()?;
    let name = inputs[0].name().clone();

    sample_by_index(name, &inputs[0], kwargs.seed, |rng| draw(&dist, rng))
}

/// Constant-parameter multi-draw fast path: the `samples` twin of
/// [`discreteuniform_sample_scalar`].
///
/// `size` consecutive draws from each row's stream, so `samples(size=1)` matches `sample` bit for
/// bit and the distribution is built once per call. Returns `Array(Int64, size)`.
#[polars_expr(output_type_func_with_kwargs=samples_i64_output)]
fn discreteuniform_samples_scalar(
    inputs: &[Series],
    kwargs: SamplesScalarKwargs<DiscreteUniformParamsKwargs>,
) -> PolarsResult<Series> {
    let dist = kwargs.params.build()?;
    let name = inputs[0].name().clone();

    samples_by_index(name, &inputs[0], kwargs.seed, kwargs.size, |rng| {
        draw(&dist, rng)
    })
}

/// Element-wise multi-draw DiscreteUniform sampler: `size` draws per row in one call, the
/// distribution built once per row. Returns `Array(Int64, size)`.
///
/// Seeding and the null/error contract follow [`samples_per_row`] and [`discreteuniform_sample`].
#[polars_expr(output_type_func_with_kwargs=samples_i64_output)]
fn discreteuniform_samples(inputs: &[Series], kwargs: SamplesKwargs) -> PolarsResult<Series> {
    let inputs = align_inputs(inputs)?;
    let min = coerce_bound(&inputs[0])?;
    let max = coerce_bound(&inputs[1])?;
    let index = inputs[2].cast(&DataType::UInt64)?;
    let name = inputs[0].name().clone();

    let rows = ternary_param_rows(&min, &max, index.u64()?, build_dist);

    samples_per_row(name, rows, kwargs.seed, kwargs.size, draw)
}
