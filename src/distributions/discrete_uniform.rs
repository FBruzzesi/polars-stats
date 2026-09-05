use polars::prelude::arity::{try_ternary_elementwise, unary_elementwise};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distr::Distribution;
use statrs::distribution::DiscreteUniform;

use crate::distributions::{
    align_inputs, validate_params_binary, value_keyed_per_row, value_keyed_scalar,
};
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

    /// Constant-parameter twin of the per-row [`du_value_keyed`], sharing its `<method>_point`
    /// bodies: validate and size the support once per call, then map `f` over the evaluation-point
    /// column in its own arithmetic (see [`coerce_points`]).
    fn value_keyed<F>(&self, value: &Series, f: F) -> PolarsResult<Series>
    where
        F: Fn(&Support, Point) -> Option<f64>,
    {
        let support = build_support(self.min, self.max)?;
        let name = value.name().clone();
        let ca: Float64Chunked = match coerce_points(value)? {
            Points::Float(v) => unary_elementwise(&v, |opt| {
                opt.and_then(|v| {
                    if v.is_nan() {
                        Some(f64::NAN)
                    } else {
                        f(&support, Point::Float(v))
                    }
                })
            }),
            Points::Int(v) => unary_elementwise(&v, |opt| {
                opt.and_then(|v| f(&support, Point::Int(v.into())))
            }),
            Points::Wide(v) => unary_elementwise(&v, |opt| {
                opt.and_then(|v| f(&support, Point::Int(v.into())))
            }),
        };
        Ok(ca.with_name(name).into_series())
    }
}

/// The validated support, with the count precomputed: the state every closed-form body reads.
///
/// `n` is `(max - min + 1) as f64`, computed in `i128` exactly as [`discreteuniform_range`]
/// returns it, so the Rust bodies and the Python moments read the same rounded count.
struct Support {
    min: i64,
    max: i64,
    n: f64,
}

/// Validate `(min, max)` via [`build_dist`] and size the support once.
fn build_support(min: i64, max: i64) -> PolarsResult<Support> {
    build_dist(min, max)?;
    let n = (max as i128 - min as i128 + 1) as f64;
    Ok(Support { min, max, n })
}

/// An evaluation point in its own arithmetic: `f64` for float columns, exact `i128` for integer
/// columns. The integer side is what keeps an `int` spelling exact on supports a `Float64` cannot
/// address (documented in docs/explanation/accuracy.md), and `i128` is wide enough that a `UInt64`
/// point above `i64::MAX` still compares exactly instead of wrapping.
///
/// [`Point`], [`Placement`] and [`coerce_points`] stay private to this file on purpose:
/// DiscreteUniform is their only consumer today. Geometric is the one remaining catalogue entry
/// with closed forms over an integer support, so hoist the trio into `mod.rs` when its port makes
/// it a second consumer, not before.
#[derive(Clone, Copy)]
enum Point {
    Float(f64),
    Int(i128),
}

/// Where a point sits relative to the support, with the two counts the closed forms read.
///
/// The float side compares and counts against `Float64`-cast bounds, mirroring how polars promotes
/// a mixed `Float64`/`Int64` operation; the integer side stays exact. The mirror is a snapshot,
/// not a dependency: the promotion rule is frozen here, so a future polars change to its supertype
/// rules moves nothing in this file. `count` and `tail` are only read where the branches keep them
/// in `[1, N - 1]`; elsewhere they are computed but discarded.
struct Placement {
    below: bool,
    at_or_above_max: bool,
    on_support: bool,
    /// `floor(value) - min + 1`, the support points at or below the point, as `Float64`.
    count: f64,
    /// `max - floor(value)`, the support points strictly above the point, as `Float64`.
    tail: f64,
}

fn place(s: &Support, point: Point) -> Placement {
    match point {
        Point::Int(v) => {
            let (lo, hi) = (i128::from(s.min), i128::from(s.max));
            Placement {
                below: v < lo,
                at_or_above_max: v >= hi,
                on_support: v >= lo && v <= hi,
                count: (v - lo + 1) as f64,
                tail: (hi - v) as f64,
            }
        },
        Point::Float(v) => {
            let (lo, hi) = (s.min as f64, s.max as f64);
            Placement {
                below: v < lo,
                at_or_above_max: v >= hi,
                on_support: v >= lo && v <= hi && v.floor() == v,
                count: v.floor() - lo + 1.0,
                tail: hi - v.floor(),
            }
        },
    }
}

// Per-method bodies, shared by the per-row plugins and their `*_scalar` twins. The drivers
// short-circuit a `NaN` float point to `NaN` before these run, matching `value_keyed_scalar`.

/// `1 / N` on the integer support, `0` off it.
fn pmf_point(s: &Support, point: Point) -> Option<f64> {
    let pos = place(s, point);
    Some(if pos.on_support { 1.0 / s.n } else { 0.0 })
}

/// `-ln(N)` on the integer support, `-inf` off it: the mass is exactly `1 / N`, so its log reads
/// straight off the count instead of a rounded quotient.
fn ln_pmf_point(s: &Support, point: Point) -> Option<f64> {
    let pos = place(s, point);
    Some(if pos.on_support {
        -s.n.ln()
    } else {
        f64::NEG_INFINITY
    })
}

/// `count * (1 / N)` inside the support, `0` below `min`, `1` from `max` up.
///
/// The explicit endpoints make `cdf(max) == 1` an answer rather than the limit of a clamp:
/// `N * (1 / N)` is not `1.0` for 483 of the first 4000 support counts.
fn cdf_point(s: &Support, point: Point) -> Option<f64> {
    let pos = place(s, point);
    Some(if pos.below {
        0.0
    } else if pos.at_or_above_max {
        1.0
    } else {
        pos.count * (1.0 / s.n)
    })
}

/// `tail * (1 / N)` inside the support, `1` below `min`, `0` from `max` up: a direct count of the
/// points above the value, not `1 - cdf`, whose subtraction absorbs the tail once the cdf rounds
/// towards `1`.
fn sf_point(s: &Support, point: Point) -> Option<f64> {
    let pos = place(s, point);
    Some(if pos.below {
        1.0
    } else if pos.at_or_above_max {
        0.0
    } else {
        pos.tail * (1.0 / s.n)
    })
}

/// `ln(count / N)`, through `ln_1p` of the survival ratio once the cdf passes one half: the naive
/// log's relative error one point below the top grows with `N` (`8.7e-16` at `N = 1e3` but
/// `2.8e-08` at `N = 1e9`).
fn ln_cdf_point(s: &Support, point: Point) -> Option<f64> {
    let pos = place(s, point);
    Some(if pos.below {
        f64::NEG_INFINITY
    } else if pos.at_or_above_max {
        0.0
    } else if pos.count * 2.0 > s.n {
        (-((s.n - pos.count) * (1.0 / s.n))).ln_1p()
    } else {
        (pos.count * (1.0 / s.n)).ln()
    })
}

/// The mirror of [`ln_cdf_point`]: the near-certain side is the lower one, so `ln_1p` sits there.
fn ln_sf_point(s: &Support, point: Point) -> Option<f64> {
    let pos = place(s, point);
    Some(if pos.below {
        0.0
    } else if pos.at_or_above_max {
        f64::NEG_INFINITY
    } else if pos.count * 2.0 > s.n {
        ((s.n - pos.count) * (1.0 / s.n)).ln()
    } else {
        (-(pos.count * (1.0 / s.n))).ln_1p()
    })
}

/// `min + ceil(q * N) - 1`, corrected and clamped; `None` (null) outside `[0, 1]`.
///
/// Matches scipy's `randint.ppf` including its correction step, which probes the point below and
/// keeps it whenever that point's cdf already reaches `q`, settling the step boundaries where an
/// ulp of noise in `q * N` would skip a support point. Both inverses lose support points above
/// `2**53`; see docs/explanation/accuracy.md.
fn ppf_value(s: &Support, q: f64) -> Option<f64> {
    if !(0.0..=1.0).contains(&q) {
        return None;
    }
    let (lo, hi) = (s.min as f64, s.max as f64);
    let candidate = lo + (q * s.n).ceil() - 1.0;
    let point_below = (candidate - 1.0).clamp(lo, hi);
    let count_below = point_below.floor() - lo + 1.0;
    let chosen = if count_below / s.n >= q {
        point_below
    } else {
        candidate
    };
    Some(chosen.clamp(lo, hi))
}

/// The smallest support point whose survival mass is at most `q`, from `max - floor(q * N)`
/// corrected on both sides and clamped; `None` (null) outside `[0, 1]`.
///
/// Entered against `q` itself rather than `ppf(1 - q)`: the complement rounds before the inverse
/// runs, and the survival steps sit at multiples of `1 / N`, exactly where `1 - q` has spent its
/// precision. A rounded `q * N` can floor one point off in either direction, so the correction
/// probes both neighbours and keeps the smallest one whose survival mass `(max - x) / N` is still
/// at most `q`.
fn isf_value(s: &Support, q: f64) -> Option<f64> {
    if !(0.0..=1.0).contains(&q) {
        return None;
    }
    let (lo, hi) = (s.min as f64, s.max as f64);
    let candidate = hi - (q * s.n).floor();
    let point_below = candidate - 1.0;
    let chosen = if (hi - point_below) / s.n <= q {
        point_below
    } else if (hi - candidate) / s.n <= q {
        candidate
    } else {
        candidate + 1.0
    };
    Some(chosen.clamp(lo, hi))
}

/// The evaluation-point column in the arithmetic its dtype earns: floats (and an all-null `Null`
/// column) run the `Float64` path, integer dtypes stay exact. `UInt64` keeps its own accessor so a
/// value above `i64::MAX` reaches [`Point::Int`] via `i128` instead of failing a cast; every other
/// integer dtype fits `Int64`. A non-numeric column is refused, as polars itself refuses it in the
/// expression forms.
enum Points {
    Float(Float64Chunked),
    Int(Int64Chunked),
    Wide(UInt64Chunked),
}

fn coerce_points(value: &Series) -> PolarsResult<Points> {
    let dtype = value.dtype();
    if dtype.is_float() || dtype == &DataType::Null {
        return Ok(Points::Float(
            value.cast(&DataType::Float64)?.f64()?.clone(),
        ));
    }
    if dtype == &DataType::UInt64 {
        return Ok(Points::Wide(value.u64()?.clone()));
    }
    if dtype.is_integer() {
        return Ok(Points::Int(value.cast(&DataType::Int64)?.i64()?.clone()));
    }
    Err(PolarsError::InvalidOperation(
        format!("value must be a numeric column, got {dtype}").into(),
    ))
}

/// Apply a closed-form `f(support, point)` element-wise over `(value, min, max)`; shared by the
/// six value-keyed plugins with an integer-exact path. Null and `NaN` contracts follow
/// [`value_keyed_per_row`]: a null bound nulls the row without building, and an invalid
/// parameterisation raises via [`build_support`] whatever the row's point is.
fn du_value_keyed<F>(inputs: &[Series], f: F) -> PolarsResult<Series>
where
    F: Fn(&Support, Point) -> Option<f64>,
{
    let inputs = align_inputs(inputs)?;
    let name = inputs[0].name().clone();
    let min = coerce_bound(&inputs[1])?;
    let max = coerce_bound(&inputs[2])?;

    let ca: Float64Chunked = match coerce_points(&inputs[0])? {
        Points::Float(v) => try_ternary_elementwise(&v, &min, &max, |v, lo, hi| {
            per_row(v.map(Point::Float), lo, hi, &f)
        })?,
        Points::Int(v) => try_ternary_elementwise(&v, &min, &max, |v, lo, hi| {
            per_row(v.map(|v| Point::Int(v.into())), lo, hi, &f)
        })?,
        Points::Wide(v) => try_ternary_elementwise(&v, &min, &max, |v, lo, hi| {
            per_row(v.map(|v| Point::Int(v.into())), lo, hi, &f)
        })?,
    };
    Ok(ca.with_name(name).into_series())
}

/// One row of [`du_value_keyed`]: build before the point is read, so an invalid parameterisation
/// raises whatever the row's point is, exactly as [`value_keyed_per_row`] orders it.
#[inline]
fn per_row<F>(
    point: Option<Point>,
    min: Option<i64>,
    max: Option<i64>,
    f: &F,
) -> PolarsResult<Option<f64>>
where
    F: Fn(&Support, Point) -> Option<f64>,
{
    let (Some(min), Some(max)) = (min, max) else {
        return Ok(None);
    };
    let support = build_support(min, max)?;
    let Some(point) = point else {
        return Ok(None);
    };
    Ok(match point {
        Point::Float(v) if v.is_nan() => Some(f64::NAN),
        _ => f(&support, point),
    })
}

/// Element-wise pmf: `1 / N` on the integer support, `0` off it, `NaN` for a `NaN` value.
/// See [`du_value_keyed`] for the null/error contract.
#[polars_expr(output_type=Float64)]
fn discreteuniform_pmf(inputs: &[Series]) -> PolarsResult<Series> {
    du_value_keyed(inputs, pmf_point)
}

/// Element-wise log-pmf: `-ln(N)` on the integer support, `-inf` off it.
#[polars_expr(output_type=Float64)]
fn discreteuniform_ln_pmf(inputs: &[Series]) -> PolarsResult<Series> {
    du_value_keyed(inputs, ln_pmf_point)
}

/// Element-wise cdf `P(X <= floor(value))`; see [`cdf_point`] for the endpoint contract.
#[polars_expr(output_type=Float64)]
fn discreteuniform_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    du_value_keyed(inputs, cdf_point)
}

/// Element-wise survival function `P(X > floor(value))`; see [`sf_point`].
#[polars_expr(output_type=Float64)]
fn discreteuniform_sf(inputs: &[Series]) -> PolarsResult<Series> {
    du_value_keyed(inputs, sf_point)
}

/// Element-wise log-cdf, `ln_1p`-stable near the top of the support; see [`ln_cdf_point`].
#[polars_expr(output_type=Float64)]
fn discreteuniform_ln_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    du_value_keyed(inputs, ln_cdf_point)
}

/// Element-wise log-sf, `ln_1p`-stable near the bottom of the support; see [`ln_sf_point`].
#[polars_expr(output_type=Float64)]
fn discreteuniform_ln_sf(inputs: &[Series]) -> PolarsResult<Series> {
    du_value_keyed(inputs, ln_sf_point)
}

/// Element-wise ppf (inverse cdf), returning the integer support point as `f64`.
/// See [`ppf_value`] for the correction and endpoint contract.
#[polars_expr(output_type=Float64)]
fn discreteuniform_ppf(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = align_inputs(inputs)?;
    let value = inputs[0].cast(&DataType::Float64)?;
    let min = coerce_bound(&inputs[1])?;
    let max = coerce_bound(&inputs[2])?;
    value_keyed_per_row(
        value.f64()?,
        &min,
        &max,
        inputs[0].name().clone(),
        build_support,
        ppf_value,
    )
}

/// Element-wise inverse survival function; see [`isf_value`] for the two-sided correction.
#[polars_expr(output_type=Float64)]
fn discreteuniform_isf(inputs: &[Series]) -> PolarsResult<Series> {
    let inputs = align_inputs(inputs)?;
    let value = inputs[0].cast(&DataType::Float64)?;
    let min = coerce_bound(&inputs[1])?;
    let max = coerce_bound(&inputs[2])?;
    value_keyed_per_row(
        value.f64()?,
        &min,
        &max,
        inputs[0].name().clone(),
        build_support,
        isf_value,
    )
}

/// Constant-parameter fast path for [`discreteuniform_pmf`].
#[polars_expr(output_type=Float64)]
fn discreteuniform_pmf_scalar(
    inputs: &[Series],
    kwargs: DiscreteUniformParamsKwargs,
) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], pmf_point)
}

/// Constant-parameter fast path for [`discreteuniform_ln_pmf`].
#[polars_expr(output_type=Float64)]
fn discreteuniform_ln_pmf_scalar(
    inputs: &[Series],
    kwargs: DiscreteUniformParamsKwargs,
) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], ln_pmf_point)
}

/// Constant-parameter fast path for [`discreteuniform_cdf`].
#[polars_expr(output_type=Float64)]
fn discreteuniform_cdf_scalar(
    inputs: &[Series],
    kwargs: DiscreteUniformParamsKwargs,
) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], cdf_point)
}

/// Constant-parameter fast path for [`discreteuniform_sf`].
#[polars_expr(output_type=Float64)]
fn discreteuniform_sf_scalar(
    inputs: &[Series],
    kwargs: DiscreteUniformParamsKwargs,
) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], sf_point)
}

/// Constant-parameter fast path for [`discreteuniform_ln_cdf`].
#[polars_expr(output_type=Float64)]
fn discreteuniform_ln_cdf_scalar(
    inputs: &[Series],
    kwargs: DiscreteUniformParamsKwargs,
) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], ln_cdf_point)
}

/// Constant-parameter fast path for [`discreteuniform_ln_sf`].
#[polars_expr(output_type=Float64)]
fn discreteuniform_ln_sf_scalar(
    inputs: &[Series],
    kwargs: DiscreteUniformParamsKwargs,
) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], ln_sf_point)
}

/// Constant-parameter fast path for [`discreteuniform_ppf`].
#[polars_expr(output_type=Float64)]
fn discreteuniform_ppf_scalar(
    inputs: &[Series],
    kwargs: DiscreteUniformParamsKwargs,
) -> PolarsResult<Series> {
    let support = build_support(kwargs.min, kwargs.max)?;
    value_keyed_scalar(&inputs[0], |q| ppf_value(&support, q))
}

/// Constant-parameter fast path for [`discreteuniform_isf`].
#[polars_expr(output_type=Float64)]
fn discreteuniform_isf_scalar(
    inputs: &[Series],
    kwargs: DiscreteUniformParamsKwargs,
) -> PolarsResult<Series> {
    let support = build_support(kwargs.min, kwargs.max)?;
    value_keyed_scalar(&inputs[0], |q| isf_value(&support, q))
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
