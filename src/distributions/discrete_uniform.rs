use polars::prelude::arity::{try_ternary_elementwise, unary_elementwise};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distr::Distribution;
use statrs::distribution::DiscreteUniform;

use crate::distributions::{
    align_inputs, coerce_f64, param_keyed, params_are_constant, value_keyed_scalar,
    value_keyed_ternary, PairDomain,
};
use crate::rng::{
    sample_by_index, sample_per_row_ternary, samples_by_index, samples_i64_output,
    samples_per_row_ternary, SampleKwargs, SampleScalarKwargs, SamplesKwargs, SamplesScalarKwargs,
};

/// `min == max` is the legitimate one-point mass. Every closed form divides by the count
/// `max - min + 1`, so it must fit `i64`; the width is computed in `i128` so the check cannot wrap.
/// The bounds have no rule of their own: an integer column is one by dtype, through [`coerce_bound`].
const BOUNDS: PairDomain<i64> = PairDomain {
    names: ("min", "max"),
    rule: "greater than or equal to min, with a support width max - min + 1 that fits in i64",
    accepts: |min, max| min <= max && i64::try_from(i128::from(max) - i128::from(min) + 1).is_ok(),
};

/// `statrs` itself only rejects `max < min`, so behind [`BOUNDS`] its error is unreachable.
fn build_dist(min: i64, max: i64) -> PolarsResult<DiscreteUniform> {
    BOUNDS.check(min, max)?;
    DiscreteUniform::new(min, max).map_err(|e| polars_err!(ComputeError: "{e}"))
}

fn check_params(min: &Int64Chunked, max: &Int64Chunked) -> PolarsResult<()> {
    BOUNDS.check_columns(min, max)
}

/// A bound column as `Int64`: the signed counterpart of binomial's `coerce_n`. A float dtype is
/// refused rather than cast (a cast would truncate `2.7` to `2`); the strict cast reports values
/// outside the range. A `Null`-dtype column widens to all-null so null-in-null-out holds; the dtype
/// gate stays column-level, so a float column is refused even when every value in it is null.
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

/// Constant parameters, deserialised once per call; every `_scalar` twin builds once through here.
#[derive(serde::Deserialize)]
struct DiscreteUniformParams {
    min: i64,
    max: i64,
}

impl DiscreteUniformParams {
    fn build(&self) -> PolarsResult<DiscreteUniform> {
        build_dist(self.min, self.max)
    }

    fn point_keyed<Body>(&self, value: &Series, body: Body) -> PolarsResult<Series>
    where
        Body: Fn(&Support, Point) -> Option<f64>,
    {
        point_keyed_scalar(value, self.min, self.max, body)
    }

    fn value_keyed<Body>(&self, value: &Series, body: Body) -> PolarsResult<Series>
    where
        Body: Fn(&Support, f64) -> Option<f64>,
    {
        value_keyed_scalar(value, &build_support(self.min, self.max)?, body)
    }
}

/// The validated support with its count precomputed. `n` is `(max - min + 1) as f64`, computed in
/// `i128` exactly as [`discreteuniform_range`] returns it, so the Rust bodies and the Python moments
/// read the same rounded count.
struct Support {
    min: i64,
    max: i64,
    n: f64,
}

fn build_support(min: i64, max: i64) -> PolarsResult<Support> {
    build_dist(min, max)?;
    let n = (max as i128 - min as i128 + 1) as f64;
    Ok(Support { min, max, n })
}

/// An evaluation point in its own arithmetic: `f64` for float columns, exact `i128` for integer
/// columns, so an `int` spelling stays exact on supports a `Float64` cannot address and a `UInt64`
/// point above `i64::MAX` compares exactly instead of wrapping.
#[derive(Clone, Copy)]
enum Point {
    Float(f64),
    Int(i128),
}

/// Where a point sits relative to the support, with the two counts the closed forms read. The float
/// side compares against `Float64`-cast bounds, as polars promotes a mixed `Float64`/`Int64`
/// operation; the integer side stays exact. `count` and `tail` are only read where the branches keep
/// them in `[1, N - 1]`.
struct Placement {
    below: bool,
    at_or_above_max: bool,
    on_support: bool,
    /// `floor(value) - min + 1`, the support points at or below the point.
    count: f64,
    /// `max - floor(value)`, the support points strictly above the point.
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

fn pmf_point(s: &Support, point: Point) -> Option<f64> {
    let pos = place(s, point);
    Some(if pos.on_support { 1.0 / s.n } else { 0.0 })
}

/// `-ln(N)` straight off the count, not the log of the rounded quotient `1 / N`.
fn ln_pmf_point(s: &Support, point: Point) -> Option<f64> {
    let pos = place(s, point);
    Some(if pos.on_support {
        -s.n.ln()
    } else {
        f64::NEG_INFINITY
    })
}

/// `count / N` inside the support, `0` below `min`, `1` from `max` up. The explicit endpoint makes
/// `cdf(max) == 1` an answer rather than the limit of a clamp: `N * (1 / N)` is not `1.0` for 483
/// of the first 4000 support counts.
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

/// `tail / N`, a direct count of the points above the value, not `1 - cdf`.
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
/// log's relative error one point below the top grows with `N` (`2.8e-08` at `N = 1e9`).
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

/// `min + ceil(q * N) - 1`, with scipy's correction step (keep the point below when its cdf already
/// reaches `q`, so an ulp of noise in `q * N` cannot skip a support point) and clamped; null outside
/// `[0, 1]`. Both inverses lose support points above `2**53`.
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

/// The smallest support point whose survival mass is at most `q`, from `max - floor(q * N)`, solved
/// on `q` itself: the survival steps sit at multiples of `1 / N`, exactly where `1 - q` has spent its
/// precision. A rounded `q * N` can floor one point off in either direction, so both neighbours are
/// probed; null outside `[0, 1]`.
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

/// The evaluation-point column in the arithmetic its dtype earns: integer dtypes stay exact, with
/// `UInt64` on its own accessor so a value above `i64::MAX` reaches [`Point::Int`] via `i128`.
enum Points {
    Float(Float64Chunked),
    Int(Int64Chunked),
    Wide(UInt64Chunked),
}

fn coerce_points(value: &Series) -> PolarsResult<Points> {
    let dtype = value.dtype();
    if dtype == &DataType::UInt64 {
        return Ok(Points::Wide(value.u64()?.clone()));
    }
    if dtype.is_integer() {
        return Ok(Points::Int(value.cast(&DataType::Int64)?.i64()?.clone()));
    }
    Ok(Points::Float(coerce_f64(value)?))
}

/// Constant-bounds path of [`point_keyed`]: size the support once, then map `body` over the point
/// column in its own arithmetic. A `NaN` float point is `NaN` before `body` runs.
fn point_keyed_scalar<Body>(value: &Series, min: i64, max: i64, body: Body) -> PolarsResult<Series>
where
    Body: Fn(&Support, Point) -> Option<f64>,
{
    let support = build_support(min, max)?;
    let out: Float64Chunked = match coerce_points(value)? {
        Points::Float(v) => unary_elementwise(&v, |opt| {
            opt.and_then(|v| {
                if v.is_nan() {
                    Some(f64::NAN)
                } else {
                    body(&support, Point::Float(v))
                }
            })
        }),
        Points::Int(v) => unary_elementwise(&v, |opt| {
            opt.and_then(|v| body(&support, Point::Int(v.into())))
        }),
        Points::Wide(v) => unary_elementwise(&v, |opt| {
            opt.and_then(|v| body(&support, Point::Int(v.into())))
        }),
    };
    Ok(out.into_series())
}

/// `value_keyed_ternary` with the point kept in its own arithmetic, for the six methods with an
/// integer-exact path; same null, `NaN` and constant-parameter contracts.
fn point_keyed<Body>(inputs: &[Series], body: Body) -> PolarsResult<Series>
where
    Body: Fn(&Support, Point) -> Option<f64>,
{
    if params_are_constant(inputs) {
        let (min, max) = (coerce_bound(&inputs[1])?, coerce_bound(&inputs[2])?);
        check_params(&min, &max)?;
        if let Some((min, max)) = min.get(0).zip(max.get(0)) {
            return point_keyed_scalar(&inputs[0], min, max, body);
        }
    }

    let inputs = align_inputs(inputs)?;
    let min = coerce_bound(&inputs[1])?;
    let max = coerce_bound(&inputs[2])?;
    check_params(&min, &max)?;

    let out: Float64Chunked = match coerce_points(&inputs[0])? {
        Points::Float(v) => try_ternary_elementwise(&v, &min, &max, |v, lo, hi| {
            per_row(v.map(Point::Float), lo, hi, &body)
        })?,
        Points::Int(v) => try_ternary_elementwise(&v, &min, &max, |v, lo, hi| {
            per_row(v.map(|v| Point::Int(v.into())), lo, hi, &body)
        })?,
        Points::Wide(v) => try_ternary_elementwise(&v, &min, &max, |v, lo, hi| {
            per_row(v.map(|v| Point::Int(v.into())), lo, hi, &body)
        })?,
    };
    Ok(out.into_series())
}

#[inline]
fn per_row<Body>(
    point: Option<Point>,
    min: Option<i64>,
    max: Option<i64>,
    body: &Body,
) -> PolarsResult<Option<f64>>
where
    Body: Fn(&Support, Point) -> Option<f64>,
{
    let (Some(min), Some(max), Some(point)) = (min, max, point) else {
        return Ok(None);
    };
    if let Point::Float(v) = point {
        if v.is_nan() {
            return Ok(Some(f64::NAN));
        }
    }
    Ok(body(&build_support(min, max)?, point))
}

#[polars_expr(output_type=Float64)]
fn discreteuniform_pmf(inputs: &[Series]) -> PolarsResult<Series> {
    point_keyed(inputs, pmf_point)
}

#[polars_expr(output_type=Float64)]
fn discreteuniform_ln_pmf(inputs: &[Series]) -> PolarsResult<Series> {
    point_keyed(inputs, ln_pmf_point)
}

#[polars_expr(output_type=Float64)]
fn discreteuniform_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    point_keyed(inputs, cdf_point)
}

#[polars_expr(output_type=Float64)]
fn discreteuniform_sf(inputs: &[Series]) -> PolarsResult<Series> {
    point_keyed(inputs, sf_point)
}

#[polars_expr(output_type=Float64)]
fn discreteuniform_ln_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    point_keyed(inputs, ln_cdf_point)
}

#[polars_expr(output_type=Float64)]
fn discreteuniform_ln_sf(inputs: &[Series]) -> PolarsResult<Series> {
    point_keyed(inputs, ln_sf_point)
}

#[polars_expr(output_type=Float64)]
fn discreteuniform_ppf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_ternary(
        inputs,
        coerce_bound,
        coerce_bound,
        check_params,
        build_support,
        ppf_value,
    )
}

#[polars_expr(output_type=Float64)]
fn discreteuniform_isf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_ternary(
        inputs,
        coerce_bound,
        coerce_bound,
        check_params,
        build_support,
        isf_value,
    )
}

#[polars_expr(output_type=Float64)]
fn discreteuniform_pmf_scalar(
    inputs: &[Series],
    kwargs: DiscreteUniformParams,
) -> PolarsResult<Series> {
    kwargs.point_keyed(&inputs[0], pmf_point)
}

#[polars_expr(output_type=Float64)]
fn discreteuniform_ln_pmf_scalar(
    inputs: &[Series],
    kwargs: DiscreteUniformParams,
) -> PolarsResult<Series> {
    kwargs.point_keyed(&inputs[0], ln_pmf_point)
}

#[polars_expr(output_type=Float64)]
fn discreteuniform_cdf_scalar(
    inputs: &[Series],
    kwargs: DiscreteUniformParams,
) -> PolarsResult<Series> {
    kwargs.point_keyed(&inputs[0], cdf_point)
}

#[polars_expr(output_type=Float64)]
fn discreteuniform_sf_scalar(
    inputs: &[Series],
    kwargs: DiscreteUniformParams,
) -> PolarsResult<Series> {
    kwargs.point_keyed(&inputs[0], sf_point)
}

#[polars_expr(output_type=Float64)]
fn discreteuniform_ln_cdf_scalar(
    inputs: &[Series],
    kwargs: DiscreteUniformParams,
) -> PolarsResult<Series> {
    kwargs.point_keyed(&inputs[0], ln_cdf_point)
}

#[polars_expr(output_type=Float64)]
fn discreteuniform_ln_sf_scalar(
    inputs: &[Series],
    kwargs: DiscreteUniformParams,
) -> PolarsResult<Series> {
    kwargs.point_keyed(&inputs[0], ln_sf_point)
}

#[polars_expr(output_type=Float64)]
fn discreteuniform_ppf_scalar(
    inputs: &[Series],
    kwargs: DiscreteUniformParams,
) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], ppf_value)
}

#[polars_expr(output_type=Float64)]
fn discreteuniform_isf_scalar(
    inputs: &[Series],
    kwargs: DiscreteUniformParams,
) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], isf_value)
}

/// The support count `N = max - min + 1` as `Float64`, which the Python moments divide by.
///
/// Callers that need one count per row rather than a broadcast scalar append a third input whose
/// height the bounds broadcast up to; the body never reads it. `_ppf` and `_isf` need that: their
/// step-boundary correction divides by the count and compares against the quantile with no slack,
/// and a length-1 count makes the in-memory engine evaluate the division as a reciprocal multiply,
/// one ulp from the true division and exactly at the steps where that flips a support point.
#[polars_expr(output_type=Float64)]
fn discreteuniform_range(inputs: &[Series]) -> PolarsResult<Series> {
    param_keyed(
        inputs,
        coerce_bound,
        coerce_bound,
        check_params,
        |lo, hi| Ok((i128::from(hi) - i128::from(lo) + 1) as f64),
    )
}

#[inline]
fn draw(dist: &DiscreteUniform, rng: &mut impl rand::Rng) -> i64 {
    <DiscreteUniform as Distribution<i64>>::sample(dist, rng)
}

#[polars_expr(output_type=Int64)]
fn discreteuniform_sample(inputs: &[Series], kwargs: SampleKwargs) -> PolarsResult<Series> {
    sample_per_row_ternary(
        inputs,
        kwargs,
        coerce_bound,
        coerce_bound,
        check_params,
        build_dist,
        draw,
    )
}

#[polars_expr(output_type=Int64)]
fn discreteuniform_sample_scalar(
    inputs: &[Series],
    kwargs: SampleScalarKwargs<DiscreteUniformParams>,
) -> PolarsResult<Series> {
    let dist = kwargs.params.build()?;
    sample_by_index(&inputs[0], kwargs.seed, |rng| draw(&dist, rng))
}

#[polars_expr(output_type_func_with_kwargs=samples_i64_output)]
fn discreteuniform_samples_scalar(
    inputs: &[Series],
    kwargs: SamplesScalarKwargs<DiscreteUniformParams>,
) -> PolarsResult<Series> {
    let dist = kwargs.params.build()?;
    samples_by_index(&inputs[0], kwargs.seed, kwargs.size, |rng| draw(&dist, rng))
}

#[polars_expr(output_type_func_with_kwargs=samples_i64_output)]
fn discreteuniform_samples(inputs: &[Series], kwargs: SamplesKwargs) -> PolarsResult<Series> {
    samples_per_row_ternary(
        inputs,
        kwargs,
        coerce_bound,
        coerce_bound,
        check_params,
        build_dist,
        draw,
    )
}
