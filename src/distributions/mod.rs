//! Shared drivers behind every `#[polars_expr]` in the distribution files.
//!
//! A plugin is a one-line shell over one driver here or in `crate::rng`; the driver owns input
//! alignment, the dtype gate, the parameter domain pass, null propagation and the row loop. Driver
//! names count plugin inputs: `value_keyed_binary` takes `(value, param)`, `value_keyed_ternary`
//! takes `(value, a, b)`.
//!
//! Contracts every driver keeps:
//!
//! * A null in any input nulls the row before the body runs.
//! * A `NaN` evaluation point is `NaN`, for every method including `ppf`, and never reaches a body:
//!   the regularized incomplete beta behind `Beta::cdf` panics on `NaN`, and `NaN.floor() as u64` is
//!   `0`, which would make Binomial answer a confident `P(X <= 0)`.
//! * An invalid parameter raises `ComputeError` from a pass over the whole column, before any row is
//!   built, so it raises beside a null or `NaN` point too.
//! * When every parameter is length 1 ([`params_are_constant`]) it is checked and built once and the
//!   value column takes [`value_keyed_scalar`], the same path the kwargs `_scalar` twins take. A null
//!   constant falls through to the row loop, which nulls every row and still runs the value dtype gate.

pub mod bernoulli;
pub mod beta;
pub mod binomial;
pub mod discrete_uniform;
pub mod exponential;
pub mod geometric;
pub mod lognormal;
pub mod normal;
pub mod uniform;

use std::borrow::Cow;

use polars::prelude::arity::{try_binary_elementwise, try_ternary_elementwise, unary_elementwise};
use polars::prelude::*;

/// Broadcast every length-1 input up to the call's row count, and reject lengths that cannot align.
///
/// Polars broadcasts nothing into a plugin and the `*_elementwise` kernels truncate to their shortest
/// input, so every multi-input plugin calls this before any cast. Length-1 inputs never set the row
/// count: an all-length-1 call stays length 1, and a 0-row frame beside a `pl.lit` stays empty.
pub(crate) fn align_inputs(inputs: &[Series]) -> PolarsResult<Cow<'_, [Series]>> {
    let mut anchor: Option<&Series> = None;
    let mut needs_broadcast = false;

    for input in inputs {
        if input.len() == 1 {
            needs_broadcast = true;
        } else if let Some(anchor) = anchor {
            polars_ensure!(
                input.len() == anchor.len(),
                ShapeMismatch:
                "inputs have incompatible lengths: '{}' has length {} but '{}' has length {}. \
                 Every input must share one length, or be length 1 to broadcast",
                anchor.name(), anchor.len(), input.name(), input.len()
            );
        } else {
            anchor = Some(input);
        }
    }

    let row_count = match anchor {
        Some(anchor) if needs_broadcast => anchor.len(),
        _ => return Ok(Cow::Borrowed(inputs)),
    };

    Ok(Cow::Owned(
        inputs
            .iter()
            .map(|input| {
                if input.len() == 1 {
                    input.new_from_index(0, row_count)
                } else {
                    input.clone()
                }
            })
            .collect(),
    ))
}

/// Every parameter (every input after the evaluation point) is length 1: a `pl.lit`, or an
/// aggregate whose length polars only knows once it has run.
pub(crate) fn params_are_constant(inputs: &[Series]) -> bool {
    inputs[1..].iter().all(|param| param.len() == 1)
}

/// The dtype gate every evaluation point and float parameter passes: `Int*`, `UInt*`, `Float*` and
/// `Decimal` cast to `Float64`, a `Null`-typed column to all-null, anything else raises. Polars' own
/// cast would read `Boolean` as `0` / `1`, parse `String` and take a temporal dtype's integer
/// representation.
pub(crate) fn coerce_f64(input: &Series) -> PolarsResult<Float64Chunked> {
    let dtype = input.dtype();
    polars_ensure!(
        dtype.is_numeric() || dtype.is_null(),
        ComputeError: "'{}' must be a numeric column (Int*, UInt*, Float*, Decimal), got {}",
        input.name(), dtype
    );
    Ok(input.cast(&DataType::Float64)?.f64()?.clone())
}

/// What one float parameter must satisfy on its own. Every rule spells finiteness because the
/// `statrs` constructors do not (`Normal::new(0.0, f64::INFINITY)` is `Ok`).
pub(crate) struct ParamDomain {
    pub(crate) name: &'static str,
    /// Read as "`name` must be {rule}".
    pub(crate) rule: &'static str,
    pub(crate) accepts: fn(f64) -> bool,
}

impl ParamDomain {
    pub(crate) const fn finite(name: &'static str) -> Self {
        Self {
            name,
            rule: "finite",
            accepts: f64::is_finite,
        }
    }

    pub(crate) const fn positive(name: &'static str) -> Self {
        Self {
            name,
            rule: "finite and strictly positive",
            accepts: |x| x.is_finite() && x > 0.0,
        }
    }

    pub(crate) const fn probability(name: &'static str) -> Self {
        Self {
            name,
            rule: "in [0, 1]",
            accepts: |p| p.is_finite() && (0.0..=1.0).contains(&p),
        }
    }

    pub(crate) fn check(&self, value: f64) -> PolarsResult<()> {
        polars_ensure!(
            (self.accepts)(value),
            ComputeError: "{} must be {}, got {}", self.name, self.rule, value
        );
        Ok(())
    }

    /// Nulls are skipped: a missing parameter is a missing answer, not an invalid one.
    pub(crate) fn check_column(&self, column: &Float64Chunked) -> PolarsResult<()> {
        column
            .iter()
            .flatten()
            .try_for_each(|value| self.check(value))
    }
}

/// What two parameters must satisfy together, checked only where both are present.
pub(crate) struct PairDomain<N> {
    pub(crate) names: (&'static str, &'static str),
    /// Read as "`names.1` must be {rule}".
    pub(crate) rule: &'static str,
    pub(crate) accepts: fn(N, N) -> bool,
}

impl<N: Copy + std::fmt::Display> PairDomain<N> {
    pub(crate) fn check(&self, a: N, b: N) -> PolarsResult<()> {
        let (a_name, b_name) = self.names;
        polars_ensure!(
            (self.accepts)(a, b),
            ComputeError: "{} must be {}, got {}={}, {}={}", b_name, self.rule, a_name, a, b_name, b
        );
        Ok(())
    }

    pub(crate) fn check_columns<T>(
        &self,
        a: &ChunkedArray<T>,
        b: &ChunkedArray<T>,
    ) -> PolarsResult<()>
    where
        T: PolarsNumericType<Native = N>,
    {
        a.iter().zip(b.iter()).try_for_each(|pair| match pair {
            (Some(a), Some(b)) => self.check(a, b),
            _ => Ok(()),
        })
    }
}

/// The one-parameter validator plugin: the parameter unchanged after `domain`'s column pass, nulls
/// included. The Python moments derive from it, so an invalid parameter raises there too.
pub(crate) fn validated_param(inputs: &[Series], domain: &ParamDomain) -> PolarsResult<Series> {
    let param = coerce_f64(&inputs[0])?;
    domain.check_column(&param)?;
    Ok(param.into_series())
}

/// Two parameters and no evaluation point: `body(a, b)` per row, null where either is null, after
/// `check_params`'s pass over both columns. Backs the two-parameter validators and the moments with
/// no closed form (`entropy`). The output is named after the first input.
pub(crate) fn param_keyed<A, B, CoerceA, CoerceB, Check, Body>(
    inputs: &[Series],
    coerce_a: CoerceA,
    coerce_b: CoerceB,
    check_params: Check,
    body: Body,
) -> PolarsResult<Series>
where
    A: PolarsNumericType,
    B: PolarsNumericType,
    CoerceA: Fn(&Series) -> PolarsResult<ChunkedArray<A>>,
    CoerceB: Fn(&Series) -> PolarsResult<ChunkedArray<B>>,
    Check: Fn(&ChunkedArray<A>, &ChunkedArray<B>) -> PolarsResult<()>,
    Body: Fn(A::Native, B::Native) -> PolarsResult<f64>,
{
    let inputs = align_inputs(inputs)?;
    let a = coerce_a(&inputs[0])?;
    let b = coerce_b(&inputs[1])?;
    check_params(&a, &b)?;

    let out: Float64Chunked = try_binary_elementwise(&a, &b, |a, b| match (a, b) {
        (Some(a), Some(b)) => body(a, b).map(Some),
        _ => Ok(None),
    })?;
    Ok(out.into_series())
}

/// The two-parameter validator plugin: the second parameter where both are present, null elsewhere.
pub(crate) fn validated_pair<A, CoerceA, Check>(
    inputs: &[Series],
    coerce_a: CoerceA,
    check_params: Check,
) -> PolarsResult<Series>
where
    A: PolarsNumericType,
    CoerceA: Fn(&Series) -> PolarsResult<ChunkedArray<A>>,
    Check: Fn(&ChunkedArray<A>, &Float64Chunked) -> PolarsResult<()>,
{
    param_keyed(inputs, coerce_a, coerce_f64, check_params, |_, b| Ok(b))
}

/// The constant-parameter value-keyed path: `body(state, value)` over the evaluation column, with
/// `state` built once by the caller. Both the kwargs `_scalar` twins and the drivers' constant branch
/// call this, so the two spellings of a constant parameterisation run one instantiation.
pub(crate) fn value_keyed_scalar<State, Body>(
    value: &Series,
    state: &State,
    body: Body,
) -> PolarsResult<Series>
where
    Body: Fn(&State, f64) -> Option<f64>,
{
    let value = coerce_f64(value)?;
    let out: Float64Chunked = unary_elementwise(&value, |point| {
        point.and_then(|v| {
            if v.is_nan() {
                Some(f64::NAN)
            } else {
                body(state, v)
            }
        })
    });
    Ok(out.into_series())
}

/// Value-keyed driver over `(value, param)`: `build` turns the parameter into the per-row state
/// (`statrs` distribution or closed-form branch table) and `body` evaluates it at the point.
///
/// `build` and `body` stay generic `Fn`s so they monomorphise into the row loop; a `&dyn Fn` or a
/// `fn` pointer would cost an indirect call per row.
pub(crate) fn value_keyed_binary<State, Build, Body>(
    inputs: &[Series],
    domain: &ParamDomain,
    build: Build,
    body: Body,
) -> PolarsResult<Series>
where
    Build: Fn(f64) -> PolarsResult<State>,
    Body: Fn(&State, f64) -> Option<f64>,
{
    if params_are_constant(inputs) {
        let param = coerce_f64(&inputs[1])?;
        domain.check_column(&param)?;
        if let Some(param) = param.get(0) {
            return value_keyed_scalar(&inputs[0], &build(param)?, body);
        }
    }

    let inputs = align_inputs(inputs)?;
    let value = coerce_f64(&inputs[0])?;
    let param = coerce_f64(&inputs[1])?;
    domain.check_column(&param)?;

    let out: Float64Chunked = try_binary_elementwise(
        &value,
        &param,
        |value, param| -> PolarsResult<Option<f64>> {
            let (Some(value), Some(param)) = (value, param) else {
                return Ok(None);
            };
            if value.is_nan() {
                return Ok(Some(f64::NAN));
            }
            Ok(body(&build(param)?, value))
        },
    )?;
    Ok(out.into_series())
}

/// Value-keyed driver over `(value, a, b)`. Each parameter has its own coercer, so a mixed
/// `(UInt64, Float64)` parameterisation (Binomial's `n` beside its `p`) fits; `check_params` is the
/// caller's pass over both columns, each alone and the pair together.
pub(crate) fn value_keyed_ternary<A, B, State, CoerceA, CoerceB, Check, Build, Body>(
    inputs: &[Series],
    coerce_a: CoerceA,
    coerce_b: CoerceB,
    check_params: Check,
    build: Build,
    body: Body,
) -> PolarsResult<Series>
where
    A: PolarsNumericType,
    B: PolarsNumericType,
    CoerceA: Fn(&Series) -> PolarsResult<ChunkedArray<A>>,
    CoerceB: Fn(&Series) -> PolarsResult<ChunkedArray<B>>,
    Check: Fn(&ChunkedArray<A>, &ChunkedArray<B>) -> PolarsResult<()>,
    Build: Fn(A::Native, B::Native) -> PolarsResult<State>,
    Body: Fn(&State, f64) -> Option<f64>,
{
    if params_are_constant(inputs) {
        let (a, b) = (coerce_a(&inputs[1])?, coerce_b(&inputs[2])?);
        check_params(&a, &b)?;
        if let Some((a, b)) = a.get(0).zip(b.get(0)) {
            return value_keyed_scalar(&inputs[0], &build(a, b)?, body);
        }
    }

    let inputs = align_inputs(inputs)?;
    let value = coerce_f64(&inputs[0])?;
    let a = coerce_a(&inputs[1])?;
    let b = coerce_b(&inputs[2])?;
    check_params(&a, &b)?;

    let out: Float64Chunked =
        try_ternary_elementwise(&value, &a, &b, |value, a, b| -> PolarsResult<Option<f64>> {
            let (Some(value), Some(a), Some(b)) = (value, a, b) else {
                return Ok(None);
            };
            if value.is_nan() {
                return Ok(Some(f64::NAN));
            }
            Ok(body(&build(a, b)?, value))
        })?;
    Ok(out.into_series())
}

/// [`value_keyed_binary`] for a closed form: `derive` turns the parameter into the method's branch
/// table and cannot fail, `select` picks the branch the point lands on. `derive` runs once per call
/// on a constant parameter and once per row on a column.
pub(crate) fn value_keyed_derived_binary<Branches, Derive, Select>(
    inputs: &[Series],
    domain: &ParamDomain,
    derive: Derive,
    select: Select,
) -> PolarsResult<Series>
where
    Derive: Fn(f64) -> Branches,
    Select: Fn(&Branches, f64) -> Option<f64>,
{
    value_keyed_binary(inputs, domain, |param| Ok(derive(param)), select)
}

/// [`value_keyed_ternary`] for a closed form over two float parameters.
pub(crate) fn value_keyed_derived_ternary<Branches, Check, Derive, Select>(
    inputs: &[Series],
    check_params: Check,
    derive: Derive,
    select: Select,
) -> PolarsResult<Series>
where
    Check: Fn(&Float64Chunked, &Float64Chunked) -> PolarsResult<()>,
    Derive: Fn(f64, f64) -> Branches,
    Select: Fn(&Branches, f64) -> Option<f64>,
{
    value_keyed_ternary(
        inputs,
        coerce_f64,
        coerce_f64,
        check_params,
        |a, b| Ok(derive(a, b)),
        select,
    )
}

/// `exp(t) - 1` as `2 exp(t / 2) sinh(t / 2)`, which has no subtraction to cancel. Not
/// `f64::exp_m1`, which differs by one ulp on roughly a fifth of the left tail; `_base.expm1` spells
/// the same identity for the moments assembled in Polars. `sinh(t / 2)` overflows above `|t| ~ 1420`;
/// every caller crosses to a direct form long before that.
#[inline]
pub(crate) fn expm1(t: f64) -> f64 {
    let half = t / 2.0;
    2.0 * half.exp() * half.sinh()
}

/// `ln|exp(t) - 1|` as `ln(2) + t / 2 + ln|sinh(t / 2)|`: [`expm1`] read term by term on the log
/// scale, so the rounding stays relative where `expm1(t).ln()` would round the answer away. The
/// absolute value carries `t < 0`, where this is the log of `1 - exp(t)`.
#[inline]
pub(crate) fn ln_abs_expm1(t: f64) -> f64 {
    let half = t / 2.0;
    std::f64::consts::LN_2 + half + half.sinh().abs().ln()
}

/// A closed form on a support whose floor is the integer `FLOOR`: a constant below it, the
/// parameter's arm on it.
pub(crate) struct Sides<Arm, const FLOOR: i8> {
    pub(crate) below_support: f64,
    pub(crate) on_support: Arm,
}

impl<Arm: Fn(f64) -> f64, const FLOOR: i8> Sides<Arm, FLOOR> {
    /// A bare `<`, so `-0.0` sits where `0.0` does; a `NaN` point never reaches here.
    pub(crate) fn at(&self, value: f64) -> Option<f64> {
        Some(if value < f64::from(FLOOR) {
            self.below_support
        } else {
            (self.on_support)(value)
        })
    }
}

/// The `select` every inverse shares: the arm on `[0, 1]`, null outside it. `-0.0` is inside.
#[inline]
pub(crate) fn on_unit_interval<Arm: Fn(f64) -> f64>(arm: &Arm, quantile: f64) -> Option<f64> {
    (0.0..=1.0).contains(&quantile).then(|| arm(quantile))
}
