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
pub mod cauchy;
pub mod discrete_uniform;
pub mod exponential;
pub mod gamma;
pub mod geometric;
pub mod lognormal;
pub mod normal;
pub mod pareto;
pub mod poisson;
pub mod uniform;
pub mod weibull;

use std::borrow::Cow;

use polars::prelude::arity::{try_binary_elementwise, try_ternary_elementwise, unary_elementwise};
use polars::prelude::*;
use polars_arrow::array::Array as _;
use polars_core::utils::align_chunks_binary;

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
#[derive(Clone, Copy)]
pub(crate) enum Rule {
    Finite,
    Positive,
    Probability,
    PositiveProbability,
}

impl Rule {
    #[inline(always)]
    fn accepts(self, x: f64) -> bool {
        match self {
            Self::Finite => x.is_finite(),
            Self::Positive => x.is_finite() && x > 0.0,
            Self::Probability => x.is_finite() && (0.0..=1.0).contains(&x),
            Self::PositiveProbability => x.is_finite() && x > 0.0 && x <= 1.0,
        }
    }

    /// Read as "`name` must be {describe}".
    fn describe(self) -> &'static str {
        match self {
            Self::Finite => "finite",
            Self::Positive => "finite and strictly positive",
            Self::Probability => "in [0, 1]",
            Self::PositiveProbability => "in (0, 1]",
        }
    }
}

pub(crate) struct ParamDomain {
    pub(crate) name: &'static str,
    pub(crate) rule: Rule,
}

impl ParamDomain {
    pub(crate) const fn finite(name: &'static str) -> Self {
        Self {
            name,
            rule: Rule::Finite,
        }
    }

    pub(crate) const fn positive(name: &'static str) -> Self {
        Self {
            name,
            rule: Rule::Positive,
        }
    }

    pub(crate) const fn probability(name: &'static str) -> Self {
        Self {
            name,
            rule: Rule::Probability,
        }
    }

    pub(crate) fn check(&self, value: f64) -> PolarsResult<()> {
        polars_ensure!(
            self.rule.accepts(value),
            ComputeError: "{} must be {}, got {}", self.name, self.rule.describe(), value
        );
        Ok(())
    }

    /// Nulls are skipped: a missing parameter is a missing answer, not an invalid one.
    pub(crate) fn check_column(&self, column: &Float64Chunked) -> PolarsResult<()> {
        // One instantiation per rule, so `accepts` inlines into the fold.
        let passes = match self.rule {
            Rule::Finite => all_accept(column, |x| Rule::Finite.accepts(x)),
            Rule::Positive => all_accept(column, |x| Rule::Positive.accepts(x)),
            Rule::Probability => all_accept(column, |x| Rule::Probability.accepts(x)),
            Rule::PositiveProbability => {
                all_accept(column, |x| Rule::PositiveProbability.accepts(x))
            },
        };
        if passes {
            return Ok(());
        }
        column
            .iter()
            .flatten()
            .try_for_each(|value| self.check(value))
    }
}

/// Whether every non-null value of `column` passes. A null slot's backing value is unspecified (in
/// practice `0.0`), so only a chunk without nulls reads its raw values.
fn all_accept(column: &Float64Chunked, accepts: impl Fn(f64) -> bool) -> bool {
    column.downcast_iter().all(|chunk| {
        if chunk.null_count() == 0 {
            // `&` rather than `all`: without an early exit the loop vectorises.
            chunk.values().iter().fold(true, |ok, &v| ok & accepts(v))
        } else {
            chunk.iter().flatten().all(|&v| accepts(v))
        }
    })
}

/// What two parameters must satisfy together, checked only where both are present. Implemented by
/// one unit struct per joint rule, so the column pass is monomorphised over `accepts`.
pub(crate) trait PairDomain {
    type N: NumericNative;
    const NAMES: (&'static str, &'static str);
    /// Read as "`NAMES.1` must be {RULE}".
    const RULE: &'static str;

    fn accepts(a: Self::N, b: Self::N) -> bool;

    fn check(a: Self::N, b: Self::N) -> PolarsResult<()> {
        let (a_name, b_name) = Self::NAMES;
        polars_ensure!(
            Self::accepts(a, b),
            ComputeError: "{} must be {}, got {}={}, {}={}", b_name, Self::RULE, a_name, a, b_name, b
        );
        Ok(())
    }

    /// `a` and `b` must have one length; their chunk layouts may differ.
    fn check_columns<T>(a: &ChunkedArray<T>, b: &ChunkedArray<T>) -> PolarsResult<()>
    where
        T: PolarsNumericType<Native = Self::N>,
    {
        let (a_aligned, b_aligned) = align_chunks_binary(a, b);
        let passes = a_aligned
            .downcast_iter()
            .zip(b_aligned.downcast_iter())
            .all(|(a, b)| {
                if a.null_count() == 0 && b.null_count() == 0 {
                    let pairs = a.values().iter().zip(b.values().iter());
                    pairs.fold(true, |ok, (&a, &b)| ok & Self::accepts(a, b))
                } else {
                    a.iter().zip(b.iter()).all(|pair| match pair {
                        (Some(&a), Some(&b)) => Self::accepts(a, b),
                        _ => true,
                    })
                }
            });
        if passes {
            return Ok(());
        }
        a.iter().zip(b.iter()).try_for_each(|pair| match pair {
            (Some(a), Some(b)) => Self::check(a, b),
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

/// `2^-53`, below which `t^2 / 2` is under half an ulp of `t` and `exp(t) - 1` is `t` itself. The
/// identity in [`expm1`] agrees down to `t ~ 4.5e-308` and then halves `t` into the subnormals,
/// which costs its last bit and flushes the smallest subnormal to `0`, so `t` is returned outright.
const EXPM1_IS_IDENTITY_BELOW: f64 = 1.1102230246251565e-16;

/// `exp(t) - 1` as `2 exp(t / 2) sinh(t / 2)`, which has no subtraction to cancel. Not
/// `f64::exp_m1`, which differs by one ulp on roughly a fifth of the left tail; `_base.expm1` spells
/// the same identity for the moments assembled in Polars. `sinh(t / 2)` overflows above `|t| ~ 1420`;
/// every caller crosses to a direct form long before that.
#[inline]
pub(crate) fn expm1(t: f64) -> f64 {
    if t.abs() < EXPM1_IS_IDENTITY_BELOW {
        return t;
    }
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

/// A closed form on a support with a lower bound: a constant below `floor`, the parameter's arm on
/// `[floor, inf)`.
pub(crate) struct Sides<Arm> {
    pub(crate) floor: f64,
    pub(crate) below_support: f64,
    pub(crate) on_support: Arm,
}

impl<Arm: Fn(f64) -> f64> Sides<Arm> {
    /// A bare `<`, so `-0.0` sits where `0.0` does; a `NaN` point never reaches here.
    pub(crate) fn at(&self, value: f64) -> Option<f64> {
        Some(if value < self.floor {
            self.below_support
        } else {
            (self.on_support)(value)
        })
    }

    /// The same form read at `transform(x)` on `[floor, inf)`, its below-support constant kept: one
    /// distribution as another on a transformed variate.
    pub(crate) fn read_at(
        self,
        floor: f64,
        transform: impl Fn(f64) -> f64,
    ) -> Sides<impl Fn(f64) -> f64> {
        let on_support = self.on_support;
        Sides {
            floor,
            below_support: self.below_support,
            on_support: move |x: f64| on_support(transform(x)),
        }
    }
}

/// `c_0 + c_1 x + c_2 x^2 + ...` by Horner's rule.
pub(crate) fn polynomial(coefficients: &[f64], x: f64) -> f64 {
    coefficients
        .iter()
        .rev()
        .fold(0.0, |acc, coefficient| acc * x + coefficient)
}

/// `scale * exp(t)` as `(scale * exp(t / 2)) * exp(t / 2)`: `exp(t)` alone overflows past
/// `t ~ 709.8`, where the answer under a small `scale` is still ordinary; the half exponent holds to
/// `t ~ 1419`, past which `scale * exp(t)` has left `f64` for every normal `scale`.
pub(crate) fn scale_exp(scale: f64, t: f64) -> f64 {
    let half = (t / 2.0).exp();
    (scale * half) * half
}

/// The `select` every inverse shares: the arm on `[0, 1]`, null outside it. `-0.0` is inside.
#[inline]
pub(crate) fn on_unit_interval<Arm: Fn(f64) -> f64>(arm: &Arm, quantile: f64) -> Option<f64> {
    (0.0..=1.0).contains(&quantile).then(|| arm(quantile))
}
