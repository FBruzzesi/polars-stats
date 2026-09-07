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

use polars::prelude::arity::{
    binary_elementwise, ternary_elementwise, try_ternary_elementwise, unary_elementwise,
};
use polars::prelude::*;

/// Broadcast every length-1 input up to the call's row count, and reject lengths that cannot align.
///
/// The row count is the one input length that is not 1, so length-1 inputs never set it: an
/// all-length-1 call stays length 1, and a 0-row frame beside a `pl.lit` stays empty.
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

    // Borrow when there is nothing to expand: no anchor means every input is already length 1, and
    // `!needs_broadcast` means they already agree, which is every call whose parameters are columns.
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

/// What one float parameter must satisfy on its own, checked once per call: [`Self::check`] on a
/// constant, [`Self::check_column`] over the whole column before any row is built.
///
/// Every rule spells finiteness because the `statrs` constructors do not (`Normal::new(0.0,
/// f64::INFINITY)` is `Ok`); behind the pass, the constructor inside a row loop cannot fail.
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
    pub(crate) fn check_column(&self, ca: &Float64Chunked) -> PolarsResult<()> {
        ca.iter().flatten().try_for_each(|value| self.check(value))
    }
}

/// What two parameters must satisfy together (`Uniform`'s `max > min`, `DiscreteUniform`'s
/// `min <= max`), checked only where both are present.
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

/// Shared driver for the constant-parameter value-keyed fast paths.
///
/// The constant-parameter counterpart of each distribution's `value_keyed` helper: when every
/// distribution parameter is a Python scalar, the caller validates them and builds the
/// distribution **once**, and only the evaluation-point column crosses FFI. This maps `f` over
/// that single column, where `f` is the same per-method body the per-row path uses, so the two
/// paths cannot drift and output is bit-identical.
///
/// Null contract: `value` is the only nullable input this driver sees; a null `value` propagates,
/// and `f` returning `None` nulls the row on the method's own terms (`ppf` outside `[0, 1]`).
///
/// `NaN` contract: a `NaN` evaluation point short-circuits to `NaN` (scipy semantics) before `f`
/// runs, for every method including `ppf`. The short-circuit is central (here and in
/// [`value_keyed_per_row`]) rather than per body, because two bodies need it and none may be
/// forgotten: the regularized incomplete beta behind `Beta` `cdf`/`sf` panics on `NaN`, aborting
/// the whole query, and binomial's support mapping saturates (`NaN.floor() as u64` is `0`),
/// returning a confident `P(X <= 0)`. Pinned by `tests/distributions/plugin_nan_test.py`.
pub(crate) fn value_keyed_scalar<F>(value: &Series, f: F) -> PolarsResult<Series>
where
    F: Fn(f64) -> Option<f64>,
{
    let value = value.cast(&DataType::Float64)?;
    let value_ca = value.f64()?;
    let name = value_ca.name().clone();

    let ca: Float64Chunked = unary_elementwise(value_ca, |opt| {
        opt.and_then(|v| if v.is_nan() { Some(f64::NAN) } else { f(v) })
    });
    Ok(ca.with_name(name).into_series())
}

/// Constant-parameter twin of [`value_keyed_derived_per_row`], over [`value_keyed_scalar`].
///
/// Checks the parameter against `domain` and derives once per call, then maps `select` over the
/// evaluation-point column, so the parameter-only terms `derive` hoists (`Bernoulli`'s `1 - p`,
/// `Exponential`'s `ln(rate)`) are computed once instead of per row.
pub(crate) fn value_keyed_derived_scalar<Branches, Derive, Select>(
    value: &Series,
    param: f64,
    domain: &ParamDomain,
    derive: Derive,
    select: Select,
) -> PolarsResult<Series>
where
    Derive: Fn(f64) -> Branches,
    Select: Fn(&Branches, f64) -> Option<f64>,
{
    domain.check(param)?;
    let branches = derive(param);
    value_keyed_scalar(value, |v| select(&branches, v))
}

/// Shared driver for the column-parameter value-keyed per-row paths.
///
/// The column-parameter counterpart of [`value_keyed_scalar`]: at least one distribution parameter
/// is a column, so `build` constructs once per row instead of once per call. `f` is the same named
/// per-method body the fast path applies (`cdf_value`, `ppf_value`, ...), so the two paths cannot
/// drift and agree bit for bit.
///
/// The caller does the cast and the accessor (`.f64()` / `.u64()`), which fixes `A` and `B`, so a
/// mixed `(u64, f64)` parameterisation (Binomial's `UInt64` `n` beside its `Float64` `p`) fits, as in
/// [`ternary_param_rows`](crate::rng::ternary_param_rows). `S` needs no trait bound: it is whatever
/// `build` returns.
///
/// The caller has run the parameter columns through their [`ParamDomain`]s, so `build` cannot fail
/// here; it stays `PolarsResult` because the `statrs` constructors are.
///
/// Null contract: any null among `(value, p1, p2)` nulls the row without calling `build`, matching
/// the samplers. `NaN` contract: as in [`value_keyed_scalar`].
///
/// Keep `build` and `f` generic `Fn`s: they monomorphise into the row loop, where a `&dyn Fn` or a
/// `fn` pointer would cost an indirect call per row.
pub(crate) fn value_keyed_per_row<A, B, S, Build, F>(
    value: &Float64Chunked,
    p1: &ChunkedArray<A>,
    p2: &ChunkedArray<B>,
    name: PlSmallStr,
    build: Build,
    f: F,
) -> PolarsResult<Series>
where
    A: PolarsNumericType,
    B: PolarsNumericType,
    Build: Fn(A::Native, B::Native) -> PolarsResult<S>,
    F: Fn(&S, f64) -> Option<f64>,
{
    let ca: Float64Chunked = try_ternary_elementwise(
        value,
        p1,
        p2,
        |value_opt, p1_opt, p2_opt| -> PolarsResult<Option<f64>> {
            let (Some(value), Some(p1), Some(p2)) = (value_opt, p1_opt, p2_opt) else {
                return Ok(None);
            };
            if value.is_nan() {
                return Ok(Some(f64::NAN));
            }
            Ok(f(&build(p1, p2)?, value))
        },
    )?;

    Ok(ca.with_name(name).into_series())
}

/// Shared driver for the value-keyed methods of a **one-parameter** distribution that computes its
/// own closed form instead of building a `statrs` distribution.
///
/// `derive` turns the parameter into that method's branch table and `select` picks the branch the
/// evaluation point lands on. Each method derives its own type, so pairing one method's `derive`
/// with another's `select` does not compile. Both run per row here; [`value_keyed_derived_scalar`]
/// is the twin that runs `derive` once per call.
///
/// `domain`'s column pass runs once before any row is built, so nothing inside the row loop
/// validates.
///
/// Null contract: a null parameter nulls the row before the evaluation point is read; then a null
/// value nulls it. `select` returning `None` nulls it on the method's own terms (`ppf` outside
/// `[0, 1]`).
///
/// `NaN` contract: a `NaN` value short-circuits to `NaN`, as in [`value_keyed_scalar`].
///
/// Keep `derive` and `select` generic `Fn`s: they monomorphise into the row loop, where a `&dyn Fn`
/// or a `fn` pointer would cost an indirect call per row.
pub(crate) fn value_keyed_derived_per_row<Branches, Derive, Select>(
    inputs: &[Series],
    domain: &ParamDomain,
    derive: Derive,
    select: Select,
) -> PolarsResult<Series>
where
    Derive: Fn(f64) -> Branches,
    Select: Fn(&Branches, f64) -> Option<f64>,
{
    let inputs = align_inputs(inputs)?;
    let value = inputs[0].cast(&DataType::Float64)?;
    let param = inputs[1].cast(&DataType::Float64)?;
    let param_ca = param.f64()?;
    let name = inputs[0].name().clone();
    domain.check_column(param_ca)?;

    let ca: Float64Chunked = binary_elementwise(value.f64()?, param_ca, |value_opt, param_opt| {
        let param = param_opt?;
        let value = value_opt?;
        if value.is_nan() {
            Some(f64::NAN)
        } else {
            select(&derive(param), value)
        }
    });

    Ok(ca.with_name(name).into_series())
}

/// `exp(t) - 1` through the identity `2 exp(t / 2) sinh(t / 2)`, which has no subtraction to cancel.
///
/// Not `f64::exp_m1`, which differs from this identity by one ulp on roughly a fifth of the left
/// tail. Every expected value in `tests/distributions/exponential/` is this identity's rounding, and
/// `_base.expm1` spells it the same way for the moments still assembled in Polars.
///
/// `sinh(t / 2)` overflows above `|t| ~ 1420`; every caller crosses over to a direct form long before
/// that (`exponential::CDF_SINH_MAX`, `geometric::CDF_DIRECT_COMPLEMENT_MAX`).
#[inline]
pub(crate) fn expm1(t: f64) -> f64 {
    let half = t / 2.0;
    2.0 * half.exp() * half.sinh()
}

/// `ln|exp(t) - 1|` as `ln(2) + t / 2 + ln|sinh(t / 2)|`: [`expm1`]'s identity read term by term on
/// the log scale, so each term is large exactly where the answer is and the rounding stays relative;
/// `expm1(t).ln()` would round the answer's own magnitude away.
///
/// The absolute value carries both signs of `t`: a no-op for `t > 0`, and for `t < 0` the log of the
/// complement `1 - exp(t)`. Same overflow limit as [`expm1`]; `_base.log_abs_expm1` spells it the same
/// way for `LogNormal.entropy`.
#[inline]
pub(crate) fn ln_abs_expm1(t: f64) -> f64 {
    let half = t / 2.0;
    std::f64::consts::LN_2 + half + half.sinh().abs().ln()
}

/// The two sides of a support whose floor is the integer `FLOOR`, for the value-keyed methods of a
/// distribution that computes its own closed form: a constant below the floor, the parameter's arm
/// on it.
pub(crate) struct Sides<Arm, const FLOOR: i8> {
    pub(crate) below_support: f64,
    pub(crate) on_support: Arm,
}

impl<Arm: Fn(f64) -> f64, const FLOOR: i8> Sides<Arm, FLOOR> {
    /// `value < FLOOR` takes the constant, everything else the arm, so `-0.0` sits where `0.0` does.
    ///
    /// A `NaN` value never reaches here: every driver short-circuits it. That is what lets this be a
    /// bare `<`; the `!(value >= FLOOR)` a negated predicate would spell puts `NaN` below the support.
    pub(crate) fn at(&self, value: f64) -> Option<f64> {
        Some(if value < f64::from(FLOOR) {
            self.below_support
        } else {
            (self.on_support)(value)
        })
    }
}

/// The `select` every inverse shares: the arm on the closed quantile interval `[0, 1]`, `None`
/// (null) outside it. `-0.0` is inside, since `-0.0 == 0.0`.
///
/// A `NaN` quantile never reaches here: every driver short-circuits it, which is what lets this be a
/// plain range check.
#[inline]
pub(crate) fn on_unit_interval<Arm: Fn(f64) -> f64>(arm: &Arm, quantile: f64) -> Option<f64> {
    (0.0..=1.0).contains(&quantile).then(|| arm(quantile))
}

/// Two-parameter sibling of [`value_keyed_derived_per_row`], over `ternary_elementwise`.
///
/// "Pair" counts the *distribution parameters* (`Uniform`'s two bounds); the driver itself takes
/// three `Series`.
///
/// `check_params` is the caller's column pass over the two parameter columns, each alone and the
/// pair together, run once before any row is built. Null and `NaN` contracts as in
/// [`value_keyed_derived_per_row`]: a null in either parameter nulls the row.
pub(crate) fn value_keyed_derived_pair_per_row<Branches, CheckParams, Derive, Select>(
    inputs: &[Series],
    check_params: CheckParams,
    derive: Derive,
    select: Select,
) -> PolarsResult<Series>
where
    CheckParams: Fn(&Float64Chunked, &Float64Chunked) -> PolarsResult<()>,
    Derive: Fn(f64, f64) -> Branches,
    Select: Fn(&Branches, f64) -> Option<f64>,
{
    let inputs = align_inputs(inputs)?;
    let value = inputs[0].cast(&DataType::Float64)?;
    let param_a = inputs[1].cast(&DataType::Float64)?;
    let param_b = inputs[2].cast(&DataType::Float64)?;
    let (a_ca, b_ca) = (param_a.f64()?, param_b.f64()?);
    let name = inputs[0].name().clone();
    check_params(a_ca, b_ca)?;

    let ca: Float64Chunked =
        ternary_elementwise(value.f64()?, a_ca, b_ca, |value_opt, a_opt, b_opt| {
            let (a, b) = (a_opt?, b_opt?);
            let value = value_opt?;
            if value.is_nan() {
                Some(f64::NAN)
            } else {
                select(&derive(a, b), value)
            }
        });

    Ok(ca.with_name(name).into_series())
}
