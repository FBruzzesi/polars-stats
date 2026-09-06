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
    binary_elementwise, ternary_elementwise, try_binary_elementwise, try_ternary_elementwise,
    try_unary_elementwise, unary_elementwise,
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

/// What one float parameter must satisfy on its own, stated once per distribution file and checked
/// the same way in both regimes: [`Self::check`] once per call on a constant, [`Self::check_column`]
/// over the whole column before any row is built.
///
/// Every domain spells finiteness explicitly. The `statrs` constructors do not check it
/// (`Normal::new(0.0, f64::INFINITY)` is `Ok`), so the constructor error inside a row loop is a
/// backstop, not what the contract rests on.
pub(crate) struct ParamDomain {
    pub(crate) name: &'static str,
    /// Read as "`name` must be {domain}".
    pub(crate) domain: &'static str,
    pub(crate) accepts: fn(f64) -> bool,
}

impl ParamDomain {
    fn violation(&self, value: f64) -> String {
        format!("{} must be {}, got {}", self.name, self.domain, value)
    }

    pub(crate) fn check(&self, value: f64) -> PolarsResult<()> {
        polars_ensure!((self.accepts)(value), ComputeError: "{}", self.violation(value));
        Ok(())
    }

    /// Raise on the first present value outside the domain, naming its row.
    ///
    /// A plain iterator with an early exit, not the vectorised primitives: those would build one
    /// boolean column per predicate and still need a scan to name the offender.
    pub(crate) fn check_column(&self, ca: &Float64Chunked) -> PolarsResult<()> {
        for (row, value) in ca.iter().enumerate() {
            if let Some(value) = value {
                polars_ensure!(
                    (self.accepts)(value),
                    ComputeError: "{} at row {}", self.violation(value), row
                );
            }
        }
        Ok(())
    }
}

/// What two parameters must satisfy together (`Uniform`'s `max > min`, `DiscreteUniform`'s
/// `min <= max`), checked only where both are present: a row with one null bound is `null` under
/// the null contract, and the present bound still passes its own [`ParamDomain`].
pub(crate) struct PairDomain<N> {
    pub(crate) names: (&'static str, &'static str),
    /// Read as "`names.1` must be {domain}".
    pub(crate) domain: &'static str,
    pub(crate) accepts: fn(N, N) -> bool,
}

impl<N: Copy + std::fmt::Display> PairDomain<N> {
    fn violation(&self, a: N, b: N) -> String {
        let (a_name, b_name) = self.names;
        format!(
            "{b_name} must be {}, got {a_name}={a}, {b_name}={b}",
            self.domain
        )
    }

    pub(crate) fn check(&self, a: N, b: N) -> PolarsResult<()> {
        polars_ensure!((self.accepts)(a, b), ComputeError: "{}", self.violation(a, b));
        Ok(())
    }

    /// The column twin of [`Self::check`]: raise on the first row where both are present and the
    /// pair is outside the domain, naming the row.
    pub(crate) fn check_columns<T>(
        &self,
        a: &ChunkedArray<T>,
        b: &ChunkedArray<T>,
    ) -> PolarsResult<()>
    where
        T: PolarsNumericType<Native = N>,
    {
        for (row, pair) in a.iter().zip(b.iter()).enumerate() {
            if let (Some(a), Some(b)) = pair {
                polars_ensure!(
                    (self.accepts)(a, b),
                    ComputeError: "{} at row {}", self.violation(a, b), row
                );
            }
        }
        Ok(())
    }
}

/// Shared driver for the constant-parameter value-keyed fast paths.
///
/// The constant-parameter counterpart of each distribution's `value_keyed` helper: when every
/// distribution parameter is a Python scalar, the caller validates them and builds the
/// distribution **once**, and only the evaluation-point column crosses FFI. This maps `f` over
/// that single column, where `f` is the same per-method body the per-row path uses (a named
/// function in each distribution file), so the two paths cannot drift and output is bit-identical.
///
/// Null contract: `value` is the only nullable input this driver sees. The distribution
/// parameters are not passed here at all; the caller validates them once (via `build_dist`,
/// which raises on an invalid or non-finite parameterisation before this runs) and bakes them
/// into `f` through a pre-built `dist`. So a null `value` propagates to null, and `f` returning
/// `None` nulls the row on the method's own terms (e.g. `ppf` outside `[0, 1]`), matching the
/// per-row path element for element.
///
/// `NaN` contract: a `NaN` evaluation point short-circuits to `NaN` (scipy semantics) before `f`
/// runs, for every method including `ppf`. The short-circuit is central (here and in
/// [`value_keyed_per_row`]) rather than per body, because two bodies genuinely need it and none
/// may be forgotten: the regularized incomplete beta behind `Beta` `cdf`/`sf` panics on `NaN`
/// (aborting the whole query), and binomial's support mapping saturates (`NaN.floor() as u64` is
/// `0`), returning a confident `P(X <= 0)`. The Python-side `propagate_null_and_nan` guard cannot
/// substitute: polars evaluates every `when`/`then`/`otherwise` branch over the full column, so a
/// plugin runs on the `NaN` rows even though the guard discards their output. Pinned by
/// `tests/distributions/plugin_nan_test.py`.
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
/// evaluation-point column. The Python side routes here only once the parameter is a Python scalar,
/// so `derive` always sees `Some`, and the parameter-only terms it hoists (`Bernoulli`'s `1 - p`,
/// `Exponential`'s `ln(rate)`) are computed once instead of per row.
pub(crate) fn value_keyed_derived_scalar<Branches, Derive, Select>(
    value: &Series,
    param: f64,
    domain: &ParamDomain,
    derive: Derive,
    select: Select,
) -> PolarsResult<Series>
where
    Derive: Fn(Option<f64>) -> Branches,
    Select: Fn(&Branches, f64) -> Option<f64>,
{
    domain.check(param)?;
    let branches = derive(Some(param));
    value_keyed_scalar(value, |v| select(&branches, v))
}

/// Shared driver for the column-parameter value-keyed per-row paths.
///
/// The column-parameter counterpart of [`value_keyed_scalar`]: at least one distribution parameter
/// is a column, so `build` validates and constructs once per row instead of once per call. `f` is
/// the same named per-method body the fast path applies (`cdf_value`, `ppf_value`, ...), so the two
/// paths cannot drift and agree bit for bit.
///
/// The caller does the cast and the accessor (`.f64()` / `.u64()`), which fixes `A` and `B`, so a
/// mixed `(u64, f64)` parameterisation (Binomial's `UInt64` `n` beside its `Float64` `p`) fits, as in
/// [`ternary_param_rows`](crate::rng::ternary_param_rows). `S` needs no trait bound: it is whatever
/// `build` returns.
///
/// Null contract: any null among `(value, p1, p2)` nulls the row without calling `build`, matching
/// the samplers.
///
/// `NaN` contract: as in [`value_keyed_scalar`], except that the short-circuit runs after `build`,
/// so an invalid parameterisation still raises on a `NaN` row.
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
            let (Some(p1), Some(p2)) = (p1_opt, p2_opt) else {
                return Ok(None);
            };
            // Build before the value is read at all, so an invalid parameterisation raises whatever
            // the row's value is. Only a null or `NaN` *value* propagates. Pinned by
            // `tests/distributions/plugin_nan_test.py` and `invalid_param_null_value_test.py`.
            let dist = build(p1, p2)?;
            let Some(value) = value_opt else {
                return Ok(None);
            };
            Ok(if value.is_nan() {
                Some(f64::NAN)
            } else {
                f(&dist, value)
            })
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
/// Validation is `domain`'s column pass, run once over the parameter column before any row is
/// built: an invalid present parameter raises whatever else its row holds, a null or `NaN` value
/// included. Nothing inside the row loop validates.
///
/// Null contract, and the reason this is not [`value_keyed_per_row`]: that driver nulls the row on
/// **any** null input without calling `build`, which is right for a `statrs`-backed distribution and
/// wrong here. A null parameter reaches `derive` as `None`, so the branches whose answer is a
/// parameter-free constant still answer: `Bernoulli`'s `pmf(2) = 0` and `Exponential`'s
/// `cdf(-1) = 0` survive a null parameter, and each distribution's `null_param(s)_test.py` pins it.
/// A null **value** nulls the row, matching the samplers.
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
    Derive: Fn(Option<f64>) -> Branches,
    Select: Fn(&Branches, f64) -> Option<f64>,
{
    let inputs = align_inputs(inputs)?;
    let value = inputs[0].cast(&DataType::Float64)?;
    let param = inputs[1].cast(&DataType::Float64)?;
    let param_ca = param.f64()?;
    let name = inputs[0].name().clone();
    domain.check_column(param_ca)?;

    let ca: Float64Chunked = binary_elementwise(value.f64()?, param_ca, |value_opt, param_opt| {
        let value = value_opt?;
        if value.is_nan() {
            Some(f64::NAN)
        } else {
            select(&derive(param_opt), value)
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
/// distribution that computes its own closed form.
///
/// Below the floor every answer is a parameter-free constant, so a null parameter must not null it:
/// `below_support` is a plain `f64`, leaving nothing to thread the parameter through by accident.
/// `on_support` is `None` exactly when the parameter is null. Each distribution's
/// `null_param(s)_test.py` pins its constants.
pub(crate) struct Sides<Arm, const FLOOR: i8> {
    pub(crate) below_support: f64,
    pub(crate) on_support: Option<Arm>,
}

impl<Arm: Fn(f64) -> f64, const FLOOR: i8> Sides<Arm, FLOOR> {
    /// `value < FLOOR` takes the constant, everything else the arm, so `-0.0` sits where `0.0` does.
    ///
    /// A `NaN` value never reaches here: every driver short-circuits it. That is what lets this be a
    /// bare `<`; the `!(value >= FLOOR)` a negated predicate would spell puts `NaN` below the support.
    pub(crate) fn at(&self, value: f64) -> Option<f64> {
        if value < f64::from(FLOOR) {
            Some(self.below_support)
        } else {
            self.on_support.as_ref().map(|arm| arm(value))
        }
    }
}

/// The closed quantile domain `[0, 1]`, for the inverses of a distribution that computes its own
/// closed form rather than building a `statrs` one.
///
/// One slot rather than the two a support needs, because no part of either inverse survives a
/// null parameter, at any quantile in range or out; each distribution's `null_param(s)_test.py` pins
/// it.
pub(crate) struct Domain<Arm> {
    pub(crate) inside: Option<Arm>,
}

impl<Arm: Fn(f64) -> f64> Domain<Arm> {
    /// `None` (null) outside `[0, 1]`; `-0.0` is inside, since `-0.0 == 0.0`.
    ///
    /// A `NaN` quantile never reaches here: every driver short-circuits it, which is what lets this
    /// be a plain range check.
    pub(crate) fn at(&self, quantile: f64) -> Option<f64> {
        if !(0.0..=1.0).contains(&quantile) {
            return None;
        }
        self.inside.as_ref().map(|arm| arm(quantile))
    }
}

/// Two-parameter sibling of [`value_keyed_derived_per_row`], over `ternary_elementwise`.
///
/// "Pair" counts the *distribution parameters* (`Uniform`'s two bounds); the driver itself takes
/// three `Series`. Kept beside the one-parameter version rather than generified over arity, because
/// `derive` needs both `Option`s in scope to answer from one bound while the other is null.
///
/// `validate` is the caller's column pass over the two parameter columns, each bound alone and the
/// pair together, run once before any row is built. Nothing inside the row loop validates.
///
/// Null contract: as in [`value_keyed_derived_per_row`], with the parameter half now two-sided. A
/// null **value** nulls the row. A null parameter reaches `derive` as `None`, so the branches a
/// single known parameter already settles still answer.
///
/// `NaN` contract: a `NaN` value short-circuits to `NaN`.
pub(crate) fn value_keyed_derived_pair_per_row<Branches, Validate, Derive, Select>(
    inputs: &[Series],
    validate: Validate,
    derive: Derive,
    select: Select,
) -> PolarsResult<Series>
where
    Validate: Fn(&Float64Chunked, &Float64Chunked) -> PolarsResult<()>,
    Derive: Fn(Option<f64>, Option<f64>) -> Branches,
    Select: Fn(&Branches, f64) -> Option<f64>,
{
    let inputs = align_inputs(inputs)?;
    let value = inputs[0].cast(&DataType::Float64)?;
    let param_a = inputs[1].cast(&DataType::Float64)?;
    let param_b = inputs[2].cast(&DataType::Float64)?;
    let (a, b) = (param_a.f64()?, param_b.f64()?);
    let name = inputs[0].name().clone();
    validate(a, b)?;

    let ca: Float64Chunked = ternary_elementwise(value.f64()?, a, b, |value_opt, a_opt, b_opt| {
        let value = value_opt?;
        if value.is_nan() {
            Some(f64::NAN)
        } else {
            select(&derive(a_opt, b_opt), value)
        }
    });

    Ok(ca.with_name(name).into_series())
}

/// Shared driver for the two-parameter validation plugins: `validate` builds the distribution per
/// row and returns the `Float64` to emit, either a parameter itself (`sigma`) or a quantity derived
/// from both (Uniform's `max - min`), `?`-propagating the `InvalidOperation` out of `build_dist`.
/// Any null input nulls the row without calling `validate`, matching the samplers.
///
/// These plugins are what let the closed-form moments report an invalid parameterisation through
/// the same Rust `build_dist` as the value-keyed methods and the samplers. The constant-parameter
/// fast path calls the same plugin on length-1 `pl.lit` inputs, so it is built once instead of per
/// row.
///
/// The two parameter dtypes are independent, so a mixed `(u64, f64)` parameterisation (Binomial)
/// fits, as in [`ternary_param_rows`](crate::rng::ternary_param_rows): the caller does the cast and
/// the accessor (`.f64()` / `.u64()`), which fixes `A` and `B`.
///
/// Takes no output name: polars resolves a plugin expression's output name from its first input, so
/// the frame column follows `inputs[0]` (pinned by `tests/distributions/output_name_test.py`).
///
/// Keep `validate` a generic `F: Fn`: it monomorphises into the row loop, where a `&dyn Fn` or a
/// `fn` pointer would cost an indirect call per row.
pub(crate) fn validate_params_binary<A, B, F>(
    a: &ChunkedArray<A>,
    b: &ChunkedArray<B>,
    validate: F,
) -> PolarsResult<Series>
where
    A: PolarsNumericType,
    B: PolarsNumericType,
    F: Fn(A::Native, B::Native) -> PolarsResult<f64>,
{
    let ca: Float64Chunked =
        try_binary_elementwise(a, b, |a_opt, b_opt| -> PolarsResult<Option<f64>> {
            match (a_opt, b_opt) {
                (Some(a), Some(b)) => Ok(Some(validate(a, b)?)),
                _ => Ok(None),
            }
        })?;

    Ok(ca.into_series())
}

/// Single-parameter counterpart of [`validate_params_binary`], same contracts
/// (`bernoulli_proba`, `exponential_rate`).
pub(crate) fn validate_params_unary<A, F>(a: &ChunkedArray<A>, validate: F) -> PolarsResult<Series>
where
    A: PolarsNumericType,
    F: Fn(A::Native) -> PolarsResult<f64>,
{
    let ca: Float64Chunked = try_unary_elementwise(a, |a_opt| -> PolarsResult<Option<f64>> {
        match a_opt {
            Some(a) => Ok(Some(validate(a)?)),
            None => Ok(None),
        }
    })?;

    Ok(ca.into_series())
}
