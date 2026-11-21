//! The regularized incomplete beta `I_x(a, b)` behind `log_cdf` / `log_sf` is evaluated here in log space, not
//! through statrs' `beta_reg`: its continued fraction stops at 141 terms, which truncates silently from shapes of
//! about `1e4`, its prefactor sums `ln_gamma` terms that cancel to `4.7e-9` absolute in the log at `a = b = 1e6`,
//! and its linear result underflows where the log is still representable.

use std::cell::OnceCell;

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distr::Distribution as RandDistribution;
use statrs::distribution::{Beta, Continuous, ContinuousCDF};
use statrs::statistics::Distribution as StatrsDistribution;

use crate::distributions::gamma::{
    ln_1p_mx, Tail, Tails, CONVERGENCE_EPS, LN_2PI, STIRLING_FROM, STIRLING_REMAINDER_SERIES,
};
use crate::distributions::{
    coerce_f64, expm1, param_keyed, polynomial, validated_pair, value_keyed_derived_ternary,
    value_keyed_scalar, value_keyed_ternary, ParamDomain,
};
use crate::rng::{
    sample_by_index, sample_per_row_ternary, samples_by_index, samples_f64_output,
    samples_per_row_ternary, SampleKwargs, SampleScalarKwargs, SamplesKwargs, SamplesScalarKwargs,
};

const A: ParamDomain = ParamDomain::positive("a");
const B: ParamDomain = ParamDomain::positive("b");

fn check_params(a: &Float64Chunked, b: &Float64Chunked) -> PolarsResult<()> {
    A.check_column(a)?;
    B.check_column(b)
}

fn build_dist(a: f64, b: f64) -> PolarsResult<Beta> {
    Beta::new(a, b).map_err(|e| polars_err!(ComputeError: "{e}"))
}

/// Constant parameters, deserialised once per call. Every `_scalar` twin builds through
/// [`Self::build`], so it cannot rebuild per row.
#[derive(serde::Deserialize)]
struct BetaParams {
    a: f64,
    b: f64,
}

impl BetaParams {
    fn build(&self) -> PolarsResult<Beta> {
        A.check(self.a)?;
        B.check(self.b)?;
        build_dist(self.a, self.b)
    }

    fn value_keyed<Body>(&self, value: &Series, body: Body) -> PolarsResult<Series>
    where
        Body: Fn(&Beta, f64) -> Option<f64>,
    {
        value_keyed_scalar(value, &self.build()?, body)
    }

    fn incomplete_beta_keyed(
        &self,
        value: &Series,
        select: impl Fn(&IncompleteBeta, f64) -> Option<f64>,
    ) -> PolarsResult<Series> {
        A.check(self.a)?;
        B.check(self.b)?;
        value_keyed_scalar(value, &IncompleteBeta::new(self.a, self.b), select)
    }
}

/// Binet's remainder `mu(z) = ln Gamma(z) - (z - 1/2) ln z + z - ln(2 pi) / 2`, from [`STIRLING_FROM`] on.
fn stirling_remainder(z: f64) -> f64 {
    polynomial(&STIRLING_REMAINDER_SERIES, (z * z).recip()) / z
}

/// `ln Gamma(b + a) - ln Gamma(b)`, relatively exact as `a -> 0` where the difference of two `ln_gamma` calls
/// keeps only their absolute rounding. Below [`STIRLING_FROM`] `b` is shifted up by the recurrence, which
/// contributes `ln prod(1 + a / z)`; from there Stirling's formula is `(z - 1/2) ln(1 + a / z) - a + a ln(z + a)`
/// plus the remainders' increment, whose rounding is `eps a` against a value of at least `a ln 10`.
fn ln_gamma_increment(b: f64, a: f64) -> f64 {
    let (shifted, ln_steps) = shift_to_stirling(b, a);
    (shifted - 0.5) * (a / shifted).ln_1p() - a
        + a * (shifted + a).ln()
        + stirling_remainder_increment(shifted, a)
        - ln_steps
}

/// `b` shifted to [`STIRLING_FROM`] by unit steps `z`, and `ln prod(1 + a / z)` over them. The product is kept as
/// its excess over `1`, so one `ln_1p` reads it to full relative precision, without a branch per step. The excess
/// overflows only when a first `z` is below about `1e-300 a`, and then each step takes its own `ln_1p`.
fn shift_to_stirling(b: f64, a: f64) -> (f64, f64) {
    let mut shifted = b;
    let mut excess = 0.0_f64;
    while shifted < STIRLING_FROM {
        let step = a / shifted;
        excess = excess.mul_add(step, excess + step);
        shifted += 1.0;
    }
    if excess.is_finite() {
        return (shifted, excess.ln_1p());
    }
    let (mut shifted, mut ln_steps) = (b, 0.0);
    while shifted < STIRLING_FROM {
        ln_steps += (a / shifted).ln_1p();
        shifted += 1.0;
    }
    (shifted, ln_steps)
}

/// `mu(z + a) - mu(z)` term by term, each `z^-n ((z / (z + a))^n - 1)` as `z^-n (r - 1)(1 + r + ... + r^(n-1))`
/// with `r - 1 = -a / (z + a)`: the literal difference of two remainders near `1 / (12 z)` keeps only their
/// absolute rounding, `1e-18`, where the increment itself is `O(a)`.
fn stirling_remainder_increment(z: f64, a: f64) -> f64 {
    let r = z / (z + a);
    let (inverse, inverse_squared) = (z.recip(), (z * z).recip());
    let (mut geometric, mut r_power, mut z_power, mut total) = (0.0, 1.0, inverse, 0.0);
    for (index, coefficient) in STIRLING_REMAINDER_SERIES.iter().enumerate() {
        for _ in 0..if index == 0 { 1 } else { 2 } {
            geometric += r_power;
            r_power *= r;
        }
        total += coefficient * z_power * geometric;
        z_power *= inverse_squared;
    }
    total * (-a / (z + a))
}

/// From [`STIRLING_FROM`] on in both shapes the prefactor is DiDonato and Morris' `brcomp` form, which has
/// no `a ln a`-sized terms to cancel; below it `ln B(a, b)` is read off [`series_offset`] at the smaller shape, so
/// no term goes through statrs' `ln_gamma` and its `1e-14`.
enum LnPrefactor {
    LnBeta(f64),
    /// `ln(x0^a y0^b / B(a, b)) = ln(a b / (2 pi (a + b))) / 2 - mu(a) - mu(b) + mu(a + b)` at the mean
    /// `x0 = a / (a + b)`, with `ln x0` and `ln y0`.
    Stirling {
        at_mean: f64,
        ln_x0: f64,
        ln_y0: f64,
    },
}

/// The regularized incomplete beta of shapes `(a, b)`, with every shape-only term hoisted.
///
/// Every evaluation takes the point twice, as `x` and as its complement `y = 1 - x`, and reads each logarithm
/// off whichever of the two is the smaller, so a caller holding only `y` exactly never forms `1 - y`.
pub(super) struct IncompleteBeta {
    a: f64,
    b: f64,
    prefactor: LnPrefactor,
    /// `-ln(s B(a, b))` for the smaller shape `s`, relatively exact, where `s <= 1` and its series reads it.
    small_series: Option<f64>,
    /// The same for the larger shape where it is at most `1` too, computed on first use: a point reads one side's
    /// series, and a build per row evaluates one point.
    large_series: OnceCell<f64>,
}

/// Terms after which the series or the continued fraction gives up and answers `NaN`, `3.5 ms` of work. Only a point
/// at the mean itself, `lambda = 0`, needs many: the continued fraction takes 551 terms at `a = b = 1e6` and about
/// `4.4 (a + b)^(1/3)` beyond, so it answers there up to `a + b ~ 1e16`; 1.4 standard deviations away takes 155 at
/// `1e6`.
const MAX_TERMS: u32 = 1_000_000;

/// `-ln(a B(a, b))` as `ln Gamma(a + b) - ln Gamma(b) - ln Gamma(1 + a)`, every term `O(a)` as `a -> 0`.
fn series_offset(a: f64, b: f64) -> f64 {
    ln_gamma_increment(b, a) - ln_gamma_increment(1.0, a)
}

/// `shape (ln(1 + excess) - excess)`: [`ln_1p_mx`] near the mean, and from the logs it was read off where
/// `1 + excess` is far from `1` and forming it from `excess` would round.
fn excess_term(shape: f64, excess: f64, ln_ratio: f64) -> f64 {
    if excess.abs() < 0.5 {
        shape * ln_1p_mx(excess)
    } else {
        shape * (ln_ratio - excess)
    }
}

impl IncompleteBeta {
    pub(super) fn new(a: f64, b: f64) -> Self {
        let (small, large) = if a <= b { (a, b) } else { (b, a) };
        let small_offset = (small < STIRLING_FROM).then(|| series_offset(small, large));
        let prefactor = match small_offset {
            Some(offset) => LnPrefactor::LnBeta(-offset - small.ln()),
            None => {
                let remainders =
                    stirling_remainder(a) + stirling_remainder(b) - stirling_remainder(a + b);
                LnPrefactor::Stirling {
                    at_mean: 0.5 * (small.ln() - (small / large).ln_1p() - LN_2PI) - remainders,
                    ln_x0: -(b / a).ln_1p(),
                    ln_y0: -(a / b).ln_1p(),
                }
            },
        };
        Self {
            a,
            b,
            prefactor,
            small_series: small_offset.filter(|_| small <= 1.0),
            large_series: OnceCell::new(),
        }
    }

    /// `-ln(shape B(a, b))` for `shape` = `a` ([`Tail::Lower`]) or `b`, where that shape is at most `1`.
    fn series_offset(&self, side: Tail) -> Option<f64> {
        let (shape, other) = match side {
            Tail::Lower => (self.a, self.b),
            Tail::Upper => (self.b, self.a),
        };
        if shape > 1.0 {
            None
        } else if (side == Tail::Lower) == (self.a <= self.b) {
            self.small_series
        } else {
            Some(
                *self
                    .large_series
                    .get_or_init(|| series_offset(shape, other)),
            )
        }
    }

    /// `lambda = a - (a + b) x = (a + b) y - b`, by `fma` off the smaller of `x` and `y`, which the caller holds
    /// exactly.
    fn excess(&self, x: f64, y: f64) -> f64 {
        let (a, b) = (self.a, self.b);
        if x <= y {
            (-(a + b)).mul_add(x, a)
        } else {
            (a + b).mul_add(y, -b)
        }
    }

    /// `ln(x^a y^b / B(a, b))` at a point given as `x`, `y = 1 - x` and their logarithms. In the Stirling form it
    /// is `a (ln(1 + e) - e) + b (ln(1 + f) - f)` plus its value at the mean, with `x = x0 (1 + e)`,
    /// `y = y0 (1 + f)` and `a e + b f = 0`, the excess `lambda = a - (a + b) x` read by [`Self::excess`].
    pub(super) fn ln_prefactor(&self, x: f64, y: f64, ln_x: f64, ln_y: f64) -> f64 {
        let (a, b) = (self.a, self.b);
        match self.prefactor {
            LnPrefactor::LnBeta(ln_beta) => a * ln_x + b * ln_y - ln_beta,
            LnPrefactor::Stirling {
                at_mean,
                ln_x0,
                ln_y0,
            } => {
                let lambda = self.excess(x, y);
                excess_term(a, -lambda / a, ln_x - ln_x0)
                    + excess_term(b, lambda / b, ln_y - ln_y0)
                    + at_mean
            },
        }
    }

    /// Both tails at the point `(x, y)`, `x + y = 1`: `I_x(a, b)` is [`Tail::Lower`], its complement
    /// `I_y(b, a)` [`Tail::Upper`].
    ///
    /// The point is oriented as `(shape, other_shape, u) = (a, b, x)` or `(b, a, y)`, so the tail evaluated there is
    /// never formed as a complement. Where the side `u < (shape + 1) / (a + b + 2)` has `shape <= 1`, that tail can
    /// sit next to `1` with its complement carrying the answer, so the power series gives both; there `u < 2/3`,
    /// which bounds the series' length. Otherwise the side below the mean, `lambda >= 0`, has `shape > 1` (the two
    /// sides differ only where both shapes are), its tail is below `1 - e^-1`, and the continued fraction times the
    /// prefactor gives it.
    pub(super) fn tails(&self, x: f64, y: f64) -> Tails {
        let (a, b) = (self.a, self.b);
        if x == 0.0 || y == 0.0 {
            let smaller = if x == 0.0 { Tail::Lower } else { Tail::Upper };
            return Tails {
                ln_prefactor: f64::NEG_INFINITY,
                smaller,
                ln_smaller: f64::NEG_INFINITY,
            };
        }
        let ln_x = if x <= y { x.ln() } else { (-y).ln_1p() };
        let ln_y = if y <= x { y.ln() } else { (-x).ln_1p() };
        let ln_prefactor = self.ln_prefactor(x, y, ln_x, ln_y);
        let series = if x < (a + 1.0) / (a + b + 2.0) {
            self.series_offset(Tail::Lower)
                .map(|offset| (Tail::Lower, a, b, x, a * ln_x + offset))
        } else {
            self.series_offset(Tail::Upper)
                .map(|offset| (Tail::Upper, b, a, y, b * ln_y + offset))
        };
        let (smaller, ln_smaller) = if let Some((side, shape, other_shape, u, exponent)) = series {
            let (ln_side, ln_other) = power_series(shape, other_shape, u, exponent);
            if ln_side <= ln_other {
                (side, ln_side)
            } else {
                (side.complement(), ln_other)
            }
        } else {
            let lambda = self.excess(x, y);
            let (side, shape, other_shape, u, v) = if lambda >= 0.0 {
                (Tail::Lower, a, b, x, y)
            } else {
                (Tail::Upper, b, a, y, x)
            };
            let fraction = continued_fraction(shape, other_shape, u, v, lambda.abs());
            (side, ln_prefactor + fraction.ln())
        };
        Tails {
            ln_prefactor,
            smaller,
            ln_smaller,
        }
    }
}

/// `(ln I_x(a, b), ln(1 - I_x(a, b)))` for `a <= 1`, by DLMF 8.17.7's hypergeometric series
/// `I_x(a, b) = e^E (1 + a S)`, `S = sum_{j >= 1} (1 - b)_j x^j / (j! (a + j))`, `E = a ln x - ln(a B(a, b))`.
/// The complement is `-expm1(E) - e^E a S`, both terms `O(a)` as `a -> 0`, so it keeps its relative precision
/// where `1 - I` would round it away. On its side `b x < a + 1 <= 2`, so the terms grow at most `e^2` before they
/// fall.
fn power_series(a: f64, b: f64, x: f64, exponent: f64) -> (f64, f64) {
    let mut coefficient = 1.0;
    let mut sum = 0.0;
    for j in 1..MAX_TERMS {
        let j = f64::from(j);
        coefficient *= (j - b) * x / j;
        let term = coefficient / (a + j);
        sum += term;
        if term.abs() <= CONVERGENCE_EPS * sum.abs() {
            let lower = exponent + (a * sum).ln_1p();
            let upper = -expm1(exponent) - exponent.exp() * a * sum;
            return (lower, upper.ln());
        }
    }
    (f64::NAN, f64::NAN)
}

/// The continued fraction `r` with `I_x(a, b) = x^a y^b r / B(a, b)`, `y = 1 - x`, below the mean, where
/// `lambda = a - (a + b) x >= 0`: DiDonato and Morris' `bfrac` (TOMS 708), a three-term recurrence rescaled every
/// step. It reads `lambda` and `y`, never `1 - x`: the textbook Lentz fraction (DLMF 8.17.22) forms `1 - x` in every
/// odd partial denominator, which at `x = 1 - 1e-8` from a rounded complement is `1e-8` relative off.
fn continued_fraction(a: f64, b: f64, x: f64, y: f64, lambda: f64) -> f64 {
    let c = lambda + 1.0;
    let c0 = b / a;
    let c1 = a.recip() + 1.0;
    let y_plus_one = y + 1.0;
    let mut growth = 1.0;
    let mut s = a + 1.0;
    let mut ratio = c1 / c;
    // Normalised by the first denominator `c / c1`, as every later step is: at `lambda ~ 1e200` it times `beta`
    // overflows.
    let (mut numerator_previous, mut denominator_previous, mut numerator, mut denominator) =
        (0.0, ratio, ratio, 1.0);
    for m in 1..MAX_TERMS {
        let n = f64::from(m);
        let t = n / a;
        let w = n * (b - n) * x;
        let e = a / s;
        let alpha = growth * (growth + c0) * e * e * (w * x);
        let e = (t + 1.0) / (c1 + t + t);
        let beta = n + w / s + e * (c + n * y_plus_one);
        growth = t + 1.0;
        s += 2.0;
        let numerator_next = alpha * numerator_previous + beta * numerator;
        let denominator_next = alpha * denominator_previous + beta * denominator;
        let previous_ratio = ratio;
        ratio = numerator_next / denominator_next;
        if (ratio - previous_ratio).abs() <= CONVERGENCE_EPS * ratio {
            return ratio;
        }
        numerator_previous = numerator / denominator_next;
        denominator_previous = denominator / denominator_next;
        numerator = ratio;
        denominator = 1.0;
    }
    f64::NAN
}

fn value_keyed<Body>(inputs: &[Series], body: Body) -> PolarsResult<Series>
where
    Body: Fn(&Beta, f64) -> Option<f64>,
{
    value_keyed_ternary(
        inputs,
        coerce_f64,
        coerce_f64,
        check_params,
        build_dist,
        body,
    )
}

#[polars_expr(output_type=Float64)]
fn beta_params(inputs: &[Series]) -> PolarsResult<Series> {
    validated_pair(inputs, coerce_f64, check_params)
}

#[inline]
fn draw(dist: &Beta, rng: &mut impl rand::Rng) -> f64 {
    RandDistribution::sample(dist, rng)
}

#[polars_expr(output_type=Float64)]
fn beta_sample(inputs: &[Series], kwargs: SampleKwargs) -> PolarsResult<Series> {
    sample_per_row_ternary(
        inputs,
        kwargs,
        coerce_f64,
        coerce_f64,
        check_params,
        build_dist,
        draw,
    )
}

#[polars_expr(output_type=Float64)]
fn beta_sample_scalar(
    inputs: &[Series],
    kwargs: SampleScalarKwargs<BetaParams>,
) -> PolarsResult<Series> {
    let dist = kwargs.params.build()?;
    sample_by_index(&inputs[0], kwargs.seed, |rng| draw(&dist, rng))
}

#[polars_expr(output_type_func_with_kwargs=samples_f64_output)]
fn beta_samples_scalar(
    inputs: &[Series],
    kwargs: SamplesScalarKwargs<BetaParams>,
) -> PolarsResult<Series> {
    let dist = kwargs.params.build()?;
    samples_by_index(&inputs[0], kwargs.seed, kwargs.size, |rng| draw(&dist, rng))
}

#[polars_expr(output_type_func_with_kwargs=samples_f64_output)]
fn beta_samples(inputs: &[Series], kwargs: SamplesKwargs) -> PolarsResult<Series> {
    samples_per_row_ternary(
        inputs,
        kwargs,
        coerce_f64,
        coerce_f64,
        check_params,
        build_dist,
        draw,
    )
}

fn pdf_value(dist: &Beta, v: f64) -> Option<f64> {
    Some(dist.pdf(v))
}

fn ln_pdf_value(dist: &Beta, v: f64) -> Option<f64> {
    Some(dist.ln_pdf(v))
}

// The regularized incomplete beta behind `cdf` / `sf` panics on a `NaN` point; the drivers'
// short-circuit is what keeps it out.

fn cdf_value(dist: &Beta, v: f64) -> Option<f64> {
    Some(dist.cdf(v))
}

fn sf_value(dist: &Beta, v: f64) -> Option<f64> {
    Some(dist.sf(v))
}

fn ln_cdf_value(beta: &IncompleteBeta, v: f64) -> Option<f64> {
    Some(if v <= 0.0 {
        f64::NEG_INFINITY
    } else if v >= 1.0 {
        0.0
    } else {
        beta.tails(v, 1.0 - v).ln_probability(Tail::Lower)
    })
}

fn ln_sf_value(beta: &IncompleteBeta, v: f64) -> Option<f64> {
    Some(if v <= 0.0 {
        0.0
    } else if v >= 1.0 {
        f64::NEG_INFINITY
    } else {
        beta.tails(v, 1.0 - v).ln_probability(Tail::Upper)
    })
}

/// Null outside `[0, 1]`; `statrs` panics there. The endpoints map to the support bounds.
fn ppf_value(dist: &Beta, q: f64) -> Option<f64> {
    if !(0.0..=1.0).contains(&q) {
        None
    } else {
        Some(dist.inverse_cdf(q))
    }
}

fn isf_value(dist: &Beta, q: f64) -> Option<f64> {
    ppf_value(dist, 1.0 - q)
}

#[polars_expr(output_type=Float64)]
fn beta_pdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, pdf_value)
}

#[polars_expr(output_type=Float64)]
fn beta_ln_pdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, ln_pdf_value)
}

#[polars_expr(output_type=Float64)]
fn beta_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, cdf_value)
}

#[polars_expr(output_type=Float64)]
fn beta_ln_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, IncompleteBeta::new, ln_cdf_value)
}

#[polars_expr(output_type=Float64)]
fn beta_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, sf_value)
}

#[polars_expr(output_type=Float64)]
fn beta_ln_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, IncompleteBeta::new, ln_sf_value)
}

#[polars_expr(output_type=Float64)]
fn beta_ppf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, ppf_value)
}

#[polars_expr(output_type=Float64)]
fn beta_isf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, isf_value)
}

#[polars_expr(output_type=Float64)]
fn beta_pdf_scalar(inputs: &[Series], kwargs: BetaParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], pdf_value)
}

#[polars_expr(output_type=Float64)]
fn beta_ln_pdf_scalar(inputs: &[Series], kwargs: BetaParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], ln_pdf_value)
}

#[polars_expr(output_type=Float64)]
fn beta_cdf_scalar(inputs: &[Series], kwargs: BetaParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], cdf_value)
}

#[polars_expr(output_type=Float64)]
fn beta_ln_cdf_scalar(inputs: &[Series], kwargs: BetaParams) -> PolarsResult<Series> {
    kwargs.incomplete_beta_keyed(&inputs[0], ln_cdf_value)
}

#[polars_expr(output_type=Float64)]
fn beta_sf_scalar(inputs: &[Series], kwargs: BetaParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], sf_value)
}

#[polars_expr(output_type=Float64)]
fn beta_ln_sf_scalar(inputs: &[Series], kwargs: BetaParams) -> PolarsResult<Series> {
    kwargs.incomplete_beta_keyed(&inputs[0], ln_sf_value)
}

#[polars_expr(output_type=Float64)]
fn beta_ppf_scalar(inputs: &[Series], kwargs: BetaParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], ppf_value)
}

#[polars_expr(output_type=Float64)]
fn beta_isf_scalar(inputs: &[Series], kwargs: BetaParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], isf_value)
}

/// Differential entropy (nats), `ln B(a, b) - (a - 1) psi(a) - (b - 1) psi(b) + (a + b - 2) psi(a + b)`:
/// log-Beta and digamma have no Polars expression to move to.
#[polars_expr(output_type=Float64)]
fn beta_entropy(inputs: &[Series]) -> PolarsResult<Series> {
    param_keyed(inputs, coerce_f64, coerce_f64, check_params, |a, b| {
        Ok(build_dist(a, b)?
            .entropy()
            .expect("Beta::entropy is Some for every valid (a, b)"))
    })
}
