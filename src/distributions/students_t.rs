//! `StudentsT(dof, loc, scale)` is evaluated here rather than through statrs' `StudentsT`: its density switches to the
//! normal from `dof = 1e8`, which is unboundedly off in the polynomial tails; its `cdf` squares the standardised point,
//! which overflows to a tail of `0` past `|t| ~ 1.3e154`; and its `inverse_cdf` is the incomplete-beta inverse that
//! hangs `Beta.ppf`. The tails are `beta.rs`'s log-space incomplete beta at `x = dof / (dof + t^2)`, with
//! `y = t^2 / (dof + t^2)` formed directly beside it.

use std::f64::consts::{LN_2, SQRT_2};

use polars::prelude::arity::unary_elementwise;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distr::Distribution as RandDistribution;
use statrs::distribution::StudentsT;
use statrs::function::erf::erfc_inv;
use statrs::function::gamma::digamma;

use crate::distributions::beta::IncompleteBeta;
use crate::distributions::gamma::LN_2PI_E;
use crate::distributions::{
    bracketed_newton, cauchy, coerce_f64, ln_abs_expm1, on_unit_interval, polynomial, scale_exp,
    validated_triple, value_keyed_quaternary, value_keyed_scalar, ParamDomain, Tail, Tails,
};
use crate::rng::{
    sample_by_index, sample_per_row_quaternary, samples_by_index, samples_f64_output,
    samples_per_row_quaternary, SampleKwargs, SampleScalarKwargs, SamplesKwargs,
    SamplesScalarKwargs,
};

const DOF: ParamDomain = ParamDomain::positive("dof");
const LOC: ParamDomain = ParamDomain::finite("loc");
const SCALE: ParamDomain = ParamDomain::positive("scale");

fn check_params(
    dof: &Float64Chunked,
    loc: &Float64Chunked,
    scale: &Float64Chunked,
) -> PolarsResult<()> {
    DOF.check_column(dof)?;
    LOC.check_column(loc)?;
    SCALE.check_column(scale)
}

fn build_dist(dof: f64, loc: f64, scale: f64) -> PolarsResult<StudentsT> {
    StudentsT::new(loc, scale, dof).map_err(|e| polars_err!(ComputeError: "{e}"))
}

/// Constant parameters, deserialised once per call. Every `_scalar` twin checks them once here.
#[derive(serde::Deserialize)]
struct StudentsTParams {
    dof: f64,
    loc: f64,
    scale: f64,
}

impl StudentsTParams {
    fn check(&self) -> PolarsResult<()> {
        DOF.check(self.dof)?;
        LOC.check(self.loc)?;
        SCALE.check(self.scale)
    }

    fn build(&self) -> PolarsResult<StudentsT> {
        self.check()?;
        build_dist(self.dof, self.loc, self.scale)
    }

    fn value_keyed(
        &self,
        value: &Series,
        select: impl Fn(&ScaledT, f64) -> Option<f64>,
    ) -> PolarsResult<Series> {
        self.check()?;
        value_keyed_scalar(value, &ScaledT::new(self.dof, self.loc, self.scale), select)
    }
}

/// From this `ln(t^2 / dof)` on, `x = dof / (dof + t^2)` is below `1e-300`: its logarithm is read off `ln(t^2 / dof)`
/// and the far tail off the leading term of its power series, whose correction is `O(x)`. `x` itself would turn
/// subnormal within another decade and then `0`.
const FAR_FROM_LN_RATIO: f64 = 690.0; // keep in sync with `STUDENTS_T_FAR_FROM_LN_RATIO` in `tools/accuracy/audit.py`

/// The largest `r` whose `ln` is not above [`FAR_FROM_LN_RATIO`], so a normal `r` is classified without its `ln`.
const FAR_FROM_RATIO: f64 = 4.604_606_404_783_251e299;

/// `r = t^2 / dof` at a standardised point `t`: itself up to [`FAR_FROM_LN_RATIO`], its logarithm past it.
#[derive(Clone, Copy)]
enum Ratio {
    Near(f64),
    Far { ln_r: f64 },
}

impl Ratio {
    fn from_ln(ln_r: f64) -> Self {
        if ln_r > FAR_FROM_LN_RATIO {
            Self::Far { ln_r }
        } else {
            Self::Near(ln_r.exp())
        }
    }
}

/// The standard `t` distribution of `dof` degrees of freedom, with every `dof`-only term hoisted.
struct StandardT {
    dof: f64,
    ln_dof: f64,
    sqrt_dof: f64,
    half_dof: f64,
    half_dof_plus_half: f64,
    /// `I_x(dof / 2, 1 / 2)`, twice the tail beyond `|t|`; its complement `I_y(1 / 2, dof / 2)` is twice the mass
    /// between `0` and `|t|`.
    beta: IncompleteBeta,
    ln_beta: f64,
    /// The log density at `0`, `-ln(sqrt(dof) B(dof / 2, 1 / 2))`.
    ln_peak: f64,
}

impl StandardT {
    fn new(dof: f64) -> Self {
        let half_dof = 0.5 * dof;
        let beta = IncompleteBeta::new(half_dof, 0.5);
        let ln_beta = beta.ln_beta();
        let ln_dof = dof.ln();
        Self {
            dof,
            ln_dof,
            sqrt_dof: dof.sqrt(),
            half_dof,
            half_dof_plus_half: half_dof + 0.5,
            beta,
            ln_beta,
            ln_peak: -0.5 * ln_dof - ln_beta,
        }
    }

    /// `ln((dof / 2) B(dof / 2, 1 / 2))`, relatively exact as `dof -> 0`. Only the far tail and the inverse read it.
    fn ln_half_dof_beta(&self) -> f64 {
        self.beta
            .series_offset_on(Tail::Lower)
            .map_or_else(|| self.half_dof.ln() + self.ln_beta, |offset| -offset)
    }

    /// The ratio at `t = distance / scale`, never formed as `t` itself where it over- or underflows.
    fn ratio_at(&self, distance: f64, scale: f64, ln_scale: f64) -> Ratio {
        let w = distance / scale / self.sqrt_dof;
        let r = w * w;
        if r.is_normal() {
            return if r > FAR_FROM_RATIO {
                Ratio::Far { ln_r: r.ln() }
            } else {
                Ratio::Near(r)
            };
        }
        if distance == 0.0 {
            return Ratio::Near(0.0);
        }
        let ln_r = if w.is_normal() {
            2.0 * w.abs().ln()
        } else {
            2.0 * (distance.abs().ln() - ln_scale) - self.ln_dof
        };
        Ratio::from_ln(ln_r)
    }

    fn ratio_at_ln_t(&self, ln_t: f64) -> Ratio {
        let t = ln_t.exp();
        if t.is_normal() {
            self.ratio_at(t, 1.0, 0.0)
        } else {
            Ratio::from_ln(2.0 * ln_t - self.ln_dof)
        }
    }

    /// `(dof + 1) / 2 ln(1 + r)`, the log density's fall from the peak.
    fn ln_decay(&self, ratio: Ratio) -> f64 {
        let ln_1p_r = match ratio {
            Ratio::Near(r) => r.ln_1p(),
            Ratio::Far { ln_r } => ln_r + (-ln_r).exp().ln_1p(),
        };
        self.half_dof_plus_half * ln_1p_r
    }

    fn ln_density(&self, ratio: Ratio) -> f64 {
        self.ln_peak - self.ln_decay(ratio)
    }

    /// `I_x(dof / 2, 1 / 2)` ([`Tail::Lower`]) and `I_y(1 / 2, dof / 2)` at the ratio, each formed directly.
    fn tails(&self, ratio: Ratio) -> Tails {
        let ln_r = match ratio {
            Ratio::Near(r) => {
                let total = 1.0 + r;
                return self.beta.tails(total.recip(), r / total);
            },
            Ratio::Far { ln_r } => ln_r,
        };
        let ln_x = -(ln_r + (-ln_r).exp().ln_1p());
        let ln_y = -ln_x.exp();
        let ln_far = self.half_dof * ln_x - self.ln_half_dof_beta();
        let (smaller, ln_smaller) = if ln_far <= -LN_2 {
            (Tail::Lower, ln_far)
        } else {
            (Tail::Upper, ln_one_minus_exp(ln_far))
        };
        Tails {
            ln_prefactor: self.half_dof * ln_x + 0.5 * ln_y - self.ln_beta,
            smaller,
            ln_smaller,
        }
    }

    /// `|t|` with `P(T < -|t|) = p`, for `p` in `[0, 1/2]`.
    fn quantile_magnitude(&self, p: f64) -> Magnitude {
        if p == 0.0 {
            return Magnitude::Value(f64::INFINITY);
        }
        if p == 0.5 {
            return Magnitude::Value(0.0);
        }
        if self.dof == 1.0 {
            Magnitude::Value(-cauchy::standard_quantile(p))
        } else if self.dof == 2.0 {
            Magnitude::Value((1.0 - 2.0 * p) / (2.0 * p * (1.0 - p)).sqrt())
        } else if self.dof == 4.0 {
            Magnitude::Value(quartic_quantile_magnitude(p))
        } else {
            let ln_t = self.ln_quantile_magnitude(p);
            let t = ln_t.exp();
            if t.is_normal() {
                Magnitude::Value(self.refine(t, p.ln()))
            } else {
                Magnitude::Log(ln_t)
            }
        }
    }

    /// One Newton step on `ln P` in `t` itself: the solve in `ln t` resolves `t` only to `eps |ln t|` relative.
    fn refine(&self, t: f64, ln_p: f64) -> f64 {
        let ratio = self.ratio_at(t, 1.0, 0.0);
        let ln_far = self.tails(ratio).ln_probability(Tail::Lower) - LN_2;
        let refined = t + (ln_far - ln_p) * (ln_far - self.ln_density(ratio)).exp();
        if refined.is_finite() {
            refined
        } else {
            t
        }
    }

    /// `ln|t|` with `P(T < -|t|) = p`, for `p` in `(0, 1/2)`: Newton on `ln P` in `ln|t|`, every proposal clamped into a
    /// bisection bracket.
    ///
    /// The bracket is exact. With `x = dof / (dof + t^2)` and `a = dof / 2`, the hypergeometric series of the far
    /// tail gives `x^a / (a B) <= I_x(a, 1/2) <= x^a / (a B) (1 - x)^(-1/2)`. The left bound reaching `2p` puts `|t|` at
    /// least at `t_g`; past `t_g`, `(1 - x)^(-1/2)` is at most its value at `t_g`, which puts `|t|` at most at `t_h`.
    /// Deep in the tail the two meet, and `t_g` is the seed; elsewhere the seed is the Cornish-Fisher expansion
    /// (Abramowitz and Stegun 26.7.5).
    fn ln_quantile_magnitude(&self, p: f64) -> f64 {
        let ln_p = p.ln();
        let a = self.half_dof;
        let ln_x_lower = (LN_2 + ln_p + self.ln_half_dof_beta()) / a;
        let (mut lo, mut hi) = if ln_x_lower < 0.0 {
            let ln_y_lower = ln_one_minus_exp(ln_x_lower);
            let ln_x_upper = ln_x_lower + 0.5 * ln_y_lower / a;
            (self.ln_t_at_ln_x(ln_x_lower), self.ln_t_at_ln_x(ln_x_upper))
        } else {
            (MIN_LN_QUANTILE, MAX_LN_QUANTILE)
        };
        if lo > MAX_LN_QUANTILE {
            return f64::INFINITY;
        }
        lo -= BRACKET_SLACK * lo.abs().max(1.0);
        hi = (hi + BRACKET_SLACK * hi.abs().max(1.0)).min(MAX_LN_QUANTILE);
        bracketed_newton(self.cornish_fisher(p).ln(), lo, hi, |u| {
            let ratio = self.ratio_at_ln_t(u);
            let ln_far = self.tails(ratio).ln_probability(Tail::Lower) - LN_2;
            (ln_p - ln_far, (self.ln_density(ratio) + u - ln_far).exp())
        })
    }

    /// `ln|t|` at `ln x`: `t^2 = dof (1 - x) / x`.
    fn ln_t_at_ln_x(&self, ln_x: f64) -> f64 {
        0.5 * (self.ln_dof - ln_x + ln_one_minus_exp(ln_x))
    }

    /// The Cornish-Fisher `|t|` at the normal quantile of `p`, through the `dof^-4` term.
    fn cornish_fisher(&self, p: f64) -> f64 {
        let z = SQRT_2 * erfc_inv(2.0 * p);
        let z2 = z * z;
        let terms = [
            z,
            z * polynomial(&[1.0, 1.0], z2) / 4.0,
            z * polynomial(&[3.0, 16.0, 5.0], z2) / 96.0,
            z * polynomial(&[-15.0, 17.0, 19.0, 3.0], z2) / 384.0,
            z * polynomial(&[-945.0, -1920.0, 1482.0, 776.0, 79.0], z2) / 92160.0,
        ];
        polynomial(&terms, self.dof.recip())
    }
}

/// The bracket where no bound holds: `|t|` from `e^-800`, below the quantile at the largest `p` short of `1/2` for any
/// `dof` up to `f64::MAX`, to `e^1500`, which `scale * |t|` overflows at every positive `scale`.
const MIN_LN_QUANTILE: f64 = -800.0;
const MAX_LN_QUANTILE: f64 = 1500.0;

/// Relative widening of the exact bracket, against the rounding of the bounds themselves.
const BRACKET_SLACK: f64 = 1e-9;

/// `ln(1 - exp(l))` for `l <= 0`, relatively exact on both sides of `l = -ln 2`.
fn ln_one_minus_exp(l: f64) -> f64 {
    if l < -LN_2 {
        (-l.exp()).ln_1p()
    } else {
        ln_abs_expm1(l)
    }
}

/// The quantile at `dof = 4`, the root of a quartic: with `p (1 - p) = c^2 / 4` and `1 - 2p = s`,
/// `theta = atan2(s, c)`, `|t|^2 = 8 sin(2 theta / 3) sin(theta / 3) / c`. The textbook
/// `2 sqrt(cos(acos(c) / 3) / c - 1)` cancels in the difference and in `acos` near the center.
fn quartic_quantile_magnitude(p: f64) -> f64 {
    let c = 2.0 * (p * (1.0 - p)).sqrt();
    let theta = (1.0 - 2.0 * p).atan2(c);
    2.0 * (2.0 * (2.0 * theta / 3.0).sin() * (theta / 3.0).sin() / c).sqrt()
}

/// A standard quantile's magnitude, or its logarithm where `|t|` overflows while `scale |t|` may not.
enum Magnitude {
    Value(f64),
    Log(f64),
}

impl Magnitude {
    fn scaled(self, scale: f64) -> f64 {
        match self {
            Self::Value(t) => scale * t,
            Self::Log(ln_t) => scale_exp(scale, ln_t),
        }
    }
}

/// `StudentsT(dof, loc, scale)`: the standard distribution read at `t = (x - loc) / scale`, which is never formed
/// before the ratio `t^2 / dof` needs it.
struct ScaledT {
    standard: StandardT,
    loc: f64,
    scale: f64,
    ln_scale: f64,
}

impl ScaledT {
    fn new(dof: f64, loc: f64, scale: f64) -> Self {
        Self {
            standard: StandardT::new(dof),
            loc,
            scale,
            ln_scale: scale.ln(),
        }
    }

    fn tails_at(&self, distance: f64) -> Tails {
        let ratio = self.standard.ratio_at(distance, self.scale, self.ln_scale);
        self.standard.tails(ratio)
    }

    /// `P(X - loc <= distance)`: `x - loc` gives the cdf at `x`, `loc - x` the sf, by the symmetry.
    fn lower_tail(&self, distance: f64) -> f64 {
        let tails = self.tails_at(distance);
        if distance < 0.0 {
            0.5 * tails.probability(Tail::Lower)
        } else {
            0.5 + 0.5 * tails.probability(Tail::Upper)
        }
    }

    /// The log of [`Self::lower_tail`]. On the near side it is `ln(1 - I_x / 2)` or `ln((1 + I_y) / 2)`, whichever
    /// incomplete beta is the smaller, so neither rounds a small answer away.
    fn ln_lower_tail(&self, distance: f64) -> f64 {
        let tails = self.tails_at(distance);
        if distance < 0.0 {
            return tails.ln_probability(Tail::Lower) - LN_2;
        }
        let smaller = tails.ln_smaller.exp();
        match tails.smaller {
            Tail::Lower => (-0.5 * smaller).ln_1p(),
            Tail::Upper => smaller.ln_1p() - LN_2,
        }
    }

    /// `ppf(q) - loc`, solved on `min(q, 1 - q)` (exact by Sterbenz past `1/2`); `isf(q) - loc` is its negation.
    fn quantile_offset(&self, q: f64) -> f64 {
        if q <= 0.5 {
            -self.standard.quantile_magnitude(q).scaled(self.scale)
        } else {
            self.standard.quantile_magnitude(1.0 - q).scaled(self.scale)
        }
    }

    fn ln_density_at(&self, distance: f64) -> f64 {
        let ratio = self.standard.ratio_at(distance, self.scale, self.ln_scale);
        self.standard.ln_density(ratio) - self.ln_scale
    }
}

fn ln_pdf_value(dist: &ScaledT, v: f64) -> Option<f64> {
    Some(dist.ln_density_at(v - dist.loc))
}

/// `exp` of the log density, as `Gamma`'s: `(1 + t^2 / dof)^(-(dof + 1) / 2)` over- or underflows on its own where
/// the density is ordinary.
fn pdf_value(dist: &ScaledT, v: f64) -> Option<f64> {
    ln_pdf_value(dist, v).map(f64::exp)
}

fn cdf_value(dist: &ScaledT, v: f64) -> Option<f64> {
    Some(dist.lower_tail(v - dist.loc))
}

fn ln_cdf_value(dist: &ScaledT, v: f64) -> Option<f64> {
    Some(dist.ln_lower_tail(v - dist.loc))
}

fn sf_value(dist: &ScaledT, v: f64) -> Option<f64> {
    Some(dist.lower_tail(dist.loc - v))
}

fn ln_sf_value(dist: &ScaledT, v: f64) -> Option<f64> {
    Some(dist.ln_lower_tail(dist.loc - v))
}

fn ppf_value(dist: &ScaledT, q: f64) -> Option<f64> {
    on_unit_interval(&|q| dist.loc + dist.quantile_offset(q), q)
}

/// `isf(q) = ppf(1 - q)` through the symmetry `2 loc - ppf(q)`, solved on `q` itself.
fn isf_value(dist: &ScaledT, q: f64) -> Option<f64> {
    on_unit_interval(&|q| dist.loc - dist.quantile_offset(q), q)
}

fn value_keyed(
    inputs: &[Series],
    select: impl Fn(&ScaledT, f64) -> Option<f64>,
) -> PolarsResult<Series> {
    value_keyed_quaternary(
        inputs,
        check_params,
        |dof, loc, scale| Ok(ScaledT::new(dof, loc, scale)),
        select,
    )
}

#[polars_expr(output_type=Float64)]
fn studentst_pdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, pdf_value)
}

#[polars_expr(output_type=Float64)]
fn studentst_ln_pdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, ln_pdf_value)
}

#[polars_expr(output_type=Float64)]
fn studentst_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, cdf_value)
}

#[polars_expr(output_type=Float64)]
fn studentst_ln_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, ln_cdf_value)
}

#[polars_expr(output_type=Float64)]
fn studentst_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, sf_value)
}

#[polars_expr(output_type=Float64)]
fn studentst_ln_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, ln_sf_value)
}

#[polars_expr(output_type=Float64)]
fn studentst_ppf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, ppf_value)
}

#[polars_expr(output_type=Float64)]
fn studentst_isf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, isf_value)
}

#[polars_expr(output_type=Float64)]
fn studentst_pdf_scalar(inputs: &[Series], kwargs: StudentsTParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], pdf_value)
}

#[polars_expr(output_type=Float64)]
fn studentst_ln_pdf_scalar(inputs: &[Series], kwargs: StudentsTParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], ln_pdf_value)
}

#[polars_expr(output_type=Float64)]
fn studentst_cdf_scalar(inputs: &[Series], kwargs: StudentsTParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], cdf_value)
}

#[polars_expr(output_type=Float64)]
fn studentst_ln_cdf_scalar(inputs: &[Series], kwargs: StudentsTParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], ln_cdf_value)
}

#[polars_expr(output_type=Float64)]
fn studentst_sf_scalar(inputs: &[Series], kwargs: StudentsTParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], sf_value)
}

#[polars_expr(output_type=Float64)]
fn studentst_ln_sf_scalar(inputs: &[Series], kwargs: StudentsTParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], ln_sf_value)
}

#[polars_expr(output_type=Float64)]
fn studentst_ppf_scalar(inputs: &[Series], kwargs: StudentsTParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], ppf_value)
}

#[polars_expr(output_type=Float64)]
fn studentst_isf_scalar(inputs: &[Series], kwargs: StudentsTParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], isf_value)
}

/// The validated `scale`, which the Python moments gate on.
#[polars_expr(output_type=Float64)]
fn studentst_scale(inputs: &[Series]) -> PolarsResult<Series> {
    validated_triple(inputs, check_params)
}

/// From this `dof` the standard entropy is its asymptotic series: the digamma difference in the direct form is
/// `~1 / dof` and cancels digits that grow with `dof`. The 13 terms below are good to `6.4e-17` relative here.
const ENTROPY_SERIES_FROM: f64 = 20.0;

/// The coefficients `c_j` of `dof^-j`, `j = 1..=13`, in the standard entropy less `ln(2 pi e) / 2`.
const ENTROPY_SERIES: [f64; 13] = [
    1.0,
    0.25,
    -1.0 / 6.0,
    -0.125,
    0.3,
    0.25,
    -17.0 / 14.0,
    -17.0 / 16.0,
    155.0 / 18.0,
    7.75,
    -2073.0 / 22.0,
    -691.0 / 8.0,
    38227.0 / 26.0,
];

/// The standard distribution's differential entropy (nats),
/// `(dof + 1) / 2 (psi((dof + 1) / 2) - psi(dof / 2)) + ln(sqrt(dof) B(dof / 2, 1 / 2))`.
fn standard_entropy(dof: f64) -> f64 {
    if dof < ENTROPY_SERIES_FROM {
        let standard = StandardT::new(dof);
        return standard.half_dof_plus_half
            * (digamma(standard.half_dof_plus_half) - digamma(standard.half_dof))
            - standard.ln_peak;
    }
    let inverse_dof = dof.recip();
    0.5 * LN_2PI_E + polynomial(&ENTROPY_SERIES, inverse_dof) * inverse_dof
}

/// The standard distribution's entropy; Python adds `ln(scale)`. Digamma has no Polars expression to move to.
#[polars_expr(output_type=Float64)]
fn studentst_entropy(inputs: &[Series]) -> PolarsResult<Series> {
    let dof = coerce_f64(&inputs[0])?;
    DOF.check_column(&dof)?;
    let out: Float64Chunked = unary_elementwise(&dof, |row| row.map(standard_entropy));
    Ok(out.into_series())
}

/// statrs' draw: `Gamma(dof / 2, 1 / 2)`, then a normal of scale `scale sqrt(dof / g)` (Devroye). At a tiny `dof`
/// the gamma draw underflows to `0` and the draw is `+-inf`, as scipy's.
#[inline]
fn draw(dist: &StudentsT, rng: &mut impl rand::Rng) -> f64 {
    RandDistribution::sample(dist, rng)
}

#[polars_expr(output_type=Float64)]
fn studentst_sample(inputs: &[Series], kwargs: SampleKwargs) -> PolarsResult<Series> {
    sample_per_row_quaternary(inputs, kwargs, check_params, build_dist, draw)
}

#[polars_expr(output_type=Float64)]
fn studentst_sample_scalar(
    inputs: &[Series],
    kwargs: SampleScalarKwargs<StudentsTParams>,
) -> PolarsResult<Series> {
    let dist = kwargs.params.build()?;
    sample_by_index(&inputs[0], kwargs.seed, |rng| draw(&dist, rng))
}

#[polars_expr(output_type_func_with_kwargs=samples_f64_output)]
fn studentst_samples_scalar(
    inputs: &[Series],
    kwargs: SamplesScalarKwargs<StudentsTParams>,
) -> PolarsResult<Series> {
    let dist = kwargs.params.build()?;
    samples_by_index(&inputs[0], kwargs.seed, kwargs.size, |rng| draw(&dist, rng))
}

#[polars_expr(output_type_func_with_kwargs=samples_f64_output)]
fn studentst_samples(inputs: &[Series], kwargs: SamplesKwargs) -> PolarsResult<Series> {
    samples_per_row_quaternary(inputs, kwargs, check_params, build_dist, draw)
}
