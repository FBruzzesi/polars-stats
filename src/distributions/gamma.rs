//! `Gamma(shape, rate)` is the standard gamma of the same shape read at `t = rate x`, so every
//! value-keyed method is the regularized incomplete gamma `P(shape, t)`, its complement `Q`, or an
//! inverse of one of them. All of them are evaluated here in log space, not through statrs'
//! `gamma_lr` / `gamma_ur`: their prefactor `shape ln t - t - ln_gamma(shape)` is `2.2e-7` relative
//! off at `shape = 1e8`, their `1 - P` leaves the upper tail `1.5e-7` off at `shape = 1e-8`, and
//! they return `0` once the prefactor drops below `e^-709.78`, where the answer is still representable.

use std::f64::consts::{FRAC_2_SQRT_PI, LN_2, SQRT_2, TAU};

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distr::Distribution as RandDistribution;
use statrs::distribution::Gamma;
use statrs::function::erf::erfc_inv;
use statrs::function::gamma::{digamma, ln_gamma};

use crate::distributions::{
    coerce_f64, expm1, on_unit_interval, param_keyed, polynomial, validated_pair,
    value_keyed_derived_ternary, value_keyed_scalar, ParamDomain,
};
use crate::rng::{
    sample_by_index, sample_per_row_ternary, samples_by_index, samples_f64_output,
    samples_per_row_ternary, SampleKwargs, SampleScalarKwargs, SamplesKwargs, SamplesScalarKwargs,
};

const SHAPE: ParamDomain = ParamDomain::positive("shape");
const RATE: ParamDomain = ParamDomain::positive("rate");

fn check_params(shape: &Float64Chunked, rate: &Float64Chunked) -> PolarsResult<()> {
    SHAPE.check_column(shape)?;
    RATE.check_column(rate)
}

fn build_dist(shape: f64, rate: f64) -> PolarsResult<Gamma> {
    Gamma::new(shape, rate).map_err(|e| polars_err!(ComputeError: "{e}"))
}

/// Constant parameters, deserialised once per call. Every `_scalar` twin checks them once here.
#[derive(serde::Deserialize)]
struct GammaParams {
    shape: f64,
    rate: f64,
}

impl GammaParams {
    fn check(&self) -> PolarsResult<()> {
        SHAPE.check(self.shape)?;
        RATE.check(self.rate)
    }

    fn build(&self) -> PolarsResult<Gamma> {
        self.check()?;
        build_dist(self.shape, self.rate)
    }

    fn value_keyed(
        &self,
        value: &Series,
        select: impl Fn(&ScaledGamma, f64) -> Option<f64>,
    ) -> PolarsResult<Series> {
        self.check()?;
        value_keyed_scalar(value, &ScaledGamma::new(self.shape, self.rate), select)
    }
}

const EULER: f64 = 0.5772156649015329;
const LN_2PI: f64 = 1.8378770664093453;

/// Below this `|a|` the Taylor series of `ln Gamma(1 + a)` keeps full relative precision, where
/// statrs' `ln_gamma(1.0 + a)` rounds `1 + a` first and is off by `~1e-16` absolute.
const LN_GAMMA_1P_SERIES_BELOW: f64 = 0.2;

/// `(-1)^k zeta(k) / k` for `k = 2..=24`: the Taylor coefficients of `ln Gamma(1 + a)` past `-euler a`,
/// enough for `2^-56` relative at `|a| = 0.2`.
const LN_GAMMA_1P_SERIES: [f64; 23] = [
    0.8224670334241132,
    -0.40068563438653143,
    0.27058080842778454,
    -0.20738555102867398,
    0.1695571769974082,
    -0.1440498967688461,
    0.12550966952474304,
    -0.11133426586956469,
    0.1000994575127818,
    -0.09095401714582904,
    0.083353840546109,
    -0.0769325164113522,
    0.07143294629536133,
    -0.06666870588242046,
    0.06250095514121304,
    -0.058823978658684585,
    0.055555767627403614,
    -0.05263167937961666,
    0.05000004769810169,
    -0.047619070330142226,
    0.04545455629320467,
    -0.04347826605304026,
    0.04166666915034121,
];

/// `ln Gamma(1 + a)` below [`LN_GAMMA_1P_SERIES_BELOW`], relatively exact as `a -> 0`.
fn ln_gamma_1p_series(a: f64) -> f64 {
    a * (-EULER + a * polynomial(&LN_GAMMA_1P_SERIES, a))
}

/// From this shape the prefactor is read off Stirling's formula, which cancels the `a ln a` terms
/// exactly instead of rounding each of them.
const STIRLING_FROM: f64 = 10.0;

/// `B_2k / (2k (2k - 1))` for `k = 1..=7`: the asymptotic series of Binet's remainder
/// `mu(a) = ln Gamma(a + 1) - (a + 1/2) ln a + a - ln(2 pi) / 2`, truncated at `3e-17` at `a = 10`.
const STIRLING_REMAINDER_SERIES: [f64; 7] = [
    0.08333333333333333,
    -0.002777777777777778,
    0.0007936507936507937,
    -0.0005952380952380953,
    0.0008417508417508417,
    -0.0019175269175269176,
    0.00641025641025641,
];

/// `ln(1 + s) - s`, scipy's `log1pmx`. Below `|s| = 1/2` it is
/// `-s^2 / (2 + s) + 2 (w^3 / 3 + w^5 / 5 + ...)` with `w = s / (2 + s)`, the `atanh` series of
/// `ln(1 + s)` with the linear term cancelled exactly: the literal difference keeps only `~ulp / s`
/// relative.
fn ln_1p_mx(s: f64) -> f64 {
    if s.abs() >= 0.5 {
        return s.ln_1p() - s;
    }
    let lead = -s * s / (2.0 + s);
    let w = s / (2.0 + s);
    let w_squared = w * w;
    let mut power = w * w_squared;
    let mut odd = 3.0;
    let mut sum = 0.0;
    loop {
        let term = power / odd;
        sum += term;
        if term.abs() <= INCOMPLETE_GAMMA_EPS * lead.abs() {
            return lead + 2.0 * sum;
        }
        power *= w_squared;
        odd += 2.0;
    }
}

/// Relative convergence target of the series and continued fractions below.
const INCOMPLETE_GAMMA_EPS: f64 = f64::EPSILON / 2.0;

/// Terms after which a series or continued fraction gives up and answers `NaN`, a guard no shape
/// reaches: Temme's expansion takes the bulk from [`TEMME_FROM`] on, which bounds what is left to
/// the series and the continued fraction at any shape.
const MAX_TERMS: usize = 10_000_000;

/// The Lentz floor, standing in for a vanishing partial denominator.
const LENTZ_FLOOR: f64 = 1e-300;

/// From this shape the tails within [`TEMME_WITHIN`] of the mean are Temme's uniform expansion,
/// which costs the same at any shape, where the series needs `~9 sqrt(shape)` terms at the mean.
const TEMME_FROM: f64 = 100.0;

/// The relative distance `|t - a| / a` from the mean inside which Temme's expansion is used. Past
/// it the series and the continued fraction shrink by a fixed ratio per term at any shape.
const TEMME_WITHIN: f64 = 0.3;

/// `d_{k,n}` of DLMF 8.12.12, `C_k(eta) = sum_n d_{k,n} eta^n` for `k = 0..=6` truncated at
/// `15 - 2k` powers: from [`TEMME_FROM`] on and within [`TEMME_WITHIN`] of the mean the dropped terms
/// are below `1e-17` of the tail. Computed by series reversion at 90 digits.
const TEMME_COEFFICIENTS: [&[f64]; 7] = [
    &[
        -0.3333333333333333,
        0.08333333333333333,
        -0.014814814814814815,
        0.0011574074074074073,
        0.0003527336860670194,
        -0.0001787551440329218,
        3.919263178522438e-05,
        -2.185448510679992e-06,
        -1.85406221071516e-06,
        8.296711340953087e-07,
        -1.7665952736826078e-07,
        6.707853543401498e-09,
        1.0261809784240309e-08,
        -4.382036018453353e-09,
        9.14769958223679e-10,
    ],
    &[
        -0.001851851851851852,
        -0.003472222222222222,
        0.0026455026455026454,
        -0.0009902263374485596,
        0.00020576131687242798,
        -4.018775720164609e-07,
        -1.8098550334489977e-05,
        7.64916091608111e-06,
        -1.6120900894563446e-06,
        4.647127802807434e-09,
        1.378633446915721e-07,
        -5.752545603517705e-08,
        1.1951628599778148e-08,
    ],
    &[
        0.004133597883597883,
        -0.0026813271604938273,
        0.0007716049382716049,
        2.0093878600823047e-06,
        -0.0001073665322636516,
        5.2923448829120125e-05,
        -1.2760635188618728e-05,
        3.423578734096138e-08,
        1.3721957309062934e-06,
        -6.298992138380055e-07,
        1.4280614206064242e-07,
    ],
    &[
        0.0006494341563786008,
        0.00022947209362139917,
        -0.0004691894943952557,
        0.00026772063206283885,
        -7.561801671883977e-05,
        -2.396505113867297e-07,
        1.1082654115347302e-05,
        -5.6749528269915965e-06,
        1.4230900732435883e-06,
    ],
    &[
        -0.0008618882909167117,
        0.0007840392217200666,
        -0.0002990724803031902,
        -1.4638452578843418e-06,
        6.641498215465122e-05,
        -3.968365047179435e-05,
        1.1375726970678419e-05,
    ],
    &[
        -0.00033679855336635813,
        -6.972813758365857e-05,
        0.0002772753244959392,
        -0.00019932570516188847,
        6.797780477937208e-05,
    ],
    &[
        0.0005313079364639922,
        -0.0005921664373536939,
        0.0002708782096718045,
    ],
];

/// Below this `w`, [`erfcx`] is its Taylor polynomial about `w = 1`.
const ERFCX_TAYLOR_BELOW: f64 = 2.0;

/// The Taylor coefficients of `erfcx` about `w = 1`, `3e-16` relative on `[0, 2]`.
const ERFCX_TAYLOR: [f64; 32] = [
    0.427583576155807,
    -0.27321201478389856,
    0.15437156137190844,
    -0.07922696894132675,
    0.037572296215290846,
    -0.016661869090414363,
    0.006970142374958827,
    -0.0027690647758444385,
    0.001050269399778597,
    -0.0003819545280146314,
    0.00013366297435279316,
    -4.514391884760696e-05,
    1.4753175917531031e-05,
    -4.675498912319374e-06,
    1.4396681436016656e-06,
    -4.314441024956944e-07,
    1.260280051382464e-07,
    -3.593130557146447e-08,
    1.0010744396309103e-08,
    -2.7284801237005647e-09,
    7.282264272608537e-10,
    -1.9050035204187724e-10,
    4.8884188656270594e-11,
    -1.2314448990052753e-11,
    3.0474783055181535e-12,
    -7.413576547627679e-13,
    1.7739389621195276e-13,
    -4.17750932259863e-14,
    9.68705735614046e-15,
    -2.2129679910238512e-15,
    4.982726243411073e-16,
    -1.1062550752791896e-16,
];

/// `erfcx(w) = e^(w^2) erfc(w)` for `w >= 0`, the scaled complementary error function: its Taylor
/// polynomial below [`ERFCX_TAYLOR_BELOW`], and above it Laplace's continued fraction
/// `1 / (sqrt(pi) (w + (1/2) / (w + 1 / (w + (3/2) / (w + ...)))))` evaluated backward from
/// `200 / w^2 + 10` levels, which holds `3e-16` from `w = 2` on. Both stay relative where `erfc(w)`
/// cancels against `1` or `e^(-w^2)` underflows, and neither calls statrs' `erfc`.
fn erfcx(w: f64) -> f64 {
    if w < ERFCX_TAYLOR_BELOW {
        return polynomial(&ERFCX_TAYLOR, w - 1.0);
    }
    let depth = (200.0 / (w * w)).ceil() as usize + 10;
    let fraction = (1..=depth)
        .rev()
        .fold(w, |fraction, level| w + 0.5 * level as f64 / fraction);
    0.5 * FRAC_2_SQRT_PI / fraction
}

/// Which regularized incomplete gamma: `P(a, t)` below `t`, or `Q(a, t) = 1 - P(a, t)` above it.
#[derive(Clone, Copy, PartialEq)]
enum Tail {
    Lower,
    Upper,
}

impl Tail {
    fn complement(self) -> Self {
        match self {
            Self::Lower => Self::Upper,
            Self::Upper => Self::Lower,
        }
    }
}

/// The smaller of `P` and `Q` at a point, on the log scale; the other one is its complement.
struct Tails {
    /// `ln(t^a e^-t / Gamma(a + 1))`.
    ln_prefactor: f64,
    smaller: Tail,
    ln_smaller: f64,
}

impl Tails {
    fn ln_probability(&self, tail: Tail) -> f64 {
        if self.smaller == tail {
            self.ln_smaller
        } else {
            (-self.ln_smaller.exp()).ln_1p()
        }
    }

    /// The complement is `-expm1` of the smaller tail's log only while that tail is not small: the
    /// `sinh` identity behind [`expm1`] overflows past `|ln| ~ 1420`, and below `e^-1` the plain
    /// `1 - exp` has nothing to cancel.
    fn probability(&self, tail: Tail) -> f64 {
        if self.smaller == tail {
            self.ln_smaller.exp()
        } else if self.ln_smaller > -1.0 {
            -expm1(self.ln_smaller)
        } else {
            1.0 - self.ln_smaller.exp()
        }
    }
}

/// The standard gamma of shape `a`, with every shape-only term of its incomplete gamma hoisted.
///
/// Every evaluation takes the point twice, as `t` and as `u = ln t`, so a `t` that has underflowed
/// keeps its logarithm: `a ln t` is where a small point's digits live.
struct IncompleteGamma {
    a: f64,
    ln_a: f64,
    /// `ln Gamma(a + 1)`, Stirling's formula from [`STIRLING_FROM`] on, so no shape of that size
    /// calls statrs' `ln_gamma`.
    ln_gamma_1p: f64,
    /// `ln Gamma(a)`, relatively exact across the zero at `a = 1`.
    ln_gamma: f64,
    /// `ln(2 pi a) / 2 + mu(a)`, what `ln Gamma(a + 1)` adds to `a ln a - a`, from [`STIRLING_FROM`] on.
    stirling_offset: Option<f64>,
}

impl IncompleteGamma {
    fn new(a: f64) -> Self {
        let ln_a = a.ln();
        let stirling_offset = (a >= STIRLING_FROM).then(|| {
            0.5 * (LN_2PI + ln_a) + polynomial(&STIRLING_REMAINDER_SERIES, 1.0 / (a * a)) / a
        });
        let ln_gamma_1p = match stirling_offset {
            Some(offset) => a * (ln_a - 1.0) + offset,
            None if a < LN_GAMMA_1P_SERIES_BELOW => ln_gamma_1p_series(a),
            None => ln_gamma(1.0 + a),
        };
        let ln_gamma = if (a - 1.0).abs() < LN_GAMMA_1P_SERIES_BELOW {
            ln_gamma_1p_series(a - 1.0)
        } else {
            ln_gamma_1p - ln_a
        };
        Self {
            a,
            ln_a,
            ln_gamma_1p,
            ln_gamma,
            stirling_offset,
        }
    }

    /// `ln(t^a e^-t / Gamma(a + 1))`. From [`STIRLING_FROM`] on it is
    /// `a (ln(t / a) - t / a + 1) - ln(2 pi a) / 2 - mu(a)`, which has no `a ln a` to cancel: near the
    /// mean the bracket is [`ln_1p_mx`] of the exact excess `(t - a) / a`.
    fn ln_prefactor(&self, t: f64, u: f64) -> f64 {
        let a = self.a;
        let Some(stirling_offset) = self.stirling_offset else {
            return a * u - t - self.ln_gamma_1p;
        };
        let bracket = if t < 0.5 * a {
            let ratio = t / a;
            let ln_ratio = if ratio.is_normal() {
                ratio.ln()
            } else {
                u - self.ln_a
            };
            a * ln_ratio + (a - t)
        } else {
            a * ln_1p_mx((t - a) / a)
        };
        bracket - stirling_offset
    }

    /// `ln(t^(a - 1) e^-t / Gamma(a))`, the log density. Below [`STIRLING_FROM`] it is summed term by
    /// term so a vanishing `(a - 1) u` leaves `-t` intact, where `ln_prefactor - u` would round it away.
    /// Below [`LN_GAMMA_1P_SERIES_BELOW`] `ln Gamma(a) ~ -ln a` would cancel `-u` near `t = a`, so the
    /// pair is read as `ln(a / t)` instead, from the Sterbenz-exact `a - t` within a factor of two.
    fn ln_density(&self, t: f64, u: f64) -> f64 {
        let a = self.a;
        if self.stirling_offset.is_some() {
            return self.ln_prefactor(t, u) + self.ln_a - u;
        }
        if a >= LN_GAMMA_1P_SERIES_BELOW {
            return (a - 1.0) * u - t - self.ln_gamma;
        }
        let ratio = a / t;
        let ln_ratio = if (0.5..=2.0).contains(&ratio) {
            ((a - t) / t).ln_1p()
        } else if ratio.is_normal() {
            ratio.ln()
        } else {
            self.ln_a - u
        };
        a * u + ln_ratio - t - self.ln_gamma_1p
    }

    /// Both tails at `t > 0` finite, by the region split of Cephes' `igamc`: the power series for `P`
    /// below the mean, Legendre's continued fraction for `Q` above it, and DLMF 8.7.3 for `Q` where
    /// `t` is small and so is the shape. From [`TEMME_FROM`] on the bulk is Temme's expansion instead.
    fn tails(&self, t: f64, u: f64) -> Tails {
        let a = self.a;
        let temme_excess = (a >= TEMME_FROM)
            .then(|| (t - a) / a)
            .filter(|excess| excess.abs() < TEMME_WITHIN);
        if let (Some(excess), Some(stirling_offset)) = (temme_excess, self.stirling_offset) {
            return self.temme(excess, stirling_offset);
        }
        let ln_prefactor = self.ln_prefactor(t, u);
        let lower_is_smaller = if t > 1.1 {
            t < a
        } else if t <= 0.5 {
            a * u < -0.4
        } else {
            1.1 * t < a
        };
        let (smaller, ln_smaller) = if lower_is_smaller {
            (Tail::Lower, ln_prefactor + self.ln_lower_series(t))
        } else if t > 1.1 {
            let ln_q = ln_prefactor + self.ln_a - u - self.ln_upper_fraction(t);
            (Tail::Upper, ln_q)
        } else {
            (Tail::Upper, self.upper_small_argument(t, u).ln())
        };
        Tails {
            ln_prefactor,
            smaller,
            ln_smaller,
        }
    }

    /// Both tails at `t = a (1 + excess)` by Temme's uniform expansion, DLMF 8.12.3 and 8.12.4: with
    /// `eta = sign(excess) sqrt(-2 ln_1p_mx(excess))`, `w^2 = a eta^2 / 2` and `S = sum_k C_k(eta) a^-k`,
    /// `Q` and `P` are `e^(-w^2) (erfcx(w) / 2 +- S / sqrt(2 pi a))`. The `e^(-w^2)` both terms carry
    /// is factored out and added on the log scale, and the bracket cannot cancel: it is
    /// `1 / (|excess| sqrt(2 pi a))` to leading order. The prefactor is `-w^2` less the Stirling
    /// offset, bit for bit the `a ln_1p_mx(excess)` bracket [`Self::ln_prefactor`] reads there.
    fn temme(&self, excess: f64, stirling_offset: f64) -> Tails {
        let a = self.a;
        let half_eta_squared = -ln_1p_mx(excess);
        let eta = (2.0 * half_eta_squared).sqrt().copysign(excess);
        let w_squared = a * half_eta_squared;
        let series = polynomial(&TEMME_COEFFICIENTS.map(|c| polynomial(c, eta)), a.recip());
        let remainder = series / (TAU * a).sqrt();
        let half_erfcx = 0.5 * erfcx(w_squared.sqrt());
        let (smaller, bracket) = if excess < 0.0 {
            (Tail::Lower, half_erfcx - remainder)
        } else {
            (Tail::Upper, half_erfcx + remainder)
        };
        Tails {
            ln_prefactor: -w_squared - stirling_offset,
            smaller,
            ln_smaller: bracket.ln() - w_squared,
        }
    }

    /// `ln(1 + t / (a + 1) + t^2 / ((a + 1)(a + 2)) + ...)` for `t < a`, where the terms fall, so that
    /// `P = prefactor * sum`. It stops once the geometric bound on the rest, `term * r / (1 - r)` with
    /// `r = t / (a + n + 1)`, is below the target: near the mean the terms shrink so slowly that
    /// stopping on the last term alone leaves `1e-13`.
    fn ln_lower_series(&self, t: f64) -> f64 {
        let mut term = 1.0;
        let mut sum = 1.0;
        let mut denominator = self.a;
        for _ in 0..MAX_TERMS {
            denominator += 1.0;
            term *= t / denominator;
            sum += term;
            if term * t <= INCOMPLETE_GAMMA_EPS * sum * (denominator + 1.0 - t) {
                return sum.ln();
            }
        }
        f64::NAN
    }

    /// `ln(f / t)` for `t >= a`, with Legendre's continued fraction `Gamma(a, t) = t^a e^-t / f`,
    /// `f = t + 1 - a - 1 (1 - a) / (t + 3 - a - 2 (2 - a) / (t + 5 - a - ...))`, by modified Lentz on
    /// the fraction divided through by `t` at every level: `f` itself overflows its partial
    /// denominators long before `Q` stops being representable.
    fn ln_upper_fraction(&self, t: f64) -> f64 {
        let a = self.a;
        let excess = t - a;
        let mut value = ((excess + 1.0) / t).max(LENTZ_FLOOR);
        let mut numerator_side = value;
        let mut denominator_side = 0.0;
        for n in 1..MAX_TERMS {
            let i = n as f64;
            let partial_numerator = -i * (i - a) / t / t;
            let partial_denominator = (excess + (2.0 * i + 1.0)) / t;
            denominator_side = partial_denominator + partial_numerator * denominator_side;
            if denominator_side.abs() < LENTZ_FLOOR {
                denominator_side = LENTZ_FLOOR;
            }
            denominator_side = denominator_side.recip();
            numerator_side = partial_denominator + partial_numerator / numerator_side;
            if numerator_side.abs() < LENTZ_FLOOR {
                numerator_side = LENTZ_FLOOR;
            }
            let delta = numerator_side * denominator_side;
            value *= delta;
            if (delta - 1.0).abs() <= INCOMPLETE_GAMMA_EPS {
                return value.ln();
            }
        }
        f64::NAN
    }

    /// `Q(a, t)` for `t <= 1.1` on the side where it is small, DLMF 8.7.3:
    /// `1 - t^a / Gamma(a + 1) - (t^a / Gamma(a)) sum_{n >= 1} (-t)^n / (n! (a + n))`, with the
    /// leading difference as `-expm1` of an exact log. At a small shape `P = 1 - Q` rounds to `1`, so
    /// `Q` must never be formed as the complement there.
    fn upper_small_argument(&self, t: f64, u: f64) -> f64 {
        let a = self.a;
        let mut numerator = 1.0;
        let mut sum = 0.0;
        for n in 1..MAX_TERMS {
            let i = n as f64;
            numerator *= -t / i;
            let term = numerator / (a + i);
            sum += term;
            if term.abs() <= INCOMPLETE_GAMMA_EPS * sum.abs() {
                break;
            }
        }
        let ln_power = a * u - self.ln_gamma_1p;
        -expm1(ln_power) - a * ln_power.exp() * sum
    }

    /// The `u = ln t` with `ln(tail(e^u)) = ln_target`, `ln_target <= ln(1/2)`: Newton on the log of
    /// the tail as a function of `u`, guarded by a bracket that starts finite and bisects any step
    /// leaving it.
    ///
    /// In `u` both tails are concave, since the log of a gamma variate has a log-concave density, so
    /// from the first iterate on Newton approaches the root from one side and the bracket only
    /// catches rounding. The bracket's ends are bounds that hold for every shape: `P <= t^a /
    /// Gamma(a + 1)` below, the median below the mean `a` for `P`, and Chernoff's
    /// `Q <= 2^a e^(-t / 2)` above.
    fn ln_quantile(&self, tail: Tail, ln_target: f64) -> f64 {
        let a = self.a;
        let (mut lo, mut hi, seed) = match tail {
            Tail::Lower => {
                let power_bound = (ln_target + self.ln_gamma_1p) / a;
                (
                    power_bound,
                    self.ln_a,
                    self.seed_lower(ln_target, power_bound),
                )
            },
            Tail::Upper => {
                let power_bound = ((-ln_target.exp()).ln_1p() + self.ln_gamma_1p) / a;
                let chernoff_bound = (2.0 * (a * LN_2 - ln_target)).ln();
                (
                    power_bound,
                    chernoff_bound,
                    self.seed_upper(ln_target, power_bound),
                )
            },
        };
        let mut u = seed.max(lo).min(hi);
        let mut previous_step: f64 = 0.0;
        for _ in 0..MAX_NEWTON_STEPS {
            let tails = self.tails(u.exp(), u);
            let ln_value = tails.ln_probability(tail);
            let residual = ln_value - ln_target;
            if residual.is_nan() {
                return f64::NAN;
            }
            if residual == 0.0 {
                return u;
            }
            let slope = (self.ln_a + tails.ln_prefactor - ln_value).exp();
            let (slope, below_root) = match tail {
                Tail::Lower => (slope, residual < 0.0),
                Tail::Upper => (-slope, residual > 0.0),
            };
            if below_root {
                lo = u;
            } else {
                hi = u;
            }
            let newton = u - residual / slope;
            let step = (newton - u).abs();
            let tolerance = f64::EPSILON * u.abs().max(1.0);
            if step <= tolerance {
                return newton;
            }
            if hi - lo <= 4.0 * tolerance {
                return u;
            }
            if newton <= lo || newton >= hi {
                u = 0.5 * (lo + hi);
                previous_step = 0.0;
                continue;
            }
            if step.powi(3) <= SETTLED_STEP_FRACTION * tolerance * previous_step.powi(2) {
                return newton;
            }
            u = newton;
            previous_step = step;
        }
        f64::NAN
    }

    /// Wilson-Hilferty, `t ~ a (1 - 1 / (9a) + z / (3 sqrt a))^3` at the normal quantile `z`, where the
    /// cube is positive and the shape is not small enough for the mass to pile up at `0`.
    fn wilson_hilferty(&self, z: f64) -> Option<f64> {
        let a = self.a;
        let cube_root = 1.0 - 1.0 / (9.0 * a) + z / (3.0 * a.sqrt());
        (a >= 1.0 && cube_root > 0.0 && z.is_finite()).then(|| self.ln_a + 3.0 * cube_root.ln())
    }

    /// `power_bound`, where the series' leading term `t^a / Gamma(a + 1)` is `p`, while that term
    /// dominates; Wilson-Hilferty in the bulk.
    fn seed_lower(&self, ln_p: f64, power_bound: f64) -> f64 {
        if self.a < 1.0 || power_bound.exp() < 0.2 * (self.a + 1.0) {
            return power_bound;
        }
        self.wilson_hilferty(-SQRT_2 * erfc_inv(2.0 * ln_p.exp()))
            .unwrap_or(power_bound)
    }

    /// Wilson-Hilferty in the bulk, `t ~ -ln q + (a - 1) ln(-ln q) - ln Gamma(a)` from `Q`'s leading
    /// asymptotic term in the far tail, and `power_bound`, the lower tail's leading term at the
    /// complement, otherwise.
    fn seed_upper(&self, ln_q: f64, power_bound: f64) -> f64 {
        let a = self.a;
        if let Some(u) = self.wilson_hilferty(SQRT_2 * erfc_inv(2.0 * ln_q.exp())) {
            return u;
        }
        let far = -ln_q + (a - 1.0) * (-ln_q).ln() - self.ln_gamma;
        if far > a.max(1.0) {
            far.ln()
        } else {
            power_bound
        }
    }
}

/// The inverse's step budget, past which it answers `NaN` rather than an unconverged iterate. Only a
/// pathological seed could spend it on bisection: from these seeds Newton settles in a handful of
/// steps across `shape` in `1e-8` to `1e8`.
const MAX_NEWTON_STEPS: usize = 200;

/// Two Newton steps in a row, `p` then `s`, put the next one at about `s^3 / p^2`. Once that is below
/// this fraction of the tolerance the iterate is returned without the evaluation that would only
/// confirm it; after a bisection there is no `p` and the iteration runs on.
const SETTLED_STEP_FRACTION: f64 = 1.0 / 1024.0;

/// `Gamma(shape, rate)`: the standard gamma's incomplete gamma, read at `t = rate x`.
struct ScaledGamma {
    standard: IncompleteGamma,
    rate: f64,
    ln_rate: f64,
}

impl ScaledGamma {
    fn new(shape: f64, rate: f64) -> Self {
        Self {
            standard: IncompleteGamma::new(shape),
            rate,
            ln_rate: rate.ln(),
        }
    }

    /// `(t, ln t)` for `x > 0`; `ln t` is `ln rate + ln x` where `t` has left the normal range, so an
    /// underflowed product keeps its logarithm.
    fn standard_point(&self, x: f64) -> (f64, f64) {
        let t = self.rate * x;
        let u = if t.is_normal() {
            t.ln()
        } else {
            self.ln_rate + x.ln()
        };
        (t, u)
    }

    /// Both tails at `x > 0`, or `None` where `rate x` overflows, which is the whole upper tail as far
    /// as `f64` can say.
    fn tails_at(&self, x: f64) -> Option<Tails> {
        let (t, u) = self.standard_point(x);
        t.is_finite().then(|| self.standard.tails(t, u))
    }

    /// The `x` where `tail` is `q`, for `q` in `[0, 1]`. Past `q = 1/2` it solves the complement at
    /// `ln(1 - q)`, exact by Sterbenz, so the solve always targets the smaller tail, and a target of
    /// `ln 0` is the support edge. The point is `exp(u) / rate` as `(exp(u / 2) / rate) exp(u / 2)`:
    /// `exp(u)` alone is subnormal or infinite where the quantile under a small or large `rate` is not.
    fn quantile(&self, tail: Tail, q: f64) -> f64 {
        let (tail, ln_target) = if q <= 0.5 {
            (tail, q.ln())
        } else {
            (tail.complement(), (-q).ln_1p())
        };
        if ln_target == f64::NEG_INFINITY {
            return if tail == Tail::Lower {
                0.0
            } else {
                f64::INFINITY
            };
        }
        let half = (self.standard.ln_quantile(tail, ln_target) / 2.0).exp();
        (half / self.rate) * half
    }

    /// The density at `x = 0`: infinite below `shape = 1`, `rate` at it, `0` above it.
    fn pdf_at_origin(&self) -> f64 {
        let shape = self.standard.a;
        if shape < 1.0 {
            f64::INFINITY
        } else if shape == 1.0 {
            self.rate
        } else {
            0.0
        }
    }
}

fn ln_pdf_value(gamma: &ScaledGamma, x: f64) -> Option<f64> {
    Some(if x < 0.0 {
        f64::NEG_INFINITY
    } else if x == 0.0 {
        gamma.pdf_at_origin().ln()
    } else {
        let (t, u) = gamma.standard_point(x);
        if t.is_finite() {
            gamma.ln_rate + gamma.standard.ln_density(t, u)
        } else {
            f64::NEG_INFINITY
        }
    })
}

/// `exp` of the log density: every factor of `rate^shape x^(shape - 1) e^(-rate x) / Gamma(shape)`
/// over- or underflows on its own somewhere the density is ordinary, which is where statrs is `NaN`.
fn pdf_value(gamma: &ScaledGamma, x: f64) -> Option<f64> {
    if x == 0.0 {
        return Some(gamma.pdf_at_origin());
    }
    ln_pdf_value(gamma, x).map(f64::exp)
}

fn cdf_value(gamma: &ScaledGamma, x: f64) -> Option<f64> {
    Some(if x <= 0.0 {
        0.0
    } else {
        gamma
            .tails_at(x)
            .map_or(1.0, |tails| tails.probability(Tail::Lower))
    })
}

fn ln_cdf_value(gamma: &ScaledGamma, x: f64) -> Option<f64> {
    Some(if x <= 0.0 {
        f64::NEG_INFINITY
    } else {
        gamma
            .tails_at(x)
            .map_or(0.0, |tails| tails.ln_probability(Tail::Lower))
    })
}

fn sf_value(gamma: &ScaledGamma, x: f64) -> Option<f64> {
    Some(if x <= 0.0 {
        1.0
    } else {
        gamma
            .tails_at(x)
            .map_or(0.0, |tails| tails.probability(Tail::Upper))
    })
}

fn ln_sf_value(gamma: &ScaledGamma, x: f64) -> Option<f64> {
    Some(if x <= 0.0 {
        0.0
    } else {
        gamma
            .tails_at(x)
            .map_or(f64::NEG_INFINITY, |tails| tails.ln_probability(Tail::Upper))
    })
}

fn ppf_value(gamma: &ScaledGamma, q: f64) -> Option<f64> {
    on_unit_interval(&|q| gamma.quantile(Tail::Lower, q), q)
}

fn isf_value(gamma: &ScaledGamma, q: f64) -> Option<f64> {
    on_unit_interval(&|q| gamma.quantile(Tail::Upper, q), q)
}

#[polars_expr(output_type=Float64)]
fn gamma_pdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, ScaledGamma::new, pdf_value)
}

#[polars_expr(output_type=Float64)]
fn gamma_ln_pdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, ScaledGamma::new, ln_pdf_value)
}

#[polars_expr(output_type=Float64)]
fn gamma_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, ScaledGamma::new, cdf_value)
}

#[polars_expr(output_type=Float64)]
fn gamma_ln_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, ScaledGamma::new, ln_cdf_value)
}

#[polars_expr(output_type=Float64)]
fn gamma_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, ScaledGamma::new, sf_value)
}

#[polars_expr(output_type=Float64)]
fn gamma_ln_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, ScaledGamma::new, ln_sf_value)
}

#[polars_expr(output_type=Float64)]
fn gamma_ppf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, ScaledGamma::new, ppf_value)
}

#[polars_expr(output_type=Float64)]
fn gamma_isf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed_derived_ternary(inputs, check_params, ScaledGamma::new, isf_value)
}

#[polars_expr(output_type=Float64)]
fn gamma_pdf_scalar(inputs: &[Series], kwargs: GammaParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], pdf_value)
}

#[polars_expr(output_type=Float64)]
fn gamma_ln_pdf_scalar(inputs: &[Series], kwargs: GammaParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], ln_pdf_value)
}

#[polars_expr(output_type=Float64)]
fn gamma_cdf_scalar(inputs: &[Series], kwargs: GammaParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], cdf_value)
}

#[polars_expr(output_type=Float64)]
fn gamma_ln_cdf_scalar(inputs: &[Series], kwargs: GammaParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], ln_cdf_value)
}

#[polars_expr(output_type=Float64)]
fn gamma_sf_scalar(inputs: &[Series], kwargs: GammaParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], sf_value)
}

#[polars_expr(output_type=Float64)]
fn gamma_ln_sf_scalar(inputs: &[Series], kwargs: GammaParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], ln_sf_value)
}

#[polars_expr(output_type=Float64)]
fn gamma_ppf_scalar(inputs: &[Series], kwargs: GammaParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], ppf_value)
}

#[polars_expr(output_type=Float64)]
fn gamma_isf_scalar(inputs: &[Series], kwargs: GammaParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], isf_value)
}

/// The validated `rate`, which the Python moments gate on.
#[polars_expr(output_type=Float64)]
fn gamma_rate(inputs: &[Series]) -> PolarsResult<Series> {
    validated_pair(inputs, coerce_f64, check_params)
}

/// From this shape the entropy is its asymptotic series: statrs' `shape + ln_gamma(shape) +
/// (1 - shape) digamma(shape)` cancels terms of order `shape ln shape` down to `ln(shape) / 2`, and
/// is `2.3e-8` relative off at `shape = 1e8`. The truncated series holds `3e-16` relative here.
const ENTROPY_SERIES_FROM: f64 = 30.0;

/// The coefficients `c_j` of `a^-j`, `j = 1..=8`, in `a + ln Gamma(a) + (1 - a) psi(a) - ln(2 pi e a) / 2`:
/// `B_2k / (2k - 1)` and `-B_2k / (2k)` from the Stirling series of the two terms, the first less `1/2`.
const ENTROPY_SERIES: [f64; 8] = [
    -0.3333333333333333,
    -0.08333333333333333,
    -0.011111111111111112,
    0.008333333333333333,
    0.004761904761904762,
    -0.003968253968253968,
    -0.004761904761904762,
    0.004166666666666667,
];

const LN_2PI_E: f64 = 2.8378770664093453;

/// Differential entropy (nats), `shape - ln(rate) + ln Gamma(shape) + (1 - shape) psi(shape)`:
/// digamma has no Polars expression to move to.
#[polars_expr(output_type=Float64)]
fn gamma_entropy(inputs: &[Series]) -> PolarsResult<Series> {
    param_keyed(
        inputs,
        coerce_f64,
        coerce_f64,
        check_params,
        |shape, rate| {
            if shape < ENTROPY_SERIES_FROM {
                return Ok(shape - rate.ln() + ln_gamma(shape) + (1.0 - shape) * digamma(shape));
            }
            let inverse_shape = shape.recip();
            let correction = polynomial(&ENTROPY_SERIES, inverse_shape) * inverse_shape;
            Ok(0.5 * (LN_2PI_E + shape.ln()) + correction - rate.ln())
        },
    )
}

#[inline]
fn draw(dist: &Gamma, rng: &mut impl rand::Rng) -> f64 {
    RandDistribution::sample(dist, rng)
}

#[polars_expr(output_type=Float64)]
fn gamma_sample(inputs: &[Series], kwargs: SampleKwargs) -> PolarsResult<Series> {
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
fn gamma_sample_scalar(
    inputs: &[Series],
    kwargs: SampleScalarKwargs<GammaParams>,
) -> PolarsResult<Series> {
    let dist = kwargs.params.build()?;
    sample_by_index(&inputs[0], kwargs.seed, |rng| draw(&dist, rng))
}

#[polars_expr(output_type_func_with_kwargs=samples_f64_output)]
fn gamma_samples_scalar(
    inputs: &[Series],
    kwargs: SamplesScalarKwargs<GammaParams>,
) -> PolarsResult<Series> {
    let dist = kwargs.params.build()?;
    samples_by_index(&inputs[0], kwargs.seed, kwargs.size, |rng| draw(&dist, rng))
}

#[polars_expr(output_type_func_with_kwargs=samples_f64_output)]
fn gamma_samples(inputs: &[Series], kwargs: SamplesKwargs) -> PolarsResult<Series> {
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
