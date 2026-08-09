#![allow(clippy::unused_unit)]
use polars::prelude::arity::{try_binary_elementwise, try_ternary_elementwise};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distributions::Distribution as RandDistribution;
use statrs::distribution::{Beta, Continuous, ContinuousCDF};
use statrs::function::gamma::ln_gamma;
use statrs::statistics::Distribution as StatrsDistribution;

use crate::distributions::{param_validator, value_keyed_per_row, value_keyed_scalar_plugins};
use crate::rng::{
    sample_scalar_plugin, samples_f64_output, samples_per_row, ternary_param_rows, SampleKwargs,
    SamplesKwargs,
};

/// Construct a `statrs::Beta`, mapping the invalid-parameter case to a `ComputeError`.
///
/// `statrs::Beta::new(shape_a, shape_b)` rejects a `NaN`, infinite, or non-positive shape. That surfaces as
/// `InvalidOperation`, so an invalid shape fails the whole evaluation rather than silently nulling the row.
fn build_dist(a: f64, b: f64) -> PolarsResult<Beta> {
    Beta::new(a, b).map_err(|e| {
        PolarsError::InvalidOperation(
            format!("a and b must be finite and strictly positive, got a={a}, b={b}: {e}").into(),
        )
    })
}

param_validator! {
    /// Validate the `(a, b)` parameterisation and return the validated `b`.
    ///
    /// `inputs[0]` is `a`, `inputs[1]` is `b`. The Python closed-form moments (`mean = a / (a + b)`,
    /// `variance = a * b / ((a + b)^2 * (a + b + 1))`) are gated on this single FFI round-trip, so
    /// they raise on an invalid parameterisation exactly like the value-keyed methods. `null` in
    /// either input propagates; invalid raises via [`build_dist`].
    fn beta_params;
    params = (a: DataType::Float64 => f64, b: DataType::Float64 => f64);
    build = build_dist;
    returns = b;
    output_name = inputs[1];
}

value_keyed_per_row! {
    /// Apply a value-keyed `f(dist, value)` element-wise over `(value, a, b)`; shared by `pdf`,
    /// `ln_pdf`, `cdf`, `sf`, `ppf`. `null` propagates; an invalid shape raises via
    /// [`build_dist`]; `f` may return `None` to null a row on its own terms.
    fn value_keyed(&Beta);
    params = (DataType::Float64 => f64, DataType::Float64 => f64);
    build = build_dist;
}

/// Apply a parameter-keyed moment `f(dist)` element-wise over `(a, b)`.
///
/// `inputs[0]` is `a`, `inputs[1]` is `b`. `null` in either propagates; an invalid shape raises
/// via [`build_dist`]. Only `entropy` routes through here (the one moment without an elementary
/// closed form); `mean` and `variance` are closed forms computed in Polars and gated on
/// [`beta_params`].
fn params_keyed<F>(inputs: &[Series], f: F) -> PolarsResult<Series>
where
    F: Fn(&Beta) -> f64,
{
    let a = inputs[0].cast(&DataType::Float64)?;
    let a_ca = a.f64()?;
    let b = inputs[1].cast(&DataType::Float64)?;
    let b_ca = b.f64()?;
    let name = inputs[1].name().clone();

    let ca: Float64Chunked =
        try_binary_elementwise(a_ca, b_ca, |a_opt, b_opt| -> PolarsResult<Option<f64>> {
            match (a_opt, b_opt) {
                (Some(a), Some(b)) => {
                    let dist = build_dist(a, b)?;
                    Ok(Some(f(&dist)))
                },
                _ => Ok(None),
            }
        })?;

    Ok(ca.with_name(name).into_series())
}

/// Element-wise Beta sampler over `(a, b, row_index)`, returning `Float64`.
///
/// Per row, `null` propagates and an invalid shape raises via [`build_dist`]. Seeding and
/// chunk-invariance follow [`SampleKwargs::row_rngs`].
///
/// The draw keeps `statrs` (two `O(1)`-amortised Gamma draws, normalised); routing it through
/// `rand_distr` would buy nothing, since that is already the algorithm class it uses (unlike the
/// binomial draw, see docs/explanation/design.md).
#[polars_expr(output_type=Float64)]
fn beta_sample(inputs: &[Series], kwargs: SampleKwargs) -> PolarsResult<Series> {
    let a = inputs[0].cast(&DataType::Float64)?;
    let a_ca = a.f64()?;
    let b = inputs[1].cast(&DataType::Float64)?;
    let b_ca = b.f64()?;
    let index = inputs[2].cast(&DataType::UInt64)?;
    let index_ca = index.u64()?;
    let name = inputs[0].name().clone();

    let rngs = kwargs.row_rngs();

    let ca: Float64Chunked = try_ternary_elementwise(
        a_ca,
        b_ca,
        index_ca,
        |a_opt, b_opt, i_opt| -> PolarsResult<Option<f64>> {
            match (a_opt, b_opt, i_opt) {
                (Some(a), Some(b), Some(i)) => {
                    let dist = build_dist(a, b)?;
                    let mut rng = rngs.rng(i);
                    let draw: f64 = RandDistribution::sample(&dist, &mut rng);
                    Ok(Some(draw))
                },
                _ => Ok(None),
            }
        },
    )?;

    Ok(ca.with_name(name).into_series())
}

sample_scalar_plugin! {
    struct BetaScalarKwargs { a: f64, b: f64 }

    /// Constant-parameter fast path for [`beta_sample`].
    fn beta_sample_scalar(output_type = Float64, physical = Float64Type);

    samples = beta_samples_scalar as BetaSamplesScalarKwargs -> samples_f64_output;

    build = |kw| build_dist(kw.a, kw.b)?;
    draw = |dist, rng| RandDistribution::sample(&dist, rng);
}

/// Element-wise multi-draw Beta sampler: `size` draws per row in one call, the distribution built
/// once per row. Returns `Array(Float64, size)`.
///
/// Seeding and the null/error contract follow [`samples_per_row`] and [`beta_sample`].
#[polars_expr(output_type_func_with_kwargs=samples_f64_output)]
fn beta_samples(inputs: &[Series], kwargs: SamplesKwargs) -> PolarsResult<Series> {
    let a = inputs[0].cast(&DataType::Float64)?;
    let b = inputs[1].cast(&DataType::Float64)?;
    let index = inputs[2].cast(&DataType::UInt64)?;
    let name = inputs[0].name().clone();

    let rows = ternary_param_rows(a.f64()?, b.f64()?, index.u64()?, build_dist);

    samples_per_row::<Float64Type, _, _, _, _>(
        name,
        kwargs.seed,
        kwargs.size,
        rows,
        RandDistribution::sample,
    )
}

/// Convergence threshold and Lentz rescale floor of the incomplete-beta continued fraction,
/// `statrs`' own `prec::F64_PREC` / `fpmin` in `checked_beta_reg`, so the log-space port below
/// terminates exactly like its linear twin.
const BETA_CF_EPS: f64 = 1.1102230246251565e-16;
const BETA_CF_FPMIN: f64 = f64::MIN_POSITIVE / BETA_CF_EPS;

/// Iteration cap for [`beta_reg_cf`], deliberately **not** `statrs`' (and Numerical Recipes') 141.
///
/// The Lentz recurrence needs roughly `sqrt(a + b)` terms near the split, so 141 silently truncates
/// above shapes of ~`1e4`: measured 243 terms at `a = b = 1e5`, 526 at `1e6`, 2282 at `1e8`. Stopping
/// early does not merely lose digits, it returns a number with no relation to the integral, and
/// `statrs`' own `beta_reg` still does exactly that (`Beta(1e8, 1e8).cdf(0.5)` is `-1.147` there
/// against an exact `0.5`, and its log `+0.76`, a positive log-probability). The loop exits on the
/// convergence test in every ordinary case, so a cap this high costs nothing until it is needed and
/// covers shapes past `1e10`.
///
/// Beyond ~`1e9` the CF is no longer the limit: `ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b)` in
/// [`ln_beta_reg_direct`] cancels, leaving `~1.1e-16 * ln_gamma(a + b)` absolute in the log
/// (`4.7e-09` at `a = b = 1e6`, `3.1e-03` at `1e12`). That is the caveat in the README accuracy
/// notes; no iteration count removes it.
const BETA_CF_MAX_ITERATIONS: u32 = 100_000;

/// The modified-Lentz continued fraction `statrs` evaluates inside `checked_beta_reg` (identical
/// recurrence and rescaling, but see [`BETA_CF_MAX_ITERATIONS`] for the termination), returning the
/// raw CF value `h` with `I_x(a, b) = x^a (1 - x)^b / (a B(a, b)) * h`. Only valid on the convergent
/// side `x < (a + 1) / (a + b + 2)` (the callers own the branch split), where `h` is `O(1)`, so the
/// caller can keep the underflow-prone prefactor in log space and take `ln(h)` exactly.
fn beta_reg_cf(a: f64, b: f64, x: f64) -> f64 {
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < BETA_CF_FPMIN {
        d = BETA_CF_FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..BETA_CF_MAX_ITERATIONS {
        let m = f64::from(m);
        let m2 = m * 2.0;
        let mut aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < BETA_CF_FPMIN {
            d = BETA_CF_FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < BETA_CF_FPMIN {
            c = BETA_CF_FPMIN;
        }
        d = 1.0 / d;
        h = h * d * c;
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < BETA_CF_FPMIN {
            d = BETA_CF_FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < BETA_CF_FPMIN {
            c = BETA_CF_FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() <= BETA_CF_EPS {
            break;
        }
    }
    h
}

/// `ln I_x(a, b)` on the CF's convergent side: [`beta_reg_cf`] with the prefactor
/// `x^a (1 - x)^b / (a B(a, b))` kept in log space via `ln_gamma`. Finite where the linear form
/// underflows (once `a ln x` drops below ~-745) and agreeing with it to machine precision where it
/// does not (same CF, same termination).
fn ln_beta_reg_direct(a: f64, b: f64, x: f64) -> f64 {
    ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b)
        + a * x.ln()
        + b * (-x).ln_1p()
        + (beta_reg_cf(a, b, x) / a).ln()
}

/// Natural log of the regularized incomplete beta `ln I_x(a, b)`, stable in the left corner.
///
/// `beta_reg(a, b, x).ln()` is `-inf` once `x^a` underflows (e.g. `Beta(200, 2).cdf(0.001)`), the
/// regime `log_cdf` exists to serve. The branch split mirrors
/// `statrs::function::beta::checked_beta_reg` exactly:
///
/// * `x < (a + 1) / (a + b + 2)`: [`ln_beta_reg_direct`];
/// * otherwise `I_x` is bounded away from `0` and only its complement can be tiny, so `ln_1p` of the
///   reflected `I_{1-x}(b, a)` keeps full relative precision as `I_x -> 1`. The reflection lands on
///   that side's own convergent branch, so it is [`ln_beta_reg_direct`] again rather than statrs'
///   `beta_reg`: routing it back through the crate would reintroduce the 141-term truncation
///   [`BETA_CF_MAX_ITERATIONS`] exists to avoid, on exactly the half of the domain this branch owns.
///
/// Requires `a, b > 0`; `x` in `(0, 1)` open is the caller's contract (the value bodies map the
/// support edges). `pub(crate)` so `Binomial` binds the same port through the `I_x` identity.
pub(crate) fn ln_beta_reg(a: f64, b: f64, x: f64) -> f64 {
    if x < (a + 1.0) / (a + b + 2.0) {
        ln_beta_reg_direct(a, b, x)
    } else {
        (-ln_beta_reg_direct(b, a, 1.0 - x).exp()).ln_1p()
    }
}

/// Natural log of the complement `ln (1 - I_x(a, b)) = ln I_{1-x}(b, a)`, stable in the right
/// corner.
///
/// Not spelled `ln_beta_reg(b, a, 1 - x)`: taking the complement of `x` *before* branching would
/// round a tiny `x` away (`1 - (1 - x)` back-computes to `0` below ~1e-17) and lose the
/// near-certain side's relative precision. Branching first means each side complements at most
/// once, inside the branch that keeps it exact. Same contract as [`ln_beta_reg`].
pub(crate) fn ln_beta_reg_complement(a: f64, b: f64, x: f64) -> f64 {
    if x < (a + 1.0) / (a + b + 2.0) {
        (-ln_beta_reg_direct(a, b, x).exp()).ln_1p()
    } else {
        ln_beta_reg_direct(b, a, 1.0 - x)
    }
}

// Per-method bodies, shared by the per-row plugins and their `*_scalar` twins.

/// `true` where the density genuinely diverges: a support endpoint whose shape is `< 1`.
///
/// `statrs` disagrees with itself there. Its `pdf` returns `inf` for small shapes but `0` once it
/// switches to `ln_pdf().exp()` (shapes above ~80, so `Beta(0.1, 500).pdf(0.0)` is `0.0`), and its
/// `ln_pdf` returns `-inf` in every case. scipy returns `inf` / `inf` throughout, which is the
/// limit, so both bodies take this branch.
fn diverges_at(dist: &Beta, v: f64) -> bool {
    (v == 0.0 && dist.shape_a() < 1.0) || (v == 1.0 && dist.shape_b() < 1.0)
}

fn pdf_value(dist: &Beta, v: f64) -> Option<f64> {
    if diverges_at(dist, v) {
        Some(f64::INFINITY)
    } else {
        Some(dist.pdf(v))
    }
}

fn ln_pdf_value(dist: &Beta, v: f64) -> Option<f64> {
    if diverges_at(dist, v) {
        Some(f64::INFINITY)
    } else {
        Some(dist.ln_pdf(v))
    }
}

// `cdf` / `sf` rely on the shared drivers' `NaN` short-circuit (see `value_keyed_scalar` in
// `mod.rs`): unlike `pdf` / `ln_pdf`, which compute through to `NaN`, the regularized incomplete
// beta behind them panics on a `NaN` evaluation point, which would abort the whole query.

fn cdf_value(dist: &Beta, v: f64) -> Option<f64> {
    Some(dist.cdf(v))
}

/// Survival function, complementing on whichever side keeps the complement exact.
///
/// `statrs`' `Beta::sf` is `beta_reg(b, a, 1 - x)`, which passes the complement as the *argument*:
/// below `x ~ 1e-16` that rounds to `1.0` and the whole tail is lost (`Beta(0.05, 0.05).sf(1e-16)`
/// came back `1.0` against a true `0.9205`). Left of statrs' own convergent split the cdf is the
/// accurate branch and its complement is well conditioned, since `I_x` cannot be near `1` there;
/// right of it the input is far enough from `0` that `1 - x` is exact and statrs' form is the
/// accurate one.
fn sf_value(dist: &Beta, v: f64) -> Option<f64> {
    let (a, b) = (dist.shape_a(), dist.shape_b());
    if v > 0.0 && v < 1.0 && v < (a + 1.0) / (a + b + 2.0) {
        Some(1.0 - dist.cdf(v))
    } else {
        Some(dist.sf(v))
    }
}

/// Native log-cdf via [`ln_beta_reg`]: finite in the left corner (no `cdf().ln()` underflow) and
/// full relative precision in the right one. The support edges follow the statrs `cdf` (`0`
/// at/below `0`, `1` at/above `1`), taken in log.
fn ln_cdf_value(dist: &Beta, v: f64) -> Option<f64> {
    if v <= 0.0 {
        Some(f64::NEG_INFINITY)
    } else if v >= 1.0 {
        Some(0.0)
    } else {
        Some(ln_beta_reg(dist.shape_a(), dist.shape_b(), v))
    }
}

/// Native log-sf via [`ln_beta_reg_complement`]: finite in the right corner (no `sf().ln()`
/// underflow) and full relative precision in the left one. The support edges follow the statrs
/// `sf` (`1` at/below `0`, `0` at/above `1`), taken in log.
fn ln_sf_value(dist: &Beta, v: f64) -> Option<f64> {
    if v <= 0.0 {
        Some(0.0)
    } else if v >= 1.0 {
        Some(f64::NEG_INFINITY)
    } else {
        Some(ln_beta_reg_complement(dist.shape_a(), dist.shape_b(), v))
    }
}

/// Iteration cap for [`solve_increasing`]. Reached only if Newton never helps, in which case the
/// loop is pure bisection and 100 halvings of a bracket at most 746 wide is far past convergence.
const PPF_MAX_ITERATIONS: u32 = 100;

/// Bracket floor for the log variable, below the log of the smallest subnormal (~`-744.4`) by
/// enough that `exp` of it is exactly `0`. A quantile whose answer lies below the representable
/// range then returns `0` (and `1` on the other side), matching scipy, rather than the one
/// subnormal the bracket happens to stop at.
const PPF_LOG_FLOOR: f64 = -750.0;

/// Relative width at which the bracket is converged; `ln x` carries ~16 digits over this range.
const PPF_LOG_TOL: f64 = 1e-15;

/// Root of `f(s) = target` for a strictly increasing `f` bracketed by `[lo, hi]`, by Newton steps
/// guarded by bisection.
///
/// Every proposal that leaves the current bracket (including a `NaN` one, since the comparison is
/// false for `NaN`) is replaced by the bisection midpoint, and the loop is capped, so this
/// terminates for every input and never evaluates `f` outside the bracket. Both properties are
/// load-bearing: they are exactly what `statrs`' AS 64 inverse lacks (see [`ppf_value`]).
fn solve_increasing<F, D>(
    mut lo: f64,
    mut hi: f64,
    seed: f64,
    target: f64,
    f: F,
    derivative: D,
) -> f64
where
    F: Fn(f64) -> f64,
    D: Fn(f64, f64) -> f64,
{
    let mut s = if seed > lo && seed < hi {
        seed
    } else {
        0.5 * (lo + hi)
    };
    for _ in 0..PPF_MAX_ITERATIONS {
        let value = f(s);
        if value > target {
            hi = s;
        } else {
            lo = s;
        }
        if hi - lo <= PPF_LOG_TOL * (1.0 + hi.abs()) {
            break;
        }
        let proposal = s - (value - target) / derivative(s, value);
        let next = if proposal > lo && proposal < hi {
            proposal
        } else {
            0.5 * (lo + hi)
        };
        // The *iterate* is the answer, not the bracket midpoint: Newton converges long before the
        // side it keeps landing on has pulled the other side in, so a converged root sits inside a
        // bracket that can still be a dozen units wide.
        let converged = (next - s).abs() <= PPF_LOG_TOL * (1.0 + s.abs());
        s = next;
        if converged {
            break;
        }
    }
    s
}

/// Tail-asymptotic seed for the log variable: `ln I_x(a, b) -> a ln x - ln a - ln B(a, b)` as
/// `x -> 0`, inverted. Exact in the corner the solve exists for, and merely a starting point
/// elsewhere, since [`solve_increasing`] brackets whatever it is handed.
///
/// Worth the three `ln_gamma` calls: from the bracket midpoint instead, the first several iterations
/// are spent bisecting through the region where `exp(t)` rounds to `1` and the residual is `-inf`,
/// which is both uninformative and the most expensive place to evaluate.
fn log_tail_seed(shape: f64, other: f64, log_target: f64) -> f64 {
    let ln_beta = ln_gamma(shape) + ln_gamma(other) - ln_gamma(shape + other);
    (log_target + shape.ln() + ln_beta) / shape
}

/// Solve `ln I_x(a, b) = log_mass` for `x` below the median, in the log variable `t = ln x`.
///
/// `log_mass` is the log of the *lower* tail mass, whichever inverse holds it exactly.
fn solve_below_median(dist: &Beta, log_mass: f64) -> f64 {
    let (a, b) = (dist.shape_a(), dist.shape_b());
    let t = solve_increasing(
        PPF_LOG_FLOOR,
        0.0,
        log_tail_seed(a, b, log_mass),
        log_mass,
        |t| ln_beta_reg(a, b, t.exp()),
        |t, value| (t + dist.ln_pdf(t.exp()) - value).exp(),
    );
    t.exp().clamp(0.0, 1.0)
}

/// Largest `float64` strictly below `1`. Its complement `EPSILON / 2` is the smallest `1 - x` any
/// `x < 1` can have, so it is the deepest point the upper-branch solve can actually reach.
const LARGEST_BELOW_ONE: f64 = 1.0 - f64::EPSILON / 2.0;

/// Solve `ln (1 - I_x(a, b)) = log_mass` for `x` above the median, in the log variable
/// `u = ln(1 - x)`.
///
/// `log_mass` is the log of the *upper* tail mass, whichever inverse holds it exactly.
fn solve_above_median(dist: &Beta, log_mass: f64) -> f64 {
    let (a, b) = (dist.shape_a(), dist.shape_b());
    // Past [`LARGEST_BELOW_ONE`] every `x` rounds to `1.0` and the residual collapses to `-inf`, so
    // the solve is no longer monotone in `u` and its bracket converges onto that wall: it would
    // answer `1 - 1.1e-16` where the correctly-rounded answer is `1.0`, non-monotonically in `q`
    // (`Beta(0.5, 0.5).isf(1e-300)` came back *below* `isf(1e-100)`). Only `isf` can ask for a mass
    // this small; `ppf`'s upper branch is entered with `ln(1 - q)`, which bottoms out at `-36.7`.
    if log_mass <= ln_beta_reg_complement(a, b, LARGEST_BELOW_ONE) {
        return 1.0;
    }
    let u = solve_increasing(
        PPF_LOG_FLOOR,
        0.0,
        log_tail_seed(b, a, log_mass),
        log_mass,
        |u| ln_beta_reg_complement(a, b, -u.exp_m1()),
        |u, value| (u + dist.ln_pdf(-u.exp_m1()) - value).exp(),
    );
    (-u.exp_m1()).clamp(0.0, 1.0)
}

/// Smallest root [`solve_above_median`] can resolve, as a plain `x`.
///
/// That solve works in `u = ln(1 - x)`, which is `~-x` near zero and so carries only *absolute*
/// resolution there, floored by [`PPF_LOG_TOL`]. A root below this cannot be told from its
/// neighbours; comfortably above it, `u` still has a dozen digits.
const PPF_UPPER_SOLVE_FLOOR: f64 = 1e-8;

/// Lower tail mass at [`PPF_UPPER_SOLVE_FLOOR`]: the quantile below which `ppf`'s root is too close
/// to `0` for the upper solve, whatever the tail masses look like.
fn mass_below_floor(dist: &Beta) -> f64 {
    ln_beta_reg(dist.shape_a(), dist.shape_b(), PPF_UPPER_SOLVE_FLOOR).exp()
}

/// The [`mass_below_floor`] mirror, read from the survival side. Not `1.0 - mass_below_floor(dist)`:
/// for a distribution concentrated below the floor that complement rounds to `0.0`, and the routing
/// it then selects is the one this exists to avoid. Branch before complementing, as in
/// [`ln_beta_reg_complement`].
fn sf_above_floor(dist: &Beta) -> f64 {
    ln_beta_reg_complement(dist.shape_a(), dist.shape_b(), PPF_UPPER_SOLVE_FLOOR).exp()
}

/// Inverse cdf, by a bounded solve on the log-space incomplete beta rather than `statrs`' AS 64.
///
/// Solved on whichever side of the median is small, in that side's log variable, so the residual
/// stays well scaled where the answer is deep in a corner and both the linear cdf and its complement
/// have rounded away. That is the `q <= 0.5` half of the test, and it is the one that matters for
/// every ordinary shape pair: the small mass is where the log has resolution.
///
/// The second half is the correction. `q <= 0.5` alone assumes the smaller tail mass sits on the
/// same side as the root, which stops being true once the shapes are skewed enough to push the
/// median below [`PPF_UPPER_SOLVE_FLOOR`]. `Beta(0.001, 1)` keeps 99.93% of its mass under `x = 0.5`,
/// so every `q` in `(0.5, 0.9993]` went to the upper solve, whose variable `ln(1 - x)` cannot
/// represent a root that small, and all of them answered one constant, `6.5e-16`, against a true
/// `ppf(0.7) = 0.7^1000 ~ 1.3e-155`. That is the saturation across decades this function replaced
/// `inv_beta_reg` to avoid, re-entered through the routing rather than through the solver, so the
/// test also asks whether the root is under the floor and keeps it on the lower solve when it is.
///
/// The complement is formed as `ln_1p(-q)` rather than `(1 - q).ln()`. With the crossover no longer
/// pinned to `0.5`, a branch can be reached with a small `q`, and `(1 - q).ln()` would hand it
/// `ln(1.0)`, rounding the mass away before the solve starts.
///
/// This replaces `statrs`' `inv_beta_reg` rather than wrapping it. Its Newton step is unguarded and
/// its step-halving loop unbounded, which is not fixable from outside the crate and cost all three
/// of: a panic (`beta_reg` receiving an out-of-range probe, aborting the whole query) below
/// `q ~ 1e-165`; non-termination (>15 s for a single row) for `q` in ~`[1e-150, 1e-60]` at
/// `(a, b) = (200, 2)`; and a result pinned to one constant across 100 decades of `q`, since its
/// convergence floor is absolute (`1e-30`) rather than relative.
fn inverse_cdf(dist: &Beta, q: f64) -> f64 {
    if q <= 0.5_f64.max(mass_below_floor(dist)) {
        solve_below_median(dist, q.ln())
    } else {
        solve_above_median(dist, (-q).ln_1p())
    }
}

/// Inverse survival function: the same two solves, entered from the other side.
///
/// This is the whole of the `isf` fix for `Beta`, and it is a routing change rather than new math.
/// `ppf` holds the lower mass exactly and must round the upper one; `isf` holds the *upper* mass
/// exactly and must round the lower. Each is exact on the branch the other has to round on, so
/// running `isf` as `ppf(1 - q)` picks the rounded branch twice: `1 - q` quantises a tiny `q` to
/// `1.1e-16` absolute, and the upper branch then recovers the mass as `1 - (1 - q)`, which no longer
/// back-computes to `q`. Entering `solve_above_median` with `ln q` directly forms no complement at
/// all. Measured at `Beta(0.1, 500).isf(1e-9)`: `1.66e-09` relative before, `~1e-16` after.
///
/// Routed by the same two tests as [`inverse_cdf`], read from the survival side: `sf` is decreasing,
/// so `q >= sf(floor)` is the statement that the root sits under [`PPF_UPPER_SOLVE_FLOOR`].
fn inverse_sf(dist: &Beta, q: f64) -> f64 {
    if q >= 0.5_f64.min(sf_above_floor(dist)) {
        solve_below_median(dist, (-q).ln_1p())
    } else {
        solve_above_median(dist, q.ln())
    }
}

/// A quantile outside `[0, 1]` yields `null`; the endpoints map to the support bounds
/// (`ppf(0) = 0`, `ppf(1) = 1`), matching `scipy.stats.beta.ppf`. See [`inverse_cdf`] for the
/// interior.
fn ppf_value(dist: &Beta, q: f64) -> Option<f64> {
    if !(0.0..=1.0).contains(&q) {
        None
    } else if q == 0.0 {
        Some(0.0)
    } else if q == 1.0 {
        Some(1.0)
    } else {
        Some(inverse_cdf(dist, q))
    }
}

/// A quantile outside `[0, 1]` yields `null`; the endpoints map to the support bounds in the
/// opposite order to `ppf` (`isf(0) = 1`, `isf(1) = 0`), matching `scipy.stats.beta.isf`.
/// See [`inverse_sf`] for the interior and for why this is not `ppf(1 - q)`.
fn isf_value(dist: &Beta, q: f64) -> Option<f64> {
    if !(0.0..=1.0).contains(&q) {
        None
    } else if q == 0.0 {
        Some(1.0)
    } else if q == 1.0 {
        Some(0.0)
    } else {
        Some(inverse_sf(dist, q))
    }
}

/// Element-wise pdf via `statrs` `Continuous::pdf` (Beta function); `0` outside `[0, 1]`, and
/// divergent (`inf`) at a boundary whose shape is `< 1`. See [`value_keyed`] for the null/error
/// contract.
#[polars_expr(output_type=Float64)]
fn beta_pdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, pdf_value)
}

/// Element-wise log-pdf via native `Continuous::ln_pdf` (more accurate than `pdf().ln()`);
/// `-inf` outside `[0, 1]`.
#[polars_expr(output_type=Float64)]
fn beta_ln_pdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, ln_pdf_value)
}

/// Element-wise cdf via `statrs` `ContinuousCDF::cdf` (regularized incomplete beta); `0` below the
/// support, `1` at/above `1`, `NaN` for a `NaN` value (short-circuited in the shared drivers:
/// statrs panics on it).
#[polars_expr(output_type=Float64)]
fn beta_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, cdf_value)
}

/// Element-wise survival function via native `ContinuousCDF::sf` (accurate in the upper tail);
/// `1` below the support, `0` at/above `1`, `NaN` for a `NaN` value (short-circuited in the shared
/// drivers: statrs panics on it).
#[polars_expr(output_type=Float64)]
fn beta_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, sf_value)
}

/// Element-wise log-cdf via the stable [`ln_beta_reg`] port (finite in the left corner, unlike
/// `cdf().log()`).
#[polars_expr(output_type=Float64)]
fn beta_ln_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, ln_cdf_value)
}

/// Element-wise log-sf via the stable [`ln_beta_reg_complement`] port (finite in the right corner,
/// unlike `sf().log()`).
#[polars_expr(output_type=Float64)]
fn beta_ln_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, ln_sf_value)
}

/// Element-wise ppf via the bounded log-space solve in [`inverse_cdf`], not `statrs`' AS 64.
/// See [`ppf_value`] for the endpoint and out-of-range contract.
#[polars_expr(output_type=Float64)]
fn beta_ppf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, ppf_value)
}

/// Element-wise isf via the same solve entered from the upper tail, not `ppf(1 - q)`.
/// See [`inverse_sf`] for why, and [`isf_value`] for the endpoint and out-of-range contract.
#[polars_expr(output_type=Float64)]
fn beta_isf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, isf_value)
}

value_keyed_scalar_plugins! {
    struct BetaParamsKwargs { a: f64, b: f64 }

    build = |kw| build_dist(kw.a, kw.b)?;

    methods {
        fn beta_pdf_scalar => pdf_value;
        fn beta_ln_pdf_scalar => ln_pdf_value;
        fn beta_cdf_scalar => cdf_value;
        fn beta_sf_scalar => sf_value;
        fn beta_ln_cdf_scalar => ln_cdf_value;
        fn beta_ln_sf_scalar => ln_sf_value;
        fn beta_ppf_scalar => ppf_value;
        fn beta_isf_scalar => isf_value;
    }
}

/// Element-wise differential entropy (in nats) via `statrs` `Distribution::entropy`:
/// `ln B(a, b) - (a - 1) psi(a) - (b - 1) psi(b) + (a + b - 2) psi(a + b)`.
///
/// Kept in Rust, unlike `mean` / `variance`: the beta entropy needs log-Beta and digamma, which
/// have no elementary closed form, so there is no numerically equivalent Polars expression to move
/// it to (the same reasoning as `binomial_entropy`).
#[polars_expr(output_type=Float64)]
fn beta_entropy(inputs: &[Series]) -> PolarsResult<Series> {
    params_keyed(inputs, |dist| dist.entropy().unwrap())
}
