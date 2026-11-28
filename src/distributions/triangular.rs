//! `Triangular(min, mode, max)` is evaluated here rather than through statrs' `Triangular`: its `pdf` at an end the
//! mode sits on is `0 / 0 = NaN`, its `inverse_cdf` panics outside `[0, 1]`, and its sampler multiplies
//! `u (max - min) (mode - min)` before the root, which overflows where the quantile does not.

use std::f64::consts::LN_2;
use std::hint::select_unpredictable;
use std::ops::RangeInclusive;

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::distr::{Distribution, StandardUniform};

use crate::distributions::uniform::Bounds;
use crate::distributions::{
    on_unit_interval, validated_triple, value_keyed_quaternary, value_keyed_scalar, PairDomain,
    ParamDomain,
};
use crate::rng::{
    sample_by_index, sample_per_row_quaternary, samples_by_index, samples_f64_output,
    samples_per_row_quaternary, SampleKwargs, SampleScalarKwargs, SamplesKwargs,
    SamplesScalarKwargs,
};

const MIN: ParamDomain = ParamDomain::finite("min");
const MODE: ParamDomain = ParamDomain::finite("mode");
const MAX: ParamDomain = ParamDomain::finite("max");

/// Densities whose log is read through their excess over `1`. Inside it the support is at most `4` wide, so the
/// excess's denominator cannot overflow; outside it `|ln density| >= ln 2` and the plain `ln` keeps its digits.
const DENSITY_NEAR_ONE: RangeInclusive<f64> = 0.5..=2.0;

struct ModeFromMin;

impl PairDomain for ModeFromMin {
    type Native = f64;
    const NAMES: (&'static str, &'static str) = ("min", "mode");
    const DESCRIPTION: &'static str = "at least min";

    fn accepts(min: f64, mode: f64) -> bool {
        mode >= min
    }
}

struct ModeToMax;

impl PairDomain for ModeToMax {
    type Native = f64;
    const NAMES: (&'static str, &'static str) = ("max", "mode");
    const DESCRIPTION: &'static str = "at most max";

    fn accepts(max: f64, mode: f64) -> bool {
        mode <= max
    }
}

fn check_params(
    min: &Float64Chunked,
    mode: &Float64Chunked,
    max: &Float64Chunked,
) -> PolarsResult<()> {
    MIN.check_column(min)?;
    MODE.check_column(mode)?;
    MAX.check_column(max)?;
    Bounds::check_columns(min, max)?;
    ModeFromMin::check_columns(min, mode)?;
    ModeToMax::check_columns(max, mode)
}

/// Infallible once [`check_params`] has passed; fallible only to fit the drivers' `build` slot.
fn build_dist(min: f64, mode: f64, max: f64) -> PolarsResult<Triangle> {
    Ok(Triangle::new(min, mode, max))
}

/// Constant parameters, deserialised once per call. Every `_scalar` twin checks them once here.
#[derive(serde::Deserialize)]
struct TriangularParams {
    min: f64,
    mode: f64,
    max: f64,
}

impl TriangularParams {
    fn build(&self) -> PolarsResult<Triangle> {
        MIN.check(self.min)?;
        MODE.check(self.mode)?;
        MAX.check(self.max)?;
        Bounds::check(self.min, self.max)?;
        ModeFromMin::check(self.min, self.mode)?;
        ModeToMax::check(self.max, self.mode)?;
        Ok(Triangle::new(self.min, self.mode, self.max))
    }

    fn value_keyed(
        &self,
        value: &Series,
        select: impl Fn(&Triangle, f64) -> Option<f64>,
    ) -> PolarsResult<Series> {
        value_keyed_scalar(value, &self.build()?, select)
    }
}

/// The distribution with `max - min`, `mode - min` and `max - mode` each formed once. All three are finite;
/// `below_mode` or `above_mode` is `0` where the mode sits on that end.
struct Triangle {
    min: f64,
    mode: f64,
    max: f64,
    width: f64,
    below_mode: f64,
    above_mode: f64,
}

/// Where a point lands. The support ends are regions of their own: there the density is a one-sided limit, and the
/// piece on that side can have zero span.
enum Region {
    Below,
    AtMin,
    Inside(Piece),
    AtMax,
    Above,
}

impl Triangle {
    fn new(min: f64, mode: f64, max: f64) -> Self {
        Self {
            min,
            mode,
            max,
            width: max - min,
            below_mode: mode - min,
            above_mode: max - mode,
        }
    }

    /// An interior point reads the piece it lies on, the mode itself the upper one: `x < mode` keeps the lower span
    /// positive and `x < max` the upper.
    fn region(&self, x: f64) -> Region {
        if x < self.min {
            Region::Below
        } else if x == self.min {
            Region::AtMin
        } else if x > self.max {
            Region::Above
        } else if x == self.max {
            Region::AtMax
        } else if x < self.mode {
            Region::Inside(Piece {
                from_end: x - self.min,
                to_mode: self.mode - x,
                span: self.below_mode,
                other_span: self.above_mode,
                width: self.width,
                is_lower: true,
            })
        } else {
            Region::Inside(Piece {
                from_end: self.max - x,
                to_mode: x - self.mode,
                span: self.above_mode,
                other_span: self.below_mode,
                width: self.width,
                is_lower: false,
            })
        }
    }

    /// `2 / width`, rounded as [`Piece::density`] rounds it at the mode.
    fn peak_density(&self) -> f64 {
        2.0 * self.width.recip()
    }

    /// `ln(2 / width)`; near `1` through `ln_1p((2 - width) / width)`, whose numerator is exact there (Sterbenz).
    fn ln_peak_density(&self) -> f64 {
        let peak = self.peak_density();
        if DENSITY_NEAR_ONE.contains(&peak) {
            ((2.0 - self.width) / self.width).ln_1p()
        } else if peak.is_normal() {
            peak.ln()
        } else {
            LN_2 - self.width.ln()
        }
    }

    /// `outside` off the support and at an end the mode is not on, `at_peak()` at an end it is on.
    fn density(
        &self,
        x: f64,
        outside: f64,
        at_peak: impl Fn() -> f64,
        inside: impl Fn(&Piece) -> f64,
    ) -> f64 {
        match self.region(x) {
            Region::Below | Region::Above => outside,
            Region::AtMin if self.below_mode == 0.0 => at_peak(),
            Region::AtMax if self.above_mode == 0.0 => at_peak(),
            Region::AtMin | Region::AtMax => outside,
            Region::Inside(piece) => inside(&piece),
        }
    }

    fn pdf(&self, x: f64) -> f64 {
        self.density(x, 0.0, || self.peak_density(), Piece::density)
    }

    fn ln_pdf(&self, x: f64) -> f64 {
        self.density(
            x,
            f64::NEG_INFINITY,
            || self.ln_peak_density(),
            Piece::ln_density,
        )
    }

    /// The mass below `x` (`lower_tail`) or above it: `at_min` up to `min`, `at_max` from `max` on, and inside, the
    /// piece's `end_mass` where the tail starts from that piece's end, its `rest_mass` otherwise.
    fn tail(
        &self,
        x: f64,
        lower_tail: bool,
        (at_min, at_max): (f64, f64),
        end_mass: fn(&Piece) -> f64,
        rest_mass: fn(&Piece) -> f64,
    ) -> f64 {
        match self.region(x) {
            Region::Below | Region::AtMin => at_min,
            Region::AtMax | Region::Above => at_max,
            Region::Inside(piece) if piece.is_lower == lower_tail => end_mass(&piece),
            Region::Inside(piece) => rest_mass(&piece),
        }
    }

    fn cdf(&self, x: f64) -> f64 {
        self.tail(x, true, (0.0, 1.0), Piece::end_mass, Piece::rest_mass)
    }

    fn sf(&self, x: f64) -> f64 {
        self.tail(x, false, (1.0, 0.0), Piece::end_mass, Piece::rest_mass)
    }

    fn ln_cdf(&self, x: f64) -> f64 {
        self.tail(
            x,
            true,
            (f64::NEG_INFINITY, 0.0),
            Piece::ln_end_mass,
            Piece::ln_rest_mass,
        )
    }

    fn ln_sf(&self, x: f64) -> f64 {
        self.tail(
            x,
            false,
            (0.0, f64::NEG_INFINITY),
            Piece::ln_end_mass,
            Piece::ln_rest_mass,
        )
    }

    /// The point with mass `q` below it (`lower_tail`, `ppf`) or above it (`isf`), solved on `q` itself and read off
    /// whichever of the near end, the mode and the far end is closest, so its distance from that anchor keeps its
    /// relative digits.
    fn quantile(&self, q: f64, lower_tail: bool) -> f64 {
        let (near, far, span, other_span, sign) = if lower_tail {
            (self.min, self.max, self.below_mode, self.above_mode, 1.0)
        } else {
            (self.max, self.min, self.above_mode, self.below_mode, -1.0)
        };
        // `q width - span` in one rounding: its sign picks the piece and its size is the mass past the mode.
        let past_mode = q.mul_add(self.width, -span);
        let root_width = self.width.sqrt();
        // Both pieces are formed and one is selected without a branch: `q` is as unpredictable as a draw.
        let from_near = q.sqrt() * root_width * span.sqrt();
        let to_mode = -past_mode / (1.0 + from_near / span);
        let on_near_piece = select_unpredictable(
            from_near <= to_mode,
            near + sign * from_near,
            self.mode - sign * to_mode,
        );
        let from_far = (1.0 - q).sqrt() * root_width * other_span.sqrt();
        let from_mode = past_mode / (1.0 + from_far / other_span);
        let on_far_piece = select_unpredictable(
            from_mode <= from_far,
            self.mode + sign * from_mode,
            far - sign * from_far,
        );
        select_unpredictable(
            past_mode < 0.0,
            on_near_piece,
            select_unpredictable(past_mode > 0.0, on_far_piece, self.mode),
        )
    }
}

/// An interior point on its piece: `from_end` past the piece's support end and `to_mode` short of the mode, over the
/// piece's `span`, the other piece's `other_span` and the support's `width`. The lower piece's end is `min`. Both
/// distances are positive except `to_mode` at the mode.
struct Piece {
    from_end: f64,
    to_mode: f64,
    span: f64,
    other_span: f64,
    width: f64,
    is_lower: bool,
}

impl Piece {
    /// The mass between the piece's end and the point, `from_end^2 / (width span)`.
    fn end_mass(&self) -> f64 {
        (self.from_end / self.width) * (self.from_end / self.span)
    }

    /// The rest of the mass, `other_span / width + (to_mode / width) (1 + from_end / span)`: a sum of non-negative
    /// terms where `1 - end_mass` cancels once the piece holds nearly all the mass. `1 + from_end / span` rather than
    /// `(span + from_end) / span`, whose numerator overflows at a width near `f64::MAX`.
    fn rest_mass(&self) -> f64 {
        self.other_span / self.width
            + (self.to_mode / self.width) * (self.from_end / self.span + 1.0)
    }

    /// `ln(mass)`: `ln_1p(-complement)` above `1/2`, and from the logs of its factors where the mass is subnormal.
    fn ln_mass(mass: f64, complement: impl Fn() -> f64, ln_factors: impl Fn() -> f64) -> f64 {
        if mass > 0.5 {
            (-complement()).ln_1p()
        } else if mass.is_normal() {
            mass.ln()
        } else {
            ln_factors()
        }
    }

    fn ln_end_mass(&self) -> f64 {
        Self::ln_mass(
            self.end_mass(),
            || self.rest_mass(),
            || 2.0 * self.from_end.ln() - self.span.ln() - self.width.ln(),
        )
    }

    fn ln_rest_mass(&self) -> f64 {
        Self::ln_mass(
            self.rest_mass(),
            || self.end_mass(),
            || {
                (self.other_span + self.to_mode * (self.from_end / self.span + 1.0)).ln()
                    - self.width.ln()
            },
        )
    }

    /// `2 from_end / (span width)`. Divided by `width` rather than multiplied by `2 / width`, which overflows for a
    /// subnormal width where the density away from the mode does not.
    fn density(&self) -> f64 {
        2.0 * (self.from_end / self.span / self.width)
    }

    /// Near `1`, `ln_1p` of the excess `(2 from_end - span width) / (span width)`, its numerator in one rounding:
    /// `ln` of a density rounded near `1` keeps no relative digits. From the logs of the factors where the density
    /// is subnormal.
    fn ln_density(&self) -> f64 {
        let density = self.density();
        if DENSITY_NEAR_ONE.contains(&density) {
            let scale = self.span * self.width;
            ((-self.span).mul_add(self.width, 2.0 * self.from_end) / scale).ln_1p()
        } else if density.is_normal() {
            density.ln()
        } else {
            LN_2 + self.from_end.ln() - self.span.ln() - self.width.ln()
        }
    }
}

fn pdf_value(dist: &Triangle, x: f64) -> Option<f64> {
    Some(dist.pdf(x))
}

fn ln_pdf_value(dist: &Triangle, x: f64) -> Option<f64> {
    Some(dist.ln_pdf(x))
}

fn cdf_value(dist: &Triangle, x: f64) -> Option<f64> {
    Some(dist.cdf(x))
}

fn ln_cdf_value(dist: &Triangle, x: f64) -> Option<f64> {
    Some(dist.ln_cdf(x))
}

fn sf_value(dist: &Triangle, x: f64) -> Option<f64> {
    Some(dist.sf(x))
}

fn ln_sf_value(dist: &Triangle, x: f64) -> Option<f64> {
    Some(dist.ln_sf(x))
}

fn ppf_value(dist: &Triangle, q: f64) -> Option<f64> {
    on_unit_interval(&|q| dist.quantile(q, true), q)
}

fn isf_value(dist: &Triangle, q: f64) -> Option<f64> {
    on_unit_interval(&|q| dist.quantile(q, false), q)
}

fn value_keyed(
    inputs: &[Series],
    select: impl Fn(&Triangle, f64) -> Option<f64>,
) -> PolarsResult<Series> {
    value_keyed_quaternary(inputs, check_params, build_dist, select)
}

#[polars_expr(output_type=Float64)]
fn triangular_pdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, pdf_value)
}

#[polars_expr(output_type=Float64)]
fn triangular_ln_pdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, ln_pdf_value)
}

#[polars_expr(output_type=Float64)]
fn triangular_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, cdf_value)
}

#[polars_expr(output_type=Float64)]
fn triangular_ln_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, ln_cdf_value)
}

#[polars_expr(output_type=Float64)]
fn triangular_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, sf_value)
}

#[polars_expr(output_type=Float64)]
fn triangular_ln_sf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, ln_sf_value)
}

#[polars_expr(output_type=Float64)]
fn triangular_ppf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, ppf_value)
}

#[polars_expr(output_type=Float64)]
fn triangular_isf(inputs: &[Series]) -> PolarsResult<Series> {
    value_keyed(inputs, isf_value)
}

#[polars_expr(output_type=Float64)]
fn triangular_pdf_scalar(inputs: &[Series], kwargs: TriangularParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], pdf_value)
}

#[polars_expr(output_type=Float64)]
fn triangular_ln_pdf_scalar(inputs: &[Series], kwargs: TriangularParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], ln_pdf_value)
}

#[polars_expr(output_type=Float64)]
fn triangular_cdf_scalar(inputs: &[Series], kwargs: TriangularParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], cdf_value)
}

#[polars_expr(output_type=Float64)]
fn triangular_ln_cdf_scalar(inputs: &[Series], kwargs: TriangularParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], ln_cdf_value)
}

#[polars_expr(output_type=Float64)]
fn triangular_sf_scalar(inputs: &[Series], kwargs: TriangularParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], sf_value)
}

#[polars_expr(output_type=Float64)]
fn triangular_ln_sf_scalar(inputs: &[Series], kwargs: TriangularParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], ln_sf_value)
}

#[polars_expr(output_type=Float64)]
fn triangular_ppf_scalar(inputs: &[Series], kwargs: TriangularParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], ppf_value)
}

#[polars_expr(output_type=Float64)]
fn triangular_isf_scalar(inputs: &[Series], kwargs: TriangularParams) -> PolarsResult<Series> {
    kwargs.value_keyed(&inputs[0], isf_value)
}

/// The validated `max`, which the Python moments gate on.
#[polars_expr(output_type=Float64)]
fn triangular_max(inputs: &[Series]) -> PolarsResult<Series> {
    validated_triple(inputs, check_params)
}

/// The quantile at one uniform in `[0, 1)` on a `2^-53` grid, so every draw lies in `[min, max]`.
#[inline]
fn draw(dist: &Triangle, rng: &mut impl rand::Rng) -> f64 {
    let u: f64 = StandardUniform.sample(rng);
    dist.quantile(u, true)
}

#[polars_expr(output_type=Float64)]
fn triangular_sample(inputs: &[Series], kwargs: SampleKwargs) -> PolarsResult<Series> {
    sample_per_row_quaternary(inputs, kwargs, check_params, build_dist, draw)
}

#[polars_expr(output_type=Float64)]
fn triangular_sample_scalar(
    inputs: &[Series],
    kwargs: SampleScalarKwargs<TriangularParams>,
) -> PolarsResult<Series> {
    let dist = kwargs.params.build()?;
    sample_by_index(&inputs[0], kwargs.seed, |rng| draw(&dist, rng))
}

#[polars_expr(output_type_func_with_kwargs=samples_f64_output)]
fn triangular_samples_scalar(
    inputs: &[Series],
    kwargs: SamplesScalarKwargs<TriangularParams>,
) -> PolarsResult<Series> {
    let dist = kwargs.params.build()?;
    samples_by_index(&inputs[0], kwargs.seed, kwargs.size, |rng| draw(&dist, rng))
}

#[polars_expr(output_type_func_with_kwargs=samples_f64_output)]
fn triangular_samples(inputs: &[Series], kwargs: SamplesKwargs) -> PolarsResult<Series> {
    samples_per_row_quaternary(inputs, kwargs, check_params, build_dist, draw)
}
