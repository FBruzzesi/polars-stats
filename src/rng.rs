//! Per-row RNG foundation shared by every sampler.
//!
//! Row `i` draws from a `Pcg64Mcg` seeded from `(root_seed, i)` alone, so output does not depend on
//! how polars chunks or threads the input, and the constant-parameter and column-parameter paths
//! agree bit for bit for the same seed. `Pcg64Mcg` costs a handful of integer ops to construct, passes
//! TestU01 BigCrush, and is stable across `rand_pcg` releases and platforms.
//!
//! Every sampler plugin is a one-line shell over a driver here: [`sample_by_index`] /
//! [`samples_by_index`] when every parameter is constant (only the row index crosses FFI),
//! `sample_per_row_*` / `samples_per_row_*` when one is a column. Driver names count plugin inputs:
//! `_binary` takes `(param, row_index)`, `_ternary` takes `(a, b, row_index)`. A distribution supplies
//! `build` (a row's draw state from its parameters) and `draw` (one value from that state); the output
//! dtype follows the drawn value through [`DrawValue`]. A null in any input nulls the row.

use polars::prelude::arity::{try_binary_elementwise, try_ternary_elementwise};
use polars::prelude::*;
use polars_arrow::array::ArrayFromIter;
use polars_arrow::bitmap::Bitmap;
use polars_arrow::datatypes::reshape::ReshapeDimension;
use polars_core::runtime::RAYON;
use polars_core::utils::rayon::prelude::*;
use rand::rngs::SysRng;
use rand::TryRng;
use rand_pcg::Pcg64Mcg;
use serde::Deserialize;

use crate::distributions::{align_inputs, coerce_f64, ParamDomain};

/// splitmix64 finalizer: full-avalanche mixing of one 64-bit word, so neighbouring
/// `(root_seed, index)` pairs seed well-separated states.
#[inline]
fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Per-call source of per-row RNGs. The root seed is resolved once per plugin call, so OS entropy is
/// drawn at most once and every [`Self::row_rng`] is a few integer ops.
struct RowRngs {
    root_seed: u64,
}

impl RowRngs {
    /// The caller's seed, or one OS-entropy draw; an entropy failure is a `ComputeError`, never a
    /// panic out of the plugin.
    fn new(seed: Option<u64>) -> PolarsResult<Self> {
        let root_seed = match seed {
            Some(seed) => seed,
            None => SysRng.try_next_u64().map_err(|e| {
                polars_err!(ComputeError: "failed to draw OS entropy for the sampler root seed: {e}")
            })?,
        };
        Ok(Self { root_seed })
    }

    /// Identical `(seed, index)` pairs always yield identical streams. Two splitmix64 draws fold both
    /// inputs into the 128-bit state; the low bit is forced odd for the MCG's full period.
    #[inline]
    fn row_rng(&self, index: u64) -> Pcg64Mcg {
        let lo = splitmix64(self.root_seed ^ index.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let hi = splitmix64(lo);
        Pcg64Mcg::new((((hi as u128) << 64) | lo as u128) | 1)
    }
}

fn coerce_index(index: &Series) -> PolarsResult<UInt64Chunked> {
    Ok(index.cast(&DataType::UInt64)?.u64()?.clone())
}

/// The output column dtype a drawn value collects into.
pub(crate) trait DrawValue: Sized {
    type Data: PolarsDataType<Array: ArrayFromIter<Option<Self>>>;
}

impl DrawValue for f64 {
    type Data = Float64Type;
}

impl DrawValue for u64 {
    type Data = UInt64Type;
}

impl DrawValue for i64 {
    type Data = Int64Type;
}

impl DrawValue for bool {
    type Data = BooleanType;
}

/// Kwargs of every column-parameter single-draw plugin. The row index travels as a regular input.
#[derive(Deserialize)]
pub(crate) struct SampleKwargs {
    pub(crate) seed: Option<u64>,
}

/// Kwargs of every column-parameter multi-draw plugin; `size` is the output `Array` width. The
/// output-dtype functions read the `_scalar` variants' kwargs into this too: serde skips the extra
/// parameter keys.
#[derive(Deserialize)]
pub(crate) struct SamplesKwargs {
    pub(crate) seed: Option<u64>,
    pub(crate) size: usize,
}

/// Single-draw fast-path kwargs: the distribution's constant parameters `P`, flattened next to
/// `seed` so the wire shape stays one flat mapping.
#[derive(Deserialize)]
pub(crate) struct SampleScalarKwargs<P> {
    pub(crate) seed: Option<u64>,
    #[serde(flatten)]
    pub(crate) params: P,
}

/// Multi-draw counterpart of [`SampleScalarKwargs`].
#[derive(Deserialize)]
pub(crate) struct SamplesScalarKwargs<P> {
    pub(crate) seed: Option<u64>,
    pub(crate) size: usize,
    #[serde(flatten)]
    pub(crate) params: P,
}

/// Constant-parameter single-draw driver: the caller has built the draw state once, and the only
/// input is the dense, non-null row index. Row `i` draws from the stream keyed `(root_seed, i)`, the
/// one the per-row drivers use.
pub(crate) fn sample_by_index<V, Draw>(
    index: &Series,
    seed: Option<u64>,
    draw: Draw,
) -> PolarsResult<Series>
where
    V: DrawValue,
    ChunkedArray<V::Data>: NewChunkedArray<V::Data, V> + IntoSeries,
    Draw: Fn(&mut Pcg64Mcg) -> V,
{
    let indices = coerce_index(index)?;
    let rngs = RowRngs::new(seed)?;

    let out = ChunkedArray::<V::Data>::from_iter_values(
        index.name().clone(),
        indices
            .into_no_null_iter()
            .map(|i| draw(&mut rngs.row_rng(i))),
    );
    Ok(out.into_series())
}

/// Column-parameter single-draw driver over `(param, row_index)`. `domain`'s column pass runs before
/// any row is built, so `build` cannot fail in the loop. Output is named after the parameter column.
///
/// Keep `build` and `draw` generic `Fn`s: they monomorphise into the row loop, where a `&dyn Fn` or
/// a `fn` pointer would cost an indirect call per row.
pub(crate) fn sample_per_row_binary<V, State, Build, Draw>(
    inputs: &[Series],
    kwargs: SampleKwargs,
    domain: &ParamDomain,
    build: Build,
    draw: Draw,
) -> PolarsResult<Series>
where
    V: DrawValue,
    ChunkedArray<V::Data>: IntoSeries,
    Build: Fn(f64) -> PolarsResult<State>,
    Draw: Fn(&State, &mut Pcg64Mcg) -> V,
{
    let inputs = align_inputs(inputs)?;
    let param = coerce_f64(&inputs[0])?;
    let index = coerce_index(&inputs[1])?;
    domain.check_column(&param)?;
    let rngs = RowRngs::new(kwargs.seed)?;

    let out: ChunkedArray<V::Data> =
        try_binary_elementwise(&param, &index, |param, index| -> PolarsResult<Option<V>> {
            match (param, index) {
                (Some(param), Some(index)) => {
                    Ok(Some(draw(&build(param)?, &mut rngs.row_rng(index))))
                },
                _ => Ok(None),
            }
        })?;
    Ok(out.with_name(inputs[0].name().clone()).into_series())
}

/// Two-parameter counterpart of [`sample_per_row_binary`], over `(a, b, row_index)`. Each parameter
/// has its own coercer, so a mixed `(UInt64, Float64)` parameterisation fits; `check_params` is the
/// caller's pass over both columns. `State` is whatever `build` returns: the built distribution for
/// most callers, Uniform's checked `(min, max)` for the one that draws directly.
pub(crate) fn sample_per_row_ternary<V, A, B, State, CoerceA, CoerceB, Check, Build, Draw>(
    inputs: &[Series],
    kwargs: SampleKwargs,
    coerce_a: CoerceA,
    coerce_b: CoerceB,
    check_params: Check,
    build: Build,
    draw: Draw,
) -> PolarsResult<Series>
where
    V: DrawValue,
    ChunkedArray<V::Data>: IntoSeries,
    A: PolarsNumericType,
    B: PolarsNumericType,
    CoerceA: Fn(&Series) -> PolarsResult<ChunkedArray<A>>,
    CoerceB: Fn(&Series) -> PolarsResult<ChunkedArray<B>>,
    Check: Fn(&ChunkedArray<A>, &ChunkedArray<B>) -> PolarsResult<()>,
    Build: Fn(A::Native, B::Native) -> PolarsResult<State>,
    Draw: Fn(&State, &mut Pcg64Mcg) -> V,
{
    let inputs = align_inputs(inputs)?;
    let a = coerce_a(&inputs[0])?;
    let b = coerce_b(&inputs[1])?;
    let index = coerce_index(&inputs[2])?;
    check_params(&a, &b)?;
    let rngs = RowRngs::new(kwargs.seed)?;

    let out: ChunkedArray<V::Data> =
        try_ternary_elementwise(&a, &b, &index, |a, b, index| -> PolarsResult<Option<V>> {
            match (a, b, index) {
                (Some(a), Some(b), Some(index)) => {
                    Ok(Some(draw(&build(a, b)?, &mut rngs.row_rng(index))))
                },
                _ => Ok(None),
            }
        })?;
    Ok(out.with_name(inputs[0].name().clone()).into_series())
}

/// Total draw count below which the multi-draw fill runs serially: a fork-join dispatch costs more
/// than this few draws, and elementwise plugins run once per `group_by` / `over` partition.
const PARALLEL_FILL_MIN_DRAWS: usize = 4096;

/// Fill the row-major multi-draw buffer: `fill_row(i, slot)` writes row `i`'s `size` draws into its
/// own slice. Rows fill in parallel when the total justifies it, which is deterministic because a
/// row's draws depend only on `(root_seed, row_index)`.
fn fill_rows<V, F>(flat: &mut [V], size: usize, fill_row: F)
where
    V: Send,
    F: Fn(usize, &mut [V]) + Sync,
{
    if flat.len() >= PARALLEL_FILL_MIN_DRAWS {
        RAYON.install(|| {
            flat.par_chunks_mut(size)
                .enumerate()
                .for_each(|(row, slot)| fill_row(row, slot));
        });
    } else {
        for (row, slot) in flat.chunks_mut(size).enumerate() {
            fill_row(row, slot);
        }
    }
}

/// Constant-parameter multi-draw driver: one call returns the `Array(width=size)` column. Row `i`'s
/// `size` draws are consecutive values from the stream keyed `(root_seed, i)`, the one `sample` takes
/// its single draw from, so `samples(size=1)` is bit-identical to `sample` and growing `size` extends
/// each row without changing the existing draws.
pub(crate) fn samples_by_index<V, Draw>(
    index: &Series,
    seed: Option<u64>,
    size: usize,
    draw: Draw,
) -> PolarsResult<Series>
where
    V: DrawValue + Default + Clone + Send,
    ChunkedArray<V::Data>: NewChunkedArray<V::Data, V> + IntoSeries,
    Draw: Fn(&mut Pcg64Mcg) -> V + Sync,
{
    polars_ensure!(size > 0, InvalidOperation: "samples requires a positive size");
    let indices: Vec<u64> = coerce_index(index)?.into_no_null_iter().collect();
    let rngs = RowRngs::new(seed)?;

    let mut flat = vec![V::default(); indices.len() * size];
    fill_rows(&mut flat, size, |row, slot| {
        let mut rng = rngs.row_rng(indices[row]);
        for value in slot {
            *value = draw(&mut rng);
        }
    });

    ChunkedArray::<V::Data>::from_iter_values(index.name().clone(), flat.into_iter())
        .into_series()
        .reshape_array(&[ReshapeDimension::Infer, ReshapeDimension::new(size as i64)])
}

/// Column-parameter multi-draw driver. `rows` yields, per row, the index and a ready-to-draw state
/// (built once per row, not once per draw); a `None` row (any null input) becomes a null `Array`
/// element whose inner slots are also null, the two-layer shape of `pl.lit(None, dtype=Array(...))`.
/// Seeding as in [`samples_by_index`].
fn samples_per_row<V, State, Rows, Draw>(
    name: PlSmallStr,
    rows: Rows,
    seed: Option<u64>,
    size: usize,
    draw: Draw,
) -> PolarsResult<Series>
where
    V: DrawValue + Default + Clone + Send,
    ChunkedArray<V::Data>: NewChunkedArray<V::Data, V> + IntoSeries,
    State: Sync,
    Rows: Iterator<Item = PolarsResult<Option<(u64, State)>>>,
    Draw: Fn(&State, &mut Pcg64Mcg) -> V + Sync,
{
    polars_ensure!(size > 0, InvalidOperation: "samples requires a positive size");
    // Materialising the states first separates the part that can raise (building) from the draw
    // loop, whose rows then fill independently. A null row keeps its `V::default()` slice, masked by
    // the validity bitmaps below and never read.
    let states: Vec<Option<(u64, State)>> = rows.collect::<PolarsResult<_>>()?;
    let rngs = RowRngs::new(seed)?;

    let mut flat = vec![V::default(); states.len() * size];
    fill_rows(&mut flat, size, |row, slot| {
        if let Some((index, state)) = &states[row] {
            let mut rng = rngs.row_rng(*index);
            for value in slot {
                *value = draw(state, &mut rng);
            }
        }
    });

    let flat =
        ChunkedArray::<V::Data>::from_iter_values(name.clone(), flat.into_iter()).into_series();
    let shape = [ReshapeDimension::Infer, ReshapeDimension::new(size as i64)];

    let outer_validity: Bitmap = states.iter().map(Option::is_some).collect();
    if outer_validity.unset_bits() == 0 {
        return flat.reshape_array(&shape);
    }
    let inner_validity: Bitmap = states
        .iter()
        .flat_map(|state| std::iter::repeat_n(state.is_some(), size))
        .collect();
    let inner = flat.rechunk().chunks()[0].with_validity(Some(inner_validity));
    let out = Series::from_arrow(name.clone(), inner)?.reshape_array(&shape)?;
    let masked = out.rechunk().chunks()[0].with_validity(Some(outer_validity));
    Series::from_arrow(name, masked)
}

/// [`samples_per_row`] over `(param, row_index)`: `domain`'s column pass, then one `build` per
/// fully-non-null row.
pub(crate) fn samples_per_row_binary<V, State, Build, Draw>(
    inputs: &[Series],
    kwargs: SamplesKwargs,
    domain: &ParamDomain,
    build: Build,
    draw: Draw,
) -> PolarsResult<Series>
where
    V: DrawValue + Default + Clone + Send,
    ChunkedArray<V::Data>: NewChunkedArray<V::Data, V> + IntoSeries,
    State: Sync,
    Build: Fn(f64) -> PolarsResult<State>,
    Draw: Fn(&State, &mut Pcg64Mcg) -> V + Sync,
{
    let inputs = align_inputs(inputs)?;
    let param = coerce_f64(&inputs[0])?;
    let index = coerce_index(&inputs[1])?;
    domain.check_column(&param)?;

    let rows = param
        .iter()
        .zip(index.iter())
        .map(|(param, index)| match (param, index) {
            (Some(param), Some(index)) => Ok(Some((index, build(param)?))),
            _ => Ok(None),
        });
    samples_per_row(
        inputs[0].name().clone(),
        rows,
        kwargs.seed,
        kwargs.size,
        draw,
    )
}

/// Two-parameter counterpart of [`samples_per_row_binary`], over `(a, b, row_index)`; coercers and
/// `check_params` as in [`sample_per_row_ternary`].
pub(crate) fn samples_per_row_ternary<V, A, B, State, CoerceA, CoerceB, Check, Build, Draw>(
    inputs: &[Series],
    kwargs: SamplesKwargs,
    coerce_a: CoerceA,
    coerce_b: CoerceB,
    check_params: Check,
    build: Build,
    draw: Draw,
) -> PolarsResult<Series>
where
    V: DrawValue + Default + Clone + Send,
    ChunkedArray<V::Data>: NewChunkedArray<V::Data, V> + IntoSeries,
    A: PolarsNumericType,
    B: PolarsNumericType,
    State: Sync,
    CoerceA: Fn(&Series) -> PolarsResult<ChunkedArray<A>>,
    CoerceB: Fn(&Series) -> PolarsResult<ChunkedArray<B>>,
    Check: Fn(&ChunkedArray<A>, &ChunkedArray<B>) -> PolarsResult<()>,
    Build: Fn(A::Native, B::Native) -> PolarsResult<State>,
    Draw: Fn(&State, &mut Pcg64Mcg) -> V + Sync,
{
    let inputs = align_inputs(inputs)?;
    let a = coerce_a(&inputs[0])?;
    let b = coerce_b(&inputs[1])?;
    let index = coerce_index(&inputs[2])?;
    check_params(&a, &b)?;

    let rows = a
        .iter()
        .zip(b.iter())
        .zip(index.iter())
        .map(|((a, b), index)| match (a, b, index) {
            (Some(a), Some(b), Some(index)) => Ok(Some((index, build(a, b)?))),
            _ => Ok(None),
        });
    samples_per_row(
        inputs[0].name().clone(),
        rows,
        kwargs.seed,
        kwargs.size,
        draw,
    )
}

fn samples_output(fields: &[Field], width: usize, inner: DataType) -> PolarsResult<Field> {
    Ok(Field::new(
        fields[0].name().clone(),
        DataType::Array(Box::new(inner), width),
    ))
}

pub(crate) fn samples_f64_output(fields: &[Field], kwargs: SamplesKwargs) -> PolarsResult<Field> {
    samples_output(fields, kwargs.size, DataType::Float64)
}

pub(crate) fn samples_u64_output(fields: &[Field], kwargs: SamplesKwargs) -> PolarsResult<Field> {
    samples_output(fields, kwargs.size, DataType::UInt64)
}

pub(crate) fn samples_i64_output(fields: &[Field], kwargs: SamplesKwargs) -> PolarsResult<Field> {
    samples_output(fields, kwargs.size, DataType::Int64)
}

pub(crate) fn samples_bool_output(fields: &[Field], kwargs: SamplesKwargs) -> PolarsResult<Field> {
    samples_output(fields, kwargs.size, DataType::Boolean)
}
