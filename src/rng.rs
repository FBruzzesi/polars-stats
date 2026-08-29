//! Shared per-row RNG foundation for all distribution samplers.
//!
//! Every sampler needs the same property: a deterministic, independent random stream
//! per row, derived only from `(root_seed, row_index)`. Because the seed is a function
//! of the row index (not of position within a chunk), the output is invariant to how
//! Polars chunks or threads the input.
//!
//! The generator is [`Pcg64Mcg`]: construction is a handful of integer ops (no key
//! schedule, no keystream block), it passes TestU01 BigCrush, and its output is stable
//! across `rand_pcg` releases and platforms (so seeded results stay reproducible). That
//! makes it a safe default for any distribution, including rejection/Ziggurat samplers
//! that consume an unbounded number of words per draw.
//!
//! A one-shot hash-to-uniform would be cheaper but only serves distributions needing a
//! single uniform per draw, so it is deliberately not the foundation; see
//! `docs/explanation/design.md` for the alternatives that were rejected and why.
//!
//! Every sampler plugin is a shell over one of the drivers below; none resolves a seed or writes a
//! row loop itself. They share one argument order, `name, inputs, seed, [size], closures`, and take
//! their output dtype from the drawn value through [`DrawValue`]. The constant-parameter drivers
//! cast the index `Series` themselves; the per-row drivers take it pre-cast as `&UInt64Chunked`,
//! since [`binary_param_rows`] returns an iterator borrowing its inputs and cannot own the cast.

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

/// splitmix64 finalizer: full-avalanche mixing of a single 64-bit word.
///
/// Used to decorrelate adjacent `(root_seed, index)` pairs before they seed the
/// generator, so neighbouring rows get well-separated states rather than nearly
/// identical ones.
#[inline]
fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Resolve the root seed for a sampler call: the caller's seed if given, otherwise a
/// fresh OS-entropy draw. Called once per plugin invocation, never per row.
///
/// An OS-entropy failure (`SysRng` is fallible) surfaces as a `ComputeError` and fails the
/// evaluation, the same contract as an invalid parameter: it is a per-call error, deliberately
/// not a panic out of the plugin.
#[inline]
fn resolve_root_seed(seed: Option<u64>) -> PolarsResult<u64> {
    match seed {
        Some(seed) => Ok(seed),
        None => SysRng.try_next_u64().map_err(|e| {
            PolarsError::ComputeError(
                format!("failed to draw OS entropy for the sampler root seed: {e}").into(),
            )
        }),
    }
}

/// Per-row RNG seeded by `(root_seed, index)`.
///
/// Identical inputs always yield identical streams, so a sampler built on this is
/// genuinely elementwise: chunking and thread scheduling cannot change its output.
#[inline]
fn row_rng(root_seed: u64, index: u64) -> Pcg64Mcg {
    // Fold both inputs into a 128-bit state via two splitmix64 draws. The low bit is
    // forced odd to give the MCG its full period.
    let lo = splitmix64(root_seed ^ index.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let hi = splitmix64(lo);
    let state = (((hi as u128) << 64) | lo as u128) | 1;
    Pcg64Mcg::new(state)
}

/// Kwargs shared by every per-row sampler plugin.
///
/// A sampler's only static input is an optional root seed: the per-row index travels as
/// a regular input `Series`, not a kwarg. So every distribution deserialises the *same*
/// shape, and they share this one struct rather than each declaring an identical copy.
#[derive(Deserialize)]
pub(crate) struct SampleKwargs {
    pub(crate) seed: Option<u64>,
}

/// Resolve a root seed **once** and hand back a per-row RNG source.
///
/// Every sampler reaches this through a driver, once per plugin invocation and never inside the
/// row loop: OS entropy is drawn at most once here (only when `seed` is `None`), after which every
/// [`RowRngs::rng`] is a few integer ops. That single entry point is why seeded output is identical
/// across the scalar and column paths. Errs only when the OS entropy source fails (see
/// [`resolve_root_seed`]).
#[inline]
fn row_rngs(seed: Option<u64>) -> PolarsResult<RowRngs> {
    Ok(RowRngs {
        root_seed: resolve_root_seed(seed)?,
    })
}

/// Shared driver for the constant-parameter sampler fast paths.
///
/// When every distribution parameter is a Python scalar, the parameters are validated **once** by
/// the caller and passed as kwargs, so the only FFI input is the per-row index produced by
/// `pl.int_range(0, len)`. That index is dense and non-null by construction, which lets this skip
/// the per-row `Option`/validity bookkeeping the general `try_*_elementwise` paths must carry.
///
/// `draw` takes one value from a `&mut` per-row RNG already seeded `(root_seed, i)`, the same draw
/// the distribution's per-row plugin performs, so seeded output is identical between the two.
#[inline]
pub(crate) fn sample_by_index<V, Draw>(
    name: PlSmallStr,
    index: &Series,
    seed: Option<u64>,
    draw: Draw,
) -> PolarsResult<Series>
where
    V: DrawValue,
    ChunkedArray<V::Data>: NewChunkedArray<V::Data, V> + IntoSeries,
    Draw: Fn(&mut Pcg64Mcg) -> V,
{
    let index = index.cast(&DataType::UInt64)?;
    let index_ca = index.u64()?;
    let rngs = row_rngs(seed)?;

    let ca = ChunkedArray::<V::Data>::from_iter_values(
        name,
        index_ca.into_no_null_iter().map(|i| {
            let mut rng = rngs.rng(i);
            draw(&mut rng)
        }),
    );
    Ok(ca.into_series())
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

/// Shared driver for the column-parameter single-draw (`sample`) per-row paths: one parameter
/// column plus the row index.
///
/// The column-parameter counterpart of [`sample_by_index`], and the single-draw counterpart of
/// [`samples_per_row`]. `build` constructs the row's draw state once per row; `draw` takes one
/// value from that row's stream. Any null input nulls the row without calling `build`; an invalid
/// parameterisation `?`-raises out of the row closure.
///
/// Seeding follows [`sample_by_index`]: row `i` draws from a stream keyed `(root_seed, i)`, a
/// function of position only, never of the parameters, so the scalar and column paths stay
/// bit-identical for the same parameters.
///
/// Takes `ChunkedArray`s, not the [`binary_param_rows`] iterator its multi-draw twin consumes:
/// `try_*_elementwise` walks each concrete arrow chunk, while chaining `ChunkedArray::iter()`s
/// across chunks costs more per row than the single draw. One draw per row also does not pay for a
/// fork-join dispatch, so this fills serially and [`PARALLEL_FILL_MIN_DRAWS`] is the multi-draw
/// path's concern alone.
///
/// Keep `build` and `draw` generic `Fn`s: they monomorphise into the row loop, where a `&dyn Fn` or
/// a `fn` pointer would cost an indirect call per row.
#[inline]
pub(crate) fn sample_per_row_binary<V, A, S, Build, Draw>(
    name: PlSmallStr,
    param: &ChunkedArray<A>,
    index: &UInt64Chunked,
    seed: Option<u64>,
    build: Build,
    draw: Draw,
) -> PolarsResult<Series>
where
    V: DrawValue,
    ChunkedArray<V::Data>: IntoSeries,
    A: PolarsNumericType,
    Build: Fn(A::Native) -> PolarsResult<S>,
    Draw: Fn(&S, &mut Pcg64Mcg) -> V,
{
    let rngs = row_rngs(seed)?;

    let ca: ChunkedArray<V::Data> = try_binary_elementwise(
        param,
        index,
        |param_opt, index_opt| -> PolarsResult<Option<V>> {
            match (param_opt, index_opt) {
                (Some(param), Some(index)) => {
                    let state = build(param)?;
                    let mut rng = rngs.rng(index);
                    Ok(Some(draw(&state, &mut rng)))
                },
                _ => Ok(None),
            }
        },
    )?;

    Ok(ca.with_name(name).into_series())
}

/// Two-parameter counterpart of [`sample_per_row_binary`] (e.g. `(mu, sigma)`, `(n, p)`), same
/// contracts throughout.
///
/// The two parameter dtypes are independent, so a mixed `(u64, f64)` parameterisation (Binomial's
/// `UInt64` `n` beside its `Float64` `p`) fits, as in [`ternary_param_rows`]: the caller does the
/// cast and the accessor (`.f64()` / `.u64()`), which fixes `A` and `B`. `S` is whatever `build`
/// returns, the built distribution for most callers but Uniform's raw `(f64, f64)` bounds for the
/// one that validates then discards.
#[inline]
pub(crate) fn sample_per_row_ternary<V, A, B, S, Build, Draw>(
    name: PlSmallStr,
    a: &ChunkedArray<A>,
    b: &ChunkedArray<B>,
    index: &UInt64Chunked,
    seed: Option<u64>,
    build: Build,
    draw: Draw,
) -> PolarsResult<Series>
where
    V: DrawValue,
    ChunkedArray<V::Data>: IntoSeries,
    A: PolarsNumericType,
    B: PolarsNumericType,
    Build: Fn(A::Native, B::Native) -> PolarsResult<S>,
    Draw: Fn(&S, &mut Pcg64Mcg) -> V,
{
    let rngs = row_rngs(seed)?;

    let ca: ChunkedArray<V::Data> = try_ternary_elementwise(
        a,
        b,
        index,
        |a_opt, b_opt, index_opt| -> PolarsResult<Option<V>> {
            match (a_opt, b_opt, index_opt) {
                (Some(a), Some(b), Some(index)) => {
                    let state = build(a, b)?;
                    let mut rng = rngs.rng(index);
                    Ok(Some(draw(&state, &mut rng)))
                },
                _ => Ok(None),
            }
        },
    )?;

    Ok(ca.with_name(name).into_series())
}

/// Total draw count below which the multi-draw row fill runs serially: a fork-join dispatch
/// costs more than drawing this few values, and elementwise plugins run once per
/// `group_by` / `over` partition, so tiny calls are common.
const PARALLEL_FILL_MIN_DRAWS: usize = 4096;

/// Fill the row-major multi-draw buffer: `fill_row(i, slot)` writes row `i`'s `size` draws into
/// its own disjoint `size`-wide slice.
///
/// Rows fill in parallel on the polars thread pool when the total work justifies it. That is
/// deterministic because a row's draws depend only on `(root_seed, row_index)`, never on other
/// rows or on visit order, so the parallel fill is bit-identical to the serial one.
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

/// Shared driver for the constant-parameter multi-draw (`samples`) fast paths.
///
/// The multi-draw counterpart of [`sample_by_index`]: one plugin call returns the
/// `Array(width=size)` column directly, rather than `size` `sample` calls glued by `concat_arr`.
///
/// Row `i`'s `size` draws are **consecutive values from one per-row stream** seeded
/// `(root_seed, i)`, the same stream `sample` takes its single draw from. So `samples(size=1)` is
/// bit-identical to `sample` for the same seed, and growing `size` extends each row's array without
/// changing the existing draws. Rows can fill in parallel (see [`fill_rows`]).
#[inline]
pub(crate) fn samples_by_index<V, Draw>(
    name: PlSmallStr,
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
    // The Python layer rejects `size <= 0` before registering the plugin; this guards the
    // zero-width `Array` reshape against any other caller.
    if size == 0 {
        return Err(PolarsError::InvalidOperation(
            "samples requires a positive size".into(),
        ));
    }
    let index = index.cast(&DataType::UInt64)?;
    let indices: Vec<u64> = index.u64()?.into_no_null_iter().collect();
    let rngs = row_rngs(seed)?;

    let mut flat = vec![V::default(); indices.len() * size];
    fill_rows(&mut flat, size, |row, slot| {
        let mut rng = rngs.rng(indices[row]);
        for value in slot {
            *value = draw(&mut rng);
        }
    });

    ChunkedArray::<V::Data>::from_iter_values(name, flat.into_iter())
        .into_series()
        .reshape_array(&[ReshapeDimension::Infer, ReshapeDimension::new(size as i64)])
}

/// Shared driver for the column-parameter multi-draw (`samples`) per-row paths.
///
/// The column-parameter counterpart of [`samples_by_index`]. `rows` yields, per row, the index and
/// a ready-to-draw state (typically the built distribution, so it is constructed once per row
/// rather than once per draw). A `None` row (any null input) becomes a null `Array` element with
/// its inner slots also null, the same two-layer shape as `pl.lit(None, dtype=Array(...))`; an
/// invalid parameterisation `?`-raises out of the row iterator.
///
/// Seeding follows [`samples_by_index`]. The per-draw *consumption* of a row's stream does depend
/// on its parameters (rejection samplers draw a variable number of words), which is fine: the
/// stream is private to the row.
#[inline]
pub(crate) fn samples_per_row<V, S, Rows, Draw>(
    name: PlSmallStr,
    rows: Rows,
    seed: Option<u64>,
    size: usize,
    draw: Draw,
) -> PolarsResult<Series>
where
    V: DrawValue + Default + Clone + Send,
    ChunkedArray<V::Data>: NewChunkedArray<V::Data, V> + IntoSeries,
    S: Sync,
    Rows: Iterator<Item = PolarsResult<Option<(u64, S)>>>,
    Draw: Fn(&S, &mut Pcg64Mcg) -> V + Sync,
{
    if size == 0 {
        return Err(PolarsError::InvalidOperation(
            "samples requires a positive size".into(),
        ));
    }
    // Materialising the states up front separates the sequential part (parameter validation,
    // which can raise) from the draw loop, whose rows then fill independently. A null row keeps
    // its `V::default()` slice; it is masked by the validity bitmaps below, never read.
    let states: Vec<Option<(u64, S)>> = rows.collect::<PolarsResult<_>>()?;
    let rngs = row_rngs(seed)?;

    let mut flat = vec![V::default(); states.len() * size];
    fill_rows(&mut flat, size, |row, slot| {
        if let Some((index, state)) = &states[row] {
            let mut rng = rngs.rng(*index);
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
    // A null row nulls both layers, matching `pl.lit(None, dtype=Array(...))`: the outer bit makes
    // the element null, and the inner bits keep value-level reads (`arr.first`, `explode`, ...)
    // from leaking the placeholder defaults behind it.
    let inner_validity: Bitmap = states
        .iter()
        .flat_map(|state| std::iter::repeat_n(state.is_some(), size))
        .collect();
    let inner = flat.rechunk().chunks()[0].with_validity(Some(inner_validity));
    let out = Series::from_arrow(name.clone(), inner)?.reshape_array(&shape)?;
    let masked = out.rechunk().chunks()[0].with_validity(Some(outer_validity));
    Series::from_arrow(name, masked)
}

/// Build the per-row state iterator a single-parameter column `samples` plugin feeds to
/// [`samples_per_row`]: zip the parameter column with the row index and, on a fully-non-null row,
/// run `build` once to make the row's draw state.
///
/// Any null input nulls the whole row, matching the single-draw paths.
pub(crate) fn binary_param_rows<'a, A, S, F>(
    param: &'a ChunkedArray<A>,
    index: &'a UInt64Chunked,
    build: F,
) -> impl Iterator<Item = PolarsResult<Option<(u64, S)>>> + 'a
where
    A: PolarsNumericType,
    F: Fn(A::Native) -> PolarsResult<S> + 'a,
    S: 'a,
{
    param
        .iter()
        .zip(index.iter())
        .map(move |(p_opt, i_opt)| match (p_opt, i_opt) {
            (Some(p), Some(i)) => Ok(Some((i, build(p)?))),
            _ => Ok(None),
        })
}

/// Two-parameter counterpart of [`binary_param_rows`] (e.g. `(mu, sigma)`, `(n, p)`): zip both
/// parameter columns with the row index and, on a fully-non-null row, run `build` once. The two
/// parameter dtypes are independent, so a mixed `(u64, f64)` parameterisation (Binomial) fits.
pub(crate) fn ternary_param_rows<'a, A, B, S, F>(
    a: &'a ChunkedArray<A>,
    b: &'a ChunkedArray<B>,
    index: &'a UInt64Chunked,
    build: F,
) -> impl Iterator<Item = PolarsResult<Option<(u64, S)>>> + 'a
where
    A: PolarsNumericType,
    B: PolarsNumericType,
    F: Fn(A::Native, B::Native) -> PolarsResult<S> + 'a,
    S: 'a,
{
    a.iter()
        .zip(b.iter())
        .zip(index.iter())
        .map(move |((a_opt, b_opt), i_opt)| match (a_opt, b_opt, i_opt) {
            (Some(a), Some(b), Some(i)) => Ok(Some((i, build(a, b)?))),
            _ => Ok(None),
        })
}

/// Kwargs shared by every column-parameter multi-draw plugin, and the slice the shared
/// output-dtype functions read from the scalar variants' kwargs (serde skips their extra
/// parameter fields): the optional root seed and the draw count, which is the output `Array`
/// width.
#[derive(Deserialize)]
pub(crate) struct SamplesKwargs {
    pub(crate) seed: Option<u64>,
    pub(crate) size: usize,
}

/// Single-draw fast-path kwargs: a distribution's constant parameters `P` plus the optional root seed.
///
/// `P` is flattened, so the wire shape is one flat mapping with `seed` next to the parameter keys.
/// Composing keeps each distribution's parameter list, and its constructor order, in the one struct
/// its value-keyed fast paths already use.
#[derive(Deserialize)]
pub(crate) struct SampleScalarKwargs<P> {
    pub(crate) seed: Option<u64>,
    #[serde(flatten)]
    pub(crate) params: P,
}

/// Multi-draw counterpart of [`SampleScalarKwargs`], plus the draw count `size` (the output `Array`
/// width).
///
/// `seed` and `size` sit outside `P` and flatten alongside the parameter keys, so the shared output
/// functions can read `size` by deserialising the same bytes into [`SamplesKwargs`].
#[derive(Deserialize)]
pub(crate) struct SamplesScalarKwargs<P> {
    pub(crate) seed: Option<u64>,
    pub(crate) size: usize,
    #[serde(flatten)]
    pub(crate) params: P,
}

fn samples_output(fields: &[Field], width: usize, inner: DataType) -> PolarsResult<Field> {
    Ok(Field::new(
        fields[0].name().clone(),
        DataType::Array(Box::new(inner), width),
    ))
}

/// Output dtype of a float-valued multi-draw plugin: `Array(Float64, size)`.
pub(crate) fn samples_f64_output(fields: &[Field], kwargs: SamplesKwargs) -> PolarsResult<Field> {
    samples_output(fields, kwargs.size, DataType::Float64)
}

/// Output dtype of an integer-valued multi-draw plugin: `Array(UInt64, size)`.
pub(crate) fn samples_u64_output(fields: &[Field], kwargs: SamplesKwargs) -> PolarsResult<Field> {
    samples_output(fields, kwargs.size, DataType::UInt64)
}

/// Output dtype of a signed-integer-valued multi-draw plugin: `Array(Int64, size)`.
pub(crate) fn samples_i64_output(fields: &[Field], kwargs: SamplesKwargs) -> PolarsResult<Field> {
    samples_output(fields, kwargs.size, DataType::Int64)
}

/// Output dtype of a boolean-valued multi-draw plugin: `Array(Boolean, size)`.
pub(crate) fn samples_bool_output(fields: &[Field], kwargs: SamplesKwargs) -> PolarsResult<Field> {
    samples_output(fields, kwargs.size, DataType::Boolean)
}

/// Per-call source of per-row RNGs, all derived from one already-resolved root seed.
///
/// The resolve-once step happens at construction (via [`row_rngs`]), so the
/// type enforces the invariant every elementwise sampler depends on: resolve per call,
/// derive per row.
pub(crate) struct RowRngs {
    root_seed: u64,
}

impl RowRngs {
    /// The deterministic, independent RNG for `index`. Identical `(seed, index)` pairs
    /// always yield identical streams, so output is invariant to Polars chunking/threading.
    #[inline]
    pub(crate) fn rng(&self, index: u64) -> Pcg64Mcg {
        row_rng(self.root_seed, index)
    }
}
