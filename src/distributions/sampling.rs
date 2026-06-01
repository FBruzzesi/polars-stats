use rand::rngs::OsRng;
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use serde::Deserialize;

/// Kwargs shared by every per-row sampler: a static, optional root seed.
#[derive(Deserialize)]
pub struct SampleKwargs {
    pub seed: Option<u64>,
}

/// Resolve the root seed: the caller's value, or a fresh entropy draw when `None`.
pub fn resolve_root_seed(seed: Option<u64>) -> u64 {
    seed.unwrap_or_else(|| OsRng.next_u64())
}

/// Per-row RNG seeded by (root_seed, row_index). Identical pairs always yield identical streams,
/// so output is independent of how Polars chunks or threads the input.
pub fn row_rng(root_seed: u64, index: u64) -> ChaCha20Rng {
    let mut seed_bytes = [0u8; 32];
    seed_bytes[..8].copy_from_slice(&root_seed.to_le_bytes());
    seed_bytes[8..16].copy_from_slice(&index.to_le_bytes());
    ChaCha20Rng::from_seed(seed_bytes)
}
