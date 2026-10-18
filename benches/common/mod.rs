//! Shared dataset generators, seeds, sizes, bijective uniform query permutations,
//! self-only CPU affinity/repinning, and timing harness for `core_bench` (E1) and
//! `competitor_bench` (E2).

#![allow(dead_code)]

pub const SEED_BASE: u64 = 0xE1C0_5EBE_4C00_2026;
pub const ENTROPY_BLOCK_SOURCE: &str = "synthetic_enwik8_like_64kib (deterministic Zipf-skewed English/Wiki markup token stream, seed = 0xE1C0_5EBE_4C00_2021)";

pub mod affinity;
pub mod binary_meta;
pub mod datasets;
pub mod harness;

pub use affinity::*;
pub use datasets::*;
pub use harness::*;
