//! Plan v2 (reliability-first) regression tests, verifier G3.
//!
//! - C1: FSE `compress_parallel` framing (`parallel_blocks` of `Some(0)` /
//!   `Some(1)` emitted an unframed stream) and its per-block thread spawn.
//!
//! Every `test_v2_*` here compiles against v5.0.0; the doc comment of each test
//! states whether it fails there (and why) or is a guard that must keep passing.

use std::panic::{AssertUnwindSafe, catch_unwind};

use zipora::entropy::{FseConfig, FseDecoder, FseEncoder};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic payload>".to_string()
    }
}

/// FNV-1a, used to pin wire formats without embedding whole streams.
fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325_u64, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

// ---------------------------------------------------------------------------
// C1: FSE parallel framing
// ---------------------------------------------------------------------------

const FSE_MODE_SINGLE: u8 = 0xF5;
const FSE_MODE_PARALLEL: u8 = 0xF6;

fn fse_framing_payload() -> Vec<u8> {
    b"zipora fse framing regression test ".repeat(120)
}

/// Compress with `cfg`, then decode with a decoder built from the same config.
fn fse_roundtrip(cfg: &FseConfig, data: &[u8]) -> (Vec<u8>, zipora::Result<Vec<u8>>) {
    let compressed = FseEncoder::new(cfg.clone())
        .expect("valid FSE config")
        .compress(data)
        .expect("FSE compress");
    let decoded = FseDecoder::with_config(cfg.clone())
        .expect("valid FSE config")
        .decompress(&compressed);
    (compressed, decoded)
}

fn assert_fse_single_frame_roundtrip(parallel_blocks: Option<usize>) {
    let data = fse_framing_payload();
    let cfg = FseConfig {
        block_size: 1024,
        parallel_blocks,
        ..FseConfig::default()
    };
    assert!(
        data.len() > cfg.block_size * 2,
        "payload must reach compress_parallel"
    );
    let (compressed, decoded) = fse_roundtrip(&cfg, &data);
    let decoded = decoded.unwrap_or_else(|e| {
        panic!("parallel_blocks {parallel_blocks:?}: stream must decode, got: {e}")
    });
    assert_eq!(decoded, data);
    assert_eq!(
        compressed[0], FSE_MODE_SINGLE,
        "parallel_blocks {parallel_blocks:?} has nothing to split and must emit a single-mode frame"
    );
}

/// C1 (fails at v5.0.0): `parallel_blocks: Some(1)` reaches `compress_parallel`,
/// whose `num_blocks <= 1` fallback returned the body without the 0xF5 mode
/// byte; decode fails with "unknown FSE stream mode byte: 0x68".
#[test]
fn test_v2_fse_parallel_blocks_one_emits_framed_stream() {
    assert_fse_single_frame_roundtrip(Some(1));
}

/// C1 (fails at v5.0.0): same unframed fallback for `parallel_blocks: Some(0)`.
#[test]
fn test_v2_fse_parallel_blocks_zero_emits_framed_stream() {
    assert_fse_single_frame_roundtrip(Some(0));
}

/// C1 guard (passes at v5.0.0): `Some(2)` still produces a decodable
/// parallel-mode frame with one block per `block_size` chunk.
#[test]
fn test_v2_fse_parallel_blocks_two_roundtrips_as_parallel_frame() {
    let data = fse_framing_payload();
    let cfg = FseConfig {
        block_size: 1024,
        parallel_blocks: Some(2),
        ..FseConfig::default()
    };
    let (compressed, decoded) = fse_roundtrip(&cfg, &data);
    assert_eq!(decoded.expect("Some(2) stream must decode"), data);
    assert_eq!(compressed[0], FSE_MODE_PARALLEL);
    let blocks = u32::from_le_bytes(compressed[1..5].try_into().unwrap()) as usize;
    assert_eq!(blocks, data.len().div_ceil(cfg.block_size));
}

/// C1 guard (passes at v5.0.0, and must keep passing after the bounded worker
/// pool lands): 64 blocks with only 4 requested workers. Pins the exact
/// v5.0.0 parallel wire format (block order, per-block encoding) by length and
/// FNV-1a digest, so the rewrite cannot reorder or re-encode blocks.
#[test]
fn test_v2_fse_parallel_wire_format_is_unchanged() {
    let data: Vec<u8> = (0..16_384u32)
        .map(|i| {
            (b"etaoin shrdlu cmfwyp"[(i as usize * 7 + i as usize / 11) % 20])
                ^ ((i / 997) as u8 & 1)
        })
        .collect();
    let cfg = FseConfig {
        block_size: 256,
        parallel_blocks: Some(4),
        ..FseConfig::default()
    };
    let (compressed, decoded) = fse_roundtrip(&cfg, &data);
    assert_eq!(decoded.expect("64-block stream must decode"), data);
    assert_eq!(compressed[0], FSE_MODE_PARALLEL);
    assert_eq!(
        u32::from_le_bytes(compressed[1..5].try_into().unwrap()),
        64,
        "one block per 256-byte chunk"
    );
    // Captured from v5.0.0 (tag v5.0.0, commit 4a5f4e5).
    assert_eq!(
        compressed.len(),
        V500_FSE_PARALLEL_LEN,
        "parallel wire format changed"
    );
    assert_eq!(
        fnv1a64(&compressed),
        V500_FSE_PARALLEL_FNV,
        "parallel wire format changed"
    );
}

const V500_FSE_PARALLEL_LEN: usize = 18_829;
const V500_FSE_PARALLEL_FNV: u64 = 0x1420_7cdb_c30b_be1f;

/// C1-adjacent (fails at v5.0.0): `block_size: 0` passes `FseConfig::validate`,
/// and with `parallel_blocks` set `compress` panics in `data.chunks(0)`
/// ("chunk size must be non-zero").
#[test]
fn test_v2_fse_zero_block_size_with_parallel_blocks_does_not_panic() {
    let data = fse_framing_payload();
    let cfg = FseConfig {
        block_size: 0,
        parallel_blocks: Some(2),
        ..FseConfig::default()
    };
    let (_, decoded) = catch_unwind(AssertUnwindSafe(|| fse_roundtrip(&cfg, &data)))
        .unwrap_or_else(|p| panic!("compress panicked: {}", panic_message(&*p)));
    assert_eq!(decoded.expect("stream must decode"), data);
}

/// C1-adjacent (fails at v5.0.0): `data.len() > block_size * 2` overflows for
/// `block_size > usize::MAX / 2` (debug: "attempt to multiply with overflow";
/// release: wraps to 0 and takes the unframed single-chunk fallback).
#[test]
fn test_v2_fse_huge_block_size_with_parallel_blocks_roundtrips() {
    let data = fse_framing_payload();
    let cfg = FseConfig {
        block_size: usize::MAX / 2 + 1,
        parallel_blocks: Some(2),
        ..FseConfig::default()
    };
    let (_, decoded) = catch_unwind(AssertUnwindSafe(|| fse_roundtrip(&cfg, &data)))
        .unwrap_or_else(|p| panic!("compress panicked: {}", panic_message(&*p)));
    assert_eq!(decoded.expect("stream must decode"), data);
}

/// C1-adjacent (fails at v5.0.0): `compress_parallel` never updated the
/// encoder's statistics, so `stats()` kept describing an earlier call.
#[test]
fn test_v2_fse_parallel_compress_updates_stats() {
    let data = fse_framing_payload();
    let cfg = FseConfig {
        block_size: 1024,
        parallel_blocks: Some(2),
        ..FseConfig::default()
    };
    let mut encoder = FseEncoder::new(cfg).unwrap();
    let compressed = encoder.compress(&data).unwrap();
    assert_eq!(compressed[0], FSE_MODE_PARALLEL);
    assert_eq!(encoder.stats().input_size, data.len());
    assert_eq!(encoder.stats().output_size, compressed.len());
}
