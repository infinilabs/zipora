//! Plan v2 (reliability-first) regression tests, verifier G3.
//!
//! - C1: FSE `compress_parallel` framing (`parallel_blocks` of `Some(0)` /
//!   `Some(1)` emitted an unframed stream) and its per-block thread spawn.
//! - C4: rANS / contextual-Huffman decoders preallocating a caller-supplied
//!   `output_length` before reading a single symbol.
//!
//! Every `test_v2_*` here compiles against v5.0.0; the doc comment of each test
//! states whether it fails there (and why) or is a guard that must keep passing.

use std::panic::{AssertUnwindSafe, catch_unwind};

use zipora::entropy::huffman::InterleavingFactor;
use zipora::entropy::rans::ParallelVariant as RansVariant;
use zipora::entropy::{
    ContextualHuffmanDecoder, ContextualHuffmanEncoder, FseConfig, FseDecoder, FseEncoder,
    HuffmanOrder, ParallelX1, ParallelX2, ParallelX4, ParallelX8, Rans64Encoder, RansDecoder,
};

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

// ---------------------------------------------------------------------------
// C4: caller-supplied output_length must not be preallocated blindly
// ---------------------------------------------------------------------------

/// A length `Vec` cannot represent: preallocating it panics with "capacity
/// overflow" (catchable) instead of aborting the process like a merely huge
/// allocation would.
const UNALLOCATABLE_LEN: usize = isize::MAX as usize + 1;

fn rans_frequencies(data: &[u8]) -> [u32; 256] {
    let mut freqs = [0u32; 256];
    for &b in data {
        freqs[b as usize] += 1;
    }
    freqs
}

/// Decode a tiny valid stream with a caller-supplied `output_length`.
/// `Err(msg)` means the decoder panicked.
fn rans_decode_tiny_with_len<P: RansVariant>(
    len: usize,
) -> Result<zipora::Result<Vec<u8>>, String> {
    let data = b"zipora rans c4: output_length is a caller argument".to_vec();
    let encoder = Rans64Encoder::<P>::new(&rans_frequencies(&data)).unwrap();
    let encoded = encoder.encode(&data).unwrap();
    let decoder = RansDecoder::new(&encoder);
    assert_eq!(
        decoder.decode(&encoded, data.len()).unwrap(),
        data,
        "control round trip ({})",
        P::NAME
    );
    catch_unwind(AssertUnwindSafe(|| decoder.decode(&encoded, len))).map_err(|p| panic_message(&*p))
}

/// C4 (fails at v5.0.0 for x2/x4/x8): `decode_parallel` runs
/// `vec![0u8; output_length]` before decoding anything, so a 75-byte x4 stream
/// with an absurd `output_length` panics with "capacity overflow" (and a
/// merely huge one commits gigabytes). x1 already capped its reservation.
#[test]
fn test_v2_rans_decode_unallocatable_output_length_is_err() {
    let mut failures = Vec::new();
    for (name, outcome) in [
        (
            "x1",
            rans_decode_tiny_with_len::<ParallelX1>(UNALLOCATABLE_LEN),
        ),
        (
            "x2",
            rans_decode_tiny_with_len::<ParallelX2>(UNALLOCATABLE_LEN),
        ),
        (
            "x4",
            rans_decode_tiny_with_len::<ParallelX4>(UNALLOCATABLE_LEN),
        ),
        (
            "x8",
            rans_decode_tiny_with_len::<ParallelX8>(UNALLOCATABLE_LEN),
        ),
    ] {
        match outcome {
            Ok(Err(_)) => {}
            Ok(Ok(v)) => failures.push(format!("{name}: returned {} bytes", v.len())),
            Err(msg) => failures.push(format!("{name}: panicked: {msg}")),
        }
    }
    assert!(failures.is_empty(), "{failures:?}");
}

fn rans_roundtrip_len<P: RansVariant>(data: &[u8]) {
    let encoder = Rans64Encoder::<P>::new(&rans_frequencies(data)).unwrap();
    let encoded = encoder.encode(data).unwrap();
    let decoded = RansDecoder::new(&encoder)
        .decode(&encoded, data.len())
        .unwrap_or_else(|e| panic!("{} decode of {} bytes failed: {e}", P::NAME, data.len()));
    assert!(
        decoded == data,
        "{} round trip of {} bytes differs",
        P::NAME,
        data.len()
    );
}

/// C4 guard (passes at v5.0.0): legitimate outputs larger than the fixed
/// decoder's 1 MiB up-front reservation, with lengths that are not a multiple
/// of the stream count, still round-trip; so does a single-symbol stream whose
/// symbols cost zero bits (tiny input, multi-MiB output).
#[test]
fn test_v2_rans_parallel_decode_large_output_roundtrips() {
    let len = (3 << 20) + 3;
    let mixed: Vec<u8> = (0..len)
        .map(|i| b"abcdefgh"[(i * 7 + i / 13) % 8])
        .collect();
    rans_roundtrip_len::<ParallelX2>(&mixed);
    rans_roundtrip_len::<ParallelX4>(&mixed);
    rans_roundtrip_len::<ParallelX8>(&mixed);

    let single = vec![b'z'; (2 << 20) + 5];
    rans_roundtrip_len::<ParallelX4>(&single);
    rans_roundtrip_len::<ParallelX8>(&single);
}

/// C4 (fails at v5.0.0): `ContextualHuffmanDecoder::decode` (orders 0/1/2) and
/// the interleaved `decode_x1..x8` reserve `output_length` up front
/// (`Vec::with_capacity` / `vec![0; n]` / `resize`), so they panic with
/// "capacity overflow" for an absurd caller-supplied length.
#[test]
fn test_v2_contextual_huffman_unallocatable_output_length_is_err() {
    let text =
        b"zipora contextual huffman c4 regression: output_length is caller supplied".to_vec();
    let mut failures = Vec::new();

    for order in [
        HuffmanOrder::Order0,
        HuffmanOrder::Order1,
        HuffmanOrder::Order2,
    ] {
        let encoder = ContextualHuffmanEncoder::new(&text, order).unwrap();
        let encoded = encoder.encode(&text).unwrap();
        let decoder = ContextualHuffmanDecoder::new(encoder);
        assert_eq!(
            decoder.decode(&encoded, text.len()).unwrap(),
            text,
            "control {order:?}"
        );
        match catch_unwind(AssertUnwindSafe(|| {
            decoder.decode(&encoded, UNALLOCATABLE_LEN)
        })) {
            Ok(Err(_)) => {}
            Ok(Ok(v)) => failures.push(format!("decode {order:?}: returned {} bytes", v.len())),
            Err(p) => failures.push(format!(
                "decode {order:?}: panicked: {}",
                panic_message(&*p)
            )),
        }
    }

    let encoder = ContextualHuffmanEncoder::new(&text, HuffmanOrder::Order1).unwrap();
    for factor in [
        InterleavingFactor::X1,
        InterleavingFactor::X2,
        InterleavingFactor::X4,
        InterleavingFactor::X8,
    ] {
        let encoded = encoder.encode_with_interleaving(&text, factor).unwrap();
        assert_eq!(
            encoder
                .decode_with_interleaving(&encoded, text.len(), factor)
                .unwrap(),
            text,
            "control {factor:?}"
        );
        match catch_unwind(AssertUnwindSafe(|| {
            encoder.decode_with_interleaving(&encoded, UNALLOCATABLE_LEN, factor)
        })) {
            Ok(Err(_)) => {}
            Ok(Ok(v)) => failures.push(format!("{factor:?}: returned {} bytes", v.len())),
            Err(p) => failures.push(format!("{factor:?}: panicked: {}", panic_message(&*p))),
        }
    }

    assert!(failures.is_empty(), "{failures:?}");
}
