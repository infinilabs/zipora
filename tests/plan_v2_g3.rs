//! Plan v2 (reliability-first) regression tests, verifier G3.
//!
//! - C1: FSE `compress_parallel` framing (`parallel_blocks` of `Some(0)` /
//!   `Some(1)` emitted an unframed stream) and its per-block thread spawn.
//! - C4: rANS / contextual-Huffman decoders preallocating a caller-supplied
//!   `output_length` before reading a single symbol.
//! - C2: PA-Zip `use_reference_encoding` writes streams `decompress` cannot read
//!   (data loss), plus `DictZipBlobStore::get` cloning the whole compressor.
//!
//! Every `test_v2_*` here compiles against v5.0.0; the doc comment of each test
//! states whether it fails there (and why) or is a guard that must keep passing.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::panic::{AssertUnwindSafe, catch_unwind};

use zipora::blob_store::{BlobStore, CompressedBlobStore};
use zipora::compression::dict_zip::{
    DictZipBlobStoreBuilder, DictZipConfig, DictionaryBuilder, DictionaryBuilderConfig,
    EntropyAlgorithm, PaZipCompressor, PaZipCompressorConfig,
};
use zipora::entropy::huffman::InterleavingFactor;
use zipora::entropy::rans::ParallelVariant as RansVariant;
use zipora::entropy::{
    ContextualHuffmanDecoder, ContextualHuffmanEncoder, FseConfig, FseDecoder, FseEncoder,
    HuffmanOrder, ParallelX1, ParallelX2, ParallelX4, ParallelX8, Rans64Encoder, RansDecoder,
};
use zipora::memory::{SecureMemoryPool, SecurePoolConfig};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Per-thread allocation counter, so tests running in parallel do not see
/// each other's allocations. Counting never changes allocation behaviour.
struct CountingAlloc;

thread_local! {
    static ALLOCATED_BYTES: Cell<usize> = const { Cell::new(0) };
}

fn note_alloc(size: usize) {
    let _ = ALLOCATED_BYTES.try_with(|c| c.set(c.get().saturating_add(size)));
}

// SAFETY: every call is forwarded unchanged to `System`; the only extra work
// is bumping a const-initialised thread-local `Cell`, which never allocates.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note_alloc(layout.size());
        // SAFETY: same contract as the caller's.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        note_alloc(layout.size());
        // SAFETY: same contract as the caller's.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        note_alloc(new_size);
        // SAFETY: same contract as the caller's.
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: same contract as the caller's.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

/// Bytes allocated by the current thread while running `f`.
fn bytes_allocated_by<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let before = ALLOCATED_BYTES.with(Cell::get);
    let out = f();
    let after = ALLOCATED_BYTES.with(Cell::get);
    (out, after - before)
}

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

fn hex_decode(hex: &str) -> Vec<u8> {
    let digits: Vec<u8> = hex.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    assert_eq!(digits.len() % 2, 0, "odd number of hex digits");
    digits
        .chunks(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
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
    assert_eq!(
        decoded.capacity(),
        data.len(),
        "{} decoded buffer capacity should match exact output length",
        P::NAME
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

// ---------------------------------------------------------------------------
// C2: PA-Zip reference mode (data loss) and DictZipBlobStore::get clone
// ---------------------------------------------------------------------------

const JSON_RECORD: &[u8] = br#"{"service":"zipora-search","status":200,"region":"us-east1"}"#;

fn json_payload() -> Vec<u8> {
    JSON_RECORD.repeat(16)
}

fn pazip_compressor(config: PaZipCompressorConfig) -> PaZipCompressor {
    let dictionary = DictionaryBuilder::with_config(DictionaryBuilderConfig::default())
        .build(&json_payload())
        .expect("dictionary");
    let pool = SecureMemoryPool::new(SecurePoolConfig::new(4096, 1024, 8)).expect("pool");
    PaZipCompressor::new(dictionary, config, pool).expect("compressor")
}

/// Decode through a `&mut` binding so this file compiles (warning-free) both
/// before and after `decompress` moves to `&self`.
fn pazip_decompress(compressor: &mut PaZipCompressor, stream: &[u8]) -> zipora::Result<Vec<u8>> {
    let mut out = Vec::new();
    compressor.decompress(stream, &mut out).map(|()| out)
}

fn reference_configs() -> [(&'static str, PaZipCompressorConfig); 2] {
    [
        (
            "reference_compliant (suffix array)",
            PaZipCompressorConfig::reference_compliant(),
        ),
        (
            "use_reference_encoding (hash table)",
            PaZipCompressorConfig {
                use_reference_encoding: true,
                use_suffix_array_local_match: false,
                ..PaZipCompressorConfig::default()
            },
        ),
    ]
}

/// C2 (fails at v5.0.0, DATA LOSS): with `use_reference_encoding` the encoder
/// writes reference-format instructions that `decompress` cannot parse
/// ("Invalid PA-Zip compression instruction type"); the 960-byte payload also
/// grows to 990 bytes and `bytes_output` reports the input length (960).
#[test]
fn test_v2_pazip_reference_encoding_configs_roundtrip() {
    let input = json_payload();
    for (name, config) in reference_configs() {
        let mut compressor = pazip_compressor(config);
        let mut compressed = Vec::new();
        let stats = compressor
            .compress(&input, &mut compressed)
            .expect("compress");
        let decoded = pazip_decompress(&mut compressor, &compressed).unwrap_or_else(|e| {
            panic!(
                "{name}: {}-byte stream does not decode: {e}",
                compressed.len()
            )
        });
        assert_eq!(decoded, input, "{name}: round trip differs");
        assert_eq!(
            stats.bytes_output,
            compressed.len() as u64,
            "{name}: bytes_output"
        );
        assert!(
            compressed.len() < input.len(),
            "{name}: {} -> {} bytes, highly repetitive input must shrink",
            input.len(),
            compressed.len()
        );
    }
}

/// C2 guard (passes at v5.0.0): the default (legacy) encoder round-trips the
/// same payload plus edge-case sizes.
#[test]
fn test_v2_pazip_legacy_default_roundtrips() {
    let mut compressor = pazip_compressor(PaZipCompressorConfig::default());
    let inputs: [Vec<u8>; 7] = [
        json_payload(),
        JSON_RECORD.to_vec(),
        vec![0u8],
        vec![0x41],
        b"AB".to_vec(),
        (0..=255u8).collect(),
        b"abc".repeat(100),
    ];
    for input in inputs {
        let mut compressed = Vec::new();
        compressor
            .compress(&input, &mut compressed)
            .expect("compress");
        let decoded =
            pazip_decompress(&mut compressor, &compressed).expect("legacy stream must decode");
        assert_eq!(decoded, input);
    }
}

/// v5.0.0 reference-mode layout for `data` (its match selection never accepted
/// a match, so the stream is reference literal runs only):
/// `[(len - 1) << 3][len bytes]` per 32-byte chunk, shorter final chunk.
fn v500_reference_layout(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() + data.len().div_ceil(32));
    for chunk in data.chunks(32) {
        out.push(((chunk.len() - 1) << 3) as u8);
        out.extend_from_slice(chunk);
    }
    out
}

/// Captured from the v5.0.0 encoder (tag v5.0.0, commit 4a5f4e5):
/// `PaZipCompressorConfig::reference_compliant()` compressing `JSON_RECORD`.
const V500_REFERENCE_JSON_RECORD_HEX: &str = "\
    f87b2273657276696365223a227a69706f72612d736561726368222c2273746174\
    d87573223a3230302c22726567696f6e223a2275732d6561737431227d";

/// C2 recovery guard (passes at v5.0.0): the captured v5.0.0 reference stream
/// is exactly the literal-run layout, so it can be recognized structurally
/// (first byte 0xF8 here, never a legacy opcode 0..=7).
#[test]
fn test_v2_pazip_v500_reference_golden_is_literal_layout() {
    let golden = hex_decode(V500_REFERENCE_JSON_RECORD_HEX);
    assert_eq!(golden, v500_reference_layout(JSON_RECORD));
    assert_eq!(golden[0], 0xF8);
}

/// C2 recovery (fails at v5.0.0): streams written by v5.0.0 in reference mode
/// must decode to the original bytes ("Invalid PA-Zip compression instruction
/// type" at v5.0.0). Covers the captured golden stream and every chunking
/// shape, including the 1-byte stream `[0x00, b]` (whose `b == 0` form v5.0.0
/// silently decoded to an empty output).
#[test]
fn test_v2_pazip_decodes_v500_reference_streams() {
    let mut compressor = pazip_compressor(PaZipCompressorConfig::default());
    let golden = hex_decode(V500_REFERENCE_JSON_RECORD_HEX);
    let decoded = pazip_decompress(&mut compressor, &golden)
        .unwrap_or_else(|e| panic!("captured v5.0.0 reference stream does not decode: {e}"));
    assert_eq!(decoded, JSON_RECORD);

    let payload = json_payload();
    let mut cases: Vec<Vec<u8>> = [1usize, 2, 7, 31, 32, 33, 64, 65, 960]
        .iter()
        .map(|&n| payload[..n].to_vec())
        .collect();
    cases.push(vec![0x00]);
    cases.push(vec![0xF8; 40]);
    let mut failures = Vec::new();
    for data in cases {
        match pazip_decompress(&mut compressor, &v500_reference_layout(&data)) {
            Ok(out) if out == data => {}
            Ok(out) => failures.push(format!(
                "{} bytes: decoded {} different bytes",
                data.len(),
                out.len()
            )),
            Err(e) => failures.push(format!("{} bytes: {e}", data.len())),
        }
    }
    assert!(failures.is_empty(), "{failures:?}");
}

/// C2 recovery guard (passes at v5.0.0): only streams that match the v5.0.0
/// layout exactly are recovered; near misses keep failing.
#[test]
fn test_v2_pazip_v500_recovery_rejects_near_miss_streams() {
    let mut compressor = pazip_compressor(PaZipCompressorConfig::default());
    let good = v500_reference_layout(&json_payload()[..70]); // 32 + 32 + 6
    let near_misses = [
        // Truncated final run.
        good[..good.len() - 1].to_vec(),
        // Trailing garbage after the final run.
        [good.as_slice(), &[0x00]].concat(),
        // Non-final run shorter than 32 bytes.
        [v500_reference_layout(&json_payload()[..20]), good.clone()].concat(),
        // Header with a non-literal type in its low bits.
        [&[0xF9], &good[1..]].concat(),
    ];
    for stream in near_misses {
        assert!(
            pazip_decompress(&mut compressor, &stream).is_err(),
            "near-miss stream of {} bytes (first byte {:#04x}) must not decode",
            stream.len(),
            stream[0]
        );
    }
}

fn dict_zip_store_config() -> DictZipConfig {
    DictZipConfig {
        dict_builder_config: DictionaryBuilderConfig {
            target_dict_size: 1024,
            max_dict_size: 8192,
            validate_result: false,
            ..DictionaryBuilderConfig::default()
        },
        min_compression_size: 10,
        ..DictZipConfig::default()
    }
}

/// C2 (fails at v5.0.0, DATA LOSS through the blob store): reference-mode
/// output is only stored when an entropy layer shrinks it below the original
/// size, after which `get` fails with "Decompression failed: Invalid PA-Zip
/// compression instruction type".
#[test]
fn test_v2_dict_zip_store_reference_config_with_entropy_roundtrips() {
    let config = DictZipConfig {
        compressor_config: PaZipCompressorConfig::reference_compliant(),
        entropy_algorithm: EntropyAlgorithm::HuffmanO1,
        entropy_interleaved: 1,
        entropy_zip_ratio_require: 0.95,
        ..dict_zip_store_config()
    };
    let mut builder = DictZipBlobStoreBuilder::with_config(config).unwrap();
    builder.add_training_sample(&json_payload()).unwrap();
    let mut store = builder.finish().unwrap();
    let record = json_payload();
    let id = store.put(&record).unwrap();
    let got = store
        .get(id)
        .unwrap_or_else(|e| panic!("record written with reference_compliant() is unreadable: {e}"));
    assert_eq!(got, record);
}

/// C2 guard for the `get` clone removal (passes at v5.0.0): compressed records
/// read back byte-exactly, repeatedly and in reverse order.
#[test]
fn test_v2_dict_zip_store_get_returns_original_bytes() {
    let mut builder = DictZipBlobStoreBuilder::with_config(dict_zip_store_config()).unwrap();
    builder.add_training_sample(&json_payload()).unwrap();
    let mut store = builder.finish().unwrap();
    let records: Vec<Vec<u8>> = (1..=12).map(|k| JSON_RECORD.repeat(k)).collect();
    let ids: Vec<_> = records.iter().map(|r| store.put(r).unwrap()).collect();
    assert!(
        store.compression_stats().compressed_count > 0,
        "compressed path not exercised"
    );
    for _ in 0..2 {
        for (id, record) in ids.iter().zip(&records).rev() {
            assert_eq!(&store.get(*id).unwrap(), record);
        }
    }
}

/// C2 perf (fails at v5.0.0): every cache-miss `get` deep-clones the whole
/// `PaZipCompressor` (DFA cache, matcher state, last compressed output)
/// because `decompress` took `&mut self`; with a 64 KiB dictionary that is
/// ~290 KB of allocation to read a 120-byte record.
#[test]
fn test_v2_dict_zip_store_get_does_not_clone_compressor() {
    let config = DictZipConfig {
        dict_builder_config: DictionaryBuilderConfig {
            target_dict_size: 64 * 1024,
            max_dict_size: 256 * 1024,
            validate_result: false,
            ..DictionaryBuilderConfig::default()
        },
        min_compression_size: 10,
        ..DictZipConfig::default()
    };
    let mut builder = DictZipBlobStoreBuilder::with_config(config).unwrap();
    let corpus: Vec<u8> = (0..2000)
        .flat_map(|i| {
            format!(r#"{{"id":{i},"service":"zipora-search","status":200}}"#).into_bytes()
        })
        .collect();
    builder.add_training_sample(&corpus).unwrap();
    let mut store = builder.finish().unwrap();
    for k in 1..=16 {
        store.put(&JSON_RECORD.repeat(k)).unwrap();
    }
    let record = JSON_RECORD.repeat(2);
    let id = store.put(&record).unwrap();
    let (got, allocated) = bytes_allocated_by(|| store.get(id).unwrap());
    assert_eq!(got, record);
    assert!(
        allocated < 32 * 1024,
        "cache-miss get of a {}-byte record allocated {allocated} bytes",
        record.len()
    );
}

/// C2b (fails at v5.0.0, DATA LOSS in the default config): inputs of at least
/// 1 MiB take `compress_parallel`, which runs the legacy encoder once per
/// 64 KiB block without clearing `output_buffer`, so block k's output repeats
/// blocks 1..k (a 1,052,672-byte probe grew to 10,044,034 bytes and decoded to
/// 9,965,568 wrong bytes). `enable_multithreading: true` is the default.
#[test]
fn test_v2_pazip_default_config_large_input_roundtrips() {
    let mut compressor = pazip_compressor(PaZipCompressorConfig::default());
    let input = JSON_RECORD.repeat(17_500); // 1,050,000 bytes > 1 MiB threshold
    assert!(input.len() >= 1 << 20);
    let mut compressed = Vec::new();
    compressor
        .compress(&input, &mut compressed)
        .expect("compress");
    let decoded = pazip_decompress(&mut compressor, &compressed).expect("decode");
    assert_eq!(
        decoded.len(),
        input.len(),
        "{} -> {} bytes",
        input.len(),
        compressed.len()
    );
    assert!(decoded == input, "decoded bytes differ");
    assert!(
        compressed.len() < input.len(),
        "{} -> {} bytes",
        input.len(),
        compressed.len()
    );
}
