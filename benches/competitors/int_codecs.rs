//! Family 7: Integer Codecs (100,000 contiguous sorted `u32` integers by default).
//!
//! Zipora: `StreamVByte`, `GroupVarint`
//! Competitors: `stream-vbyte` (`Scalar` and stable-Rust SSSE3/SSE4.1 `SvbCrateSimd`),
//!              `bitpacking` (`BitPacker4x`)
//!
//! Fairness invariant (`S42-R1`): every implementation is benchmarked on **1 instance**
//! and **1 shared `sorted_u32` input slice**, with zero 8-clone input rings or heap spacers.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m128i, _mm_cvtsi128_si64, _mm_loadu_si128, _mm_min_epu8, _mm_mullo_epi32, _mm_shuffle_epi8,
    _mm_storeu_si128,
};
use std::collections::BTreeMap;
use std::hint::black_box;

use bitpacking::{BitPacker, BitPacker4x};
use stream_vbyte::{
    decode::{DecodeQuadSink, Decoder, WriteQuadToSlice, decode as svb_crate_decode},
    encode::{Encoder, encode as svb_crate_encode},
    scalar::Scalar as SvbCrateScalar,
};
use zipora::compression::stream_vbyte::{GroupVarint, StreamVByte};

use crate::common::*;

/// Compile-time lookup tables matching `stream-vbyte` v0.4.1's `src/tables.rs`
/// (`control_byte` layout: `(len0 - 1) | ((len1 - 1) << 2) | ((len2 - 1) << 4) | ((len3 - 1) << 6)`).
const SVB_CRATE_DECODE_LEN_TABLE: [u8; 256] = {
    let mut table = [0u8; 256];
    let mut code = 0usize;
    while code < 256 {
        let l0 = ((code & 0b11) + 1) as u8;
        let l1 = (((code >> 2) & 0b11) + 1) as u8;
        let l2 = (((code >> 4) & 0b11) + 1) as u8;
        let l3 = (((code >> 6) & 0b11) + 1) as u8;
        table[code] = l0 + l1 + l2 + l3;
        code += 1;
    }
    table
};

const SVB_CRATE_DECODE_SHUFFLE_TABLE: [[u8; 16]; 256] = {
    let mut table = [[0x80u8; 16]; 256];
    let mut code = 0usize;
    while code < 256 {
        let lens = [
            ((code & 0b11) + 1) as u8,
            (((code >> 2) & 0b11) + 1) as u8,
            (((code >> 4) & 0b11) + 1) as u8,
            (((code >> 6) & 0b11) + 1) as u8,
        ];
        let mut src_idx = 0u8;
        let mut lane = 0usize;
        while lane < 4 {
            let mut b = 0u8;
            while b < lens[lane] {
                table[code][lane * 4 + (b as usize)] = src_idx;
                src_idx += 1;
                b += 1;
            }
            lane += 1;
        }
        code += 1;
    }
    table
};

const SVB_CRATE_ENCODE_SHUFFLE_TABLE: [[u8; 16]; 256] = {
    let mut table = [[0x80u8; 16]; 256];
    let mut code = 0usize;
    while code < 256 {
        let lens = [
            ((code & 0b11) + 1) as u8,
            (((code >> 2) & 0b11) + 1) as u8,
            (((code >> 4) & 0b11) + 1) as u8,
            (((code >> 6) & 0b11) + 1) as u8,
        ];
        let mut dst_idx = 0usize;
        let mut lane = 0usize;
        while lane < 4 {
            let mut b = 0u8;
            while b < lens[lane] {
                table[code][dst_idx] = (lane as u8) * 4 + b;
                dst_idx += 1;
                b += 1;
            }
            lane += 1;
        }
        code += 1;
    }
    table
};

/// Stable-Rust port of `stream-vbyte` v0.4.1's `x86::Sse41` encoder and `x86::Ssse3` decoder
/// (which upstream gates behind nightly `#![feature(portable_simd)]` solely to index two bytes
/// out of an `__m128i`). Implements `stream_vbyte::encode::Encoder`, `stream_vbyte::decode::Decoder`,
/// and `stream_vbyte::decode::WriteQuadToSlice` so it plugs directly into `stream_vbyte::encode::encode`
/// and `stream_vbyte::decode::decode`.
#[cfg(target_arch = "x86_64")]
struct SvbCrateSimd;

#[cfg(target_arch = "x86_64")]
impl Encoder for SvbCrateSimd {
    #[inline]
    fn encode_quads(input: &[u32], control_bytes: &mut [u8], output: &mut [u8]) -> (usize, usize) {
        const ONES: [u8; 16] = [1; 16];
        const SHIFT: u32 = 1 | (1 << 9) | (1 << 18);
        const SHIFTS: [u32; 4] = [SHIFT; 4];
        const LANECODES: [u8; 16] = [
            0, 3, 2, 3, 1, 3, 2, 3, 128, 128, 128, 128, 128, 128, 128, 128,
        ];
        const GATHER_HI: [u8; 16] = [
            15, 11, 7, 3, 15, 11, 7, 3, 128, 128, 128, 128, 128, 128, 128, 128,
        ];
        const CONCAT: u32 = 1 | (1 << 10) | (1 << 20) | (1 << 30);
        const SUM: u32 = 1 | (1 << 8) | (1 << 16) | (1 << 24);
        const AGGREGATORS: [u32; 4] = [CONCAT, SUM, 0, 0];

        let mut nums_encoded = 0usize;
        let mut bytes_encoded = 0usize;

        // SAFETY: x86_64 baseline + SSSE3/SSE4.1 intrinsics on 16-byte stack constants and
        // slice-checked 16-byte windows (`control_bytes.len().saturating_sub(3)` guarantees at
        // least 16 remaining output bytes per quad).
        unsafe {
            let ones = _mm_loadu_si128(ONES.as_ptr().cast::<__m128i>());
            let shifts = _mm_loadu_si128(SHIFTS.as_ptr().cast::<__m128i>());
            let lanecodes = _mm_loadu_si128(LANECODES.as_ptr().cast::<__m128i>());
            let gather_hi = _mm_loadu_si128(GATHER_HI.as_ptr().cast::<__m128i>());
            let aggregators = _mm_loadu_si128(AGGREGATORS.as_ptr().cast::<__m128i>());

            let control_byte_limit = control_bytes.len().saturating_sub(3);
            for control_byte in &mut control_bytes[..control_byte_limit] {
                let to_encode = _mm_loadu_si128(
                    input[nums_encoded..nums_encoded + 4]
                        .as_ptr()
                        .cast::<__m128i>(),
                );
                let mins = _mm_min_epu8(to_encode, ones);
                let bytemaps = _mm_mullo_epi32(mins, shifts);
                let shuffled_lanecodes = _mm_shuffle_epi8(lanecodes, bytemaps);
                let hi_bytes = _mm_shuffle_epi8(shuffled_lanecodes, gather_hi);
                let code_and_length = _mm_mullo_epi32(hi_bytes, aggregators);
                let low64 = _mm_cvtsi128_si64(code_and_length) as u64;
                let code = (low64 >> 24) as u8;
                let length = ((low64 >> 56) as u8) + 4;

                let mask_bytes = &SVB_CRATE_ENCODE_SHUFFLE_TABLE[code as usize];
                let encode_mask = _mm_loadu_si128(mask_bytes.as_ptr().cast::<__m128i>());
                let encoded = _mm_shuffle_epi8(to_encode, encode_mask);
                _mm_storeu_si128(
                    output[bytes_encoded..bytes_encoded + 16]
                        .as_mut_ptr()
                        .cast::<__m128i>(),
                    encoded,
                );

                *control_byte = code;
                bytes_encoded += length as usize;
                nums_encoded += 4;
            }
        }

        (nums_encoded, bytes_encoded)
    }
}

#[cfg(target_arch = "x86_64")]
impl Decoder for SvbCrateSimd {
    type DecodedQuad = __m128i;

    #[inline]
    fn decode_quads<S: DecodeQuadSink<Self>>(
        control_bytes: &[u8],
        encoded_nums: &[u8],
        control_bytes_to_decode: usize,
        nums_already_decoded: usize,
        sink: &mut S,
    ) -> (usize, usize) {
        let mut bytes_read = 0usize;
        let mut nums_decoded = nums_already_decoded;
        let control_byte_limit = control_bytes_to_decode.min(control_bytes.len().saturating_sub(3));

        for &control_byte in &control_bytes[..control_byte_limit] {
            let length = SVB_CRATE_DECODE_LEN_TABLE[control_byte as usize];
            let mask_bytes = &SVB_CRATE_DECODE_SHUFFLE_TABLE[control_byte as usize];
            let next_4 = &encoded_nums[bytes_read..bytes_read + 16];
            // SAFETY: `next_4` is slice-checked to 16 bytes and `mask_bytes` is `[u8; 16]`.
            let decompressed = unsafe {
                let mask = _mm_loadu_si128(mask_bytes.as_ptr().cast::<__m128i>());
                let data = _mm_loadu_si128(next_4.as_ptr().cast::<__m128i>());
                _mm_shuffle_epi8(data, mask)
            };
            sink.on_quad(decompressed, nums_decoded);
            bytes_read += length as usize;
            nums_decoded += 4;
        }

        (nums_decoded - nums_already_decoded, bytes_read)
    }
}

#[cfg(target_arch = "x86_64")]
impl WriteQuadToSlice for SvbCrateSimd {
    #[inline]
    fn write_quad_to_slice(quad: Self::DecodedQuad, slice: &mut [u32]) {
        assert!(slice.len() >= 4);
        // SAFETY: `slice` has at least 4 `u32`s (16 bytes).
        unsafe {
            _mm_storeu_si128(slice.as_mut_ptr().cast::<__m128i>(), quad);
        }
    }
}

pub fn bench_int_codecs(
    records: &mut Vec<BenchRecord>,
    space: &mut BTreeMap<String, f64>,
    sizes: SuiteSizes,
) {
    trim_allocator_arena();
    let n = sizes.int_codec_elements;
    let sorted_u32 = gen_uniform_sparse(n, 120, SEED_BASE ^ 0x08);
    let mut params = BenchParams::fast(n as u64, n as u64).with_threshold(9.0);
    params.trials = 11;
    params.bursts_per_trial = 8;

    let mut svb_into_params = BenchParams::fast(n as u64, (n * 32) as u64).with_threshold(9.0);
    svb_into_params.warmup_calls = 2;
    svb_into_params.trials = 11;
    svb_into_params.bursts_per_trial = 4;

    let mut svb_deltas_params = BenchParams::fast(n as u64, (n * 16) as u64).with_threshold(9.0);
    svb_deltas_params.warmup_calls = 2;
    svb_deltas_params.trials = 11;
    svb_deltas_params.bursts_per_trial = 4;

    // 1. Zipora StreamVByte (1 instance)
    {
        let svb_stream = StreamVByte::encode_deltas(&sorted_u32);
        record_space_metric(
            space,
            "int_codecs/stream_vbyte_bits_per_int",
            ((svb_stream.controls.len() + svb_stream.data.len()) as f64 * 8.0) / (n as f64),
        );
        let mut svb_out = vec![0u32; n];

        measure_bench(
            records,
            "int_codecs",
            "int_codecs/stream_vbyte/encode_deltas",
            "ns/int",
            params,
            || {
                let encoded = StreamVByte::encode_deltas(black_box(&sorted_u32));
                black_box(encoded.data.len());
            },
        );

        measure_bench(
            records,
            "int_codecs",
            "int_codecs/stream_vbyte/decode_into",
            "ns/int",
            svb_into_params,
            || {
                for _ in 0..32usize {
                    let decoded = StreamVByte::decode_into(black_box(&svb_stream), n, &mut svb_out);
                    black_box(decoded);
                }
            },
        );

        measure_bench(
            records,
            "int_codecs",
            "int_codecs/stream_vbyte/decode_deltas",
            "ns/int",
            svb_deltas_params,
            || {
                for _ in 0..16usize {
                    let decoded = StreamVByte::decode_deltas(black_box(&svb_stream), n);
                    black_box(decoded.len());
                }
            },
        );
    }

    // 2. Zipora GroupVarint (1 instance, 1 shared `sorted_u32` input slice)
    {
        let gv_encoded = GroupVarint::encode_deltas(&sorted_u32);
        record_space_metric(
            space,
            "int_codecs/group_varint_bits_per_int",
            (gv_encoded.len() as f64 * 8.0) / (n as f64),
        );

        measure_bench(
            records,
            "int_codecs",
            "int_codecs/group_varint/encode_deltas",
            "ns/int",
            params,
            || {
                let encoded = GroupVarint::encode_deltas(black_box(&sorted_u32));
                black_box(encoded.len());
            },
        );

        measure_bench(
            records,
            "int_codecs",
            "int_codecs/group_varint/decode_deltas",
            "ns/int",
            params,
            || {
                let decoded = GroupVarint::decode_deltas(black_box(&gv_encoded), n);
                black_box(decoded.len());
            },
        );
    }

    // 3. Competitor: stream-vbyte crate (v0.4.1 Scalar + delta pass, 1 instance)
    {
        let mut init_deltas = Vec::with_capacity(n);
        let mut prev = 0u32;
        for &v in &sorted_u32 {
            init_deltas.push(v.wrapping_sub(prev));
            prev = v;
        }
        let mut svb_crate_encoded = vec![0u8; n.div_ceil(4) + n * 4];
        let enc_len = svb_crate_encode::<SvbCrateScalar>(&init_deltas, &mut svb_crate_encoded);
        svb_crate_encoded.truncate(enc_len);
        record_space_metric(
            space,
            "int_codecs/stream_vbyte_crate_bits_per_int",
            (svb_crate_encoded.len() as f64 * 8.0) / (n as f64),
        );

        {
            let mut svb_crate_out = vec![0u32; n];

            measure_bench(
                records,
                "int_codecs",
                "int_codecs/stream_vbyte_crate/encode_deltas",
                "ns/int",
                params,
                || {
                    let src = black_box(&sorted_u32);
                    let mut deltas = Vec::with_capacity(n);
                    let mut p = 0u32;
                    for &v in src {
                        deltas.push(v.wrapping_sub(p));
                        p = v;
                    }
                    let mut buf = vec![0u8; n.div_ceil(4) + n * 4];
                    let written = svb_crate_encode::<SvbCrateScalar>(&deltas, &mut buf);
                    buf.truncate(written);
                    black_box(buf.len());
                },
            );

            measure_bench(
                records,
                "int_codecs",
                "int_codecs/stream_vbyte_crate/decode_into",
                "ns/int",
                params,
                || {
                    let consumed = svb_crate_decode::<SvbCrateScalar>(
                        black_box(&svb_crate_encoded),
                        n,
                        &mut svb_crate_out,
                    );
                    black_box(consumed);
                },
            );

            measure_bench(
                records,
                "int_codecs",
                "int_codecs/stream_vbyte_crate/decode_deltas",
                "ns/int",
                params,
                || {
                    let mut out = vec![0u32; n];
                    let consumed = svb_crate_decode::<SvbCrateScalar>(
                        black_box(&svb_crate_encoded),
                        n,
                        &mut out,
                    );
                    let mut acc = 0u32;
                    for v in &mut out {
                        acc = acc.wrapping_add(*v);
                        *v = acc;
                    }
                    black_box(consumed + out.len());
                },
            );
        }

        #[cfg(target_arch = "x86_64")]
        {
            // Verify bit-exact parity between SvbCrateSimd (SSE4.1/SSSE3) and SvbCrateScalar.
            let mut simd_encoded = vec![0u8; n.div_ceil(4) + n * 4];
            let simd_len = svb_crate_encode::<SvbCrateSimd>(&init_deltas, &mut simd_encoded);
            simd_encoded.truncate(simd_len);
            assert_eq!(simd_encoded, svb_crate_encoded);
            {
                let mut simd_check_out = vec![0u32; n];
                let simd_consumed =
                    svb_crate_decode::<SvbCrateSimd>(&simd_encoded, n, &mut simd_check_out);
                assert_eq!(simd_consumed, simd_encoded.len());
                assert_eq!(simd_check_out, init_deltas);
            }

            record_space_metric(
                space,
                "int_codecs/stream_vbyte_simd_bits_per_int",
                (simd_encoded.len() as f64 * 8.0) / (n as f64),
            );

            let mut svb_simd_out = vec![0u32; n];

            measure_bench(
                records,
                "int_codecs",
                "int_codecs/stream_vbyte_simd/encode_deltas",
                "ns/int",
                params,
                || {
                    let src = black_box(&sorted_u32);
                    let mut deltas = Vec::with_capacity(n);
                    let mut p = 0u32;
                    for &v in src {
                        deltas.push(v.wrapping_sub(p));
                        p = v;
                    }
                    let mut buf = vec![0u8; n.div_ceil(4) + n * 4];
                    let written = svb_crate_encode::<SvbCrateSimd>(&deltas, &mut buf);
                    buf.truncate(written);
                    black_box(buf.len());
                },
            );

            measure_bench(
                records,
                "int_codecs",
                "int_codecs/stream_vbyte_simd/decode_into",
                "ns/int",
                svb_into_params,
                || {
                    for _ in 0..32usize {
                        let consumed = svb_crate_decode::<SvbCrateSimd>(
                            black_box(&simd_encoded),
                            n,
                            &mut svb_simd_out,
                        );
                        black_box(consumed);
                    }
                },
            );

            measure_bench(
                records,
                "int_codecs",
                "int_codecs/stream_vbyte_simd/decode_deltas",
                "ns/int",
                svb_deltas_params,
                || {
                    for _ in 0..16usize {
                        let mut out = vec![0u32; n];
                        let consumed =
                            svb_crate_decode::<SvbCrateSimd>(black_box(&simd_encoded), n, &mut out);
                        let mut acc = 0u32;
                        for v in &mut out {
                            acc = acc.wrapping_add(*v);
                            *v = acc;
                        }
                        black_box(consumed + out.len());
                    }
                },
            );
        }
    }

    // 4. Competitor: bitpacking crate (v0.9.3 BitPacker4x, 128-element SIMD blocks, 1 instance)
    {
        let bitpacker = BitPacker4x::new();
        const BP_BLOCK: usize = BitPacker4x::BLOCK_LEN; // 128
        let num_blocks = n.div_ceil(BP_BLOCK);
        let padded_len = num_blocks * BP_BLOCK;
        let mut padded_sorted = sorted_u32.clone();
        let last_val = *sorted_u32.last().unwrap_or(&0);
        padded_sorted.resize(padded_len, last_val);

        let mut bp_num_bits = Vec::with_capacity(num_blocks);
        let mut bp_compressed = vec![0u8; padded_len * 4];
        let mut bp_len = 0usize;
        let mut init_val = 0u32;
        for b_idx in 0..num_blocks {
            let chunk = &padded_sorted[b_idx * BP_BLOCK..(b_idx + 1) * BP_BLOCK];
            let nb = bitpacker.num_bits_sorted(init_val, chunk);
            bp_num_bits.push(nb);
            bp_len += bitpacker.compress_sorted(init_val, chunk, &mut bp_compressed[bp_len..], nb);
            init_val = chunk[BP_BLOCK - 1];
        }
        bp_compressed.truncate(bp_len);
        record_space_metric(
            space,
            "int_codecs/bitpacking_4x_bits_per_int",
            ((bp_compressed.len() + bp_num_bits.len()) as f64 * 8.0) / (n as f64),
        );

        let mut bp_out = vec![0u32; padded_len];

        measure_bench(
            records,
            "int_codecs",
            "int_codecs/bitpacking_4x/encode_deltas",
            "ns/int",
            params,
            || {
                let src = black_box(&padded_sorted);
                let mut num_bits_vec = Vec::with_capacity(num_blocks);
                let mut out_buf = vec![0u8; padded_len * 4];
                let mut pos = 0usize;
                let mut initial = 0u32;
                for b_idx in 0..num_blocks {
                    let chunk = &src[b_idx * BP_BLOCK..(b_idx + 1) * BP_BLOCK];
                    let nb = bitpacker.num_bits_sorted(initial, chunk);
                    num_bits_vec.push(nb);
                    pos += bitpacker.compress_sorted(initial, chunk, &mut out_buf[pos..], nb);
                    initial = chunk[BP_BLOCK - 1];
                }
                out_buf.truncate(pos);
                black_box(out_buf.len() + num_bits_vec.len());
            },
        );

        measure_bench(
            records,
            "int_codecs",
            "int_codecs/bitpacking_4x/decode_into",
            "ns/int",
            svb_into_params,
            || {
                for _ in 0..32usize {
                    let comp = black_box(&bp_compressed);
                    let nbs = black_box(&bp_num_bits);
                    let mut pos = 0usize;
                    for (b_idx, &nb) in nbs.iter().enumerate() {
                        let dst = &mut bp_out[b_idx * BP_BLOCK..(b_idx + 1) * BP_BLOCK];
                        pos += bitpacker.decompress(&comp[pos..], dst, nb);
                    }
                    black_box(pos);
                }
            },
        );

        measure_bench(
            records,
            "int_codecs",
            "int_codecs/bitpacking_4x/decode_deltas",
            "ns/int",
            svb_deltas_params,
            || {
                for _ in 0..16usize {
                    let comp = black_box(&bp_compressed);
                    let nbs = black_box(&bp_num_bits);
                    let mut out = vec![0u32; padded_len];
                    let mut pos = 0usize;
                    let mut initial = 0u32;
                    for (b_idx, &nb) in nbs.iter().enumerate() {
                        let dst = &mut out[b_idx * BP_BLOCK..(b_idx + 1) * BP_BLOCK];
                        pos += bitpacker.decompress_sorted(initial, &comp[pos..], dst, nb);
                        initial = dst[BP_BLOCK - 1];
                    }
                    out.truncate(n);
                    black_box(pos + out.len());
                }
            },
        );
    }
}
