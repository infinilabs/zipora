//! Family 7: Integer Codecs (100,000 contiguous sorted `u32` integers by default).
//!
//! Zipora: `StreamVByte`, `GroupVarint`
//! Competitors: `stream-vbyte` (`Scalar`), `bitpacking` (`BitPacker4x`)
//!
//! Fairness invariant (`S42-R1`): every implementation is benchmarked on **1 instance**
//! and **1 shared `sorted_u32` input slice**, with zero 8-clone input rings or heap spacers.

use std::collections::BTreeMap;
use std::hint::black_box;

use bitpacking::{BitPacker, BitPacker4x};
use stream_vbyte::{
    decode::decode as svb_crate_decode, encode::encode as svb_crate_encode,
    scalar::Scalar as SvbCrateScalar,
};
use zipora::compression::stream_vbyte::{GroupVarint, StreamVByte};

use crate::common::*;

pub fn bench_int_codecs(
    records: &mut Vec<BenchRecord>,
    space: &mut BTreeMap<String, f64>,
    sizes: SuiteSizes,
) {
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
    let svb_stream = StreamVByte::encode_deltas(&sorted_u32);
    let mut svb_out = vec![0u32; n];
    record_space_metric(
        space,
        "int_codecs/stream_vbyte_bits_per_int",
        ((svb_stream.controls.len() + svb_stream.data.len()) as f64 * 8.0) / (n as f64),
    );

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

    // 2. Zipora GroupVarint (1 instance, 1 shared `sorted_u32` input slice)
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
