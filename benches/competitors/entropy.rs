//! Family 6: Entropy Coding (65,536 bytes = 64 KiB synthetic `enwik8`-like block).
//!
//! Zipora: `HuffmanEncoder` O0, `ContextualHuffmanEncoder` O1/O2, `FseEncoder`,
//!         `Rans64Encoder<ParallelX1>`, `Rans64Encoder<ParallelX4>`
//! Competitors: `huffman-compress`, `constriction` (ANS), `zstd` (levels 1 & 3)
//!
//! Fairness invariant (`S42-R1`): every implementation is benchmarked on **1 instance**
//! on the calling thread over the exact same 64 KiB `block` with zero multi-instance
//! rings or heap spacers.

use std::collections::BTreeMap;
use std::hint::black_box;

use bit_vec::BitVec as HuffmanBitVec;
use constriction::stream::{
    Decode as ConstrictionDecode,
    model::SmallContiguousCategoricalEntropyModel,
    stack::DefaultAnsCoder,
};
use huffman_compress::CodeBuilder as HuffmanCompressBuilder;
use zipora::entropy::fse::{FseConfig, FseEncoder, fse_decompress};
use zipora::entropy::huffman::{
    ContextualHuffmanDecoder, ContextualHuffmanEncoder, HuffmanDecoder, HuffmanEncoder,
    HuffmanOrder,
};
use zipora::entropy::rans::{ParallelX1, ParallelX4, Rans64Decoder, Rans64Encoder};

use crate::common::*;

pub fn bench_entropy(
    records: &mut Vec<BenchRecord>,
    space: &mut BTreeMap<String, f64>,
    sizes: SuiteSizes,
) {
    let block_size = sizes.entropy_block_bytes;
    let block = gen_enwik_like_block(block_size, SEED_BASE ^ 0x07);
    let mut params = BenchParams::fast(block_size as u64, block_size as u64).with_threshold(9.0);
    params.warmup_calls = 8;
    params.trials = 11;
    params.bursts_per_trial = 8;

    // 1. Zipora Huffman O0 (64 KiB, 1 instance)
    let huff_o0_enc = HuffmanEncoder::new(&block).unwrap();
    let huff_o0_compressed = huff_o0_enc.encode(&block).unwrap();
    let huff_o0_dec = HuffmanDecoder::new(huff_o0_enc.tree().clone());
    record_space_metric(
        space,
        "entropy/huffman_o0_bits_per_byte",
        (huff_o0_compressed.len() as f64 * 8.0) / (block_size as f64),
    );

    measure_bench(
        records,
        "entropy",
        "entropy/huffman_o0/encode_64kib",
        "ns/byte",
        params,
        || {
            let encoded = huff_o0_enc.encode(black_box(&block)).unwrap();
            black_box(encoded.len());
        },
    );

    measure_bench(
        records,
        "entropy",
        "entropy/huffman_o0/decode_64kib",
        "ns/byte",
        params,
        || {
            let decoded = huff_o0_dec
                .decode(black_box(&huff_o0_compressed), block_size)
                .unwrap();
            black_box(decoded.len());
        },
    );

    // 2. Zipora Huffman O1 (64 KiB, 1 instance)
    let huff_o1_enc = ContextualHuffmanEncoder::new(&block, HuffmanOrder::Order1).unwrap();
    let huff_o1_compressed = huff_o1_enc.encode(&block).unwrap();
    record_space_metric(
        space,
        "entropy/huffman_o1_bits_per_byte",
        (huff_o1_compressed.len() as f64 * 8.0) / (block_size as f64),
    );

    measure_bench(
        records,
        "entropy",
        "entropy/huffman_o1/encode_x1_64kib",
        "ns/byte",
        params,
        || {
            let encoded = huff_o1_enc.encode_x1(black_box(&block)).unwrap();
            black_box(encoded.len());
        },
    );

    measure_bench(
        records,
        "entropy",
        "entropy/huffman_o1/encode_x4_64kib",
        "ns/byte",
        params,
        || {
            let encoded = huff_o1_enc.encode_x4(black_box(&block)).unwrap();
            black_box(encoded.len());
        },
    );

    let huff_o1_dec = ContextualHuffmanDecoder::new(huff_o1_enc);
    measure_bench(
        records,
        "entropy",
        "entropy/huffman_o1/decode_64kib",
        "ns/byte",
        params,
        || {
            let decoded = huff_o1_dec
                .decode(black_box(&huff_o1_compressed), block_size)
                .unwrap();
            black_box(decoded.len());
        },
    );
    drop(huff_o1_dec);
    drop(huff_o1_compressed);

    // 3. Zipora Huffman O2 (64 KiB, 1 instance)
    let huff_o2_enc = ContextualHuffmanEncoder::new(&block, HuffmanOrder::Order2).unwrap();
    let huff_o2_compressed = huff_o2_enc.encode(&block).unwrap();
    record_space_metric(
        space,
        "entropy/huffman_o2_bits_per_byte",
        (huff_o2_compressed.len() as f64 * 8.0) / (block_size as f64),
    );

    measure_bench(
        records,
        "entropy",
        "entropy/huffman_o2/encode_64kib",
        "ns/byte",
        params,
        || {
            let encoded = huff_o2_enc.encode(black_box(&block)).unwrap();
            black_box(encoded.len());
        },
    );

    let huff_o2_dec = ContextualHuffmanDecoder::new(huff_o2_enc);
    measure_bench(
        records,
        "entropy",
        "entropy/huffman_o2/decode_64kib",
        "ns/byte",
        params,
        || {
            let decoded = huff_o2_dec
                .decode(black_box(&huff_o2_compressed), block_size)
                .unwrap();
            black_box(decoded.len());
        },
    );
    drop(huff_o2_dec);
    drop(huff_o2_compressed);

    // 4. Zipora FSE (64 KiB, 1 instance)
    let mut fse_enc = FseEncoder::new(FseConfig::default()).unwrap();
    let fse_compressed = fse_enc.compress(&block).unwrap();
    record_space_metric(
        space,
        "entropy/fse_bits_per_byte",
        (fse_compressed.len() as f64 * 8.0) / (block_size as f64),
    );

    measure_bench(
        records,
        "entropy",
        "entropy/fse/encode_64kib",
        "ns/byte",
        params,
        || {
            let encoded = fse_enc.compress(black_box(&block)).unwrap();
            black_box(encoded.len());
        },
    );

    measure_bench(
        records,
        "entropy",
        "entropy/fse/decode_64kib",
        "ns/byte",
        params,
        || {
            let decoded = fse_decompress(black_box(&fse_compressed)).unwrap();
            black_box(decoded.len());
        },
    );

    // 5. Zipora rANS64 X1 and X4 (64 KiB, 1 instance each)
    let mut freqs = [1u32; 256];
    for &b in &block {
        freqs[b as usize] += 1;
    }
    let rans_x1_enc = Rans64Encoder::<ParallelX1>::new(&freqs).unwrap();
    let rans_x1_compressed = rans_x1_enc.encode(&block).unwrap();
    let rans_x1_dec = Rans64Decoder::<ParallelX1>::new(&rans_x1_enc);
    record_space_metric(
        space,
        "entropy/rans64_x1_bits_per_byte",
        (rans_x1_compressed.len() as f64 * 8.0) / (block_size as f64),
    );

    let rans_x4_enc = Rans64Encoder::<ParallelX4>::new(&freqs).unwrap();
    let rans_x4_compressed = rans_x4_enc.encode(&block).unwrap();
    let rans_x4_dec = Rans64Decoder::<ParallelX4>::new(&rans_x4_enc);
    record_space_metric(
        space,
        "entropy/rans64_x4_bits_per_byte",
        (rans_x4_compressed.len() as f64 * 8.0) / (block_size as f64),
    );

    measure_bench(
        records,
        "entropy",
        "entropy/rans64_x1/encode_64kib",
        "ns/byte",
        params,
        || {
            let encoded = rans_x1_enc.encode(black_box(&block)).unwrap();
            black_box(encoded.len());
        },
    );

    measure_bench(
        records,
        "entropy",
        "entropy/rans64_x1/decode_64kib",
        "ns/byte",
        params,
        || {
            let decoded = rans_x1_dec
                .decode(black_box(&rans_x1_compressed), block_size)
                .unwrap();
            black_box(decoded.len());
        },
    );

    measure_bench(
        records,
        "entropy",
        "entropy/rans64_x4/encode_64kib",
        "ns/byte",
        params,
        || {
            let encoded = rans_x4_enc.encode(black_box(&block)).unwrap();
            black_box(encoded.len());
        },
    );

    measure_bench(
        records,
        "entropy",
        "entropy/rans64_x4/decode_64kib",
        "ns/byte",
        params,
        || {
            let decoded = rans_x4_dec
                .decode(black_box(&rans_x4_compressed), block_size)
                .unwrap();
            black_box(decoded.len());
        },
    );

    // 6. Competitor: huffman-compress (static Order-0 Huffman on 64 KiB block, 1 instance)
    {
        let mut huff_weights: BTreeMap<u8, u32> = BTreeMap::new();
        for &b in &block {
            *huff_weights.entry(b).or_insert(0) += 1;
        }
        let (hc_book, hc_tree) = HuffmanCompressBuilder::from_iter(huff_weights).finish();
        let mut hc_compressed = HuffmanBitVec::with_capacity(block_size * 8);
        for &b in &block {
            hc_book.encode(&mut hc_compressed, &b).unwrap();
        }
        let hc_bytes = hc_compressed.len().div_ceil(8);
        record_space_metric(
            space,
            "entropy/huffman_compress_bits_per_byte",
            (hc_bytes as f64 * 8.0) / (block_size as f64),
        );

        let mut hc_params = params;
        hc_params.warmup_calls = 2;
        hc_params.trials = 7;
        hc_params.bursts_per_trial = 2;

        measure_bench(
            records,
            "entropy",
            "entropy/huffman_compress/encode_64kib",
            "ns/byte",
            hc_params,
            || {
                let mut buf = HuffmanBitVec::with_capacity(block_size * 8);
                for b in black_box(&block) {
                    hc_book.encode(&mut buf, b).unwrap();
                }
                black_box(buf.len());
            },
        );

        measure_bench(
            records,
            "entropy",
            "entropy/huffman_compress/decode_64kib",
            "ns/byte",
            hc_params,
            || {
                let mut out = Vec::with_capacity(block_size);
                out.extend(hc_tree.decoder(black_box(&hc_compressed), block_size));
                black_box(out.len());
            },
        );
    }

    // 7. Competitor: constriction ANS (DefaultAnsCoder + 12-bit categorical / lookup model, 1 instance)
    {
        let probs_f64: Vec<f64> = freqs.iter().map(|&c| c as f64).collect();
        let constriction_enc_model =
            SmallContiguousCategoricalEntropyModel::from_floating_point_probabilities_perfect(
                &probs_f64,
            )
            .unwrap();
        let constriction_dec_model = constriction_enc_model.to_lookup_decoder_model();

        let mut init_coder = DefaultAnsCoder::with_write_backend(Vec::with_capacity(block_size / 4));
        init_coder
            .encode_iid_symbols_reverse(
                block.iter().map(|&b| b as usize),
                constriction_enc_model.as_view(),
            )
            .unwrap();
        let constriction_compressed: Vec<u32> = init_coder.into_compressed().unwrap();
        record_space_metric(
            space,
            "entropy/constriction_ans_bits_per_byte",
            ((constriction_compressed.len() * 4) as f64 * 8.0) / (block_size as f64),
        );

        measure_bench(
            records,
            "entropy",
            "entropy/constriction_ans/encode_64kib",
            "ns/byte",
            params,
            || {
                let mut coder =
                    DefaultAnsCoder::with_write_backend(Vec::with_capacity(block_size / 4));
                coder
                    .encode_iid_symbols_reverse(
                        black_box(&block).iter().map(|&b| b as usize),
                        constriction_enc_model.as_view(),
                    )
                    .unwrap();
                let words = coder.into_compressed().unwrap();
                black_box(words.len());
            },
        );

        measure_bench(
            records,
            "entropy",
            "entropy/constriction_ans/decode_64kib",
            "ns/byte",
            params,
            || {
                let mut coder =
                    DefaultAnsCoder::from_compressed_slice(black_box(&constriction_compressed))
                        .unwrap();
                let mut out = Vec::with_capacity(block_size);
                for sym in coder.decode_iid_symbols(block_size, &constriction_dec_model) {
                    out.push(sym.unwrap() as u8);
                }
                black_box(out.len());
            },
        );
    }

    // 8. Competitor: zstd levels 1 and 3 (64 KiB whole-block compress / decompress with reused CCtx/DCtx)
    {
        let mut zstd_l1_comp = zstd::bulk::Compressor::new(1).unwrap();
        let mut zstd_l3_comp = zstd::bulk::Compressor::new(3).unwrap();
        let mut zstd_decomp = zstd::bulk::Decompressor::new().unwrap();
        let zstd_l1_compressed = zstd_l1_comp.compress(&block).unwrap();
        let zstd_l3_compressed = zstd_l3_comp.compress(&block).unwrap();
        record_space_metric(
            space,
            "entropy/zstd_l1_bits_per_byte",
            (zstd_l1_compressed.len() as f64 * 8.0) / (block_size as f64),
        );
        record_space_metric(
            space,
            "entropy/zstd_l3_bits_per_byte",
            (zstd_l3_compressed.len() as f64 * 8.0) / (block_size as f64),
        );

        measure_bench(
            records,
            "entropy",
            "entropy/zstd_l1/encode_64kib",
            "ns/byte",
            params,
            || {
                let encoded = zstd_l1_comp.compress(black_box(&block)).unwrap();
                black_box(encoded.len());
            },
        );

        measure_bench(
            records,
            "entropy",
            "entropy/zstd_l1/decode_64kib",
            "ns/byte",
            params,
            || {
                let decoded = zstd_decomp
                    .decompress(black_box(&zstd_l1_compressed), block_size)
                    .unwrap();
                black_box(decoded.len());
            },
        );

        measure_bench(
            records,
            "entropy",
            "entropy/zstd_l3/encode_64kib",
            "ns/byte",
            params,
            || {
                let encoded = zstd_l3_comp.compress(black_box(&block)).unwrap();
                black_box(encoded.len());
            },
        );

        measure_bench(
            records,
            "entropy",
            "entropy/zstd_l3/decode_64kib",
            "ns/byte",
            params,
            || {
                let decoded = zstd_decomp
                    .decompress(black_box(&zstd_l3_compressed), block_size)
                    .unwrap();
                black_box(decoded.len());
            },
        );
    }
}
