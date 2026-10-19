//! Family 6: Entropy Coding (65,536 bytes = 64 KiB synthetic `enwik8`-like block).
//!
//! Zipora: `HuffmanEncoder` O0, `ContextualHuffmanEncoder` O1/O2, `FseEncoder`,
//!         `Rans64Encoder<ParallelX1>`, `Rans64Encoder<ParallelX4>`
//! Competitors: `huffman-compress`, `constriction` (ANS), `libzstd` raw `huff0` and `FSE`
//!              entropy stages (`S42-R5`), and `zstd` whole-block (LZ77+entropy levels 1 & 3)
//!
//! Fairness invariant (`S42-R1`): every implementation is benchmarked on **1 instance**
//! on the calling thread over the exact same 64 KiB `block` with zero multi-instance
//! rings or heap spacers.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::hint::black_box;

use bit_vec::BitVec as HuffmanBitVec;
use constriction::stream::{
    Decode as ConstrictionDecode, model::SmallContiguousCategoricalEntropyModel,
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

// Raw `libzstd` (`zstd-sys`) entropy-stage symbols (`huff0` and `FSE` tANS) for `S42-R5`.
unsafe extern "C" {
    fn HUF_isError(code: usize) -> u32;
    fn HUF_compress4X_repeat(
        dst: *mut c_void,
        dst_size: usize,
        src: *const c_void,
        src_size: usize,
        max_symbol_value: u32,
        table_log: u32,
        work_space: *mut c_void,
        wksp_size: usize,
        huf_table: *mut usize,
        repeat: *mut u32,
        flags: i32,
    ) -> usize;
    fn HUF_decompress4X_hufOnly_wksp(
        dctx: *mut u32,
        dst: *mut c_void,
        dst_size: usize,
        c_src: *const c_void,
        c_src_size: usize,
        work_space: *mut c_void,
        wksp_size: usize,
        flags: i32,
    ) -> usize;
    fn HIST_count_wksp(
        count: *mut u32,
        max_symbol_value_ptr: *mut u32,
        src: *const c_void,
        src_size: usize,
        work_space: *mut u32,
        wksp_size: usize,
    ) -> usize;
    fn FSE_isError(code: usize) -> u32;
    fn FSE_optimalTableLog(max_table_log: u32, src_size: usize, max_symbol_value: u32) -> u32;
    fn FSE_normalizeCount(
        normalized_counter: *mut i16,
        table_log: u32,
        count: *const u32,
        src_size: usize,
        max_symbol_value: u32,
        use_low_prob_count: u32,
    ) -> usize;
    fn FSE_writeNCount(
        buffer: *mut c_void,
        buffer_size: usize,
        normalized_counter: *const i16,
        max_symbol_value: u32,
        table_log: u32,
    ) -> usize;
    fn FSE_buildCTable_wksp(
        ct: *mut u32,
        normalized_counter: *const i16,
        max_symbol_value: u32,
        table_log: u32,
        work_space: *mut c_void,
        wksp_size: usize,
    ) -> usize;
    fn FSE_compress_usingCTable(
        dst: *mut c_void,
        dst_capacity: usize,
        src: *const c_void,
        src_size: usize,
        ct: *const u32,
    ) -> usize;
    fn FSE_decompress_wksp_bmi2(
        dst: *mut c_void,
        dst_capacity: usize,
        c_src: *const c_void,
        c_src_size: usize,
        max_log: u32,
        work_space: *mut c_void,
        wksp_size: usize,
        bmi2: i32,
    ) -> usize;
}

struct ZstdHuf0Workspace {
    comp_wksp: Vec<u64>,
    ctable: [usize; 258],
    dtable: [u32; 4097],
    decomp_wksp: [u64; 512],
}

impl ZstdHuf0Workspace {
    fn new() -> Self {
        let mut dtable = [0u32; 4097];
        dtable[0] = 12 * 0x0100_0001;
        Self {
            comp_wksp: vec![0u64; 2048],
            ctable: [0usize; 258],
            dtable,
            decomp_wksp: [0u64; 512],
        }
    }

    #[inline]
    fn compress(&mut self, src: &[u8], dst: &mut [u8]) -> usize {
        let mut repeat: u32 = 0; // HUF_repeat_none
        // SAFETY: All buffers are sized above `HUF_WORKSPACE_SIZE` and `HUF_CTABLE_SIZE_ST(255)`,
        // 8-byte aligned, and `dst` is slice-bounded.
        let written = unsafe {
            HUF_compress4X_repeat(
                dst.as_mut_ptr().cast::<c_void>(),
                dst.len(),
                src.as_ptr().cast::<c_void>(),
                src.len(),
                255,
                11,
                self.comp_wksp.as_mut_ptr().cast::<c_void>(),
                self.comp_wksp.len() * std::mem::size_of::<u64>(),
                self.ctable.as_mut_ptr(),
                &mut repeat,
                1, // HUF_flags_bmi2
            )
        };
        assert!(unsafe { HUF_isError(written) } == 0 && written > 0);
        written
    }

    #[inline]
    fn decompress(&mut self, c_src: &[u8], dst: &mut [u8]) -> usize {
        self.dtable[0] = 12 * 0x0100_0001;
        // SAFETY: `dtable` has `HUF_DTABLE_SIZE(12) == 4097` words and `decomp_wksp` exceeds
        // `HUF_DECOMPRESS_WORKSPACE_SIZE` (4096 > 2560 bytes).
        let decoded = unsafe {
            HUF_decompress4X_hufOnly_wksp(
                self.dtable.as_mut_ptr(),
                dst.as_mut_ptr().cast::<c_void>(),
                dst.len(),
                c_src.as_ptr().cast::<c_void>(),
                c_src.len(),
                self.decomp_wksp.as_mut_ptr().cast::<c_void>(),
                self.decomp_wksp.len() * std::mem::size_of::<u64>(),
                1, // HUF_flags_bmi2
            )
        };
        assert!(unsafe { HUF_isError(decoded) } == 0 && decoded == dst.len());
        decoded
    }
}

struct ZstdFseWorkspace {
    hist_wksp: [u32; 1024],
    count: [u32; 256],
    norm: [i16; 256],
    ctable: [u32; 4096],
    ctable_wksp: [u64; 2048],
    decomp_wksp: [u64; 4096],
}

impl ZstdFseWorkspace {
    fn new() -> Self {
        Self {
            hist_wksp: [0u32; 1024],
            count: [0u32; 256],
            norm: [0i16; 256],
            ctable: [0u32; 4096],
            ctable_wksp: [0u64; 2048],
            decomp_wksp: [0u64; 4096],
        }
    }

    #[inline]
    fn compress(&mut self, src: &[u8], dst: &mut [u8]) -> usize {
        let mut max_symbol: u32 = 255;
        // SAFETY: Buffers are statically sized above `HIST_WKSP_SIZE_U32`, `FSE_CTABLE_SIZE_U32(12, 255)`,
        // and `FSE_BUILD_CTABLE_WORKSPACE_SIZE(255, 12)`, and `dst` is slice-bounded.
        unsafe {
            let max_count = HIST_count_wksp(
                self.count.as_mut_ptr(),
                &mut max_symbol,
                src.as_ptr().cast::<c_void>(),
                src.len(),
                self.hist_wksp.as_mut_ptr(),
                self.hist_wksp.len() * std::mem::size_of::<u32>(),
            );
            assert!(FSE_isError(max_count) == 0 && max_count < src.len() && max_count > 1);
            let table_log = FSE_optimalTableLog(12, src.len(), max_symbol);
            let norm_log = FSE_normalizeCount(
                self.norm.as_mut_ptr(),
                table_log,
                self.count.as_ptr(),
                src.len(),
                max_symbol,
                1,
            );
            assert!(FSE_isError(norm_log) == 0);
            let h_size = FSE_writeNCount(
                dst.as_mut_ptr().cast::<c_void>(),
                dst.len(),
                self.norm.as_ptr(),
                max_symbol,
                norm_log as u32,
            );
            assert!(FSE_isError(h_size) == 0 && h_size < dst.len());
            let build_rc = FSE_buildCTable_wksp(
                self.ctable.as_mut_ptr(),
                self.norm.as_ptr(),
                max_symbol,
                norm_log as u32,
                self.ctable_wksp.as_mut_ptr().cast::<c_void>(),
                self.ctable_wksp.len() * std::mem::size_of::<u64>(),
            );
            assert!(FSE_isError(build_rc) == 0);
            let c_size = FSE_compress_usingCTable(
                dst[h_size..].as_mut_ptr().cast::<c_void>(),
                dst.len() - h_size,
                src.as_ptr().cast::<c_void>(),
                src.len(),
                self.ctable.as_ptr(),
            );
            assert!(FSE_isError(c_size) == 0 && c_size > 0);
            h_size + c_size
        }
    }

    #[inline]
    fn decompress(&mut self, c_src: &[u8], dst: &mut [u8]) -> usize {
        // SAFETY: `decomp_wksp` (`16 KiB`, 8-byte aligned) exceeds `FSE_DECOMPRESS_WKSP_SIZE(12, 255)`
        // and `c_src` / `dst` are valid byte slices.
        let decoded = unsafe {
            FSE_decompress_wksp_bmi2(
                dst.as_mut_ptr().cast::<c_void>(),
                dst.len(),
                c_src.as_ptr().cast::<c_void>(),
                c_src.len(),
                12,
                self.decomp_wksp.as_mut_ptr().cast::<c_void>(),
                self.decomp_wksp.len() * std::mem::size_of::<u64>(),
                1,
            )
        };
        assert!(unsafe { FSE_isError(decoded) } == 0 && decoded == dst.len());
        decoded
    }
}

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

        let mut init_coder =
            DefaultAnsCoder::with_write_backend(Vec::with_capacity(block_size / 4));
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

    // 8. Competitor: libzstd raw huff0 (4-stream Order-0 Huffman stage, 1 instance)
    {
        let mut huf0_wksp = ZstdHuf0Workspace::new();
        let max_dst = block_size + (block_size >> 8) + 256;
        let mut huf0_compressed = vec![0u8; max_dst];
        let huf0_len = huf0_wksp.compress(&block, &mut huf0_compressed);
        huf0_compressed.truncate(huf0_len);
        let mut huf0_check = vec![0u8; block_size];
        huf0_wksp.decompress(&huf0_compressed, &mut huf0_check);
        assert_eq!(huf0_check, block);

        record_space_metric(
            space,
            "entropy/zstd_huf0_bits_per_byte",
            (huf0_compressed.len() as f64 * 8.0) / (block_size as f64),
        );

        let mut huf0_comp_buf = vec![0u8; max_dst];
        let mut huf0_decomp_buf = vec![0u8; block_size];

        measure_bench(
            records,
            "entropy",
            "entropy/zstd_huf0/encode_64kib",
            "ns/byte",
            params,
            || {
                let written = huf0_wksp.compress(black_box(&block), &mut huf0_comp_buf);
                black_box(written);
            },
        );

        measure_bench(
            records,
            "entropy",
            "entropy/zstd_huf0/decode_64kib",
            "ns/byte",
            params,
            || {
                let decoded =
                    huf0_wksp.decompress(black_box(&huf0_compressed), &mut huf0_decomp_buf);
                black_box(decoded);
            },
        );
    }

    // 9. Competitor: libzstd raw FSE (tANS Order-0 entropy stage, 1 instance)
    {
        let mut fse_wksp = ZstdFseWorkspace::new();
        let max_dst = block_size + (block_size >> 7) + 1024;
        let mut zfse_compressed = vec![0u8; max_dst];
        let zfse_len = fse_wksp.compress(&block, &mut zfse_compressed);
        zfse_compressed.truncate(zfse_len);
        let mut zfse_check = vec![0u8; block_size];
        fse_wksp.decompress(&zfse_compressed, &mut zfse_check);
        assert_eq!(zfse_check, block);

        record_space_metric(
            space,
            "entropy/zstd_fse_bits_per_byte",
            (zfse_compressed.len() as f64 * 8.0) / (block_size as f64),
        );

        let mut zfse_comp_buf = vec![0u8; max_dst];
        let mut zfse_decomp_buf = vec![0u8; block_size];

        measure_bench(
            records,
            "entropy",
            "entropy/zstd_fse/encode_64kib",
            "ns/byte",
            params,
            || {
                let written = fse_wksp.compress(black_box(&block), &mut zfse_comp_buf);
                black_box(written);
            },
        );

        measure_bench(
            records,
            "entropy",
            "entropy/zstd_fse/decode_64kib",
            "ns/byte",
            params,
            || {
                let decoded =
                    fse_wksp.decompress(black_box(&zfse_compressed), &mut zfse_decomp_buf);
                black_box(decoded);
            },
        );
    }

    // 10. Competitor: zstd levels 1 and 3 (64 KiB whole-block LZ77+entropy with reused CCtx/DCtx)
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
