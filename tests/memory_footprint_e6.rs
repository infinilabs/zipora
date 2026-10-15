//! Phase E (`E6`): Deterministic assertions for all memory-footprint claims.
//!
//! Asserts with explicit tolerances:
//! 1. Elias-Fano variants (`EliasFano`, `PartitionedEliasFano`, `OptimalPartitionedEliasFano`,
//!    `ClusteredEliasFano`, `HybridPostingList`) `bits_per_element()` across uniform-sparse,
//!    bursty-clustered, and dense/run-heavy distributions;
//! 2. Trie representations (`DoubleArrayTrie`, `DoubleArrayTrieMap<u32>`, `CsppTrie`)
//!    bytes/state and active bytes/key;
//! 3. `FieldnormEncoder` `1.0 B/doc` storage vs `UintVecMin0` `~1.13 B/doc`, 1,024-byte
//!    norm table, and bounded quantization error;
//! 4. Rank/Select (`Rank9`, `RankSelectSE256`, `RankSelectSE512_64`, `RankSelectInterleaved256`,
//!    `RankSelectSimple`, `RankSelectFewOne`) `space_overhead_percent()` and `mem_size()`.

use zipora::containers::UintVecMin0;
use zipora::fsa::cspp_trie::CsppTrie;
use zipora::fsa::double_array::{DoubleArrayTrie, DoubleArrayTrieMap};
use zipora::scoring::FieldnormEncoder;
use zipora::succinct::BitVector;
use zipora::succinct::elias_fano::{
    ClusteredEliasFano, EliasFano, HybridPostingList, OptimalPartitionedEliasFano,
    PartitionedEliasFano, PostingEncoding,
};
use zipora::succinct::rank_select::{
    Rank9, RankSelectFewOne, RankSelectInterleaved256, RankSelectOps, RankSelectSE256,
    RankSelectSE512_64, RankSelectSimple,
};

#[test]
fn test_e6_elias_fano_variants_bits_per_element_across_distributions() {
    let n = 10_000usize;

    // 1. Uniform-sparse distribution: n = 10,000, U = 10,000,000 (average gap = 1,000).
    // Information-theoretic EF bound: 2 + ceil(log2(U / n)) = 2 + 10 = 12 bits/elem.
    let uniform: Vec<u32> = (0..n as u32).map(|i| i * 1_000 + (i % 7)).collect();
    let ef_uni = EliasFano::from_sorted(&uniform);
    let pef_uni = PartitionedEliasFano::from_sorted(&uniform);
    let opef_uni = OptimalPartitionedEliasFano::from_sorted(&uniform);
    let cef_uni = ClusteredEliasFano::from_sorted(&uniform);

    assert_eq!(
        ef_uni.bits_per_element(),
        (ef_uni.size_bytes() as f64 * 8.0) / n as f64
    );
    assert!(
        (11.5..=14.5).contains(&ef_uni.bits_per_element()),
        "EliasFano uniform bits/elem out of expected [11.5, 14.5] range: {:.2}",
        ef_uni.bits_per_element()
    );
    assert!(
        (11.5..=15.5).contains(&pef_uni.bits_per_element()),
        "PartitionedEliasFano uniform bits/elem out of expected [11.5, 15.5] range: {:.2}",
        pef_uni.bits_per_element()
    );
    assert!(
        (11.5..=15.5).contains(&opef_uni.bits_per_element()),
        "OptimalPartitionedEliasFano uniform bits/elem out of expected [11.5, 15.5] range: {:.2}",
        opef_uni.bits_per_element()
    );
    assert!(
        cef_uni.bits_per_element() <= 16.0,
        "ClusteredEliasFano uniform bits/elem exceeded 16.0: {:.2}",
        cef_uni.bits_per_element()
    );

    // 2. Bursty / clustered distribution: 50 tight runs of 200 contiguous IDs separated by 100,000 gaps.
    // Plain EF pays global universe width (~11-12 bits/elem), while PEF/OPEF adapt per chunk.
    let mut clustered = Vec::with_capacity(n);
    let mut cur = 100u32;
    for cluster in 0..50u32 {
        for offset in 0..200u32 {
            clustered.push(cur + offset);
        }
        cur += 200 + 100_000 + (cluster * 13);
    }
    let ef_clu = EliasFano::from_sorted(&clustered);
    let pef_clu = PartitionedEliasFano::from_sorted(&clustered);
    let opef_clu = OptimalPartitionedEliasFano::from_sorted(&clustered);

    assert!(
        ef_clu.bits_per_element() >= 10.0,
        "Plain EliasFano on wide-universe clustered data must be >= 10.0 bits/elem, got {:.2}",
        ef_clu.bits_per_element()
    );
    assert!(
        pef_clu.bits_per_element() < ef_clu.bits_per_element(),
        "PEF ({:.2}) must beat plain EF ({:.2}) on clustered data",
        pef_clu.bits_per_element(),
        ef_clu.bits_per_element()
    );
    assert!(
        opef_clu.bits_per_element() < pef_clu.bits_per_element(),
        "OptimalPEF ({:.2}) must beat uniform PEF ({:.2}) on variable-cluster data",
        opef_clu.bits_per_element(),
        pef_clu.bits_per_element()
    );
    assert!(
        opef_clu.bits_per_element() <= 4.5,
        "OptimalPEF on tight clusters must be <= 4.5 bits/elem, got {:.2}",
        opef_clu.bits_per_element()
    );

    // 3a. Fully-dense distribution (0..100_000, matching benches/elias_fano_bench.rs):
    // full 128-element chunks are Run containers (40-byte CefChunkMeta per 128 elements = 2.50 bits/elem + tail = ~2.51 bits/elem).
    let fully_dense: Vec<u32> = (0..100_000u32).collect();
    let cef_fully_dense = ClusteredEliasFano::from_sorted(&fully_dense);
    let opef_fully_dense = OptimalPartitionedEliasFano::from_sorted(&fully_dense);
    let hybrid_fully_dense = HybridPostingList::from_sorted(&fully_dense);
    assert!(
        (2.50..=2.55).contains(&cef_fully_dense.bits_per_element()),
        "ClusteredEliasFano on fully-dense 0..100_000 must be in [2.50, 2.55] bits/elem (claim ~2.51), got {:.2}",
        cef_fully_dense.bits_per_element()
    );
    assert!(
        cef_fully_dense.bits_per_element() < opef_fully_dense.bits_per_element(),
        "ClusteredEliasFano ({:.2}) must beat OptimalPEF ({:.2}) on fully-dense data",
        cef_fully_dense.bits_per_element(),
        opef_fully_dense.bits_per_element()
    );
    assert_eq!(hybrid_fully_dense.encoding(), PostingEncoding::Clustered);

    // 3b. Run-heavy distribution: runs of 512 contiguous elements separated by 64-ID gaps
    // (span ~ 1.125 * n <= 2 * n, 4 full 128-element Run chunks per run -> ~2.65 bits/elem).
    let mut dense = Vec::with_capacity(n);
    let mut pos = 0u32;
    while dense.len() < n {
        let run = 512.min(n - dense.len());
        for _ in 0..run {
            dense.push(pos);
            pos += 1;
        }
        pos += 64;
    }
    let cef_dense = ClusteredEliasFano::from_sorted(&dense);
    let opef_dense = OptimalPartitionedEliasFano::from_sorted(&dense);
    let hybrid_dense = HybridPostingList::from_sorted(&dense);

    assert!(
        cef_dense.bits_per_element() <= 3.0,
        "ClusteredEliasFano on run-heavy data must be <= 3.0 bits/elem, got {:.2}",
        cef_dense.bits_per_element()
    );
    assert!(
        cef_dense.bits_per_element() < opef_dense.bits_per_element(),
        "ClusteredEliasFano ({:.2}) must beat OptimalPEF ({:.2}) on run-heavy data",
        cef_dense.bits_per_element(),
        opef_dense.bits_per_element()
    );
    assert_eq!(hybrid_dense.encoding(), PostingEncoding::Clustered);
    assert!(hybrid_dense.bits_per_element() <= 3.05);

    // 4. HybridPostingList size-based and density-based selection ladder.
    let short_sparse: Vec<u32> = (0..32u32).map(|i| i * 100).collect();
    let med_sparse: Vec<u32> = (0..128u32).map(|i| i * 100).collect();
    let large_sparse: Vec<u32> = (0..1024u32).map(|i| i * 100).collect();
    assert_eq!(
        HybridPostingList::from_sorted(&short_sparse).encoding(),
        PostingEncoding::Dense
    );
    assert_eq!(
        HybridPostingList::from_sorted(&med_sparse).encoding(),
        PostingEncoding::EliasFano
    );
    assert_eq!(
        HybridPostingList::from_sorted(&large_sparse).encoding(),
        PostingEncoding::Partitioned
    );
    assert_eq!(
        HybridPostingList::from_sorted(&uniform).encoding(),
        PostingEncoding::Optimal
    );
}

#[test]
fn test_e6_tries_bytes_per_state_and_bytes_per_key() {
    // 1. DoubleArrayTrie: 8 B/state hot transition array (DaState) + 4 B/state NInfo = 12 B/state total.
    let mut dat = DoubleArrayTrie::new();
    let n_keys = 10_000usize;
    for i in 0..n_keys {
        let key = format!("pkg::module_{:02}::symbol_{:05}", i / 100, i);
        dat.insert(key.as_bytes()).unwrap();
    }
    dat.shrink_to_fit();
    assert_eq!(dat.len(), n_keys);
    assert!(dat.total_states() > 0);
    assert_eq!(
        dat.mem_size(),
        dat.total_states() * 12,
        "DoubleArrayTrie mem_size must equal 12 bytes per state (8B DaState + 4B NInfo)"
    );

    // 2. DoubleArrayTrieMap<u32>: sentinel V::EMPTY avoids Option<u32> 8-byte alignment overhead,
    // keeping each value slot at 4 bytes instead of 8 bytes (half the value-table footprint of u64/Option<u32>).
    let mut dat_map = DoubleArrayTrieMap::<u32>::new();
    let mut dat_map_u64 = DoubleArrayTrieMap::<u64>::new();
    for i in 0..2_000u32 {
        let key = format!("term_{:05}", i);
        dat_map.insert(key.as_bytes(), i).unwrap();
        dat_map_u64.insert(key.as_bytes(), i as u64).unwrap();
    }
    dat_map.shrink_to_fit();
    dat_map_u64.shrink_to_fit();
    assert_eq!(dat_map.total_states(), dat_map_u64.total_states());
    let trie_only_bytes = dat_map.total_states() * 12;
    let u32_value_bytes = dat_map.mem_size() - trie_only_bytes;
    let u64_value_bytes = dat_map_u64.mem_size() - trie_only_bytes;
    assert!(u32_value_bytes > 0);
    assert_eq!(
        u64_value_bytes,
        u32_value_bytes * 2,
        "4-byte sentinel value slots in DoubleArrayTrieMap<u32> must use half the value-table bytes of 8-byte slots"
    );
    let map_bytes_per_state = dat_map.mem_size() as f64 / dat_map.total_states() as f64;
    assert!(
        (12.0..=16.0).contains(&map_bytes_per_state),
        "DoubleArrayTrieMap<u32> bytes/state after shrink_to_fit must be in [12.0, 16.0] (12B trie + <=4B sentinel value), got {:.2}",
        map_bytes_per_state
    );

    // 3. CsppTrie: path-compressed radix trie (~10.7 B/key active footprint on shared-prefix keys).
    let mut cspp = CsppTrie::new(0);
    for i in 0..n_keys {
        let key = format!("pkg::module_{:02}::symbol_{:05}", i / 100, i);
        cspp.insert(key.as_bytes());
    }
    assert_eq!(cspp.num_words(), n_keys);
    let stat = cspp.mem_get_stat();
    assert!(stat.used_size >= stat.frag_size);
    let active_bytes = stat.used_size - stat.frag_size;
    let active_bytes_per_key = active_bytes as f64 / cspp.num_words() as f64;
    assert!(
        (7.0..=14.5).contains(&active_bytes_per_key),
        "CsppTrie active bytes/key on shared-prefix vocabulary must be in [7.0, 14.5] (claim ~10.7 B/key), got {:.2}",
        active_bytes_per_key
    );
}

#[test]
fn test_e6_fieldnorm_encoder_one_byte_per_doc_and_quantization_bounds() {
    let num_docs = 100_000usize;
    let doc_lengths: Vec<u32> = (0..num_docs as u32)
        .map(|i| 20 + ((i.wrapping_mul(2654435761)) % 480))
        .collect();

    // 1. Fieldnorm encodes arbitrary u32 document lengths (0..=u32::MAX) into a single u8
    // code in 0..=239 while bracketing the original length (`decode(code) <= dl < decode(code + 1)`)
    // within the 3-bit mantissa bound (`<= 12.5%` relative error).
    assert_eq!(FieldnormEncoder::encode(0), 0);
    assert_eq!(FieldnormEncoder::encode(u32::MAX), 239);
    assert_eq!(FieldnormEncoder::decode(239), 15u32 << 28);
    let fieldnorm_bytes: Vec<u8> = doc_lengths
        .iter()
        .map(|&dl| FieldnormEncoder::encode(dl))
        .collect();
    let encoded_bytes = std::mem::size_of_val(fieldnorm_bytes.as_slice());
    let fieldnorm_b_per_doc = encoded_bytes as f64 / num_docs as f64;
    for (&dl, &code) in doc_lengths.iter().zip(fieldnorm_bytes.iter()) {
        assert!(
            code <= 239,
            "SmallFloat code {code} exceeds max table index 239"
        );
        let decoded = FieldnormEncoder::decode(code);
        assert!(
            decoded <= dl,
            "decoded {decoded} must not exceed original {dl}"
        );
        if code < 239 {
            let next_decoded = FieldnormEncoder::decode(code + 1);
            assert!(
                dl < next_decoded,
                "dl {dl} must be strictly less than next bucket {next_decoded} (code={code})"
            );
        }
        assert!(FieldnormEncoder::quantization_error(dl) <= 0.125);
    }

    // 2. Compare against UintVecMin0 on the decoded u32 lengths (9-bit width -> 1.125 B/doc).
    let max_decoded = fieldnorm_bytes
        .iter()
        .map(|&b| FieldnormEncoder::decode(b) as usize)
        .max()
        .unwrap_or(0);
    let mut uint_vec = UintVecMin0::new(num_docs, max_decoded);
    for (i, &b) in fieldnorm_bytes.iter().enumerate() {
        uint_vec.set(i, FieldnormEncoder::decode(b) as usize);
    }
    let uint_vec_b_per_doc = uint_vec.mem_size() as f64 / num_docs as f64;
    assert!(
        (1.12..=1.14).contains(&uint_vec_b_per_doc),
        "UintVecMin0 expected ~1.125 B/doc for 9-bit doc lengths, got {:.4}",
        uint_vec_b_per_doc
    );
    assert!(fieldnorm_b_per_doc < uint_vec_b_per_doc);

    // 3. Norm table is exactly 256 * 4 = 1,024 bytes.
    let norm_table = FieldnormEncoder::build_norm_table(250.0, 1.2, 0.75);
    assert_eq!(std::mem::size_of_val(&norm_table), 1024);

    // 4. Quantization error is 0.0 for 0..=15 and <= 12.5% (0.125) for all >= 16.
    for len in 0..=15u32 {
        assert_eq!(FieldnormEncoder::quantization_error(len), 0.0);
    }
    for exp in 4..31u32 {
        let base = 1u32 << exp;
        for step in 0..16u32 {
            let len = base.saturating_add(step.saturating_mul(base / 16));
            let err = FieldnormEncoder::quantization_error(len);
            assert!(
                (0.0..=0.125).contains(&err),
                "quantization_error({len}) = {err} exceeded 12.5%"
            );
        }
    }
}

#[test]
fn test_e6_rank_select_space_overhead_percent_and_mem_size() {
    // 1 Mbit (1,048,576 bits = 16,384 u64 words = 131,072 bytes) at 50% density.
    let num_words = 16_384usize;
    let total_bits = num_words * 64;
    let raw_bytes = num_words * 8;
    let words: Vec<u64> = (0..num_words)
        .map(|i| 0x5555_5555_5555_5555u64 ^ ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)))
        .collect();
    let bv = BitVector::from_blocks(words, total_bits);

    // 1. Rank9: 16-byte Rank9Entry per 512-bit (64-byte) block = 25.0% index overhead (+ 1 sentinel entry).
    let r9 = Rank9::new(bv.clone()).unwrap();
    let r9_overhead = r9.space_overhead_percent();
    assert!(
        (25.0..=25.1).contains(&r9_overhead),
        "Rank9 space_overhead_percent on 1 Mbit must be in [25.0, 25.1], got {:.3}%",
        r9_overhead
    );
    assert_eq!(r9.mem_size(), raw_bytes + (total_bits / 512 + 1) * 16);

    // 2. RankSelectSE256: 8-byte RankCacheSE per 256-bit (32-byte) block = 25.0% rank-only overhead,
    // plus 12.5% when both sel0_cache and sel1_cache (4B per 256 zeros/ones) are enabled (~37.5%).
    let se256_rank_only = RankSelectSE256::with_options(bv.clone(), false, false).unwrap();
    assert!(
        (25.0..=25.1).contains(&se256_rank_only.space_overhead_percent()),
        "RankSelectSE256 rank-only space_overhead_percent must be in [25.0, 25.1], got {:.3}%",
        se256_rank_only.space_overhead_percent()
    );
    let se256 = RankSelectSE256::new(bv.clone()).unwrap();
    let se256_overhead = se256.space_overhead_percent();
    assert!(
        (37.4..=37.7).contains(&se256_overhead),
        "RankSelectSE256 space_overhead_percent with sel0+sel1 on 1 Mbit 50% density must be in [37.4, 37.7], got {:.3}%",
        se256_overhead
    );
    assert!(se256.mem_size() > raw_bytes);

    // 3. RankSelectSE512_64: 12-byte packed RankCacheSE512 per 512-bit (64-byte) block = 18.75% rank-only overhead,
    // plus 6.25% when both sel0_cache and sel1_cache (4B per 512 zeros/ones) are enabled (~25.0%).
    let se512_rank_only = RankSelectSE512_64::with_options(bv.clone(), false, false).unwrap();
    assert!(
        (18.7..=18.8).contains(&se512_rank_only.space_overhead_percent()),
        "RankSelectSE512_64 rank-only space_overhead_percent must be in [18.7, 18.8], got {:.3}%",
        se512_rank_only.space_overhead_percent()
    );
    let se512 = RankSelectSE512_64::new(bv.clone()).unwrap();
    let se512_overhead = se512.space_overhead_percent();
    assert!(
        (25.0..=25.1).contains(&se512_overhead),
        "RankSelectSE512_64 space_overhead_percent with sel0+sel1 on 1 Mbit 50% density must be in [25.0, 25.1], got {:.3}%",
        se512_overhead
    );

    // 4. RankSelectInterleaved256: 64-byte InterleavedLine per 256-bit (32-byte) block = 100.0% rank-only overhead,
    // plus 3.125% select cache on 50% density (~103.125%).
    let il256_rank_only = RankSelectInterleaved256::with_options(bv.clone(), false, 512).unwrap();
    assert_eq!(il256_rank_only.space_overhead_percent(), 100.0);
    let il256 = RankSelectInterleaved256::new(bv.clone()).unwrap();
    let il256_overhead = il256.space_overhead_percent();
    assert!(
        (103.0..=103.5).contains(&il256_overhead),
        "RankSelectInterleaved256 space_overhead_percent on 1 Mbit must be in [103.0, 103.5], got {:.3}%",
        il256_overhead
    );

    // 5. RankSelectSimple: 4-byte u32 rank entry per 256-bit (32-byte) block = 12.5% overhead.
    let simple = RankSelectSimple::new(bv).unwrap();
    let simple_overhead = simple.space_overhead_percent();
    assert!(
        (12.5..=12.6).contains(&simple_overhead),
        "RankSelectSimple space_overhead_percent on 1 Mbit must be in [12.5, 12.6], got {:.3}%",
        simple_overhead
    );

    // 6. RankSelectFewOne on 0.1% sparse 1-Mbit bitvector: < 5% space overhead.
    let mut sparse_words = vec![0u64; num_words];
    for i in (0..total_bits).step_by(1000) {
        sparse_words[i / 64] |= 1u64 << (i % 64);
    }
    let sparse_bv = BitVector::from_blocks(sparse_words, total_bits);
    let few = RankSelectFewOne::from_bitvector(&sparse_bv).unwrap();
    assert!(
        few.space_overhead_percent() < 5.0,
        "RankSelectFewOne on 0.1% sparse 1 Mbit must have < 5% overhead, got {:.3}%",
        few.space_overhead_percent()
    );
}
