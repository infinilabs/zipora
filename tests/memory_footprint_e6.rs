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

    // 4. RankSelectInterleaved256: 40-byte InterleavedLine per 256-bit (32-byte) block = 25.0% rank-only overhead,
    // plus 6.25% select1 + select0 caches on 50% density (~31.25%).
    let il256_rank_only = RankSelectInterleaved256::with_options(bv.clone(), false, 512).unwrap();
    assert_eq!(il256_rank_only.space_overhead_percent(), 25.0);
    let il256 = RankSelectInterleaved256::new(bv.clone()).unwrap();
    let il256_overhead = il256.space_overhead_percent();
    assert!(
        (31.2..=31.4).contains(&il256_overhead),
        "RankSelectInterleaved256 space_overhead_percent on 1 Mbit must be in [31.2, 31.4], got {:.3}%",
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

#[test]
fn test_s51_r2_baseline_per_row_provenance_and_unrestamped_headers() {
    use std::collections::HashMap;
    use std::path::Path;
    use std::process::Command;

    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));

    for (suite, expected_total) in [
        ("core_bench", 80usize),
        ("competitor_bench", 181usize),
        ("e2e_search", 7usize),
    ] {
        let path = manifest_dir
            .join("benches/baselines/bindiego")
            .join(format!("{suite}.json"));
        let raw = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
        let doc: serde_json::Value = serde_json::from_str(&raw)
            .unwrap_or_else(|e| panic!("failed to parse {}: {e}", path.display()));

        let host_meta = &doc["host_metadata"];
        let header_commit = host_meta["git_commit"]
            .as_str()
            .unwrap_or_else(|| panic!("{suite}: missing host_metadata.git_commit"));
        let header_sha = host_meta["binary_sha256"]
            .as_str()
            .unwrap_or_else(|| panic!("{suite}: missing host_metadata.binary_sha256"));

        let benches = doc["benchmarks"]
            .as_array()
            .unwrap_or_else(|| panic!("{suite}: missing benchmarks array"));
        assert_eq!(benches.len(), expected_total, "{suite}: row count mismatch");

        let mut commit_shas: HashMap<String, String> = HashMap::new();
        let mut row_provenance: HashMap<String, (String, String)> = HashMap::new();

        for row in benches {
            let name = row["name"].as_str().expect("benchmark row name");
            let row_commit = row["measured_at_commit"].as_str().unwrap_or_else(|| {
                panic!("{suite}: benchmark '{name}' missing per-row measured_at_commit")
            });
            let row_sha = row["binary_sha256"].as_str().unwrap_or_else(|| {
                panic!("{suite}: benchmark '{name}' missing per-row binary_sha256")
            });
            assert!(
                (7..=40).contains(&row_commit.len())
                    && row_commit.chars().all(|c| c.is_ascii_hexdigit()),
                "{suite}: benchmark '{name}' has invalid measured_at_commit '{row_commit}'"
            );
            assert!(
                row_sha.len() == 64 && row_sha.chars().all(|c| c.is_ascii_hexdigit()),
                "{suite}: benchmark '{name}' has invalid binary_sha256 '{row_sha}'"
            );
            // Every commit within a suite must map to a single consistent binary_sha256.
            if let Some(prev_sha) = commit_shas.get(row_commit) {
                assert_eq!(
                    prev_sha, row_sha,
                    "{suite}: inconsistent binary_sha256 for commit {row_commit}"
                );
            } else {
                let status = Command::new("git")
                    .args(["cat-file", "-e", &format!("{row_commit}^{{commit}}")])
                    .current_dir(manifest_dir)
                    .status()
                    .expect("git cat-file");
                assert!(
                    status.success(),
                    "{suite}: measured_at_commit '{row_commit}' does not exist in git history"
                );
                commit_shas.insert(row_commit.to_string(), row_sha.to_string());
            }
            row_provenance.insert(
                name.to_string(),
                (row_commit.to_string(), row_sha.to_string()),
            );
        }

        // Top-level host_metadata must match the known binary_sha256 for header_commit.
        assert_eq!(
            commit_shas.get(header_commit).map(String::as_str),
            Some(header_sha),
            "{suite}: host_metadata ({header_commit}, {header_sha}) does not match row provenance for {header_commit}"
        );

        if suite == "competitor_bench" {
            let comparisons = doc["same_run_comparisons"]
                .as_array()
                .expect("competitor_bench: missing same_run_comparisons");
            let mut cross_count = 0usize;
            for cmp in comparisons {
                let zb = cmp["zipora_bench"].as_str().expect("zipora_bench");
                let cb = cmp["competitor_bench"].as_str().expect("competitor_bench");
                let (exp_z_commit, exp_z_sha) = &row_provenance[zb];
                let (exp_c_commit, exp_c_sha) = &row_provenance[cb];
                assert_eq!(
                    cmp["zipora_measured_at_commit"].as_str(),
                    Some(exp_z_commit.as_str()),
                    "comparison {zb} vs {cb}: zipora_measured_at_commit mismatch"
                );
                assert_eq!(
                    cmp["zipora_binary_sha256"].as_str(),
                    Some(exp_z_sha.as_str()),
                    "comparison {zb} vs {cb}: zipora_binary_sha256 mismatch"
                );
                assert_eq!(
                    cmp["competitor_measured_at_commit"].as_str(),
                    Some(exp_c_commit.as_str()),
                    "comparison {zb} vs {cb}: competitor_measured_at_commit mismatch"
                );
                assert_eq!(
                    cmp["competitor_binary_sha256"].as_str(),
                    Some(exp_c_sha.as_str()),
                    "comparison {zb} vs {cb}: competitor_binary_sha256 mismatch"
                );
                let expected_cross = exp_z_sha != exp_c_sha;
                let actual_cross = cmp["cross_binary_comparison"].as_bool().unwrap_or_else(|| {
                    panic!("comparison {zb} vs {cb}: missing cross_binary_comparison boolean")
                });
                assert_eq!(
                    actual_cross, expected_cross,
                    "comparison {zb} vs {cb}: cross_binary_comparison mismatch"
                );
                if actual_cross {
                    cross_count += 1;
                }
            }
            assert!(
                cross_count == 0 || cross_count == 25,
                "competitor_bench: expected either 0 (uniform re-baseline) or 25 cross-binary comparisons, got {cross_count}"
            );
        }
    }

    // Verify bench_compare.py validates each row's sha against known binaries for that commit
    // (not a majority vote), so a self-consistent restamp with a forged sha fails closed.
    let bench_compare = std::fs::read_to_string(manifest_dir.join("scripts/bench_compare.py"))
        .expect("read scripts/bench_compare.py");
    assert!(
        bench_compare.contains("KNOWN_BASELINE_BINARIES")
            && !bench_compare.contains("maj_commit, maj_sha = max("),
        "scripts/bench_compare.py must validate against KNOWN_BASELINE_BINARIES rather than a majority vote"
    );
    let mut forged: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(manifest_dir.join("benches/baselines/bindiego/core_bench.json"))
            .expect("read core_bench.json"),
    )
    .expect("parse core_bench.json");
    let fake_sha = "a".repeat(64);
    forged["host_metadata"]["git_commit"] = serde_json::Value::String("4b490ff".to_string());
    forged["host_metadata"]["binary_sha256"] = serde_json::Value::String(fake_sha.clone());
    for row in forged["benchmarks"].as_array_mut().unwrap() {
        row["measured_at_commit"] = serde_json::Value::String("4b490ff".to_string());
        row["binary_sha256"] = serde_json::Value::String(fake_sha.clone());
    }
    let tmp_forged = std::env::temp_dir().join(format!(
        "zipora_test_s52_r1_forged_sha_{}.json",
        std::process::id()
    ));
    std::fs::write(&tmp_forged, serde_json::to_string(&forged).unwrap()).unwrap();
    let status = std::process::Command::new("python3")
        .arg(manifest_dir.join("scripts/bench_compare.py"))
        .arg("--suite")
        .arg("core_bench")
        .arg("--baseline")
        .arg(&tmp_forged)
        .arg("--current")
        .arg(manifest_dir.join("benches/baselines/bindiego/core_bench.json"))
        .output()
        .expect("run bench_compare.py on forged sha baseline");
    let _ = std::fs::remove_file(&tmp_forged);
    assert!(
        !status.status.success(),
        "bench_compare.py must reject a baseline whose rows and host_metadata are uniformly restamped with a forged binary_sha256"
    );

    let e2_e3 = std::fs::read_to_string(manifest_dir.join("docs/review/e2_e3.md"))
        .expect("read docs/review/e2_e3.md");
    assert!(
        !e2_e3.contains("generated directly from a full run of its own binary"),
        "docs/review/e2_e3.md must not claim spliced baselines were generated from a single full run"
    );
    assert!(
        e2_e3.contains("measured_at_commit") && e2_e3.contains("25 cross-binary"),
        "docs/review/e2_e3.md must document per-row measured_at_commit provenance and all 25 cross-binary comparisons"
    );

    let e4 = std::fs::read_to_string(manifest_dir.join("docs/review/e4.md"))
        .expect("read docs/review/e4.md");
    assert!(
        !e4.contains("generated from its own compiled benchmark binary on the clean code tree"),
        "docs/review/e4.md must not claim spliced baselines came from a single clean tree without per-row provenance"
    );
    assert!(
        e4.contains("measured_at_commit") && e4.contains("25 cross-binary"),
        "docs/review/e4.md must document per-row measured_at_commit provenance and all 25 cross-binary comparisons"
    );

    let e1 = std::fs::read_to_string(manifest_dir.join("docs/review/e1.md"))
        .expect("read docs/review/e1.md");
    assert!(
        !e1.contains("natively measured at `4b490ff`, default `THP-off`"),
        "docs/review/e1.md:98 must not claim all 80 core_bench rows were measured at 4b490ff"
    );
    assert!(
        e1.contains("measured_at_commit"),
        "docs/review/e1.md must document per-row measured_at_commit provenance"
    );
}

#[test]
fn test_s51_r3_review_ledger_identifiers_and_metrics_match_code_and_baselines() {
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));

    let lockfree_pool = std::fs::read_to_string(manifest_dir.join("src/memory/lockfree_pool.rs"))
        .expect("read src/memory/lockfree_pool.rs");
    let e4 = std::fs::read_to_string(manifest_dir.join("docs/review/e4.md"))
        .expect("read docs/review/e4.md");
    let e2_e3 = std::fs::read_to_string(manifest_dir.join("docs/review/e2_e3.md"))
        .expect("read docs/review/e2_e3.md");
    let e1 = std::fs::read_to_string(manifest_dir.join("docs/review/e1.md"))
        .expect("read docs/review/e1.md");
    let gen_e3 = std::fs::read_to_string(manifest_dir.join("scripts/gen_e3_summary.py"))
        .expect("read scripts/gen_e3_summary.py");

    // 1. Nonexistent `deallocate_to_fast_bin` must not appear in active source, scripts, or Stage 15/16 ledgers.
    for (label, content) in [
        ("src/memory/lockfree_pool.rs", &lockfree_pool),
        ("docs/review/e4.md", &e4),
        ("docs/review/e2_e3.md", &e2_e3),
        ("scripts/gen_e3_summary.py", &gen_e3),
    ] {
        assert!(
            !content.contains("deallocate_to_fast_bin"),
            "{label} still references nonexistent identifier `deallocate_to_fast_bin`"
        );
    }
    assert!(
        e4.contains("push_free_fast_bin_offset") && e2_e3.contains("push_free_fast_bin_offset"),
        "docs/review/e4.md and e2_e3.md must reference PoolArena::push_free_fast_bin_offset"
    );

    // 2. suffix_array/construct trial/warmup count in benches/core_bench.rs vs benches/competitors/suffix_array.rs.
    assert!(
        !e4.contains("warmup_calls = 2, trials = 4"),
        "docs/review/e4.md must not claim benches/core_bench.rs uses warmup_calls = 2, trials = 4"
    );

    // 3. `non_empty_bins.fetch_or` instruction lowering (`lock cmpxchgq` loop, not `lock orq`).
    assert!(
        !e4.contains("lock orq"),
        "docs/review/e4.md must not claim fetch_or lowers to `lock orq` when prev == 0 is inspected"
    );
    assert!(
        e4.contains("lock cmpxchgq"),
        "docs/review/e4.md must document `lock cmpxchgq`"
    );

    // 4. `e2_e3.md` E3.5 must distinguish pre-E4(f) 4-RMW Treiber stack (24.003 / 21.853 ns/op) from Stage 16 3-RMW magazine path.
    assert!(
        !e2_e3.contains(
            "visible directly in the loop disassembly above at `9.104 ns/op` 1T and `9.201 ns/op` 4T"
        ),
        "docs/review/e2_e3.md must not attribute 9.104 / 9.201 ns/op to the pre-E4(f) 4-RMW Treiber disassembly"
    );

    // 5. StreamVByte line references and attribution in docs/review/e4.md.
    assert!(
        !e4.contains("stream_vbyte.rs:78-83") && !e4.contains("stream_vbyte.rs:646-697"),
        "docs/review/e4.md contains stale stream_vbyte.rs line numbers"
    );
    assert!(
        e4.contains("src/compression/stream_vbyte.rs:34-64")
            && e4.contains("src/compression/stream_vbyte.rs:627-700"),
        "docs/review/e4.md must cite exact stream_vbyte.rs:34-64 and 627-700 line ranges"
    );
    assert!(
        e4.contains("aligning `SHUFFLE_TABLE` (`AlignedShuffleTable`) alone")
            || e4.contains("aligning `SHUFFLE_TABLE` alone"),
        "docs/review/e4.md must disclose that aligning SHUFFLE_TABLE alone kept the 0.209-0.235 ns/int swing and the flat ~0.186 ns/int comes from the 4x-unrolled raw-pointer loop"
    );

    // 6. `stream_vbyte` unit test count in docs/review/e4.md is 21/21, not 22/22.
    assert!(
        !e4.contains("`22/22` `stream_vbyte`"),
        "docs/review/e4.md must not claim 22/22 stream_vbyte unit tests"
    );
    assert!(
        e4.contains("`21/21` `stream_vbyte`"),
        "docs/review/e4.md must report 21/21 stream_vbyte unit tests"
    );

    // 7. `docs/review/e1.md:351` radix_sort test name matches src/algorithms/radix_sort/tests.rs.
    let radix_tests =
        std::fs::read_to_string(manifest_dir.join("src/algorithms/radix_sort/tests.rs"))
            .expect("read src/algorithms/radix_sort/tests.rs");
    assert!(
        radix_tests.contains("fn test_s20_r3_parallel_merge_tree_stable_and_matches_std"),
        "expected test_s20_r3_parallel_merge_tree_stable_and_matches_std in radix_sort/tests.rs"
    );
    assert!(
        !e1.contains("test_s20_r3_par_merge_preserves_cross_half_tie_order_exhaustively")
            && e1.contains("test_s20_r3_parallel_merge_tree_stable_and_matches_std"),
        "docs/review/e1.md must cite the real test name test_s20_r3_parallel_merge_tree_stable_and_matches_std"
    );

    // 8. `docs/review/e4.md` must not make an unproven binary code-layout placement claim on huffman_o1/decode_64kib.
    assert!(
        !e4.contains("reflects binary code-layout placement rather than an algorithmic change"),
        "docs/review/e4.md must not attribute huffman_o1/decode_64kib re-baseline to an unproven layout shift"
    );
    assert!(
        e4.contains("438.26 us") && e4.contains("348.90 us"),
        "docs/review/e4.md must report the exact JSON total_ns 438.26 us -> 348.90 us for huffman_o1/decode_64kib"
    );

    // 9. S52-R8: no nonexistent identifiers, stale line ranges, anachronistic Stage 15 refs, or stale unsafe counts.
    for (label, content) in [
        ("docs/review/e4.md", &e4),
        ("docs/review/e2_e3.md", &e2_e3),
        ("docs/review/e1.md", &e1),
        ("scripts/gen_e3_summary.py", &gen_e3),
    ] {
        for forbidden in [
            "insert_code_checked",
            "Kraft-dyadic",
            "row_provenance_git_commit",
            "row_provenance_binary_sha256",
            "benches/competitors/suffix_array.rs:116-118",
            "while standalone `DoubleArrayTrie` remains `8 B/state`",
            "1,429/1,429",
            "benches/common/datasets.rs:424-441",
            "benches/common/datasets.rs:530-545",
            "benches/common/datasets.rs:564-579",
            "benches/common/harness.rs:726",
            "benches/common/harness.rs:736",
        ] {
            assert!(
                !content.contains(forbidden),
                "S52-R8: {label} still contains stale or nonexistent reference `{forbidden}`"
            );
        }
    }
    assert!(
        e4.contains("benches/competitors/suffix_array.rs:122-124")
            && e4.contains("1,427/1,427 = 100.0%"),
        "S52-R8: docs/review/e4.md must cite benches/competitors/suffix_array.rs:122-124 and 1,427/1,427 unsafe sites"
    );
    assert!(
        e2_e3.contains("benches/common/datasets.rs:533-548")
            && e2_e3.contains("benches/common/harness.rs:712")
            && e2_e3.contains("1,427/1,427 = 100.0%")
            && e2_e3.contains("later refactored into `LockFreeMemoryPool::pop_fast_bin` and `PoolArena::push_free_fast_bin_offset`"),
        "S52-R8: docs/review/e2_e3.md must have accurate datasets.rs/harness.rs line refs, 1,427 unsafe count, and non-anachronistic E3.5 Stage 15/16 pool attribution"
    );
}

#[test]
fn test_s53_r3_benchmark_provenance_out_of_timed_heap_and_once_in_e2e() {
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let datasets = std::fs::read_to_string(manifest_dir.join("benches/common/datasets.rs"))
        .expect("read benches/common/datasets.rs");
    let binary_meta = std::fs::read_to_string(manifest_dir.join("benches/common/binary_meta.rs"))
        .expect("read benches/common/binary_meta.rs");
    let e2e = std::fs::read_to_string(manifest_dir.join("benches/e2e_search_bench.rs"))
        .expect("read benches/e2e_search_bench.rs");

    assert!(
        !datasets.contains("Command::new(\"git\")")
            && !datasets.contains("detect_self_binary_path_and_sha256"),
        "S53-R3: benches/common/datasets.rs must not spawn git or read/hash the binary at init"
    );
    assert!(
        binary_meta.contains("sha256_reader")
            && !binary_meta.contains("std::fs::read(&exe)")
            && binary_meta.contains("ZIPORA_BENCH_GIT_COMMIT"),
        "S53-R3: benches/common/binary_meta.rs must stream-hash with a stack buffer and honor ZIPORA_BENCH_GIT_COMMIT"
    );
    let e2e_measure_fn = e2e
        .split("fn measure_bench<F>")
        .nth(1)
        .and_then(|s| s.split("fn read_vm_hwm_kib").next())
        .expect("locate measure_bench in e2e_search_bench.rs");
    assert!(
        !e2e_measure_fn.contains("detect_git_commit")
            && !e2e_measure_fn.contains("detect_self_binary_path_and_sha256"),
        "S53-R3: e2e_search_bench::measure_bench must not run git or self-hash inside per-benchmark measurement"
    );
    assert_eq!(
        e2e.matches("collect_host_metadata(").count(),
        2,
        "S53-R3: e2e_search_bench.rs must define collect_host_metadata once and call it once"
    );
    let rss_pos = e2e
        .find("let vm_hwm_kib = read_vm_hwm_kib();")
        .expect("find read_vm_hwm_kib call");
    let meta_pos = e2e
        .find("let host_metadata = collect_host_metadata(primary_core);")
        .expect("find collect_host_metadata call");
    assert!(
        meta_pos > rss_pos,
        "S53-R3: e2e_search_bench.rs must call collect_host_metadata only once after all timed benchmarks and read_vm_hwm_kib()"
    );
}
