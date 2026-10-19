// 10. String Search & BM25 Scoring (64 KiB strchr & strstr + 100,000-doc BM25)
//     Zipora: SimdStringSearch (sse42_strchr, sse42_strstr), Bm25BatchScorer, FieldnormEncoder
//     Competitors: memchr::memchr, memchr::memmem::{find, Finder}, aho_corasick::AhoCorasick

use std::collections::BTreeMap;
use std::hint::black_box;

use aho_corasick::AhoCorasick;
use memchr::{memchr as crate_memchr, memmem as crate_memmem};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use zipora::scoring::bm25::Bm25BatchScorer;
use zipora::scoring::fieldnorm::FieldnormEncoder;
use zipora::string::SimdStringSearch;

use crate::common::*;

pub fn bench_search_and_scoring(
    records: &mut Vec<BenchRecord>,
    space: &mut BTreeMap<String, f64>,
    sizes: SuiteSizes,
) {
    maybe_repin_self_within_ccd();
    record_space_metric(space, "search_and_scoring/strchr_strstr_aux_bytes", 0.0);
    {
        let haystack_len = 65_536usize;
        let haystack0 = &SEARCH_HAYSTACKS.0[0][..];
        let needle = &SEARCH_NEEDLE.0[..13];

        let searcher = SimdStringSearch::new();
        let search_iters = 128u64;
        let mut str_params =
            BenchParams::fast(haystack_len as u64, search_iters).with_threshold(9.0);
        str_params.warmup_calls = 64;
        str_params.trials = 15;
        str_params.bursts_per_trial = 10;

        wait_for_quiet_sibling();
        let mut pre_acc = 0usize;
        for _ in 0..4_096usize {
            if let Some(pos) = searcher.sse42_strchr(black_box(haystack0), black_box(b'Z')) {
                pre_acc = pre_acc.wrapping_add(pos);
            }
        }
        black_box(pre_acc);

        measure_bench(
            records,
            "search_and_scoring",
            "search_and_scoring/simd_strchr_64kib",
            "ns/call",
            str_params,
            || {
                let mut acc = 0usize;
                for _ in 0..search_iters {
                    if let Some(pos) = searcher.sse42_strchr(black_box(haystack0), black_box(b'Z'))
                    {
                        acc = acc.wrapping_add(pos);
                    }
                }
                black_box(acc);
            },
        );

        for _ in 0..2_048usize {
            if let Some(pos) = searcher.sse42_strstr(black_box(haystack0), black_box(needle)) {
                pre_acc = pre_acc.wrapping_add(pos);
            }
        }
        black_box(pre_acc);

        let mut strstr_params = str_params.with_threshold(16.0);
        strstr_params.warmup_calls = 64;
        strstr_params.bursts_per_trial = 8;
        measure_bench(
            records,
            "search_and_scoring",
            "search_and_scoring/simd_strstr_64kib",
            "ns/call",
            strstr_params,
            || {
                let mut acc = 0usize;
                for _ in 0..search_iters {
                    if let Some(pos) =
                        searcher.sse42_strstr(black_box(haystack0), black_box(needle))
                    {
                        acc = acc.wrapping_add(pos);
                    }
                }
                black_box(acc);
            },
        );

        // Competitors for strchr & strstr on the exact same single haystack and 13-byte needle
        measure_bench(
            records,
            "search_and_scoring",
            "search_and_scoring/memchr_strchr_64kib",
            "ns/call",
            str_params,
            || {
                let mut acc = 0usize;
                for _ in 0..search_iters {
                    if let Some(pos) = crate_memchr(black_box(b'Z'), black_box(haystack0)) {
                        acc = acc.wrapping_add(pos);
                    }
                }
                black_box(acc);
            },
        );

        measure_bench(
            records,
            "search_and_scoring",
            "search_and_scoring/memmem_strstr_64kib",
            "ns/call",
            strstr_params,
            || {
                let mut acc = 0usize;
                for _ in 0..search_iters {
                    if let Some(pos) = crate_memmem::find(black_box(haystack0), black_box(needle)) {
                        acc = acc.wrapping_add(pos);
                    }
                }
                black_box(acc);
            },
        );

        let finder = crate_memmem::Finder::new(needle);
        measure_bench(
            records,
            "search_and_scoring",
            "search_and_scoring/memmem_finder_strstr_64kib",
            "ns/call",
            strstr_params,
            || {
                let mut acc = 0usize;
                for _ in 0..search_iters {
                    if let Some(pos) = finder.find(black_box(haystack0)) {
                        acc = acc.wrapping_add(pos);
                    }
                }
                black_box(acc);
            },
        );

        let ac = AhoCorasick::new([needle]).unwrap();
        measure_bench(
            records,
            "search_and_scoring",
            "search_and_scoring/aho_corasick_strstr_64kib",
            "ns/call",
            strstr_params,
            || {
                let mut acc = 0usize;
                for _ in 0..search_iters {
                    if let Some(m) = ac.find(black_box(haystack0)) {
                        acc = acc.wrapping_add(m.start());
                    }
                }
                black_box(acc);
            },
        );
    }

    // BM25 hot paths (100,000 docs by default) + S36-R2 build_score_table
    let k1 = 1.2f32;
    let b = 0.75f32;
    let avg_dl = 120.0f32;
    let idf = 3.25f32;

    {
        let norm_table = FieldnormEncoder::build_norm_table(avg_dl, k1, b);
        let scorer = Bm25BatchScorer::new(&norm_table, idf, k1);

        let num_docs = sizes.bm25_docs;
        let mut rng = StdRng::seed_from_u64(SEED_BASE ^ 0x0B);
        let doc_ids: Vec<u32> = (0..num_docs as u32).map(|i| i * 3).collect();
        let tfs: Vec<u16> = (0..num_docs).map(|_| rng.random_range(1..=16)).collect();
        let norms: Vec<u8> = (0..num_docs * 3)
            .map(|_| rng.random_range(1..=200))
            .collect();
        let contiguous_norms: Vec<u8> = doc_ids.iter().map(|&d| norms[d as usize]).collect();
        let mut scores_out = vec![0.0f32; num_docs];

        let bm25_params = BenchParams::fast(num_docs as u64, num_docs as u64);
        measure_bench(
            records,
            "search_and_scoring",
            "search_and_scoring/bm25_score_single",
            "ns/doc",
            bm25_params,
            || {
                let mut sum = 0.0f32;
                for i in 0..num_docs {
                    sum += scorer.score(black_box(norms[doc_ids[i] as usize]), black_box(tfs[i]));
                }
                black_box(sum);
            },
        );

        measure_bench(
            records,
            "search_and_scoring",
            "search_and_scoring/bm25_batch_score_with_prefetch",
            "ns/doc",
            bm25_params.with_threshold(9.0),
            || {
                scorer.batch_score_with_prefetch(
                    black_box(&norms),
                    black_box(&doc_ids),
                    black_box(&tfs),
                    black_box(&mut scores_out),
                );
                black_box(scores_out[0]);
            },
        );

        let mut simd_params = bm25_params;
        simd_params.bursts_per_trial = 8;
        measure_bench(
            records,
            "search_and_scoring",
            "search_and_scoring/bm25_batch_score_simd",
            "ns/doc",
            simd_params,
            || {
                scorer.batch_score(
                    black_box(contiguous_norms.as_slice()),
                    black_box(tfs.as_slice()),
                    black_box(scores_out.as_mut_slice()),
                );
                black_box(scores_out[0]);
            },
        );
    }

    // S36-R2: FieldnormEncoder::build_score_table (256 * 256 = 65,536 cells = 256 KiB per call)
    let table_rounds = 500u64;
    let max_tf = 255u16;
    let cells_per_call = (max_tf as u64 + 1) * 256u64;
    let mut table_params =
        BenchParams::fast(cells_per_call, table_rounds * cells_per_call).with_threshold(9.0);
    table_params.warmup_calls = 4;
    table_params.trials = 7;
    table_params.bursts_per_trial = 3;
    measure_bench(
        records,
        "search_and_scoring",
        "search_and_scoring/fieldnorm_build_score_table",
        "ns/cell",
        table_params,
        || {
            for r in 0..table_rounds {
                let table = FieldnormEncoder::build_score_table(
                    black_box(avg_dl),
                    black_box(k1),
                    black_box(b),
                    black_box(idf + (r as f32) * 1e-6),
                    black_box(max_tf),
                );
                black_box(table[1][1]);
            }
        },
    );
}
