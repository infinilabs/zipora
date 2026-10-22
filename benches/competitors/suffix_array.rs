// 9. Suffix Array (10,000,000 = 10^7 bytes + 100,000 uniform substring queries)
//    Zipora: SuffixArray
//    Competitors: suffix::SuffixTable, divsufsort, cdivsufsort, libsais

use std::collections::BTreeMap;
use std::hint::black_box;

use libsais::SuffixArrayConstruction;
use sacabase::StringIndex as SacabaseStringIndex;
use suffix::SuffixTable;
use zipora::algorithms::suffix_array::SuffixArray;

use crate::common::*;

fn verify_sa_with<F: Fn(usize) -> usize>(
    text: &[u8],
    sa_len: usize,
    sa_at: F,
    expected_sample: &mut Option<Vec<u32>>,
    label: &str,
) {
    let n = text.len();
    assert_eq!(sa_len, n, "{label}: suffix array length mismatch");
    let step = (n / 256).max(1);
    let mut sample = Vec::with_capacity(258);
    let mut idx = 0usize;
    while idx < n {
        let p0 = sa_at(idx);
        assert!(p0 < n, "{label}: suffix index {p0} out of bounds for n={n}");
        if idx + 1 < n {
            let p1 = sa_at(idx + 1);
            assert!(p1 < n, "{label}: suffix index {p1} out of bounds for n={n}");
            assert!(
                text[p0..] < text[p1..],
                "{label}: suffix ordering violated at rank {idx} (p0={p0}, p1={p1})"
            );
        }
        sample.push(p0 as u32);
        idx += step;
    }
    sample.push(sa_at(n - 1) as u32);
    if let Some(ref exp) = *expected_sample {
        assert_eq!(
            &sample, exp,
            "{label}: sampled suffix array ranks disagree with reference"
        );
    } else {
        *expected_sample = Some(sample);
    }
}

fn verify_sa_usize_invariants(
    text: &[u8],
    sa: &[usize],
    expected_sample: &mut Option<Vec<u32>>,
    label: &str,
) {
    verify_sa_with(text, sa.len(), |i| sa[i], expected_sample, label);
}

fn verify_sa_u32_invariants(
    text: &[u8],
    sa: &[u32],
    expected_sample: &mut Option<Vec<u32>>,
    label: &str,
) {
    verify_sa_with(text, sa.len(), |i| sa[i] as usize, expected_sample, label);
}

fn verify_sa_i32_invariants(
    text: &[u8],
    sa: &[i32],
    expected_sample: &mut Option<Vec<u32>>,
    label: &str,
) {
    verify_sa_with(
        text,
        sa.len(),
        |i| {
            assert!(sa[i] >= 0, "{label}: negative suffix index at rank {i}");
            sa[i] as usize
        },
        expected_sample,
        label,
    );
}

pub fn bench_suffix_array(
    records: &mut Vec<BenchRecord>,
    space: &mut BTreeMap<String, f64>,
    sizes: SuiteSizes,
) {
    maybe_repin_self_within_ccd();
    trim_allocator_arena();
    let n = sizes.suffix_array_bytes;
    let zipora_sa_bytes = SuffixArray::memory_usage_for_text_len(n);
    record_space_metric(
        space,
        "suffix_array/zipora_bytes_per_byte",
        1.0 + zipora_sa_bytes as f64 / n.max(1) as f64,
    );
    record_space_metric(space, "suffix_array/suffix_crate_bytes_per_byte", 5.0);
    record_space_metric(space, "suffix_array/divsufsort_rust_bytes_per_byte", 5.0);
    record_space_metric(space, "suffix_array/cdivsufsort_c_bytes_per_byte", 5.0);
    record_space_metric(space, "suffix_array/libsais_c_bytes_per_byte", 5.0);

    let text0 = gen_enwik_like_block(n, SEED_BASE ^ 0x0A);

    let num_queries = sizes.suffix_array_queries;
    let sa_batch = (num_queries / 100).max(500).min(num_queries.max(1));
    let pat_len = 16usize;
    let max_start = n - pat_len;
    let sa_query_seed = SEED_BASE ^ 0x0A01;
    let sa_perm_len = ((n / 10) / 1000 * 1000).max(1_000);
    let sa_window = (max_start / sa_perm_len).max(1);
    let sa_perm = BijectivePerm::new(sa_perm_len, SEED_BASE ^ 0x0A02);
    let mut sa_search_params =
        BenchParams::heavy(n as u64, sa_batch as u64, 12).with_threshold(9.0);
    sa_search_params.warmup_calls = 8;
    sa_search_params.bursts_per_trial = 12;

    let mut sa_build_params = BenchParams::heavy(n as u64, n as u64, 4);
    sa_build_params.warmup_calls = 2;
    sa_build_params.bursts_per_trial = 1;
    let mut expected_sa_sample: Option<Vec<u32>> = None;
    let mut expected_occ_hits: Option<usize> = None;
    let warm_queries = sa_batch * 24;
    let expected_matched_queries = warm_queries - warm_queries / 10;

    // 1. Zipora: SuffixArray (1 instance, 1 text buffer)
    if should_run_any(&["suffix_array/construct", "suffix_array/search_query"]) {
        if should_run_bench("suffix_array/construct") {
            trim_allocator_arena();
            let mut sa_pending: Option<SuffixArray> = None;
            measure_bench_with_setup(
                records,
                "suffix_array",
                "suffix_array/construct",
                "ns/byte",
                sa_build_params,
                &mut sa_pending,
                |slot| {
                    let _ = slot.take();
                },
                |slot| {
                    let built = SuffixArray::new(black_box(&text0)).unwrap();
                    black_box(built.as_slice().len());
                    *slot = Some(built);
                },
            );
            if let Some(ref built) = sa_pending {
                assert_eq!(built.memory_usage(), zipora_sa_bytes);
                verify_sa_usize_invariants(
                    &text0,
                    built.as_slice(),
                    &mut expected_sa_sample,
                    "suffix_array/construct",
                );
            }
            drop(sa_pending);
            trim_allocator_arena();
        }
        if should_run_bench("suffix_array/search_query") {
            trim_allocator_arena();
            let sa0 = SuffixArray::new(&text0).unwrap();
            assert_eq!(sa0.memory_usage(), zipora_sa_bytes);
            verify_sa_usize_invariants(
                &text0,
                sa0.as_slice(),
                &mut expected_sa_sample,
                "suffix_array/search_query",
            );
            let mut warm_sa_hits = 0usize;
            for idx in 0..warm_queries {
                let pos = uniform_index_at(sa_query_seed ^ 0x55, idx, max_start);
                let mut pat = [0u8; 16];
                pat.copy_from_slice(&text0[pos..pos + pat_len]);
                let is_miss = idx % 10 == 9;
                if is_miss {
                    pat[pat_len - 1] = b'~';
                }
                let (_, count) = sa0.search(black_box(&text0), black_box(&pat[..]));
                if is_miss {
                    assert_eq!(
                        count, 0,
                        "suffix_array/search_query: expected miss at idx={idx}"
                    );
                } else {
                    assert!(
                        count >= 1,
                        "suffix_array/search_query: expected hit at idx={idx}"
                    );
                }
                warm_sa_hits = warm_sa_hits.wrapping_add(count);
            }
            expected_occ_hits = Some(warm_sa_hits);
            black_box(warm_sa_hits);

            measure_bench(
                records,
                "suffix_array",
                "suffix_array/search_query",
                "ns/op",
                sa_search_params,
                || {
                    let base = current_burst_index() * sa_batch;
                    let mut hits = 0usize;
                    let mut q_idx = 0usize;
                    sa_perm.for_each_slice(base, sa_batch, |id| {
                        let pos = (id * sa_window + uniform_index_at(sa_query_seed, id, sa_window))
                            .min(max_start.saturating_sub(1));
                        let mut pat = [0u8; 16];
                        pat.copy_from_slice(&text0[pos..pos + pat_len]);
                        if q_idx % 10 == 9 {
                            pat[pat_len - 1] = b'~';
                        }
                        q_idx = q_idx.wrapping_add(1);
                        let (_, count) = sa0.search(black_box(&text0), black_box(&pat[..]));
                        hits = hits.wrapping_add(count);
                    });
                    black_box(hits);
                },
            );
            drop(sa0);
            trim_allocator_arena();
        }
    }

    // 2. Competitor: suffix::SuffixTable (pure-Rust SA-IS, 1 instance, 1 text buffer)
    if should_run_bench("suffix_array/suffix_crate/construct")
        || should_run_bench("suffix_array/suffix_crate/search_query")
    {
        let text_str0 = std::str::from_utf8(&text0).unwrap();
        if should_run_bench("suffix_array/suffix_crate/construct") {
            trim_allocator_arena();
            let mut st_pending: Option<SuffixTable<'_, '_>> = None;
            measure_bench_with_setup(
                records,
                "suffix_array",
                "suffix_array/suffix_crate/construct",
                "ns/byte",
                sa_build_params,
                &mut st_pending,
                |slot| {
                    let _ = slot.take();
                },
                |slot| {
                    let built = SuffixTable::new(black_box(text_str0));
                    black_box(built.table().len());
                    *slot = Some(built);
                },
            );
            if let Some(ref built) = st_pending {
                verify_sa_u32_invariants(
                    &text0,
                    built.table(),
                    &mut expected_sa_sample,
                    "suffix_array/suffix_crate/construct",
                );
            }
            drop(st_pending);
            trim_allocator_arena();
        }
        if should_run_bench("suffix_array/suffix_crate/search_query") {
            trim_allocator_arena();
            let st0 = SuffixTable::new(text_str0);
            verify_sa_u32_invariants(
                &text0,
                st0.table(),
                &mut expected_sa_sample,
                "suffix_array/suffix_crate/search_query",
            );

            let mut warm_hits = 0usize;
            for idx in 0..warm_queries {
                let pos = uniform_index_at(sa_query_seed ^ 0x55, idx, max_start);
                let mut pat = [0u8; 16];
                pat.copy_from_slice(&text0[pos..pos + pat_len]);
                let is_miss = idx % 10 == 9;
                if is_miss {
                    pat[pat_len - 1] = b'~';
                }
                let pat_str = std::str::from_utf8(&pat).unwrap();
                let count = st0.positions(black_box(pat_str)).len();
                if is_miss {
                    assert_eq!(
                        count, 0,
                        "suffix_array/suffix_crate/search_query: expected miss at idx={idx}"
                    );
                } else {
                    assert!(
                        count >= 1,
                        "suffix_array/suffix_crate/search_query: expected hit at idx={idx}"
                    );
                }
                warm_hits = warm_hits.wrapping_add(count);
            }
            if let Some(exp) = expected_occ_hits {
                assert_eq!(
                    warm_hits, exp,
                    "suffix_array/suffix_crate/search_query: hit count mismatch vs Zipora"
                );
            }
            black_box(warm_hits);

            measure_bench(
                records,
                "suffix_array",
                "suffix_array/suffix_crate/search_query",
                "ns/op",
                sa_search_params,
                || {
                    let base = current_burst_index() * sa_batch;
                    let mut hits = 0usize;
                    let mut q_idx = 0usize;
                    sa_perm.for_each_slice(base, sa_batch, |id| {
                        let pos = (id * sa_window + uniform_index_at(sa_query_seed, id, sa_window))
                            .min(max_start.saturating_sub(1));
                        let mut pat = [0u8; 16];
                        pat.copy_from_slice(&text0[pos..pos + pat_len]);
                        if q_idx % 10 == 9 {
                            pat[pat_len - 1] = b'~';
                        }
                        q_idx = q_idx.wrapping_add(1);
                        let pat_str = std::str::from_utf8(&pat).unwrap();
                        let positions = st0.positions(black_box(pat_str));
                        hits = hits.wrapping_add(positions.len());
                    });
                    black_box(hits);
                },
            );
            drop(st0);
            trim_allocator_arena();
        }
    }

    // 3. Competitor: divsufsort (pure-Rust divsufsort, 1 instance, 1 text buffer)
    if should_run_bench("suffix_array/divsufsort_rust/construct")
        || should_run_bench("suffix_array/divsufsort_rust/search_query")
    {
        if should_run_bench("suffix_array/divsufsort_rust/construct") {
            trim_allocator_arena();
            let mut dss_pending: Option<sacabase::SuffixArray<'_, i32>> = None;
            measure_bench_with_setup(
                records,
                "suffix_array",
                "suffix_array/divsufsort_rust/construct",
                "ns/byte",
                sa_build_params,
                &mut dss_pending,
                |slot| {
                    let _ = slot.take();
                },
                |slot| {
                    let built = divsufsort::sort(black_box(&text0));
                    black_box(built.text().len());
                    *slot = Some(built);
                },
            );
            if let Some(built) = dss_pending.take() {
                let (_, sa_vec) = built.into_parts();
                verify_sa_i32_invariants(
                    &text0,
                    &sa_vec,
                    &mut expected_sa_sample,
                    "suffix_array/divsufsort_rust/construct",
                );
                drop(sa_vec);
            }
            trim_allocator_arena();
        }
        if should_run_bench("suffix_array/divsufsort_rust/search_query") {
            trim_allocator_arena();
            let dss0 = divsufsort::sort(&text0);

            let mut warm_hits = 0usize;
            for idx in 0..warm_queries {
                let pos = uniform_index_at(sa_query_seed ^ 0x55, idx, max_start);
                let mut pat = [0u8; 16];
                pat.copy_from_slice(&text0[pos..pos + pat_len]);
                let is_miss = idx % 10 == 9;
                if is_miss {
                    pat[pat_len - 1] = b'~';
                }
                let m = dss0.longest_substring_match(black_box(&pat[..]));
                if is_miss {
                    assert!(
                        m.len < pat_len,
                        "suffix_array/divsufsort_rust/search_query: expected miss at idx={idx}"
                    );
                } else {
                    assert_eq!(
                        m.len, pat_len,
                        "suffix_array/divsufsort_rust/search_query: expected hit at idx={idx}"
                    );
                    assert_eq!(m.as_bytes(), &pat[..]);
                    warm_hits = warm_hits.wrapping_add(1);
                }
            }
            assert_eq!(
                warm_hits, expected_matched_queries,
                "suffix_array/divsufsort_rust/search_query: matched query count mismatch"
            );
            black_box(warm_hits);

            measure_bench(
                records,
                "suffix_array",
                "suffix_array/divsufsort_rust/search_query",
                "ns/op",
                sa_search_params,
                || {
                    let base = current_burst_index() * sa_batch;
                    let mut hits = 0usize;
                    let mut q_idx = 0usize;
                    sa_perm.for_each_slice(base, sa_batch, |id| {
                        let pos = (id * sa_window + uniform_index_at(sa_query_seed, id, sa_window))
                            .min(max_start.saturating_sub(1));
                        let mut pat = [0u8; 16];
                        pat.copy_from_slice(&text0[pos..pos + pat_len]);
                        if q_idx % 10 == 9 {
                            pat[pat_len - 1] = b'~';
                        }
                        q_idx = q_idx.wrapping_add(1);
                        let m = dss0.longest_substring_match(black_box(&pat[..]));
                        if m.len == pat_len {
                            hits = hits.wrapping_add(1);
                        }
                    });
                    black_box(hits);
                },
            );
            let (_, sa_vec) = dss0.into_parts();
            verify_sa_i32_invariants(
                &text0,
                &sa_vec,
                &mut expected_sa_sample,
                "suffix_array/divsufsort_rust/search_query",
            );
            drop(sa_vec);
            trim_allocator_arena();
        }
    }

    // 4. Competitor: cdivsufsort (C libdivsufsort binding, 1 instance, 1 text buffer)
    if should_run_bench("suffix_array/cdivsufsort_c/construct")
        || should_run_bench("suffix_array/cdivsufsort_c/search_query")
    {
        if should_run_bench("suffix_array/cdivsufsort_c/construct") {
            trim_allocator_arena();
            let mut cdss_pending: Option<sacabase::SuffixArray<'_, i32>> = None;
            measure_bench_with_setup(
                records,
                "suffix_array",
                "suffix_array/cdivsufsort_c/construct",
                "ns/byte",
                sa_build_params,
                &mut cdss_pending,
                |slot| {
                    let _ = slot.take();
                },
                |slot| {
                    let built = cdivsufsort::sort(black_box(&text0));
                    black_box(built.text().len());
                    *slot = Some(built);
                },
            );
            if let Some(built) = cdss_pending.take() {
                let (_, sa_vec) = built.into_parts();
                verify_sa_i32_invariants(
                    &text0,
                    &sa_vec,
                    &mut expected_sa_sample,
                    "suffix_array/cdivsufsort_c/construct",
                );
                drop(sa_vec);
            }
            trim_allocator_arena();
        }
        if should_run_bench("suffix_array/cdivsufsort_c/search_query") {
            trim_allocator_arena();
            let cdss0 = cdivsufsort::sort(&text0);

            let mut warm_hits = 0usize;
            for idx in 0..warm_queries {
                let pos = uniform_index_at(sa_query_seed ^ 0x55, idx, max_start);
                let mut pat = [0u8; 16];
                pat.copy_from_slice(&text0[pos..pos + pat_len]);
                let is_miss = idx % 10 == 9;
                if is_miss {
                    pat[pat_len - 1] = b'~';
                }
                let m = cdss0.longest_substring_match(black_box(&pat[..]));
                if is_miss {
                    assert!(
                        m.len < pat_len,
                        "suffix_array/cdivsufsort_c/search_query: expected miss at idx={idx}"
                    );
                } else {
                    assert_eq!(
                        m.len, pat_len,
                        "suffix_array/cdivsufsort_c/search_query: expected hit at idx={idx}"
                    );
                    assert_eq!(m.as_bytes(), &pat[..]);
                    warm_hits = warm_hits.wrapping_add(1);
                }
            }
            assert_eq!(
                warm_hits, expected_matched_queries,
                "suffix_array/cdivsufsort_c/search_query: matched query count mismatch"
            );
            black_box(warm_hits);

            measure_bench(
                records,
                "suffix_array",
                "suffix_array/cdivsufsort_c/search_query",
                "ns/op",
                sa_search_params,
                || {
                    let base = current_burst_index() * sa_batch;
                    let mut hits = 0usize;
                    let mut q_idx = 0usize;
                    sa_perm.for_each_slice(base, sa_batch, |id| {
                        let pos = (id * sa_window + uniform_index_at(sa_query_seed, id, sa_window))
                            .min(max_start.saturating_sub(1));
                        let mut pat = [0u8; 16];
                        pat.copy_from_slice(&text0[pos..pos + pat_len]);
                        if q_idx % 10 == 9 {
                            pat[pat_len - 1] = b'~';
                        }
                        q_idx = q_idx.wrapping_add(1);
                        let m = cdss0.longest_substring_match(black_box(&pat[..]));
                        if m.len == pat_len {
                            hits = hits.wrapping_add(1);
                        }
                    });
                    black_box(hits);
                },
            );
            let (_, sa_vec) = cdss0.into_parts();
            verify_sa_i32_invariants(
                &text0,
                &sa_vec,
                &mut expected_sa_sample,
                "suffix_array/cdivsufsort_c/search_query",
            );
            drop(sa_vec);
            trim_allocator_arena();
        }
    }

    // 5. Competitor: libsais (C libsais single-threaded binding, 1 instance, 1 text buffer)
    if should_run_bench("suffix_array/libsais_c/construct")
        || should_run_bench("suffix_array/libsais_c/search_query")
    {
        fn build_libsais(txt: &[u8]) -> sacabase::SuffixArray<'_, i32> {
            let sa_vec: Vec<i32> = SuffixArrayConstruction::for_text(txt)
                .in_owned_buffer32()
                .single_threaded()
                .run()
                .unwrap()
                .into_vec();
            sacabase::SuffixArray::new(txt, sa_vec)
        }
        if should_run_bench("suffix_array/libsais_c/construct") {
            trim_allocator_arena();
            let mut sais_pending: Option<sacabase::SuffixArray<'_, i32>> = None;
            measure_bench_with_setup(
                records,
                "suffix_array",
                "suffix_array/libsais_c/construct",
                "ns/byte",
                sa_build_params,
                &mut sais_pending,
                |slot| {
                    let _ = slot.take();
                },
                |slot| {
                    let built = build_libsais(black_box(&text0));
                    black_box(built.text().len());
                    *slot = Some(built);
                },
            );
            if let Some(built) = sais_pending.take() {
                let (_, sa_vec) = built.into_parts();
                verify_sa_i32_invariants(
                    &text0,
                    &sa_vec,
                    &mut expected_sa_sample,
                    "suffix_array/libsais_c/construct",
                );
                drop(sa_vec);
            }
            trim_allocator_arena();
        }
        if should_run_bench("suffix_array/libsais_c/search_query") {
            trim_allocator_arena();
            let sais0 = build_libsais(&text0);

            let mut warm_hits = 0usize;
            for idx in 0..warm_queries {
                let pos = uniform_index_at(sa_query_seed ^ 0x55, idx, max_start);
                let mut pat = [0u8; 16];
                pat.copy_from_slice(&text0[pos..pos + pat_len]);
                let is_miss = idx % 10 == 9;
                if is_miss {
                    pat[pat_len - 1] = b'~';
                }
                let m = sais0.longest_substring_match(black_box(&pat[..]));
                if is_miss {
                    assert!(
                        m.len < pat_len,
                        "suffix_array/libsais_c/search_query: expected miss at idx={idx}"
                    );
                } else {
                    assert_eq!(
                        m.len, pat_len,
                        "suffix_array/libsais_c/search_query: expected hit at idx={idx}"
                    );
                    assert_eq!(m.as_bytes(), &pat[..]);
                    warm_hits = warm_hits.wrapping_add(1);
                }
            }
            assert_eq!(
                warm_hits, expected_matched_queries,
                "suffix_array/libsais_c/search_query: matched query count mismatch"
            );
            black_box(warm_hits);

            measure_bench(
                records,
                "suffix_array",
                "suffix_array/libsais_c/search_query",
                "ns/op",
                sa_search_params,
                || {
                    let base = current_burst_index() * sa_batch;
                    let mut hits = 0usize;
                    let mut q_idx = 0usize;
                    sa_perm.for_each_slice(base, sa_batch, |id| {
                        let pos = (id * sa_window + uniform_index_at(sa_query_seed, id, sa_window))
                            .min(max_start.saturating_sub(1));
                        let mut pat = [0u8; 16];
                        pat.copy_from_slice(&text0[pos..pos + pat_len]);
                        if q_idx % 10 == 9 {
                            pat[pat_len - 1] = b'~';
                        }
                        q_idx = q_idx.wrapping_add(1);
                        let m = sais0.longest_substring_match(black_box(&pat[..]));
                        if m.len == pat_len {
                            hits = hits.wrapping_add(1);
                        }
                    });
                    black_box(hits);
                },
            );
            let (_, sa_vec) = sais0.into_parts();
            verify_sa_i32_invariants(
                &text0,
                &sa_vec,
                &mut expected_sa_sample,
                "suffix_array/libsais_c/search_query",
            );
            drop(sa_vec);
            trim_allocator_arena();
        }
    }
}
