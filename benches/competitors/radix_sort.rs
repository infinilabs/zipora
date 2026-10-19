//! Family 8: Radix Sort (10,000,000 = 10^7 elements for `u32`, `u64`, `(u64, u64)`).
//!
//! Zipora: `RadixSort` (`lsd_u32`, `lsd_u64`), `KeyValueRadixSort<u64, u64>` (`kv_u64_u64`)
//! Competitors: `voracious_radix_sort`, `rdst` (single-threaded), `[T]::sort_unstable`
//!
//! Fairness invariant (`S42-R1`): every implementation uses **1 sorter and 1 work buffer**,
//! with `warmup_calls = 1`, `trials = 3`, `bursts_per_trial = 1`, and zero per-trial heap
//! spacers or multi-buffer rings.

use std::collections::BTreeMap;
use std::hint::black_box;

use rdst::{RadixKey as RdstRadixKey, RadixSort as RdstRadixSort};
use voracious_radix_sort::{RadixSort as VoraciousRadixSort, Radixable as VoraciousRadixable};
use zipora::algorithms::radix_sort::{KeyValueRadixSort, RadixSort, RadixSortConfig};

use crate::common::*;

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
struct RadixKvPair {
    key: u64,
    val: u64,
}

impl PartialEq for RadixKvPair {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl Eq for RadixKvPair {}

impl PartialOrd for RadixKvPair {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RadixKvPair {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.key.cmp(&other.key)
    }
}

impl VoraciousRadixable<u64> for RadixKvPair {
    type Key = u64;
    #[inline]
    fn key(&self) -> Self::Key {
        self.key
    }
}

impl RdstRadixKey for RadixKvPair {
    const LEVELS: usize = 8;
    #[inline]
    fn get_level(&self, level: usize) -> u8 {
        (self.key >> (level * 8)) as u8
    }
}

pub fn bench_radix_sort_lsd(
    records: &mut Vec<BenchRecord>,
    space: &mut BTreeMap<String, f64>,
    sizes: SuiteSizes,
) {
    maybe_repin_self_within_ccd();
    trim_allocator_arena();
    let n = sizes.radix_sort_elements;
    let make_sorter = || {
        RadixSort::with_config(RadixSortConfig {
            use_parallel: false,
            use_counting_sort_threshold: 0,
            ..RadixSortConfig::default()
        })
    };
    let mut sort_params = BenchParams::heavy(n as u64, n as u64, 3);
    sort_params.warmup_calls = 1;
    sort_params.bursts_per_trial = 1;

    record_space_metric(space, "radix_sort/zipora_u32_bytes_per_elem", 8.0);
    record_space_metric(space, "radix_sort/voracious_u32_bytes_per_elem", 8.0);
    record_space_metric(space, "radix_sort/rdst_u32_bytes_per_elem", 8.0);
    record_space_metric(space, "radix_sort/std_u32_bytes_per_elem", 4.0);
    record_space_metric(space, "radix_sort/zipora_u64_bytes_per_elem", 16.0);
    record_space_metric(space, "radix_sort/voracious_u64_bytes_per_elem", 16.0);
    record_space_metric(space, "radix_sort/rdst_u64_bytes_per_elem", 16.0);
    record_space_metric(space, "radix_sort/std_u64_bytes_per_elem", 8.0);

    // 1. u32 sorting (10^7 elements, 1 work buffer per side)
    if should_run_any(&[
        "radix_sort/lsd_u32",
        "radix_sort/voracious/u32",
        "radix_sort/rdst/u32",
        "radix_sort/std_unstable/u32",
    ]) {
        trim_allocator_arena();
        let template_u32 = gen_radix_u32_template(n);
        let mut work_u32 = vec![0u32; n];

        if should_run_bench("radix_sort/lsd_u32") {
            trim_allocator_arena();
            let mut sorter = make_sorter();
            measure_bench_with_setup(
                records,
                "radix_sort",
                "radix_sort/lsd_u32",
                "ns/elem",
                sort_params,
                &mut work_u32,
                |buf| {
                    buf.copy_from_slice(&template_u32);
                },
                |buf| {
                    sorter.sort_u32(black_box(buf.as_mut_slice())).unwrap();
                    black_box(buf[0]);
                },
            );
            let _ = &sorter;
            trim_allocator_arena();
        }

        if should_run_bench("radix_sort/voracious/u32") {
            trim_allocator_arena();
            measure_bench_with_setup(
                records,
                "radix_sort",
                "radix_sort/voracious/u32",
                "ns/elem",
                sort_params,
                &mut work_u32,
                |buf| {
                    buf.copy_from_slice(&template_u32);
                },
                |buf| {
                    black_box(buf.as_mut_slice()).voracious_sort();
                    black_box(buf[0]);
                },
            );
            trim_allocator_arena();
        }

        if should_run_bench("radix_sort/rdst/u32") {
            trim_allocator_arena();
            measure_bench_with_setup(
                records,
                "radix_sort",
                "radix_sort/rdst/u32",
                "ns/elem",
                sort_params,
                &mut work_u32,
                |buf| {
                    buf.copy_from_slice(&template_u32);
                },
                |buf| {
                    black_box(buf.as_mut_slice())
                        .radix_sort_builder()
                        .with_single_threaded_tuner()
                        .sort();
                    black_box(buf[0]);
                },
            );
            trim_allocator_arena();
        }

        if should_run_bench("radix_sort/std_unstable/u32") {
            trim_allocator_arena();
            measure_bench_with_setup(
                records,
                "radix_sort",
                "radix_sort/std_unstable/u32",
                "ns/elem",
                sort_params,
                &mut work_u32,
                |buf| {
                    buf.copy_from_slice(&template_u32);
                },
                |buf| {
                    black_box(buf.as_mut_slice()).sort_unstable();
                    black_box(buf[0]);
                },
            );
            trim_allocator_arena();
        }
        drop(work_u32);
        drop(template_u32);
        trim_allocator_arena();
    }

    // 2. u64 sorting (10^7 elements, 1 work buffer per side)
    if should_run_any(&[
        "radix_sort/lsd_u64",
        "radix_sort/voracious/u64",
        "radix_sort/rdst/u64",
        "radix_sort/std_unstable/u64",
    ]) {
        trim_allocator_arena();
        let template_u64 = gen_radix_u64_template(n);
        let mut work_u64 = vec![0u64; n];

        if should_run_bench("radix_sort/lsd_u64") {
            trim_allocator_arena();
            let mut sorter = make_sorter();
            measure_bench_with_setup(
                records,
                "radix_sort",
                "radix_sort/lsd_u64",
                "ns/elem",
                sort_params,
                &mut work_u64,
                |buf| {
                    buf.copy_from_slice(&template_u64);
                },
                |buf| {
                    sorter.sort_u64(black_box(buf.as_mut_slice())).unwrap();
                    black_box(buf[0]);
                },
            );
            let _ = &sorter;
            trim_allocator_arena();
        }

        if should_run_bench("radix_sort/voracious/u64") {
            trim_allocator_arena();
            measure_bench_with_setup(
                records,
                "radix_sort",
                "radix_sort/voracious/u64",
                "ns/elem",
                sort_params,
                &mut work_u64,
                |buf| {
                    buf.copy_from_slice(&template_u64);
                },
                |buf| {
                    black_box(buf.as_mut_slice()).voracious_sort();
                    black_box(buf[0]);
                },
            );
            trim_allocator_arena();
        }

        if should_run_bench("radix_sort/rdst/u64") {
            trim_allocator_arena();
            measure_bench_with_setup(
                records,
                "radix_sort",
                "radix_sort/rdst/u64",
                "ns/elem",
                sort_params,
                &mut work_u64,
                |buf| {
                    buf.copy_from_slice(&template_u64);
                },
                |buf| {
                    black_box(buf.as_mut_slice())
                        .radix_sort_builder()
                        .with_single_threaded_tuner()
                        .sort();
                    black_box(buf[0]);
                },
            );
            trim_allocator_arena();
        }

        if should_run_bench("radix_sort/std_unstable/u64") {
            trim_allocator_arena();
            measure_bench_with_setup(
                records,
                "radix_sort",
                "radix_sort/std_unstable/u64",
                "ns/elem",
                sort_params,
                &mut work_u64,
                |buf| {
                    buf.copy_from_slice(&template_u64);
                },
                |buf| {
                    black_box(buf.as_mut_slice()).sort_unstable();
                    black_box(buf[0]);
                },
            );
            trim_allocator_arena();
        }
        drop(work_u64);
        drop(template_u64);
        trim_allocator_arena();
    }
}

pub fn bench_radix_sort_kv(
    records: &mut Vec<BenchRecord>,
    space: &mut BTreeMap<String, f64>,
    primary_core: usize,
    sizes: SuiteSizes,
) {
    maybe_repin_self_within_ccd();
    trim_allocator_arena();
    let n = sizes.radix_sort_elements;
    record_space_metric(space, "radix_sort/zipora_kv_bytes_per_elem", 56.0);
    record_space_metric(space, "radix_sort/voracious_kv_bytes_per_elem", 32.0);
    record_space_metric(space, "radix_sort/rdst_kv_bytes_per_elem", 32.0);
    record_space_metric(space, "radix_sort/std_kv_bytes_per_elem", 16.0);

    let _ = primary_core;
    let rayon_core_for = |p_core: usize| -> usize {
        let ccd_base = (p_core / 8) * 8;
        if p_core != ccd_base + 2 {
            ccd_base + 2
        } else {
            ccd_base + 3
        }
    };
    let init_rayon_core = rayon_core_for(current_self_pinned_core());
    set_allow_self_repin(false);
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .start_handler(move |_| {
            pin_current_thread_to_core(init_rayon_core);
        })
        .build_global();
    rayon::broadcast(|_| {});

    let template_kv = gen_radix_kv_template(n);
    let mut work_kv = vec![(0u64, 0u64); n];

    let kv_sorter = KeyValueRadixSort::<u64, u64>::new();
    let mut kv_params = BenchParams::heavy(n as u64, n as u64, 3);
    kv_params.warmup_calls = 1;
    kv_params.bursts_per_trial = 1;

    if should_run_bench("radix_sort/kv_u64_u64") {
        measure_bench_with_setup(
            records,
            "radix_sort",
            "radix_sort/kv_u64_u64",
            "ns/elem",
            kv_params,
            &mut work_kv,
            |buf| {
                buf.copy_from_slice(&template_kv);
            },
            |buf| {
                kv_sorter
                    .sort_by_key(black_box(buf.as_mut_slice()))
                    .unwrap();
                black_box(buf[0].0);
            },
        );
    }

    if should_run_bench("radix_sort/voracious/kv_u64_u64")
        || should_run_bench("radix_sort/rdst/kv_u64_u64")
        || should_run_bench("radix_sort/std_unstable/kv_u64_u64")
    {
        let template_pairs: Vec<RadixKvPair> = template_kv
            .iter()
            .map(|&(key, val)| RadixKvPair { key, val })
            .collect();
        let mut work_pairs = vec![RadixKvPair::default(); n];

        if should_run_bench("radix_sort/voracious/kv_u64_u64") {
            measure_bench_with_setup(
                records,
                "radix_sort",
                "radix_sort/voracious/kv_u64_u64",
                "ns/elem",
                kv_params,
                &mut work_pairs,
                |buf| {
                    buf.copy_from_slice(&template_pairs);
                },
                |buf| {
                    black_box(buf.as_mut_slice()).voracious_sort();
                    black_box(buf[0].key);
                },
            );
        }

        if should_run_bench("radix_sort/rdst/kv_u64_u64") {
            measure_bench_with_setup(
                records,
                "radix_sort",
                "radix_sort/rdst/kv_u64_u64",
                "ns/elem",
                kv_params,
                &mut work_pairs,
                |buf| {
                    buf.copy_from_slice(&template_pairs);
                },
                |buf| {
                    black_box(buf.as_mut_slice())
                        .radix_sort_builder()
                        .with_single_threaded_tuner()
                        .sort();
                    black_box(buf[0].key);
                },
            );
        }

        if should_run_bench("radix_sort/std_unstable/kv_u64_u64") {
            measure_bench_with_setup(
                records,
                "radix_sort",
                "radix_sort/std_unstable/kv_u64_u64",
                "ns/elem",
                kv_params,
                &mut work_pairs,
                |buf| {
                    buf.copy_from_slice(&template_pairs);
                },
                |buf| {
                    black_box(buf.as_mut_slice()).sort_unstable_by_key(|p| p.key);
                    black_box(buf[0].key);
                },
            );
        }
    }

    set_allow_self_repin(true);
}
