//! Family 3: Tries (1,000,000 keys across 3 distributions, lookup on all 1,000,000 keys).
//!
//! Zipora: `DoubleArrayTrie`, `CsppTrie`, `ZiporaTrie`
//! Competitors: `fst::Set`, `qp_trie::Trie`, `radix_trie::Trie`
//!
//! Fairness invariant (`S42-R1`): every implementation is benchmarked on **1 trie instance**,
//! with the **exact same `lookup_batch = n / 250` (`4,000` keys)**, the **exact same
//! `BijectivePerm` query stream** via `run_trie_lookup_bench`, and **zero per-side heap
//! spacers or multi-instance rings**.

use std::collections::BTreeMap;
use std::hint::black_box;

use fst::Set as FstSet;
use qp_trie::Trie as QpTrie;
use radix_trie::{Trie as RadixTrie, TrieCommon as _};
use zipora::fsa::cspp_trie::CsppTrie;
use zipora::fsa::{DoubleArrayTrie, TrieStrategy, ZiporaTrie, ZiporaTrieConfig};
use zipora::succinct::rank_select::RankSelectInterleaved256;

use crate::common::*;

fn mallinfo_used_bytes() -> usize {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        // SAFETY: `libc::mallinfo2` is thread-safe and reads glibc allocator statistics by value.
        let mi = unsafe { libc::mallinfo2() };
        mi.uordblks.saturating_add(mi.hblkhd)
    }
    #[cfg(not(all(target_os = "linux", target_env = "gnu")))]
    {
        0
    }
}

#[allow(clippy::too_many_arguments)]
fn run_trie_lookup_bench<F>(
    records: &mut Vec<BenchRecord>,
    name: &str,
    dist_idx: usize,
    lookup_batch: usize,
    warm_perm: &BijectivePerm,
    perm: &BijectivePerm,
    lookup_params: BenchParams,
    contains_fn: F,
) where
    F: Fn(&[u8]) -> bool,
{
    if !should_run_bench(name) {
        return;
    }
    maybe_repin_self_within_ccd();
    set_allow_self_repin(false);
    let mut warm = 0usize;
    for w_burst in 0..8usize {
        let w_start = w_burst * lookup_batch;
        warm_perm.for_each_slice(w_start, lookup_batch, |id| {
            if contains_trie_key_by_id(dist_idx, id, &contains_fn) {
                warm += 1;
            }
        });
    }
    black_box(warm);

    measure_bench(records, "tries", name, "ns/op", lookup_params, || {
        let start = current_burst_index() * lookup_batch;
        let mut found = 0usize;
        perm.for_each_slice(start, lookup_batch, |id| {
            if contains_trie_key_by_id(dist_idx, id, &contains_fn) {
                found += 1;
            }
        });
        debug_assert_eq!(found, lookup_batch);
        black_box(found);
    });
    set_allow_self_repin(true);
}

pub fn bench_tries(
    records: &mut Vec<BenchRecord>,
    space: &mut BTreeMap<String, f64>,
    sizes: SuiteSizes,
) {
    let n = sizes.trie_keys;
    let lookup_batch = (n / 250).max(1_000).min(n.max(1));

    for dist_idx in 0..3usize {
        let dist_label = match dist_idx {
            0 => "url_paths",
            1 => "random_16b",
            _ => "shared_prefix",
        };
        let run_dat = should_run_any(&[
            &format!("tries/dat/{dist_label}/build_sorted"),
            &format!("tries/dat/{dist_label}/lookup"),
        ]);
        let run_cspp = should_run_any(&[
            &format!("tries/cspp/{dist_label}/build"),
            &format!("tries/cspp/{dist_label}/lookup"),
        ]);
        let run_fst = should_run_any(&[
            &format!("tries/fst/{dist_label}/build_sorted"),
            &format!("tries/fst/{dist_label}/lookup"),
        ]);
        let run_qp = should_run_any(&[
            &format!("tries/qp_trie/{dist_label}/build"),
            &format!("tries/qp_trie/{dist_label}/lookup"),
        ]);
        let run_radix = should_run_any(&[
            &format!("tries/radix_trie/{dist_label}/build"),
            &format!("tries/radix_trie/{dist_label}/lookup"),
        ]);
        if !run_dat && !run_cspp && !run_fst && !run_qp && !run_radix {
            continue;
        }
        maybe_repin_self_within_ccd();
        let perm = BijectivePerm::new(n, SEED_BASE ^ 0x0500 ^ (dist_idx as u64));
        let warm_perm = BijectivePerm::new(n, SEED_BASE ^ 0x0550 ^ (dist_idx as u64));
        let num_keys = n as u64;

        let mut lookup_params = BenchParams::heavy(num_keys, lookup_batch as u64, 14);
        lookup_params.warmup_calls = 10;
        lookup_params.bursts_per_trial = 18;

        let mut build_params = BenchParams::heavy(num_keys, num_keys, 3);
        build_params.warmup_calls = 1;
        build_params.bursts_per_trial = 1;

        // 1. Zipora DoubleArrayTrie (1 instance)
        if run_dat {
            maybe_repin_self_within_ccd();
            let (dist_name, packed) = gen_trie_distribution(dist_idx, n);
            let key_slices = packed.slices();
            let mut dat_slot: Option<DoubleArrayTrie> = None;
            measure_bench_with_setup(
                records,
                "tries",
                &format!("tries/dat/{dist_name}/build_sorted"),
                "ns/key",
                build_params,
                &mut dat_slot,
                |slot| {
                    let _ = slot.take();
                },
                |slot| {
                    let d = DoubleArrayTrie::build_from_sorted(black_box(&key_slices)).unwrap();
                    black_box(d.len());
                    *slot = Some(d);
                },
            );
            let dat = dat_slot
                .take()
                .unwrap_or_else(|| DoubleArrayTrie::build_from_sorted(&key_slices).unwrap());
            drop(key_slices);
            drop(packed);

            let bytes_per_state = dat.mem_size() as f64 / dat.total_states().max(1) as f64;
            let bytes_per_key = dat.mem_size() as f64 / n.max(1) as f64;
            record_space_metric(
                space,
                &format!("tries/dat_{dist_name}_bytes_per_state"),
                bytes_per_state,
            );
            record_space_metric(
                space,
                &format!("tries/dat_{dist_name}_bytes_per_key"),
                bytes_per_key,
            );

            run_trie_lookup_bench(
                records,
                &format!("tries/dat/{dist_name}/lookup"),
                dist_idx,
                lookup_batch,
                &warm_perm,
                &perm,
                lookup_params,
                |k| dat.contains(k),
            );
            drop(dat);
        }

        // 2. Zipora CsppTrie (1 instance)
        if run_cspp {
            let dist_name = dist_label;
            maybe_repin_self_within_ccd();
            let (_, packed) = gen_trie_distribution(dist_idx, n);
            let key_len = packed.key_len;
            let mut cspp_slot: Option<CsppTrie> = None;
            measure_bench_with_setup(
                records,
                "tries",
                &format!("tries/cspp/{dist_name}/build"),
                "ns/key",
                build_params,
                &mut cspp_slot,
                |slot| {
                    let _ = slot.take();
                },
                |slot| {
                    let mut c = CsppTrie::new(0);
                    for k in packed.data.chunks_exact(key_len) {
                        c.insert(black_box(k));
                    }
                    black_box(c.num_words());
                    *slot = Some(c);
                },
            );
            let cspp = cspp_slot.take().unwrap_or_else(|| {
                let mut c = CsppTrie::new(0);
                for k in packed.data.chunks_exact(key_len) {
                    c.insert(k);
                }
                c
            });
            drop(packed);

            let stat = cspp.mem_get_stat();
            let active_bytes = stat.used_size.saturating_sub(stat.frag_size);
            record_space_metric(
                space,
                &format!("tries/cspp_{dist_name}_bytes_per_key"),
                active_bytes as f64 / cspp.num_words().max(1) as f64,
            );

            run_trie_lookup_bench(
                records,
                &format!("tries/cspp/{dist_name}/lookup"),
                dist_idx,
                lookup_batch,
                &warm_perm,
                &perm,
                lookup_params,
                |k| cspp.contains(k),
            );
            drop(cspp);
        }

        // 3. Competitor: fst::Set (1 instance)
        if run_fst {
            let dist_name = dist_label;
            maybe_repin_self_within_ccd();
            let (_, packed) = gen_trie_distribution(dist_idx, n);
            let key_slices = packed.slices();
            let mut fst_slot: Option<FstSet<Vec<u8>>> = None;
            measure_bench_with_setup(
                records,
                "tries",
                &format!("tries/fst/{dist_name}/build_sorted"),
                "ns/key",
                build_params,
                &mut fst_slot,
                |slot| {
                    let _ = slot.take();
                },
                |slot| {
                    let set = FstSet::from_iter(key_slices.iter().copied()).unwrap();
                    black_box(set.len());
                    *slot = Some(set);
                },
            );
            let fst_set = fst_slot
                .take()
                .unwrap_or_else(|| FstSet::from_iter(key_slices.iter().copied()).unwrap());
            drop(key_slices);
            drop(packed);

            record_space_metric(
                space,
                &format!("tries/fst_{dist_name}_bytes_per_key"),
                fst_set.as_fst().size() as f64 / n.max(1) as f64,
            );

            run_trie_lookup_bench(
                records,
                &format!("tries/fst/{dist_name}/lookup"),
                dist_idx,
                lookup_batch,
                &warm_perm,
                &perm,
                lookup_params,
                |k| fst_set.contains(k),
            );
            drop(fst_set);
        }

        // 4. Competitor: qp_trie::Trie<&[u8], ()> (1 instance)
        if run_qp {
            let dist_name = dist_label;
            maybe_repin_self_within_ccd();
            let (_, packed) = gen_trie_distribution(dist_idx, n);
            let key_len = packed.key_len;
            let raw_key_bytes = packed.data.len();
            let mut qp_slot: Option<QpTrie<&[u8], ()>> = None;
            measure_bench_with_setup(
                records,
                "tries",
                &format!("tries/qp_trie/{dist_name}/build"),
                "ns/key",
                build_params,
                &mut qp_slot,
                |slot| {
                    let _ = slot.take();
                },
                |slot| {
                    let mut trie = QpTrie::new();
                    for k in packed.data.chunks_exact(key_len) {
                        trie.insert(black_box(k), ());
                    }
                    black_box(trie.count());
                    *slot = Some(trie);
                },
            );
            let qp_trie = qp_slot.take().unwrap_or_else(|| {
                let mut trie = QpTrie::new();
                for k in packed.data.chunks_exact(key_len) {
                    trie.insert(k, ());
                }
                trie
            });

            let before_clone = mallinfo_used_bytes();
            let qp_cloned = black_box(qp_trie.clone());
            let after_clone = mallinfo_used_bytes();
            let qp_heap_bytes = after_clone
                .saturating_sub(before_clone)
                .max(n * std::mem::size_of::<(&[u8], usize)>());
            drop(qp_cloned);
            record_space_metric(
                space,
                &format!("tries/qp_trie_{dist_name}_bytes_per_key"),
                (qp_heap_bytes + raw_key_bytes) as f64 / n.max(1) as f64,
            );

            run_trie_lookup_bench(
                records,
                &format!("tries/qp_trie/{dist_name}/lookup"),
                dist_idx,
                lookup_batch,
                &warm_perm,
                &perm,
                lookup_params,
                |k| qp_trie.contains_key(k),
            );
            drop(qp_trie);
            drop(packed);
        }

        // 5. Competitor: radix_trie::Trie<&[u8], ()> (1 instance)
        if run_radix {
            let dist_name = dist_label;
            maybe_repin_self_within_ccd();
            let (_, packed) = gen_trie_distribution(dist_idx, n);
            let key_len = packed.key_len;
            let raw_key_bytes = packed.data.len();
            let mut radix_slot: Option<RadixTrie<&[u8], ()>> = None;
            measure_bench_with_setup(
                records,
                "tries",
                &format!("tries/radix_trie/{dist_name}/build"),
                "ns/key",
                build_params,
                &mut radix_slot,
                |slot| {
                    let _ = slot.take();
                },
                |slot| {
                    let mut trie = RadixTrie::new();
                    for k in packed.data.chunks_exact(key_len) {
                        trie.insert(black_box(k), ());
                    }
                    black_box(trie.len());
                    *slot = Some(trie);
                },
            );
            let radix_trie = radix_slot.take().unwrap_or_else(|| {
                let mut trie = RadixTrie::new();
                for k in packed.data.chunks_exact(key_len) {
                    trie.insert(k, ());
                }
                trie
            });

            let before_clone = mallinfo_used_bytes();
            let radix_cloned = black_box(radix_trie.clone());
            let after_clone = mallinfo_used_bytes();
            let radix_heap_bytes = after_clone
                .saturating_sub(before_clone)
                .max(n * std::mem::size_of::<(&[u8], usize)>());
            drop(radix_cloned);
            record_space_metric(
                space,
                &format!("tries/radix_trie_{dist_name}_bytes_per_key"),
                (radix_heap_bytes + raw_key_bytes) as f64 / n.max(1) as f64,
            );

            run_trie_lookup_bench(
                records,
                &format!("tries/radix_trie/{dist_name}/lookup"),
                dist_idx,
                lookup_batch,
                &warm_perm,
                &perm,
                lookup_params,
                |k| radix_trie.get(k).is_some(),
            );
            drop(radix_trie);
            drop(packed);
        }
    }

    // 6. ZiporaTrie (DoubleArray strategy, 1 instance)
    if !should_run_any(&["tries/zipora_trie_da/build", "tries/zipora_trie_da/lookup"]) {
        return;
    }
    maybe_repin_self_within_ccd();
    let num_zipora_keys = n as u64;
    let zperm = BijectivePerm::new(n, SEED_BASE ^ 0x0500);
    let zwarm_perm = BijectivePerm::new(n, SEED_BASE ^ 0x0550);
    let mut zlookup_params = BenchParams::heavy(num_zipora_keys, lookup_batch as u64, 14);
    zlookup_params.warmup_calls = 10;
    zlookup_params.bursts_per_trial = 18;
    let mut zbuild_params = BenchParams::heavy(num_zipora_keys, num_zipora_keys, 3);
    zbuild_params.warmup_calls = 1;
    zbuild_params.bursts_per_trial = 1;

    let config = ZiporaTrieConfig {
        trie_strategy: TrieStrategy::DoubleArray {
            initial_capacity: (n / 2) + 90_112,
            growth_factor: 2.0,
            free_list_management: true,
            auto_shrink: false,
        },
        ..ZiporaTrieConfig::default()
    };

    let (_, zipora_packed) = gen_trie_distribution(0, n);
    let zkey_len = zipora_packed.key_len;
    let mut z_pending: Option<ZiporaTrie<RankSelectInterleaved256>> = None;
    measure_bench_with_setup(
        records,
        "tries",
        "tries/zipora_trie_da/build",
        "ns/key",
        zbuild_params,
        &mut z_pending,
        |slot| {
            let _ = slot.take();
        },
        |slot| {
            let mut z: ZiporaTrie<RankSelectInterleaved256> =
                ZiporaTrie::with_config(config.clone());
            for k in zipora_packed.data.chunks_exact(zkey_len) {
                z.insert(black_box(k)).unwrap();
            }
            black_box(z.len());
            *slot = Some(z);
        },
    );
    let ztrie = z_pending.take().unwrap_or_else(|| {
        let mut z: ZiporaTrie<RankSelectInterleaved256> = ZiporaTrie::with_config(config);
        for k in zipora_packed.data.chunks_exact(zkey_len) {
            z.insert(k).unwrap();
        }
        z
    });
    drop(zipora_packed);

    run_trie_lookup_bench(
        records,
        "tries/zipora_trie_da/lookup",
        0,
        lookup_batch,
        &zwarm_perm,
        &zperm,
        zlookup_params,
        |k| ztrie.contains(k),
    );
    drop(ztrie);
}
