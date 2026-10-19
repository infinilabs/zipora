//! Family 4: Hash Maps (1,000,000 `u64` keys & 1,000,000 16-byte string keys, zero replay).
//!
//! Zipora: `ZiporaHashMap`, `GoldHashMap`
//! Competitors: `hashbrown::HashMap` + `ahash`, `std::collections::HashMap`
//!
//! Fairness invariant (`S42-R1`):
//! - Every implementation uses **1 map instance** (`ZiporaHashMap`, `GoldHashMap`,
//!   `hashbrown+ahash`, and `std::collections::HashMap`), removing the 2-map rings on
//!   Zipora/Gold and the 4-map rings on `hashbrown`/`std`.
//! - Every implementation is pre-allocated for `1,000,000` entries (~2M hash slots) so
//!   no map rehashes during `insert` or `get`.
//! - Every implementation executes the **exact same `get_batch = 4,000` keys per burst**,
//!   the **exact same `prefill = n - 12 * get_batch` insert window**, and the **exact same
//!   `BijectivePerm` lookup stream** via shared generic `run_insert_bench` / `run_get_bench`
//!   drivers with zero per-side heap spacers.

use std::collections::hash_map::DefaultHasher as StdSipHasher13;
use std::collections::{BTreeMap, HashMap as StdHashMap};
use std::hash::BuildHasherDefault;
use std::hint::black_box;

use hashbrown::HashMap as HashbrownMap;
use zipora::hash_map::{
    GoldHashMap, GoldHashMapConfig, HashStorageStrategy, ZiporaHashMap, ZiporaHashMapConfig,
};

use crate::common::*;

type DeterministicStdHashMap<K, V> = StdHashMap<K, V, BuildHasherDefault<StdSipHasher13>>;

#[allow(clippy::too_many_arguments)]
fn run_insert_bench<K, M, FMake, FKey, FIns, FLen, FClear>(
    records: &mut Vec<BenchRecord>,
    name: &str,
    n: usize,
    prefill: usize,
    get_batch: usize,
    insert_params: BenchParams,
    make_map: FMake,
    make_key: FKey,
    insert_fn: FIns,
    len_fn: FLen,
    mut clear_fn: FClear,
) where
    FMake: FnOnce() -> M,
    FKey: Fn(usize) -> K,
    FIns: Fn(&mut M, K, u64),
    FLen: Fn(&M) -> usize,
    FClear: FnMut(&mut M),
{
    if !should_run_bench(name) {
        return;
    }
    trim_allocator_arena();
    maybe_repin_self_within_ccd();
    let mut map = make_map();
    for i in 0..n {
        insert_fn(&mut map, make_key(i), i as u64);
    }
    clear_fn(&mut map);
    for i in 0..prefill {
        insert_fn(&mut map, make_key(i), i as u64);
    }
    measure_bench_with_setup(
        records,
        "hash_maps",
        name,
        "ns/op",
        insert_params,
        &mut map,
        |m| {
            if len_fn(m) >= n {
                clear_fn(m);
                for i in 0..prefill {
                    insert_fn(m, make_key(i), i as u64);
                }
            }
        },
        |m| {
            let b_idx = current_burst_index();
            let start = n + b_idx * get_batch;
            for id in start..start + get_batch {
                let k = make_key(id);
                insert_fn(m, black_box(k), id as u64);
            }
            black_box(len_fn(m));
        },
    );
    drop(map);
    trim_allocator_arena();
}

#[allow(clippy::too_many_arguments)]
fn run_get_bench<K, M, FMake, FKey, FIns, FGet>(
    records: &mut Vec<BenchRecord>,
    name: &str,
    n: usize,
    get_batch: usize,
    perm: &BijectivePerm,
    get_params: BenchParams,
    make_map: FMake,
    make_key: FKey,
    mut insert_fn: FIns,
    get_fn: FGet,
) where
    FMake: FnOnce() -> M,
    FKey: Fn(usize) -> K,
    FIns: FnMut(&mut M, K, u64),
    FGet: Fn(&M, &K) -> Option<u64>,
{
    if !should_run_bench(name) {
        return;
    }
    trim_allocator_arena();
    maybe_repin_self_within_ccd();
    let mut map = make_map();
    for id in 0..n {
        insert_fn(&mut map, make_key(id), id as u64);
    }
    let mut warm_sum = 0u64;
    for w in 0..8usize {
        perm.for_each_slice(w * get_batch, get_batch, |id| {
            let k = make_key(id);
            if let Some(v) = get_fn(&map, black_box(&k)) {
                warm_sum = warm_sum.wrapping_add(v);
            }
        });
    }
    black_box(warm_sum);
    measure_bench(records, "hash_maps", name, "ns/op", get_params, || {
        let b_idx = current_burst_index();
        let start = b_idx * get_batch;
        let mut sum = 0u64;
        perm.for_each_slice(start, get_batch, |id| {
            let k = make_key(id);
            if let Some(v) = get_fn(&map, black_box(&k)) {
                sum = sum.wrapping_add(v);
            }
        });
        black_box(sum);
    });
    drop(map);
    trim_allocator_arena();
}

pub fn bench_hash_maps(
    records: &mut Vec<BenchRecord>,
    space: &mut BTreeMap<String, f64>,
    sizes: SuiteSizes,
) {
    let n = sizes.hash_map_keys;
    let get_batch = (n / 250).max(1_000).min(n.max(1));
    let cap = (n * 2).next_power_of_two();
    let u64_perm = BijectivePerm::new(n, SEED_BASE ^ 0x0601);
    let str16_perm = BijectivePerm::new(n, SEED_BASE ^ 0x0602);
    trim_allocator_arena();

    let make_zconfig = || ZiporaHashMapConfig {
        initial_capacity: cap,
        storage_strategy: HashStorageStrategy::Standard {
            initial_capacity: cap,
            growth_factor: 2.0,
        },
        ..ZiporaHashMapConfig::default()
    };
    let make_hasher = || {
        ahash::RandomState::with_seeds(
            0x517c_c1b7_2722_0a95,
            0x9e37_79b9_7f4a_7c15,
            0xbf58_476d_1ce4_e5b9,
            0x94d0_49bb_1331_11eb,
        )
    };
    // `next_prime((n * 10) / 7)` = `1,982,627` buckets at `n = 1,000,000` (`max_load = 1,387,838`),
    // matching the ~2M-bucket table scale of `ZiporaHashMap`/`hashbrown`/`std` without rehashing.
    let gconfig = GoldHashMapConfig {
        initial_capacity: (n * 10 / 7).max(16),
        ..GoldHashMapConfig::default()
    };

    let mut insert_params = BenchParams::heavy(n as u64, get_batch as u64, 10).with_threshold(22.0);
    insert_params.warmup_calls = 8;
    insert_params.bursts_per_trial = 12;

    let mut get_params = BenchParams::heavy(n as u64, get_batch as u64, 14).with_threshold(22.0);
    get_params.warmup_calls = 12;
    get_params.bursts_per_trial = 18;

    let prefill = n.saturating_sub(insert_params.bursts_per_trial * get_batch);

    // 1. ZiporaHashMap u64 (1 instance)
    if should_run_any(&[
        "hash_maps/zipora_hash_map/insert_u64",
        "hash_maps/zipora_hash_map/get_u64",
    ]) {
        record_space_metric(
            space,
            "hash_maps/zipora_u64_bytes_per_entry",
            (cap * 40) as f64 / n.max(1) as f64,
        );
        run_insert_bench(
            records,
            "hash_maps/zipora_hash_map/insert_u64",
            n,
            prefill,
            get_batch,
            insert_params,
            || {
                ZiporaHashMap::<u64, u64>::with_config_and_hasher(make_zconfig(), make_hasher())
                    .unwrap()
            },
            make_u64_key,
            |m, k, v| {
                m.insert(k, v).unwrap();
            },
            |m| m.len(),
            |m| m.clear(),
        );
        run_get_bench(
            records,
            "hash_maps/zipora_hash_map/get_u64",
            n,
            get_batch,
            &u64_perm,
            get_params,
            || {
                ZiporaHashMap::<u64, u64>::with_config_and_hasher(make_zconfig(), make_hasher())
                    .unwrap()
            },
            make_u64_key,
            |m, k, v| {
                m.insert(k, v).unwrap();
            },
            |m, k| m.get(k).copied(),
        );
    }

    // 2. ZiporaHashMap [u8; 16] (1 instance)
    if should_run_any(&[
        "hash_maps/zipora_hash_map/insert_str16",
        "hash_maps/zipora_hash_map/get_str16",
    ]) {
        record_space_metric(
            space,
            "hash_maps/zipora_str16_bytes_per_entry",
            (cap * 48) as f64 / n.max(1) as f64,
        );
        run_insert_bench(
            records,
            "hash_maps/zipora_hash_map/insert_str16",
            n,
            prefill,
            get_batch,
            insert_params,
            || {
                ZiporaHashMap::<[u8; 16], u64>::with_config_and_hasher(
                    make_zconfig(),
                    make_hasher(),
                )
                .unwrap()
            },
            make_str16_key,
            |m, k, v| {
                m.insert(k, v).unwrap();
            },
            |m| m.len(),
            |m| m.clear(),
        );
        run_get_bench(
            records,
            "hash_maps/zipora_hash_map/get_str16",
            n,
            get_batch,
            &str16_perm,
            get_params,
            || {
                ZiporaHashMap::<[u8; 16], u64>::with_config_and_hasher(
                    make_zconfig(),
                    make_hasher(),
                )
                .unwrap()
            },
            make_str16_key,
            |m, k, v| {
                m.insert(k, v).unwrap();
            },
            |m, k| m.get(k).copied(),
        );
    }

    // 3. GoldHashMap u64 (1 instance)
    if should_run_any(&[
        "hash_maps/gold_hash_map/insert_u64",
        "hash_maps/gold_hash_map/get_u64",
    ]) {
        let g_ref: GoldHashMap<u64, u64> = GoldHashMap::with_config(gconfig.clone());
        record_space_metric(
            space,
            "hash_maps/gold_u64_bytes_per_entry",
            (g_ref.capacity() * 4 + n * 24) as f64 / n.max(1) as f64,
        );
        drop(g_ref);
        run_insert_bench(
            records,
            "hash_maps/gold_hash_map/insert_u64",
            n,
            prefill,
            get_batch,
            insert_params,
            || GoldHashMap::<u64, u64>::with_config(gconfig.clone()),
            make_u64_key,
            |m, k, v| {
                let _ = m.insert(k, v);
            },
            |m| m.len(),
            |m| m.clear(),
        );
        run_get_bench(
            records,
            "hash_maps/gold_hash_map/get_u64",
            n,
            get_batch,
            &u64_perm,
            get_params,
            || GoldHashMap::<u64, u64>::with_config(gconfig.clone()),
            make_u64_key,
            |m, k, v| {
                let _ = m.insert(k, v);
            },
            |m, k| m.get(k).copied(),
        );
    }

    // 4. GoldHashMap [u8; 16] (1 instance)
    if should_run_any(&[
        "hash_maps/gold_hash_map/insert_str16",
        "hash_maps/gold_hash_map/get_str16",
    ]) {
        let g_ref: GoldHashMap<[u8; 16], u64> = GoldHashMap::with_config(gconfig.clone());
        record_space_metric(
            space,
            "hash_maps/gold_str16_bytes_per_entry",
            (g_ref.capacity() * 4 + n * 32) as f64 / n.max(1) as f64,
        );
        drop(g_ref);
        run_insert_bench(
            records,
            "hash_maps/gold_hash_map/insert_str16",
            n,
            prefill,
            get_batch,
            insert_params,
            || GoldHashMap::<[u8; 16], u64>::with_config(gconfig.clone()),
            make_str16_key,
            |m, k, v| {
                let _ = m.insert(k, v);
            },
            |m| m.len(),
            |m| m.clear(),
        );
        run_get_bench(
            records,
            "hash_maps/gold_hash_map/get_str16",
            n,
            get_batch,
            &str16_perm,
            get_params,
            || GoldHashMap::<[u8; 16], u64>::with_config(gconfig),
            make_str16_key,
            |m, k, v| {
                let _ = m.insert(k, v);
            },
            |m, k| m.get(k).copied(),
        );
    }

    // 5. Competitor: hashbrown::HashMap<u64, u64, ahash::RandomState> (1 instance)
    if should_run_any(&[
        "hash_maps/hashbrown_ahash/insert_u64",
        "hash_maps/hashbrown_ahash/get_u64",
    ]) {
        let hb_ref: HashbrownMap<u64, u64, ahash::RandomState> =
            HashbrownMap::with_capacity_and_hasher(n, make_hasher());
        record_space_metric(
            space,
            "hash_maps/hashbrown_u64_bytes_per_entry",
            hb_ref.allocation_size() as f64 / n.max(1) as f64,
        );
        drop(hb_ref);
        run_insert_bench(
            records,
            "hash_maps/hashbrown_ahash/insert_u64",
            n,
            prefill,
            get_batch,
            insert_params,
            || {
                HashbrownMap::<u64, u64, ahash::RandomState>::with_capacity_and_hasher(
                    n,
                    make_hasher(),
                )
            },
            make_u64_key,
            |m, k, v| {
                m.insert(k, v);
            },
            |m| m.len(),
            |m| m.clear(),
        );
        run_get_bench(
            records,
            "hash_maps/hashbrown_ahash/get_u64",
            n,
            get_batch,
            &u64_perm,
            get_params,
            || {
                HashbrownMap::<u64, u64, ahash::RandomState>::with_capacity_and_hasher(
                    n,
                    make_hasher(),
                )
            },
            make_u64_key,
            |m, k, v| {
                m.insert(k, v);
            },
            |m, k| m.get(k).copied(),
        );
    }

    // 6. Competitor: hashbrown::HashMap<[u8; 16], u64, ahash::RandomState> (1 instance)
    if should_run_any(&[
        "hash_maps/hashbrown_ahash/insert_str16",
        "hash_maps/hashbrown_ahash/get_str16",
    ]) {
        let hb_ref: HashbrownMap<[u8; 16], u64, ahash::RandomState> =
            HashbrownMap::with_capacity_and_hasher(n, make_hasher());
        record_space_metric(
            space,
            "hash_maps/hashbrown_str16_bytes_per_entry",
            hb_ref.allocation_size() as f64 / n.max(1) as f64,
        );
        drop(hb_ref);
        run_insert_bench(
            records,
            "hash_maps/hashbrown_ahash/insert_str16",
            n,
            prefill,
            get_batch,
            insert_params,
            || {
                HashbrownMap::<[u8; 16], u64, ahash::RandomState>::with_capacity_and_hasher(
                    n,
                    make_hasher(),
                )
            },
            make_str16_key,
            |m, k, v| {
                m.insert(k, v);
            },
            |m| m.len(),
            |m| m.clear(),
        );
        run_get_bench(
            records,
            "hash_maps/hashbrown_ahash/get_str16",
            n,
            get_batch,
            &str16_perm,
            get_params,
            || {
                HashbrownMap::<[u8; 16], u64, ahash::RandomState>::with_capacity_and_hasher(
                    n,
                    make_hasher(),
                )
            },
            make_str16_key,
            |m, k, v| {
                m.insert(k, v);
            },
            |m, k| m.get(k).copied(),
        );
    }

    // 7. Competitor: std::collections::HashMap<u64, u64> (1 instance)
    if should_run_any(&["hash_maps/std/insert_u64", "hash_maps/std/get_u64"]) {
        let hb_ref: HashbrownMap<u64, u64, ahash::RandomState> =
            HashbrownMap::with_capacity_and_hasher(n, make_hasher());
        record_space_metric(
            space,
            "hash_maps/std_u64_bytes_per_entry",
            hb_ref.allocation_size() as f64 / n.max(1) as f64,
        );
        drop(hb_ref);
        run_insert_bench(
            records,
            "hash_maps/std/insert_u64",
            n,
            prefill,
            get_batch,
            insert_params,
            || {
                DeterministicStdHashMap::<u64, u64>::with_capacity_and_hasher(
                    n,
                    BuildHasherDefault::default(),
                )
            },
            make_u64_key,
            |m, k, v| {
                m.insert(k, v);
            },
            |m| m.len(),
            |m| m.clear(),
        );
        run_get_bench(
            records,
            "hash_maps/std/get_u64",
            n,
            get_batch,
            &u64_perm,
            get_params,
            || {
                DeterministicStdHashMap::<u64, u64>::with_capacity_and_hasher(
                    n,
                    BuildHasherDefault::default(),
                )
            },
            make_u64_key,
            |m, k, v| {
                m.insert(k, v);
            },
            |m, k| m.get(k).copied(),
        );
    }

    // 8. Competitor: std::collections::HashMap<[u8; 16], u64> (1 instance)
    if should_run_any(&["hash_maps/std/insert_str16", "hash_maps/std/get_str16"]) {
        let hb_ref: HashbrownMap<[u8; 16], u64, ahash::RandomState> =
            HashbrownMap::with_capacity_and_hasher(n, make_hasher());
        record_space_metric(
            space,
            "hash_maps/std_str16_bytes_per_entry",
            hb_ref.allocation_size() as f64 / n.max(1) as f64,
        );
        drop(hb_ref);
        run_insert_bench(
            records,
            "hash_maps/std/insert_str16",
            n,
            prefill,
            get_batch,
            insert_params,
            || {
                DeterministicStdHashMap::<[u8; 16], u64>::with_capacity_and_hasher(
                    n,
                    BuildHasherDefault::default(),
                )
            },
            make_str16_key,
            |m, k, v| {
                m.insert(k, v);
            },
            |m| m.len(),
            |m| m.clear(),
        );
        run_get_bench(
            records,
            "hash_maps/std/get_str16",
            n,
            get_batch,
            &str16_perm,
            get_params,
            || {
                DeterministicStdHashMap::<[u8; 16], u64>::with_capacity_and_hasher(
                    n,
                    BuildHasherDefault::default(),
                )
            },
            make_str16_key,
            |m, k, v| {
                m.insert(k, v);
            },
            |m, k| m.get(k).copied(),
        );
    }
}
