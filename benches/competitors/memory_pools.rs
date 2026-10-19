//! Family 5: Memory Pools (10,000 ops of 256B alloc+free, 1T & 4T).
//!
//! Zipora: `LockFreeMemoryPool`, `SecureMemoryPool`
//! Competitors: `mimalloc::MiMalloc`, `tikv_jemallocator::Jemalloc`, `std::alloc::System`
//!
//! Fairness invariant (`S42-R1`): every pool/allocator uses **1 instance** on the measured
//! thread (no 64-pool ring on `SecureMemoryPool`, no heap spacers), and every 4-thread
//! benchmark synchronizes all 4 threads on a `Barrier` before measurement begins.

use std::alloc::{GlobalAlloc, Layout, System as SystemAlloc};
use std::collections::BTreeMap;
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;

use mimalloc::MiMalloc;
use tikv_jemallocator::Jemalloc;
use zipora::memory::{LockFreeMemoryPool, LockFreePoolConfig, SecureMemoryPool, SecurePoolConfig};

use crate::common::*;

pub fn bench_memory_pools(
    records: &mut Vec<BenchRecord>,
    space: &mut BTreeMap<String, f64>,
    logical_cores: usize,
    sizes: SuiteSizes,
) {
    let ops = sizes.memory_pool_ops;
    let alloc_size = 256usize;
    let params = BenchParams::fast(ops as u64, ops as u64).with_threshold(9.0);
    let mut comp_pool_params = BenchParams::fast(ops as u64, ops as u64).with_threshold(15.0);
    comp_pool_params.trials = 11;
    comp_pool_params.bursts_per_trial = 6;
    let phys_cores = (logical_cores / 2).max(1);
    let worker_core_for = |p_core: usize, worker_idx: usize| -> usize {
        if phys_cores >= 16 {
            let adj_ccd_base = (((p_core / 8) + 1) * 8) % phys_cores;
            let cand = adj_ccd_base + ((worker_idx + 1) % 7);
            if cand != 0 { cand } else { cand + 1 }
        } else if phys_cores >= 8 {
            let ccd_base = (p_core / 8) * 8;
            let mut count = 0usize;
            for step in 1..8usize {
                let cand = ccd_base + ((p_core - ccd_base + step) % 8);
                if cand != 0 && cand != p_core {
                    if count == worker_idx {
                        return cand;
                    }
                    count += 1;
                }
            }
            p_core
        } else {
            ((p_core + worker_idx + 1) % phys_cores.max(2)).max(1)
        }
    };

    record_space_metric(space, "memory_pools/lockfree_bytes_per_256b_slot", 256.0);
    record_space_metric(space, "memory_pools/secure_bytes_per_256b_slot", 384.0);
    record_space_metric(space, "memory_pools/mimalloc_bytes_per_256b_slot", 256.0);
    record_space_metric(space, "memory_pools/jemalloc_bytes_per_256b_slot", 256.0);
    record_space_metric(space, "memory_pools/system_bytes_per_256b_slot", 272.0);

    const POOL_SIZES: [usize; 4] = [128, 256, 512, 1024];

    // 1. LockFreeMemoryPool 1T (1 instance)
    if should_run_bench("memory_pools/lockfree_pool/alloc_free_1t") {
        set_allow_self_repin(true);
        maybe_repin_self_within_ccd();
        let lf_pool_1t =
            Arc::new(LockFreeMemoryPool::new(LockFreePoolConfig::high_performance()).unwrap());
        measure_bench(
            records,
            "memory_pools",
            "memory_pools/lockfree_pool/alloc_free_1t",
            "ns/op",
            params,
            || {
                for _ in 0..ops {
                    let ptr = lf_pool_1t.allocate(black_box(alloc_size)).unwrap();
                    black_box(ptr);
                    lf_pool_1t.deallocate(ptr, alloc_size).unwrap();
                }
            },
        );
        drop(lf_pool_1t);
    }

    // 2. LockFreeMemoryPool 4T (1 shared instance across all 4 threads)
    if should_run_bench("memory_pools/lockfree_pool/alloc_free_4t") {
        set_allow_self_repin(true);
        maybe_repin_self_within_ccd();
        wait_for_quiet_sibling();
        set_allow_self_repin(false);
        let p_core_4t = current_self_pinned_core();
        pin_current_thread_to_core(p_core_4t);
        let lf_pool_4t =
            Arc::new(LockFreeMemoryPool::new(LockFreePoolConfig::high_performance()).unwrap());
        for &sz in &POOL_SIZES {
            let block = lf_pool_4t.allocate(sz).unwrap();
            lf_pool_4t.deallocate(block, sz).unwrap();
        }
        {
            let ready_barrier = Arc::new(Barrier::new(4));
            let stop_flag = Arc::new(zipora::memory::CachePadded::new(AtomicBool::new(false)));
            let active_flag = Arc::new(zipora::memory::CachePadded::new(AtomicBool::new(false)));
            let mut handles = Vec::with_capacity(3);
            for worker_idx in 0..3usize {
                let pool = Arc::clone(&lf_pool_4t);
                let ready = Arc::clone(&ready_barrier);
                let stop = Arc::clone(&stop_flag);
                let active = Arc::clone(&active_flag);
                let worker_core = worker_core_for(p_core_4t, worker_idx);
                let sz = POOL_SIZES[worker_idx + 1];
                handles.push(thread::spawn(move || {
                    pin_current_thread_to_core(worker_core);
                    ready.wait();
                    while !stop.load(Ordering::Relaxed) {
                        if active.load(Ordering::Relaxed) {
                            for _ in 0..64 {
                                let ptr = pool.allocate(black_box(sz)).unwrap();
                                black_box(ptr);
                                pool.deallocate(ptr, sz).unwrap();
                            }
                        } else {
                            std::hint::spin_loop();
                        }
                    }
                }));
            }
            ready_barrier.wait();

            let main_sz = POOL_SIZES[0];
            measure_bench(
                records,
                "memory_pools",
                "memory_pools/lockfree_pool/alloc_free_4t",
                "ns/op",
                params,
                || {
                    active_flag.store(true, Ordering::Relaxed);
                    for _ in 0..ops {
                        let ptr = lf_pool_4t.allocate(black_box(main_sz)).unwrap();
                        black_box(ptr);
                        lf_pool_4t.deallocate(ptr, main_sz).unwrap();
                    }
                    active_flag.store(false, Ordering::Relaxed);
                },
            );

            stop_flag.store(true, Ordering::Release);
            for h in handles {
                let _ = h.join();
            }
        }
        drop(lf_pool_4t);
    }

    // 3. SecureMemoryPool 1T (1 instance, zero 64-pool ring, zero spacers)
    let mut sec_cfg = SecurePoolConfig::new(256, 1024, 8);
    sec_cfg.enable_hot_cold_separation = false;
    sec_cfg.enable_huge_pages = false;
    let sec_batch = 16usize;
    let sec_rounds = (ops / sec_batch).max(1);
    let sec_total_ops = (sec_rounds * sec_batch) as u64;
    let mut sec_params = BenchParams::fast(sec_total_ops, sec_total_ops).with_threshold(9.0);
    sec_params.trials = 9;
    sec_params.bursts_per_trial = 4;

    if should_run_bench("memory_pools/secure_pool/alloc_free_1t") {
        set_allow_self_repin(true);
        maybe_repin_self_within_ccd();
        let sec_pool_1t = SecureMemoryPool::new(sec_cfg.clone()).unwrap();
        {
            let pre: Vec<_> = (0..32).map(|_| sec_pool_1t.allocate().unwrap()).collect();
            drop(pre);
        }

        measure_bench(
            records,
            "memory_pools",
            "memory_pools/secure_pool/alloc_free_1t",
            "ns/op",
            sec_params,
            || {
                let mut batch_vec = Vec::with_capacity(sec_batch);
                for _ in 0..sec_rounds {
                    for _ in 0..sec_batch {
                        batch_vec.push(sec_pool_1t.allocate().unwrap());
                    }
                    black_box(batch_vec.len());
                    batch_vec.clear();
                }
            },
        );
        drop(sec_pool_1t);
    }

    // 4. SecureMemoryPool 4T (1 instance per thread, zero 64-pool ring, zero spacers)
    if should_run_bench("memory_pools/secure_pool/alloc_free_4t") {
        set_allow_self_repin(true);
        maybe_repin_self_within_ccd();
        wait_for_quiet_sibling();
        set_allow_self_repin(false);
        let sec_p_core = current_self_pinned_core();
        pin_current_thread_to_core(sec_p_core);
        let sec_pool_4t = SecureMemoryPool::new(sec_cfg.clone()).unwrap();
        {
            let pre: Vec<_> = (0..32).map(|_| sec_pool_4t.allocate().unwrap()).collect();
            drop(pre);
        }
        {
            let ready_barrier = Arc::new(Barrier::new(4));
            let stop_flag = Arc::new(zipora::memory::CachePadded::new(AtomicBool::new(false)));
            let active_flag = Arc::new(zipora::memory::CachePadded::new(AtomicBool::new(false)));
            let mut handles = Vec::with_capacity(3);
            for worker_idx in 0..3usize {
                let pool = SecureMemoryPool::new(sec_cfg.clone()).unwrap();
                let ready = Arc::clone(&ready_barrier);
                let stop = Arc::clone(&stop_flag);
                let active = Arc::clone(&active_flag);
                let worker_core = worker_core_for(sec_p_core, worker_idx);
                handles.push(thread::spawn(move || {
                    pin_current_thread_to_core(worker_core);
                    let pre: Vec<_> = (0..32).map(|_| pool.allocate().unwrap()).collect();
                    drop(pre);
                    let mut w_batch = Vec::with_capacity(sec_batch);
                    ready.wait();
                    while !stop.load(Ordering::Relaxed) {
                        if active.load(Ordering::Relaxed) {
                            for _ in 0..sec_batch {
                                w_batch.push(pool.allocate().unwrap());
                            }
                            black_box(w_batch.len());
                            w_batch.clear();
                        } else {
                            std::hint::spin_loop();
                        }
                    }
                }));
            }
            ready_barrier.wait();

            measure_bench(
                records,
                "memory_pools",
                "memory_pools/secure_pool/alloc_free_4t",
                "ns/op",
                sec_params,
                || {
                    active_flag.store(true, Ordering::Relaxed);
                    let mut batch_vec = Vec::with_capacity(sec_batch);
                    for _ in 0..sec_rounds {
                        for _ in 0..sec_batch {
                            batch_vec.push(sec_pool_4t.allocate().unwrap());
                        }
                        black_box(batch_vec.len());
                        batch_vec.clear();
                    }
                    active_flag.store(false, Ordering::Relaxed);
                },
            );

            stop_flag.store(true, Ordering::Release);
            for h in handles {
                let _ = h.join();
            }
        }
        drop(sec_pool_4t);
    }

    let layout_256 = Layout::from_size_align(alloc_size, 16).unwrap();
    let pool_layouts: [Layout; 4] = [
        Layout::from_size_align(POOL_SIZES[0], 16).unwrap(),
        Layout::from_size_align(POOL_SIZES[1], 16).unwrap(),
        Layout::from_size_align(POOL_SIZES[2], 16).unwrap(),
        Layout::from_size_align(POOL_SIZES[3], 16).unwrap(),
    ];

    // 5. Competitor: mimalloc (1T & 4T)
    if should_run_bench("memory_pools/mimalloc/alloc_free_1t") {
        set_allow_self_repin(true);
        maybe_repin_self_within_ccd();
        measure_bench(
            records,
            "memory_pools",
            "memory_pools/mimalloc/alloc_free_1t",
            "ns/op",
            comp_pool_params,
            || {
                for _ in 0..ops {
                    let lay = black_box(layout_256);
                    // SAFETY: `lay` is a valid non-zero `Layout` (256 bytes, 16-byte aligned),
                    // and `ptr` is immediately deallocated with the identical `lay`.
                    unsafe {
                        let ptr = MiMalloc.alloc(lay);
                        black_box(ptr);
                        MiMalloc.dealloc(ptr, lay);
                    }
                }
            },
        );
    }

    if should_run_bench("memory_pools/mimalloc/alloc_free_4t") {
        set_allow_self_repin(true);
        maybe_repin_self_within_ccd();
        wait_for_quiet_sibling();
        set_allow_self_repin(false);
        let p_core_4t = current_self_pinned_core();
        pin_current_thread_to_core(p_core_4t);
        let ready_barrier = Arc::new(Barrier::new(4));
        let stop_flag = Arc::new(zipora::memory::CachePadded::new(AtomicBool::new(false)));
        let active_flag = Arc::new(zipora::memory::CachePadded::new(AtomicBool::new(false)));
        let mut handles = Vec::with_capacity(3);
        for worker_idx in 0..3usize {
            let ready = Arc::clone(&ready_barrier);
            let stop = Arc::clone(&stop_flag);
            let active = Arc::clone(&active_flag);
            let worker_core = worker_core_for(p_core_4t, worker_idx);
            let lay = pool_layouts[worker_idx + 1];
            handles.push(thread::spawn(move || {
                pin_current_thread_to_core(worker_core);
                ready.wait();
                while !stop.load(Ordering::Relaxed) {
                    if active.load(Ordering::Relaxed) {
                        for _ in 0..64 {
                            let l = black_box(lay);
                            // SAFETY: `l` is a valid non-zero `Layout`.
                            unsafe {
                                let ptr = MiMalloc.alloc(l);
                                black_box(ptr);
                                MiMalloc.dealloc(ptr, l);
                            }
                        }
                    } else {
                        std::hint::spin_loop();
                    }
                }
            }));
        }
        ready_barrier.wait();

        let main_lay = pool_layouts[0];
        measure_bench(
            records,
            "memory_pools",
            "memory_pools/mimalloc/alloc_free_4t",
            "ns/op",
            comp_pool_params,
            || {
                active_flag.store(true, Ordering::Relaxed);
                for _ in 0..ops {
                    let l = black_box(main_lay);
                    // SAFETY: `l` is a valid non-zero `Layout`.
                    unsafe {
                        let ptr = MiMalloc.alloc(l);
                        black_box(ptr);
                        MiMalloc.dealloc(ptr, l);
                    }
                }
                active_flag.store(false, Ordering::Relaxed);
            },
        );

        stop_flag.store(true, Ordering::Release);
        for h in handles {
            let _ = h.join();
        }
    }

    // 6. Competitor: tikv-jemallocator (1T & 4T)
    if should_run_bench("memory_pools/jemalloc/alloc_free_1t") {
        set_allow_self_repin(true);
        maybe_repin_self_within_ccd();
        measure_bench(
            records,
            "memory_pools",
            "memory_pools/jemalloc/alloc_free_1t",
            "ns/op",
            comp_pool_params,
            || {
                for _ in 0..ops {
                    let lay = black_box(layout_256);
                    // SAFETY: `lay` is a valid non-zero `Layout` (256 bytes, 16-byte aligned).
                    unsafe {
                        let ptr = Jemalloc.alloc(lay);
                        black_box(ptr);
                        Jemalloc.dealloc(ptr, lay);
                    }
                }
            },
        );
    }

    if should_run_bench("memory_pools/jemalloc/alloc_free_4t") {
        set_allow_self_repin(true);
        maybe_repin_self_within_ccd();
        wait_for_quiet_sibling();
        set_allow_self_repin(false);
        let p_core_4t = current_self_pinned_core();
        pin_current_thread_to_core(p_core_4t);
        let ready_barrier = Arc::new(Barrier::new(4));
        let stop_flag = Arc::new(zipora::memory::CachePadded::new(AtomicBool::new(false)));
        let active_flag = Arc::new(zipora::memory::CachePadded::new(AtomicBool::new(false)));
        let mut handles = Vec::with_capacity(3);
        for worker_idx in 0..3usize {
            let ready = Arc::clone(&ready_barrier);
            let stop = Arc::clone(&stop_flag);
            let active = Arc::clone(&active_flag);
            let worker_core = worker_core_for(p_core_4t, worker_idx);
            let lay = pool_layouts[worker_idx + 1];
            handles.push(thread::spawn(move || {
                pin_current_thread_to_core(worker_core);
                ready.wait();
                while !stop.load(Ordering::Relaxed) {
                    if active.load(Ordering::Relaxed) {
                        for _ in 0..64 {
                            let l = black_box(lay);
                            // SAFETY: `l` is a valid non-zero `Layout`.
                            unsafe {
                                let ptr = Jemalloc.alloc(l);
                                black_box(ptr);
                                Jemalloc.dealloc(ptr, l);
                            }
                        }
                    } else {
                        std::hint::spin_loop();
                    }
                }
            }));
        }
        ready_barrier.wait();

        let main_lay = pool_layouts[0];
        measure_bench(
            records,
            "memory_pools",
            "memory_pools/jemalloc/alloc_free_4t",
            "ns/op",
            comp_pool_params,
            || {
                active_flag.store(true, Ordering::Relaxed);
                for _ in 0..ops {
                    let l = black_box(main_lay);
                    // SAFETY: `l` is a valid non-zero `Layout`.
                    unsafe {
                        let ptr = Jemalloc.alloc(l);
                        black_box(ptr);
                        Jemalloc.dealloc(ptr, l);
                    }
                }
                active_flag.store(false, Ordering::Relaxed);
            },
        );

        stop_flag.store(true, Ordering::Release);
        for h in handles {
            let _ = h.join();
        }
    }

    // 7. Competitor: std::alloc::System (1T & 4T)
    if should_run_bench("memory_pools/system_alloc/alloc_free_1t") {
        set_allow_self_repin(true);
        maybe_repin_self_within_ccd();
        measure_bench(
            records,
            "memory_pools",
            "memory_pools/system_alloc/alloc_free_1t",
            "ns/op",
            comp_pool_params,
            || {
                for _ in 0..ops {
                    let lay = black_box(layout_256);
                    // SAFETY: `lay` is a valid non-zero `Layout` (256 bytes, 16-byte aligned).
                    unsafe {
                        let ptr = SystemAlloc.alloc(lay);
                        black_box(ptr);
                        SystemAlloc.dealloc(ptr, lay);
                    }
                }
            },
        );
    }

    if should_run_bench("memory_pools/system_alloc/alloc_free_4t") {
        set_allow_self_repin(true);
        maybe_repin_self_within_ccd();
        wait_for_quiet_sibling();
        set_allow_self_repin(false);
        let p_core_4t = current_self_pinned_core();
        pin_current_thread_to_core(p_core_4t);
        let ready_barrier = Arc::new(Barrier::new(4));
        let stop_flag = Arc::new(zipora::memory::CachePadded::new(AtomicBool::new(false)));
        let active_flag = Arc::new(zipora::memory::CachePadded::new(AtomicBool::new(false)));
        let mut handles = Vec::with_capacity(3);
        for worker_idx in 0..3usize {
            let ready = Arc::clone(&ready_barrier);
            let stop = Arc::clone(&stop_flag);
            let active = Arc::clone(&active_flag);
            let worker_core = worker_core_for(p_core_4t, worker_idx);
            let lay = pool_layouts[worker_idx + 1];
            handles.push(thread::spawn(move || {
                pin_current_thread_to_core(worker_core);
                ready.wait();
                while !stop.load(Ordering::Relaxed) {
                    if active.load(Ordering::Relaxed) {
                        for _ in 0..64 {
                            let l = black_box(lay);
                            // SAFETY: `l` is a valid non-zero `Layout`.
                            unsafe {
                                let ptr = SystemAlloc.alloc(l);
                                black_box(ptr);
                                SystemAlloc.dealloc(ptr, l);
                            }
                        }
                    } else {
                        std::hint::spin_loop();
                    }
                }
            }));
        }
        ready_barrier.wait();

        let main_lay = pool_layouts[0];
        measure_bench(
            records,
            "memory_pools",
            "memory_pools/system_alloc/alloc_free_4t",
            "ns/op",
            comp_pool_params,
            || {
                active_flag.store(true, Ordering::Relaxed);
                for _ in 0..ops {
                    let l = black_box(main_lay);
                    // SAFETY: `l` is a valid non-zero `Layout`.
                    unsafe {
                        let ptr = SystemAlloc.alloc(l);
                        black_box(ptr);
                        SystemAlloc.dealloc(ptr, l);
                    }
                }
                active_flag.store(false, Ordering::Relaxed);
            },
        );

        stop_flag.store(true, Ordering::Release);
        for h in handles {
            let _ = h.join();
        }
    }

    set_allow_self_repin(true);
    pin_current_thread_to_core(current_self_pinned_core());
}
