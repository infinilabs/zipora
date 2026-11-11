//! Regression tests added during the v5.0.2 release review (plan.md Phase 2 rework).
//!
//! One module per topic. Tests that install a `#[global_allocator]` live in their own
//! `tests/plan_v2_alloc_*.rs` files.

/// SortedUintVec / ZipOffsetBlobStore: block minimums wider than sample_width are rejected (every preset, including the 32-bit default).
mod zip_offset {
    use zipora::blob_store::{
        BlobStore, SortedUintVecBuilder, SortedUintVecConfig, ZipOffsetBlobStoreBuilder,
        ZipOffsetBlobStoreConfig,
    };

    /// `SortedUintVecConfig::default()` (also the offset config of
    /// `ZipOffsetBlobStoreConfig::default()` / `security_optimized()`) has
    /// `sample_width == 32`, and `sample_width == 0` does not even validate.
    /// A block starting at 4 GiB was silently stored as 0 by v5.0.1.
    #[test]
    fn test_v2_t21_default_sorted_uint_vec_config_was_affected() {
        let cfg = SortedUintVecConfig::default();
        assert_eq!(cfg.sample_width, 32);
        assert!(
            SortedUintVecConfig {
                sample_width: 0,
                ..cfg
            }
            .validate()
            .is_err()
        );
        assert_eq!(
            ZipOffsetBlobStoreConfig::default()
                .offset_config
                .sample_width,
            32
        );

        let mut builder = SortedUintVecBuilder::new();
        builder
            .extend((0..64u64).chain((0..64u64).map(|i| (1u64 << 32) + i)))
            .unwrap();
        match builder.finish() {
            Ok(vec) => panic!(
                "finish() returned Ok and get(64) = {:?} instead of {}",
                vec.get(64),
                1u64 << 32
            ),
            Err(e) => assert!(
                e.to_string().contains("sample_width 32"),
                "unexpected error: {e}"
            ),
        }
    }

    /// End to end with the 24-bit offsets of `compression_optimized()`
    /// (`SortedUintVecConfig::memory_optimized()`): a store whose content passes
    /// 16 MiB. Without compression or checksums, v5.0.1's `finish()` succeeds and
    /// records past 16 MiB silently read back the wrong bytes; v5.0.2's
    /// `ZipOffsetBlobStoreBuilder::finish()` must return the new error.
    #[test]
    fn test_v2_t21_zip_offset_store_past_16mib_with_24bit_offsets() {
        assert_eq!(
            ZipOffsetBlobStoreConfig::compression_optimized()
                .offset_config
                .sample_width,
            24
        );
        let config = ZipOffsetBlobStoreConfig {
            compress_level: 0,
            checksum_level: 0,
            offset_config: SortedUintVecConfig::memory_optimized(),
            use_secure_memory: false,
            enable_simd: true,
        };
        const REC: usize = 60; // 64 records * 60 B = 3840 B < 2^12 (offset_width 12)
        let n = (16usize << 20) / REC + 1_000;
        let record = |i: usize| -> Vec<u8> {
            let mut r = (i as u64).to_le_bytes().to_vec();
            r.resize(REC, (i % 251) as u8);
            r
        };
        let mut builder = ZipOffsetBlobStoreBuilder::with_config(config).unwrap();
        for i in 0..n {
            builder.add_record(&record(i)).unwrap();
        }
        match builder.finish() {
            Ok(store) => {
                let wrong = (0..n)
                    .filter(|&i| store.get(i as u32).ok() != Some(record(i)))
                    .count();
                panic!("finish() returned Ok; {wrong} of {n} records read back wrong");
            }
            Err(e) => assert!(
                e.to_string().contains("sample_width 24"),
                "unexpected error: {e}"
            ),
        }
    }
}

/// ZReorderMap rejects a 10th varint byte above 0x01.
mod reorder_map {
    #![cfg(feature = "mmap")]

    use std::io::Write;
    use zipora::blob_store::ZReorderMap;

    /// Header `[size: u64 LE][sign: i64 LE = 1]` followed by `entries`.
    fn map_file(size: u64, entries: &[u8]) -> tempfile::NamedTempFile {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&size.to_le_bytes()).unwrap();
        file.write_all(&1i64.to_le_bytes()).unwrap();
        file.write_all(entries).unwrap();
        file.flush().unwrap();
        file
    }

    /// Sequence entry for value 7 whose run length is `0x81, 0x80 x 8, 0x02`,
    /// i.e. 1 + 2^64, which does not fit `usize`. v5.0.1 computes
    /// `0x02 << 63 == 0`, silently reads a run length of 1, and yields
    /// `[7, 9, 11]`. v5.0.2 must fail `open()` with "var_uint overflow".
    #[test]
    fn test_v2_t21_reorder_map_rejects_overlong_var_uint() {
        let mut entries = vec![14u8, 0, 0, 0, 0]; // (7 << 1) | 0: sequence, var_uint length follows
        entries.extend_from_slice(&[0x81, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x02]);
        entries.extend_from_slice(&[19, 0, 0, 0, 0]); // (9 << 1) | 1: single 9
        entries.extend_from_slice(&[23, 0, 0, 0, 0]); // (11 << 1) | 1: single 11
        let file = map_file(3, &entries);
        match ZReorderMap::open(file.path()) {
            Ok(map) => panic!(
                "over-long var_uint accepted; map yields {:?}",
                map.collect::<Vec<usize>>()
            ),
            Err(e) => assert!(
                e.to_string().contains("var_uint overflow"),
                "unexpected error: {e}"
            ),
        }
    }

    /// Guard: a non-canonical but valid 10-byte run length (10th byte 0x00,
    /// value 2) still decodes in both versions.
    #[test]
    fn test_v2_t21_reorder_map_noncanonical_10_byte_var_uint_guard() {
        let mut entries = vec![14u8, 0, 0, 0, 0]; // sequence starting at 7
        entries.extend_from_slice(&[0x82, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x00]);
        entries.extend_from_slice(&[19, 0, 0, 0, 0]); // single 9
        let file = map_file(3, &entries);
        let map = ZReorderMap::open(file.path()).unwrap();
        assert_eq!(map.collect::<Vec<usize>>(), vec![7, 8, 9]);
    }
}

/// NestLoudsTrieBlobStore: key_count, total_key_size and keys() track the distinct live keys.
mod key_count {
    use std::collections::BTreeMap;

    use zipora::blob_store::{BlobStore, NestLoudsTrieBlobStore, TrieBlobStoreConfig};
    use zipora::{RankSelectInterleaved256, RecordId};

    type Store = NestLoudsTrieBlobStore<RankSelectInterleaved256>;

    fn run(name: &str, config: TrieBlobStoreConfig) {
        let mut store = Store::new(config).expect("store");
        // key -> (latest record id, latest value)
        let mut latest: BTreeMap<Vec<u8>, (RecordId, Vec<u8>)> = BTreeMap::new();
        // record ids that were superseded by a later put of the same key and are still live
        let mut superseded: Vec<RecordId> = Vec::new();

        let mut x: u64 = 0x2545_F491_4F6C_DD1D;
        for step in 0..6000u64 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let op = x % 10;
            if op < 7 {
                let key = format!("k/{:03}/{}", x % 997, (x >> 20) % 7).into_bytes();
                let value = format!("v{step}").into_bytes();
                let id = store.put_with_key(&key, &value).expect("put_with_key");
                if let Some((old_id, _)) = latest.insert(key, (id, value)) {
                    superseded.push(old_id);
                }
            } else if op < 9 {
                // Removing a superseded record must not change key_count.
                if let Some(id) = superseded.pop() {
                    store.remove(id).expect("remove superseded record");
                }
            } else {
                // Removing the latest record deletes the key.
                let victim = latest
                    .keys()
                    .nth((x >> 8) as usize % latest.len().max(1))
                    .cloned();
                if let Some(key) = victim {
                    let (id, _) = latest.remove(&key).expect("tracked key");
                    store.remove(id).expect("remove latest record");
                }
            }

            if step % 250 == 0 || step == 5999 {
                let live_keys = latest.len();
                let live_key_bytes: usize = latest.keys().map(|k| k.len()).sum();
                assert_eq!(
                    store.key_count(),
                    live_keys,
                    "{name} step {step}: key_count() vs distinct live keys"
                );
                assert_eq!(
                    store.trie_stats().total_key_size,
                    live_key_bytes,
                    "{name} step {step}: total_key_size vs live key bytes"
                );
                assert_eq!(
                    store.keys().expect("keys").len(),
                    live_keys,
                    "{name} step {step}: keys() vs distinct live keys"
                );
            }
        }

        for (key, (_, value)) in &latest {
            assert_eq!(
                &store.get_by_key(key).expect("get_by_key"),
                value,
                "{name}: latest value for {:?}",
                String::from_utf8_lossy(key)
            );
        }
    }

    #[test]
    fn test_v2_c3_key_count_matches_live_keys_patricia() {
        run("patricia", TrieBlobStoreConfig::default());
    }

    #[test]
    fn test_v2_c3_key_count_matches_live_keys_double_array() {
        run(
            "double_array",
            TrieBlobStoreConfig::memory_optimized().statistics(true),
        );
    }
}

/// PaZipCompressor::compress appends to the caller's buffer on both paths and counts only the appended bytes.
mod pazip_append {
    use zipora::compression::dict_zip::{
        DictionaryBuilder, DictionaryBuilderConfig, PaZipCompressor, PaZipCompressorConfig,
    };
    use zipora::memory::{SecureMemoryPool, SecurePoolConfig};

    const JSON_RECORD: &[u8] = br#"{"service":"zipora-search","status":200,"region":"us-east1"}"#;
    const PREFIX: &[u8] = b"CALLER-OWNED-HEADER\x00\x01\x02";

    fn fnv1a64(bytes: &[u8]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &b in bytes {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    /// Fresh compressor with the default config (`enable_multithreading = true`,
    /// `multithreading_threshold = 64 KiB`) and a deterministic dictionary.
    fn compressor() -> PaZipCompressor {
        let dictionary = DictionaryBuilder::with_config(DictionaryBuilderConfig::default())
            .build(&JSON_RECORD.repeat(16))
            .expect("dictionary");
        let pool = SecureMemoryPool::new(SecurePoolConfig::new(4096, 1024, 8)).expect("pool");
        PaZipCompressor::new(dictionary, PaZipCompressorConfig::default(), pool)
            .expect("compressor")
    }

    /// Deterministic JSON-lines payload of exactly `len` bytes.
    fn input(len: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(len + 160);
        let mut i: u64 = 0;
        while out.len() < len {
            let line = format!(
                "{{\"id\":{},\"service\":\"zipora-search\",\"status\":{},\"region\":\"us-east{}\",\"latency_us\":{}}}\n",
                i,
                200 + (i % 7),
                i % 3,
                (i.wrapping_mul(2_654_435_761) >> 7) % 100_000
            );
            out.extend_from_slice(line.as_bytes());
            i += 1;
        }
        out.truncate(len);
        out
    }

    fn check(len: usize) {
        let data = input(len);

        let mut fresh = Vec::new();
        let fresh_stats = compressor()
            .compress(&data, &mut fresh)
            .expect("compress into empty buffer");
        println!(
            "len={len}: empty-buffer stream len={} fnv1a64={:016x} stats.bytes_output={}",
            fresh.len(),
            fnv1a64(&fresh),
            fresh_stats.bytes_output
        );

        let mut out = PREFIX.to_vec();
        let stats = compressor()
            .compress(&data, &mut out)
            .expect("compress into prefixed buffer");
        println!(
            "len={len}: prefixed buffer total len={} stats.bytes_output={}",
            out.len(),
            stats.bytes_output
        );

        assert!(
            out.starts_with(PREFIX),
            "len={len}: caller prefix was not preserved"
        );
        assert!(
            out[PREFIX.len()..] == fresh[..],
            "len={len}: appended stream differs from the empty-buffer stream"
        );
        assert_eq!(
            stats.bytes_output as usize,
            fresh.len(),
            "len={len}: bytes_output must count only the appended bytes"
        );

        let mut decoded = Vec::new();
        compressor()
            .decompress(&out[PREFIX.len()..], &mut decoded)
            .expect("decompress");
        assert!(decoded == data, "len={len}: round trip mismatch");
    }

    /// >= 1 MiB with the default config: `compress_parallel` takes the 64 KiB block path.
    #[test]
    fn test_v2_m3_parallel_block_path_appends_identical_stream() {
        check((1 << 20) + 12_345);
    }

    /// 64 KiB <= len < 1 MiB: `compress_parallel` falls back to `compress_sequential`.
    #[test]
    fn test_v2_m3_sequential_path_appends_identical_stream() {
        check(200 * 1024);
    }
}

/// VersionManager::min_version never passes the version of a live token.
mod min_version {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    use zipora::fsa::version_sync::{ConcurrencyLevel, VersionManager};

    #[test]
    fn test_v2_s1_min_version_never_exceeds_live_token_version() {
        let iters: u64 = if cfg!(miri) { 64 } else { 20_000 };
        for level in [
            ConcurrencyLevel::MultiWriteMultiRead,
            ConcurrencyLevel::OneWriteMultiRead,
        ] {
            let mgr = Arc::new(VersionManager::new(level));
            let violations = Arc::new(AtomicU64::new(0));
            std::thread::scope(|s| {
                for _ in 0..4 {
                    let mgr = Arc::clone(&mgr);
                    let violations = Arc::clone(&violations);
                    s.spawn(move || {
                        for _ in 0..iters {
                            let r = mgr.acquire_reader_token().unwrap();
                            let v = r.version();
                            if mgr.min_version() > v {
                                violations.fetch_add(1, Ordering::Relaxed);
                            }
                            // OneWriteMultiRead: the single writer slot may be busy; that's fine.
                            if let Ok(w) = mgr.acquire_writer_token() {
                                if mgr.min_version() > v {
                                    violations.fetch_add(1, Ordering::Relaxed);
                                }
                                drop(w);
                            }
                            if mgr.min_version() > v {
                                violations.fetch_add(1, Ordering::Relaxed);
                            }
                            drop(r);
                        }
                    });
                }
            });
            assert_eq!(
                violations.load(Ordering::Relaxed),
                0,
                "{level:?}: min_version advanced past a live token"
            );
            assert_eq!(mgr.active_readers(), 0);
            assert_eq!(mgr.active_writers(), 0);
            assert_eq!(mgr.min_version(), mgr.current_version());
        }
    }
}

/// ThreadLocalMemoryPool::allocate from a late thread-local destructor falls back to a standalone allocation.
mod late_tls {
    use std::cell::RefCell;
    use std::sync::Arc;
    use std::sync::atomic::Ordering;

    use thread_local::ThreadLocal;
    use zipora::memory::{ThreadLocalMemoryPool, ThreadLocalPoolConfig};

    fn new_pool() -> Arc<ThreadLocalMemoryPool> {
        ThreadLocalMemoryPool::new(ThreadLocalPoolConfig {
            use_secure_memory: false,
            ..ThreadLocalPoolConfig::default()
        })
        .unwrap()
    }

    struct LateAlloc(Option<Arc<ThreadLocalMemoryPool>>);

    impl Drop for LateAlloc {
        fn drop(&mut self) {
            if let Some(pool) = self.0.take() {
                let mut a = pool.allocate(64).expect("late-TLS allocate must succeed");
                a.as_mut_slice().fill(0x5A);
                assert!(a.as_slice().iter().all(|&b| b == 0x5A));
                let mut b = pool
                    .allocate(64)
                    .expect("second late-TLS allocate must succeed");
                b.as_mut_slice().fill(0x33);
                drop(a);
                drop(b);
                let big = pool
                    .allocate(8192)
                    .expect("oversize late-TLS allocate must succeed");
                assert!(big.as_slice().iter().all(|&x| x == 0));
                drop(big);
            }
        }
    }

    thread_local! {
        static LATE_A: RefCell<LateAlloc> = const { RefCell::new(LateAlloc(None)) };
        static LATE_B: RefCell<LateAlloc> = const { RefCell::new(LateAlloc(None)) };
    }

    /// Scenario A (the v5.0.1 abort): the thread used some `thread_local::ThreadLocal`, so the
    /// crate's THREAD_GUARD is registered after LATE_A and destroyed before it, but the thread
    /// never allocated from any `ThreadLocalMemoryPool` (zipora's OS_TID stays uninitialized).
    #[test]
    fn test_v2_s2_scenario_a_thread_guard_already_destroyed() {
        let pool = new_pool();
        let p = Arc::clone(&pool);
        std::thread::spawn(move || {
            LATE_A.with(|c| c.borrow_mut().0 = Some(p));
            let other: ThreadLocal<u8> = ThreadLocal::new();
            let _ = other.get_or(|| 1);
        })
        .join()
        .expect("thread must exit without aborting");

        let s = pool.stats().unwrap();
        assert_eq!(
            s.arena_allocations.load(Ordering::Relaxed),
            0,
            "no slab may be created during late teardown"
        );
        assert_eq!(s.hot_allocations.load(Ordering::Relaxed), 0);
        assert_eq!(
            s.cache_misses.load(Ordering::Relaxed),
            3,
            "all three late allocations must take the standalone fallback"
        );
    }

    /// Scenario B: the thread never touched any `thread_local::ThreadLocal` before teardown, so
    /// `get_or` can still register THREAD_GUARD from the late destructor and the slab path works.
    #[test]
    fn test_v2_s2_scenario_b_thread_guard_never_initialized() {
        let pool = new_pool();
        let p = Arc::clone(&pool);
        std::thread::spawn(move || {
            LATE_B.with(|c| c.borrow_mut().0 = Some(p));
        })
        .join()
        .expect("thread must exit without aborting");

        let s = pool.stats().unwrap();
        assert_eq!(s.arena_allocations.load(Ordering::Relaxed), 1);
        assert_eq!(
            s.cache_misses.load(Ordering::Relaxed),
            1,
            "only the 8 KiB request is standalone"
        );
    }
}

/// FseEncoder output does not depend on how many worker threads encode the blocks.
mod fse_inline {
    use zipora::entropy::{FseConfig, FseDecoder, FseEncoder};

    fn fnv1a64(bytes: &[u8]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &b in bytes {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    /// Deterministic skewed payload (xorshift64*) over a 4-symbol alphabet.
    fn payload(len: usize) -> Vec<u8> {
        const ALPHABET: &[u8] = b"aaaaabbbccd";
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
        (0..len)
            .map(|_| {
                x ^= x >> 12;
                x ^= x << 25;
                x ^= x >> 27;
                let r = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
                ALPHABET[((r >> 33) % ALPHABET.len() as u64) as usize]
            })
            .collect()
    }

    #[test]
    fn test_v2_c1_fse_parallel_output_is_independent_of_worker_threads() {
        let avail = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let data = payload(1 << 20);
        for parallel_blocks in [Some(2), Some(4), Some(16)] {
            let cfg = FseConfig {
                parallel_blocks,
                block_size: 64 * 1024,
                ..FseConfig::default()
            };
            let compressed = FseEncoder::new(cfg.clone())
                .expect("encoder")
                .compress(&data)
                .expect("compress");
            println!(
                "available_parallelism={avail} parallel_blocks={parallel_blocks:?}: \
             len={} fnv1a64={:016x}",
                compressed.len(),
                fnv1a64(&compressed)
            );
            assert_eq!(compressed[0], 0xF6, "parallel framing byte");
            let decoded = FseDecoder::with_config(cfg)
                .expect("decoder")
                .decompress(&compressed)
                .expect("decompress");
            assert!(decoded == data, "round trip mismatch");
        }
    }
}

/// LruPageCache::drop: write errors are logged, and pages flushed explicitly are not written twice.
mod lru_drop {
    use std::io::Write;
    use std::sync::Mutex;

    use zipora::cache::{LruPageCache, PageCacheConfig};

    struct Capture(Mutex<Vec<String>>);

    impl log::Log for Capture {
        fn enabled(&self, _: &log::Metadata<'_>) -> bool {
            true
        }
        fn log(&self, record: &log::Record<'_>) {
            self.0.lock().unwrap().push(format!("{}", record.args()));
        }
        fn flush(&self) {}
    }

    static LOGGER: Capture = Capture(Mutex::new(Vec::new()));

    /// Every write to `/dev/full` fails with `ENOSPC`. The explicit flush must
    /// report it; `Drop` must retry, log a warning, and return normally.
    #[test]
    fn test_v2_drop_with_failing_write_logs_and_does_not_panic() {
        if std::fs::metadata("/dev/full").is_err() {
            eprintln!("skipped: no /dev/full");
            return;
        }
        log::set_logger(&LOGGER).unwrap();
        log::set_max_level(log::LevelFilter::Warn);

        let cache = LruPageCache::new(PageCacheConfig::balanced()).unwrap();
        let fid = cache.open_file("/dev/full").unwrap();
        cache.write(fid, 0, b"x").unwrap();
        assert!(
            cache.flush_file(fid).is_err(),
            "flush_file must report ENOSPC"
        );
        drop(cache);

        let logs = LOGGER.0.lock().unwrap();
        assert!(
            logs.iter()
                .any(|m| m.contains("LruPageCache::drop failed to flush")),
            "expected a drop-time warning, got {logs:?}"
        );
    }

    /// Guard: pages written back by `flush_file` are clean, so `Drop` must not
    /// write them again over newer on-disk data.
    #[test]
    fn test_v2_drop_after_explicit_flush_does_not_rewrite() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&[0u8; 4096]).unwrap();
        file.flush().unwrap();

        let cache = LruPageCache::new(PageCacheConfig::balanced()).unwrap();
        let fid = cache.open_file(file.path()).unwrap();
        cache.write(fid, 0, b"CACHED").unwrap();
        cache.flush_file(fid).unwrap();
        std::fs::write(file.path(), b"EXTERN").unwrap();
        drop(cache);

        assert_eq!(std::fs::read(file.path()).unwrap(), b"EXTERN");
    }
}

/// FixedCapacityMemoryPool::secure_clear on 4-aligned blocks (u32 words plus byte tail).
mod fcp_secure_clear {
    use zipora::memory::{FixedCapacityMemoryPool, FixedCapacityPoolConfig};

    #[test]
    fn test_v2_s3_secure_clear_u32_branch_and_tail() {
        let pool = FixedCapacityMemoryPool::new(FixedCapacityPoolConfig {
            max_block_size: 20,
            total_blocks: 2,
            alignment: 4,
            enable_stats: false,
            eager_allocation: true,
            secure_clear: true,
        })
        .unwrap();

        let mut a = pool.allocate(20).unwrap();
        let mut b = pool.allocate(20).unwrap();
        let (pa, pb) = (a.as_ptr() as usize, b.as_ptr() as usize);
        assert_eq!(pa.abs_diff(pb), 20);
        assert!(
            pa % 8 == 4 || pb % 8 == 4,
            "one block must take the u32 branch"
        );
        assert_eq!(a.size(), 20);
        a.as_mut_slice().fill(0xAA);
        b.as_mut_slice().fill(0xBB);
        drop(a);
        drop(b);

        let x = pool.allocate(20).unwrap();
        let y = pool.allocate(20).unwrap();
        assert!(x.as_slice().iter().all(|&v| v == 0), "block not wiped");
        assert!(y.as_slice().iter().all(|&v| v == 0), "block not wiped");
    }
}

/// CacheObliviousSort: the L1 path stays O(n log n) on reverse-sorted input.
mod cache_oblivious_l1 {
    use std::cell::Cell;
    use std::cmp::Ordering;
    use zipora::algorithms::{CacheObliviousConfig, CacheObliviousSort};
    use zipora::memory::CacheHierarchy;

    thread_local! {
        static COMPARISONS: Cell<u64> = const { Cell::new(0) };
    }

    /// 8-byte element whose `Ord` counts comparisons.
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct Counted8(u64);

    impl Ord for Counted8 {
        fn cmp(&self, other: &Self) -> Ordering {
            COMPARISONS.with(|c| c.set(c.get() + 1));
            self.0.cmp(&other.0)
        }
    }

    impl PartialOrd for Counted8 {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    fn reverse_sorted_comparisons(n: u64, cache_hierarchy: CacheHierarchy) -> u64 {
        let mut data: Vec<Counted8> = (0..n).rev().map(Counted8).collect();
        let mut sorter = CacheObliviousSort::with_config(CacheObliviousConfig {
            cache_hierarchy,
            ..CacheObliviousConfig::default()
        });
        COMPARISONS.with(|c| c.set(0));
        sorter.sort(&mut data).unwrap();
        assert!(
            data.windows(2).all(|w| w[0].0 <= w[1].0),
            "output is not sorted"
        );
        COMPARISONS.with(|c| c.get())
    }

    fn bound(n: u64) -> u64 {
        30 * n * (n as f64).log2().ceil() as u64
    }

    /// Default cache sizes (L1 32 KiB, also what x86 AMD hosts and Miri get):
    /// 4,096 reverse-sorted 8-byte elements.
    #[test]
    fn test_v2_l1_insertion_sort_default_cache_sizes() {
        let n = 4_096;
        let c = reverse_sorted_comparisons(n, CacheHierarchy::default());
        println!(
            "default sizes, n = {n}: {c} comparisons (bound {})",
            bound(n)
        );
        assert!(
            c <= bound(n),
            "{c} comparisons for n = {n} (bound {})",
            bound(n)
        );
    }

    /// The cache sizes of `plan_v2_cache_oblivious` (L1 = 8.5 n), with 8-byte
    /// instead of 16-byte elements: the same 199,990,000 comparisons the plan
    /// quotes for the (now fixed) L2 Lomuto path.
    #[test]
    fn test_v2_l1_insertion_sort_plan_v2_cache_sizes() {
        let n: u64 = 20_000;
        let c = reverse_sorted_comparisons(
            n,
            CacheHierarchy {
                l1_size: (n * 8 + n / 2) as usize,
                l2_size: 1 << 30,
                l3_size: 1 << 31,
                ..CacheHierarchy::default()
            },
        );
        println!(
            "plan_v2 sizes, n = {n}: {c} comparisons (bound {})",
            bound(n)
        );
        assert!(
            c <= bound(n),
            "{c} comparisons for n = {n} (bound {})",
            bound(n)
        );
    }
}

/// Substring search (`zipora::string`, `zipora::io::simd_memory`, `SimdPatternMatcher`): results
/// equal a naive search, and the adversarial `a...a` / `a...ab` input stays linear.
mod substring_search {
    use std::time::{Duration, Instant};

    use zipora::compression::{SimdPatternConfig, SimdPatternMatcher};
    use zipora::io::simd_memory::search;

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            // xorshift64*
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }

        fn below(&mut self, n: u64) -> u64 {
            self.next() % n.max(1)
        }
    }

    /// Start offsets of all (overlapping) occurrences, in order, at most `max`.
    fn naive_all(input: &[u8], pattern: &[u8], max: usize) -> Vec<usize> {
        let mut out = Vec::new();
        if pattern.is_empty() || input.len() < pattern.len() {
            return out;
        }
        for i in 0..=input.len() - pattern.len() {
            if out.len() >= max {
                break;
            }
            if &input[i..i + pattern.len()] == pattern {
                out.push(i);
            }
        }
        out
    }

    fn positions(
        matcher: &mut SimdPatternMatcher,
        input: &[u8],
        pattern: &[u8],
        max: usize,
    ) -> Vec<usize> {
        matcher
            .find_pattern_matches(input, pattern, max)
            .unwrap()
            .iter()
            .map(|m| m.input_position)
            .collect()
    }

    /// Small alphabets give many candidates and many overlapping matches. Needle lengths
    /// straddle every strategy switch (16/17, 32/33, 35/36).
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_v2_substring_results_match_naive_search() {
        let mut rng = Rng(0x00C0_FFEE_D00D_F00D);
        let searcher = search::SimdStringSearch::new();
        // Early termination off, so the whole capped match list comes back. `enable_bmi2`
        // selects the AVX2 tier on AVX2+BMI2 hosts; `enable_simd: false` the scalar tier.
        let mut avx2 = SimdPatternMatcher::with_config(SimdPatternConfig {
            enable_bmi2: true,
            enable_early_termination: false,
            ..Default::default()
        });
        let mut scalar = SimdPatternMatcher::with_config(SimdPatternConfig {
            enable_simd: false,
            enable_early_termination: false,
            ..Default::default()
        });

        for it in 0..12_000u64 {
            let alphabet = [1u64, 2, 2, 3, 4, 26][(it % 6) as usize];
            let hay_len = match it % 4 {
                0 => rng.below(40),
                1 => rng.below(200),
                2 => rng.below(700),
                _ => rng.below(3000),
            } as usize;
            let hay: Vec<u8> = (0..hay_len)
                .map(|_| b'a' + rng.below(alphabet) as u8)
                .collect();
            let needle_len = [
                0usize, 1, 2, 7, 16, 17, 31, 32, 33, 35, 36, 40, 64, 65, 129, 300,
            ][rng.below(16) as usize];
            // Mostly a slice of the haystack (so it occurs), sometimes random or with one
            // byte that never occurs.
            let mut needle: Vec<u8> = if hay_len >= needle_len && needle_len > 0 && it % 5 != 0 {
                let start = rng.below((hay_len - needle_len + 1) as u64) as usize;
                hay[start..start + needle_len].to_vec()
            } else {
                (0..needle_len)
                    .map(|_| b'a' + rng.below(alphabet) as u8)
                    .collect()
            };
            if !needle.is_empty() && it % 7 == 0 {
                let k = rng.below(needle.len() as u64) as usize;
                needle[k] = b'z';
            }
            let ctx =
                format!("iteration {it}: haystack {hay_len} bytes, needle {needle_len} bytes");

            let want_first = if needle.is_empty() {
                Some(0)
            } else {
                naive_all(&hay, &needle, 1).first().copied()
            };
            assert_eq!(searcher.find_pattern(&hay, &needle), want_first, "{ctx}");
            assert_eq!(search::find_pattern(&hay, &needle), want_first, "{ctx}");
            assert_eq!(search::scalar_strstr(&hay, &needle), want_first, "{ctx}");
            #[cfg(target_arch = "x86_64")]
            assert_eq!(search::sse42_strstr(&hay, &needle), want_first, "{ctx}");

            for max in [0usize, 1, 2, 5, usize::MAX] {
                let want = naive_all(&hay, &needle, max);
                assert_eq!(
                    positions(&mut scalar, &hay, &needle, max),
                    want,
                    "scalar tier, max {max}, {ctx}"
                );
                if needle.len() > 35 {
                    assert_eq!(
                        positions(&mut avx2, &hay, &needle, max),
                        want,
                        "{:?} tier, max {max}, {ctx}",
                        avx2.simd_tier()
                    );
                }
            }
        }
    }

    /// 1 MiB of `a` and the needle `a^65535 b`: every position is a candidate that fails only
    /// at its last byte. A first-byte filter with a full compare needs about 6.5e10 byte
    /// comparisons (1.4 s in a release build); a linear-time search needs milliseconds. Best
    /// of three runs, so a stalled CI runner does not fail the test.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_v2_substring_search_adversarial_input_is_linear() {
        const LIMIT: Duration = Duration::from_millis(500);

        fn assert_linear(label: &str, mut found: impl FnMut() -> bool) {
            let mut best = Duration::MAX;
            for _ in 0..3 {
                let start = Instant::now();
                assert!(!found(), "{label}: the needle does not occur");
                best = best.min(start.elapsed());
            }
            assert!(best < LIMIT, "{label} took {best:?} (limit {LIMIT:?})");
        }

        let haystack = vec![b'a'; 1 << 20];
        let mut needle = vec![b'a'; 65_535];
        needle.push(b'b');
        let (h, n) = (haystack.as_slice(), needle.as_slice());

        let string_searcher = zipora::string::SimdStringSearch::new();
        assert_linear("string::SimdStringSearch::sse42_strstr", || {
            string_searcher.sse42_strstr(h, n).is_some()
        });
        assert_linear("string::sse42_strstr", || {
            zipora::string::sse42_strstr(h, n).is_some()
        });
        assert_linear("FastStr::find", || {
            zipora::FastStr::new(h)
                .find(zipora::FastStr::new(n))
                .is_some()
        });

        let io_searcher = search::SimdStringSearch::new();
        assert_linear("io::simd_memory::SimdStringSearch::find_pattern", || {
            io_searcher.find_pattern(h, n).is_some()
        });
        assert_linear("io::simd_memory::search::scalar_strstr", || {
            search::scalar_strstr(h, n).is_some()
        });

        for config in [
            SimdPatternConfig::default(),
            SimdPatternConfig {
                enable_bmi2: true,
                ..Default::default()
            },
            SimdPatternConfig {
                enable_simd: false,
                ..Default::default()
            },
        ] {
            let mut matcher = SimdPatternMatcher::with_config(config);
            let label = format!(
                "SimdPatternMatcher::find_pattern_matches ({:?} tier)",
                matcher.simd_tier()
            );
            assert_linear(&label, || {
                !matcher.find_pattern_matches(h, n, 1).unwrap().is_empty()
            });
        }
    }
}
