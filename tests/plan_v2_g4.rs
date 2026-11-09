//! Plan v2 (reliability-first 5.0.x) regression tests, group G4.
//!
//! Every test here goes through the public API. Tests whose name ends in
//! `_guard` pass at v5.0.0 and pin behaviour a fix must keep; the others fail
//! at v5.0.0 for the reason given in their doc comment.

use std::io::Write;

// ---------------------------------------------------------------------------
// C5.3 `MemoryMappedInput::read_var_int` (src/io/mmap.rs)
// ---------------------------------------------------------------------------

#[cfg(feature = "mmap")]
mod mmap_var_int {
    use super::*;
    use zipora::DataInput;
    use zipora::io::{MemoryMappedInput, VarInt};

    /// Writes `bytes` followed by `pad` zero bytes and opens it. A file of at
    /// most 4 KiB uses the buffered strategy, a larger one a real mapping;
    /// both share `read_var_int`.
    fn open(bytes: &[u8], pad: usize) -> (tempfile::NamedTempFile, MemoryMappedInput) {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(bytes).unwrap();
        file.write_all(&vec![0u8; pad]).unwrap();
        file.flush().unwrap();
        let input = MemoryMappedInput::from_path(file.path()).unwrap();
        (file, input)
    }

    fn ten_byte_varint(last: u8) -> Vec<u8> {
        let mut bytes = vec![0xFF; 9];
        bytes.push(last);
        bytes
    }

    /// The 10th byte of a LEB128 `u64` carries bit 63 only. `VarInt::read_from`
    /// rejects `byte > 1` at shift 63; the mmap reader has no such check and
    /// `(byte & 0x7F) << 63` silently drops the high bits, returning
    /// `Ok(0x7FFF_FFFF_FFFF_FFFF)` (or `Ok(u64::MAX)`) for an encoding of a
    /// value >= 2^64.
    #[test]
    fn test_v2_mmap_read_var_int_rejects_u64_overflow() {
        for last in [0x02u8, 0x03, 0x7F] {
            let bytes = ten_byte_varint(last);
            assert!(
                VarInt::decode(&bytes).is_err(),
                "reference decoder rejects 0x{last:02X}"
            );
            for pad in [0usize, 8192] {
                let (_file, mut input) = open(&bytes, pad);
                let got = input.read_var_int();
                assert!(
                    got.is_err(),
                    "10th varint byte 0x{last:02X} (file {} bytes, strategy {:?}): \
                     read_var_int returned {got:?} for a value that overflows u64",
                    bytes.len() + pad,
                    input.strategy()
                );
            }
        }
    }

    /// Guard: the longest valid encodings still decode and agree with
    /// `VarInt::decode`.
    #[test]
    fn test_v2_mmap_read_var_int_max_values_guard() {
        for value in [u64::MAX, 1u64 << 63, (1u64 << 63) - 1, 0, 127, 128] {
            let bytes = VarInt::encode(value);
            assert_eq!(VarInt::decode(&bytes).unwrap().0, value);
            for pad in [0usize, 8192] {
                let (_file, mut input) = open(&bytes, pad);
                assert_eq!(input.read_var_int().unwrap(), value, "pad {pad}");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// C5.4 `EntropyBitOps::pack_bits` (src/entropy/bit_ops.rs)
// ---------------------------------------------------------------------------

mod pack_bits {
    use zipora::entropy::EntropyBitOps;

    /// `pack_bits` validates `width <= 32` and `offset <= 64` but not
    /// `offset + width <= 64`, so `64 - offset - width` underflows: a panic
    /// with overflow checks (test profile), a silent `Ok` that drops the value
    /// in release. It must return `Err`.
    #[test]
    fn test_v2_pack_bits_rejects_field_past_bit_64() {
        let ops = EntropyBitOps::new();
        for (offset, width) in [(60u32, 8u32), (64, 1), (33, 32), (57, 8)] {
            let mut stream = 0u64;
            let got = ops.pack_bits(&mut stream, 0xFF, offset, width);
            assert!(
                got.is_err(),
                "offset {offset} + width {width} > 64 must be rejected, got {got:?}"
            );
            assert_eq!(stream, 0, "rejected call must not modify the stream");
        }
    }

    /// Guard: fields that end exactly at bit 64 (or are empty) still pack.
    #[test]
    fn test_v2_pack_bits_boundaries_guard() {
        let ops = EntropyBitOps::new();
        let cases = [
            (0xFFu32, 56u32, 8u32, 0xFFu64),
            (0xFFFF_FFFF, 32, 32, 0xFFFF_FFFF),
            (0xFFFF_FFFF, 0, 32, 0xFFFF_FFFF_0000_0000),
            (0x1, 63, 1, 0x1),
            (0x0, 64, 0, 0x0),
            (0xAB, 0, 0, 0x0),
        ];
        for (value, offset, width, expected) in cases {
            let mut stream = 0u64;
            ops.pack_bits(&mut stream, value, offset, width).unwrap();
            assert_eq!(stream, expected, "offset {offset} width {width}");
        }
        let mut stream = 0u64;
        assert!(ops.pack_bits(&mut stream, 1, 65, 0).is_err());
        assert!(ops.pack_bits(&mut stream, 1, 0, 33).is_err());
    }
}

// ---------------------------------------------------------------------------
// C6.6 `LruPageCache` (src/cache/basic_cache.rs)
// ---------------------------------------------------------------------------

mod lru_page_cache {
    use super::*;
    use zipora::cache::{LruPageCache, PAGE_SIZE, PageCacheConfig};

    fn file_with(len: usize) -> tempfile::NamedTempFile {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&vec![0u8; len]).unwrap();
        file.flush().unwrap();
        file
    }

    /// (a) `write()` only marks pages dirty; `LruPageCache` has no `Drop`, so
    /// dropping it without `flush_file`/`close_file` discards the writes.
    #[test]
    fn test_v2_lru_cache_drop_flushes_dirty_pages() {
        let file = file_with(2 * PAGE_SIZE);
        {
            let cache = LruPageCache::new(PageCacheConfig::balanced()).unwrap();
            let fid = cache.open_file(file.path()).unwrap();
            cache.write(fid, 100, b"PERSIST-ME").unwrap();
            cache.write(fid, PAGE_SIZE as u64 + 7, b"PAGE-ONE").unwrap();
        }
        let disk = std::fs::read(file.path()).unwrap();
        assert_eq!(
            &disk[100..110],
            b"PERSIST-ME",
            "dirty page 0 was lost when the cache was dropped without flush_file"
        );
        assert_eq!(&disk[PAGE_SIZE + 7..PAGE_SIZE + 15], b"PAGE-ONE");
    }

    /// (b) Every miss also calls `record_hit(InitialFree)` (and
    /// `record_hit(EvictedOthers)` when it evicts), so `N` cold reads report
    /// `N` hits: a 50% hit ratio (63.6% with evictions) where 0% is true.
    #[test]
    fn test_v2_lru_cache_cold_reads_report_zero_hit_ratio() {
        const PAGES: usize = 8;
        let file = file_with(PAGES * PAGE_SIZE);
        for capacity_pages in [PAGES * 4, 2] {
            let config = PageCacheConfig::balanced()
                .with_capacity(capacity_pages * PAGE_SIZE)
                .with_shards(1);
            let cache = LruPageCache::new(config).unwrap();
            let fid = cache.open_file(file.path()).unwrap();
            for page in 0..PAGES {
                cache.read(fid, (page * PAGE_SIZE) as u64, 16).unwrap();
            }
            let stats = cache.stats();
            assert_eq!(
                stats.total_misses, PAGES as u64,
                "capacity {capacity_pages} pages"
            );
            assert_eq!(
                (stats.total_hits, stats.hit_ratio),
                (0, 0.0),
                "capacity {capacity_pages} pages: {PAGES} distinct cold reads must not count as hits"
            );
            // One warm read is exactly one hit.
            cache
                .read(fid, ((PAGES - 1) * PAGE_SIZE) as u64, 16)
                .unwrap();
            assert_eq!(
                cache.stats().total_hits,
                1,
                "capacity {capacity_pages} pages"
            );
        }
    }

    /// (c) `PageId` is `u32`, so `offset_to_page_id` truncates offsets at or
    /// above 2^44 (16 TiB). `write` then computes `PAGE_SIZE - offset_in_page`
    /// with `offset_in_page ~ 2^44` (overflow panic in the test profile, a
    /// ~16 TiB `resize` in release). Offsets the 32-bit page id cannot
    /// address must be rejected with `Err`. A virtual file (`fd == -1`) needs
    /// no backing storage.
    #[test]
    fn test_v2_lru_cache_rejects_offsets_beyond_32bit_page_ids() {
        let cache = LruPageCache::new(PageCacheConfig::balanced()).unwrap();
        let fid = cache.register_file(-1).unwrap();
        let limit = 1u64 << 44;
        assert!(cache.write(fid, limit, b"x").is_err(), "write at 16 TiB");
        assert!(
            cache.write(fid, limit - 2, b"xyzw").is_err(),
            "write across 16 TiB"
        );
        assert!(cache.read(fid, limit, 1).is_err(), "read at 16 TiB");
        assert!(cache.read(fid, limit - 2, 4).is_err(), "read across 16 TiB");
        assert!(cache.prefetch(fid, limit, 1).is_err(), "prefetch at 16 TiB");
        // The last addressable bytes are still accepted (EOF of an empty file).
        // No write there: flushing a dirty page of a virtual file grows its
        // backing `Vec` to the page offset.
        assert!(cache.read(fid, limit - 4, 4).unwrap().is_empty());
    }

    /// (c) Same truncation on the read side, observable only with a file
    /// larger than 16 TiB: v5.0.0 maps offset 2^44 to page 0 and returns an
    /// empty buffer for data that exists. Needs a filesystem that allows a
    /// sparse file above 16 TiB (tmpfs, xfs, btrfs; not ext4); otherwise the
    /// test logs and returns.
    #[test]
    fn test_v2_lru_cache_read_beyond_16tib_is_not_silently_short() {
        use std::os::unix::fs::FileExt;
        let limit = 1u64 << 44;
        let candidates = [std::env::temp_dir(), "/dev/shm".into(), "/tmp".into()];
        let Some(file) = candidates.iter().find_map(|dir| {
            let file = tempfile::Builder::new()
                .prefix("zipora-v2-sparse")
                .tempfile_in(dir)
                .ok()?;
            file.as_file().set_len(limit + 2 * PAGE_SIZE as u64).ok()?;
            Some(file)
        }) else {
            eprintln!("skipped: no filesystem here supports a sparse file above 16 TiB");
            return;
        };
        file.as_file().write_all_at(b"HELLO", limit).unwrap();

        let cache = LruPageCache::new(PageCacheConfig::balanced()).unwrap();
        let fid = cache.open_file(file.path()).unwrap();
        if let Ok(buf) = cache.read(fid, limit, 5) {
            assert_eq!(
                buf.data(),
                b"HELLO",
                "read at 16 TiB returned {:?}: the page id was truncated to 0",
                buf.data()
            );
        }
    }
}

// ---------------------------------------------------------------------------
// C6.8 `FastVec`'s `Extend` (src/containers/fast_vec/traits.rs)
// ---------------------------------------------------------------------------

mod fast_vec_extend {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use zipora::FastVec;

    /// The `Extend` impl falls back to `let _ = self.push(item)`, so every
    /// element `push` rejects is silently dropped. It is reached through
    /// `collect`/`FromIterator` and any code generic over `Extend` (the
    /// inherent, fallible `FastVec::extend` shadows it for method calls).
    /// `push` fails on allocation failure or when the length cannot grow; a
    /// zero-sized element type at `len == usize::MAX` reaches the latter
    /// deterministically. `Vec<()>` panics with "capacity overflow" here, and
    /// so must `FastVec`'s `Extend`.
    #[test]
    fn test_v2_fast_vec_extend_does_not_silently_drop_elements() {
        let mut v: FastVec<()> = FastVec::new();
        v.reserve(1).unwrap(); // ZST: capacity becomes usize::MAX
        assert_eq!(v.capacity(), usize::MAX);
        // SAFETY: `()` needs no initialisation and `usize::MAX <= capacity()`.
        unsafe { v.set_len(usize::MAX) };
        assert!(v.push(()).is_err(), "push at usize::MAX must fail");
        // The inherent, fallible `extend` already reports the failure.
        assert!(v.extend(std::iter::once(())).is_err());

        let extended = catch_unwind(AssertUnwindSafe(|| {
            <FastVec<()> as Extend<()>>::extend(&mut v, std::iter::once(()))
        }));
        assert!(
            extended.is_err(),
            "Extend::extend returned normally although push rejected the element (silently dropped)"
        );
        assert_eq!(v.len(), usize::MAX);
    }
}

// ---------------------------------------------------------------------------
// Algorithmic worst case: substring search (src/string/simd_search.rs,
// src/string/fast_str.rs)
// ---------------------------------------------------------------------------

mod strstr_guard {
    use zipora::FastStr;
    use zipora::string::{SimdStringSearch, sse42_strstr};

    fn naive(haystack: &[u8], needle: &[u8]) -> Option<usize> {
        if needle.len() > haystack.len() {
            return None;
        }
        (0..=haystack.len() - needle.len()).find(|&i| &haystack[i..i + needle.len()] == needle)
    }

    struct XorShift(u64);
    impl XorShift {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    fn check(searcher: &SimdStringSearch, haystack: &[u8], needle: &[u8]) {
        let expected = naive(haystack, needle);
        let simd_expected = if needle.is_empty() { None } else { expected };
        assert_eq!(
            searcher.sse42_strstr(haystack, needle),
            simd_expected,
            "sse42_strstr hay {} needle {:?}",
            haystack.len(),
            needle
        );
        assert_eq!(
            sse42_strstr(haystack, needle),
            simd_expected,
            "global sse42_strstr"
        );
        assert_eq!(
            FastStr::new(haystack).find(FastStr::new(needle)),
            expected,
            "FastStr::find hay {} needle {:?}",
            haystack.len(),
            needle
        );
    }

    /// Guard (passes at v5.0.0): every search entry point agrees with a naive
    /// reference on periodic, adversarial and random inputs, including the
    /// `a…ab` worst case at small sizes. A Two-Way / `memchr::memmem` fix must
    /// keep this green.
    #[test]
    fn test_v2_strstr_matches_naive_reference_guard() {
        let searcher = SimdStringSearch::new();
        // Adversarial periodic inputs.
        for n in [0usize, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 200, 1000] {
            let hay_a = vec![b'a'; n];
            let mut hay_ab = hay_a.clone();
            if n > 0 {
                hay_ab[n - 1] = b'b';
            }
            for m in [1usize, 2, 15, 16, 17, 32, 33, 64, 65, 300] {
                let mut needle = vec![b'a'; m.saturating_sub(1)];
                needle.push(b'b');
                for hay in [&hay_a, &hay_ab] {
                    check(&searcher, hay, &needle);
                    check(&searcher, hay, &vec![b'a'; m]);
                }
            }
        }
        // Overlapping self-similar needles.
        let hay = b"abaabaabaababaabaabaabab".repeat(9);
        for needle in [
            &b"abaabab"[..],
            &b"aabaab"[..],
            &b"babaa"[..],
            &b"abab"[..],
            &b"bb"[..],
            &b"abaabaabaabab"[..],
        ] {
            check(&searcher, &hay, needle);
        }
        // Random small alphabets (many partial matches).
        let mut rng = XorShift(0x5EED_1234_ABCD_0001);
        for _ in 0..3000 {
            let n = (rng.next() % 300) as usize;
            let m = 1 + (rng.next() % 40) as usize;
            let alphabet = 1 + (rng.next() % 3) as u8;
            let hay: Vec<u8> = (0..n)
                .map(|_| b'a' + (rng.next() % alphabet as u64) as u8)
                .collect();
            let needle: Vec<u8> = if n >= m && rng.next().is_multiple_of(2) {
                let start = (rng.next() as usize) % (n - m + 1);
                hay[start..start + m].to_vec()
            } else {
                (0..m)
                    .map(|_| b'a' + (rng.next() % alphabet as u64) as u8)
                    .collect()
            };
            check(&searcher, &hay, &needle);
        }
    }
}
