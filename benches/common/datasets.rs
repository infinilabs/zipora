//! Deterministic dataset generators and Feistel bijective query permutations.

#![allow(dead_code)]

use std::hint::black_box;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use zipora::succinct::rank_select::RankSelectOps;

use super::SEED_BASE;

#[inline(always)]
pub fn splitmix64_at(seed: u64, idx: usize) -> u64 {
    let mut z = seed.wrapping_add((idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline(always)]
pub fn uniform_index_at(seed: u64, idx: usize, range: usize) -> usize {
    ((splitmix64_at(seed, idx) as u128 * range as u128) >> 64) as usize
}

/// Exact 2-round Feistel-like bijection over `0..n` (when `n` is a multiple of `1,000`)
/// backed by L1-resident permutation tables (`<= 16 KiB` total).
///
/// Visits all `n` (`1,000,000` at spec size) keys in the structure in a uniform
/// pseudo-random order with zero replayed samples, while stratifying every `m`-key
/// sub-slice across all `m` (`1,000`) outer buckets `q2 in 0..m`.
pub struct BijectivePerm {
    pub p1: [u32; 1000],
    pub p2: Vec<u32>,
    pub p3: [u32; 1000],
    pub p4: Vec<u32>,
}

impl BijectivePerm {
    pub fn new(n: usize, seed: u64) -> Self {
        assert!(
            n.is_multiple_of(1000) && n > 0,
            "n must be a positive multiple of 1000"
        );
        let m = n / 1000;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut p1 = [0u32; 1000];
        for (i, slot) in p1.iter_mut().enumerate() {
            *slot = i as u32;
        }
        p1.shuffle(&mut rng);

        let mut p2: Vec<u32> = (0..m as u32).collect();
        p2.shuffle(&mut rng);

        let mut p3 = [0u32; 1000];
        for slot in &mut p3 {
            *slot = rng.random_range(0..m as u32);
        }

        let mut p4: Vec<u32> = (0..m).map(|_| rng.random_range(0..1000u32)).collect();
        p4.shuffle(&mut rng);

        Self { p1, p2, p3, p4 }
    }

    #[inline(always)]
    pub fn for_each_slice<F: FnMut(usize)>(&self, start: usize, count: usize, mut f: F) {
        let m = self.p2.len();
        let r_steps = (count / m).max(1);
        let mut r = (start / m) % 1000;
        for _ in 0..r_steps {
            let mut q_idx = self.p3[r] as usize;
            for _ in 0..m {
                let q2 = self.p2[q_idx] as usize;
                let mut r_idx = r + self.p4[q2] as usize;
                if r_idx >= 1000 {
                    r_idx -= 1000;
                }
                let r2 = self.p1[r_idx] as usize;
                if m == 1000 {
                    f(r2 * 1000 + q2);
                } else {
                    f(q2 * 1000 + r2);
                }
                q_idx += 1;
                if q_idx == m {
                    q_idx = 0;
                }
            }
            r += 1;
            if r == 1000 {
                r = 0;
            }
        }
    }
}

pub fn gen_rank_words(num_words: usize) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(SEED_BASE ^ 0x01);
    (0..num_words).map(|_| rng.random::<u64>()).collect()
}

pub fn warm_rank_pages<R: RankSelectOps>(rs: &R, num_bits: usize) {
    let mut acc = 0usize;
    for bit in (0..num_bits).step_by(4096) {
        acc = acc.wrapping_add(rs.rank1(bit));
    }
    black_box(acc);
}

pub fn warm_select1_pages<R: RankSelectOps>(rs: &R, max_ones: usize) {
    let mut acc = 0usize;
    for k in (0..max_ones).step_by(4096) {
        acc = acc.wrapping_add(rs.select1(k).unwrap_or(0));
    }
    black_box(acc);
}

pub fn warm_select0_pages<R: RankSelectOps>(rs: &R, max_zeros: usize) {
    let mut acc = 0usize;
    for k in (0..max_zeros).step_by(4096) {
        acc = acc.wrapping_add(rs.select0(k).unwrap_or(0));
    }
    black_box(acc);
}

pub fn gen_uniform_sparse(n: usize, avg_gap: u32, seed: u64) -> Vec<u32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n);
    let mut cur: u32 = 0;
    for _ in 0..n {
        cur = cur.saturating_add(rng.random_range(1..=2 * avg_gap));
        out.push(cur);
    }
    out
}

pub fn gen_zipf_clustered(n: usize, seed: u64) -> Vec<u32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n);
    let mut cur: u32 = 0;
    while out.len() < n {
        let run = rng.random_range(50..=200).min(n - out.len());
        for _ in 0..run {
            out.push(cur);
            cur = cur.saturating_add(1);
        }
        cur = cur.saturating_add(rng.random_range(5_000..=50_000));
    }
    out
}

pub const TRIE_RAND_SEED: u64 = SEED_BASE ^ 0x05;

#[inline(always)]
pub fn make_url_path_key(id: usize) -> [u8; 42] {
    let mut out = *b"org/infinilabs/mod_00/sub_000/item_0000000";
    let m = (id & 31) as u32;
    out[19] = b'0' + (m / 10) as u8;
    out[20] = b'0' + (m % 10) as u8;
    let s = ((id >> 5) & 255) as u32;
    out[26] = b'0' + (s / 100) as u8;
    out[27] = b'0' + ((s / 10) % 10) as u8;
    out[28] = b'0' + (s % 10) as u8;
    let mut v = id as u32;
    out[41] = b'0' + (v % 10) as u8;
    v /= 10;
    out[40] = b'0' + (v % 10) as u8;
    v /= 10;
    out[39] = b'0' + (v % 10) as u8;
    v /= 10;
    out[38] = b'0' + (v % 10) as u8;
    v /= 10;
    out[37] = b'0' + (v % 10) as u8;
    v /= 10;
    out[36] = b'0' + (v % 10) as u8;
    v /= 10;
    out[35] = b'0' + (v % 10) as u8;
    out
}

#[inline(always)]
pub fn make_random_16b_key(id: usize) -> [u8; 16] {
    const ALPHABET: &[u8; 36] = b"abcdefghijklmnopqrstuvwxyz0123456789";
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut z0 = TRIE_RAND_SEED ^ (id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z0 = (z0 ^ (z0 >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z0 ^= z0 >> 27;
    let mut z1 = z0.wrapping_add(0x94D0_49BB_1331_11EB);
    z1 = (z1 ^ (z1 >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z1 ^= z1 >> 27;

    let mut key = [0u8; 16];
    for i in 0..5 {
        key[i] = ALPHABET[((z0 >> (i * 12)) as usize) % 36];
        key[i + 5] = ALPHABET[((z1 >> (i * 12)) as usize) % 36];
    }
    let v = id as u32;
    key[10] = HEX[((v >> 20) & 0xf) as usize];
    key[11] = HEX[((v >> 16) & 0xf) as usize];
    key[12] = HEX[((v >> 12) & 0xf) as usize];
    key[13] = HEX[((v >> 8) & 0xf) as usize];
    key[14] = HEX[((v >> 4) & 0xf) as usize];
    key[15] = HEX[(v & 0xf) as usize];
    key
}

#[inline(always)]
pub fn make_shared_prefix_key(id: usize) -> [u8; 32] {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut key = *b"common_prefix_header_v1_00000000";
    let v = id as u32;
    key[24] = HEX[((v >> 28) & 0xf) as usize];
    key[25] = HEX[((v >> 24) & 0xf) as usize];
    key[26] = HEX[((v >> 20) & 0xf) as usize];
    key[27] = HEX[((v >> 16) & 0xf) as usize];
    key[28] = HEX[((v >> 12) & 0xf) as usize];
    key[29] = HEX[((v >> 8) & 0xf) as usize];
    key[30] = HEX[((v >> 4) & 0xf) as usize];
    key[31] = HEX[(v & 0xf) as usize];
    key
}

pub struct PackedKeySet {
    pub data: Vec<u8>,
    pub key_len: usize,
}

impl PackedKeySet {
    pub fn from_sorted_fixed<const N: usize>(keys: &[[u8; N]]) -> Self {
        let mut data = Vec::with_capacity(keys.len() * N);
        for k in keys {
            data.extend_from_slice(k);
        }
        Self { data, key_len: N }
    }

    pub fn slices(&self) -> Vec<&[u8]> {
        self.data.chunks_exact(self.key_len).collect()
    }
}

pub fn gen_trie_distribution(dist_idx: usize, n: usize) -> (&'static str, PackedKeySet) {
    match dist_idx {
        0 => {
            let mut paths: Vec<[u8; 42]> = (0..n).map(make_url_path_key).collect();
            paths.sort_unstable();
            paths.dedup();
            ("url_paths", PackedKeySet::from_sorted_fixed(&paths))
        }
        1 => {
            let mut random_16b: Vec<[u8; 16]> = (0..n).map(make_random_16b_key).collect();
            random_16b.sort_unstable();
            random_16b.dedup();
            ("random_16b", PackedKeySet::from_sorted_fixed(&random_16b))
        }
        _ => {
            let mut shared_prefix: Vec<[u8; 32]> = (0..n).map(make_shared_prefix_key).collect();
            shared_prefix.sort_unstable();
            shared_prefix.dedup();
            (
                "shared_prefix",
                PackedKeySet::from_sorted_fixed(&shared_prefix),
            )
        }
    }
}

#[repr(C, align(64))]
pub struct AlignedKey<const N: usize>(pub [u8; N]);

#[inline(always)]
pub fn contains_trie_key_by_id<F: FnMut(&[u8]) -> bool>(
    dist_idx: usize,
    id: usize,
    mut contains: F,
) -> bool {
    match dist_idx {
        0 => {
            let k = AlignedKey(make_url_path_key(id));
            contains(black_box(&k.0[..]))
        }
        1 => {
            let k = AlignedKey(make_random_16b_key(id));
            contains(black_box(&k.0[..]))
        }
        _ => {
            let k = AlignedKey(make_shared_prefix_key(id));
            contains(black_box(&k.0[..]))
        }
    }
}

#[inline(always)]
pub fn make_u64_key(id: usize) -> u64 {
    splitmix64_at(SEED_BASE ^ 0x06, id)
}

#[inline(always)]
pub fn make_str16_key(id: usize) -> [u8; 16] {
    let lo = splitmix64_at(SEED_BASE ^ 0x0610, id) | 0x2020_2020_2020_2020;
    let hi = (id as u64) ^ 0x6b65_795f_0000_0000;
    let mut buf = [0u8; 16];
    buf[..8].copy_from_slice(&lo.to_le_bytes());
    buf[8..].copy_from_slice(&hi.to_le_bytes());
    buf
}

/// Zipf-skewed English words and MediaWiki markup tokens.
pub fn gen_enwik_like_block(size: usize, seed: u64) -> Vec<u8> {
    const TOKENS: &[&[u8]] = &[
        b"the ",
        b"of ",
        b"and ",
        b"in ",
        b"to ",
        b"a ",
        b"is ",
        b"for ",
        b"that ",
        b"with ",
        b"on ",
        b"as ",
        b"are ",
        b"by ",
        b"this ",
        b"be ",
        b"from ",
        b"or ",
        b"an ",
        b"at ",
        b"[[Category:",
        b"Wikipedia",
        b"http://",
        b"archive",
        b"reference",
        b"section",
        b"compression",
        b"succinct",
        b"automaton",
        b"structure",
        b"algorithm",
        b"performance ",
    ];
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(size + 64);
    while out.len() < size {
        let r = rng.random_range(0..100usize);
        let idx = if r < 55 {
            rng.random_range(0..8)
        } else if r < 85 {
            rng.random_range(8..20)
        } else {
            rng.random_range(20..TOKENS.len())
        };
        out.extend_from_slice(TOKENS[idx]);
    }
    out.truncate(size);
    out
}

#[repr(C, align(4096))]
pub struct AlignedHaystacks(pub [[u8; 65_536]; 4]);

#[repr(C, align(64))]
pub struct AlignedNeedle(pub [u8; 16]);

pub static SEARCH_HAYSTACKS: AlignedHaystacks = {
    let mut buf = [b'a'; 65_536];
    let needle = *b"needle_target";
    let start = 65_536 - 32;
    let mut i = 0;
    while i < 13 {
        buf[start + i] = needle[i];
        i += 1;
    }
    buf[65_536 - 5] = b'Z';
    AlignedHaystacks([buf, buf, buf, buf])
};

pub static SEARCH_NEEDLE: AlignedNeedle = AlignedNeedle(*b"needle_target\0\0\0");

pub fn gen_radix_u32_template(n: usize) -> Vec<u32> {
    let mut template_u32 = vec![0u32; n];
    let mut s = SEED_BASE ^ 0x0901;
    for slot in template_u32.iter_mut() {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        *slot = (z ^ (z >> 29)) as u32;
    }
    template_u32
}

pub fn gen_radix_u64_template(n: usize) -> Vec<u64> {
    let mut template_u64 = vec![0u64; n];
    let mut s = SEED_BASE ^ 0x0902;
    for slot in template_u64.iter_mut() {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        *slot = z ^ (z >> 27);
    }
    template_u64
}

pub fn gen_radix_kv_template(n: usize) -> Vec<(u64, u64)> {
    let mut template_kv = vec![(0u64, 0u64); n];
    let mut s = SEED_BASE ^ 0x0903;
    for (i, slot) in template_kv.iter_mut().enumerate() {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        *slot = (z ^ (z >> 27), i as u64);
    }
    template_kv
}

use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

struct PreallocBenchBuffers {
    record_pool: Vec<(String, String, String, Vec<f64>)>,
    space_pool: Vec<(String, f64)>,
    space_used: usize,
}

static PREALLOC_BUFFERS: OnceLock<Mutex<PreallocBenchBuffers>> = OnceLock::new();

fn get_prealloc_buffers() -> &'static Mutex<PreallocBenchBuffers> {
    PREALLOC_BUFFERS.get_or_init(|| {
        let mut record_pool = Vec::with_capacity(512);
        for _ in 0..512 {
            record_pool.push((
                String::with_capacity(128),
                String::with_capacity(64),
                String::with_capacity(32),
                Vec::with_capacity(32),
            ));
        }
        let mut space_pool = Vec::with_capacity(256);
        for _ in 0..256 {
            space_pool.push((String::with_capacity(128), 0.0));
        }
        Mutex::new(PreallocBenchBuffers {
            record_pool,
            space_pool,
            space_used: 0,
        })
    })
}

/// Pops a pre-allocated `(name, category, unit, samples)` buffer tuple from the bottom-of-`brk`
/// pool initialized by [`init_bench_allocator_policy`], so `measure_bench_inner` performs zero
/// heap allocations while a benchmark's multi-megabyte data structure is alive on `brk`.
#[inline(never)]
pub fn take_prealloc_record_buffers(
    name: &str,
    category: &str,
    unit: &str,
) -> (String, String, String, Vec<f64>) {
    init_bench_allocator_policy();
    let mut guard = get_prealloc_buffers().lock().unwrap();
    if let Some((mut n_buf, mut c_buf, mut u_buf, mut s_buf)) = guard.record_pool.pop() {
        n_buf.clear();
        assert!(n_buf.capacity() >= name.len());
        n_buf.push_str(name);
        c_buf.clear();
        assert!(c_buf.capacity() >= category.len());
        c_buf.push_str(category);
        u_buf.clear();
        assert!(u_buf.capacity() >= unit.len());
        u_buf.push_str(unit);
        s_buf.clear();
        (n_buf, c_buf, u_buf, s_buf)
    } else {
        (
            name.to_string(),
            category.to_string(),
            unit.to_string(),
            Vec::with_capacity(32),
        )
    }
}

/// Records a `(key, val)` space metric into the pre-allocated bottom-of-`brk` buffer pool
/// without allocating `String` or `BTreeMap` nodes while benchmark data structures are on `brk`.
#[inline(never)]
pub fn record_prealloc_space_metric(space: &mut BTreeMap<String, f64>, key: &str, val: f64) {
    init_bench_allocator_policy();
    let mut guard = get_prealloc_buffers().lock().unwrap();
    let idx = guard.space_used;
    if idx < guard.space_pool.len() {
        let (k_buf, v_slot) = &mut guard.space_pool[idx];
        k_buf.clear();
        assert!(k_buf.capacity() >= key.len());
        k_buf.push_str(key);
        *v_slot = val;
        guard.space_used = idx + 1;
    } else {
        drop(guard);
        space.insert(key.to_string(), val);
    }
}

/// Flushes all pre-allocated space metrics recorded via [`record_prealloc_space_metric`] into
/// the output `BTreeMap` after all benchmarks have completed.
#[inline(never)]
pub fn flush_prealloc_space_metrics(space: &mut BTreeMap<String, f64>) {
    if let Some(m) = PREALLOC_BUFFERS.get() {
        let mut guard = m.lock().unwrap();
        let used = guard.space_used;
        for i in 0..used {
            let (ref k, v) = guard.space_pool[i];
            space.insert(k.clone(), v);
        }
        guard.space_used = 0;
    }
}

/// Configures a single fixed glibc allocator policy at the start of every benchmark binary:
/// - `libc::mallopt(libc::M_MMAP_MAX, 0)` disables `mmap()`-backed heap allocations for all
///   sizes (including `> 32 MiB` buffers in `dat/*/build_sorted`, `radix_sort/*`, and
///   `suffix_array/*`), routing all system-allocator requests through `brk`.
/// - `libc::mallopt(libc::M_TRIM_THRESHOLD, -1)` disables automatic `systrim()` (`sbrk(-delta)`)
///   inside `free()` during timed benchmark loops, so pages faulted during `warmup_calls`
///   remain mapped across all timed trials in both full-suite and isolated `--filter` runs.
/// - Pre-allocates all `BenchRecord` and `space_metrics` buffers at the bottom of `brk` so
///   `measure_bench_inner` and `record_space_metric` never allocate above a benchmark's working
///   set or pin `av->top` across `malloc_trim(0)`.
pub fn init_bench_allocator_policy() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        use std::sync::Once;
        static INIT_POLICY: Once = Once::new();
        INIT_POLICY.call_once(|| {
            // SAFETY: `libc::mallopt` sets global glibc ptmalloc2 parameters using valid
            // documented option constants (`M_MMAP_MAX`, `M_TRIM_THRESHOLD`).
            unsafe {
                libc::mallopt(libc::M_MMAP_MAX, 0);
                libc::mallopt(libc::M_TRIM_THRESHOLD, -1);
            }
        });
    }
    let _ = get_prealloc_buffers();
}

/// Consolidates glibc `fastbins` and returns unused top-of-heap / interior pages
/// to the OS via `libc::malloc_trim(0)` (after ensuring [`init_bench_allocator_policy`]
/// is active) so full-suite runs and isolated `--filter` retries start each multi-megabyte
/// benchmark from the same clean allocator state.
pub fn trim_allocator_arena() {
    init_bench_allocator_policy();
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        // SAFETY: `libc::malloc_trim(0)` is thread-safe and consolidates fastbins / releases
        // unused pages from glibc's arenas back to the OS between benchmarks.
        unsafe {
            libc::malloc_trim(0);
        }
    }
}
