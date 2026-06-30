//! Differential tests for ClusteredEliasFano: every `get`/`next_geq` result is
//! checked against a plain `Vec<u64>` oracle across sparse/clustered/dense/mixed
//! inputs and edge cases. These encode the invariant that the clustered encoding
//! is a *lossless, behavior-identical* representation regardless of which
//! per-chunk container is chosen.

use super::{ChunkKind, ClusteredEliasFano};

/// Oracle: first (index, value) with value >= target.
fn oracle_next_geq(v: &[u64], target: u64) -> Option<(usize, u64)> {
    let i = v.partition_point(|&x| x < target);
    if i < v.len() {
        Some((i, v[i]))
    } else {
        None
    }
}

/// Check get + next_geq against the oracle for many probes.
fn assert_matches_oracle(values: &[u64]) {
    let cef = ClusteredEliasFano::from_sorted_u64(values);
    assert_eq!(cef.len(), values.len());

    // get(i) for every i.
    for (i, &want) in values.iter().enumerate() {
        assert_eq!(cef.get(i), Some(want), "get({i})");
    }
    assert_eq!(cef.get(values.len()), None, "get past end");

    if values.is_empty() {
        assert_eq!(cef.next_geq(0), None);
        return;
    }

    let max = *values.last().unwrap();
    // Probe set: every value, value±1, 0, max+1, and a sweep.
    let mut probes: Vec<u64> = vec![0, max, max + 1];
    for &v in values {
        probes.push(v);
        probes.push(v.saturating_sub(1));
        probes.push(v + 1);
    }
    let step = (max / 97).max(1);
    let mut t = 0u64;
    while t <= max + 2 {
        probes.push(t);
        t += step;
    }
    for &t in &probes {
        assert_eq!(
            cef.next_geq(t),
            oracle_next_geq(values, t),
            "next_geq({t}) on len={}",
            values.len()
        );
    }
}

fn gen_sparse(n: usize, seed: u64) -> Vec<u64> {
    let mut x = seed | 1;
    let mut cur = 0u64;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            cur += 1 + (x % 200);
            cur
        })
        .collect()
}

fn gen_clustered(n: usize, seed: u64) -> Vec<u64> {
    let mut x = seed | 1;
    let mut out = Vec::with_capacity(n);
    let mut cur = 0u64;
    while out.len() < n {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        let run = (50 + (x % 150)) as usize;
        for _ in 0..run.min(n - out.len()) {
            out.push(cur);
            cur += 1;
        }
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        cur += 5_000 + (x % 45_000);
    }
    out
}

#[test]
fn empty() {
    assert_matches_oracle(&[]);
}

#[test]
fn single() {
    assert_matches_oracle(&[42]);
}

#[test]
fn two_elements() {
    assert_matches_oracle(&[5, 9]);
}

#[test]
fn fully_dense_one_chunk() {
    let v: Vec<u64> = (0..128).collect();
    assert_matches_oracle(&v);
    // A perfectly contiguous chunk must be encoded as a Run.
    let cef = ClusteredEliasFano::from_sorted_u64(&v);
    assert_eq!(cef.chunk_kinds(), vec![ChunkKind::Run]);
}

#[test]
fn fully_dense_multi_chunk() {
    let v: Vec<u64> = (1000..1000 + 500).collect();
    assert_matches_oracle(&v);
    let cef = ClusteredEliasFano::from_sorted_u64(&v);
    // 500 elements / 128 = 4 chunks; all contiguous → all Run.
    assert!(cef.chunk_kinds().iter().all(|&k| k == ChunkKind::Run));
}

#[test]
fn dense_bitmap_chunk() {
    // 100 elements spread over universe 200: too dense for EF to beat a bitmap,
    // not contiguous → should pick Bitmap.
    let v: Vec<u64> = (0..100).map(|i| i * 2).collect();
    assert_matches_oracle(&v);
    let cef = ClusteredEliasFano::from_sorted_u64(&v);
    assert!(
        cef.chunk_kinds().contains(&ChunkKind::Bitmap),
        "expected a Bitmap chunk, got {:?}",
        cef.chunk_kinds()
    );
}

#[test]
fn exactly_chunk_boundary() {
    for n in [127usize, 128, 129, 255, 256, 257] {
        let v: Vec<u64> = (0..n as u64).map(|i| i * 7).collect();
        assert_matches_oracle(&v);
    }
}

#[test]
fn sparse_random() {
    for seed in [1u64, 2, 12345, 0xDEAD_BEEF] {
        assert_matches_oracle(&gen_sparse(1000, seed));
    }
}

#[test]
fn clustered_random() {
    for seed in [1u64, 7, 99, 0xCAFE] {
        assert_matches_oracle(&gen_clustered(2000, seed));
    }
}

#[test]
fn mixed_distribution() {
    // Concatenate dense run + sparse tail + another dense run (strictly increasing).
    let mut v: Vec<u64> = (0..200).collect();
    let mut cur = 10_000u64;
    for k in 0..200 {
        cur += 1 + (k % 300);
        v.push(cur);
    }
    let base = *v.last().unwrap() + 1000;
    v.extend(base..base + 200);
    assert_matches_oracle(&v);
    let kinds = ClusteredEliasFano::from_sorted_u64(&v).chunk_kinds();
    // Should exercise more than one container kind.
    let distinct: std::collections::HashSet<_> = kinds.iter().collect();
    assert!(distinct.len() >= 2, "expected mixed kinds, got {kinds:?}");
}

#[test]
fn u32_builder_matches() {
    let v32: Vec<u32> = (0..500u32).map(|i| i * 3).collect();
    let v64: Vec<u64> = v32.iter().map(|&x| x as u64).collect();
    let a = ClusteredEliasFano::from_sorted(&v32);
    let b = ClusteredEliasFano::from_sorted_u64(&v64);
    assert_eq!(a.len(), b.len());
    for i in 0..v32.len() {
        assert_eq!(a.get(i), b.get(i));
    }
}

/// Oracle: count of common elements in two sorted slices.
fn oracle_intersect_count(a: &[u64], b: &[u64]) -> usize {
    let (mut i, mut j, mut c) = (0, 0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                c += 1;
                i += 1;
                j += 1;
            }
        }
    }
    c
}

fn assert_intersect_matches(a: &[u64], b: &[u64]) {
    let ca = ClusteredEliasFano::from_sorted_u64(a);
    let cb = ClusteredEliasFano::from_sorted_u64(b);
    let want = oracle_intersect_count(a, b);
    assert_eq!(ca.intersect_count(&cb), want, "a∩b");
    assert_eq!(cb.intersect_count(&ca), want, "b∩a (symmetry)");
}

#[test]
fn intersect_dense_run_paths() {
    let a: Vec<u64> = (0..1000).collect();
    let b: Vec<u64> = (500..1500).collect();
    assert_intersect_matches(&a, &b); // overlap 500..1000 → 500
    assert_intersect_matches(&a, &a); // self → 1000
    assert_intersect_matches(&a, &(2000..3000).collect::<Vec<_>>()); // disjoint → 0
}

#[test]
fn intersect_mixed_and_random() {
    for seed in [3u64, 17, 0xBEEF] {
        let a = gen_clustered(1500, seed);
        let b = gen_clustered(1500, seed ^ 0xFF);
        assert_intersect_matches(&a, &b);
        let s = gen_sparse(800, seed);
        let d: Vec<u64> = (100..900).map(|i| i * 2).collect();
        assert_intersect_matches(&s, &d);
    }
}

#[test]
fn intersect_edge_cases() {
    assert_intersect_matches(&[], &[1, 2, 3]);
    assert_intersect_matches(&[5], &[5]);
    assert_intersect_matches(&[5], &[6]);
    assert_intersect_matches(&[1, 2, 3], &[]);
}

#[test]
fn space_beats_opef_on_dense() {
    // On fully-dense data, Run containers carry ~0 payload, so clustered EF must
    // be no worse than OPEF (the strongest existing encoder) and far below the
    // 8 bytes/elem of a raw array. The exact margin is reported by the benchmark.
    use super::super::optimal::OptimalPartitionedEliasFano;
    let v: Vec<u64> = (0..10_000).collect();
    let cef = ClusteredEliasFano::from_sorted_u64(&v);
    let opef = OptimalPartitionedEliasFano::from_sorted_u64(&v);
    assert!(
        cef.bits_per_element() <= opef.bits_per_element(),
        "dense: CEF {} > OPEF {}",
        cef.bits_per_element(),
        opef.bits_per_element()
    );
    assert!(cef.bits_per_element() < 8.0);
}
