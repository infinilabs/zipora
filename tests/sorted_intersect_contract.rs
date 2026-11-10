//! Contract tests for `zipora::algorithms::simd_set_intersect`.
//!
//! The functions require strictly ascending inputs (sets, e.g. DocID posting
//! lists). On such inputs every strategy must equal a reference merge. Inputs
//! that break the precondition (duplicates, unsorted) give an unspecified
//! result, but every call must still return.

use std::sync::mpsc;
use std::time::Duration;

use zipora::algorithms::simd_set_intersect::{
    sorted_intersect_adaptive, sorted_intersect_count, sorted_intersect_galloping,
    sorted_intersect_simd,
};
use zipora::algorithms::simd_set_union::sorted_union_count;

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

    /// A strictly ascending set of at most `max_len` values in `base..base + universe`.
    fn set(&mut self, max_len: u64, universe: u64, base: u32) -> Vec<u32> {
        let len = self.below(max_len + 1);
        let mut v: Vec<u32> = (0..len)
            .map(|_| base + self.below(universe) as u32)
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }
}

fn reference(a: &[u32], b: &[u32]) -> Vec<u32> {
    let (mut i, mut j, mut out) = (0, 0, Vec::new());
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    out
}

/// Every public entry point against the reference merge, for one ordered pair of sets.
fn check_sets(a: &[u32], b: &[u32]) {
    let want = reference(a, b);
    let mut out = Vec::new();

    assert_eq!(
        sorted_intersect_adaptive(a, b, &mut out),
        want.len(),
        "adaptive count {a:?} {b:?}"
    );
    assert_eq!(out, want, "adaptive {a:?} {b:?}");

    assert_eq!(
        sorted_intersect_simd(a, b, &mut out),
        want.len(),
        "simd count {a:?} {b:?}"
    );
    assert_eq!(out, want, "simd {a:?} {b:?}");

    assert_eq!(
        sorted_intersect_galloping(a, b, &mut out),
        want.len(),
        "galloping count {a:?} {b:?}"
    );
    assert_eq!(out, want, "galloping {a:?} {b:?}");

    assert_eq!(
        sorted_intersect_count(a, b),
        want.len(),
        "count {a:?} {b:?}"
    );
    assert_eq!(
        sorted_union_count(a, b),
        a.len() + b.len() - want.len(),
        "union count {a:?} {b:?}"
    );
}

/// All ordered pairs of subsets of a 10-value universe, at the bottom and at
/// the top of the `u32` range. Lengths 0..=10 cover empty inputs, the scalar
/// path (fewer than 8 elements), 4-lane SIMD blocks and every tail length.
#[test]
fn sets_match_reference_exhaustive_small_universe() {
    for base in [0u32, u32::MAX - 9] {
        let subsets: Vec<Vec<u32>> = (0u32..1 << 10)
            .map(|mask| {
                (0..10)
                    .filter(|k| mask >> k & 1 == 1)
                    .map(|k| base + k)
                    .collect()
            })
            .collect();
        for a in &subsets {
            for b in &subsets {
                check_sets(a, b);
            }
        }
    }
}

/// Random sets long enough for 16-lane blocks, with dense and sparse overlap,
/// including values next to `u32::MAX`.
#[test]
fn sets_match_reference_random_balanced() {
    let mut rng = Rng(0x5EED_0001);
    for it in 0..20_000u64 {
        let universe = [24u64, 64, 300, 4096, 1 << 20][(it % 5) as usize];
        let base = if it % 7 == 0 {
            u32::MAX - universe as u32
        } else {
            0
        };
        let a = rng.set(600, universe, base);
        let b = rng.set(600, universe, base);
        check_sets(&a, &b);
        check_sets(&b, &a);
    }
}

/// Size ratios above 32 take the galloping strategy in the adaptive entry points.
#[test]
fn sets_match_reference_random_skewed() {
    let mut rng = Rng(0x5EED_0002);
    for it in 0..5_000u64 {
        let universe = [512u64, 4096, 1 << 16][(it % 3) as usize];
        let small = rng.set(12, universe, 0);
        let large = rng.set(4000, universe, 0);
        check_sets(&small, &large);
        check_sets(&large, &small);
    }
}

fn returns_within(label: &str, secs: u64, f: impl FnOnce() + Send + 'static) {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        f();
        let _ = tx.send(());
    });
    assert!(
        rx.recv_timeout(Duration::from_secs(secs)).is_ok(),
        "{label}: a call did not return within {secs} s on input that violates the precondition"
    );
}

fn call_all(a: &[u32], b: &[u32]) {
    let mut out = Vec::new();
    let n = sorted_intersect_adaptive(a, b, &mut out);
    assert_eq!(n, out.len());
    let n = sorted_intersect_simd(a, b, &mut out);
    assert_eq!(n, out.len());
    let n = sorted_intersect_galloping(a, b, &mut out);
    assert_eq!(n, out.len());
    let _ = sorted_intersect_count(a, b);
    let _ = sorted_union_count(a, b);
}

/// Inputs on which the per-step duplicate guard tried for 5.0.2 never
/// returned (4-, 8- and 16-lane shapes).
#[test]
fn unsorted_inputs_return_known_shapes() {
    returns_within("4 lanes", 10, || call_all(&[9, 9, 5, 1], &[9, 10, 11, 12]));
    returns_within("8 lanes", 10, || {
        call_all(&[9, 9, 5, 1, 0, 0, 0, 0], &(9..17).collect::<Vec<u32>>())
    });
    returns_within("16 lanes", 10, || {
        let a: Vec<u32> = [9u32, 9, 5, 1].into_iter().chain([0; 12]).collect();
        call_all(&a, &(9..25).collect::<Vec<u32>>())
    });
}

/// Random inputs with duplicates and without any order. The result is
/// unspecified; the calls must return and report the length they wrote.
#[test]
fn unsorted_and_duplicate_inputs_return_random() {
    returns_within("random", 60, || {
        let mut rng = Rng(0x5EED_0003);
        for it in 0..200_000u64 {
            let max_len = [6u64, 20, 70, 300][(it % 4) as usize];
            let values = [3u64, 12, 100][(it % 3) as usize];
            let (la, lb) = (rng.below(max_len + 1), rng.below(max_len + 1));
            let mut a: Vec<u32> = (0..la).map(|_| rng.below(values) as u32).collect();
            let mut b: Vec<u32> = (0..lb).map(|_| rng.below(values) as u32).collect();
            if it % 2 == 0 {
                // sorted, with duplicates
                a.sort_unstable();
                b.sort_unstable();
            }
            call_all(&a, &b);
        }
    });
}
