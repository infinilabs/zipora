//! Structure-aware `arbitrary` fuzz target for `HybridPostingList` (`next_geq` and
//! `intersect_count` vs naive sorted-slice oracle, plan.md B2.1).
#![no_main]
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use zipora::succinct::elias_fano::{HybridPostingList, PostingEncoding};

#[derive(Arbitrary, Debug)]
struct PostingInput {
    list_a: Vec<u32>,
    list_b: Vec<u32>,
    queries: Vec<u32>,
    encoding_sel: u8,
}

fn normalize_sorted(mut v: Vec<u32>) -> Vec<u32> {
    v.truncate(1024);
    v.sort_unstable();
    v.dedup();
    v
}

fn naive_next_geq(sorted: &[u32], target: u64) -> Option<(usize, u64)> {
    if target > u32::MAX as u64 {
        return None;
    }
    let t = target as u32;
    let idx = sorted.partition_point(|&x| x < t);
    sorted.get(idx).map(|&val| (idx, val as u64))
}

fn naive_intersect_count(a: &[u32], b: &[u32]) -> usize {
    let (mut i, mut j, mut count) = (0usize, 0usize, 0usize);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                count += 1;
                i += 1;
                j += 1;
            }
        }
    }
    count
}

fuzz_target!(|input: PostingInput| {
    let a = normalize_sorted(input.list_a);
    let b = normalize_sorted(input.list_b);
    if a.is_empty() || b.is_empty() {
        return;
    }

    let enc = match input.encoding_sel % 6 {
        0 => None,
        1 => Some(PostingEncoding::Dense),
        2 => Some(PostingEncoding::EliasFano),
        3 => Some(PostingEncoding::Partitioned),
        4 => Some(PostingEncoding::Optimal),
        _ => Some(PostingEncoding::Clustered),
    };

    let pl_a = match enc {
        Some(e) => HybridPostingList::with_encoding(&a, e),
        None => HybridPostingList::from_sorted(&a),
    };
    let pl_b = match enc {
        Some(e) => HybridPostingList::with_encoding(&b, e),
        None => HybridPostingList::from_sorted(&b),
    };

    assert_eq!(pl_a.len(), a.len());
    assert_eq!(pl_b.len(), b.len());

    for (idx, &expected) in a.iter().enumerate().take(64) {
        assert_eq!(pl_a.get(idx), Some(expected as u64));
    }
    assert_eq!(pl_a.get(a.len()), None);

    for &q in input.queries.iter().take(64) {
        assert_eq!(
            pl_a.next_geq(q as u64),
            naive_next_geq(&a, q as u64),
            "next_geq mismatch on query {q}"
        );
    }
    assert_eq!(pl_a.next_geq(0), naive_next_geq(&a, 0));
    let last_a = *a.last().unwrap() as u64;
    assert_eq!(pl_a.next_geq(last_a), naive_next_geq(&a, last_a));
    assert_eq!(pl_a.next_geq(last_a + 1), None);

    assert_eq!(
        pl_a.intersect_count(&pl_b),
        naive_intersect_count(&a, &b),
        "intersect_count mismatch"
    );
});
