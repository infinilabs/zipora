//! Parallel merge of the sorted runs left behind by the chunked radix sorts.
//!
//! The parallel LSD paths of `AdvancedRadixSort` and of `RadixSort::sort_u32`
//! / `sort_u64` sort contiguous chunks independently and then have to merge
//! them. Until S20-R3 (`AdvancedRadixSort`) and S20-R3b (`RadixSort`) that
//! merge was a single-threaded k-way heap merge, which made the parallel sorts
//! slower than the sequential ones, and for `AdvancedRadixSort` reordered
//! equal keys from different chunks. The runs are now merged as a balanced
//! binary tree whose levels and individual pairwise merges are split across
//! rayon tasks, and the merge is stable.

/// Merge the sorted, contiguous `run_len`-long runs of `data` (the last run
/// may be shorter) into one sorted sequence. Stable: equal elements keep their
/// input order. No-op when `data` holds at most one run.
pub(super) fn merge_sorted_runs<T: Copy + Ord + Send + Sync>(data: &mut [T], run_len: usize) {
    if data.is_empty() || run_len == 0 || run_len >= data.len() {
        return;
    }
    let bounds: Vec<usize> = (0..data.len())
        .step_by(run_len)
        .chain(std::iter::once(data.len()))
        .collect();
    // Both buffers start out holding every run, so the leaves of the merge
    // tree never copy.
    let mut scratch: Vec<T> = {
        use rayon::prelude::*;
        data.par_iter().copied().collect()
    };
    merge_run_tree(data, &mut scratch, &bounds, true);
}

/// Merge the sorted runs `bounds[i]..bounds[i + 1]` (absolute offsets; `a` and
/// `b` both start at `bounds[0]`) into `a` when `into_a`, otherwise into `b`.
///
/// Precondition: every run is present in both `a` and `b`. A node's children
/// merge into the opposite buffer of the node's own target, so a leaf (single
/// run) is already where its parent reads it, and an ancestor overwrites a
/// region only after all of its descendants have finished reading it.
fn merge_run_tree<T: Copy + Ord + Send + Sync>(
    a: &mut [T],
    b: &mut [T],
    bounds: &[usize],
    into_a: bool,
) {
    let runs = bounds.len().saturating_sub(1);
    if runs <= 1 {
        return;
    }
    let mid = runs / 2;
    let split = bounds[mid] - bounds[0];
    {
        let (a_lo, a_hi) = a.split_at_mut(split);
        let (b_lo, b_hi) = b.split_at_mut(split);
        rayon::join(
            || merge_run_tree(a_lo, b_lo, &bounds[..=mid], !into_a),
            || merge_run_tree(a_hi, b_hi, &bounds[mid..], !into_a),
        );
    }
    if into_a {
        let (lo, hi) = b.split_at(split);
        par_merge(lo, hi, a);
    } else {
        let (lo, hi) = a.split_at(split);
        par_merge(lo, hi, b);
    }
}

/// Merges producing at most this many elements run sequentially.
const PAR_MERGE_SEQ_CUTOFF: usize = 1 << 13;

/// Stable merge of sorted `left` and `right` into `out`, where
/// `out.len() == left.len() + right.len()`.
fn par_merge<T: Copy + Ord + Send + Sync>(left: &[T], right: &[T], out: &mut [T]) {
    par_merge_with_cutoff(left, right, out, PAR_MERGE_SEQ_CUTOFF);
}

/// [`par_merge`] with an explicit sequential cutoff (clamped to at least 2).
///
/// Above the cutoff the larger input is cut at its midpoint and the other at
/// the matching partition point, and the two halves are merged in parallel.
/// Equal elements from `left` always precede those from `right`: cutting
/// `left` at pivot `p` sends the `right` elements `< p` to the lower half, and
/// cutting `right` at pivot `q` sends the `left` elements `<= q` there.
///
/// Termination: with `out.len() >= 3` the larger input has at least 2
/// elements, so its cut lies strictly inside it and both halves are smaller
/// than `out`; the clamp keeps `out.len() == 2` (a 1 + 1 merge, whose cut
/// could be empty) on the sequential path.
pub(super) fn par_merge_with_cutoff<T: Copy + Ord + Send + Sync>(
    left: &[T],
    right: &[T],
    out: &mut [T],
    cutoff: usize,
) {
    if out.len() <= cutoff.max(2) || left.is_empty() || right.is_empty() {
        seq_merge(left, right, out);
        return;
    }
    let (l_cut, r_cut) = if left.len() >= right.len() {
        let l_cut = left.len() / 2;
        let pivot = left[l_cut];
        (l_cut, right.partition_point(|x| *x < pivot))
    } else {
        let r_cut = right.len() / 2;
        let pivot = right[r_cut];
        (left.partition_point(|x| *x <= pivot), r_cut)
    };
    let (l_lo, l_hi) = left.split_at(l_cut);
    let (r_lo, r_hi) = right.split_at(r_cut);
    let (out_lo, out_hi) = out.split_at_mut(l_cut + r_cut);
    rayon::join(
        || par_merge_with_cutoff(l_lo, r_lo, out_lo, cutoff),
        || par_merge_with_cutoff(l_hi, r_hi, out_hi, cutoff),
    );
}

/// Sequential stable merge (ties are taken from `left`).
fn seq_merge<T: Copy + Ord>(left: &[T], right: &[T], out: &mut [T]) {
    let (mut i, mut j, mut k) = (0, 0, 0);
    while i < left.len() && j < right.len() {
        let take_right = right[j] < left[i];
        out[k] = if take_right { right[j] } else { left[i] };
        j += usize::from(take_right);
        i += usize::from(!take_right);
        k += 1;
    }
    let rest = &left[i..];
    out[k..k + rest.len()].copy_from_slice(rest);
    out[k + rest.len()..].copy_from_slice(&right[j..]);
}
