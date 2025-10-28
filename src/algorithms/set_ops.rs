//! Set Operations Library
//!
//! Complete implementation of set operations following the C++ reference implementation
//! from topling-zip/src/terark/set_op.hpp.
//!
//! This module provides efficient implementations of:
//! - Multiset operations (preserving duplicates)
//! - Unique set operations (removing duplicates)
//! - Adaptive algorithms that select optimal strategy based on input sizes
//!
//! # Performance Characteristics
//!
//! - `multiset_intersection`: O(n + m) linear scan
//! - `multiset_1small_intersection`: O(n * log(m)) binary search for small first set
//! - `multiset_fast_intersection`: Adaptive selection based on size ratio
//! - All operations are zero-allocation when possible
//!
//! # Examples
//!
//! ```
//! use zipora::algorithms::set_ops::*;
//!
//! let a = vec![1, 2, 2, 3, 4];
//! let b = vec![2, 2, 3, 5];
//!
//! // Multiset intersection - preserves duplicates
//! let result = multiset_intersection(&a, &b, |x, y| x.cmp(y));
//! assert_eq!(result, vec![2, 2, 3]);
//!
//! // Set intersection - unique elements only
//! let result = set_intersection(&a, &b, |x, y| x.cmp(y));
//! assert_eq!(result, vec![2, 3]);
//!
//! // Remove duplicates in-place
//! let mut data = vec![1, 1, 2, 2, 2, 3];
//! let new_len = set_unique_default(&mut data);
//! data.truncate(new_len);
//! assert_eq!(data, vec![1, 2, 3]);
//! ```

use std::cmp::Ordering;

//---------------------------------------------------------------
// Multiset Intersection Operations (copy from first sequence)
//---------------------------------------------------------------

/// Multiset intersection - result copied from first sequence
///
/// Performs a linear scan (O(n + m)) to find common elements between two sorted sequences.
/// When elements are equal, copies from the first sequence and does NOT increment second iterator,
/// allowing duplicates from first to be preserved.
///
/// # Arguments
///
/// * `first1` - First sorted sequence
/// * `first2` - Second sorted sequence
/// * `pred` - Comparison function returning Ordering
///
/// # Returns
///
/// Vector containing intersection elements copied from first sequence
///
/// # Examples
///
/// ```
/// use zipora::algorithms::set_ops::multiset_intersection;
///
/// let a = vec![1, 2, 2, 3, 4, 5];
/// let b = vec![2, 2, 3, 6, 7];
/// let result = multiset_intersection(&a, &b, |x, y| x.cmp(y));
/// assert_eq!(result, vec![2, 2, 3]);
/// ```
pub fn multiset_intersection<T, F>(first1: &[T], first2: &[T], pred: F) -> Vec<T>
where
    T: Clone,
    F: Fn(&T, &T) -> Ordering,
{
    let mut result = Vec::new();
    let mut i1 = 0;
    let mut i2 = 0;

    while i1 < first1.len() && i2 < first2.len() {
        match pred(&first1[i1], &first2[i2]) {
            Ordering::Less => i1 += 1,    // first1[i1] < first2[i2]
            Ordering::Greater => i2 += 1, // first1[i1] > first2[i2]
            Ordering::Equal => {
                result.push(first1[i1].clone());
                i1 += 1;
                // NOTE: Do NOT increment i2 - allows duplicates from first2
            }
        }
    }
    result
}

/// Multiset intersection optimized for small first sequence
///
/// Uses binary search (O(n * log(m))) when the first sequence is much smaller than the second.
/// For each element in first sequence, uses equal_range to find matching range in second sequence.
///
/// # Arguments
///
/// * `first1` - Small first sorted sequence
/// * `first2` - Large second sorted sequence
/// * `pred` - Comparison function returning Ordering
///
/// # Returns
///
/// Vector containing intersection elements copied from first sequence
///
/// # Performance
///
/// Best when `first1.len() << first2.len()` (e.g., size ratio > 32)
///
/// # Examples
///
/// ```
/// use zipora::algorithms::set_ops::multiset_1small_intersection;
///
/// let a = vec![2, 3, 5];  // Small set
/// let b = vec![1, 2, 2, 3, 3, 3, 4, 5, 5, 6];  // Large set
/// let result = multiset_1small_intersection(&a, &b, |x, y| x.cmp(y));
/// assert_eq!(result, vec![2, 3, 5]);
/// ```
pub fn multiset_1small_intersection<T, F>(first1: &[T], first2: &[T], pred: F) -> Vec<T>
where
    T: Clone,
    F: Fn(&T, &T) -> Ordering,
{
    let mut result = Vec::new();
    let mut i1 = 0;
    let mut i2 = 0;

    while i1 < first1.len() && i2 < first2.len() {
        // Binary search for range of equal elements in first2
        let range = equal_range(&first2[i2..], &first1[i1], &pred);

        if range.start == range.end {
            // No match found
            i1 += 1;
        } else {
            // Found matching range, copy all from first1 that match
            let range_start = &first2[i2 + range.start];
            while i1 < first1.len() && pred(&first1[i1], range_start) == Ordering::Equal {
                result.push(first1[i1].clone());
                i1 += 1;
            }
        }
        i2 += range.end;
    }
    result
}

/// Adaptive multiset intersection - selects best algorithm based on size ratio
///
/// Automatically chooses between linear scan and binary search approaches based on
/// the size ratio between the two sequences.
///
/// # Arguments
///
/// * `first1` - First sorted sequence
/// * `first2` - Second sorted sequence
/// * `pred` - Comparison function returning Ordering
/// * `threshold` - Size ratio threshold (default 32)
///
/// # Algorithm Selection
///
/// - If `first1.len() * threshold < first2.len()`: Uses binary search (1small variant)
/// - Otherwise: Uses linear scan
///
/// # Examples
///
/// ```
/// use zipora::algorithms::set_ops::multiset_fast_intersection;
///
/// let a = vec![1, 2, 3];
/// let b: Vec<i32> = (1..=100).collect();
/// let result = multiset_fast_intersection(&a, &b, |x, y| x.cmp(y), 32);
/// assert_eq!(result, vec![1, 2, 3]);
/// ```
pub fn multiset_fast_intersection<T, F>(
    first1: &[T],
    first2: &[T],
    pred: F,
    threshold: usize,
) -> Vec<T>
where
    T: Clone,
    F: Fn(&T, &T) -> Ordering,
{
    // Adaptive selection based on size ratio
    if first1.len() * threshold < first2.len() {
        multiset_1small_intersection(first1, first2, pred)
    } else {
        multiset_intersection(first1, first2, pred)
    }
}

//---------------------------------------------------------------
// Multiset Intersection2 Operations (copy from second sequence)
//---------------------------------------------------------------

/// Multiset intersection - result copied from second sequence
///
/// Similar to multiset_intersection but copies elements from the second sequence instead of first.
/// When elements are equal, copies from second sequence and does NOT increment first iterator.
///
/// # Arguments
///
/// * `first1` - First sorted sequence
/// * `first2` - Second sorted sequence
/// * `pred` - Comparison function returning Ordering
///
/// # Examples
///
/// ```
/// use zipora::algorithms::set_ops::multiset_intersection2;
///
/// let a = vec![1, 2, 2, 3];
/// let b = vec![2, 2, 2, 3, 4];
/// let result = multiset_intersection2(&a, &b, |x, y| x.cmp(y));
/// // Copies from second sequence: 2, 2, 2, 3
/// assert_eq!(result, vec![2, 2, 2, 3]);
/// ```
pub fn multiset_intersection2<T, F>(first1: &[T], first2: &[T], pred: F) -> Vec<T>
where
    T: Clone,
    F: Fn(&T, &T) -> Ordering,
{
    let mut result = Vec::new();
    let mut i1 = 0;
    let mut i2 = 0;

    while i1 < first1.len() && i2 < first2.len() {
        match pred(&first1[i1], &first2[i2]) {
            Ordering::Less => i1 += 1,    // first1[i1] < first2[i2]
            Ordering::Greater => i2 += 1, // first1[i1] > first2[i2]
            Ordering::Equal => {
                result.push(first2[i2].clone());
                i2 += 1;
                // NOTE: Do NOT increment i1 - allows duplicates from first1
            }
        }
    }
    result
}

/// Multiset intersection2 optimized for small first sequence
///
/// Uses binary search for each element in first sequence. Copies entire matching ranges
/// from the second sequence.
///
/// # Examples
///
/// ```
/// use zipora::algorithms::set_ops::multiset_1small_intersection2;
///
/// let a = vec![2, 3];
/// let b = vec![1, 2, 2, 2, 3, 3, 4];
/// let result = multiset_1small_intersection2(&a, &b, |x, y| x.cmp(y));
/// assert_eq!(result, vec![2, 2, 2, 3, 3]);
/// ```
pub fn multiset_1small_intersection2<T, F>(first1: &[T], first2: &[T], pred: F) -> Vec<T>
where
    T: Clone,
    F: Fn(&T, &T) -> Ordering,
{
    let mut result = Vec::new();
    let mut i1 = 0;
    let mut i2 = 0;

    while i1 < first1.len() && i2 < first2.len() {
        // Binary search for range of equal elements in first2
        let range = equal_range(&first2[i2..], &first1[i1], &pred);

        if range.start != range.end {
            // Copy all matching elements from second sequence
            for idx in (i2 + range.start)..(i2 + range.end) {
                result.push(first2[idx].clone());
            }
        }
        i2 += range.end;
        i1 += 1;
    }
    result
}

/// Adaptive multiset intersection2
///
/// # Examples
///
/// ```
/// use zipora::algorithms::set_ops::multiset_fast_intersection2;
///
/// let a = vec![1, 2];
/// let b: Vec<i32> = (1..=100).collect();
/// let result = multiset_fast_intersection2(&a, &b, |x, y| x.cmp(y), 32);
/// assert_eq!(result, vec![1, 2]);
/// ```
pub fn multiset_fast_intersection2<T, F>(
    first1: &[T],
    first2: &[T],
    pred: F,
    threshold: usize,
) -> Vec<T>
where
    T: Clone,
    F: Fn(&T, &T) -> Ordering,
{
    if first1.len() * threshold < first2.len() {
        multiset_1small_intersection2(first1, first2, pred)
    } else {
        multiset_intersection2(first1, first2, pred)
    }
}

//---------------------------------------------------------------
// Set Unique Operation
//---------------------------------------------------------------

/// Remove duplicate elements from sorted sequence (in-place)
///
/// Removes consecutive duplicate elements from a sequence, similar to C++ std::unique.
/// The predicate should return true when elements are equal.
///
/// # Arguments
///
/// * `data` - Mutable slice to remove duplicates from (must be sorted)
/// * `pred` - Equality predicate (returns true if elements are equal)
///
/// # Returns
///
/// New length of unique elements. Elements beyond this length are unspecified.
///
/// # Note
///
/// Following the C++ implementation pattern from lines 171-191 of set_op.hpp.
/// The algorithm uses swap-based movement to avoid extra allocations.
///
/// # Examples
///
/// ```
/// use zipora::algorithms::set_ops::set_unique;
///
/// let mut data = vec![1, 1, 2, 2, 2, 3, 4, 4, 5];
/// let new_len = set_unique(&mut data, |a, b| a == b);
/// data.truncate(new_len);
/// assert_eq!(data, vec![1, 2, 3, 4, 5]);
/// ```
pub fn set_unique<T, F>(data: &mut [T], pred: F) -> usize
where
    F: Fn(&T, &T) -> bool,
{
    if data.len() <= 1 {
        return data.len();
    }

    let mut write_pos = 0;
    let mut read_pos = 0;

    while read_pos < data.len() {
        if write_pos == 0 || !pred(&data[write_pos - 1], &data[read_pos]) {
            if write_pos != read_pos {
                // Move element (swap to avoid extra allocation)
                data.swap(write_pos, read_pos);
            }
            write_pos += 1;
        }
        read_pos += 1;
    }

    write_pos
}

/// Remove duplicates with default equality
///
/// # Examples
///
/// ```
/// use zipora::algorithms::set_ops::set_unique_default;
///
/// let mut data = vec![1, 1, 2, 3, 3, 3, 4];
/// let new_len = set_unique_default(&mut data);
/// data.truncate(new_len);
/// assert_eq!(data, vec![1, 2, 3, 4]);
/// ```
pub fn set_unique_default<T: PartialEq>(data: &mut [T]) -> usize {
    set_unique(data, |a, b| a == b)
}

//---------------------------------------------------------------
// Multiset Union Operation
//---------------------------------------------------------------

/// Multiset union - combines both sequences preserving duplicates
///
/// Merges two sorted sequences including all duplicates from both.
///
/// # Examples
///
/// ```
/// use zipora::algorithms::set_ops::multiset_union;
///
/// let a = vec![1, 2, 3, 4];
/// let b = vec![3, 4, 5, 6];
/// let result = multiset_union(&a, &b, |x, y| x.cmp(y));
/// assert_eq!(result, vec![1, 2, 3, 3, 4, 4, 5, 6]);
/// ```
pub fn multiset_union<T, F>(first1: &[T], first2: &[T], pred: F) -> Vec<T>
where
    T: Clone,
    F: Fn(&T, &T) -> Ordering,
{
    let mut result = Vec::with_capacity(first1.len() + first2.len());
    let mut i1 = 0;
    let mut i2 = 0;

    while i1 < first1.len() && i2 < first2.len() {
        match pred(&first1[i1], &first2[i2]) {
            Ordering::Less => {
                result.push(first1[i1].clone());
                i1 += 1;
            }
            Ordering::Greater => {
                result.push(first2[i2].clone());
                i2 += 1;
            }
            Ordering::Equal => {
                // Include both equal elements
                result.push(first1[i1].clone());
                result.push(first2[i2].clone());
                i1 += 1;
                i2 += 1;
            }
        }
    }

    // Copy remaining elements
    while i1 < first1.len() {
        result.push(first1[i1].clone());
        i1 += 1;
    }
    while i2 < first2.len() {
        result.push(first2[i2].clone());
        i2 += 1;
    }

    result
}

//---------------------------------------------------------------
// Multiset Difference Operation
//---------------------------------------------------------------

/// Multiset difference - elements in first but not in second
///
/// Returns elements from first sequence that are not present in second sequence.
///
/// # Examples
///
/// ```
/// use zipora::algorithms::set_ops::multiset_difference;
///
/// let a = vec![1, 2, 2, 3, 4];
/// let b = vec![2, 3, 3, 5];
/// let result = multiset_difference(&a, &b, |x, y| x.cmp(y));
/// assert_eq!(result, vec![1, 2, 4]);
/// ```
pub fn multiset_difference<T, F>(first1: &[T], first2: &[T], pred: F) -> Vec<T>
where
    T: Clone,
    F: Fn(&T, &T) -> Ordering,
{
    let mut result = Vec::new();
    let mut i1 = 0;
    let mut i2 = 0;

    while i1 < first1.len() && i2 < first2.len() {
        match pred(&first1[i1], &first2[i2]) {
            Ordering::Less => {
                result.push(first1[i1].clone());
                i1 += 1;
            }
            Ordering::Greater => {
                i2 += 1;
            }
            Ordering::Equal => {
                // Skip equal elements
                i1 += 1;
                i2 += 1;
            }
        }
    }

    // Copy remaining elements from first sequence
    while i1 < first1.len() {
        result.push(first1[i1].clone());
        i1 += 1;
    }

    result
}

//---------------------------------------------------------------
// Unique Set Operations
//---------------------------------------------------------------

/// Set intersection - unique elements only
///
/// Returns intersection with duplicates removed.
///
/// # Examples
///
/// ```
/// use zipora::algorithms::set_ops::set_intersection;
///
/// let a = vec![1, 2, 2, 3, 4];
/// let b = vec![2, 2, 3, 5];
/// let result = set_intersection(&a, &b, |x, y| x.cmp(y));
/// assert_eq!(result, vec![2, 3]);  // Unique only
/// ```
pub fn set_intersection<T, F>(first1: &[T], first2: &[T], pred: F) -> Vec<T>
where
    T: Clone + PartialEq,
    F: Fn(&T, &T) -> Ordering,
{
    let mut result = multiset_intersection(first1, first2, pred);
    let new_len = set_unique_default(&mut result);
    result.truncate(new_len);
    result
}

/// Set union - unique elements only
///
/// Returns union with duplicates removed.
///
/// # Examples
///
/// ```
/// use zipora::algorithms::set_ops::set_union;
///
/// let a = vec![1, 2, 2, 3];
/// let b = vec![2, 3, 4, 4];
/// let result = set_union(&a, &b, |x, y| x.cmp(y));
/// assert_eq!(result, vec![1, 2, 3, 4]);  // Unique only
/// ```
pub fn set_union<T, F>(first1: &[T], first2: &[T], pred: F) -> Vec<T>
where
    T: Clone + PartialEq,
    F: Fn(&T, &T) -> Ordering,
{
    let mut result = multiset_union(first1, first2, pred);
    let new_len = set_unique_default(&mut result);
    result.truncate(new_len);
    result
}

/// Set difference - unique elements only
///
/// Returns difference with duplicates removed.
///
/// # Examples
///
/// ```
/// use zipora::algorithms::set_ops::set_difference;
///
/// let a = vec![1, 2, 2, 3, 4, 4];
/// let b = vec![2, 3];
/// let result = set_difference(&a, &b, |x, y| x.cmp(y));
/// // multiset_difference removes one occurrence: [1, 2, 4]
/// // set_difference then removes duplicates: [1, 2, 4]
/// assert_eq!(result, vec![1, 2, 4]);  // Unique only
/// ```
pub fn set_difference<T, F>(first1: &[T], first2: &[T], pred: F) -> Vec<T>
where
    T: Clone + PartialEq,
    F: Fn(&T, &T) -> Ordering,
{
    let mut result = multiset_difference(first1, first2, pred);
    let new_len = set_unique_default(&mut result);
    result.truncate(new_len);
    result
}

//---------------------------------------------------------------
// Helper Functions
//---------------------------------------------------------------

/// Find equal range using binary search (lower_bound + upper_bound)
fn equal_range<T, F>(data: &[T], value: &T, pred: &F) -> std::ops::Range<usize>
where
    F: Fn(&T, &T) -> Ordering,
{
    let lower = lower_bound(data, value, pred);
    let upper = upper_bound(data, value, pred);
    lower..upper
}

/// Find first position where value could be inserted (lower bound)
fn lower_bound<T, F>(data: &[T], value: &T, pred: &F) -> usize
where
    F: Fn(&T, &T) -> Ordering,
{
    let mut left = 0;
    let mut right = data.len();

    while left < right {
        let mid = left + (right - left) / 2;
        if pred(&data[mid], value) == Ordering::Less {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    left
}

/// Find first position after all elements equal to value (upper bound)
fn upper_bound<T, F>(data: &[T], value: &T, pred: &F) -> usize
where
    F: Fn(&T, &T) -> Ordering,
{
    let mut left = 0;
    let mut right = data.len();

    while left < right {
        let mid = left + (right - left) / 2;
        if pred(value, &data[mid]) == Ordering::Less {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    left
}

//---------------------------------------------------------------
// Tests
//---------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn cmp_i32(a: &i32, b: &i32) -> Ordering {
        a.cmp(b)
    }

    // Multiset Intersection Tests
    #[test]
    fn test_multiset_intersection_basic() {
        let a = vec![1, 2, 2, 3, 4, 5];
        let b = vec![2, 2, 3, 6, 7];
        let result = multiset_intersection(&a, &b, cmp_i32);
        assert_eq!(result, vec![2, 2, 3]);
    }

    #[test]
    fn test_multiset_intersection_empty() {
        let a: Vec<i32> = vec![];
        let b = vec![1, 2, 3];
        let result = multiset_intersection(&a, &b, cmp_i32);
        assert_eq!(result, Vec::<i32>::new());

        let a = vec![1, 2, 3];
        let b: Vec<i32> = vec![];
        let result = multiset_intersection(&a, &b, cmp_i32);
        assert_eq!(result, Vec::<i32>::new());
    }

    #[test]
    fn test_multiset_intersection_no_overlap() {
        let a = vec![1, 2, 3];
        let b = vec![4, 5, 6];
        let result = multiset_intersection(&a, &b, cmp_i32);
        assert_eq!(result, Vec::<i32>::new());
    }

    #[test]
    fn test_multiset_intersection_duplicates() {
        let a = vec![1, 1, 1, 2, 2, 3];
        let b = vec![1, 1, 2, 2, 2, 3, 3];
        let result = multiset_intersection(&a, &b, cmp_i32);
        // From first sequence: 1, 1, 1, 2, 2, 3
        assert_eq!(result, vec![1, 1, 1, 2, 2, 3]);
    }

    #[test]
    fn test_multiset_intersection_complete_overlap() {
        let a = vec![1, 2, 3, 4, 5];
        let b = vec![1, 2, 3, 4, 5];
        let result = multiset_intersection(&a, &b, cmp_i32);
        assert_eq!(result, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_multiset_1small_intersection() {
        let a = vec![2, 3, 5]; // Small set
        let b = vec![1, 2, 2, 3, 3, 3, 4, 5, 5, 6]; // Large set
        let result = multiset_1small_intersection(&a, &b, cmp_i32);
        assert_eq!(result, vec![2, 3, 5]);
    }

    #[test]
    fn test_multiset_1small_intersection_many_duplicates() {
        let a = vec![2, 2, 2, 3];
        let b = vec![1, 2, 2, 2, 2, 3, 3, 4];
        let result = multiset_1small_intersection(&a, &b, cmp_i32);
        assert_eq!(result, vec![2, 2, 2, 3]);
    }

    #[test]
    fn test_multiset_fast_intersection_small_first() {
        let a = vec![1, 2, 3];
        let b: Vec<i32> = (1..=100).collect();
        let result = multiset_fast_intersection(&a, &b, cmp_i32, 32);
        assert_eq!(result, vec![1, 2, 3]);
    }

    #[test]
    fn test_multiset_fast_intersection_large_first() {
        let a: Vec<i32> = (1..=100).collect();
        let b = vec![1, 2, 3];
        let result = multiset_fast_intersection(&a, &b, cmp_i32, 32);
        assert_eq!(result, vec![1, 2, 3]);
    }

    // Multiset Intersection2 Tests
    #[test]
    fn test_multiset_intersection2() {
        let a = vec![1, 2, 2, 3];
        let b = vec![2, 2, 2, 3, 4];
        let result = multiset_intersection2(&a, &b, cmp_i32);
        // From second sequence
        assert_eq!(result, vec![2, 2, 2, 3]);
    }

    #[test]
    fn test_multiset_intersection2_asymmetric() {
        let a = vec![1, 1, 1, 2];
        let b = vec![1, 2, 2];
        let result = multiset_intersection2(&a, &b, cmp_i32);
        // Copies from second: 1 (once), 2 (twice)
        assert_eq!(result, vec![1, 2, 2]);
    }

    #[test]
    fn test_multiset_1small_intersection2() {
        let a = vec![2, 3];
        let b = vec![1, 2, 2, 2, 3, 3, 4];
        let result = multiset_1small_intersection2(&a, &b, cmp_i32);
        assert_eq!(result, vec![2, 2, 2, 3, 3]);
    }

    #[test]
    fn test_multiset_fast_intersection2() {
        let a = vec![1, 2];
        let b: Vec<i32> = (1..=100).collect();
        let result = multiset_fast_intersection2(&a, &b, cmp_i32, 32);
        assert_eq!(result, vec![1, 2]);
    }

    // Set Unique Tests
    #[test]
    fn test_set_unique() {
        let mut data = vec![1, 1, 2, 2, 2, 3, 4, 4, 5];
        let new_len = set_unique(&mut data, |a, b| a == b);
        data.truncate(new_len);
        assert_eq!(data, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_set_unique_single() {
        let mut data = vec![42];
        let new_len = set_unique_default(&mut data);
        assert_eq!(new_len, 1);
        assert_eq!(data, vec![42]);
    }

    #[test]
    fn test_set_unique_empty() {
        let mut data: Vec<i32> = vec![];
        let new_len = set_unique_default(&mut data);
        assert_eq!(new_len, 0);
    }

    #[test]
    fn test_set_unique_no_duplicates() {
        let mut data = vec![1, 2, 3, 4, 5];
        let new_len = set_unique_default(&mut data);
        assert_eq!(new_len, 5);
        assert_eq!(data, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_set_unique_all_same() {
        let mut data = vec![7, 7, 7, 7, 7];
        let new_len = set_unique_default(&mut data);
        data.truncate(new_len);
        assert_eq!(data, vec![7]);
    }

    // Multiset Union Tests
    #[test]
    fn test_multiset_union() {
        let a = vec![1, 2, 3, 4];
        let b = vec![3, 4, 5, 6];
        let result = multiset_union(&a, &b, cmp_i32);
        assert_eq!(result, vec![1, 2, 3, 3, 4, 4, 5, 6]);
    }

    #[test]
    fn test_multiset_union_no_overlap() {
        let a = vec![1, 2, 3];
        let b = vec![4, 5, 6];
        let result = multiset_union(&a, &b, cmp_i32);
        assert_eq!(result, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_multiset_union_with_duplicates() {
        let a = vec![1, 1, 2, 3];
        let b = vec![2, 2, 3, 4];
        let result = multiset_union(&a, &b, cmp_i32);
        assert_eq!(result, vec![1, 1, 2, 2, 2, 3, 3, 4]);
    }

    #[test]
    fn test_multiset_union_empty() {
        let a: Vec<i32> = vec![];
        let b = vec![1, 2, 3];
        let result = multiset_union(&a, &b, cmp_i32);
        assert_eq!(result, vec![1, 2, 3]);

        let a = vec![1, 2, 3];
        let b: Vec<i32> = vec![];
        let result = multiset_union(&a, &b, cmp_i32);
        assert_eq!(result, vec![1, 2, 3]);
    }

    // Multiset Difference Tests
    #[test]
    fn test_multiset_difference() {
        let a = vec![1, 2, 2, 3, 4];
        let b = vec![2, 3, 3, 5];
        let result = multiset_difference(&a, &b, cmp_i32);
        assert_eq!(result, vec![1, 2, 4]);
    }

    #[test]
    fn test_multiset_difference_no_overlap() {
        let a = vec![1, 2, 3];
        let b = vec![4, 5, 6];
        let result = multiset_difference(&a, &b, cmp_i32);
        assert_eq!(result, vec![1, 2, 3]);
    }

    #[test]
    fn test_multiset_difference_complete_overlap() {
        let a = vec![1, 2, 3];
        let b = vec![1, 2, 3];
        let result = multiset_difference(&a, &b, cmp_i32);
        assert_eq!(result, Vec::<i32>::new());
    }

    #[test]
    fn test_multiset_difference_empty() {
        let a: Vec<i32> = vec![];
        let b = vec![1, 2, 3];
        let result = multiset_difference(&a, &b, cmp_i32);
        assert_eq!(result, Vec::<i32>::new());

        let a = vec![1, 2, 3];
        let b: Vec<i32> = vec![];
        let result = multiset_difference(&a, &b, cmp_i32);
        assert_eq!(result, vec![1, 2, 3]);
    }

    // Unique Set Operations Tests
    #[test]
    fn test_set_intersection_unique() {
        let a = vec![1, 2, 2, 3, 4];
        let b = vec![2, 2, 3, 5];
        let result = set_intersection(&a, &b, cmp_i32);
        assert_eq!(result, vec![2, 3]); // Unique only
    }

    #[test]
    fn test_set_union_unique() {
        let a = vec![1, 2, 2, 3];
        let b = vec![2, 3, 4, 4];
        let result = set_union(&a, &b, cmp_i32);
        assert_eq!(result, vec![1, 2, 3, 4]); // Unique only
    }

    #[test]
    fn test_set_difference_unique() {
        let a = vec![1, 2, 2, 3, 4, 4];
        let b = vec![2, 3];
        let result = set_difference(&a, &b, cmp_i32);
        // multiset_difference removes matching elements: [1, 2, 4]
        // set_difference then removes duplicates: [1, 2, 4]
        assert_eq!(result, vec![1, 2, 4]); // Unique only
    }

    // Large Dataset Tests
    #[test]
    fn test_large_datasets_intersection() {
        // Test with 1M elements
        let a: Vec<i32> = (0..1_000_000).filter(|x| x % 2 == 0).collect();
        let b: Vec<i32> = (0..1_000_000).filter(|x| x % 3 == 0).collect();
        let result = multiset_fast_intersection(&a, &b, cmp_i32, 32);
        // Elements divisible by both 2 and 3 (i.e., by 6)
        let expected: Vec<i32> = (0..1_000_000).filter(|x| x % 6 == 0).collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_large_datasets_union() {
        let a: Vec<i32> = (0..10_000).filter(|x| x % 2 == 0).collect();
        let b: Vec<i32> = (0..10_000).filter(|x| x % 3 == 0).collect();
        let result = multiset_union(&a, &b, cmp_i32);

        // Verify result is sorted
        for i in 1..result.len() {
            assert!(result[i - 1] <= result[i]);
        }

        // Verify size
        let expected_size = a.len() + b.len();
        assert_eq!(result.len(), expected_size);
    }

    #[test]
    fn test_large_datasets_difference() {
        let a: Vec<i32> = (0..10_000).collect();
        let b: Vec<i32> = (5_000..15_000).collect();
        let result = multiset_difference(&a, &b, cmp_i32);
        let expected: Vec<i32> = (0..5_000).collect();
        assert_eq!(result, expected);
    }

    // Performance Comparison Tests
    #[test]
    fn test_performance_comparison() {
        // Small first, large second - should use binary search
        let small: Vec<i32> = (0..100).collect();
        let large: Vec<i32> = (0..10_000).collect();

        let result1 = multiset_1small_intersection(&small, &large, cmp_i32);
        let result2 = multiset_intersection(&small, &large, cmp_i32);
        let result3 = multiset_fast_intersection(&small, &large, cmp_i32, 32);

        assert_eq!(result1, result2);
        assert_eq!(result1, result3);
    }

    #[test]
    fn test_adaptive_selection_boundary() {
        // Test threshold boundary condition
        let a: Vec<i32> = (0..10).collect();
        let b: Vec<i32> = (0..320).collect(); // Exactly 32x larger

        let result = multiset_fast_intersection(&a, &b, cmp_i32, 32);
        let expected: Vec<i32> = (0..10).collect();
        assert_eq!(result, expected);
    }

    // Helper Function Tests
    #[test]
    fn test_equal_range() {
        let data = vec![1, 2, 2, 2, 3, 3, 4, 5];
        let range = equal_range(&data, &2, &cmp_i32);
        assert_eq!(range, 1..4); // Indices of 2s

        let range = equal_range(&data, &3, &cmp_i32);
        assert_eq!(range, 4..6); // Indices of 3s

        let range = equal_range(&data, &10, &cmp_i32);
        assert_eq!(range, 8..8); // Not found
    }

    #[test]
    fn test_lower_bound() {
        let data = vec![1, 2, 2, 2, 3, 3, 4, 5];
        assert_eq!(lower_bound(&data, &2, &cmp_i32), 1);
        assert_eq!(lower_bound(&data, &3, &cmp_i32), 4);
        assert_eq!(lower_bound(&data, &0, &cmp_i32), 0);
        assert_eq!(lower_bound(&data, &10, &cmp_i32), 8);
    }

    #[test]
    fn test_upper_bound() {
        let data = vec![1, 2, 2, 2, 3, 3, 4, 5];
        assert_eq!(upper_bound(&data, &2, &cmp_i32), 4);
        assert_eq!(upper_bound(&data, &3, &cmp_i32), 6);
        assert_eq!(upper_bound(&data, &0, &cmp_i32), 0);
        assert_eq!(upper_bound(&data, &10, &cmp_i32), 8);
    }

    // Edge Cases
    #[test]
    fn test_single_element_sequences() {
        let a = vec![5];
        let b = vec![5];
        assert_eq!(multiset_intersection(&a, &b, cmp_i32), vec![5]);
        assert_eq!(multiset_union(&a, &b, cmp_i32), vec![5, 5]);
        assert_eq!(multiset_difference(&a, &b, cmp_i32), Vec::<i32>::new());

        let a = vec![3];
        let b = vec![5];
        assert_eq!(multiset_intersection(&a, &b, cmp_i32), Vec::<i32>::new());
        assert_eq!(multiset_union(&a, &b, cmp_i32), vec![3, 5]);
        assert_eq!(multiset_difference(&a, &b, cmp_i32), vec![3]);
    }

    #[test]
    fn test_all_duplicates() {
        let a = vec![1, 1, 1, 1];
        let b = vec![1, 1, 1];
        let result = multiset_intersection(&a, &b, cmp_i32);
        assert_eq!(result, vec![1, 1, 1, 1]);

        let result = multiset_intersection2(&a, &b, cmp_i32);
        assert_eq!(result, vec![1, 1, 1]);
    }
}
