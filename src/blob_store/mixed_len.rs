//! MixedLenBlobStore - Hybrid storage for mixed fixed/variable-length records
//!
//! # Overview
//!
//! MixedLenBlobStore optimizes storage for datasets where a significant portion of records
//! share the same length (fixed-length) while others have varying lengths (variable-length).
//! It uses a rank/select bitmap to distinguish between the two types and stores them separately.
//!
//! # Algorithm
//!
//! 1. **Length Analysis**: Determine the most common record length (dominant fixed length)
//!    - Scans all records and counts length frequencies
//!    - Selects the most frequent length as the "fixed length"
//!
//! 2. **Separation**: Split records into two groups
//!    - **Fixed-length**: Records matching the dominant length (stored in packed array)
//!    - **Variable-length**: All other records (stored with offset array)
//!
//! 3. **Bitmap Encoding**: Use rank/select to map record ID → storage location
//!    - Bitmap bit: 1 = fixed-length, 0 = variable-length
//!    - rank1(rec_id) = index in fixed-length array
//!    - rank0(rec_id) = index in variable-length array
//!
//! # Memory Layout
//!
//! ```text
//! Record ID:     0    1    2    3    4    5    6
//! Lengths:      [10] [10] [15] [10] [20] [10] [10]
//! Bitmap:       [1]  [1]  [0]  [1]  [0]  [1]  [1]
//!                |    |    |    |    |    |    |
//! Fixed (10):   [0]  [1]       [2]       [3]  [4]  <- rank1 index
//! Variable:            [0]       [1]              <- rank0 index
//! ```
//!
//! # Example
//!
//! ```rust
//! use zipora::blob_store::MixedLenBlobStore;
//!
//! let data = vec![
//!     vec![1, 2, 3, 4, 5],      // len=5 (fixed)
//!     vec![6, 7, 8, 9, 10],     // len=5 (fixed)
//!     vec![11, 12, 13],         // len=3 (variable)
//!     vec![14, 15, 16, 17, 18], // len=5 (fixed)
//! ];
//!
//! let store = MixedLenBlobStore::build_from(&data).unwrap();
//! assert_eq!(store.fixed_len(), 5);
//! assert_eq!(store.fixed_count(), 3);
//! assert_eq!(store.variable_count(), 1);
//! # Ok::<(), zipora::error::ZiporaError>(())
//! ```
//!
//! # Use Cases
//!
//! - Database records with common row sizes
//! - Network packets with standard MTU sizes
//! - Fixed-size image thumbnails with occasional variations
//! - Any dataset with ≥50% records of the same size
//!
//! # Performance Characteristics
//!
//! - **Build Time**: O(N) where N = total records
//! - **Query Time**: O(1) for fixed-length, O(1) for variable-length
//! - **Space**: Optimal when ≥50% records share the same length
//! - **Read-Only**: No dynamic updates after build

use crate::containers::UintVecMin0;
use crate::succinct::rank_select::{RankSelectInterleaved256, RankSelectOps};
use crate::blob_store::traits::{BlobStore, BatchBlobStore, IterableBlobStore, BlobStoreStats};
use crate::error::{Result, ZiporaError};
use crate::RecordId;
use std::collections::HashMap;

/// MixedLenBlobStore - Hybrid storage for mixed fixed/variable-length records
///
/// Read-only blob store optimized for datasets with a dominant fixed length.
pub struct MixedLenBlobStore {
    /// The dominant fixed record length in bytes
    fixed_len: usize,
    /// Number of fixed-length records
    fixed_num: usize,
    /// Bitmap: 1 = fixed-length record, 0 = variable-length record
    is_fixed_len: RankSelectInterleaved256,
    /// Packed fixed-length records (fixed_num * fixed_len bytes)
    /// Records stored sequentially without offsets
    fixed_len_values: Vec<u8>,
    /// Variable-length record data (concatenated)
    var_len_values: Vec<u8>,
    /// Offsets for variable-length records (var_num + 1 entries)
    /// var_len_offsets[i]..var_len_offsets[i+1] = record i's data
    var_len_offsets: UintVecMin0,
    /// Total number of records (fixed + variable)
    num_records: usize,
    /// Statistics
    stats: BlobStoreStats,
}

impl MixedLenBlobStore {
    /// Build from data, auto-detecting the dominant fixed length
    ///
    /// # Arguments
    ///
    /// * `data` - Records to store
    ///
    /// # Example
    ///
    /// ```rust
    /// use zipora::blob_store::MixedLenBlobStore;
    ///
    /// let data = vec![
    ///     vec![1, 2, 3],
    ///     vec![4, 5, 6],
    ///     vec![7, 8],
    /// ];
    ///
    /// let store = MixedLenBlobStore::build_from(&data).unwrap();
    /// assert_eq!(store.fixed_len(), 3); // Most common length
    /// # Ok::<(), zipora::error::ZiporaError>(())
    /// ```
    pub fn build_from(data: &[Vec<u8>]) -> Result<Self> {
        if data.is_empty() {
            return Ok(Self::default());
        }

        // Step 1: Determine dominant fixed length
        let fixed_len = Self::determine_fixed_length(data);

        // Step 2: Separate fixed and variable records
        let mut is_fixed = Vec::with_capacity(data.len());
        let mut fixed_values = Vec::new();
        let mut var_values = Vec::new();
        let mut var_offsets = vec![0usize];

        for record in data {
            if record.len() == fixed_len {
                is_fixed.push(true);
                fixed_values.extend_from_slice(record);
            } else {
                is_fixed.push(false);
                var_values.extend_from_slice(record);
                var_offsets.push(var_values.len());
            }
        }

        // Step 3: Build rank/select bitmap
        let mut bv = crate::succinct::BitVector::new();
        for &bit in &is_fixed {
            bv.push(bit)?;
        }
        let is_fixed_len = RankSelectInterleaved256::new(bv)?;

        // Step 4: Build variable-length offset array
        let var_len_offsets = UintVecMin0::build_from_usize(&var_offsets).0;

        // Calculate number of fixed-length records
        // Special case: if fixed_len is 0, count how many times we saw it in is_fixed
        let fixed_num = if fixed_len == 0 {
            is_fixed.iter().filter(|&&b| b).count()
        } else {
            fixed_values.len() / fixed_len
        };

        let total_size = data.iter().map(|r| r.len()).sum();
        let mut stats = BlobStoreStats::default();
        stats.blob_count = data.len();
        stats.total_size = total_size;
        stats.average_size = if data.is_empty() {
            0.0
        } else {
            total_size as f64 / data.len() as f64
        };

        Ok(Self {
            fixed_len,
            fixed_num,
            is_fixed_len,
            fixed_len_values: fixed_values,
            var_len_values: var_values,
            var_len_offsets,
            num_records: data.len(),
            stats,
        })
    }

    /// Build with explicit fixed length (skip auto-detection)
    ///
    /// Useful when you know the dominant length in advance.
    pub fn build_from_with_fixed_len(data: &[Vec<u8>], fixed_len: usize) -> Result<Self> {
        if data.is_empty() {
            return Ok(Self::default());
        }

        let mut is_fixed = Vec::with_capacity(data.len());
        let mut fixed_values = Vec::new();
        let mut var_values = Vec::new();
        let mut var_offsets = vec![0usize];

        for record in data {
            if record.len() == fixed_len {
                is_fixed.push(true);
                fixed_values.extend_from_slice(record);
            } else {
                is_fixed.push(false);
                var_values.extend_from_slice(record);
                var_offsets.push(var_values.len());
            }
        }

        let mut bv = crate::succinct::BitVector::new();
        for &bit in &is_fixed {
            bv.push(bit)?;
        }
        let is_fixed_len = RankSelectInterleaved256::new(bv)?;
        let var_len_offsets = UintVecMin0::build_from_usize(&var_offsets).0;

        // Calculate number of fixed-length records
        // Special case: if fixed_len is 0, count how many times we saw it in is_fixed
        let fixed_num = if fixed_len == 0 {
            is_fixed.iter().filter(|&&b| b).count()
        } else {
            fixed_values.len() / fixed_len
        };

        let total_size = data.iter().map(|r| r.len()).sum();
        let mut stats = BlobStoreStats::default();
        stats.blob_count = data.len();
        stats.total_size = total_size;
        stats.average_size = if data.is_empty() {
            0.0
        } else {
            total_size as f64 / data.len() as f64
        };

        Ok(Self {
            fixed_len,
            fixed_num,
            is_fixed_len,
            fixed_len_values: fixed_values,
            var_len_values: var_values,
            var_len_offsets,
            num_records: data.len(),
            stats,
        })
    }

    /// Determine the most common record length in the dataset
    fn determine_fixed_length(data: &[Vec<u8>]) -> usize {
        let mut len_counts: HashMap<usize, usize> = HashMap::new();

        for record in data {
            *len_counts.entry(record.len()).or_insert(0) += 1;
        }

        len_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(len, _)| len)
            .unwrap_or(0)
    }

    /// Get the dominant fixed length
    pub fn fixed_len(&self) -> usize {
        self.fixed_len
    }

    /// Get the number of fixed-length records
    pub fn fixed_count(&self) -> usize {
        self.fixed_num
    }

    /// Get the number of variable-length records
    pub fn variable_count(&self) -> usize {
        self.num_records - self.fixed_num
    }

    /// Get memory usage statistics
    pub fn memory_stats(&self) -> MemoryStats {
        MemoryStats {
            fixed_values_size: self.fixed_len_values.len(),
            var_values_size: self.var_len_values.len(),
            var_offsets_size: self.var_len_offsets.mem_size(),
            bitmap_size: self.is_fixed_len.space_overhead_percent() as usize,
            total_size: self.fixed_len_values.len()
                + self.var_len_values.len()
                + self.var_len_offsets.mem_size(),
            fixed_count: self.fixed_num,
            variable_count: self.variable_count(),
        }
    }

    /// Check if a record ID corresponds to a fixed-length record
    pub fn is_fixed_length(&self, id: RecordId) -> bool {
        if id as usize >= self.num_records {
            false
        } else {
            self.is_fixed_len.get(id as usize).unwrap_or(false)
        }
    }
}

impl Default for MixedLenBlobStore {
    fn default() -> Self {
        // Create empty bitmap
        let empty_bv = crate::succinct::BitVector::new();
        let empty_bitmap = RankSelectInterleaved256::new(empty_bv).unwrap_or_else(|_| {
            // If creation fails, create a minimal empty structure
            panic!("Failed to create empty RankSelectInterleaved256")
        });

        Self {
            fixed_len: 0,
            fixed_num: 0,
            is_fixed_len: empty_bitmap,
            fixed_len_values: Vec::new(),
            var_len_values: Vec::new(),
            var_len_offsets: UintVecMin0::new_empty(),
            num_records: 0,
            stats: BlobStoreStats::default(),
        }
    }
}

impl BlobStore for MixedLenBlobStore {
    fn get(&self, id: RecordId) -> Result<Vec<u8>> {
        let idx = id as usize;
        if idx >= self.num_records {
            return Err(ZiporaError::not_found(format!(
                "Record {} not found (max {})",
                id,
                self.num_records - 1
            )));
        }

        if self.is_fixed_len.get(idx).unwrap_or(false) {
            // Fixed-length record
            let fixed_id = self.is_fixed_len.rank1(idx);
            let offset = fixed_id * self.fixed_len;

            if offset + self.fixed_len > self.fixed_len_values.len() {
                return Err(ZiporaError::invalid_data(format!(
                    "Invalid fixed-length offset={} len={} buffer_size={}",
                    offset,
                    self.fixed_len,
                    self.fixed_len_values.len()
                )));
            }

            Ok(self.fixed_len_values[offset..offset + self.fixed_len].to_vec())
        } else {
            // Variable-length record
            let var_id = self.is_fixed_len.rank0(idx);
            let beg = self.var_len_offsets.get(var_id);
            let end = self.var_len_offsets.get(var_id + 1);

            if end > self.var_len_values.len() {
                return Err(ZiporaError::invalid_data(format!(
                    "Invalid variable-length range [{}, {}) buffer_size={}",
                    beg,
                    end,
                    self.var_len_values.len()
                )));
            }

            Ok(self.var_len_values[beg..end].to_vec())
        }
    }

    fn put(&mut self, _data: &[u8]) -> Result<RecordId> {
        Err(ZiporaError::not_supported(
            "MixedLenBlobStore is read-only after build"
        ))
    }

    fn remove(&mut self, _id: RecordId) -> Result<()> {
        Err(ZiporaError::not_supported(
            "MixedLenBlobStore is read-only"
        ))
    }

    fn contains(&self, id: RecordId) -> bool {
        (id as usize) < self.num_records
    }

    fn size(&self, id: RecordId) -> Result<Option<usize>> {
        let idx = id as usize;
        if idx >= self.num_records {
            return Ok(None);
        }

        if self.is_fixed_len.get(idx).unwrap_or(false) {
            // Fixed-length record
            Ok(Some(self.fixed_len))
        } else {
            // Variable-length record
            let var_id = self.is_fixed_len.rank0(idx);
            let beg = self.var_len_offsets.get(var_id);
            let end = self.var_len_offsets.get(var_id + 1);
            Ok(Some(end - beg))
        }
    }

    fn len(&self) -> usize {
        self.num_records
    }

    fn stats(&self) -> BlobStoreStats {
        self.stats.clone()
    }
}

impl BatchBlobStore for MixedLenBlobStore {
    fn put_batch<I>(&mut self, _blobs: I) -> Result<Vec<RecordId>>
    where
        I: IntoIterator<Item = Vec<u8>>,
    {
        Err(ZiporaError::not_supported(
            "MixedLenBlobStore is read-only"
        ))
    }

    fn get_batch<I>(&self, ids: I) -> Result<Vec<Option<Vec<u8>>>>
    where
        I: IntoIterator<Item = RecordId>,
    {
        ids.into_iter()
            .map(|id| {
                if self.contains(id) {
                    self.get(id).map(Some)
                } else {
                    Ok(None)
                }
            })
            .collect()
    }

    fn remove_batch<I>(&mut self, _ids: I) -> Result<usize>
    where
        I: IntoIterator<Item = RecordId>,
    {
        Err(ZiporaError::not_supported(
            "MixedLenBlobStore is read-only"
        ))
    }
}

impl IterableBlobStore for MixedLenBlobStore {
    type IdIter = std::ops::Range<RecordId>;

    fn iter_ids(&self) -> Self::IdIter {
        0..self.num_records as RecordId
    }
}

/// Memory usage statistics for MixedLenBlobStore
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Size of fixed-length values array in bytes
    pub fixed_values_size: usize,
    /// Size of variable-length values array in bytes
    pub var_values_size: usize,
    /// Size of variable-length offsets array in bytes
    pub var_offsets_size: usize,
    /// Size of rank/select bitmap in bytes
    pub bitmap_size: usize,
    /// Total size in bytes
    pub total_size: usize,
    /// Number of fixed-length records
    pub fixed_count: usize,
    /// Number of variable-length records
    pub variable_count: usize,
}

impl MemoryStats {
    /// Calculate the percentage of fixed-length records
    pub fn fixed_percentage(&self) -> f64 {
        let total = self.fixed_count + self.variable_count;
        if total == 0 {
            0.0
        } else {
            (self.fixed_count as f64 / total as f64) * 100.0
        }
    }

    /// Calculate metadata overhead (bitmap + offsets) as percentage of data
    pub fn metadata_overhead_percent(&self) -> f64 {
        let data_size = self.fixed_values_size + self.var_values_size;
        if data_size == 0 {
            0.0
        } else {
            ((self.bitmap_size + self.var_offsets_size) as f64 / data_size as f64) * 100.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_data() {
        let data: Vec<Vec<u8>> = vec![];
        let store = MixedLenBlobStore::build_from(&data).unwrap();

        assert_eq!(store.len(), 0);
        assert!(store.is_empty());
        assert_eq!(store.fixed_count(), 0);
        assert_eq!(store.variable_count(), 0);
    }

    #[test]
    fn test_all_fixed_length() {
        let data = vec![
            vec![1, 2, 3, 4, 5],
            vec![6, 7, 8, 9, 10],
            vec![11, 12, 13, 14, 15],
            vec![16, 17, 18, 19, 20],
        ];

        let store = MixedLenBlobStore::build_from(&data).unwrap();

        assert_eq!(store.fixed_len(), 5);
        assert_eq!(store.fixed_count(), 4);
        assert_eq!(store.variable_count(), 0);

        for i in 0..4 {
            assert_eq!(store.get(i).unwrap(), data[i as usize]);
            assert!(store.is_fixed_length(i));
        }
    }

    #[test]
    fn test_all_variable_length() {
        let data = vec![
            vec![1],
            vec![2, 3],
            vec![4, 5, 6],
            vec![7, 8, 9, 10],
            vec![11, 12, 13, 14, 15],
        ];

        let store = MixedLenBlobStore::build_from(&data).unwrap();

        // All different lengths, so dominant is arbitrary (first occurrence wins in some impls)
        assert_eq!(store.fixed_count() + store.variable_count(), 5);

        for i in 0..5 {
            assert_eq!(store.get(i).unwrap(), data[i as usize]);
        }
    }

    #[test]
    fn test_mixed_5050() {
        let data = vec![
            vec![1, 2, 3, 4, 5],      // len=5 (fixed)
            vec![6, 7],               // len=2 (variable)
            vec![8, 9, 10, 11, 12],   // len=5 (fixed)
            vec![13, 14, 15],         // len=3 (variable)
            vec![16, 17, 18, 19, 20], // len=5 (fixed)
        ];

        let store = MixedLenBlobStore::build_from(&data).unwrap();

        assert_eq!(store.fixed_len(), 5);
        assert_eq!(store.fixed_count(), 3);
        assert_eq!(store.variable_count(), 2);

        // Verify all records
        for i in 0..5 {
            assert_eq!(store.get(i).unwrap(), data[i as usize]);
        }

        // Check which are fixed
        assert!(store.is_fixed_length(0));
        assert!(!store.is_fixed_length(1));
        assert!(store.is_fixed_length(2));
        assert!(!store.is_fixed_length(3));
        assert!(store.is_fixed_length(4));
    }

    #[test]
    fn test_dominant_fixed_length_detection() {
        let data = vec![
            vec![1, 2, 3],     // len=3 (appears 5 times)
            vec![4, 5, 6],
            vec![7, 8],        // len=2 (appears 2 times)
            vec![9, 10, 11],
            vec![12, 13],
            vec![14, 15, 16],
            vec![17, 18, 19],
            vec![20, 21, 22],
        ];

        let store = MixedLenBlobStore::build_from(&data).unwrap();

        assert_eq!(store.fixed_len(), 3); // Most common length
        assert_eq!(store.fixed_count(), 6);
        assert_eq!(store.variable_count(), 2);
    }

    #[test]
    fn test_explicit_fixed_length() {
        let data = vec![
            vec![1, 2, 3, 4],
            vec![5, 6],
            vec![7, 8, 9, 10],
            vec![11, 12, 13],
            vec![14, 15, 16, 17],
        ];

        let store = MixedLenBlobStore::build_from_with_fixed_len(&data, 4).unwrap();

        assert_eq!(store.fixed_len(), 4);
        assert_eq!(store.fixed_count(), 3); // Records 0, 2, 4
        assert_eq!(store.variable_count(), 2); // Records 1, 3

        for i in 0..5 {
            assert_eq!(store.get(i).unwrap(), data[i as usize]);
        }
    }

    #[test]
    fn test_size_method() {
        let data = vec![
            vec![1, 2, 3, 4, 5],      // len=5 (fixed)
            vec![6, 7, 8],            // len=3 (variable)
            vec![9, 10, 11, 12, 13],  // len=5 (fixed)
        ];

        let store = MixedLenBlobStore::build_from(&data).unwrap();

        assert_eq!(store.size(0).unwrap(), Some(5));
        assert_eq!(store.size(1).unwrap(), Some(3));
        assert_eq!(store.size(2).unwrap(), Some(5));
        assert_eq!(store.size(999).unwrap(), None);
    }

    #[test]
    fn test_batch_operations() {
        let data = vec![
            vec![1, 2, 3],
            vec![4, 5, 6],
            vec![7, 8],
        ];

        let store = MixedLenBlobStore::build_from(&data).unwrap();

        let results = store.get_batch(vec![0, 2, 999]).unwrap();
        assert_eq!(results[0], Some(vec![1, 2, 3]));
        assert_eq!(results[1], Some(vec![7, 8]));
        assert_eq!(results[2], None);
    }

    #[test]
    fn test_iteration() {
        let data = vec![
            vec![1, 2, 3],
            vec![4, 5, 6],
            vec![7, 8, 9],
        ];

        let store = MixedLenBlobStore::build_from(&data).unwrap();

        let ids: Vec<_> = store.iter_ids().collect();
        assert_eq!(ids, vec![0, 1, 2]);

        let blobs: Vec<_> = store.iter_blobs().collect::<Result<Vec<_>>>().unwrap();
        assert_eq!(blobs.len(), 3);
        assert_eq!(blobs[0].1, vec![1, 2, 3]);
        assert_eq!(blobs[1].1, vec![4, 5, 6]);
        assert_eq!(blobs[2].1, vec![7, 8, 9]);
    }

    #[test]
    fn test_read_only_operations() {
        let data = vec![vec![1, 2, 3]];
        let mut store = MixedLenBlobStore::build_from(&data).unwrap();

        assert!(store.put(&[4, 5, 6]).is_err());
        assert!(store.remove(0).is_err());
        assert!(store.put_batch(vec![vec![7, 8]]).is_err());
        assert!(store.remove_batch(vec![0]).is_err());
    }

    #[test]
    fn test_record_not_found() {
        let data = vec![vec![1, 2, 3]];
        let store = MixedLenBlobStore::build_from(&data).unwrap();

        assert!(store.get(999).is_err());
        assert!(!store.contains(999));
    }

    #[test]
    fn test_memory_stats() {
        let data = vec![
            vec![1, 2, 3, 4, 5],
            vec![6, 7, 8, 9, 10],
            vec![11, 12],
            vec![13, 14, 15, 16, 17],
        ];

        let store = MixedLenBlobStore::build_from(&data).unwrap();
        let stats = store.memory_stats();

        assert!(stats.fixed_values_size > 0);
        assert!(stats.var_values_size > 0);
        assert_eq!(stats.fixed_count, 3);
        assert_eq!(stats.variable_count, 1);
        assert_eq!(stats.fixed_percentage(), 75.0);

        println!("Fixed values: {} bytes", stats.fixed_values_size);
        println!("Variable values: {} bytes", stats.var_values_size);
        println!("Variable offsets: {} bytes", stats.var_offsets_size);
        println!("Bitmap: {} bytes", stats.bitmap_size);
        println!("Total: {} bytes", stats.total_size);
        println!("Fixed percentage: {:.2}%", stats.fixed_percentage());
        println!("Metadata overhead: {:.2}%", stats.metadata_overhead_percent());
    }

    #[test]
    fn test_zero_length_records() {
        let data = vec![
            vec![],
            vec![],
            vec![1, 2, 3],
            vec![],
        ];

        let store = MixedLenBlobStore::build_from(&data).unwrap();

        assert_eq!(store.fixed_len(), 0); // Most common length
        assert_eq!(store.fixed_count(), 3);
        assert_eq!(store.variable_count(), 1);

        assert_eq!(store.get(0).unwrap(), Vec::<u8>::new());
        assert_eq!(store.get(1).unwrap(), Vec::<u8>::new());
        assert_eq!(store.get(2).unwrap(), vec![1, 2, 3]);
        assert_eq!(store.get(3).unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn test_large_dataset() {
        let mut data = Vec::new();

        // 700 records of length 100 (fixed)
        for i in 0..700 {
            data.push(vec![i as u8; 100]);
        }

        // 300 records of varying lengths (variable)
        for i in 0..300 {
            data.push(vec![(i + 100) as u8; 50 + (i % 50)]);
        }

        let store = MixedLenBlobStore::build_from(&data).unwrap();

        assert_eq!(store.fixed_len(), 100);
        assert_eq!(store.fixed_count(), 700);
        assert_eq!(store.variable_count(), 300);

        // Verify samples
        assert_eq!(store.get(0).unwrap().len(), 100);
        assert_eq!(store.get(699).unwrap().len(), 100);
        assert!(store.get(700).unwrap().len() >= 50);
        assert!(store.get(999).unwrap().len() < 100);

        let stats = store.memory_stats();
        assert_eq!(stats.fixed_percentage(), 70.0);
    }

    #[test]
    fn test_rank_select_correctness() {
        let data = vec![
            vec![1, 2, 3],        // idx=0, fixed, rank1=0
            vec![4, 5, 6],        // idx=1, fixed, rank1=1
            vec![7, 8],           // idx=2, var, rank0=0
            vec![9, 10, 11],      // idx=3, fixed, rank1=2
            vec![12, 13, 14, 15], // idx=4, var, rank0=1
            vec![16, 17, 18],     // idx=5, fixed, rank1=3
        ];

        let store = MixedLenBlobStore::build_from(&data).unwrap();

        // Verify bitmap and rank operations work correctly
        assert!(store.is_fixed_length(0));
        assert!(store.is_fixed_length(1));
        assert!(!store.is_fixed_length(2));
        assert!(store.is_fixed_length(3));
        assert!(!store.is_fixed_length(4));
        assert!(store.is_fixed_length(5));

        // Verify data retrieval uses correct indices
        assert_eq!(store.get(0).unwrap(), vec![1, 2, 3]);
        assert_eq!(store.get(1).unwrap(), vec![4, 5, 6]);
        assert_eq!(store.get(2).unwrap(), vec![7, 8]);
        assert_eq!(store.get(3).unwrap(), vec![9, 10, 11]);
        assert_eq!(store.get(4).unwrap(), vec![12, 13, 14, 15]);
        assert_eq!(store.get(5).unwrap(), vec![16, 17, 18]);
    }

    #[test]
    fn test_single_record() {
        let data = vec![vec![1, 2, 3, 4, 5]];
        let store = MixedLenBlobStore::build_from(&data).unwrap();

        assert_eq!(store.len(), 1);
        assert_eq!(store.fixed_len(), 5);
        assert_eq!(store.fixed_count(), 1);
        assert_eq!(store.variable_count(), 0);
        assert_eq!(store.get(0).unwrap(), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_efficiency_vs_separate_stores() {
        // Scenario: 80% records of length 256, 20% varying
        let mut data = Vec::new();

        for i in 0..800 {
            data.push(vec![i as u8; 256]);
        }

        for i in 0..200 {
            data.push(vec![(i + 100) as u8; 128 + (i % 128)]);
        }

        let store = MixedLenBlobStore::build_from(&data).unwrap();
        let stats = store.memory_stats();

        // Fixed-length records: no offset overhead (just packed data)
        let fixed_data_size = 800 * 256;
        assert_eq!(stats.fixed_values_size, fixed_data_size);

        // Variable-length records: data + offset array
        assert!(stats.var_values_size > 0);
        assert!(stats.var_offsets_size > 0);

        // Total should be much better than storing all with offsets
        let naive_overhead = 1000 * 8; // 1000 offset entries * 8 bytes
        let mixed_overhead = stats.var_offsets_size + stats.bitmap_size;
        assert!(mixed_overhead < naive_overhead);

        println!("MixedLen overhead: {} bytes", mixed_overhead);
        println!("Naive overhead: {} bytes", naive_overhead);
        println!("Savings: {} bytes", naive_overhead - mixed_overhead);
    }
}
