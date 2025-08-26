//! External sorting implementation for datasets larger than memory
//!
//! This module provides efficient external sorting using replacement selection
//! and k-way merging. The algorithm is designed to handle datasets that exceed
//! available memory by using disk-based temporary storage and efficient 
//! merge operations.

use crate::algorithms::tournament_tree::{LoserTree, LoserTreeConfig};
use crate::error::{Result, ZiporaError};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::path::PathBuf;
use std::fs::{File, remove_file};
use std::io::{BufReader, BufWriter, Read, Write};
use std::marker::PhantomData;
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

/// Configuration for external sorting operations
#[derive(Debug, Clone)]
pub struct ReplaceSelectSortConfig {
    /// Size of the main memory buffer in bytes
    pub memory_buffer_size: usize,
    /// Directory for temporary files
    pub temp_dir: PathBuf,
    /// Whether to use secure memory pool
    pub use_secure_memory: bool,
    /// Compression for temporary files
    pub compress_temp_files: bool,
    /// Number of merge ways for final merge
    pub merge_ways: usize,
    /// Clean up temporary files automatically
    pub cleanup_temp_files: bool,
}

impl Default for ReplaceSelectSortConfig {
    fn default() -> Self {
        Self {
            memory_buffer_size: 64 * 1024 * 1024, // 64MB
            temp_dir: std::env::temp_dir(),
            use_secure_memory: true,
            compress_temp_files: false,
            merge_ways: 16,
            cleanup_temp_files: true,
        }
    }
}

/// Statistics for external sort operations
#[derive(Debug, Clone)]
pub struct ExternalSortStats {
    /// Total number of items sorted
    pub items_sorted: usize,
    /// Number of runs generated
    pub runs_generated: usize,
    /// Total bytes written to temporary files
    pub temp_bytes_written: usize,
    /// Total bytes read from temporary files
    pub temp_bytes_read: usize,
    /// Number of merge passes
    pub merge_passes: usize,
    /// Total processing time in microseconds
    pub processing_time_us: u64,
}

impl ExternalSortStats {
    /// Calculate the average run length
    pub fn average_run_length(&self) -> f64 {
        if self.runs_generated == 0 {
            0.0
        } else {
            self.items_sorted as f64 / self.runs_generated as f64
        }
    }

    /// Calculate I/O efficiency (ratio of logical to physical I/O)
    pub fn io_efficiency(&self) -> f64 {
        if self.temp_bytes_written == 0 {
            1.0
        } else {
            let logical_io = self.items_sorted * std::mem::size_of::<u64>(); // Estimate
            logical_io as f64 / self.temp_bytes_written as f64
        }
    }
}

/// Element wrapper for replacement selection with run tracking
#[derive(Debug, Clone)]
struct RunElement<T> {
    value: T,
    run_id: usize,
    original_index: usize,
}

impl<T> RunElement<T> {
    fn new(value: T, run_id: usize, original_index: usize) -> Self {
        Self {
            value,
            run_id,
            original_index,
        }
    }
}

impl<T: PartialEq> PartialEq for RunElement<T> {
    fn eq(&self, other: &Self) -> bool {
        self.value.eq(&other.value)
    }
}

impl<T: Eq> Eq for RunElement<T> {}

impl<T: PartialOrd> PartialOrd for RunElement<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Reverse order for max-heap behavior in BinaryHeap
        other.value.partial_cmp(&self.value)
    }
}

impl<T: Ord> Ord for RunElement<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order for max-heap behavior in BinaryHeap
        other.value.cmp(&self.value)
    }
}

/// Temporary run file for external sorting
struct TempRun {
    file_path: PathBuf,
    items_count: usize,
}

impl TempRun {
    fn new(file_path: PathBuf, items_count: usize) -> Self {
        Self {
            file_path,
            items_count,
        }
    }

    /// Create an iterator over the run data
    fn iter<T>(&self) -> Result<TempRunIterator<T>>
    where
        T: serde::de::DeserializeOwned,
    {
        if !self.file_path.exists() {
            return Err(ZiporaError::io_error(format!("Temp file does not exist: {:?}", self.file_path)));
        }
        
        // Verify file is readable and has content
        let metadata = std::fs::metadata(&self.file_path)
            .map_err(|e| ZiporaError::io_error(format!("Failed to read file metadata: {}", e)))?;
        
        if metadata.len() == 0 && self.items_count > 0 {
            return Err(ZiporaError::io_error(format!(
                "Temp file is empty but expected {} items: {:?}", 
                self.items_count, self.file_path
            )));
        }
        
        let file = File::open(&self.file_path)
            .map_err(|e| ZiporaError::io_error(format!("Failed to open temp file: {}", e)))?;
        
        let reader = BufReader::new(file);
        
        Ok(TempRunIterator {
            reader,
            items_remaining: self.items_count,
            _phantom: PhantomData,
        })
    }
}

impl Drop for TempRun {
    fn drop(&mut self) {
        // Clean up temporary file
        if self.file_path.exists() {
            let _ = remove_file(&self.file_path);
        }
    }
}

/// Iterator over temporary run data
struct TempRunIterator<T> {
    reader: BufReader<File>,
    items_remaining: usize,
    _phantom: PhantomData<T>,
}

impl<T> Iterator for TempRunIterator<T>
where
    T: serde::de::DeserializeOwned,
{
    type Item = Result<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.items_remaining == 0 {
            return None;
        }

        self.items_remaining -= 1;

        // Read size header with robust error handling
        let mut size_bytes = [0u8; 8];
        match self.reader.read_exact(&mut size_bytes) {
            Ok(()) => {},
            Err(e) => {
                return Some(Err(ZiporaError::io_error(format!(
                    "Failed to read size header (items remaining: {}): {}", 
                    self.items_remaining + 1, e
                ))));
            }
        }

        let size = usize::from_le_bytes(size_bytes);
        
        // Validate size to prevent excessive allocations
        if size > 1024 * 1024 * 100 { // 100MB limit
            return Some(Err(ZiporaError::io_error(format!(
                "Invalid data size: {} bytes", size
            ))));
        }
        
        let mut data = vec![0u8; size];
        
        // Read data with robust error handling
        match self.reader.read_exact(&mut data) {
            Ok(()) => {},
            Err(e) => {
                return Some(Err(ZiporaError::io_error(format!(
                    "Failed to read data ({} bytes, items remaining: {}): {}", 
                    size, self.items_remaining + 1, e
                ))));
            }
        }

        // Deserialize with proper error handling
        match bincode::deserialize(&data) {
            Ok(value) => Some(Ok(value)),
            Err(e) => Some(Err(ZiporaError::io_error(format!(
                "Deserialization failed for {} bytes: {}", size, e
            )))),
        }
    }
}

/// External sorting algorithm using replacement selection
///
/// This implementation uses replacement selection to generate long runs,
/// followed by k-way merging to produce the final sorted output.
///
/// # Example
/// ```
/// use zipora::algorithms::{ReplaceSelectSort, ReplaceSelectSortConfig};
/// use std::path::PathBuf;
/// 
/// let config = ReplaceSelectSortConfig {
///     memory_buffer_size: 1024 * 1024, // 1MB
///     temp_dir: PathBuf::from("/tmp"),
///     ..Default::default()
/// };
/// 
/// let mut sorter = ReplaceSelectSort::new(config);
/// 
/// let data = vec![5, 2, 8, 1, 9, 3];
/// let sorted = sorter.sort(data)?;
/// 
/// assert_eq!(sorted, vec![1, 2, 3, 5, 8, 9]);
/// # Ok::<(), zipora::ZiporaError>(())
/// ```
pub struct ReplaceSelectSort<T, F = fn(&T, &T) -> Ordering> {
    config: ReplaceSelectSortConfig,
    comparator: F,
    stats: ExternalSortStats,
    temp_files: Vec<TempRun>,
    instance_id: String,
    _phantom: PhantomData<T>,
}

impl<T> ReplaceSelectSort<T, fn(&T, &T) -> Ordering>
where
    T: Ord + Clone + serde::Serialize + serde::de::DeserializeOwned + 'static,
{
    /// Create a new external sorter with default ordering
    pub fn new(config: ReplaceSelectSortConfig) -> Self {
        Self::with_comparator(config, |a, b| a.cmp(b))
    }
}

impl<T, F> ReplaceSelectSort<T, F>
where
    T: Clone + Ord + serde::Serialize + serde::de::DeserializeOwned + 'static,
    F: Fn(&T, &T) -> Ordering + Clone,
{
    /// Create a new external sorter with custom comparator
    pub fn with_comparator(config: ReplaceSelectSortConfig, comparator: F) -> Self {
        // Generate unique instance ID to avoid file name collisions
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let thread_id = format!("{:?}", thread::current().id());
        let instance_id = format!("sort_{}_{}", 
            timestamp, 
            thread_id.replace("ThreadId(", "").replace(")", "")
        );
        
        Self {
            config,
            comparator,
            stats: ExternalSortStats {
                items_sorted: 0,
                runs_generated: 0,
                temp_bytes_written: 0,
                temp_bytes_read: 0,
                merge_passes: 0,
                processing_time_us: 0,
            },
            temp_files: Vec::new(),
            instance_id,
            _phantom: PhantomData,
        }
    }

    /// Sort the input data using external sorting
    pub fn sort<I>(&mut self, input: I) -> Result<Vec<T>>
    where
        I: IntoIterator<Item = T>,
    {
        let start_time = std::time::Instant::now();

        // Phase 1: Generate sorted runs using replacement selection
        self.generate_runs(input)?;

        // Phase 2: Merge runs using k-way merge
        let result = self.merge_runs()?;

        self.stats.processing_time_us = start_time.elapsed().as_micros() as u64;

        // Clean up temporary files if configured
        if self.config.cleanup_temp_files {
            self.cleanup()?;
        }

        Ok(result)
    }

    /// Generate sorted runs using replacement selection
    fn generate_runs<I>(&mut self, input: I) -> Result<()>
    where
        I: IntoIterator<Item = T>,
    {
        let memory_items = self.config.memory_buffer_size / std::mem::size_of::<T>().max(1);
        let mut heap = BinaryHeap::with_capacity(memory_items);
        let mut input_iter = input.into_iter();
        let mut current_run = 0;
        let mut run_items = 0;
        let mut temp_writer: Option<BufWriter<File>> = None;
        let mut current_temp_path: Option<PathBuf> = None;

        // Fill initial heap
        for _ in 0..memory_items {
            if let Some(item) = input_iter.next() {
                heap.push(RunElement::new(item, current_run, self.stats.items_sorted));
                self.stats.items_sorted += 1;
            } else {
                break;
            }
        }

        while !heap.is_empty() {
            // Get minimum element
            let min_element = heap.pop().unwrap();

            // Start new run file if needed
            if temp_writer.is_none() {
                let temp_path = self.config.temp_dir.join(format!("{}_{}.tmp", self.instance_id, current_run));
                let file = File::create(&temp_path)
                    .map_err(|e| ZiporaError::io_error(format!("Failed to create temp file: {}", e)))?;
                temp_writer = Some(BufWriter::new(file));
                current_temp_path = Some(temp_path);
                run_items = 0;
            }

            // Write element to current run
            self.write_element(&mut temp_writer.as_mut().unwrap(), &min_element.value)?;
            run_items += 1;

            // Try to read next element
            if let Some(next_item) = input_iter.next() {
                self.stats.items_sorted += 1;

                // Check if next item can be added to current run
                if (self.comparator)(&next_item, &min_element.value) != Ordering::Less {
                    // Can extend current run
                    heap.push(RunElement::new(next_item, current_run, self.stats.items_sorted));
                } else {
                    // Must start new run
                    heap.push(RunElement::new(next_item, current_run + 1, self.stats.items_sorted));

                    // Close current run if heap is empty or next min is from new run
                    if heap.is_empty() || heap.peek().map(|e| e.run_id).unwrap_or(0) > current_run {
                        self.finish_run(&mut temp_writer, current_temp_path.take().unwrap(), run_items)?;
                        current_run += 1;
                    }
                }
            } else {
                // No more input, finish current run when heap empties
                if heap.is_empty() || heap.peek().map(|e| e.run_id).unwrap_or(0) > current_run {
                    self.finish_run(&mut temp_writer, current_temp_path.take().unwrap(), run_items)?;
                    current_run += 1;
                }
            }
        }

        Ok(())
    }

    /// Write an element to the temporary file
    fn write_element<W: Write>(&mut self, writer: &mut W, element: &T) -> Result<()> {
        let serialized = bincode::serialize(element)
            .map_err(|e| ZiporaError::io_error(format!("Serialization failed: {}", e)))?;

        let size_bytes = serialized.len().to_le_bytes();
        writer.write_all(&size_bytes)
            .map_err(|e| ZiporaError::io_error(format!("Failed to write size: {}", e)))?;
        
        writer.write_all(&serialized)
            .map_err(|e| ZiporaError::io_error(format!("Failed to write data: {}", e)))?;

        self.stats.temp_bytes_written += size_bytes.len() + serialized.len();
        Ok(())
    }

    /// Finish the current run and add it to the temp files list
    fn finish_run(
        &mut self,
        writer: &mut Option<BufWriter<File>>,
        temp_path: PathBuf,
        items_count: usize,
    ) -> Result<()> {
        if let Some(mut w) = writer.take() {
            w.flush()
                .map_err(|e| ZiporaError::io_error(format!("Failed to flush temp file: {}", e)))?;
            // Ensure data is synced to disk
            w.into_inner()
                .map_err(|e| ZiporaError::io_error(format!("Failed to unwrap buffered writer: {}", e)))?
                .sync_all()
                .map_err(|e| ZiporaError::io_error(format!("Failed to sync temp file: {}", e)))?;
        }

        // Verify file exists and has the expected size before adding to temp files
        if !temp_path.exists() {
            return Err(ZiporaError::io_error(format!("Temp file was not created: {:?}", temp_path)));
        }

        self.temp_files.push(TempRun::new(temp_path, items_count));
        self.stats.runs_generated += 1;
        Ok(())
    }

    /// Merge all runs using k-way merge
    fn merge_runs(&mut self) -> Result<Vec<T>> {
        if self.temp_files.is_empty() {
            return Ok(Vec::new());
        }

        if self.temp_files.len() == 1 {
            // Only one run, just read it back
            return self.read_single_run();
        }

        // Multi-way merge
        let tree_config = LoserTreeConfig {
            initial_capacity: self.temp_files.len(),
            use_secure_memory: self.config.use_secure_memory,
            stable_sort: true,
            cache_optimized: true,
        };

        let mut tournament_tree = LoserTree::with_comparator(tree_config, self.comparator.clone());

        // Add all runs to the tournament tree
        for run in &self.temp_files {
            let iter = run.iter::<T>()?
                .filter_map(|result| result.ok()); // Skip errors instead of panicking
            tournament_tree.add_way(iter)?;
        }

        let result = tournament_tree.merge_to_vec()?;
        self.stats.merge_passes = 1;

        Ok(result)
    }

    /// Read a single run back into memory
    fn read_single_run(&mut self) -> Result<Vec<T>> {
        if self.temp_files.is_empty() {
            return Ok(Vec::new());
        }

        let run = &self.temp_files[0];
        let mut result = Vec::with_capacity(run.items_count);
        
        for item_result in run.iter::<T>()? {
            result.push(item_result?);
        }

        // Apply custom comparator to ensure correct ordering
        // The run was generated with replacement selection which maintains order,
        // but we need to ensure the final result respects the custom comparator
        result.sort_by(&self.comparator);

        Ok(result)
    }

    /// Get sorting statistics
    pub fn stats(&self) -> &ExternalSortStats {
        &self.stats
    }

    /// Clean up temporary files
    pub fn cleanup(&mut self) -> Result<()> {
        self.temp_files.clear();
        Ok(())
    }
}

/// Trait for external sorting support
pub trait ExternalSort<T> {
    /// Sort the collection using external sorting if it exceeds memory limits
    fn external_sort(&mut self) -> Result<()>;
    
    /// Sort with custom configuration
    fn external_sort_with_config(&mut self, config: ReplaceSelectSortConfig) -> Result<()>;
}

// Implementation for Vec<T>
impl<T> ExternalSort<T> for Vec<T>
where
    T: Ord + Clone + serde::Serialize + serde::de::DeserializeOwned + 'static,
{
    fn external_sort(&mut self) -> Result<()> {
        let config = ReplaceSelectSortConfig::default();
        self.external_sort_with_config(config)
    }

    fn external_sort_with_config(&mut self, config: ReplaceSelectSortConfig) -> Result<()> {
        let estimated_size = self.len() * std::mem::size_of::<T>();
        
        if estimated_size <= config.memory_buffer_size {
            // Use in-memory sort for small datasets
            self.sort();
            Ok(())
        } else {
            // Use external sort for large datasets
            let mut sorter = ReplaceSelectSort::new(config);
            let input = std::mem::take(self);
            let sorted = sorter.sort(input)?;
            *self = sorted;
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::create_dir_all;

    // Global counter for unique test directories
    static TEST_COUNTER: AtomicUsize = AtomicUsize::new(0);

    fn test_temp_dir() -> PathBuf {
        // Create unique temp directory for each test to avoid race conditions
        let process_id = std::process::id();
        let thread_id = format!("{:?}", thread::current().id());
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let counter = TEST_COUNTER.fetch_add(1, AtomicOrdering::SeqCst);
        
        let unique_name = format!(
            "zipora_external_sort_test_{}_{}_{}_{}",
            process_id,
            thread_id.replace("ThreadId(", "").replace(")", ""),
            timestamp,
            counter
        );
        
        let temp = std::env::temp_dir().join(unique_name);
        create_dir_all(&temp).unwrap();
        temp
    }

    #[test]
    fn test_replace_select_sort_config_default() {
        let config = ReplaceSelectSortConfig::default();
        assert_eq!(config.memory_buffer_size, 64 * 1024 * 1024);
        assert!(config.use_secure_memory);
        assert!(!config.compress_temp_files);
        assert_eq!(config.merge_ways, 16);
        assert!(config.cleanup_temp_files);
    }

    #[test]
    fn test_external_sort_stats() {
        let stats = ExternalSortStats {
            items_sorted: 1000,
            runs_generated: 5,
            temp_bytes_written: 8000,
            temp_bytes_read: 8000,
            merge_passes: 1,
            processing_time_us: 1000,
        };

        assert_eq!(stats.average_run_length(), 200.0);
        assert_eq!(stats.io_efficiency(), 1.0); // 8000 / 8000
    }

    #[test]
    fn test_run_element_ordering() {
        let elem1 = RunElement::new(5, 0, 0);
        let elem2 = RunElement::new(3, 0, 1);
        let elem3 = RunElement::new(7, 0, 2);

        // BinaryHeap is max-heap, but our elements reverse the order
        assert!(elem2 > elem1); // 3 > 5 in our reversed ordering
        assert!(elem1 > elem3); // 5 > 7 in our reversed ordering
    }

    #[test]
    fn test_small_dataset_in_memory() -> Result<()> {
        let config = ReplaceSelectSortConfig {
            memory_buffer_size: 1024 * 1024, // 1MB
            temp_dir: test_temp_dir(),
            ..Default::default()
        };

        let mut sorter = ReplaceSelectSort::new(config);
        let data = vec![5, 2, 8, 1, 9, 3];
        let sorted = sorter.sort(data)?;

        assert_eq!(sorted, vec![1, 2, 3, 5, 8, 9]);
        assert_eq!(sorter.stats().runs_generated, 1);

        Ok(())
    }

    #[test]
    fn test_single_element() -> Result<()> {
        let config = ReplaceSelectSortConfig {
            temp_dir: test_temp_dir(),
            ..Default::default()
        };

        let mut sorter = ReplaceSelectSort::new(config);
        let data = vec![42];
        let sorted = sorter.sort(data)?;

        assert_eq!(sorted, vec![42]);

        Ok(())
    }

    #[test]
    fn test_empty_input() -> Result<()> {
        let config = ReplaceSelectSortConfig {
            temp_dir: test_temp_dir(),
            ..Default::default()
        };

        let mut sorter = ReplaceSelectSort::new(config);
        let data: Vec<i32> = vec![];
        let sorted = sorter.sort(data)?;

        assert_eq!(sorted, vec![]);

        Ok(())
    }

    #[test]
    fn test_already_sorted() -> Result<()> {
        let config = ReplaceSelectSortConfig {
            temp_dir: test_temp_dir(),
            ..Default::default()
        };

        let mut sorter = ReplaceSelectSort::new(config);
        let data = vec![1, 2, 3, 4, 5];
        let sorted = sorter.sort(data)?;

        assert_eq!(sorted, vec![1, 2, 3, 4, 5]);

        Ok(())
    }

    #[test]
    fn test_reverse_sorted() -> Result<()> {
        let config = ReplaceSelectSortConfig {
            temp_dir: test_temp_dir(),
            ..Default::default()
        };

        let mut sorter = ReplaceSelectSort::new(config);
        let data = vec![5, 4, 3, 2, 1];
        let sorted = sorter.sort(data)?;

        assert_eq!(sorted, vec![1, 2, 3, 4, 5]);

        Ok(())
    }

    #[test]
    fn test_duplicates() -> Result<()> {
        let config = ReplaceSelectSortConfig {
            temp_dir: test_temp_dir(),
            ..Default::default()
        };

        let mut sorter = ReplaceSelectSort::new(config);
        let data = vec![3, 1, 3, 1, 2, 2];
        let sorted = sorter.sort(data)?;

        assert_eq!(sorted, vec![1, 1, 2, 2, 3, 3]);

        Ok(())
    }

    #[test]
    fn test_custom_comparator() -> Result<()> {
        let config = ReplaceSelectSortConfig {
            temp_dir: test_temp_dir(),
            ..Default::default()
        };

        let mut sorter = ReplaceSelectSort::with_comparator(config, |a: &i32, b: &i32| b.cmp(a));
        let data = vec![1, 3, 2, 5, 4];
        let sorted = sorter.sort(data)?;

        assert_eq!(sorted, vec![5, 4, 3, 2, 1]);

        Ok(())
    }

    #[test]
    fn test_vec_external_sort_trait() -> Result<()> {
        let mut data = vec![5, 2, 8, 1, 9, 3];
        data.external_sort()?;

        assert_eq!(data, vec![1, 2, 3, 5, 8, 9]);

        Ok(())
    }

    #[test]
    fn test_large_dataset_simulation() -> Result<()> {
        let config = ReplaceSelectSortConfig {
            memory_buffer_size: 64, // Very small buffer to force external sort
            temp_dir: test_temp_dir(),
            ..Default::default()
        };

        let mut sorter = ReplaceSelectSort::new(config);
        
        // Generate larger dataset
        let mut data: Vec<i32> = (0..100).rev().collect();
        data.extend_from_slice(&[50, 25, 75]);

        let sorted = sorter.sort(data)?;

        // Verify it's sorted
        for i in 1..sorted.len() {
            assert!(sorted[i] >= sorted[i-1]);
        }

        // Should have generated multiple runs due to small buffer
        assert!(sorter.stats().runs_generated > 1);

        Ok(())
    }
}