//! External sorting implementation for datasets larger than memory
//!
//! This module provides efficient external sorting using replacement selection
//! and k-way merging. The algorithm is designed to handle datasets that exceed
//! available memory by using disk-based temporary storage and efficient
//! merge operations.

use crate::algorithms::tournament_tree::{EnhancedLoserTree, LoserTreeConfig};
use crate::error::{Result, ZiporaError};
use std::cell::RefCell;
use std::cmp::Ordering;
use std::fs::{File, remove_file};
use std::io::{BufReader, BufWriter, Read, Write};
use std::marker::PhantomData;
use std::path::PathBuf;
use std::rc::Rc;
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

/// Where a merge's run iterators store a read or deserialization error; the
/// merge checks it once the loser tree is drained.
type ReadErrorSlot = Rc<RefCell<Option<ZiporaError>>>;

/// Configuration for external sorting operations
#[derive(Debug, Clone)]
pub struct ReplaceSelectSortConfig {
    /// Size of the main memory buffer in bytes
    pub memory_buffer_size: usize,
    /// Directory for temporary files
    pub temp_dir: PathBuf,
    /// Limitation: has no effect. The run buffer and the merge tree use the
    /// global allocator; the flag is only passed on to
    /// `LoserTreeConfig::use_secure_memory`, which has no effect either.
    pub use_secure_memory: bool,
    /// Limitation: has no effect; temporary files are never compressed.
    pub compress_temp_files: bool,
    /// Maximum number of runs merged at once (values below 2 count as 2).
    /// With more runs than this, intermediate passes merge groups of runs
    /// into new temporary runs until at most `merge_ways` remain, so at most
    /// `merge_ways` run files are open for reading at a time (plus the
    /// output file of an intermediate pass). Runs merged by an intermediate
    /// pass are deleted right away, whatever `cleanup_temp_files` says.
    pub merge_ways: usize,
    /// Delete the run files at the end of a successful `sort`. When false,
    /// the runs of the last `sort` stay on disk until the next `sort`,
    /// [`cleanup`](ReplaceSelectSort::cleanup), or drop of the sorter.
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

/// Statistics for external sort operations.
///
/// `items_sorted`, `runs_generated` and `temp_bytes_written` add up over
/// all `sort` calls on a sorter; `merge_passes` and `processing_time_us`
/// describe the last call.
#[derive(Debug, Clone)]
pub struct ExternalSortStats {
    /// Number of items sorted
    pub items_sorted: usize,
    /// Number of runs generated (runs written by intermediate merge passes
    /// are not counted)
    pub runs_generated: usize,
    /// Bytes written to temporary files, including intermediate merge passes
    pub temp_bytes_written: usize,
    /// Limitation: not tracked; always 0.
    pub temp_bytes_read: usize,
    /// Number of merge passes in the last `sort`: intermediate passes plus
    /// the final merge (0 when there was at most one run)
    pub merge_passes: usize,
    /// Wall time of the last `sort`, in microseconds
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

    /// Estimated I/O efficiency: `items_sorted * 8` bytes (as if every item
    /// were a `u64`) divided by `temp_bytes_written`, which includes each
    /// record's `usize` length prefix and every intermediate merge pass; 1.0
    /// when nothing was written.
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
}

impl<T> RunElement<T> {
    fn new(value: T, run_id: usize) -> Self {
        Self { value, run_id }
    }
}

impl<T: PartialEq> PartialEq for RunElement<T> {
    fn eq(&self, other: &Self) -> bool {
        self.run_id == other.run_id && self.value.eq(&other.value)
    }
}

impl<T: Eq> Eq for RunElement<T> {}

impl<T: PartialOrd> PartialOrd for RunElement<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match other.run_id.partial_cmp(&self.run_id) {
            Some(Ordering::Equal) => other.value.partial_cmp(&self.value),
            ord => ord,
        }
    }
}

impl<T: Ord> Ord for RunElement<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .run_id
            .cmp(&self.run_id)
            .then_with(|| other.value.cmp(&self.value))
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
            return Err(ZiporaError::io_error(format!(
                "Temp file does not exist: {:?}",
                self.file_path
            )));
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
            Ok(()) => {}
            Err(e) => {
                return Some(Err(ZiporaError::io_error(format!(
                    "Failed to read size header (items remaining: {}): {}",
                    self.items_remaining + 1,
                    e
                ))));
            }
        }

        let size = usize::from_le_bytes(size_bytes);

        // Validate size to prevent excessive allocations
        if size > 1024 * 1024 * 100 {
            // 100MB limit
            return Some(Err(ZiporaError::io_error(format!(
                "Invalid data size: {} bytes",
                size
            ))));
        }

        let mut data = vec![0u8; size];

        // Read data with robust error handling
        match self.reader.read_exact(&mut data) {
            Ok(()) => {}
            Err(e) => {
                return Some(Err(ZiporaError::io_error(format!(
                    "Failed to read data ({} bytes, items remaining: {}): {}",
                    size,
                    self.items_remaining + 1,
                    e
                ))));
            }
        }

        // Deserialize with proper error handling
        match bincode::deserialize(&data) {
            Ok(value) => Some(Ok(value)),
            Err(e) => Some(Err(ZiporaError::io_error(format!(
                "Deserialization failed for {} bytes: {}",
                size, e
            )))),
        }
    }
}

/// External sorting algorithm using replacement selection
///
/// This implementation uses replacement selection to generate long runs,
/// followed by k-way merges of at most
/// [`merge_ways`](ReplaceSelectSortConfig::merge_ways) runs at a time to
/// produce the final sorted output.
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
    /// Intermediate merge files created so far; numbers the next one.
    merge_files_created: usize,
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
            .map(|d| d.as_nanos())
            .unwrap_or_else(|_| {
                // Fallback for systems where time might go backwards (e.g. during NTP sync)
                let start = std::time::Instant::now();
                start.elapsed().as_nanos()
            });
        let thread_id = format!("{:?}", thread::current().id());
        let instance_id = format!(
            "sort_{}_{}",
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
            merge_files_created: 0,
            _phantom: PhantomData,
        }
    }

    /// Sort the input data using external sorting.
    ///
    /// Runs left on disk by an earlier call (`cleanup_temp_files: false`, or
    /// an error) are deleted first.
    pub fn sort<I>(&mut self, input: I) -> Result<Vec<T>>
    where
        I: IntoIterator<Item = T>,
    {
        let start_time = std::time::Instant::now();

        // Earlier runs would be merged into this result, and this call's run
        // files reuse their names.
        self.cleanup()?;

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

    /// Push an element onto the replacement-selection min-heap ordered by `(run_id, comparator)`.
    fn heap_push(heap: &mut Vec<RunElement<T>>, elem: RunElement<T>, comparator: &F) {
        heap.push(elem);
        let mut idx = heap.len() - 1;
        while idx > 0 {
            let parent = (idx - 1) / 2;
            let ord = heap[idx]
                .run_id
                .cmp(&heap[parent].run_id)
                .then_with(|| comparator(&heap[idx].value, &heap[parent].value));
            if ord == Ordering::Less {
                heap.swap(idx, parent);
                idx = parent;
            } else {
                break;
            }
        }
    }

    /// Pop the minimum element from the replacement-selection min-heap ordered by `(run_id, comparator)`.
    fn heap_pop(heap: &mut Vec<RunElement<T>>, comparator: &F) -> Option<RunElement<T>> {
        if heap.is_empty() {
            return None;
        }
        let min = heap.swap_remove(0);
        Self::sift_down(heap, 0, comparator);
        Some(min)
    }

    /// Restore the min-heap order (by `(run_id, comparator)`) of the subtree
    /// rooted at `idx`, given that both of its child subtrees are heaps.
    fn sift_down(heap: &mut [RunElement<T>], mut idx: usize, comparator: &F) {
        let len = heap.len();
        loop {
            let left = 2 * idx + 1;
            let right = 2 * idx + 2;
            let mut smallest = idx;

            if left < len {
                let ord = heap[left]
                    .run_id
                    .cmp(&heap[smallest].run_id)
                    .then_with(|| comparator(&heap[left].value, &heap[smallest].value));
                if ord == Ordering::Less {
                    smallest = left;
                }
            }
            if right < len {
                let ord = heap[right]
                    .run_id
                    .cmp(&heap[smallest].run_id)
                    .then_with(|| comparator(&heap[right].value, &heap[smallest].value));
                if ord == Ordering::Less {
                    smallest = right;
                }
            }
            if smallest != idx {
                heap.swap(idx, smallest);
                idx = smallest;
            } else {
                break;
            }
        }
    }

    /// Generate sorted runs using replacement selection
    fn generate_runs<I>(&mut self, input: I) -> Result<()>
    where
        I: IntoIterator<Item = T>,
    {
        let memory_items = (self.config.memory_buffer_size / std::mem::size_of::<T>().max(1)).max(1);
        let mut input_iter = input.into_iter().peekable();
        let mut current_run = 0;
        let mut run_items = 0;
        // The run being written owns its file from creation, so an error
        // deletes it. Bound before the writer so that the writer is dropped
        // (closed) first.
        let mut current_run_file: Option<TempRun> = None;
        let mut temp_writer: Option<BufWriter<File>> = None;

        // Fill the buffer. An exact upper size hint avoids reserving the whole
        // buffer for a small input; without one, reserve the buffer as before.
        let capacity = memory_items.min(input_iter.size_hint().1.unwrap_or(memory_items));
        let mut heap: Vec<RunElement<T>> = Vec::with_capacity(capacity);
        heap.extend(
            input_iter
                .by_ref()
                .take(memory_items)
                .map(|item| RunElement::new(item, current_run)),
        );
        self.stats.items_sorted += heap.len();

        // The whole input fits in the buffer: it all goes to one run, so one
        // stable sort replaces the heap (S20-R5). Unlike the heap path, this
        // keeps comparator-equal elements in input order.
        if input_iter.peek().is_none() {
            if heap.is_empty() {
                return Ok(());
            }
            heap.sort_by(|a, b| (self.comparator)(&a.value, &b.value));
            let (writer, temp_path) = self.create_run_file(current_run)?;
            let run_file = TempRun::new(temp_path, 0);
            let mut writer = writer;
            for element in &heap {
                self.write_element(&mut writer, &element.value)?;
            }
            return self.finish_run(&mut Some(writer), run_file, heap.len());
        }

        // Bottom-up heap construction, O(n) instead of n pushes.
        for idx in (0..heap.len() / 2).rev() {
            Self::sift_down(&mut heap, idx, &self.comparator);
        }

        while let Some(min_element) = Self::heap_pop(&mut heap, &self.comparator) {
            if min_element.run_id > current_run {
                if temp_writer.is_some() {
                    self.finish_run(
                        &mut temp_writer,
                        current_run_file.take().ok_or_else(|| {
                            ZiporaError::invalid_state(
                                "external sort: temp path missing at run boundary",
                            )
                        })?,
                        run_items,
                    )?;
                }
                current_run = min_element.run_id;
            }

            // Start new run file if needed
            if temp_writer.is_none() {
                let (writer, temp_path) = self.create_run_file(current_run)?;
                current_run_file = Some(TempRun::new(temp_path, 0));
                temp_writer = Some(writer);
                run_items = 0;
            }

            // Write element to current run
            self.write_element(
                temp_writer.as_mut().ok_or_else(|| {
                    ZiporaError::invalid_state("external sort: temp writer missing for active run")
                })?,
                &min_element.value,
            )?;
            run_items += 1;

            // Try to read next element
            if let Some(next_item) = input_iter.next() {
                self.stats.items_sorted += 1;
                let target_run =
                    if (self.comparator)(&next_item, &min_element.value) != Ordering::Less {
                        current_run
                    } else {
                        current_run + 1
                    };
                Self::heap_push(
                    &mut heap,
                    RunElement::new(next_item, target_run),
                    &self.comparator,
                );
            }
        }

        if temp_writer.is_some() {
            self.finish_run(
                &mut temp_writer,
                current_run_file.take().ok_or_else(|| {
                    ZiporaError::invalid_state("external sort: temp path missing at final run")
                })?,
                run_items,
            )?;
        }

        Ok(())
    }

    /// Create the temporary file for run `run_id`.
    fn create_run_file(&self, run_id: usize) -> Result<(BufWriter<File>, PathBuf)> {
        let temp_path = self
            .config
            .temp_dir
            .join(format!("{}_{}.tmp", self.instance_id, run_id));
        Self::create_temp_file(temp_path)
    }

    /// Create the temporary file for the next intermediate merge output.
    /// Merge files are numbered separately (`{instance_id}_m{n}.tmp`), so
    /// their names never collide with run files or with each other.
    fn create_merge_file(&mut self) -> Result<(BufWriter<File>, PathBuf)> {
        let temp_path = self.config.temp_dir.join(format!(
            "{}_m{}.tmp",
            self.instance_id, self.merge_files_created
        ));
        self.merge_files_created += 1;
        Self::create_temp_file(temp_path)
    }

    /// Create (or truncate) the temporary file at `temp_path`.
    fn create_temp_file(temp_path: PathBuf) -> Result<(BufWriter<File>, PathBuf)> {
        let file = File::create(&temp_path)
            .map_err(|e| ZiporaError::io_error(format!("Failed to create temp file: {}", e)))?;
        Ok((BufWriter::new(file), temp_path))
    }

    /// Write an element to the temporary file
    fn write_element<W: Write>(&mut self, writer: &mut W, element: &T) -> Result<()> {
        let serialized = bincode::serialize(element)
            .map_err(|e| ZiporaError::io_error(format!("Serialization failed: {}", e)))?;

        let size_bytes = serialized.len().to_le_bytes();
        writer
            .write_all(&size_bytes)
            .map_err(|e| ZiporaError::io_error(format!("Failed to write size: {}", e)))?;

        writer
            .write_all(&serialized)
            .map_err(|e| ZiporaError::io_error(format!("Failed to write data: {}", e)))?;

        self.stats.temp_bytes_written += size_bytes.len() + serialized.len();
        Ok(())
    }

    /// Flush a run file's buffer and sync the file to disk.
    fn sync_run_file(mut writer: BufWriter<File>) -> Result<()> {
        writer
            .flush()
            .map_err(|e| ZiporaError::io_error(format!("Failed to flush temp file: {}", e)))?;
        // Ensure data is synced to disk
        writer
            .into_inner()
            .map_err(|e| ZiporaError::io_error(format!("Failed to unwrap buffered writer: {}", e)))?
            .sync_all()
            .map_err(|e| ZiporaError::io_error(format!("Failed to sync temp file: {}", e)))
    }

    /// Finish the current run and add it to the temp files list
    fn finish_run(
        &mut self,
        writer: &mut Option<BufWriter<File>>,
        mut run: TempRun,
        items_count: usize,
    ) -> Result<()> {
        if let Some(w) = writer.take() {
            Self::sync_run_file(w)?;
        }

        // Verify file exists and has the expected size before adding to temp files
        if !run.file_path.exists() {
            return Err(ZiporaError::io_error(format!(
                "Temp file was not created: {:?}",
                run.file_path
            )));
        }

        run.items_count = items_count;
        self.temp_files.push(run);
        self.stats.runs_generated += 1;
        Ok(())
    }

    /// Merge all runs into the sorted output, at most `merge_ways` (at
    /// least 2) runs at a time.
    ///
    /// While there are more runs than that, intermediate passes merge
    /// contiguous groups of runs into new temporary runs, each taking the
    /// place of its group; the final merge then reads the remaining runs.
    /// Every merge gives ties to the earlier run, so the output is the same
    /// as merging all runs at once.
    fn merge_runs(&mut self) -> Result<Vec<T>> {
        let ways = self.config.merge_ways.max(2);
        let mut passes = 0;
        while self.temp_files.len() > ways {
            self.merge_pass(ways)?;
            passes += 1;
        }

        let result = match self.temp_files.len() {
            0 => Vec::new(),
            // Only one run, just read it back
            1 => self.read_single_run()?,
            _ => {
                let (mut tree, read_error) = self.open_merge_tree(&self.temp_files)?;
                let total = self.temp_files.iter().map(|run| run.items_count).sum();
                let mut result = Vec::with_capacity(total);
                while let Some(value) = tree.pop()? {
                    result.push(value);
                }
                if let Some(err) = read_error.borrow_mut().take() {
                    return Err(err);
                }
                passes += 1;
                result
            }
        };
        self.stats.merge_passes = passes;
        Ok(result)
    }

    /// One intermediate pass over more than `ways` runs: merge contiguous
    /// groups of at most `ways` runs from the front, each into a new run
    /// that takes the group's place, until the run count is the largest
    /// power of `ways` below it. Every later pass is then a full `ways`-way
    /// merge, which keeps the data rewritten to a minimum for runs of equal
    /// length.
    fn merge_pass(&mut self, ways: usize) -> Result<()> {
        let len = self.temp_files.len();
        // Largest power of `ways` below `len`; at least `ways`, as `len > ways`.
        let mut target = ways;
        while let Some(next) = target.checked_mul(ways) {
            if next >= len {
                break;
            }
            target = next;
        }

        let mut to_remove = len - target;
        let mut runs = std::mem::take(&mut self.temp_files).into_iter();
        let mut merged = Vec::with_capacity(target);
        while to_remove > 0 {
            let group: Vec<TempRun> = runs.by_ref().take(ways.min(to_remove + 1)).collect();
            if group.len() < 2 {
                merged.extend(group);
                break;
            }
            to_remove -= group.len() - 1;
            merged.push(self.merge_group_to_run(&group)?);
            // `group` is dropped here, which deletes its files.
        }
        merged.extend(runs);
        self.temp_files = merged;
        Ok(())
    }

    /// Merge `group` into a new temporary run, with `group.len()` run files
    /// open for reading plus the new run's file.
    fn merge_group_to_run(&mut self, group: &[TempRun]) -> Result<TempRun> {
        let (mut tree, read_error) = self.open_merge_tree(group)?;
        let (writer, path) = self.create_merge_file()?;
        // `run` owns the new file, so the file is deleted if the merge fails;
        // `writer` is bound after `run` so that it is dropped (closed) first.
        let mut run = TempRun::new(path, 0);
        let mut writer = writer;
        while let Some(value) = tree.pop()? {
            self.write_element(&mut writer, &value)?;
            run.items_count += 1;
        }
        if let Some(err) = read_error.borrow_mut().take() {
            return Err(err);
        }
        Self::sync_run_file(writer)?;
        Ok(run)
    }

    /// Open `runs` and build a stable loser tree over them (ties go to the
    /// earlier run). A run's iterator stops at a read or deserialization
    /// error and stores it in the returned slot, which the caller checks
    /// after draining the tree.
    fn open_merge_tree(
        &self,
        runs: &[TempRun],
    ) -> Result<(EnhancedLoserTree<T, F>, ReadErrorSlot)> {
        let tree_config = LoserTreeConfig {
            initial_capacity: runs.len(),
            use_secure_memory: self.config.use_secure_memory,
            stable_sort: true,
            cache_optimized: true,
            use_simd: cfg!(feature = "simd"),
            prefetch_distance: 2,
            alignment: 64,
        };

        let mut tournament_tree =
            EnhancedLoserTree::with_comparator(tree_config, self.comparator.clone());

        let read_error: ReadErrorSlot = Rc::new(RefCell::new(None));

        // Add the runs to the tournament tree while capturing any I/O or deserialization error
        for run in runs {
            let err_slot = Rc::clone(&read_error);
            let iter = run.iter::<T>()?.scan(err_slot, |slot, item_res| match item_res {
                Ok(val) => Some(val),
                Err(e) => {
                    *slot.borrow_mut() = Some(e);
                    None
                }
            });
            tournament_tree.add_way(iter)?;
        }

        Ok((tournament_tree, read_error))
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
        let elem1 = RunElement::new(5, 0);
        let elem2 = RunElement::new(3, 0);
        let elem3 = RunElement::new(7, 0);

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

        assert_eq!(sorted, Vec::<i32>::new());

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
            assert!(sorted[i] >= sorted[i - 1]);
        }

        // Should have generated multiple runs due to small buffer
        assert!(sorter.stats().runs_generated > 1);

        Ok(())
    }

    #[test]
    fn test_c8_9_external_sort_custom_comparator_multi_run_and_tiny_buffer() -> Result<()> {
        // 1. memory_buffer_size < size_of::<i32>() previously computed memory_items == 0
        // and silently discarded the entire input (`vec![]`).
        let tiny_config = ReplaceSelectSortConfig {
            memory_buffer_size: 1,
            temp_dir: test_temp_dir(),
            ..Default::default()
        };
        let mut tiny_sorter = ReplaceSelectSort::new(tiny_config);
        let input = vec![9, 1, 8, 2, 7, 3, 6, 4, 5];
        assert_eq!(tiny_sorter.sort(input)?, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);

        // 2. Multi-run sort with custom descending comparator previously ignored both
        // `run_id` and `self.comparator` inside `generate_runs`, producing unsorted output.
        let desc_config = ReplaceSelectSortConfig {
            memory_buffer_size: 16, // 4 i32s in memory -> forces multiple runs
            temp_dir: test_temp_dir(),
            ..Default::default()
        };
        let mut desc_sorter =
            ReplaceSelectSort::with_comparator(desc_config, |a: &i32, b: &i32| b.cmp(a));
        let data: Vec<i32> = (0..40).map(|i| (i * 17 + 3) % 29).collect();
        let mut expected = data.clone();
        expected.sort_by(|a, b| b.cmp(a));

        let sorted = desc_sorter.sort(data)?;
        assert!(desc_sorter.stats().runs_generated > 1);
        assert_eq!(sorted, expected);

        Ok(())
    }

    #[test]
    fn test_s20_r5_in_memory_fast_path_matches_heap_path() -> Result<()> {
        // Guard for the S20-R5 in-memory fast path: inputs just below, at and
        // above the buffer size take the sort-once path or the heap path. With
        // a descending comparator (not `T::cmp`) all must come out in the
        // comparator's order, with every item counted and, when the input
        // fits, exactly one run (none for an empty input).
        for b in [1usize, 2, 16, 100] {
            for n in [b - 1, b, b + 1, 3 * b + 5] {
                let config = ReplaceSelectSortConfig {
                    memory_buffer_size: b * std::mem::size_of::<i32>(),
                    temp_dir: test_temp_dir(),
                    use_secure_memory: false,
                    ..Default::default()
                };
                let mut sorter =
                    ReplaceSelectSort::with_comparator(config, |x: &i32, y: &i32| y.cmp(x));
                let data: Vec<i32> = (0..n as i32).map(|i| (i * 37 + 11) % 23 - 11).collect();
                let mut want = data.clone();
                want.sort_by(|x, y| y.cmp(x));

                let got = sorter.sort(data)?;
                let ctx = format!("buffer_items={b} n={n}");
                assert_eq!(got, want, "{ctx}");
                assert_eq!(sorter.stats().items_sorted, n, "{ctx}");
                if n <= b {
                    assert_eq!(sorter.stats().runs_generated, usize::from(n > 0), "{ctx}");
                }
            }
        }
        Ok(())
    }

    #[test]
    fn test_s20_r9_merge_passes_bounded_by_merge_ways() -> Result<()> {
        // S20-R9: `merge_ways` was never read and every run was merged at
        // once, so `merge_passes` was always 1. 256 descending items with a
        // 4-item buffer give exactly 64 runs; merging at most `ways` runs at
        // a time takes ceil(log_ways(64)) passes (`merge_ways` < 2 counts as
        // 2), and no temporary file is left after the sort.
        let data: Vec<u64> = (0..256).rev().collect();
        let want: Vec<u64> = (0..256).collect();
        for (ways, passes) in [
            (0, 6),
            (1, 6),
            (2, 6),
            (3, 4),
            (4, 3),
            (8, 2),
            (16, 2),
            (63, 2),
            (64, 1),
            (usize::MAX, 1),
        ] {
            let dir = test_temp_dir();
            let config = ReplaceSelectSortConfig {
                memory_buffer_size: 4 * std::mem::size_of::<u64>(),
                temp_dir: dir.clone(),
                merge_ways: ways,
                ..Default::default()
            };
            let mut sorter = ReplaceSelectSort::new(config);
            let got = sorter.sort(data.clone())?;
            let ctx = format!("merge_ways={ways}");
            assert_eq!(got, want, "{ctx}");
            assert_eq!(sorter.stats().runs_generated, 64, "{ctx}");
            assert_eq!(sorter.stats().merge_passes, passes, "{ctx}");
            assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0, "{ctx}");
            std::fs::remove_dir(&dir).unwrap();
        }

        // Comparator-equal items spread over many runs: every grouping must
        // give the order of the single all-run merge (ties go to the earlier
        // run), which is sorted by key and a permutation of the input.
        let tied: Vec<u64> = (0..256).map(|i| ((i * 7 % 5) << 32) | i).collect();
        let sort_tied = |ways: usize| -> Result<(Vec<u64>, ExternalSortStats)> {
            let config = ReplaceSelectSortConfig {
                memory_buffer_size: 4 * std::mem::size_of::<u64>(),
                temp_dir: test_temp_dir(),
                merge_ways: ways,
                ..Default::default()
            };
            let by_key = |a: &u64, b: &u64| (a >> 32).cmp(&(b >> 32));
            let mut sorter = ReplaceSelectSort::with_comparator(config, by_key);
            let out = sorter.sort(tied.clone())?;
            Ok((out, sorter.stats().clone()))
        };
        let (single, stats) = sort_tied(usize::MAX)?;
        assert_eq!(stats.merge_passes, 1);
        assert!(
            stats.runs_generated > 16,
            "runs_generated={}",
            stats.runs_generated
        );
        assert!(single.windows(2).all(|w| w[0] >> 32 <= w[1] >> 32));
        let (mut got_items, mut want_items) = (single.clone(), tied.clone());
        got_items.sort_unstable();
        want_items.sort_unstable();
        assert_eq!(got_items, want_items);
        for ways in [2, 3, 5, 16] {
            let (got, stats) = sort_tied(ways)?;
            assert!(stats.merge_passes > 1, "merge_ways={ways}: {stats:?}");
            assert_eq!(got, single, "merge_ways={ways}");
        }
        Ok(())
    }

    /// S20-R9: every run used to be opened at once, so with a limit of 64
    /// open files, sorting 200 runs failed with "Too many open files". The
    /// limit is set in a child process (the ignored test below) so that it
    /// does not affect other tests.
    #[test]
    #[cfg(unix)]
    #[cfg_attr(miri, ignore = "spawns a process")]
    fn test_s20_r9_merge_respects_open_file_limit() {
        let exe = std::env::current_exe().unwrap();
        let out = std::process::Command::new("sh")
            .arg("-c")
            .arg(
                "ulimit -n 64 && exec \"$0\" --exact \
                 algorithms::external_sort::tests::test_s20_r9_open_file_limit_child \
                 --ignored --test-threads=1",
            )
            .arg(&exe)
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            out.status.success() && stdout.contains("test result: ok. 1 passed"),
            "child: {}\n--- stdout\n{stdout}\n--- stderr\n{stderr}",
            out.status
        );
    }

    /// Child of `test_s20_r9_merge_respects_open_file_limit`, which runs it
    /// under `ulimit -n 64`: 1600 descending items with an 8-item buffer give
    /// 200 runs, merged with the default `merge_ways` (16).
    #[test]
    #[ignore = "run by test_s20_r9_merge_respects_open_file_limit under ulimit -n 64"]
    fn test_s20_r9_open_file_limit_child() -> Result<()> {
        let dir = test_temp_dir();
        let config = ReplaceSelectSortConfig {
            memory_buffer_size: 8 * std::mem::size_of::<u64>(),
            temp_dir: dir.clone(),
            ..Default::default()
        };
        let mut sorter = ReplaceSelectSort::new(config);
        let sorted = sorter.sort((0..1600u64).rev())?;
        assert_eq!(sorted, (0..1600u64).collect::<Vec<_>>());
        assert_eq!(sorter.stats().runs_generated, 200);
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
        std::fs::remove_dir(&dir).unwrap();
        Ok(())
    }

    #[test]
    fn test_s20_r9b_sort_again_without_cleanup() -> Result<()> {
        // S20-R9b: with `cleanup_temp_files: false` the runs of one `sort`
        // stayed in the run list, so the next `sort` on the same sorter
        // merged them into its own result (and had truncated some of their
        // files by reusing the run file names).
        for ways in [16, 2] {
            let dir = test_temp_dir();
            let config = ReplaceSelectSortConfig {
                memory_buffer_size: 4 * std::mem::size_of::<u64>(),
                temp_dir: dir.clone(),
                merge_ways: ways,
                cleanup_temp_files: false,
                ..Default::default()
            };
            let mut sorter = ReplaceSelectSort::new(config);
            let ctx = format!("merge_ways={ways}");
            let first = sorter.sort((0..40u64).rev())?;
            assert_eq!(first, (0..40).collect::<Vec<_>>(), "{ctx}");
            let second = sorter.sort((100..120u64).rev())?;
            assert_eq!(second, (100..120).collect::<Vec<_>>(), "{ctx}");
            // Only the second sort's runs are left on disk until the sorter
            // is dropped: its 5 runs, or the 2 merged runs that the final
            // 2-way merge read.
            let left = std::fs::read_dir(&dir).unwrap().count();
            assert_eq!(left, if ways == 2 { 2 } else { 5 }, "{ctx}");
            drop(sorter);
            assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0, "{ctx}");
            std::fs::remove_dir(&dir).unwrap();
        }
        Ok(())
    }

    #[test]
    fn test_s20_r9b_failed_sort_leaves_no_run_file() {
        // S20-R9b: a write error in the middle of a run (here an item that
        // fails to serialize) left that run's file behind for good: it was
        // not in the run list yet, so neither the next `sort`, `cleanup` nor
        // drop deleted it. n = 14 takes the in-memory path (one run); n = 40
        // with a 4-item buffer takes the heap path (runs 0..=5 complete, run
        // 6 being written when 13 fails).

        /// Serializes like a `u32`, except that 13 fails to serialize.
        #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, serde::Deserialize)]
        struct FailOn13(u32);

        impl serde::Serialize for FailOn13 {
            fn serialize<S: serde::Serializer>(
                &self,
                serializer: S,
            ) -> std::result::Result<S::Ok, S::Error> {
                if self.0 == 13 {
                    return Err(serde::ser::Error::custom("FailOn13"));
                }
                serializer.serialize_u32(self.0)
            }
        }

        // (n, files left in the temp dir after the failed sort and drop)
        let mut left = Vec::new();
        for (n, buffer_items) in [(14u32, 256), (40, 4)] {
            let dir = test_temp_dir();
            let config = ReplaceSelectSortConfig {
                memory_buffer_size: buffer_items * std::mem::size_of::<FailOn13>(),
                temp_dir: dir.clone(),
                ..Default::default()
            };
            let mut sorter = ReplaceSelectSort::new(config);
            let err = sorter.sort((0..n).rev().map(FailOn13)).unwrap_err();
            assert!(err.to_string().contains("FailOn13"), "n={n}: {err}");
            drop(sorter);
            left.push((n, std::fs::read_dir(&dir).unwrap().count()));
            std::fs::remove_dir_all(&dir).unwrap();
        }
        assert_eq!(left, [(14, 0), (40, 0)]);
    }

    #[test]
    fn test_s20_r6d_documented_stats_hold() -> Result<()> {
        // Guard for the S20-R6d docs: the counters add up over `sort` calls,
        // `merge_passes` describes the last call, `temp_bytes_read` stays 0,
        // and `compress_temp_files` does not change what is written.
        let record = std::mem::size_of::<usize>() + std::mem::size_of::<u64>();
        let sort_twice = |compress: bool| -> Result<ExternalSortStats> {
            let config = ReplaceSelectSortConfig {
                memory_buffer_size: 4 * std::mem::size_of::<u64>(),
                temp_dir: test_temp_dir(),
                compress_temp_files: compress,
                ..Default::default()
            };
            let mut sorter = ReplaceSelectSort::new(config);
            sorter.sort((0..40u64).rev())?; // 10 runs, one merge pass
            assert_eq!(sorter.stats().merge_passes, 1);
            sorter.sort([3u64, 1, 2])?; // fits the buffer: one run, no merge
            Ok(sorter.stats().clone())
        };
        let plain = sort_twice(false)?;
        assert_eq!(plain.items_sorted, 43);
        assert_eq!(plain.runs_generated, 11);
        assert_eq!(plain.temp_bytes_written, 43 * record);
        assert_eq!(plain.merge_passes, 0);
        assert_eq!(plain.temp_bytes_read, 0);
        let compressed = sort_twice(true)?;
        assert_eq!(compressed.temp_bytes_written, plain.temp_bytes_written);
        Ok(())
    }
}
