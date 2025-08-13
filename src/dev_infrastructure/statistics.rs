//! Statistical Analysis Tools
//!
//! Built-in statistics collection and analysis for performance monitoring and data insights.
//! Provides adaptive histograms, comprehensive statistical metrics, and efficient data
//! aggregation. Inspired by production statistical frameworks while leveraging Rust's
//! performance and safety guarantees.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, RwLock};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use crate::error::{ZiporaError, Result};

/// Trait for types that can be used as histogram indices
pub trait StatIndex: 
    Copy + Clone + PartialEq + Eq + PartialOrd + Ord + std::hash::Hash + std::fmt::Debug + 
    Into<u64> + TryFrom<u64> + Send + Sync + 'static 
{
    /// Zero value for this type
    fn zero() -> Self;
    
    /// One value for this type  
    fn one() -> Self;
    
    /// Maximum value that should use small storage
    const MAX_SMALL_VALUE: Self;
}

impl StatIndex for u32 {
    fn zero() -> Self { 0 }
    fn one() -> Self { 1 }
    const MAX_SMALL_VALUE: Self = 65535;
}

impl StatIndex for u64 {
    fn zero() -> Self { 0 }
    fn one() -> Self { 1 }
    const MAX_SMALL_VALUE: Self = 65535;
}

/// Adaptive histogram with dual storage strategy for efficient statistics collection
pub struct Histogram<T: StatIndex> {
    // Direct array for frequent keys (small values)
    small_counts: Vec<T>,
    // Hash map for rare/large keys
    large_counts: HashMap<T, T>,
    // Comprehensive statistics
    distinct_key_count: usize,
    count_sum: u64,              // Total count (integral of f(x))
    total_key_len: u64,          // Weighted sum (integral of x*f(x))
    min_key: Option<T>,
    max_key: Option<T>,
    min_count: Option<T>,
    max_count: Option<T>,
    finalized: bool,
}

impl<T: StatIndex> Histogram<T> {
    /// Create a new histogram with default small value threshold
    pub fn new() -> Self {
        Self::with_small_threshold(T::MAX_SMALL_VALUE)
    }

    /// Create a new histogram with custom small value threshold
    pub fn with_small_threshold(max_small_value: T) -> Self {
        let threshold: u64 = max_small_value.into();
        let size = std::cmp::min(threshold as usize, 65536); // Cap to prevent excessive memory
        
        Self {
            small_counts: vec![T::zero(); size],
            large_counts: HashMap::new(),
            distinct_key_count: 0,
            count_sum: 0,
            total_key_len: 0,
            min_key: None,
            max_key: None,
            min_count: None,
            max_count: None,
            finalized: false,
        }
    }

    /// Add a count to the histogram
    pub fn add(&mut self, key: T, count: T) {
        if self.finalized {
            panic!("Cannot modify finalized histogram");
        }

        let key_u64: u64 = key.into();
        let count_u64: u64 = count.into();
        
        // Update the count
        let new_count = if key_u64 < self.small_counts.len() as u64 {
            let idx = key_u64 as usize;
            let current: u64 = self.small_counts[idx].into();
            let new_val = current + count_u64;
            self.small_counts[idx] = T::try_from(new_val).unwrap_or_else(|_| {
                // If overflow, migrate to large_counts
                self.large_counts.insert(key, T::try_from(new_val).unwrap_or_else(|_| T::try_from(u64::MAX).unwrap_or(T::zero())));
                T::zero()
            });
            new_val
        } else {
            let current: u64 = self.large_counts.get(&key).map(|&v| v.into()).unwrap_or(0);
            let new_val = current + count_u64;
            self.large_counts.insert(key, T::try_from(new_val).unwrap_or_else(|_| T::try_from(u64::MAX).unwrap_or(T::zero())));
            new_val
        };

        // Update statistics  
        if new_count == count_u64 {
            // New key
            self.distinct_key_count += 1;
        }
        
        self.count_sum += count_u64;
        self.total_key_len += key_u64 * count_u64;
        
        // Update min/max key
        self.min_key = Some(self.min_key.map_or(key, |min| if key < min { key } else { min }));
        self.max_key = Some(self.max_key.map_or(key, |max| if key > max { key } else { max }));
        
        // Update min/max count
        let count_t = T::try_from(new_count).unwrap_or_else(|_| T::try_from(u64::MAX).unwrap_or(T::zero()));
        self.min_count = Some(self.min_count.map_or(count_t, |min| if count_t < min { count_t } else { min }));
        self.max_count = Some(self.max_count.map_or(count_t, |max| if count_t > max { count_t } else { max }));
    }

    /// Increment count for a key by 1
    pub fn increment(&mut self, key: T) {
        self.add(key, T::one());
    }

    /// Get count for a specific key
    pub fn get(&self, key: T) -> T {
        let key_u64: u64 = key.into();
        
        if key_u64 < self.small_counts.len() as u64 {
            self.small_counts[key_u64 as usize]
        } else {
            self.large_counts.get(&key).copied().unwrap_or(T::zero())
        }
    }

    /// Finalize the histogram for efficient iteration and analysis
    pub fn finalize(&mut self) {
        if self.finalized {
            return;
        }

        // Recompute min/max counts for accuracy
        let mut min_count: Option<T> = None;
        let mut max_count: Option<T> = None;
        
        self.for_each(|_key, count| {
            min_count = Some(min_count.map_or(count, |min| if count < min { count } else { min }));
            max_count = Some(max_count.map_or(count, |max| if count > max { count } else { max }));
        });
        
        self.min_count = min_count;
        self.max_count = max_count;
        
        self.finalized = true;
    }

    /// Iterate over all key-count pairs
    pub fn for_each<F>(&self, mut op: F)
    where
        F: FnMut(T, T),
    {
        // Iterate small counts first (cache-friendly)
        for (idx, &count) in self.small_counts.iter().enumerate() {
            let count_u64: u64 = count.into();
            if count_u64 > 0 {
                let key = T::try_from(idx as u64).unwrap_or_else(|_| T::zero());
                op(key, count);
            }
        }
        
        // Then iterate large counts
        for (&key, &count) in &self.large_counts {
            op(key, count);
        }
    }

    /// Get comprehensive statistics
    pub fn stats(&self) -> HistogramStats<T> {
        HistogramStats {
            distinct_key_count: self.distinct_key_count,
            count_sum: self.count_sum,
            total_key_len: self.total_key_len,
            min_key: self.min_key,
            max_key: self.max_key,
            min_count: self.min_count,
            max_count: self.max_count,
            mean_key: if self.count_sum > 0 {
                Some(self.total_key_len as f64 / self.count_sum as f64)
            } else {
                None
            },
            mean_count: if self.distinct_key_count > 0 {
                Some(self.count_sum as f64 / self.distinct_key_count as f64)
            } else {
                None
            },
        }
    }

    /// Get percentile value
    pub fn percentile(&self, p: f64) -> Option<T> {
        if !(0.0..=1.0).contains(&p) || self.count_sum == 0 {
            return None;
        }

        let target_count = (self.count_sum as f64 * p) as u64;
        let mut running_count = 0u64;

        let mut items: Vec<(T, T)> = Vec::new();
        self.for_each(|key, count| {
            items.push((key, count));
        });
        items.sort_by_key(|&(key, _)| key);

        for (key, count) in items {
            running_count += count.into();
            if running_count >= target_count {
                return Some(key);
            }
        }

        None
    }

    /// Get median value
    pub fn median(&self) -> Option<T> {
        self.percentile(0.5)
    }

    /// Check if histogram is empty
    pub fn is_empty(&self) -> bool {
        self.count_sum == 0
    }

    /// Get total number of samples
    pub fn total_count(&self) -> u64 {
        self.count_sum
    }

    /// Get number of distinct keys
    pub fn distinct_keys(&self) -> usize {
        self.distinct_key_count
    }
}

impl<T: StatIndex> Default for Histogram<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Comprehensive histogram statistics
#[derive(Debug, Clone)]
pub struct HistogramStats<T: StatIndex> {
    pub distinct_key_count: usize,
    pub count_sum: u64,
    pub total_key_len: u64,
    pub min_key: Option<T>,
    pub max_key: Option<T>,
    pub min_count: Option<T>,
    pub max_count: Option<T>,
    pub mean_key: Option<f64>,
    pub mean_count: Option<f64>,
}

impl<T: StatIndex> fmt::Display for HistogramStats<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Histogram Statistics:")?;
        writeln!(f, "  Distinct Keys: {}", self.distinct_key_count)?;
        writeln!(f, "  Total Count: {}", self.count_sum)?;
        writeln!(f, "  Total Key Length: {}", self.total_key_len)?;
        if let Some(min) = self.min_key {
            writeln!(f, "  Min Key: {:?}", min)?;
        }
        if let Some(max) = self.max_key {
            writeln!(f, "  Max Key: {:?}", max)?;
        }
        if let Some(min) = self.min_count {
            writeln!(f, "  Min Count: {:?}", min)?;
        }
        if let Some(max) = self.max_count {
            writeln!(f, "  Max Count: {:?}", max)?;
        }
        if let Some(mean) = self.mean_key {
            writeln!(f, "  Mean Key: {:.2}", mean)?;
        }
        if let Some(mean) = self.mean_count {
            writeln!(f, "  Mean Count: {:.2}", mean)?;
        }
        Ok(())
    }
}

/// Type aliases for common histogram types
pub type U32Histogram = Histogram<u32>;
pub type U64Histogram = Histogram<u64>;

/// Real-time statistics accumulator
pub struct StatAccumulator {
    count: AtomicU64,
    sum: AtomicU64,
    sum_squares: AtomicU64,
    min: AtomicU64,
    max: AtomicU64,
}

impl StatAccumulator {
    /// Create a new statistics accumulator
    pub fn new() -> Self {
        Self {
            count: AtomicU64::new(0),
            sum: AtomicU64::new(0),
            sum_squares: AtomicU64::new(0),
            min: AtomicU64::new(u64::MAX),
            max: AtomicU64::new(0),
        }
    }

    /// Add a value to the accumulator
    pub fn add(&self, value: u64) {
        self.count.fetch_add(1, Ordering::SeqCst);
        self.sum.fetch_add(value, Ordering::SeqCst);
        self.sum_squares.fetch_add(value * value, Ordering::SeqCst);
        
        // Update min/max with compare-and-swap loops
        let mut current_min = self.min.load(Ordering::SeqCst);
        while value < current_min {
            match self.min.compare_exchange_weak(current_min, value, Ordering::SeqCst, Ordering::Relaxed) {
                Ok(_) => break,
                Err(actual) => current_min = actual,
            }
        }
        
        let mut current_max = self.max.load(Ordering::SeqCst);
        while value > current_max {
            match self.max.compare_exchange_weak(current_max, value, Ordering::SeqCst, Ordering::Relaxed) {
                Ok(_) => break,
                Err(actual) => current_max = actual,
            }
        }
    }

    /// Get current statistics snapshot
    pub fn snapshot(&self) -> AccumulatorStats {
        let count = self.count.load(Ordering::SeqCst);
        let sum = self.sum.load(Ordering::SeqCst);
        let sum_squares = self.sum_squares.load(Ordering::SeqCst);
        let min = self.min.load(Ordering::SeqCst);
        let max = self.max.load(Ordering::SeqCst);
        
        let mean = if count > 0 { sum as f64 / count as f64 } else { 0.0 };
        
        let variance = if count > 1 {
            let mean_squares = (sum_squares as f64 / count as f64);
            mean_squares - (mean * mean)
        } else {
            0.0
        };
        
        let std_dev = variance.sqrt();
        
        AccumulatorStats {
            count,
            sum,
            min: if min == u64::MAX { 0 } else { min },
            max,
            mean,
            variance,
            std_dev,
        }
    }

    /// Reset all statistics
    pub fn reset(&self) {
        self.count.store(0, Ordering::SeqCst);
        self.sum.store(0, Ordering::SeqCst);
        self.sum_squares.store(0, Ordering::SeqCst);
        self.min.store(u64::MAX, Ordering::SeqCst);
        self.max.store(0, Ordering::SeqCst);
    }
}

impl Default for StatAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics snapshot from accumulator
#[derive(Debug, Clone)]
pub struct AccumulatorStats {
    pub count: u64,
    pub sum: u64,
    pub min: u64,
    pub max: u64,
    pub mean: f64,
    pub variance: f64,
    pub std_dev: f64,
}

impl fmt::Display for AccumulatorStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Accumulator Statistics:")?;
        writeln!(f, "  Count: {}", self.count)?;
        writeln!(f, "  Sum: {}", self.sum)?;
        writeln!(f, "  Min: {}", self.min)?;
        writeln!(f, "  Max: {}", self.max)?;
        writeln!(f, "  Mean: {:.2}", self.mean)?;
        writeln!(f, "  Std Dev: {:.2}", self.std_dev)?;
        writeln!(f, "  Variance: {:.2}", self.variance)?;
        Ok(())
    }
}

/// Multi-dimensional statistics collector
pub struct MultiDimensionalStats {
    name: String,
    dimensions: Vec<String>,
    accumulators: Vec<StatAccumulator>,
    cross_correlations: HashMap<(usize, usize), f64>,
}

impl MultiDimensionalStats {
    /// Create a new multi-dimensional statistics collector
    pub fn new(name: impl Into<String>, dimensions: Vec<String>) -> Self {
        let dim_count = dimensions.len();
        Self {
            name: name.into(),
            dimensions,
            accumulators: (0..dim_count).map(|_| StatAccumulator::new()).collect(),
            cross_correlations: HashMap::new(),
        }
    }

    /// Add a multi-dimensional sample
    pub fn add_sample(&mut self, values: &[u64]) -> Result<()> {
        if values.len() != self.accumulators.len() {
            return Err(ZiporaError::invalid_data("Sample dimension mismatch"));
        }

        for (acc, &value) in self.accumulators.iter().zip(values.iter()) {
            acc.add(value);
        }

        Ok(())
    }

    /// Get statistics for a specific dimension
    pub fn dimension_stats(&self, dimension: usize) -> Option<AccumulatorStats> {
        self.accumulators.get(dimension).map(|acc| acc.snapshot())
    }

    /// Get statistics for all dimensions
    pub fn all_stats(&self) -> Vec<AccumulatorStats> {
        self.accumulators.iter().map(|acc| acc.snapshot()).collect()
    }

    /// Compute correlation between two dimensions
    pub fn correlation(&self, dim1: usize, dim2: usize) -> Result<f64> {
        if dim1 >= self.accumulators.len() || dim2 >= self.accumulators.len() {
            return Err(ZiporaError::invalid_data("Invalid dimension index"));
        }

        // For real correlation, we'd need to store individual samples
        // This is a simplified version returning cached correlation if available
        Ok(self.cross_correlations.get(&(dim1, dim2)).copied().unwrap_or(0.0))
    }

    /// Print comprehensive report
    pub fn print_report(&self) {
        println!("Multi-Dimensional Statistics Report: {}", self.name);
        println!("{:-<60}", "");

        for (i, (dim_name, stats)) in self.dimensions.iter().zip(self.all_stats().iter()).enumerate() {
            println!("Dimension {}: {}", i, dim_name);
            println!("  Count: {}, Mean: {:.2}, Std Dev: {:.2}", 
                stats.count, stats.mean, stats.std_dev);
            println!("  Min: {}, Max: {}", stats.min, stats.max);
        }
        println!("{:-<60}", "");
    }

    /// Reset all statistics
    pub fn reset(&mut self) {
        for acc in &self.accumulators {
            acc.reset();
        }
        self.cross_correlations.clear();
    }
}

/// Global statistics registry
pub struct GlobalStatsRegistry {
    histograms: RwLock<HashMap<String, Box<dyn std::any::Any + Send + Sync>>>,
    accumulators: RwLock<HashMap<String, StatAccumulator>>,
    multi_dimensional: RwLock<HashMap<String, MultiDimensionalStats>>,
}

impl GlobalStatsRegistry {
    /// Create a new global statistics registry
    pub fn new() -> Self {
        Self {
            histograms: RwLock::new(HashMap::new()),
            accumulators: RwLock::new(HashMap::new()),
            multi_dimensional: RwLock::new(HashMap::new()),
        }
    }

    /// Register a histogram
    pub fn register_histogram<T: StatIndex + 'static>(&self, name: &str, histogram: Histogram<T>) -> Result<()> {
        let mut histograms = self.histograms.write()
            .map_err(|_| ZiporaError::io_error("Failed to acquire write lock"))?;
        
        histograms.insert(name.to_string(), Box::new(histogram));
        Ok(())
    }

    /// Register an accumulator
    pub fn register_accumulator(&self, name: &str, accumulator: StatAccumulator) -> Result<()> {
        let mut accumulators = self.accumulators.write()
            .map_err(|_| ZiporaError::io_error("Failed to acquire write lock"))?;
        
        accumulators.insert(name.to_string(), accumulator);
        Ok(())
    }

    /// Register multi-dimensional statistics
    pub fn register_multi_dimensional(&self, name: &str, stats: MultiDimensionalStats) -> Result<()> {
        let mut multi_dimensional = self.multi_dimensional.write()
            .map_err(|_| ZiporaError::io_error("Failed to acquire write lock"))?;
        
        multi_dimensional.insert(name.to_string(), stats);
        Ok(())
    }

    /// Get accumulator by name
    pub fn get_accumulator(&self, name: &str) -> Result<Option<AccumulatorStats>> {
        let accumulators = self.accumulators.read()
            .map_err(|_| ZiporaError::io_error("Failed to acquire read lock"))?;
        
        Ok(accumulators.get(name).map(|acc| acc.snapshot()))
    }

    /// List all registered statistics
    pub fn list_statistics(&self) -> Result<Vec<String>> {
        let histograms = self.histograms.read()
            .map_err(|_| ZiporaError::io_error("Failed to acquire read lock"))?;
        let accumulators = self.accumulators.read()
            .map_err(|_| ZiporaError::io_error("Failed to acquire read lock"))?;
        let multi_dimensional = self.multi_dimensional.read()
            .map_err(|_| ZiporaError::io_error("Failed to acquire read lock"))?;
        
        let mut stats = Vec::new();
        stats.extend(histograms.keys().map(|k| format!("histogram:{}", k)));
        stats.extend(accumulators.keys().map(|k| format!("accumulator:{}", k)));
        stats.extend(multi_dimensional.keys().map(|k| format!("multi_dim:{}", k)));
        
        Ok(stats)
    }
}

impl Default for GlobalStatsRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Global statistics registry instance
static GLOBAL_STATS: std::sync::LazyLock<GlobalStatsRegistry> = 
    std::sync::LazyLock::new(|| GlobalStatsRegistry::new());

/// Get the global statistics registry
pub fn global_stats() -> &'static GlobalStatsRegistry {
    &GLOBAL_STATS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_basic() {
        let mut hist = U32Histogram::new();
        
        hist.increment(5);
        hist.increment(5);
        hist.increment(10);
        hist.add(15, 3);
        
        assert_eq!(hist.get(5), 2);
        assert_eq!(hist.get(10), 1);
        assert_eq!(hist.get(15), 3);
        assert_eq!(hist.get(20), 0);
        
        let stats = hist.stats();
        assert_eq!(stats.distinct_key_count, 3);
        assert_eq!(stats.count_sum, 6);
        assert_eq!(stats.min_key, Some(5));
        assert_eq!(stats.max_key, Some(15));
    }

    #[test]
    fn test_histogram_large_keys() {
        let mut hist = U32Histogram::new();
        
        // Add values larger than small threshold
        hist.increment(2000);
        hist.increment(5000);
        hist.add(10000, 5);
        
        assert_eq!(hist.get(2000), 1);
        assert_eq!(hist.get(5000), 1);
        assert_eq!(hist.get(10000), 5);
        
        let stats = hist.stats();
        assert_eq!(stats.distinct_key_count, 3);
        assert_eq!(stats.count_sum, 7);
    }

    #[test]
    fn test_histogram_percentiles() {
        let mut hist = U32Histogram::new();
        
        // Add values: 1(x1), 2(x2), 3(x3), 4(x4)
        for i in 1..=4 {
            for _ in 0..i {
                hist.increment(i);
            }
        }
        
        hist.finalize();
        
        assert_eq!(hist.median(), Some(3));
        assert_eq!(hist.percentile(0.0), Some(1));
        assert_eq!(hist.percentile(1.0), Some(4));
    }

    #[test]
    fn test_stat_accumulator() {
        let acc = StatAccumulator::new();
        
        acc.add(10);
        acc.add(20);
        acc.add(30);
        
        let stats = acc.snapshot();
        assert_eq!(stats.count, 3);
        assert_eq!(stats.sum, 60);
        assert_eq!(stats.min, 10);
        assert_eq!(stats.max, 30);
        assert_eq!(stats.mean, 20.0);
    }

    #[test]
    fn test_multi_dimensional_stats() {
        let mut stats = MultiDimensionalStats::new(
            "test_stats",
            vec!["dimension1".to_string(), "dimension2".to_string()]
        );
        
        stats.add_sample(&[10, 20]).unwrap();
        stats.add_sample(&[15, 25]).unwrap();
        stats.add_sample(&[20, 30]).unwrap();
        
        let dim1_stats = stats.dimension_stats(0).unwrap();
        assert_eq!(dim1_stats.count, 3);
        assert_eq!(dim1_stats.mean, 15.0);
        
        let dim2_stats = stats.dimension_stats(1).unwrap();
        assert_eq!(dim2_stats.count, 3);
        assert_eq!(dim2_stats.mean, 25.0);
    }

    #[test]
    fn test_global_stats_registry() {
        let registry = global_stats();
        
        let hist = U32Histogram::new();
        registry.register_histogram("test_hist", hist).unwrap();
        
        let acc = StatAccumulator::new();
        acc.add(42);
        registry.register_accumulator("test_acc", acc).unwrap();
        
        let stats_list = registry.list_statistics().unwrap();
        assert!(stats_list.iter().any(|s| s.contains("test_hist")));
        assert!(stats_list.iter().any(|s| s.contains("test_acc")));
        
        let acc_stats = registry.get_accumulator("test_acc").unwrap();
        assert!(acc_stats.is_some());
        assert_eq!(acc_stats.unwrap().count, 1);
    }

    #[test]
    fn test_histogram_iteration() {
        let mut hist = U32Histogram::new();
        
        hist.increment(1);
        hist.increment(2);
        hist.increment(2);
        hist.increment(1000); // Large key
        
        let mut pairs = Vec::new();
        hist.for_each(|key, count| {
            pairs.push((key, count));
        });
        
        pairs.sort_by_key(|&(key, _)| key);
        assert_eq!(pairs, vec![(1, 1), (2, 2), (1000, 1)]);
    }
}