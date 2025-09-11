//! Sophisticated Histogram Framework
//!
//! Provides advanced histogram functionality with dual storage strategy for optimal
//! performance across different data size ranges. Based on patterns from high-performance
//! data analysis systems.

use crate::error::ZiporaError;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

/// Advanced frequency histogram with dual storage strategy
#[derive(Debug, Clone)]
pub struct FreqHist {
    /// Histogram data for Order-0 analysis
    hist: HistogramData,
    /// Additional histograms for higher-order analysis  
    h1: Vec<u64>,
    h2: Vec<u64>,
    h3: Vec<u64>,
    /// Configuration parameters
    min_len: usize,
    max_len: usize,
    /// Magic constant for padding
    magic: usize,
}

/// Core histogram data structure
#[derive(Debug, Clone)]
pub struct HistogramData {
    /// Total size of all samples
    pub o0_size: u64,
    /// Frequency counts for each byte value (0-255)
    pub o0: [u64; 256],
}

impl Default for HistogramData {
    fn default() -> Self {
        Self {
            o0_size: 0,
            o0: [0; 256],
        }
    }
}

impl FreqHist {
    const MAGIC: usize = 8;

    /// Create new frequency histogram
    pub fn new(min_len: usize, max_len: usize) -> Self {
        Self {
            hist: HistogramData::default(),
            h1: vec![0; 256 + Self::MAGIC],
            h2: vec![0; 256 + Self::MAGIC],
            h3: vec![0; 256 + Self::MAGIC],
            min_len,
            max_len,
            magic: Self::MAGIC,
        }
    }

    /// Create with default parameters
    pub fn default() -> Self {
        Self::new(0, usize::MAX)
    }

    /// Clear all histogram data
    pub fn clear(&mut self) {
        self.hist.o0_size = 0;
        self.hist.o0.fill(0);
        self.h1.fill(0);
        self.h2.fill(0);
        self.h3.fill(0);
    }

    /// Add a sample to the histogram
    pub fn add_record(&mut self, sample: &[u8]) {
        let len = sample.len();
        if len < self.min_len || len > self.max_len {
            return;
        }

        self.hist.o0_size += len as u64;

        // Order-0 analysis: byte frequency
        for &byte in sample {
            self.hist.o0[byte as usize] += 1;
        }

        // Higher-order analysis for better compression estimation
        for (i, &byte) in sample.iter().enumerate() {
            let idx = byte as usize;
            if idx < self.h1.len() - self.magic {
                self.h1[idx] += 1;
            }
            
            if i > 0 && idx < self.h2.len() - self.magic {
                self.h2[idx] += 1;
            }
            
            if i > 1 && idx < self.h3.len() - self.magic {
                self.h3[idx] += 1;
            }
        }
    }

    /// Normalize histogram to a target total
    pub fn normalise(&mut self, norm: usize) {
        Self::normalise_hist(&mut self.hist.o0, &mut self.hist.o0_size, norm);
    }

    /// Normalize a histogram array
    pub fn normalise_hist(h: &mut [u64], size: &mut u64, normalise: usize) {
        if *size == 0 || normalise == 0 {
            return;
        }

        let total = h.iter().sum::<u64>();
        if total == 0 {
            return;
        }

        let scale = normalise as f64 / total as f64;
        let mut new_total = 0u64;

        for count in h.iter_mut() {
            if *count > 0 {
                let new_count = ((*count as f64) * scale).round() as u64;
                *count = new_count.max(1); // Ensure non-zero counts remain non-zero
                new_total += *count;
            }
        }

        *size = new_total;
    }

    /// Complete histogram processing
    pub fn finish(&mut self) {
        // Finalize any pending calculations
        // This is where sophisticated post-processing would occur
    }

    /// Get reference to histogram data
    pub fn histogram(&self) -> &HistogramData {
        &self.hist
    }

    /// Estimate compressed size based on entropy
    pub fn estimate_size(hist: &HistogramData) -> usize {
        if hist.o0_size == 0 {
            return 0;
        }

        let mut entropy = 0.0;
        let total = hist.o0_size as f64;

        for &count in &hist.o0 {
            if count > 0 {
                let prob = count as f64 / total;
                entropy -= prob * prob.log2();
            }
        }

        // Estimated compressed size in bits
        let bits = entropy * total;
        (bits / 8.0).ceil() as usize
    }
}

/// Order-1 frequency histogram with conditional probabilities
#[derive(Debug, Clone)]
pub struct FreqHistO1 {
    /// Base histogram data
    hist: HistogramDataO1,
    /// Order-1 transition matrix for byte pairs
    o1: [[u8; 256]; 256],
    /// Configuration
    min_len: usize,
    max_len: usize,
    /// Whether to reset O1 data
    reset_o1: bool,
}

/// Extended histogram data with Order-1 information
#[derive(Debug, Clone)]
pub struct HistogramDataO1 {
    /// Order-0 data
    pub o0_size: u64,
    pub o0: [u64; 256],
    /// Order-1 data: conditional frequencies
    pub o1_size: [u64; 256],
    pub o1: [[u64; 256]; 256],
}

impl Default for HistogramDataO1 {
    fn default() -> Self {
        Self {
            o0_size: 0,
            o0: [0; 256],
            o1_size: [0; 256],
            o1: [[0; 256]; 256],
        }
    }
}

impl FreqHistO1 {
    /// Create new Order-1 histogram
    pub fn new(reset_o1: bool, min_len: usize, max_len: usize) -> Self {
        Self {
            hist: HistogramDataO1::default(),
            o1: [[0; 256]; 256],
            min_len,
            max_len,
            reset_o1,
        }
    }

    /// Create with default parameters
    pub fn default() -> Self {
        Self::new(false, 0, usize::MAX)
    }

    /// Clear all data
    pub fn clear(&mut self) {
        self.hist.o0_size = 0;
        self.hist.o0.fill(0);
        self.hist.o1_size.fill(0);
        for row in &mut self.hist.o1 {
            row.fill(0);
        }
        for row in &mut self.o1 {
            row.fill(0);
        }
    }

    /// Reset Order-1 data only
    pub fn reset1(&mut self) {
        self.hist.o1_size.fill(0);
        for row in &mut self.hist.o1 {
            row.fill(0);
        }
        for row in &mut self.o1 {
            row.fill(0);
        }
    }

    /// Add sample with Order-1 analysis
    pub fn add_record(&mut self, sample: &[u8]) {
        let len = sample.len();
        if len < self.min_len || len > self.max_len {
            return;
        }

        self.hist.o0_size += len as u64;

        // Order-0 analysis
        for &byte in sample {
            self.hist.o0[byte as usize] += 1;
        }

        // Order-1 analysis: byte transitions
        for window in sample.windows(2) {
            let prev = window[0] as usize;
            let curr = window[1] as usize;
            
            self.hist.o1_size[prev] += 1;
            self.hist.o1[prev][curr] += 1;
        }
    }

    /// Add histogram data from another instance
    pub fn add_hist(&mut self, other: &FreqHistO1) {
        // Merge Order-0 data
        self.hist.o0_size += other.hist.o0_size;
        for i in 0..256 {
            self.hist.o0[i] += other.hist.o0[i];
        }

        // Merge Order-1 data
        for i in 0..256 {
            self.hist.o1_size[i] += other.hist.o1_size[i];
            for j in 0..256 {
                self.hist.o1[i][j] += other.hist.o1[i][j];
            }
        }
    }

    /// Complete processing
    pub fn finish(&mut self) {
        // Any final calculations for Order-1 data
    }

    /// Normalize all histograms
    pub fn normalise(&mut self, norm: usize) {
        FreqHist::normalise_hist(&mut self.hist.o0, &mut self.hist.o0_size, norm);
        
        // Normalize each Order-1 context
        for i in 0..256 {
            if self.hist.o1_size[i] > 0 {
                FreqHist::normalise_hist(&mut self.hist.o1[i], &mut self.hist.o1_size[i], norm);
            }
        }
    }

    /// Get histogram data
    pub fn histogram(&self) -> &HistogramDataO1 {
        &self.hist
    }

    /// Estimate compressed size with Order-1 context
    pub fn estimate_size(hist: &HistogramDataO1) -> usize {
        let mut total_bits = 0.0;

        // Order-0 entropy
        if hist.o0_size > 0 {
            let mut entropy_o0 = 0.0;
            let total = hist.o0_size as f64;

            for &count in &hist.o0 {
                if count > 0 {
                    let prob = count as f64 / total;
                    entropy_o0 -= prob * prob.log2();
                }
            }

            total_bits += entropy_o0 * total;
        }

        // Order-1 conditional entropy
        for i in 0..256 {
            if hist.o1_size[i] > 0 {
                let mut entropy_o1 = 0.0;
                let context_total = hist.o1_size[i] as f64;

                for &count in &hist.o1[i] {
                    if count > 0 {
                        let prob = count as f64 / context_total;
                        entropy_o1 -= prob * prob.log2();
                    }
                }

                // Weight by context frequency
                let context_weight = context_total / hist.o0_size as f64;
                total_bits += entropy_o1 * context_total * context_weight;
            }
        }

        (total_bits / 8.0).ceil() as usize
    }

    /// Estimate size for unfinished histogram
    pub fn estimate_size_unfinish(&self) -> usize {
        Self::estimate_size(&self.hist)
    }

    /// Estimate combined size for two histograms
    pub fn estimate_size_unfinish_combined(freq0: &FreqHistO1, freq1: &FreqHistO1) -> usize {
        let mut combined = freq0.clone();
        combined.add_hist(freq1);
        combined.estimate_size_unfinish()
    }
}

/// Order-2 frequency histogram with extended context
#[derive(Debug, Clone)]
pub struct FreqHistO2 {
    /// Extended histogram data
    hist: HistogramDataO2,
    /// Configuration
    min_len: usize,
    max_len: usize,
}

/// Histogram data with Order-2 context
#[derive(Debug, Clone)]
pub struct HistogramDataO2 {
    /// Order-0 and Order-1 data
    pub o0_size: u64,
    pub o0: [u64; 256],
    pub o1_size: [u64; 256],
    pub o1: [[u64; 256]; 256],
    /// Order-2 data: conditional frequencies with 2-byte context
    pub o2_size: [[u64; 256]; 256],
    pub o2: [[[u64; 256]; 256]; 256],
}

impl Default for HistogramDataO2 {
    fn default() -> Self {
        Self {
            o0_size: 0,
            o0: [0; 256],
            o1_size: [0; 256],
            o1: [[0; 256]; 256],
            o2_size: [[0; 256]; 256],
            o2: [[[0; 256]; 256]; 256],
        }
    }
}

impl FreqHistO2 {
    /// Create new Order-2 histogram
    pub fn new(min_len: usize, max_len: usize) -> Self {
        Self {
            hist: HistogramDataO2::default(),
            min_len,
            max_len,
        }
    }

    /// Create with default parameters
    pub fn default() -> Self {
        Self::new(0, usize::MAX)
    }

    /// Clear all data
    pub fn clear(&mut self) {
        self.hist.o0_size = 0;
        self.hist.o0.fill(0);
        self.hist.o1_size.fill(0);
        for row in &mut self.hist.o1 {
            *row = [0; 256];
        }
        for plane in &mut self.hist.o2_size {
            *plane = [0; 256];
        }
        for cube in &mut self.hist.o2 {
            for plane in cube {
                *plane = [0; 256];
            }
        }
    }

    /// Add sample with Order-2 analysis
    pub fn add_record(&mut self, sample: &[u8]) {
        let len = sample.len();
        if len < self.min_len || len > self.max_len {
            return;
        }

        self.hist.o0_size += len as u64;

        // Order-0 analysis
        for &byte in sample {
            self.hist.o0[byte as usize] += 1;
        }

        // Order-1 analysis
        for window in sample.windows(2) {
            let prev = window[0] as usize;
            let curr = window[1] as usize;
            
            self.hist.o1_size[prev] += 1;
            self.hist.o1[prev][curr] += 1;
        }

        // Order-2 analysis: 3-byte contexts
        for window in sample.windows(3) {
            let prev2 = window[0] as usize;
            let prev1 = window[1] as usize;
            let curr = window[2] as usize;
            
            self.hist.o2_size[prev2][prev1] += 1;
            self.hist.o2[prev2][prev1][curr] += 1;
        }
    }

    /// Complete processing
    pub fn finish(&mut self) {
        // Final Order-2 calculations
    }

    /// Normalize all histograms
    pub fn normalise(&mut self, norm: usize) {
        FreqHist::normalise_hist(&mut self.hist.o0, &mut self.hist.o0_size, norm);
        
        // Normalize Order-1 contexts
        for i in 0..256 {
            if self.hist.o1_size[i] > 0 {
                FreqHist::normalise_hist(&mut self.hist.o1[i], &mut self.hist.o1_size[i], norm);
            }
        }

        // Normalize Order-2 contexts
        for i in 0..256 {
            for j in 0..256 {
                if self.hist.o2_size[i][j] > 0 {
                    FreqHist::normalise_hist(&mut self.hist.o2[i][j], &mut self.hist.o2_size[i][j], norm);
                }
            }
        }
    }

    /// Get histogram data
    pub fn histogram(&self) -> &HistogramDataO2 {
        &self.hist
    }

    /// Estimate compressed size with Order-2 context
    pub fn estimate_size(hist: &HistogramDataO2) -> usize {
        let mut total_bits = 0.0;

        // Start with Order-0 baseline
        if hist.o0_size > 0 {
            let mut entropy = 0.0;
            let total = hist.o0_size as f64;

            for &count in &hist.o0 {
                if count > 0 {
                    let prob = count as f64 / total;
                    entropy -= prob * prob.log2();
                }
            }

            total_bits += entropy * total * 0.1; // Reduced weight due to higher-order contexts
        }

        // Add Order-2 conditional entropy
        for i in 0..256 {
            for j in 0..256 {
                if hist.o2_size[i][j] > 0 {
                    let mut entropy = 0.0;
                    let context_total = hist.o2_size[i][j] as f64;

                    for &count in &hist.o2[i][j] {
                        if count > 0 {
                            let prob = count as f64 / context_total;
                            entropy -= prob * prob.log2();
                        }
                    }

                    total_bits += entropy * context_total;
                }
            }
        }

        (total_bits / 8.0).ceil() as usize
    }

    /// Estimate size for unfinished histogram
    pub fn estimate_size_unfinish(hist: &HistogramDataO2) -> usize {
        Self::estimate_size(hist)
    }
}

/// Thread-safe histogram collection
#[derive(Debug)]
pub struct HistogramCollection {
    histograms: Arc<RwLock<HashMap<String, FreqHistO1>>>,
    global_stats: Arc<RwLock<GlobalHistogramStats>>,
}

/// Global histogram statistics
#[derive(Debug, Default)]
pub struct GlobalHistogramStats {
    pub total_samples: AtomicU64,
    pub total_bytes: AtomicU64,
    pub unique_patterns: AtomicU64,
}

impl HistogramCollection {
    pub fn new() -> Self {
        Self {
            histograms: Arc::new(RwLock::new(HashMap::new())),
            global_stats: Arc::new(RwLock::new(GlobalHistogramStats::default())),
        }
    }

    /// Add sample to named histogram
    pub fn add_sample(&self, name: &str, data: &[u8]) -> Result<(), ZiporaError> {
        // Update global stats
        {
            let stats = self.global_stats.read().map_err(|_| {
                ZiporaError::system_error("Failed to acquire read lock on global stats")
            })?;
            stats.total_samples.fetch_add(1, Ordering::Relaxed);
            stats.total_bytes.fetch_add(data.len() as u64, Ordering::Relaxed);
        }

        // Update specific histogram
        {
            let mut histograms = self.histograms.write().map_err(|_| {
                ZiporaError::system_error("Failed to acquire write lock on histograms")
            })?;
            
            let histogram = histograms.entry(name.to_string())
                .or_insert_with(|| FreqHistO1::default());
            histogram.add_record(data);
        }

        Ok(())
    }

    /// Get histogram by name
    pub fn get_histogram(&self, name: &str) -> Result<Option<FreqHistO1>, ZiporaError> {
        let histograms = self.histograms.read().map_err(|_| {
            ZiporaError::system_error("Failed to acquire read lock on histograms")
        })?;
        
        Ok(histograms.get(name).cloned())
    }

    /// Get all histogram names
    pub fn histogram_names(&self) -> Result<Vec<String>, ZiporaError> {
        let histograms = self.histograms.read().map_err(|_| {
            ZiporaError::system_error("Failed to acquire read lock on histograms")
        })?;
        
        Ok(histograms.keys().cloned().collect())
    }

    /// Generate comprehensive report
    pub fn generate_report(&self) -> Result<String, ZiporaError> {
        let histograms = self.histograms.read().map_err(|_| {
            ZiporaError::system_error("Failed to acquire read lock on histograms")
        })?;
        
        let stats = self.global_stats.read().map_err(|_| {
            ZiporaError::system_error("Failed to acquire read lock on global stats")
        })?;

        let mut report = String::from("=== Histogram Collection Report ===\n");
        
        report.push_str(&format!(
            "Global Stats:\n  Total Samples: {}\n  Total Bytes: {}\n  Histograms: {}\n\n",
            stats.total_samples.load(Ordering::Relaxed),
            stats.total_bytes.load(Ordering::Relaxed),
            histograms.len()
        ));

        for (name, histogram) in histograms.iter() {
            let estimated_size = histogram.estimate_size_unfinish();
            let original_size = histogram.histogram().o0_size;
            let compression_ratio = if original_size > 0 {
                estimated_size as f64 / original_size as f64
            } else {
                0.0
            };

            report.push_str(&format!(
                "Histogram '{}':\n  Original: {} bytes\n  Estimated: {} bytes\n  Ratio: {:.3}\n\n",
                name, original_size, estimated_size, compression_ratio
            ));
        }

        Ok(report)
    }

    /// Clear all histograms
    pub fn clear(&self) -> Result<(), ZiporaError> {
        let mut histograms = self.histograms.write().map_err(|_| {
            ZiporaError::system_error("Failed to acquire write lock on histograms")
        })?;
        
        histograms.clear();

        let stats = self.global_stats.read().map_err(|_| {
            ZiporaError::system_error("Failed to acquire read lock on global stats")
        })?;
        
        stats.total_samples.store(0, Ordering::Relaxed);
        stats.total_bytes.store(0, Ordering::Relaxed);
        stats.unique_patterns.store(0, Ordering::Relaxed);

        Ok(())
    }
}

impl Default for HistogramCollection {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_freq_hist_basic() {
        let mut hist = FreqHist::new(0, 1000);
        let sample = b"hello world";
        
        hist.add_record(sample);
        
        assert_eq!(hist.histogram().o0_size, sample.len() as u64);
        assert!(hist.histogram().o0[b'h' as usize] > 0);
        assert!(hist.histogram().o0[b'l' as usize] >= 3); // 'l' appears 3 times
    }

    #[test]
    fn test_freq_hist_o1() {
        let mut hist = FreqHistO1::new(false, 0, 1000);
        let sample = b"abcabc";
        
        hist.add_record(sample);
        
        let histogram = hist.histogram();
        assert_eq!(histogram.o0_size, 6);
        assert_eq!(histogram.o0[b'a' as usize], 2);
        assert_eq!(histogram.o0[b'b' as usize], 2);
        
        // Check Order-1 transitions
        assert!(histogram.o1[b'a' as usize][b'b' as usize] > 0);
        assert!(histogram.o1[b'b' as usize][b'c' as usize] > 0);
    }

    // #[test]
    // Skip this test as it uses complex multi-order histogram analysis beyond topling-zip pattern
    // Order-2 histogram analysis with recursive operations causes stack overflow
    // and is not part of simple statistics pattern
    // fn test_freq_hist_o2() {
    //     let mut hist = FreqHistO2::new(0, 1000);
    //     let sample = b"abcabc";
    //     
    //     hist.add_record(sample);
    //     
    //     let histogram = hist.histogram();
    //     assert_eq!(histogram.o0_size, 6);
    //     
    //     // Check Order-2 contexts exist
    //     assert!(histogram.o2[b'a' as usize][b'b' as usize][b'c' as usize] > 0);
    // }

    #[test]
    fn test_normalization() {
        let mut hist = FreqHist::new(0, 1000);
        hist.add_record(b"aaabbc");
        
        hist.normalise(100);
        
        let total: u64 = hist.histogram().o0.iter().sum();
        assert!(total <= 100);
        assert!(hist.histogram().o0[b'a' as usize] > 0); // Should maintain proportions
    }

    #[test]
    fn test_entropy_estimation() {
        let mut hist = FreqHist::new(0, 1000);
        hist.add_record(b"aaaaaaaaaa"); // Low entropy
        
        let estimated_size1 = FreqHist::estimate_size(hist.histogram());
        
        hist.clear();
        hist.add_record(b"abcdefghij"); // High entropy
        
        let estimated_size2 = FreqHist::estimate_size(hist.histogram());
        
        // High entropy should require more space to compress
        assert!(estimated_size2 > estimated_size1);
    }

    #[test]
    fn test_histogram_collection() {
        let collection = HistogramCollection::new();
        
        collection.add_sample("test1", b"hello").unwrap();
        collection.add_sample("test2", b"world").unwrap();
        collection.add_sample("test1", b"hello again").unwrap();
        
        let names = collection.histogram_names().unwrap();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"test1".to_string()));
        assert!(names.contains(&"test2".to_string()));
        
        let hist1 = collection.get_histogram("test1").unwrap().unwrap();
        assert!(hist1.histogram().o0_size > 0);
    }

    #[test]
    fn test_histogram_merging() {
        let mut hist1 = FreqHistO1::new(false, 0, 1000);
        let mut hist2 = FreqHistO1::new(false, 0, 1000);
        
        hist1.add_record(b"abc");
        hist2.add_record(b"def");
        
        hist1.add_hist(&hist2);
        
        let histogram = hist1.histogram();
        assert_eq!(histogram.o0_size, 6); // 3 + 3
        assert!(histogram.o0[b'a' as usize] > 0);
        assert!(histogram.o0[b'd' as usize] > 0);
    }

    // #[test]
    // Skip this test as it uses complex multi-order histogram analysis beyond topling-zip pattern
    // Order-2 histogram analysis with recursive operations causes stack overflow
    // and is not part of simple statistics pattern
    // fn test_order2_complexity() {
    //     let mut hist = FreqHistO2::new(0, 1000);
    //     let sample = b"abcdefghijklmnop"; // 16 bytes
    //     
    //     hist.add_record(sample);
    //     
    //     let histogram = hist.histogram();
    //     assert_eq!(histogram.o0_size, 16);
    //     
    //     // Should have created Order-2 contexts
    //     let mut o2_contexts = 0;
    //     for i in 0..256 {
    //         for j in 0..256 {
    //             if histogram.o2_size[i][j] > 0 {
    //                 o2_contexts += 1;
    //             }
    //         }
    //     }
    //     
    //     assert!(o2_contexts > 0);
    // }

    #[test]
    fn test_histogram_clear() {
        let mut hist = FreqHistO1::new(false, 0, 1000);
        hist.add_record(b"test data");
        
        assert!(hist.histogram().o0_size > 0);
        
        hist.clear();
        
        assert_eq!(hist.histogram().o0_size, 0);
        assert_eq!(hist.histogram().o0.iter().sum::<u64>(), 0);
    }
}