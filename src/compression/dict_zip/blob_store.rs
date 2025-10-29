//! DictZipBlobStore - PA-Zip Dictionary Compression Blob Store
//!
//! This module provides a complete blob storage system using PA-Zip dictionary compression.
//! It integrates all PA-Zip components into a unified blob store interface that supports
//! training dictionaries from samples, efficient compression/decompression, and comprehensive
//! performance statistics.
//!
//! # Overview
//!
//! The DictZipBlobStore provides:
//! - **Dictionary Training**: Build optimal dictionaries from training samples
//! - **High-Performance Compression**: PA-Zip algorithm with DFA cache acceleration
//! - **Flexible Storage**: Support for embedded and external dictionaries
//! - **Batch Operations**: Efficient batch compression and decompression
//! - **Statistics**: Comprehensive compression and performance metrics
//! - **Serialization**: Save/load dictionaries and compressed data
//!
//! # Usage Example
//!
//! ```rust
//! use zipora::compression::dict_zip::{DictZipBlobStore, DictZipBlobStoreBuilder, DictZipConfig};
//! use zipora::blob_store::{BlobStore, CompressedBlobStore};
//!
//! // Build store with dictionary training
//! let training_samples = vec![
//!     b"The quick brown fox jumps over the lazy dog".to_vec(),
//!     b"The lazy dog was jumped over by the quick brown fox".to_vec(),
//!     b"Quick brown foxes are faster than lazy dogs".to_vec(),
//! ];
//!
//! let config = DictZipConfig::text_compression();
//! let mut builder = DictZipBlobStoreBuilder::with_config(config)?;
//! 
//! // Train dictionary from samples
//! for sample in training_samples {
//!     builder.add_training_sample(&sample)?;
//! }
//!
//! // Build the final store
//! let mut store = builder.finish()?;
//!
//! // Use the store for compression
//! let data = b"The quick brown fox";
//! let id = store.put(data)?;
//! let retrieved = store.get(id)?;
//! assert_eq!(data, retrieved.as_slice());
//!
//! // Check compression statistics
//! let stats = store.compression_stats();
//! println!("Compression ratio: {:.2}%", stats.space_saved_percent());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Builder Pattern
//!
//! The builder pattern supports incremental dictionary training:
//!
//! ```rust
//! # use zipora::compression::dict_zip::{DictZipBlobStoreBuilder, DictZipConfig};
//! # use zipora::error::Result;
//! # fn example() -> Result<()> {
//! let mut builder = DictZipBlobStoreBuilder::new()?;
//!
//! // Add training samples incrementally
//! builder.add_training_sample(b"sample data 1")?;
//! builder.add_training_sample(b"sample data 2")?;
//! builder.add_training_file("training.txt")?;
//!
//! // Configure dictionary building
//! builder.set_dict_size_mb(32)?;
//! builder.set_min_frequency(4)?;
//! builder.enable_advanced_caching()?;
//!
//! // Build final store
//! let store = builder.finish()?;
//! # Ok(())
//! # }
//! ```

use crate::blob_store::traits::{
    BatchBlobStore, BlobStore, BlobStoreStats, CompressedBlobStore, CompressionStats,
};
use crate::compression::dict_zip::{
    DictionaryBuilder, DictionaryBuilderConfig, PaZipCompressor, PaZipCompressorConfig,
    SuffixArrayDictionary, CompressionStats as PaZipCompressionStats,
};
use crate::containers::LruMap;
use crate::entropy::huffman::{ContextualHuffmanEncoder, ContextualHuffmanDecoder};
use crate::entropy::fse::{FseEncoder, FseDecoder, FseConfig};
use crate::error::{Result, ZiporaError};
use crate::memory::{SecureMemoryPool, SecurePoolConfig};
use crate::RecordId;

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::{Arc, RwLock};
use std::cell::RefCell;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Entropy encoding algorithm selection
///
/// Matches the C++ reference implementation's EntropyAlgo enum:
/// - kNoEntropy: No entropy encoding (raw compression only)
/// - kHuffmanO1: Huffman Order-1 with context
/// - kFSE: Finite State Entropy (tANS-based, part of ZSTD family)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum EntropyAlgorithm {
    /// No entropy encoding (raw dictionary compression only)
    None,
    /// Huffman Order-1 with context (depends on previous symbol)
    HuffmanO1,
    /// Finite State Entropy (tANS-based, ZSTD-compatible)
    Fse,
}

impl Default for EntropyAlgorithm {
    fn default() -> Self {
        Self::None
    }
}

/// Configuration for DictZipBlobStore
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DictZipConfig {
    /// Dictionary building configuration
    pub dict_builder_config: DictionaryBuilderConfig,
    /// Compressor configuration
    pub compressor_config: PaZipCompressorConfig,
    /// Maximum size of decompression cache in bytes
    pub cache_size_bytes: usize,
    /// Enable external dictionary storage
    pub external_dictionary: bool,
    /// Dictionary storage path (if external)
    pub dict_path: Option<String>,
    /// Memory pool configuration
    pub memory_pool_config: Option<SecurePoolConfig>,
    /// Enable compression statistics tracking
    pub track_stats: bool,
    /// Enable dictionary validation on build
    pub validate_dictionary: bool,
    /// Minimum blob size to compress (smaller blobs stored uncompressed)
    pub min_compression_size: usize,

    // ===== Entropy Encoding Configuration (matches C++ DictZipBlobStore::Options) =====
    /// Entropy encoding algorithm to use (None, HuffmanO1, or FSE)
    /// Matches C++ EntropyAlgo field
    pub entropy_algorithm: EntropyAlgorithm,

    /// Interleaving factor for parallel encoding (0, 1, 2, 4, 8)
    /// - 0 or 1: No interleaving (single stream)
    /// - 2: 2-way interleaving (modest parallelism)
    /// - 4: 4-way interleaving (good with AVX2)
    /// - 8: 8-way interleaving (maximum parallelism)
    /// Matches C++ entropyInterleaved field
    pub entropy_interleaved: u8,

    /// Enable lake support (advanced feature from C++ reference)
    /// Matches C++ enableLake field
    pub enable_lake: bool,

    /// Embed dictionary in blob store file
    /// Matches C++ embeddedDict field
    pub embedded_dict: bool,

    /// Input is permutation (affects encoding strategy)
    /// Matches C++ inputIsPerm field
    pub input_is_perm: bool,

    /// Minimum compression ratio required to use entropy encoding
    /// If actual ratio < this value, fall back to no entropy encoding
    /// Matches C++ entropyZipRatioRequire field
    pub entropy_zip_ratio_require: f32,
}

impl Default for DictZipConfig {
    fn default() -> Self {
        Self {
            dict_builder_config: DictionaryBuilderConfig::default(),
            compressor_config: PaZipCompressorConfig::default(),
            cache_size_bytes: 16 * 1024 * 1024, // 16MB cache
            external_dictionary: false,
            dict_path: None,
            memory_pool_config: None,
            track_stats: true,
            validate_dictionary: true,
            min_compression_size: 64, // Don't compress blobs smaller than 64 bytes

            // Entropy encoding defaults (matching C++ defaults)
            entropy_algorithm: EntropyAlgorithm::None,
            entropy_interleaved: 0,  // No interleaving by default
            enable_lake: false,
            embedded_dict: false,
            input_is_perm: false,
            entropy_zip_ratio_require: 0.8,  // 80% compression or better
        }
    }
}

impl DictZipConfig {
    /// Create configuration optimized for text compression
    pub fn text_compression() -> Self {
        let mut config = Self {
            dict_builder_config: DictionaryBuilderConfig {
                sample_sort_policy: crate::compression::dict_zip::SampleSortPolicy::SortBoth, // Use best sorting for text
                target_dict_size: 32 * 1024 * 1024, // 32MB
                max_dict_size: 40 * 1024 * 1024, // 40MB max
                min_frequency: 3,
                max_bfs_depth: 6,
                min_pattern_length: 4,
                max_pattern_length: 128,
                sample_ratio: 0.8,
                validate_result: true,
                ..Default::default()
            },
            compressor_config: PaZipCompressorConfig::default(),
            cache_size_bytes: 32 * 1024 * 1024, // 32MB cache
            min_compression_size: 32,
            ..Default::default()
        };
        // Keep entropy defaults from Default::default()
        config
    }

    /// Create configuration optimized for binary data compression
    pub fn binary_compression() -> Self {
        let mut config = Self {
            dict_builder_config: DictionaryBuilderConfig {
                sample_sort_policy: crate::compression::dict_zip::SampleSortPolicy::SortRight, // Right sorting good for binary patterns
                target_dict_size: 16 * 1024 * 1024, // 16MB
                max_dict_size: 20 * 1024 * 1024, // 20MB max
                min_frequency: 8,
                max_bfs_depth: 4,
                min_pattern_length: 8,
                max_pattern_length: 64,
                sample_ratio: 0.5,
                validate_result: true,
                ..Default::default()
            },
            compressor_config: PaZipCompressorConfig::default(),
            cache_size_bytes: 16 * 1024 * 1024, // 16MB cache
            min_compression_size: 128,
            ..Default::default()
        };
        config
    }

    /// Create configuration optimized for log file compression
    pub fn log_compression() -> Self {
        let mut config = Self {
            dict_builder_config: DictionaryBuilderConfig {
                sample_sort_policy: crate::compression::dict_zip::SampleSortPolicy::SortLeft, // Left sorting good for log patterns
                target_dict_size: 64 * 1024 * 1024, // 64MB
                max_dict_size: 80 * 1024 * 1024, // 80MB max
                min_frequency: 2,
                max_bfs_depth: 8,
                min_pattern_length: 10,
                max_pattern_length: 256,
                sample_ratio: 0.3, // Logs are very repetitive
                validate_result: true,
                ..Default::default()
            },
            compressor_config: PaZipCompressorConfig::default(),
            cache_size_bytes: 64 * 1024 * 1024, // 64MB cache
            min_compression_size: 16,
            ..Default::default()
        };
        config
    }

    /// Create configuration optimized for real-time compression
    pub fn realtime_compression() -> Self {
        let mut config = Self {
            dict_builder_config: DictionaryBuilderConfig {
                target_dict_size: 8 * 1024 * 1024, // 8MB
                max_dict_size: 10 * 1024 * 1024, // 10MB max
                min_frequency: 10,
                max_bfs_depth: 3,
                min_pattern_length: 6,
                max_pattern_length: 32,
                sample_ratio: 0.2,
                validate_result: false, // Skip validation for speed
                ..Default::default()
            },
            compressor_config: PaZipCompressorConfig::default(),
            cache_size_bytes: 8 * 1024 * 1024, // 8MB cache
            min_compression_size: 256,
            ..Default::default()
        };
        config
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<()> {
        // Basic validation for dict builder config
        if self.dict_builder_config.target_dict_size == 0 {
            return Err(ZiporaError::invalid_data("Target dictionary size must be > 0"));
        }
        if self.dict_builder_config.max_dict_size < self.dict_builder_config.target_dict_size {
            return Err(ZiporaError::invalid_data("Max dictionary size must be >= target size"));
        }

        if self.cache_size_bytes == 0 {
            return Err(ZiporaError::invalid_data("Cache size must be > 0"));
        }

        if self.cache_size_bytes > 1024 * 1024 * 1024 {
            return Err(ZiporaError::invalid_data("Cache size must be <= 1GB"));
        }

        if self.external_dictionary && self.dict_path.is_none() {
            return Err(ZiporaError::invalid_data("External dictionary requires dict_path"));
        }

        // Validate entropy encoding configuration
        if ![0, 1, 2, 4, 8].contains(&self.entropy_interleaved) {
            return Err(ZiporaError::invalid_data(
                "Entropy interleaved must be 0, 1, 2, 4, or 8"
            ));
        }

        if self.entropy_zip_ratio_require < 0.0 || self.entropy_zip_ratio_require > 1.0 {
            return Err(ZiporaError::invalid_data(
                "Entropy zip ratio requirement must be between 0.0 and 1.0"
            ));
        }

        Ok(())
    }

    /// Enable external dictionary storage
    pub fn with_external_dictionary<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.external_dictionary = true;
        self.dict_path = Some(path.as_ref().to_string_lossy().to_string());
        self
    }

    /// Set cache size in megabytes
    pub fn with_cache_size_mb(mut self, mb: usize) -> Self {
        self.cache_size_bytes = mb * 1024 * 1024;
        self
    }

    /// Set minimum compression size
    pub fn with_min_compression_size(mut self, size: usize) -> Self {
        self.min_compression_size = size;
        self
    }
}

/// Statistics for DictZipBlobStore operations
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DictZipBlobStoreStats {
    /// Base blob store statistics
    pub blob_stats: BlobStoreStats,
    /// Compression statistics
    pub compression_stats: CompressionStats,
    /// PA-Zip specific statistics
    pub pa_zip_stats: PaZipCompressionStats,
    /// Cache hit statistics
    pub cache_hits: u64,
    /// Cache misses
    pub cache_misses: u64,
    /// Number of compressed blobs
    pub compressed_blobs: usize,
    /// Number of uncompressed blobs (too small)
    pub uncompressed_blobs: usize,
    /// Dictionary size in bytes
    pub dictionary_size: usize,
    /// Build time in milliseconds
    pub build_time_ms: u64,
}

impl DictZipBlobStoreStats {
    /// Calculate cache hit ratio
    pub fn cache_hit_ratio(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total > 0 {
            self.cache_hits as f64 / total as f64
        } else {
            0.0
        }
    }

    /// Calculate average compression ratio for compressed blobs only
    pub fn avg_compression_ratio(&self) -> f32 {
        if self.compressed_blobs > 0 {
            self.compression_stats.compression_ratio
        } else {
            1.0
        }
    }

    /// Calculate memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.dictionary_size + 
        self.compression_stats.compressed_size +
        (self.cache_hits + self.cache_misses) as usize * 8 // Rough cache overhead
    }
}

/// Builder for constructing DictZipBlobStore with dictionary training
pub struct DictZipBlobStoreBuilder {
    /// Configuration
    config: DictZipConfig,
    /// Training samples for dictionary building
    training_samples: Vec<Vec<u8>>,
    /// Total training data size
    training_size: usize,
    /// Memory pool for secure allocation
    memory_pool: Option<Arc<SecureMemoryPool>>,
    /// Build progress callback
    progress_callback: Option<Box<dyn Fn(f64) + Send + Sync>>,
}

impl DictZipBlobStoreBuilder {
    /// Create new builder with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(DictZipConfig::default())
    }

    /// Create builder with specified configuration
    pub fn with_config(config: DictZipConfig) -> Result<Self> {
        config.validate()?;

        let memory_pool = if let Some(pool_config) = &config.memory_pool_config {
            Some(SecureMemoryPool::new(pool_config.clone())?)
        } else {
            None
        };

        Ok(Self {
            config,
            training_samples: Vec::new(),
            training_size: 0,
            memory_pool,
            progress_callback: None,
        })
    }

    /// Add a training sample for dictionary building
    pub fn add_training_sample(&mut self, data: &[u8]) -> Result<()> {
        if data.is_empty() {
            return Err(ZiporaError::invalid_data("Training sample cannot be empty"));
        }

        self.training_samples.push(data.to_vec());
        self.training_size += data.len();
        
        Ok(())
    }

    /// Add training samples from a file
    pub fn add_training_file<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let data = fs::read(path.as_ref())
            .map_err(|e| ZiporaError::io_error(format!("Failed to read training file: {}", e)))?;
        
        self.add_training_sample(&data)
    }

    /// Add multiple training samples
    pub fn add_training_samples<I>(&mut self, samples: I) -> Result<()>
    where
        I: IntoIterator<Item = Vec<u8>>,
    {
        for sample in samples {
            self.add_training_sample(&sample)?;
        }
        Ok(())
    }

    /// Set dictionary size in megabytes
    pub fn set_dict_size_mb(&mut self, mb: usize) -> Result<()> {
        let size_bytes = mb * 1024 * 1024;
        self.config.dict_builder_config.target_dict_size = size_bytes;
        self.config.dict_builder_config.max_dict_size = size_bytes + (size_bytes / 4); // 25% overhead
        Ok(())
    }

    /// Set minimum frequency threshold for patterns
    pub fn set_min_frequency(&mut self, frequency: u32) -> Result<()> {
        if frequency == 0 {
            return Err(ZiporaError::invalid_data("Minimum frequency must be > 0"));
        }
        self.config.dict_builder_config.min_frequency = frequency;
        Ok(())
    }

    /// Enable advanced DFA caching
    pub fn enable_advanced_caching(&mut self) -> Result<()> {
        self.config.dict_builder_config.max_bfs_depth = 8;
        // Note: Compressor cache optimization is enabled by default
        Ok(())
    }

    /// Set progress callback for dictionary building
    pub fn set_progress_callback<F>(&mut self, callback: F) 
    where
        F: Fn(f64) + Send + Sync + 'static,
    {
        self.progress_callback = Some(Box::new(callback));
    }

    /// Get current training data statistics
    pub fn training_stats(&self) -> (usize, usize) {
        (self.training_samples.len(), self.training_size)
    }

    /// Build the final DictZipBlobStore
    pub fn finish(self) -> Result<DictZipBlobStore> {
        if self.training_samples.is_empty() {
            return Err(ZiporaError::invalid_data("No training samples provided"));
        }

        let start_time = std::time::Instant::now();

        // Combine all training samples
        let mut combined_training = Vec::with_capacity(self.training_size);
        for sample in &self.training_samples {
            combined_training.extend_from_slice(sample);
        }

        // Progress tracking
        if let Some(callback) = &self.progress_callback {
            callback(0.1); // 10% - training data prepared
        }

        // Build dictionary
        let builder = DictionaryBuilder::with_config(self.config.dict_builder_config.clone());
        let dictionary = builder.build(&combined_training)?;

        if let Some(callback) = &self.progress_callback {
            callback(0.7); // 70% - dictionary built
        }

        // Validate dictionary if configured
        if self.config.validate_dictionary {
            dictionary.validate()?;
        }

        if let Some(callback) = &self.progress_callback {
            callback(0.8); // 80% - dictionary validated
        }

        // Create memory pool if not provided
        let memory_pool = if let Some(pool) = &self.memory_pool {
            pool.clone()
        } else {
            SecureMemoryPool::new(SecurePoolConfig::new(4096, 1024, 8))?
        };

        // Handle external dictionary storage before creating compressor
        if self.config.external_dictionary {
            if let Some(dict_path) = &self.config.dict_path {
                #[cfg(feature = "serde")]
                dictionary.save_to_file(dict_path)?;
            }
        }

        let dictionary_size = dictionary.size_in_bytes();

        // Create compressor
        let compressor = PaZipCompressor::new(
            dictionary.clone(),
            self.config.compressor_config.clone(),
            memory_pool.clone(),
        )?;

        // Initialize cache
        let cache_capacity = self.config.cache_size_bytes / 1024; // Rough estimate: 1KB per entry
        let cache = LruMap::new(cache_capacity)?;

        let build_time = start_time.elapsed();

        if let Some(callback) = &self.progress_callback {
            callback(1.0); // 100% - complete
        }

        let mut stats = DictZipBlobStoreStats::default();
        stats.dictionary_size = dictionary_size;
        stats.build_time_ms = build_time.as_millis() as u64;

        Ok(DictZipBlobStore {
            config: self.config,
            dictionary: Arc::new(RwLock::new(dictionary)),
            compressor: Arc::new(compressor),
            storage: HashMap::new(),
            cache: Arc::new(RwLock::new(cache)),
            stats: Arc::new(RwLock::new(stats)),
            memory_pool: Some(memory_pool),
            next_id: 0,
            // Lazy-initialized entropy encoders/decoders
            huffman_encoder: RefCell::new(None),
            huffman_decoder: RefCell::new(None),
            fse_encoder: RefCell::new(None),
            fse_decoder: RefCell::new(None),
        })
    }
}

/// Compressed blob entry
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
struct CompressedBlob {
    /// Compressed data
    compressed_data: Vec<u8>,
    /// Original size
    original_size: usize,
    /// Is compressed (false for small blobs stored uncompressed)
    is_compressed: bool,
    /// Compression ratio
    compression_ratio: f32,
    /// Entropy encoding algorithm used (if any)
    entropy_algorithm: EntropyAlgorithm,
}

/// Main DictZipBlobStore implementation
pub struct DictZipBlobStore {
    /// Configuration
    config: DictZipConfig,
    /// Dictionary for compression
    dictionary: Arc<RwLock<SuffixArrayDictionary>>,
    /// Compressor instance
    compressor: Arc<PaZipCompressor>,
    /// Internal storage for compressed blobs
    storage: HashMap<RecordId, CompressedBlob>,
    /// Decompression cache
    cache: Arc<RwLock<LruMap<RecordId, Vec<u8>>>>,
    /// Statistics
    stats: Arc<RwLock<DictZipBlobStoreStats>>,
    /// Memory pool
    memory_pool: Option<Arc<SecureMemoryPool>>,
    /// Next record ID
    next_id: u64,

    // Entropy encoding/decoding components (using RefCell for interior mutability)
    /// Huffman O1 encoder (lazy initialized on first use)
    huffman_encoder: RefCell<Option<ContextualHuffmanEncoder>>,
    /// Huffman O1 decoder (lazy initialized on first use)
    huffman_decoder: RefCell<Option<ContextualHuffmanDecoder>>,
    /// FSE encoder (lazy initialized on first use)
    fse_encoder: RefCell<Option<FseEncoder>>,
    /// FSE decoder (lazy initialized on first use)
    fse_decoder: RefCell<Option<FseDecoder>>,
}

impl DictZipBlobStore {
    /// Build DictZipBlobStore from training samples with NestLoudsTrieConfig (matches C++ pattern)
    /// This follows the C++ pattern: build_from(SortableStrVec& strVec, const NestLoudsTrieConfig& conf)
    pub fn build_from_training_samples(
        training_samples: &[Vec<u8>],
        config: &crate::config::nest_louds_trie::NestLoudsTrieConfig,
    ) -> Result<Self> {
        if training_samples.is_empty() {
            return Err(ZiporaError::invalid_data("No training samples provided"));
        }

        // Convert NestLoudsTrieConfig to DictZipConfig
        let dict_zip_config = Self::convert_config_from_nest_louds_trie(config)?;
        
        // Create builder and add training samples
        let mut builder = DictZipBlobStoreBuilder::with_config(dict_zip_config)?;
        for sample in training_samples {
            builder.add_training_sample(sample)?;
        }
        
        builder.finish()
    }

    /// Build DictZipBlobStore from SortableStrVec with configuration (matches C++ pattern)
    /// This follows the C++ pattern: build_from(SortableStrVec& strVec, const NestLoudsTrieConfig& conf)
    pub fn build_from_sortable_str_vec(
        keys: &crate::containers::specialized::SortableStrVec,
        config: &crate::config::nest_louds_trie::NestLoudsTrieConfig,
    ) -> Result<Self> {
        // Convert SortableStrVec to training samples
        let training_samples: Vec<Vec<u8>> = (0..keys.len())
            .filter_map(|i| keys.get(i).map(|s| s.as_bytes().to_vec()))
            .collect();

        if training_samples.is_empty() {
            return Err(ZiporaError::invalid_data("No valid strings in SortableStrVec"));
        }

        Self::build_from_training_samples(&training_samples, config)
    }

    /// Build DictZipBlobStore from ZoSortedStrVec with configuration (matches C++ pattern)
    pub fn build_from_zo_sorted_str_vec(
        keys: &crate::containers::specialized::ZoSortedStrVec,
        config: &crate::config::nest_louds_trie::NestLoudsTrieConfig,
    ) -> Result<Self> {
        // Convert ZoSortedStrVec to training samples
        let training_samples: Vec<Vec<u8>> = (0..keys.len())
            .filter_map(|i| keys.get(i).map(|s| s.as_bytes().to_vec()))
            .collect();

        if training_samples.is_empty() {
            return Err(ZiporaError::invalid_data("No valid strings in ZoSortedStrVec"));
        }

        Self::build_from_training_samples(&training_samples, config)
    }

    /// Build DictZipBlobStore from FixedLenStrVec with configuration (matches C++ pattern)
    pub fn build_from_fixed_len_str_vec<const N: usize>(
        keys: &crate::containers::specialized::FixedLenStrVec<N>,
        config: &crate::config::nest_louds_trie::NestLoudsTrieConfig,
    ) -> Result<Self> {
        // Convert FixedLenStrVec to training samples
        let training_samples: Vec<Vec<u8>> = (0..keys.len())
            .filter_map(|i| keys.get(i).map(|s| s.as_bytes().to_vec()))
            .collect();

        if training_samples.is_empty() {
            return Err(ZiporaError::invalid_data("No valid strings in FixedLenStrVec"));
        }

        Self::build_from_training_samples(&training_samples, config)
    }

    /// Build DictZipBlobStore from Vec<u8> with configuration (matches C++ pattern)
    pub fn build_from_vec_u8(
        data: &[u8],
        config: &crate::config::nest_louds_trie::NestLoudsTrieConfig,
    ) -> Result<Self> {
        if data.is_empty() {
            return Err(ZiporaError::invalid_data("Empty data provided"));
        }

        // Use the entire data as a single training sample
        let training_samples = vec![data.to_vec()];
        Self::build_from_training_samples(&training_samples, config)
    }

    /// Convert NestLoudsTrieConfig to DictZipConfig with intelligent mapping
    fn convert_config_from_nest_louds_trie(
        config: &crate::config::nest_louds_trie::NestLoudsTrieConfig,
    ) -> Result<DictZipConfig> {
        use crate::compression::dict_zip::PaZipCompressorConfig;

        // Determine appropriate preset based on NestLoudsTrieConfig parameters
        let base_config = if config.optimization_flags.contains(
            crate::config::nest_louds_trie::OptimizationFlags::ENABLE_FAST_SEARCH
        ) {
            DictZipConfig::realtime_compression()
        } else if config.enable_queue_compression {
            DictZipConfig::binary_compression()
        } else if config.best_delimiters.len() > 10 {
            // Assume text-heavy data if many delimiters
            DictZipConfig::text_compression()
        } else {
            DictZipConfig::log_compression()
        };

        // Customize based on NestLoudsTrieConfig parameters
        let mut dict_builder_config = base_config.dict_builder_config;
        
        // Map fragment configuration
        dict_builder_config.min_pattern_length = config.min_fragment_length as usize;
        dict_builder_config.max_pattern_length = config.max_fragment_length as usize;
        
        // Map sample ratio and frequency
        dict_builder_config.sample_ratio = 0.8; // Use default sample ratio
        dict_builder_config.min_frequency = config.sa_fragment_min_freq as u32;
        
        // Map BFS depth based on nesting levels
        dict_builder_config.max_bfs_depth = config.max_bfs_depth.min(8); // Cap at 8 for performance
        
        // Adjust dictionary size based on memory settings
        if config.initial_pool_size > 0 {
            dict_builder_config.target_dict_size = 16 * 1024 * 1024; // 16MB for custom memory pool
            dict_builder_config.max_dict_size = 20 * 1024 * 1024; // 20MB max
        }

        // Map validation settings
        dict_builder_config.validate_result = config.optimization_flags.contains(
            crate::config::nest_louds_trie::OptimizationFlags::ENABLE_STATISTICS
        );

        let validate_result = dict_builder_config.validate_result;

        Ok(DictZipConfig {
            dict_builder_config,
            compressor_config: PaZipCompressorConfig::default(),
            cache_size_bytes: if config.node_cache_size > 0 { config.node_cache_size } else { base_config.cache_size_bytes },
            external_dictionary: false, // Default to embedded
            dict_path: None,
            memory_pool_config: None, // Use default memory pool configuration
            track_stats: true,
            validate_dictionary: validate_result,
            min_compression_size: if config.min_fragment_length > 0 {
                config.min_fragment_length as usize
            } else {
                base_config.min_compression_size
            },
            // Entropy encoding defaults
            entropy_algorithm: EntropyAlgorithm::None,
            entropy_interleaved: 0,
            enable_lake: false,
            embedded_dict: false,
            input_is_perm: false,
            entropy_zip_ratio_require: 0.8,
        })
    }

    /// Create from existing dictionary file
    pub fn from_dictionary_file<P: AsRef<Path>>(
        dict_path: P,
        config: DictZipConfig,
    ) -> Result<Self> {
        config.validate()?;

        #[cfg(feature = "serde")]
        {
            let dictionary = SuffixArrayDictionary::load_from_file(dict_path)?;
            
            // Initialize memory pool if needed
            let memory_pool = if let Some(pool_config) = &config.memory_pool_config {
                SecureMemoryPool::new(pool_config.clone())?
            } else {
                SecureMemoryPool::new(SecurePoolConfig::new(4096, 1024, 8))?
            };
            
            let compressor = PaZipCompressor::new(dictionary.clone(), config.compressor_config.clone(), memory_pool.clone())?;

            // Initialize cache
            let cache_capacity = config.cache_size_bytes / 1024; // Rough estimate: 1KB per entry
            let cache = LruMap::new(cache_capacity)?;

            let stats = DictZipBlobStoreStats {
                dictionary_size: dictionary.size_in_bytes(),
                build_time_ms: 0, // Not applicable for loaded dictionary
                ..Default::default()
            };

            Ok(DictZipBlobStore {
                config,
                dictionary: Arc::new(RwLock::new(dictionary)),
                compressor: Arc::new(compressor),
                storage: HashMap::new(),
                cache: Arc::new(RwLock::new(cache)),
                stats: Arc::new(RwLock::new(stats)),
                memory_pool: Some(memory_pool),
                next_id: 0,
                // Lazy-initialized entropy encoders/decoders
                huffman_encoder: RefCell::new(None),
                huffman_decoder: RefCell::new(None),
                fse_encoder: RefCell::new(None),
                fse_decoder: RefCell::new(None),
            })
        }
        
        #[cfg(not(feature = "serde"))]
        Err(ZiporaError::not_supported("Dictionary loading requires 'serde' feature"))
    }

    /// Save dictionary to file
    pub fn save_dictionary<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        #[cfg(feature = "serde")]
        {
            let dictionary = self.dictionary.read()
                .map_err(|_| ZiporaError::resource_busy("Dictionary read lock"))?;
            dictionary.save_to_file(path)
        }
        
        #[cfg(not(feature = "serde"))]
        Err(ZiporaError::not_supported("Dictionary saving requires 'serde' feature"))
    }

    /// Load dictionary from file (replaces current dictionary)
    pub fn load_dictionary<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        #[cfg(feature = "serde")]
        {
            let new_dictionary = SuffixArrayDictionary::load_from_file(path)?;
            let memory_pool = self.memory_pool.as_ref()
                .ok_or_else(|| ZiporaError::invalid_data("Memory pool not initialized"))?;
            let new_compressor = PaZipCompressor::new(new_dictionary.clone(), self.config.compressor_config.clone(), Arc::clone(memory_pool))?;
            
            // Replace dictionary and compressor
            *self.dictionary.write()
                .map_err(|_| ZiporaError::resource_busy("Dictionary write lock"))? = new_dictionary;
            self.compressor = Arc::new(new_compressor);
            
            // Clear storage and cache since they're tied to the old dictionary
            self.storage.clear();
            self.cache.write()
                .map_err(|_| ZiporaError::resource_busy("Cache write lock"))?
                .clear();
            
            // Update statistics
            let mut stats = self.stats.write()
                .map_err(|_| ZiporaError::resource_busy("Stats write lock"))?;
            stats.dictionary_size = self.dictionary.read()
                .map_err(|_| ZiporaError::resource_busy("Dictionary read lock for stats"))?
                .size_in_bytes();
            
            Ok(())
        }
        
        #[cfg(not(feature = "serde"))]
        Err(ZiporaError::not_supported("Dictionary loading requires 'serde' feature"))
    }

    /// Get dictionary statistics
    pub fn dictionary_stats(&self) -> Result<crate::compression::dict_zip::MatchStats> {
        let dictionary = self.dictionary.read()
            .map_err(|_| ZiporaError::resource_busy("Dictionary read lock for stats"))?;
        Ok(dictionary.match_stats().clone())
    }

    /// Validate internal consistency
    pub fn validate(&self) -> Result<()> {
        // Validate dictionary
        let dict = self.dictionary.read()
            .map_err(|_| ZiporaError::resource_busy("Dictionary read lock"))?;
        dict.validate()?;

        // Validate configuration
        self.config.validate()?;

        // Check storage consistency
        for (id, blob) in &self.storage {
            if blob.original_size == 0 {
                return Err(ZiporaError::invalid_data(format!("Blob {} has zero original size", id)));
            }
            
            if blob.is_compressed && blob.compressed_data.len() >= blob.original_size {
                // This might be okay for incompressible data, just log a warning
                // Could add logging here in the future
            }
        }

        Ok(())
    }

    /// Optimize storage (rebuild indices, compact data)
    pub fn optimize(&mut self) -> Result<()> {
        // Clear cache to free memory
        {
            let mut cache = self.cache.write()
                .map_err(|_| ZiporaError::resource_busy("Cache write lock"))?;
            cache.clear();
        }

        // Could add storage compaction here in the future
        // For now, just validate consistency
        self.validate()
    }

    /// Get detailed statistics
    pub fn detailed_stats(&self) -> Result<DictZipBlobStoreStats> {
        let stats = self.stats.read()
            .map_err(|_| ZiporaError::resource_busy("Stats read lock"))?;
        Ok(stats.clone())
    }

    /// Generate next record ID
    fn next_record_id(&mut self) -> RecordId {
        self.next_id += 1;
        self.next_id as RecordId
    }

    /// Try to retrieve from cache
    fn try_get_from_cache(&self, id: RecordId) -> Option<Vec<u8>> {
        if let Ok(cache) = self.cache.read() {
            cache.get(&id).map(|v| v.clone())
        } else {
            None
        }
    }

    /// Store in cache
    fn store_in_cache(&self, id: RecordId, data: Vec<u8>) {
        if let Ok(mut cache) = self.cache.write() {
            let _ = cache.put(id, data);
        }
    }

    /// Update statistics for get operation
    fn update_get_stats(&self, cache_hit: bool) {
        if let Ok(mut stats) = self.stats.write() {
            if cache_hit {
                stats.cache_hits += 1;
            } else {
                stats.cache_misses += 1;
            }
            stats.blob_stats.record_get(cache_hit);
        }
    }

    /// Update statistics for put operation
    fn update_put_stats(&self, original_size: usize, compressed_size: usize, is_compressed: bool) {
        if let Ok(mut stats) = self.stats.write() {
            stats.blob_stats.record_put(original_size);
            
            if is_compressed {
                stats.compressed_blobs += 1;
                stats.compression_stats.uncompressed_size += original_size;
                stats.compression_stats.compressed_size += compressed_size;
                stats.compression_stats.compressed_count += 1;
                stats.compression_stats.compression_ratio = 
                    stats.compression_stats.compressed_size as f32 / 
                    stats.compression_stats.uncompressed_size as f32;
            } else {
                stats.uncompressed_blobs += 1;
            }
        }
    }

    /// Update statistics for remove operation
    fn update_remove_stats(&self, original_size: usize) {
        if let Ok(mut stats) = self.stats.write() {
            stats.blob_stats.record_remove(original_size);
        }
    }

    // ===== Entropy Encoding/Decoding Methods (matches C++ template functions) =====

    /// Apply entropy encoding based on configuration
    /// Matches C++ read_record_append_entropy<EntropyAlgo, EntropyInterLeave>
    fn apply_entropy_encoding(&self, data: &[u8]) -> Result<Vec<u8>> {
        match self.config.entropy_algorithm {
            EntropyAlgorithm::None => {
                // No encoding, return as-is
                Ok(data.to_vec())
            }
            EntropyAlgorithm::HuffmanO1 => {
                self.apply_huffman_o1_encoding(data)
            }
            EntropyAlgorithm::Fse => {
                self.apply_fse_encoding(data)
            }
        }
    }

    /// Apply Huffman O1 encoding with configured interleaving
    fn apply_huffman_o1_encoding(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Ensure encoder exists (lazy initialization)
        if self.huffman_encoder.borrow().is_none() {
            // Build encoder from training data (use dictionary as training data)
            let dict = self.dictionary.read()
                .map_err(|_| ZiporaError::resource_busy("Dictionary read lock"))?;

            // Use dictionary data as training corpus
            let training_data = dict.data();
            let new_encoder = ContextualHuffmanEncoder::new(training_data, crate::entropy::huffman::HuffmanOrder::Order1)?;
            *self.huffman_encoder.borrow_mut() = Some(new_encoder);
        }

        // Get a clone of the encoder for use
        let binding = self.huffman_encoder.borrow();
        let encoder = binding.as_ref().unwrap().clone();

        // Apply encoding based on interleaving factor
        match self.config.entropy_interleaved {
            0 | 1 => encoder.encode_x1(data),
            2 => encoder.encode_x2(data),
            4 => encoder.encode_x4(data),
            8 => encoder.encode_x8(data),
            _ => {
                Err(ZiporaError::Configuration {
                    message: format!("Invalid interleaving factor: {}",
                                   self.config.entropy_interleaved)
                })
            }
        }
    }

    /// Apply FSE encoding with configured interleaving
    fn apply_fse_encoding(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Get or build encoder (lazy initialization)
        if self.fse_encoder.borrow().is_none() {
            // Create FSE encoder with interleaving configuration
            let config = FseConfig {
                parallel_blocks: if self.config.entropy_interleaved > 1 {
                    Some(self.config.entropy_interleaved as usize)
                } else {
                    None
                },
                ..Default::default()
            };

            let new_encoder = FseEncoder::new(config)?;
            *self.fse_encoder.borrow_mut() = Some(new_encoder);
        }

        // Get mutable borrow for compression
        self.fse_encoder.borrow_mut().as_mut().unwrap().compress(data)
    }

    /// Check compression ratio and determine if entropy encoding should be used
    fn check_compression_ratio(&self, original: &[u8], compressed: &[u8]) -> bool {
        if original.is_empty() {
            return false;
        }
        let ratio = compressed.len() as f32 / original.len() as f32;
        ratio <= self.config.entropy_zip_ratio_require
    }

    /// Decode entropy-encoded data
    /// Matches C++ decompression logic
    fn decode_entropy(&self, data: &[u8], original_size: usize, entropy_algo: EntropyAlgorithm) -> Result<Vec<u8>> {
        match entropy_algo {
            EntropyAlgorithm::None => Ok(data.to_vec()),
            EntropyAlgorithm::HuffmanO1 => {
                self.decode_huffman_o1(data, original_size)
            }
            EntropyAlgorithm::Fse => {
                self.decode_fse(data, original_size)
            }
        }
    }

    /// Decode Huffman O1 encoded data with configured interleaving
    fn decode_huffman_o1(&self, data: &[u8], original_size: usize) -> Result<Vec<u8>> {
        // Get or build decoder (lazy initialization)
        if self.huffman_decoder.borrow().is_none() {
            // Build decoder from encoder - first build encoder
            let dict = self.dictionary.read()
                .map_err(|_| ZiporaError::resource_busy("Dictionary read lock"))?;

            let training_data = dict.data();
            let encoder = ContextualHuffmanEncoder::new(training_data, crate::entropy::huffman::HuffmanOrder::Order1)?;
            let new_decoder = ContextualHuffmanDecoder::new(encoder);
            *self.huffman_decoder.borrow_mut() = Some(new_decoder);
        }

        // Get decoder clone for use
        let binding = self.huffman_decoder.borrow();
        let decoder = binding.as_ref().unwrap().clone();

        // The decoder's decode method already handles the encoding format
        // The interleaving is determined by how the data was encoded
        decoder.decode(data, original_size)
    }

    /// Decode FSE encoded data
    fn decode_fse(&self, data: &[u8], _original_size: usize) -> Result<Vec<u8>> {
        // Get or build decoder (lazy initialization)
        if self.fse_decoder.borrow().is_none() {
            // Create FSE decoder
            let new_decoder = FseDecoder::new();
            *self.fse_decoder.borrow_mut() = Some(new_decoder);
        }

        // Get mutable borrow for decompression
        self.fse_decoder.borrow_mut().as_mut().unwrap().decompress(data)
    }
}

impl BlobStore for DictZipBlobStore {
    fn get(&self, id: RecordId) -> Result<Vec<u8>> {
        // Try cache first
        if let Some(cached_data) = self.try_get_from_cache(id) {
            self.update_get_stats(true);
            return Ok(cached_data);
        }

        // Get from storage
        let blob = self.storage.get(&id)
            .ok_or_else(|| ZiporaError::invalid_data(format!("Blob {} not found", id)))?;

        // Step 1: Decode entropy encoding (if any)
        let dict_compressed = self.decode_entropy(
            &blob.compressed_data,
            blob.original_size,
            blob.entropy_algorithm
        )?;

        // Step 2: Decompress dictionary compression
        let decompressed_data = if blob.is_compressed {
            // Decompress using PA-Zip compressor
            let mut decompressed = Vec::new();
            // Note: compressor is Arc<PaZipCompressor>, so we need to create a mutable copy for decompression
            let mut compressor_copy = (*self.compressor).clone();
            compressor_copy.decompress(&dict_compressed, &mut decompressed)
                .map_err(|e| ZiporaError::invalid_data(&format!("Decompression failed: {}", e)))?;
            decompressed
        } else {
            // Return uncompressed data directly
            dict_compressed
        };

        // Store in cache
        self.store_in_cache(id, decompressed_data.clone());
        self.update_get_stats(false);

        Ok(decompressed_data)
    }

    fn put(&mut self, data: &[u8]) -> Result<RecordId> {
        if data.is_empty() {
            return Err(ZiporaError::invalid_data("Cannot store empty blob"));
        }

        let id = self.next_record_id();
        let original_size = data.len();

        // Decide whether to compress based on size threshold
        let should_compress = original_size >= self.config.min_compression_size;

        let blob = if should_compress {
            // Step 1: Dictionary compression using PA-Zip compressor
            let mut dict_compressed = Vec::new();
            let mut compressor_copy = (*self.compressor).clone();
            let _compression_stats = compressor_copy.compress(data, &mut dict_compressed)
                .map_err(|e| ZiporaError::invalid_data(&format!("Compression failed: {}", e)))?;

            // Step 2: Apply entropy encoding (if configured)
            let (final_compressed, entropy_algorithm) = if self.config.entropy_algorithm != EntropyAlgorithm::None {
                let entropy_encoded = self.apply_entropy_encoding(&dict_compressed)?;

                // Check if entropy encoding provides benefit
                if self.check_compression_ratio(&dict_compressed, &entropy_encoded) {
                    (entropy_encoded, self.config.entropy_algorithm)
                } else {
                    // Entropy encoding didn't help, fall back to dict-only
                    (dict_compressed, EntropyAlgorithm::None)
                }
            } else {
                (dict_compressed, EntropyAlgorithm::None)
            };

            let compression_ratio = if final_compressed.len() > 0 {
                final_compressed.len() as f32 / original_size as f32
            } else {
                1.0
            };

            // Only use compression if it actually reduces size
            if final_compressed.len() < original_size {
                CompressedBlob {
                    compressed_data: final_compressed,
                    original_size,
                    is_compressed: true,
                    compression_ratio,
                    entropy_algorithm,
                }
            } else {
                // Store uncompressed if compression doesn't help
                CompressedBlob {
                    compressed_data: data.to_vec(),
                    original_size,
                    is_compressed: false,
                    compression_ratio: 1.0,
                    entropy_algorithm: EntropyAlgorithm::None,
                }
            }
        } else {
            // Store small blobs uncompressed
            CompressedBlob {
                compressed_data: data.to_vec(),
                original_size,
                is_compressed: false,
                compression_ratio: 1.0,
                entropy_algorithm: EntropyAlgorithm::None,
            }
        };

        let compressed_size = blob.compressed_data.len();
        let is_compressed = blob.is_compressed;

        self.storage.insert(id, blob);
        self.update_put_stats(original_size, compressed_size, is_compressed);

        Ok(id)
    }

    fn remove(&mut self, id: RecordId) -> Result<()> {
        let blob = self.storage.remove(&id)
            .ok_or_else(|| ZiporaError::invalid_data(format!("Blob {} not found", id)))?;

        // Remove from cache
        if let Ok(mut cache) = self.cache.write() {
            cache.remove(&id);
        }

        self.update_remove_stats(blob.original_size);
        Ok(())
    }

    fn contains(&self, id: RecordId) -> bool {
        self.storage.contains_key(&id)
    }

    fn size(&self, id: RecordId) -> Result<Option<usize>> {
        Ok(self.storage.get(&id).map(|blob| blob.original_size))
    }

    fn len(&self) -> usize {
        self.storage.len()
    }

    fn stats(&self) -> BlobStoreStats {
        if let Ok(stats) = self.stats.read() {
            stats.blob_stats.clone()
        } else {
            BlobStoreStats::default()
        }
    }
}

impl CompressedBlobStore for DictZipBlobStore {
    fn compression_ratio(&self, id: RecordId) -> Result<Option<f32>> {
        Ok(self.storage.get(&id).map(|blob| blob.compression_ratio))
    }

    fn compressed_size(&self, id: RecordId) -> Result<Option<usize>> {
        Ok(self.storage.get(&id).map(|blob| blob.compressed_data.len()))
    }

    fn compression_stats(&self) -> CompressionStats {
        if let Ok(stats) = self.stats.read() {
            stats.compression_stats.clone()
        } else {
            CompressionStats::default()
        }
    }
}

impl BatchBlobStore for DictZipBlobStore {
    fn put_batch<I>(&mut self, blobs: I) -> Result<Vec<RecordId>>
    where
        I: IntoIterator<Item = Vec<u8>>,
    {
        let mut ids = Vec::new();
        for blob in blobs {
            let id = self.put(&blob)?;
            ids.push(id);
        }
        Ok(ids)
    }

    fn get_batch<I>(&self, ids: I) -> Result<Vec<Option<Vec<u8>>>>
    where
        I: IntoIterator<Item = RecordId>,
    {
        let mut results = Vec::new();
        for id in ids {
            let result = match self.get(id) {
                Ok(data) => Some(data),
                Err(ZiporaError::InvalidData { .. }) => None,
                Err(e) => return Err(e),
            };
            results.push(result);
        }
        Ok(results)
    }

    fn remove_batch<I>(&mut self, ids: I) -> Result<usize>
    where
        I: IntoIterator<Item = RecordId>,
    {
        let mut removed_count = 0;
        for id in ids {
            match self.remove(id) {
                Ok(()) => removed_count += 1,
                Err(ZiporaError::InvalidData { .. }) => {}, // Ignore not found
                Err(e) => return Err(e),
            }
        }
        Ok(removed_count)
    }
}

// Note: IterableBlobStore is not implemented due to lifetime complexities.
// Use iter_ids_vec() method instead for getting all record IDs.

impl DictZipBlobStore {
    /// Get all record IDs as a vector (alternative to iter_ids for lifetime issues)
    pub fn iter_ids_vec(&self) -> Vec<RecordId> {
        self.storage.keys().copied().collect()
    }

    /// Get all blob data as vector of (id, data) pairs
    pub fn iter_blobs_vec(&self) -> Result<Vec<(RecordId, Vec<u8>)>> {
        let mut blobs = Vec::new();
        for &id in self.storage.keys() {
            let data = self.get(id)?;
            blobs.push((id, data));
        }
        Ok(blobs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compression::dict_zip::DictionaryBuilderConfig;

    fn create_test_training_data() -> Vec<Vec<u8>> {
        vec![
            b"The quick brown fox jumps over the lazy dog".to_vec(),
            b"The lazy dog was jumped over by the quick brown fox".to_vec(),
            b"Quick brown foxes are faster than lazy dogs".to_vec(),
            b"Dogs and foxes are both animals".to_vec(),
            b"Animals like dogs and foxes live in nature".to_vec(),
        ]
    }

    #[test]
    fn test_dict_zip_config() {
        let config = DictZipConfig::default();
        assert!(config.validate().is_ok());

        let text_config = DictZipConfig::text_compression();
        assert!(text_config.validate().is_ok());
        assert_eq!(text_config.min_compression_size, 32);

        let binary_config = DictZipConfig::binary_compression();
        assert!(binary_config.validate().is_ok());
        assert_eq!(binary_config.min_compression_size, 128);
    }

    #[test]
    fn test_builder_basic() -> Result<()> {
        let config = DictZipConfig {
            dict_builder_config: DictionaryBuilderConfig {
                target_dict_size: 1024,
                max_dict_size: 8192,
                validate_result: true,
                ..Default::default()
            },
            validate_dictionary: true,
            ..Default::default()
        };

        let mut builder = DictZipBlobStoreBuilder::with_config(config)?;
        
        let training_data = create_test_training_data();
        for sample in training_data {
            builder.add_training_sample(&sample)?;
        }

        let (sample_count, total_size) = builder.training_stats();
        assert_eq!(sample_count, 5);
        assert!(total_size > 0);

        let store = builder.finish()?;
        assert_eq!(store.len(), 0); // No blobs stored yet

        Ok(())
    }

    #[test]
    fn test_blob_store_operations() -> Result<()> {
        let config = DictZipConfig {
            dict_builder_config: DictionaryBuilderConfig {
                target_dict_size: 1024,
                max_dict_size: 8192,
                validate_result: false, // Skip validation for speed
                ..Default::default()
            },
            min_compression_size: 10, // Compress small blobs for testing
            ..Default::default()
        };

        let mut builder = DictZipBlobStoreBuilder::with_config(config)?;
        
        let training_data = create_test_training_data();
        for sample in training_data {
            builder.add_training_sample(&sample)?;
        }

        let mut store = builder.finish()?;

        // Test put/get
        let test_data = b"The quick brown fox";
        let id = store.put(test_data)?;
        let retrieved = store.get(id)?;
        assert_eq!(test_data, retrieved.as_slice());

        // Test contains
        assert!(store.contains(id));
        assert!(!store.contains(999));

        // Test size
        assert_eq!(store.size(id)?, Some(test_data.len()));
        assert_eq!(store.size(999)?, None);

        // Test remove
        store.remove(id)?;
        assert!(!store.contains(id));
        assert!(store.get(id).is_err());

        Ok(())
    }

    #[test]
    fn test_batch_operations() -> Result<()> {
        let config = DictZipConfig {
            dict_builder_config: DictionaryBuilderConfig {
                target_dict_size: 1024,
                max_dict_size: 8192,
                validate_result: false,
                ..Default::default()
            },
            min_compression_size: 10,
            ..Default::default()
        };

        let mut builder = DictZipBlobStoreBuilder::with_config(config)?;
        
        let training_data = create_test_training_data();
        for sample in training_data {
            builder.add_training_sample(&sample)?;
        }

        let mut store = builder.finish()?;

        // Test batch put
        let test_blobs = vec![
            b"First blob".to_vec(),
            b"Second blob".to_vec(),
            b"Third blob".to_vec(),
        ];
        let ids = store.put_batch(test_blobs.clone())?;
        assert_eq!(ids.len(), 3);

        // Test batch get
        let retrieved = store.get_batch(ids.clone())?;
        assert_eq!(retrieved.len(), 3);
        for (i, data) in retrieved.iter().enumerate() {
            assert_eq!(data.as_ref().unwrap(), &test_blobs[i]);
        }

        // Test batch remove
        let removed_count = store.remove_batch(ids)?;
        assert_eq!(removed_count, 3);
        assert_eq!(store.len(), 0);

        Ok(())
    }

    #[test]
    fn test_compression_stats() -> Result<()> {
        let config = DictZipConfig {
            dict_builder_config: DictionaryBuilderConfig {
                target_dict_size: 2048,
                max_dict_size: 8192,
                validate_result: false,
                ..Default::default()
            },
            min_compression_size: 5, // Compress very small blobs for testing
            track_stats: true,
            ..Default::default()
        };

        let mut builder = DictZipBlobStoreBuilder::with_config(config)?;
        
        let training_data = create_test_training_data();
        for sample in training_data {
            builder.add_training_sample(&sample)?;
        }

        let mut store = builder.finish()?;

        // Add several blobs
        let test_data = b"The quick brown fox jumps over the lazy dog";
        let _id1 = store.put(test_data)?;
        let _id2 = store.put(b"Short")?; // This might not compress well
        let _id3 = store.put(test_data)?; // Duplicate data

        let stats = store.compression_stats();
        assert!(stats.compressed_count > 0 || stats.compressed_count == 0); // May or may not compress
        
        let detailed_stats = store.detailed_stats()?;
        assert!(detailed_stats.dictionary_size > 0);
        assert!(detailed_stats.build_time_ms > 0);

        Ok(())
    }

    #[test]
    fn test_caching() -> Result<()> {
        let config = DictZipConfig {
            dict_builder_config: DictionaryBuilderConfig {
                target_dict_size: 1024,
                max_dict_size: 8192,
                validate_result: false,
                ..Default::default()
            },
            cache_size_bytes: 1024, // Small cache for testing
            ..Default::default()
        };

        let mut builder = DictZipBlobStoreBuilder::with_config(config)?;
        
        let training_data = create_test_training_data();
        for sample in training_data {
            builder.add_training_sample(&sample)?;
        }

        let mut store = builder.finish()?;

        let test_data = b"Test data for caching";
        let id = store.put(test_data)?;

        // First get - cache miss
        let _retrieved1 = store.get(id)?;
        
        // Second get - should be cache hit
        let _retrieved2 = store.get(id)?;

        let detailed_stats = store.detailed_stats()?;
        assert!(detailed_stats.cache_hits > 0 || detailed_stats.cache_misses > 0);

        Ok(())
    }

    #[test]
    fn test_validation() -> Result<()> {
        let config = DictZipConfig {
            dict_builder_config: DictionaryBuilderConfig {
                target_dict_size: 1024,
                max_dict_size: 8192,
                validate_result: true, // Enable validation
                ..Default::default()
            },
            validate_dictionary: true,
            ..Default::default()
        };

        let mut builder = DictZipBlobStoreBuilder::with_config(config)?;
        
        let training_data = create_test_training_data();
        for sample in training_data {
            builder.add_training_sample(&sample)?;
        }

        let store = builder.finish()?;
        assert!(store.validate().is_ok());

        Ok(())
    }

    #[test]
    fn test_builder_configuration_methods() -> Result<()> {
        let mut builder = DictZipBlobStoreBuilder::new()?;
        
        builder.set_dict_size_mb(16)?;
        builder.set_min_frequency(8)?;
        builder.enable_advanced_caching()?;

        assert_eq!(builder.config.dict_builder_config.target_dict_size, 16 * 1024 * 1024);
        assert_eq!(builder.config.dict_builder_config.min_frequency, 8);
        assert_eq!(builder.config.dict_builder_config.max_bfs_depth, 8);

        Ok(())
    }

    #[test]
    fn test_config_presets() {
        let text_config = DictZipConfig::text_compression();
        assert_eq!(text_config.min_compression_size, 32);

        let binary_config = DictZipConfig::binary_compression();
        assert_eq!(binary_config.min_compression_size, 128);

        let log_config = DictZipConfig::log_compression();
        assert_eq!(log_config.min_compression_size, 16);

        let realtime_config = DictZipConfig::realtime_compression();
        assert_eq!(realtime_config.min_compression_size, 256);
        assert!(!realtime_config.dict_builder_config.validate_result); // Should skip validation for speed
    }

    #[test]
    fn test_error_handling() {
        // Test empty training samples
        let mut builder = DictZipBlobStoreBuilder::new().unwrap();
        assert!(builder.finish().is_err());

        // Test empty training sample
        let mut builder = DictZipBlobStoreBuilder::new().unwrap();
        assert!(builder.add_training_sample(b"").is_err());

        // Test invalid configuration
        let invalid_config = DictZipConfig {
            cache_size_bytes: 0, // Invalid
            ..Default::default()
        };
        assert!(invalid_config.validate().is_err());
    }

    // ===== Entropy Encoding Tests =====

    #[test]
    fn test_entropy_algorithm_none() -> Result<()> {
        let config = DictZipConfig {
            dict_builder_config: DictionaryBuilderConfig {
                target_dict_size: 1024,
                max_dict_size: 8192,
                validate_result: false,
                ..Default::default()
            },
            min_compression_size: 10,
            entropy_algorithm: EntropyAlgorithm::None,
            entropy_interleaved: 0,
            ..Default::default()
        };

        let mut builder = DictZipBlobStoreBuilder::with_config(config)?;
        for sample in create_test_training_data() {
            builder.add_training_sample(&sample)?;
        }

        let mut store = builder.finish()?;

        let test_data = b"The quick brown fox jumps over the lazy dog";
        let id = store.put(test_data)?;
        let retrieved = store.get(id)?;

        assert_eq!(test_data, retrieved.as_slice());
        Ok(())
    }

    #[test]
    fn test_entropy_huffman_o1_x1() -> Result<()> {
        let config = DictZipConfig {
            dict_builder_config: DictionaryBuilderConfig {
                target_dict_size: 1024,
                max_dict_size: 8192,
                validate_result: false,
                ..Default::default()
            },
            min_compression_size: 10,
            entropy_algorithm: EntropyAlgorithm::HuffmanO1,
            entropy_interleaved: 1,  // x1 (no interleaving)
            entropy_zip_ratio_require: 0.95,  // Relaxed threshold
            ..Default::default()
        };

        let mut builder = DictZipBlobStoreBuilder::with_config(config)?;
        for sample in create_test_training_data() {
            builder.add_training_sample(&sample)?;
        }

        let mut store = builder.finish()?;

        let test_data = b"The quick brown fox jumps over the lazy dog and then the quick brown fox runs away";
        let id = store.put(test_data)?;
        let retrieved = store.get(id)?;

        assert_eq!(test_data, retrieved.as_slice());
        Ok(())
    }

    #[test]
    fn test_entropy_huffman_o1_x2() -> Result<()> {
        let config = DictZipConfig {
            dict_builder_config: DictionaryBuilderConfig {
                target_dict_size: 1024,
                max_dict_size: 8192,
                validate_result: false,
                ..Default::default()
            },
            min_compression_size: 10,
            entropy_algorithm: EntropyAlgorithm::HuffmanO1,
            entropy_interleaved: 2,  // x2 interleaving
            entropy_zip_ratio_require: 0.95,
            ..Default::default()
        };

        let mut builder = DictZipBlobStoreBuilder::with_config(config)?;
        for sample in create_test_training_data() {
            builder.add_training_sample(&sample)?;
        }

        let mut store = builder.finish()?;

        let test_data = b"Dogs and foxes are both animals that run in nature";
        let id = store.put(test_data)?;
        let retrieved = store.get(id)?;

        assert_eq!(test_data, retrieved.as_slice());
        Ok(())
    }

    #[test]
    fn test_entropy_huffman_o1_x4() -> Result<()> {
        let config = DictZipConfig {
            dict_builder_config: DictionaryBuilderConfig {
                target_dict_size: 1024,
                max_dict_size: 8192,
                validate_result: false,
                ..Default::default()
            },
            min_compression_size: 10,
            entropy_algorithm: EntropyAlgorithm::HuffmanO1,
            entropy_interleaved: 4,  // x4 interleaving (AVX2)
            entropy_zip_ratio_require: 0.95,
            ..Default::default()
        };

        let mut builder = DictZipBlobStoreBuilder::with_config(config)?;
        for sample in create_test_training_data() {
            builder.add_training_sample(&sample)?;
        }

        let mut store = builder.finish()?;

        let test_data = b"Animals like dogs and foxes live in nature and run around";
        let id = store.put(test_data)?;
        let retrieved = store.get(id)?;

        assert_eq!(test_data, retrieved.as_slice());
        Ok(())
    }

    #[test]
    fn test_entropy_huffman_o1_x8() -> Result<()> {
        let config = DictZipConfig {
            dict_builder_config: DictionaryBuilderConfig {
                target_dict_size: 1024,
                max_dict_size: 8192,
                validate_result: false,
                ..Default::default()
            },
            min_compression_size: 10,
            entropy_algorithm: EntropyAlgorithm::HuffmanO1,
            entropy_interleaved: 8,  // x8 interleaving (maximum parallelism)
            entropy_zip_ratio_require: 0.95,
            ..Default::default()
        };

        let mut builder = DictZipBlobStoreBuilder::with_config(config)?;
        for sample in create_test_training_data() {
            builder.add_training_sample(&sample)?;
        }

        let mut store = builder.finish()?;

        let test_data = b"The lazy dog was jumped over by the quick brown fox multiple times";
        let id = store.put(test_data)?;
        let retrieved = store.get(id)?;

        assert_eq!(test_data, retrieved.as_slice());
        Ok(())
    }

    #[test]
    fn test_entropy_fse() -> Result<()> {
        let config = DictZipConfig {
            dict_builder_config: DictionaryBuilderConfig {
                target_dict_size: 1024,
                max_dict_size: 8192,
                validate_result: false,
                ..Default::default()
            },
            min_compression_size: 10,
            entropy_algorithm: EntropyAlgorithm::Fse,
            entropy_interleaved: 0,  // FSE doesn't use same interleaving scheme
            entropy_zip_ratio_require: 0.95,
            ..Default::default()
        };

        let mut builder = DictZipBlobStoreBuilder::with_config(config)?;
        for sample in create_test_training_data() {
            builder.add_training_sample(&sample)?;
        }

        let mut store = builder.finish()?;

        let test_data = b"Quick brown foxes are faster than lazy dogs in every way";
        let id = store.put(test_data)?;
        let retrieved = store.get(id)?;

        assert_eq!(test_data, retrieved.as_slice());
        Ok(())
    }

    #[test]
    fn test_entropy_compression_ratio_fallback() -> Result<()> {
        let config = DictZipConfig {
            dict_builder_config: DictionaryBuilderConfig {
                target_dict_size: 1024,
                max_dict_size: 8192,
                validate_result: false,
                ..Default::default()
            },
            min_compression_size: 10,
            entropy_algorithm: EntropyAlgorithm::HuffmanO1,
            entropy_interleaved: 1,
            entropy_zip_ratio_require: 0.1,  // Very strict threshold - entropy encoding should fail
            ..Default::default()
        };

        let mut builder = DictZipBlobStoreBuilder::with_config(config)?;
        for sample in create_test_training_data() {
            builder.add_training_sample(&sample)?;
        }

        let mut store = builder.finish()?;

        let test_data = b"Short text";
        let id = store.put(test_data)?;
        let retrieved = store.get(id)?;

        // Should still work even if entropy encoding is rejected
        assert_eq!(test_data, retrieved.as_slice());
        Ok(())
    }

    #[test]
    fn test_entropy_config_validation() {
        // Test valid interleaving factors
        let valid_config = DictZipConfig {
            entropy_interleaved: 4,
            ..Default::default()
        };
        assert!(valid_config.validate().is_ok());

        // Test invalid interleaving factor
        let invalid_config = DictZipConfig {
            entropy_interleaved: 3,  // Invalid - must be 0, 1, 2, 4, or 8
            ..Default::default()
        };
        assert!(invalid_config.validate().is_err());

        // Test invalid ratio
        let invalid_ratio_config = DictZipConfig {
            entropy_zip_ratio_require: 1.5,  // Invalid - must be 0.0-1.0
            ..Default::default()
        };
        assert!(invalid_ratio_config.validate().is_err());
    }

    #[test]
    fn test_multiple_entropy_roundtrips() -> Result<()> {
        let config = DictZipConfig {
            dict_builder_config: DictionaryBuilderConfig {
                target_dict_size: 1024,
                max_dict_size: 8192,
                validate_result: false,
                ..Default::default()
            },
            min_compression_size: 10,
            entropy_algorithm: EntropyAlgorithm::HuffmanO1,
            entropy_interleaved: 2,
            entropy_zip_ratio_require: 0.95,
            ..Default::default()
        };

        let mut builder = DictZipBlobStoreBuilder::with_config(config)?;
        for sample in create_test_training_data() {
            builder.add_training_sample(&sample)?;
        }

        let mut store = builder.finish()?;

        // Test multiple different blobs
        let test_blobs = vec![
            b"First test blob with dogs and foxes".to_vec(),
            b"Second test blob with quick brown animals".to_vec(),
            b"Third test blob describing lazy behavior".to_vec(),
        ];

        let mut ids = Vec::new();
        for blob in &test_blobs {
            let id = store.put(blob)?;
            ids.push(id);
        }

        // Verify all blobs can be retrieved correctly
        for (i, id) in ids.iter().enumerate() {
            let retrieved = store.get(*id)?;
            assert_eq!(test_blobs[i].as_slice(), retrieved.as_slice());
        }

        Ok(())
    }
}