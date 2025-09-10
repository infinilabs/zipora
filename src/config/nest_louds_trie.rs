//! Configuration for Nested LOUDS Trie data structures.
//! 
//! This module provides comprehensive configuration for Nested LOUDS Trie implementations,
//! enabling fine-grained control over construction parameters, optimization settings,
//! memory management, and performance characteristics.

use super::{Config, ValidationError, parse_env_var, parse_env_bool};
use crate::error::{Result, ZiporaError};
use std::collections::HashSet;
use std::path::Path;
use bitflags::bitflags;

bitflags! {
    /// Optimization flags for Nested LOUDS Trie construction.
    /// 
    /// These flags control various optimization strategies during trie construction
    /// and operation, allowing fine-tuned performance characteristics.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct OptimizationFlags: u64 {
        /// Reserved flag for future use
        const RESERVED = 0x0001;
        
        /// Enable forward delimiter search optimization
        /// 
        /// This optimization improves performance when searching for delimiters
        /// in forward direction during fragment processing.
        const SEARCH_DELIM_FORWARD = 0x0002;
        
        /// Cut fragments on punctuation marks
        /// 
        /// This optimization automatically splits fragments at punctuation
        /// boundaries, improving compression and search performance.
        const CUT_FRAG_ON_PUNCT = 0x0004;
        
        /// Use DAWG string pool optimization
        /// 
        /// This enables Directed Acyclic Word Graph string pool sharing,
        /// reducing memory usage for common string patterns.
        const USE_DAWG_STR_POOL = 0x0008;
        
        /// Enable suffix array local matching
        /// 
        /// This optimization uses suffix arrays for fast local string matching,
        /// improving search performance at the cost of memory usage.
        const USE_SUFFIX_ARRAY_LOCAL_MATCH = 0x0010;
        
        /// Enable queue compression for memory efficiency
        /// 
        /// This reduces temporary memory usage during construction by
        /// compressing intermediate data structures.
        const ENABLE_QUEUE_COMPRESSION = 0x0020;
        
        /// Enable fast search optimizations
        /// 
        /// This enables various fast search optimizations including
        /// SIMD acceleration and hardware-specific optimizations.
        const ENABLE_FAST_SEARCH = 0x0040;
        
        /// Use mixed core link strategy
        /// 
        /// This optimization uses mixed core link strategies for
        /// better performance on varied data patterns.
        const USE_MIXED_CORE_LINK = 0x0080;
        
        /// Speed up nested trie building
        /// 
        /// This enables aggressive optimizations during nested trie
        /// construction, trading memory for build speed.
        const SPEEDUP_NEST_TRIE_BUILD = 0x0100;
        
        /// Enable statistics collection
        /// 
        /// This enables comprehensive statistics collection during
        /// construction and operation for performance analysis.
        const ENABLE_STATISTICS = 0x0200;
        
        /// Enable performance profiling
        /// 
        /// This enables detailed performance profiling and timing
        /// collection for optimization analysis.
        const ENABLE_PROFILING = 0x0400;
        
        /// Use huge pages for memory allocation
        /// 
        /// This enables huge page allocation for improved memory
        /// performance on systems that support it.
        const USE_HUGEPAGES = 0x0800;
        
        /// Enable cache optimization
        /// 
        /// This enables cache-aware optimizations including
        /// cache-line alignment and prefetch hints.
        const ENABLE_CACHE_OPTIMIZATION = 0x1000;
        
        /// Enable SIMD acceleration
        /// 
        /// This enables SIMD acceleration using AVX2, BMI2,
        /// and other hardware acceleration features.
        const ENABLE_SIMD_ACCELERATION = 0x2000;
        
        /// Enable parallel construction
        /// 
        /// This enables parallel trie construction using
        /// multiple threads for improved build performance.
        const ENABLE_PARALLEL_CONSTRUCTION = 0x4000;
    }
}

impl Default for OptimizationFlags {
    fn default() -> Self {
        // Enable commonly beneficial optimizations by default
        Self::SEARCH_DELIM_FORWARD
            | Self::CUT_FRAG_ON_PUNCT
            | Self::ENABLE_FAST_SEARCH
            | Self::ENABLE_CACHE_OPTIMIZATION
            | Self::ENABLE_SIMD_ACCELERATION
    }
}

/// Temporary file usage levels for memory management during construction.
/// 
/// These levels control how temporary files are used to manage memory
/// during large trie construction operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TempLevel {
    /// Default smart temporary file usage (level 0)
    /// 
    /// The system automatically determines the best temporary file
    /// strategy based on available memory and data size.
    Smart = 0,
    
    /// Use temporary files for BFS queue (level 1)
    /// 
    /// Large breadth-first search queues are stored in temporary
    /// files to reduce memory usage.
    BfsQueue = 1,
    
    /// Swap out link vectors for large objects (level 2)
    /// 
    /// Link vectors are swapped to temporary files when loading
    /// large objects from temporary storage.
    SwapLinkVec = 2,
    
    /// Write link vectors with 2x size and restore (level 3)
    /// 
    /// Link vectors are written to temporary files with double
    /// size allocation and restored when required.
    LinkVecDoubleSize = 3,
    
    /// Save nested string pool to temporary files (level 4)
    /// 
    /// The entire nested string pool is saved to temporary
    /// files to minimize memory usage.
    SaveNestStrPool = 4,
}

impl Default for TempLevel {
    fn default() -> Self {
        Self::Smart
    }
}

/// Compression algorithm selection for core string compression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionAlgorithm {
    /// No compression (fastest)
    None,
    /// LZ4 compression (fast, moderate compression)
    Lz4,
    /// Zstd compression with specified level (1-22)
    Zstd(u8),
    /// Dictionary-based compression
    Dictionary,
    /// Adaptive compression based on data characteristics
    Adaptive,
}

impl Default for CompressionAlgorithm {
    fn default() -> Self {
        Self::Zstd(6) // Balanced compression level
    }
}

/// Comprehensive configuration for Nested LOUDS Trie construction and operation.
/// 
/// This configuration provides fine-grained control over all aspects of
/// Nested LOUDS Trie behavior, including construction parameters, optimization
/// settings, memory management, and performance characteristics.
#[derive(Debug, Clone)]
pub struct NestLoudsTrieConfig {
    // Build-time parameters
    /// Nesting level for the trie (1-16)
    /// 
    /// Higher nesting levels can improve compression for highly repetitive
    /// data but increase construction time and memory usage.
    pub nest_level: u8,
    
    /// Maximum fragment length (-1 = no limit)
    /// 
    /// Negative values indicate no splitting by line feeds.
    /// Positive values set the maximum fragment length for splitting.
    pub max_fragment_length: i32,
    
    /// Minimum fragment length (1-1024)
    /// 
    /// Fragments shorter than this length are not processed separately,
    /// improving performance for small fragments.
    pub min_fragment_length: u32,
    
    /// Minimum link string length (1-255)
    /// 
    /// Strings shorter than this length are not considered for linking,
    /// reducing overhead for short strings.
    pub min_link_str_length: u8,
    
    // Compression settings
    /// Core string compression level (0-22)
    /// 
    /// Higher levels provide better compression but slower performance.
    /// Level 0 disables compression.
    pub core_str_compression_level: u8,
    
    /// Suffix array fragment minimum frequency (1-255)
    /// 
    /// Fragments must appear at least this many times to be considered
    /// for suffix array optimization.
    pub sa_fragment_min_freq: u8,
    
    /// Minimum length for compression (1-255)
    /// 
    /// Strings shorter than this length are not compressed,
    /// avoiding overhead for small strings.
    pub compression_min_length: u8,
    
    /// Compression algorithm selection
    pub compression_algorithm: CompressionAlgorithm,
    
    // Optimization flags
    /// Optimization flags controlling various performance strategies
    pub optimization_flags: OptimizationFlags,
    
    /// Best delimiter characters for fragment splitting
    /// 
    /// Characters that are considered optimal delimiters for
    /// splitting text into fragments.
    pub best_delimiters: HashSet<u8>,
    
    // Memory management
    /// Common prefix shared by all strings
    /// 
    /// If all strings share a common prefix, it can be stored
    /// separately to improve compression and performance.
    pub common_prefix: String,
    
    /// Temporary directory for large operations
    /// 
    /// Directory used for temporary files during construction
    /// of large tries. Empty string uses system default.
    pub temp_directory: String,
    
    /// Temporary file usage level
    pub temp_level: TempLevel,
    
    /// Initial memory pool size in bytes
    /// 
    /// Pre-allocated memory pool size for construction operations.
    /// Larger values reduce allocation overhead.
    pub initial_pool_size: usize,
    
    /// Load factor for hash operations (0.1-0.9)
    /// 
    /// Load factor used for internal hash tables and maps.
    /// Lower values use more memory but provide better performance.
    pub load_factor: f64,
    
    // Performance tuning
    /// Input is already sorted (skip top-level sorting)
    /// 
    /// If true, the top-level sort will be omitted, improving
    /// performance when input is known to be sorted.
    pub is_input_sorted: bool,
    
    /// Nesting scale factor (1-255)
    /// 
    /// If nestStrVec * nestScale < inputStrVec, nesting stops.
    /// Value 1 disables nesting, 255 maximizes nesting.
    pub nest_scale: u8,
    
    /// Enable queue compression to reduce temporary file size
    pub enable_queue_compression: bool,
    
    /// Use mixed core link strategy for varied data patterns
    pub use_mixed_core_link: bool,
    
    /// Speed up nested trie building with aggressive optimizations
    pub speedup_nest_trie_build: bool,
    
    // Advanced features
    /// Debug output level (0-3)
    /// 
    /// Controls the verbosity of debug output during construction.
    /// 0 = no debug output, 3 = maximum verbosity.
    pub debug_level: u8,
    
    /// Enable comprehensive statistics collection
    pub enable_statistics: bool,
    
    /// Enable detailed performance profiling
    pub enable_profiling: bool,
    
    /// Maximum depth for breadth-first search operations
    /// 
    /// Limits the depth of BFS operations to control memory usage
    /// and processing time.
    pub max_bfs_depth: u32,
    
    /// Frequency threshold for pattern caching
    /// 
    /// Patterns that appear at least this many times are cached
    /// for improved access performance.
    pub cache_frequency_threshold: u32,
    
    /// Enable adaptive optimization based on data characteristics
    pub enable_adaptive_optimization: bool,
    
    /// Number of threads for parallel construction (0 = auto)
    /// 
    /// Number of threads to use for parallel construction operations.
    /// 0 uses the number of available CPU cores.
    pub parallel_threads: u32,
    
    /// Enable memory-mapped file operations for large datasets
    pub enable_mmap_operations: bool,
    
    /// Cache size for frequently accessed nodes (bytes)
    /// 
    /// Size of the cache for frequently accessed trie nodes.
    /// Larger caches improve performance but use more memory.
    pub node_cache_size: usize,
}

impl Default for NestLoudsTrieConfig {
    fn default() -> Self {
        let mut best_delimiters = HashSet::new();
        // Common text delimiters
        best_delimiters.extend([b' ', b'\t', b'\n', b'\r', b'.', b',', b';', b':', b'!', b'?']);
        
        Self {
            // Build-time parameters
            nest_level: 3,
            max_fragment_length: 1024,
            min_fragment_length: 8,
            min_link_str_length: 4,
            
            // Compression settings
            core_str_compression_level: 6,
            sa_fragment_min_freq: 2,
            compression_min_length: 32,
            compression_algorithm: CompressionAlgorithm::default(),
            
            // Optimization flags
            optimization_flags: OptimizationFlags::default(),
            best_delimiters,
            
            // Memory management
            common_prefix: String::new(),
            temp_directory: String::new(),
            temp_level: TempLevel::default(),
            initial_pool_size: 64 * 1024 * 1024, // 64MB
            load_factor: 0.75,
            
            // Performance tuning
            is_input_sorted: false,
            nest_scale: 8,
            enable_queue_compression: false,
            use_mixed_core_link: false,
            speedup_nest_trie_build: false,
            
            // Advanced features
            debug_level: 0,
            enable_statistics: false,
            enable_profiling: false,
            max_bfs_depth: 1000,
            cache_frequency_threshold: 10,
            enable_adaptive_optimization: true,
            parallel_threads: 0, // Auto-detect
            enable_mmap_operations: true,
            node_cache_size: 16 * 1024 * 1024, // 16MB
        }
    }
}

impl Config for NestLoudsTrieConfig {
    fn validate(&self) -> Result<()> {
        let mut errors = Vec::new();
        
        // Validate nest level
        if self.nest_level == 0 || self.nest_level > 16 {
            errors.push(ValidationError::new(
                "nest_level",
                &self.nest_level.to_string(),
                "nest level must be between 1 and 16"
            ).with_suggestion("typical values: 2-4 for most use cases"));
        }
        
        // Validate fragment lengths
        if self.min_fragment_length == 0 {
            errors.push(ValidationError::new(
                "min_fragment_length",
                &self.min_fragment_length.to_string(),
                "minimum fragment length must be at least 1"
            ));
        }
        
        if self.max_fragment_length > 0 && self.max_fragment_length < self.min_fragment_length as i32 {
            errors.push(ValidationError::new(
                "max_fragment_length",
                &self.max_fragment_length.to_string(),
                "maximum fragment length must be greater than minimum fragment length"
            ));
        }
        
        // Validate compression settings
        if self.core_str_compression_level > 22 {
            errors.push(ValidationError::new(
                "core_str_compression_level",
                &self.core_str_compression_level.to_string(),
                "compression level must be between 0 and 22"
            ).with_suggestion("typical values: 1-6 for speed, 7-15 for balance, 16-22 for compression"));
        }
        
        // Validate load factor
        if self.load_factor <= 0.0 || self.load_factor >= 1.0 {
            errors.push(ValidationError::new(
                "load_factor",
                &self.load_factor.to_string(),
                "load factor must be between 0.0 and 1.0 (exclusive)"
            ).with_suggestion("typical values: 0.5-0.8"));
        }
        
        // Validate nest scale
        if self.nest_scale == 0 {
            errors.push(ValidationError::new(
                "nest_scale",
                &self.nest_scale.to_string(),
                "nest scale must be at least 1"
            ).with_suggestion("1 = disable nesting, 8 = default, 255 = maximum nesting"));
        }
        
        // Validate memory sizes
        if self.initial_pool_size == 0 {
            errors.push(ValidationError::new(
                "initial_pool_size",
                &self.initial_pool_size.to_string(),
                "initial pool size must be greater than 0"
            ).with_suggestion("typical values: 16MB-1GB depending on data size"));
        }
        
        // Return first error if any
        if !errors.is_empty() {
            return Err(ZiporaError::configuration(format!(
                "Configuration validation failed: {}",
                errors.into_iter()
                    .map(|e| e.to_string())
                    .collect::<Vec<_>>()
                    .join("; ")
            )));
        }
        
        Ok(())
    }
    
    fn from_env_with_prefix(prefix: &str) -> Result<Self> {
        let mut config = Self::default();
        
        // Build-time parameters
        config.nest_level = parse_env_var(&format!("{}TRIE_NEST_LEVEL", prefix), config.nest_level);
        config.max_fragment_length = parse_env_var(&format!("{}TRIE_MAX_FRAGMENT_LENGTH", prefix), config.max_fragment_length);
        config.min_fragment_length = parse_env_var(&format!("{}TRIE_MIN_FRAGMENT_LENGTH", prefix), config.min_fragment_length);
        config.min_link_str_length = parse_env_var(&format!("{}TRIE_MIN_LINK_STR_LENGTH", prefix), config.min_link_str_length);
        
        // Compression settings
        config.core_str_compression_level = parse_env_var(&format!("{}TRIE_COMPRESSION_LEVEL", prefix), config.core_str_compression_level);
        config.sa_fragment_min_freq = parse_env_var(&format!("{}TRIE_SA_FRAGMENT_MIN_FREQ", prefix), config.sa_fragment_min_freq);
        config.compression_min_length = parse_env_var(&format!("{}TRIE_COMPRESSION_MIN_LENGTH", prefix), config.compression_min_length);
        
        // Memory management
        if let Ok(temp_dir) = std::env::var(&format!("{}TRIE_TEMP_DIRECTORY", prefix)) {
            config.temp_directory = temp_dir;
        }
        config.initial_pool_size = parse_env_var(&format!("{}TRIE_INITIAL_POOL_SIZE", prefix), config.initial_pool_size);
        config.load_factor = parse_env_var(&format!("{}TRIE_LOAD_FACTOR", prefix), config.load_factor);
        
        // Performance tuning
        config.is_input_sorted = parse_env_bool(&format!("{}TRIE_INPUT_SORTED", prefix), config.is_input_sorted);
        config.nest_scale = parse_env_var(&format!("{}TRIE_NEST_SCALE", prefix), config.nest_scale);
        config.enable_queue_compression = parse_env_bool(&format!("{}TRIE_QUEUE_COMPRESSION", prefix), config.enable_queue_compression);
        config.use_mixed_core_link = parse_env_bool(&format!("{}TRIE_MIXED_CORE_LINK", prefix), config.use_mixed_core_link);
        config.speedup_nest_trie_build = parse_env_bool(&format!("{}TRIE_SPEEDUP_BUILD", prefix), config.speedup_nest_trie_build);
        
        // Advanced features
        config.debug_level = parse_env_var(&format!("{}TRIE_DEBUG_LEVEL", prefix), config.debug_level);
        config.enable_statistics = parse_env_bool(&format!("{}TRIE_ENABLE_STATISTICS", prefix), config.enable_statistics);
        config.enable_profiling = parse_env_bool(&format!("{}TRIE_ENABLE_PROFILING", prefix), config.enable_profiling);
        config.max_bfs_depth = parse_env_var(&format!("{}TRIE_MAX_BFS_DEPTH", prefix), config.max_bfs_depth);
        config.cache_frequency_threshold = parse_env_var(&format!("{}TRIE_CACHE_FREQ_THRESHOLD", prefix), config.cache_frequency_threshold);
        config.enable_adaptive_optimization = parse_env_bool(&format!("{}TRIE_ADAPTIVE_OPTIMIZATION", prefix), config.enable_adaptive_optimization);
        config.parallel_threads = parse_env_var(&format!("{}TRIE_PARALLEL_THREADS", prefix), config.parallel_threads);
        config.enable_mmap_operations = parse_env_bool(&format!("{}TRIE_ENABLE_MMAP", prefix), config.enable_mmap_operations);
        config.node_cache_size = parse_env_var(&format!("{}TRIE_NODE_CACHE_SIZE", prefix), config.node_cache_size);
        
        config.validate()?;
        Ok(config)
    }
    
    fn performance_preset() -> Self {
        let mut config = Self::default();
        
        // Optimize for maximum performance
        config.nest_level = 2; // Reduced nesting for speed
        config.core_str_compression_level = 3; // Fast compression
        config.optimization_flags = OptimizationFlags::SEARCH_DELIM_FORWARD
            | OptimizationFlags::ENABLE_FAST_SEARCH
            | OptimizationFlags::USE_MIXED_CORE_LINK
            | OptimizationFlags::SPEEDUP_NEST_TRIE_BUILD
            | OptimizationFlags::ENABLE_CACHE_OPTIMIZATION
            | OptimizationFlags::ENABLE_SIMD_ACCELERATION
            | OptimizationFlags::ENABLE_PARALLEL_CONSTRUCTION
            | OptimizationFlags::USE_HUGEPAGES;
        
        config.initial_pool_size = 256 * 1024 * 1024; // 256MB
        config.load_factor = 0.6; // Lower load factor for speed
        config.speedup_nest_trie_build = true;
        config.use_mixed_core_link = true;
        config.parallel_threads = 0; // Use all available cores
        config.node_cache_size = 64 * 1024 * 1024; // 64MB cache
        
        config
    }
    
    fn memory_preset() -> Self {
        let mut config = Self::default();
        
        // Optimize for minimal memory usage
        config.nest_level = 5; // Higher nesting for compression
        config.core_str_compression_level = 15; // High compression
        config.optimization_flags = OptimizationFlags::SEARCH_DELIM_FORWARD
            | OptimizationFlags::CUT_FRAG_ON_PUNCT
            | OptimizationFlags::USE_DAWG_STR_POOL
            | OptimizationFlags::ENABLE_QUEUE_COMPRESSION;
        
        config.temp_level = TempLevel::SaveNestStrPool; // Use temp files aggressively
        config.initial_pool_size = 16 * 1024 * 1024; // 16MB
        config.load_factor = 0.9; // High load factor for memory efficiency
        config.enable_queue_compression = true;
        config.compression_min_length = 8; // Compress smaller strings
        config.node_cache_size = 4 * 1024 * 1024; // 4MB cache
        config.parallel_threads = 1; // Single-threaded to save memory
        
        config
    }
    
    fn realtime_preset() -> Self {
        let mut config = Self::default();
        
        // Optimize for low latency and predictable performance
        config.nest_level = 2; // Reduced nesting for predictability
        config.core_str_compression_level = 1; // Minimal compression
        config.optimization_flags = OptimizationFlags::ENABLE_FAST_SEARCH
            | OptimizationFlags::ENABLE_CACHE_OPTIMIZATION
            | OptimizationFlags::ENABLE_SIMD_ACCELERATION;
        
        config.temp_level = TempLevel::Smart; // Minimize temp file usage
        config.initial_pool_size = 128 * 1024 * 1024; // 128MB pre-allocated
        config.load_factor = 0.7; // Balanced load factor
        config.enable_queue_compression = false; // Avoid compression overhead
        config.speedup_nest_trie_build = true;
        config.max_bfs_depth = 100; // Limit depth for predictability
        config.parallel_threads = 2; // Limited parallelism for predictability
        config.node_cache_size = 32 * 1024 * 1024; // 32MB cache
        
        config
    }
    
    fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let serialized = serde_json::to_string_pretty(self)
            .map_err(|e| ZiporaError::configuration(format!("Failed to serialize config: {}", e)))?;
        
        std::fs::write(path, serialized)
            .map_err(|e| ZiporaError::configuration(format!("Failed to write config file: {}", e)))?;
        
        Ok(())
    }
    
    fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| ZiporaError::configuration(format!("Failed to read config file: {}", e)))?;
        
        let config: Self = serde_json::from_str(&content)
            .map_err(|e| ZiporaError::configuration(format!("Failed to parse config file: {}", e)))?;
        
        config.validate()?;
        Ok(config)
    }
}

impl NestLoudsTrieConfig {
    /// Create a new configuration builder.
    /// 
    /// # Returns
    /// 
    /// A new configuration builder instance.
    pub fn builder() -> NestLoudsTrieConfigBuilder {
        NestLoudsTrieConfigBuilder::new()
    }
    
    /// Set the best delimiters from a string of characters.
    /// 
    /// # Arguments
    /// 
    /// * `delimiters` - String containing delimiter characters
    pub fn set_best_delimiters(&mut self, delimiters: &str) {
        self.best_delimiters.clear();
        self.best_delimiters.extend(delimiters.bytes());
    }
    
    /// Enable or disable a specific optimization flag.
    /// 
    /// # Arguments
    /// 
    /// * `flag` - The optimization flag to modify
    /// * `enabled` - Whether to enable or disable the flag
    pub fn set_optimization_flag(&mut self, flag: OptimizationFlags, enabled: bool) {
        if enabled {
            self.optimization_flags |= flag;
        } else {
            self.optimization_flags &= !flag;
        }
    }
    
    /// Check if a specific optimization flag is enabled.
    /// 
    /// # Arguments
    /// 
    /// * `flag` - The optimization flag to check
    /// 
    /// # Returns
    /// 
    /// `true` if the flag is enabled, `false` otherwise.
    pub fn has_optimization_flag(&self, flag: OptimizationFlags) -> bool {
        self.optimization_flags.contains(flag)
    }
}

/// Builder for constructing Nested LOUDS Trie configurations.
/// 
/// This builder provides a fluent API for constructing complex configurations
/// with validation and sensible defaults.
#[derive(Debug, Clone)]
pub struct NestLoudsTrieConfigBuilder {
    config: NestLoudsTrieConfig,
}

impl NestLoudsTrieConfigBuilder {
    /// Create a new configuration builder with default values.
    pub fn new() -> Self {
        Self {
            config: NestLoudsTrieConfig::default(),
        }
    }
    
    /// Set the nesting level.
    /// 
    /// # Arguments
    /// 
    /// * `level` - Nesting level (1-16)
    pub fn nest_level(mut self, level: u8) -> Self {
        self.config.nest_level = level;
        self
    }
    
    /// Set the maximum fragment length.
    /// 
    /// # Arguments
    /// 
    /// * `length` - Maximum fragment length (-1 for no limit)
    pub fn max_fragment_length(mut self, length: i32) -> Self {
        self.config.max_fragment_length = length;
        self
    }
    
    /// Set the minimum fragment length.
    /// 
    /// # Arguments
    /// 
    /// * `length` - Minimum fragment length
    pub fn min_fragment_length(mut self, length: u32) -> Self {
        self.config.min_fragment_length = length;
        self
    }
    
    /// Set the compression level.
    /// 
    /// # Arguments
    /// 
    /// * `level` - Compression level (0-22)
    pub fn compression_level(mut self, level: u8) -> Self {
        self.config.core_str_compression_level = level;
        self
    }
    
    /// Set the compression algorithm.
    /// 
    /// # Arguments
    /// 
    /// * `algorithm` - Compression algorithm to use
    pub fn compression_algorithm(mut self, algorithm: CompressionAlgorithm) -> Self {
        self.config.compression_algorithm = algorithm;
        self
    }
    
    /// Enable queue compression.
    /// 
    /// # Arguments
    /// 
    /// * `enabled` - Whether to enable queue compression
    pub fn enable_queue_compression(mut self, enabled: bool) -> Self {
        self.config.enable_queue_compression = enabled;
        self
    }
    
    /// Set temporary directory.
    /// 
    /// # Arguments
    /// 
    /// * `dir` - Temporary directory path
    pub fn temp_directory<S: Into<String>>(mut self, dir: S) -> Self {
        self.config.temp_directory = dir.into();
        self
    }
    
    /// Set the initial memory pool size.
    /// 
    /// # Arguments
    /// 
    /// * `size` - Pool size in bytes
    pub fn initial_pool_size(mut self, size: usize) -> Self {
        self.config.initial_pool_size = size;
        self
    }
    
    /// Enable or disable statistics collection.
    /// 
    /// # Arguments
    /// 
    /// * `enabled` - Whether to enable statistics
    pub fn enable_statistics(mut self, enabled: bool) -> Self {
        self.config.enable_statistics = enabled;
        self
    }
    
    /// Enable or disable profiling.
    /// 
    /// # Arguments
    /// 
    /// * `enabled` - Whether to enable profiling
    pub fn enable_profiling(mut self, enabled: bool) -> Self {
        self.config.enable_profiling = enabled;
        self
    }
    
    /// Set optimization flags.
    /// 
    /// # Arguments
    /// 
    /// * `flags` - Optimization flags to set
    pub fn optimization_flags(mut self, flags: OptimizationFlags) -> Self {
        self.config.optimization_flags = flags;
        self
    }
    
    /// Set the number of parallel threads.
    /// 
    /// # Arguments
    /// 
    /// * `threads` - Number of threads (0 for auto-detect)
    pub fn parallel_threads(mut self, threads: u32) -> Self {
        self.config.parallel_threads = threads;
        self
    }
    
    /// Build the configuration.
    /// 
    /// # Returns
    /// 
    /// The constructed configuration, or an error if validation fails.
    pub fn build(self) -> Result<NestLoudsTrieConfig> {
        self.config.validate()?;
        Ok(self.config)
    }
}

impl Default for NestLoudsTrieConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// Add serde support for JSON serialization
use serde::{Deserialize, Serialize};

impl Serialize for NestLoudsTrieConfig {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        
        let mut state = serializer.serialize_struct("NestLoudsTrieConfig", 25)?;
        state.serialize_field("nest_level", &self.nest_level)?;
        state.serialize_field("max_fragment_length", &self.max_fragment_length)?;
        state.serialize_field("min_fragment_length", &self.min_fragment_length)?;
        state.serialize_field("min_link_str_length", &self.min_link_str_length)?;
        state.serialize_field("core_str_compression_level", &self.core_str_compression_level)?;
        state.serialize_field("sa_fragment_min_freq", &self.sa_fragment_min_freq)?;
        state.serialize_field("compression_min_length", &self.compression_min_length)?;
        state.serialize_field("optimization_flags", &self.optimization_flags.bits())?;
        state.serialize_field("best_delimiters", &self.best_delimiters.iter().cloned().collect::<Vec<u8>>())?;
        state.serialize_field("common_prefix", &self.common_prefix)?;
        state.serialize_field("temp_directory", &self.temp_directory)?;
        state.serialize_field("temp_level", &(self.temp_level as u8))?;
        state.serialize_field("initial_pool_size", &self.initial_pool_size)?;
        state.serialize_field("load_factor", &self.load_factor)?;
        state.serialize_field("is_input_sorted", &self.is_input_sorted)?;
        state.serialize_field("nest_scale", &self.nest_scale)?;
        state.serialize_field("enable_queue_compression", &self.enable_queue_compression)?;
        state.serialize_field("use_mixed_core_link", &self.use_mixed_core_link)?;
        state.serialize_field("speedup_nest_trie_build", &self.speedup_nest_trie_build)?;
        state.serialize_field("debug_level", &self.debug_level)?;
        state.serialize_field("enable_statistics", &self.enable_statistics)?;
        state.serialize_field("enable_profiling", &self.enable_profiling)?;
        state.serialize_field("max_bfs_depth", &self.max_bfs_depth)?;
        state.serialize_field("cache_frequency_threshold", &self.cache_frequency_threshold)?;
        state.serialize_field("enable_adaptive_optimization", &self.enable_adaptive_optimization)?;
        state.serialize_field("parallel_threads", &self.parallel_threads)?;
        state.serialize_field("enable_mmap_operations", &self.enable_mmap_operations)?;
        state.serialize_field("node_cache_size", &self.node_cache_size)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for NestLoudsTrieConfig {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{self, MapAccess, Visitor};
        use std::fmt;
        
        struct ConfigVisitor;
        
        impl<'de> Visitor<'de> for ConfigVisitor {
            type Value = NestLoudsTrieConfig;
            
            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("NestLoudsTrieConfig struct")
            }
            
            fn visit_map<V>(self, mut map: V) -> std::result::Result<NestLoudsTrieConfig, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut config = NestLoudsTrieConfig::default();
                
                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "nest_level" => config.nest_level = map.next_value()?,
                        "max_fragment_length" => config.max_fragment_length = map.next_value()?,
                        "min_fragment_length" => config.min_fragment_length = map.next_value()?,
                        "min_link_str_length" => config.min_link_str_length = map.next_value()?,
                        "core_str_compression_level" => config.core_str_compression_level = map.next_value()?,
                        "sa_fragment_min_freq" => config.sa_fragment_min_freq = map.next_value()?,
                        "compression_min_length" => config.compression_min_length = map.next_value()?,
                        "optimization_flags" => {
                            let bits: u64 = map.next_value()?;
                            config.optimization_flags = OptimizationFlags::from_bits(bits)
                                .ok_or_else(|| de::Error::custom("Invalid optimization flags"))?;
                        },
                        "best_delimiters" => {
                            let delims: Vec<u8> = map.next_value()?;
                            config.best_delimiters = delims.into_iter().collect();
                        },
                        "common_prefix" => config.common_prefix = map.next_value()?,
                        "temp_directory" => config.temp_directory = map.next_value()?,
                        "temp_level" => {
                            let level: u8 = map.next_value()?;
                            config.temp_level = match level {
                                0 => TempLevel::Smart,
                                1 => TempLevel::BfsQueue,
                                2 => TempLevel::SwapLinkVec,
                                3 => TempLevel::LinkVecDoubleSize,
                                4 => TempLevel::SaveNestStrPool,
                                _ => return Err(de::Error::custom("Invalid temp level")),
                            };
                        },
                        "initial_pool_size" => config.initial_pool_size = map.next_value()?,
                        "load_factor" => config.load_factor = map.next_value()?,
                        "is_input_sorted" => config.is_input_sorted = map.next_value()?,
                        "nest_scale" => config.nest_scale = map.next_value()?,
                        "enable_queue_compression" => config.enable_queue_compression = map.next_value()?,
                        "use_mixed_core_link" => config.use_mixed_core_link = map.next_value()?,
                        "speedup_nest_trie_build" => config.speedup_nest_trie_build = map.next_value()?,
                        "debug_level" => config.debug_level = map.next_value()?,
                        "enable_statistics" => config.enable_statistics = map.next_value()?,
                        "enable_profiling" => config.enable_profiling = map.next_value()?,
                        "max_bfs_depth" => config.max_bfs_depth = map.next_value()?,
                        "cache_frequency_threshold" => config.cache_frequency_threshold = map.next_value()?,
                        "enable_adaptive_optimization" => config.enable_adaptive_optimization = map.next_value()?,
                        "parallel_threads" => config.parallel_threads = map.next_value()?,
                        "enable_mmap_operations" => config.enable_mmap_operations = map.next_value()?,
                        "node_cache_size" => config.node_cache_size = map.next_value()?,
                        _ => {
                            // Skip unknown fields for forward compatibility
                            map.next_value::<serde_json::Value>()?;
                        }
                    }
                }
                
                Ok(config)
            }
        }
        
        deserializer.deserialize_map(ConfigVisitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_default_config() {
        let config = NestLoudsTrieConfig::default();
        assert!(config.validate().is_ok());
    }
    
    #[test]
    fn test_builder_pattern() {
        let config = NestLoudsTrieConfig::builder()
            .nest_level(4)
            .compression_level(9)
            .enable_statistics(true)
            .build()
            .expect("Failed to build config");
        
        assert_eq!(config.nest_level, 4);
        assert_eq!(config.core_str_compression_level, 9);
        assert!(config.enable_statistics);
    }
    
    #[test]
    fn test_presets() {
        let perf_config = NestLoudsTrieConfig::performance_preset();
        assert!(perf_config.validate().is_ok());
        assert!(perf_config.has_optimization_flag(OptimizationFlags::ENABLE_SIMD_ACCELERATION));
        
        let mem_config = NestLoudsTrieConfig::memory_preset();
        assert!(mem_config.validate().is_ok());
        assert!(mem_config.enable_queue_compression);
        
        let rt_config = NestLoudsTrieConfig::realtime_preset();
        assert!(rt_config.validate().is_ok());
        assert_eq!(rt_config.core_str_compression_level, 1);
    }
    
    #[test]
    fn test_validation() {
        let mut config = NestLoudsTrieConfig::default();
        
        // Test invalid nest level
        config.nest_level = 0;
        assert!(config.validate().is_err());
        
        config.nest_level = 17;
        assert!(config.validate().is_err());
        
        // Test invalid load factor
        config = NestLoudsTrieConfig::default();
        config.load_factor = 0.0;
        assert!(config.validate().is_err());
        
        config.load_factor = 1.0;
        assert!(config.validate().is_err());
    }
    
    #[test]
    fn test_optimization_flags() {
        let mut config = NestLoudsTrieConfig::default();
        
        assert!(config.has_optimization_flag(OptimizationFlags::ENABLE_SIMD_ACCELERATION));
        
        config.set_optimization_flag(OptimizationFlags::USE_HUGEPAGES, true);
        assert!(config.has_optimization_flag(OptimizationFlags::USE_HUGEPAGES));
        
        config.set_optimization_flag(OptimizationFlags::USE_HUGEPAGES, false);
        assert!(!config.has_optimization_flag(OptimizationFlags::USE_HUGEPAGES));
    }
    
    #[test]
    fn test_serialization() {
        let config = NestLoudsTrieConfig::default();
        
        // Test JSON serialization
        let json = serde_json::to_string(&config).expect("Failed to serialize");
        let deserialized: NestLoudsTrieConfig = serde_json::from_str(&json).expect("Failed to deserialize");
        
        // Verify key fields match
        assert_eq!(config.nest_level, deserialized.nest_level);
        assert_eq!(config.max_fragment_length, deserialized.max_fragment_length);
        assert_eq!(config.optimization_flags, deserialized.optimization_flags);
    }
}