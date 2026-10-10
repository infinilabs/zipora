//! # Enhanced CPU Feature Detection System
//!
//! Comprehensive runtime CPU feature detection with adaptive algorithm selection.
//! Inspired by production-grade feature detection systems and high-performance
//! libraries with additional Rust-specific optimizations.
//!
//! # Architecture
//!
//! This module implements sophisticated hardware detection following Phase 1.1 of the
//! systematic SIMD implementation plan:
//! - Enhanced x86_64 feature detection (SSE4.1/4.2, AVX, AVX2, AVX-512, BMI1/2, etc.)
//! - Comprehensive ARM64 feature detection (NEON, CRC32, Crypto, SVE)
//! - Cache characteristics detection with accurate sizing
//! - Build system integration with feature flags
//! - Runtime optimal algorithm selection

use std::collections::HashMap;
use std::sync::OnceLock;

/// Comprehensive CPU feature flags for runtime detection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CpuFeature {
    // x86_64 Basic SSE/AVX features
    SSE2,
    SSE3,
    SSSE3,
    SSE4_1,
    SSE4_2,
    AVX,
    AVX2,

    // x86_64 BMI and specialized instructions
    BMI1,
    BMI2,
    POPCNT,
    LZCNT,
    TZCNT,
    PREFETCHW,

    // x86_64 Crypto and specialized features
    PCLMULQDQ,
    AES,
    RDRAND,
    RDSEED,

    // x86_64 AVX-512 feature family
    AVX512F,         // Foundation
    AVX512DQ,        // Doubleword and Quadword Instructions
    AVX512CD,        // Conflict Detection Instructions
    AVX512BW,        // Byte and Word Instructions
    AVX512VL,        // Vector Length Extensions
    AVX512VPOPCNTDQ, // Vector Population Count D/Q
    AVX512VBMI,      // Vector Bit Manipulation Instructions
    AVX512IFMA,      // Integer Fused Multiply-Add

    // ARM64 features
    NEON,
    CRC32,
    AesArm,
    SHA1,
    SHA2,
    SHA3,
    Crypto,
    SVE,  // Scalable Vector Extension
    SVE2, // Scalable Vector Extension 2

    // Universal features
    UnalignedAccess,

    // Memory and cache features
    Prefetch,
    ClflushOpt,
    Clwb,
}

/// Primary CPU feature set with comprehensive performance characteristics
///
/// This is the main CPU features interface, providing comprehensive hardware detection
/// and optimization strategy selection for SIMD operations.
#[derive(Debug, Clone)]
pub struct CpuFeatures {
    // x86_64 SSE/AVX features
    pub has_sse2: bool,
    pub has_sse3: bool,
    pub has_ssse3: bool,
    pub has_sse41: bool,
    pub has_sse42: bool,
    pub has_avx: bool,
    pub has_avx2: bool,
    pub has_avx512f: bool,
    pub has_avx512dq: bool,
    pub has_avx512cd: bool,
    pub has_avx512bw: bool,
    pub has_avx512vl: bool,
    pub has_avx512vpopcntdq: bool,
    pub has_avx512vbmi: bool,
    pub has_avx512ifma: bool,

    // x86_64 BMI and specialized instructions
    pub has_bmi1: bool,
    pub has_bmi2: bool,
    pub has_popcnt: bool,
    pub has_lzcnt: bool,
    pub has_tzcnt: bool,
    pub has_prefetchw: bool,

    // x86_64 Crypto and RNG features
    pub has_pclmulqdq: bool,
    pub has_aes: bool,
    pub has_rdrand: bool,
    pub has_rdseed: bool,

    // ARM64 features
    pub has_neon: bool,
    pub has_crc32: bool,
    pub has_aes_arm: bool,
    pub has_sha1: bool,
    pub has_sha2: bool,
    pub has_sha3: bool,
    pub has_crypto: bool,
    pub has_sve: bool,
    pub has_sve2: bool,

    // Universal and memory/cache features
    pub has_unaligned_access: bool,
    pub has_prefetch: bool,
    pub has_clflushopt: bool,
    pub has_clwb: bool,

    // Cache characteristics
    pub l1_cache_size: usize,
    pub l2_cache_size: usize,
    pub l3_cache_size: usize,
    pub cache_line_size: usize,

    // System characteristics
    pub logical_cores: usize,
    pub physical_cores: usize,
    pub vendor: String,
    pub model: String,

    // Performance optimization tier
    pub optimization_tier: u8,
    pub simd_tier: u8,
}

/// Legacy CPU feature set for backward compatibility
#[derive(Debug, Clone)]
pub struct CpuFeatureSet {
    /// Available CPU features
    pub features: HashMap<CpuFeature, bool>,
    /// CPU vendor (Intel, AMD, ARM, etc.)
    pub vendor: String,
    /// CPU model name
    pub model: String,
    /// Number of logical cores
    pub logical_cores: usize,
    /// Number of physical cores
    pub physical_cores: usize,
    /// Cache line size (typically 64 bytes)
    pub cache_line_size: usize,
    /// L1 cache size (data)
    pub l1_cache_size: usize,
    /// L2 cache size
    pub l2_cache_size: usize,
    /// L3 cache size
    pub l3_cache_size: usize,
    /// SIMD optimization tier (0=scalar, 1=basic, 2=advanced, 3=cutting-edge)
    pub simd_tier: u8,
}

impl CpuFeatures {
    /// Create new AdvancedCpuFeatures with all features disabled
    pub fn new() -> Self {
        Self {
            // x86_64 SSE/AVX features
            has_sse2: false,
            has_sse3: false,
            has_ssse3: false,
            has_sse41: false,
            has_sse42: false,
            has_avx: false,
            has_avx2: false,
            has_avx512f: false,
            has_avx512dq: false,
            has_avx512cd: false,
            has_avx512bw: false,
            has_avx512vl: false,
            has_avx512vpopcntdq: false,
            has_avx512vbmi: false,
            has_avx512ifma: false,

            // x86_64 BMI and specialized instructions
            has_bmi1: false,
            has_bmi2: false,
            has_popcnt: false,
            has_lzcnt: false,
            has_tzcnt: false,
            has_prefetchw: false,

            // x86_64 Crypto and RNG features
            has_pclmulqdq: false,
            has_aes: false,
            has_rdrand: false,
            has_rdseed: false,

            // ARM64 features
            has_neon: false,
            has_crc32: false,
            has_aes_arm: false,
            has_sha1: false,
            has_sha2: false,
            has_sha3: false,
            has_crypto: false,
            has_sve: false,
            has_sve2: false,

            // Universal and memory/cache features
            has_unaligned_access: false,
            has_prefetch: false,
            has_clflushopt: false,
            has_clwb: false,

            // Cache characteristics (default values)
            l1_cache_size: 32 * 1024,       // 32KB
            l2_cache_size: 256 * 1024,      // 256KB
            l3_cache_size: 8 * 1024 * 1024, // 8MB
            cache_line_size: 64,            // 64 bytes

            // System characteristics
            logical_cores: 1,
            physical_cores: 1,
            vendor: String::new(),
            model: String::new(),

            // Performance optimization tier
            optimization_tier: 0,
            simd_tier: 0,
        }
    }

    /// Detect and configure SIMD optimization strategy
    pub fn detect_and_configure_simd(&mut self) {
        // Determine optimization tier based on available features
        self.optimization_tier = self.calculate_optimization_tier();
        self.simd_tier = self.calculate_simd_tier();
    }

    /// Calculate optimization tier
    fn calculate_optimization_tier(&self) -> u8 {
        if self.has_avx512f && self.has_avx512bw && self.has_avx512vpopcntdq {
            5 // Tier 5: AVX-512 with popcount
        } else if self.has_avx2 && self.has_bmi2 {
            4 // Tier 4: AVX2 + BMI2 
        } else if self.has_bmi2 {
            3 // Tier 3: BMI2
        } else if self.has_popcnt || self.has_neon {
            2 // Tier 2: POPCNT or NEON
        } else {
            1 // Tier 1: Scalar fallback
        }
    }

    /// Calculate SIMD tier
    fn calculate_simd_tier(&self) -> u8 {
        if self.has_avx512f {
            4 // Cutting-edge: AVX-512
        } else if self.has_bmi2 && self.has_avx2 {
            3 // Advanced: BMI2 + AVX2
        } else if self.has_avx2 {
            2 // Intermediate: AVX2
        } else if self.has_popcnt || self.has_neon {
            1 // Basic: POPCNT or NEON
        } else {
            0 // Scalar fallback
        }
    }

    /// Get optimal SIMD implementation for rank/select operations
    pub fn optimal_rank_select_variant(&self) -> &'static str {
        if self.has_avx512f && self.has_avx512bw && self.has_avx512vpopcntdq {
            "avx512_popcnt"
        } else if self.has_bmi2 && self.has_avx2 {
            "bmi2_avx2"
        } else if self.has_avx2 {
            "avx2"
        } else if self.has_bmi2 {
            "bmi2"
        } else if self.has_popcnt {
            "popcnt"
        } else if self.has_neon {
            "neon"
        } else {
            "scalar"
        }
    }

    /// Get optimal string search implementation
    pub fn optimal_string_search_variant(&self) -> &'static str {
        if self.has_sse42 {
            "sse42_pcmpestri"
        } else if self.has_avx2 {
            "avx2_search"
        } else if self.has_neon {
            "neon_search"
        } else {
            "scalar"
        }
    }

    /// Get optimal memory copy implementation
    pub fn optimal_memcpy_variant(&self) -> &'static str {
        if self.has_avx512f {
            "avx512_memcpy"
        } else if self.has_avx2 {
            "avx2_memcpy"
        } else if self.has_neon {
            "neon_memcpy"
        } else {
            "scalar_memcpy"
        }
    }

    /// Get optimal Base64 implementation
    pub fn optimal_base64_variant(&self) -> &'static str {
        if self.has_avx2 {
            "avx2"
        } else if self.has_sse42 {
            "sse42"
        } else if self.has_neon {
            "neon"
        } else {
            "scalar"
        }
    }

    /// Get recommended chunk size for bulk operations
    pub fn recommended_chunk_size(&self) -> usize {
        match self.optimization_tier {
            5 => 64 * 1024, // AVX-512: 64KB chunks
            4 => 32 * 1024, // AVX2+BMI2: 32KB chunks
            3 => 16 * 1024, // BMI2: 16KB chunks
            2 => 8 * 1024,  // Basic SIMD: 8KB chunks
            _ => 4 * 1024,  // Scalar: 4KB chunks
        }
    }

    /// Check if prefetching should be used
    pub fn should_use_prefetch(&self) -> bool {
        self.has_prefetchw || self.optimization_tier >= 3
    }

    /// Check if hardware has optimal memory access patterns
    pub fn has_optimal_memory_access(&self) -> bool {
        // Check for features that indicate good memory performance
        self.cache_line_size == 64
            && (cfg!(target_arch = "x86_64") || cfg!(target_arch = "aarch64"))
    }

    /// Get recommended memory alignment for SIMD operations
    pub fn recommended_alignment(&self) -> usize {
        if self.has_avx512f {
            64 // 512-bit alignment
        } else if self.has_avx2 {
            32 // 256-bit alignment
        } else if self.has_neon {
            16 // 128-bit alignment
        } else {
            8 // 64-bit alignment
        }
    }

    /// Check if a specific CPU feature is available
    pub fn has_feature(&self, feature: CpuFeature) -> bool {
        match feature {
            // x86_64 Basic SSE/AVX features
            CpuFeature::SSE2 => self.has_sse2,
            CpuFeature::SSE3 => self.has_sse3,
            CpuFeature::SSSE3 => self.has_ssse3,
            CpuFeature::SSE4_1 => self.has_sse41,
            CpuFeature::SSE4_2 => self.has_sse42,
            CpuFeature::AVX => self.has_avx,
            CpuFeature::AVX2 => self.has_avx2,

            // x86_64 BMI and specialized instructions
            CpuFeature::BMI1 => self.has_bmi1,
            CpuFeature::BMI2 => self.has_bmi2,
            CpuFeature::POPCNT => self.has_popcnt,
            CpuFeature::LZCNT => self.has_lzcnt,
            CpuFeature::TZCNT => self.has_tzcnt,
            CpuFeature::PREFETCHW => self.has_prefetchw,

            // x86_64 Crypto and specialized features
            CpuFeature::PCLMULQDQ => self.has_pclmulqdq,
            CpuFeature::AES => self.has_aes,
            CpuFeature::RDRAND => self.has_rdrand,
            CpuFeature::RDSEED => self.has_rdseed,

            // x86_64 AVX-512 feature family
            CpuFeature::AVX512F => self.has_avx512f,
            CpuFeature::AVX512DQ => self.has_avx512dq,
            CpuFeature::AVX512CD => self.has_avx512cd,
            CpuFeature::AVX512BW => self.has_avx512bw,
            CpuFeature::AVX512VL => self.has_avx512vl,
            CpuFeature::AVX512VPOPCNTDQ => self.has_avx512vpopcntdq,
            CpuFeature::AVX512VBMI => self.has_avx512vbmi,
            CpuFeature::AVX512IFMA => self.has_avx512ifma,

            // ARM64 features
            CpuFeature::NEON => self.has_neon,
            CpuFeature::CRC32 => self.has_crc32,
            CpuFeature::AesArm => self.has_aes_arm,
            CpuFeature::SHA1 => self.has_sha1,
            CpuFeature::SHA2 => self.has_sha2,
            CpuFeature::SHA3 => self.has_sha3,
            CpuFeature::Crypto => self.has_crypto,
            CpuFeature::SVE => self.has_sve,
            CpuFeature::SVE2 => self.has_sve2,

            // Universal features
            CpuFeature::UnalignedAccess => self.has_unaligned_access,

            // Memory and cache features
            CpuFeature::Prefetch => self.has_prefetch,
            CpuFeature::ClflushOpt => self.has_clflushopt,
            CpuFeature::Clwb => self.has_clwb,
        }
    }
}

impl Default for CpuFeatures {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuFeatureSet {
    /// Check if a specific feature is available
    pub fn has_feature(&self, feature: CpuFeature) -> bool {
        self.features.get(&feature).copied().unwrap_or(false)
    }

    /// Get the optimal SIMD instruction set for rank/select operations
    pub fn optimal_rank_select_variant(&self) -> &'static str {
        if self.has_feature(CpuFeature::AVX512F) && self.has_feature(CpuFeature::AVX512BW) {
            "avx512"
        } else if self.has_feature(CpuFeature::BMI2) && self.has_feature(CpuFeature::AVX2) {
            "bmi2_avx2"
        } else if self.has_feature(CpuFeature::AVX2) {
            "avx2"
        } else if self.has_feature(CpuFeature::POPCNT) {
            "popcnt"
        } else if self.has_feature(CpuFeature::NEON) {
            "neon"
        } else {
            "scalar"
        }
    }

    /// Get the optimal Base64 implementation
    pub fn optimal_base64_variant(&self) -> &'static str {
        if self.has_feature(CpuFeature::AVX2) {
            "avx2"
        } else if self.has_feature(CpuFeature::SSE4_2) {
            "sse42"
        } else if self.has_feature(CpuFeature::NEON) {
            "neon"
        } else {
            "scalar"
        }
    }

    /// Get the SIMD optimization tier
    pub fn get_simd_tier(&self) -> u8 {
        if self.has_feature(CpuFeature::AVX512F) {
            4 // Cutting-edge: AVX-512
        } else if self.has_feature(CpuFeature::BMI2) && self.has_feature(CpuFeature::AVX2) {
            3 // Advanced: BMI2 + AVX2
        } else if self.has_feature(CpuFeature::AVX2) {
            2 // Intermediate: AVX2
        } else if self.has_feature(CpuFeature::POPCNT) || self.has_feature(CpuFeature::NEON) {
            1 // Basic: POPCNT or NEON
        } else {
            0 // Scalar fallback
        }
    }

    /// Check if hardware has optimal memory access patterns
    pub fn has_optimal_memory_access(&self) -> bool {
        // Check for features that indicate good memory performance
        self.cache_line_size == 64
            && (self.has_feature(CpuFeature::UnalignedAccess) || cfg!(target_arch = "x86_64"))
    }

    /// Get recommended buffer alignment for SIMD operations
    pub fn recommended_alignment(&self) -> usize {
        if self.has_feature(CpuFeature::AVX512F) {
            64 // 512-bit alignment
        } else if self.has_feature(CpuFeature::AVX2) {
            32 // 256-bit alignment
        } else if self.has_feature(CpuFeature::SSE2) || self.has_feature(CpuFeature::NEON) {
            16 // 128-bit alignment
        } else {
            8 // 64-bit alignment
        }
    }
}

/// Runtime CPU feature detection interface
pub struct RuntimeCpuFeatures;

impl Default for RuntimeCpuFeatures {
    fn default() -> Self {
        Self::new()
    }
}

impl RuntimeCpuFeatures {
    /// Create a new runtime feature detector
    pub fn new() -> Self {
        Self
    }

    /// Detect all available CPU features
    pub fn detect_features(&self) -> CpuFeatures {
        let mut features = CpuFeatures::new();
        features.has_unaligned_access = cfg!(any(target_arch = "x86_64", target_arch = "aarch64"));

        // Detect CPU info
        let (vendor, model) = self.get_cpu_info();
        features.vendor = vendor;
        features.model = model;

        // Detect core counts
        let (logical_cores, physical_cores) = self.get_core_count();
        features.logical_cores = logical_cores;
        features.physical_cores = physical_cores;

        // Detect cache info with enhanced detection
        let cache_info = self.get_enhanced_cache_info();
        features.cache_line_size = cache_info.0;
        features.l1_cache_size = cache_info.1;
        features.l2_cache_size = cache_info.2;
        features.l3_cache_size = cache_info.3;

        // Platform-specific feature detection
        #[cfg(target_arch = "x86_64")]
        {
            self.detect_x86_features(&mut features);
        }

        #[cfg(target_arch = "aarch64")]
        {
            self.detect_arm_features(&mut features);
        }

        // Configure SIMD optimization strategy
        features.detect_and_configure_simd();

        features
    }

    /// Enhanced x86_64 feature detection for CpuFeatures
    ///
    /// Every flag here gates a real `#[target_feature]` kernel, so detection
    /// has to answer "may this process execute the instruction?", not "does
    /// the silicon know the opcode?". Reading the CPUID leaves directly — what
    /// `raw_cpuid`'s `has_avx()`/`has_avx2()`/`has_avx512f()` do — only answers
    /// the second question: touching YMM/ZMM registers additionally requires
    /// the OS to have enabled extended state saving (CR4.OSXSAVE plus the XCR0
    /// SSE/AVX/opmask/ZMM bits). Where it has not, the first AVX instruction
    /// raises #UD and the process dies with SIGILL.
    ///
    /// `is_x86_feature_detected!` performs that XCR0 handshake, caches the
    /// answer, and is already what the other detection sites in this crate
    /// use, so delegate rather than keeping a second, laxer implementation.
    #[cfg(target_arch = "x86_64")]
    fn detect_x86_features(&self, features: &mut CpuFeatures) {
        #[cfg(miri)]
        let _ = features;

        #[cfg(not(miri))]
        {
            // Basic SSE/AVX features
            features.has_sse2 = is_x86_feature_detected!("sse2");
            features.has_sse3 = is_x86_feature_detected!("sse3");
            features.has_ssse3 = is_x86_feature_detected!("ssse3");
            features.has_sse41 = is_x86_feature_detected!("sse4.1");
            features.has_sse42 = is_x86_feature_detected!("sse4.2");
            features.has_avx = is_x86_feature_detected!("avx");
            features.has_avx2 = is_x86_feature_detected!("avx2");

            // BMI and bit manipulation instructions
            features.has_bmi1 = is_x86_feature_detected!("bmi1");
            features.has_bmi2 = is_x86_feature_detected!("bmi2");
            features.has_popcnt = is_x86_feature_detected!("popcnt");
            features.has_lzcnt = is_x86_feature_detected!("lzcnt");
            features.has_tzcnt = features.has_bmi1;

            // Crypto and hardware RNG features
            features.has_pclmulqdq = is_x86_feature_detected!("pclmulqdq");
            features.has_aes = is_x86_feature_detected!("aes");
            features.has_rdrand = is_x86_feature_detected!("rdrand");
            features.has_rdseed = is_x86_feature_detected!("rdseed");

            // AVX-512 feature family
            features.has_avx512f = is_x86_feature_detected!("avx512f");
            features.has_avx512dq = is_x86_feature_detected!("avx512dq");
            features.has_avx512cd = is_x86_feature_detected!("avx512cd");
            features.has_avx512bw = is_x86_feature_detected!("avx512bw");
            features.has_avx512vl = is_x86_feature_detected!("avx512vl");
            features.has_avx512vpopcntdq = is_x86_feature_detected!("avx512vpopcntdq");
            features.has_avx512vbmi = is_x86_feature_detected!("avx512vbmi");
            features.has_avx512ifma = is_x86_feature_detected!("avx512ifma");

            // Memory and cache hints
            features.has_prefetch = is_x86_feature_detected!("sse");
            let cpuid = raw_cpuid::CpuId::new();
            if let Some(ext) = cpuid.get_extended_feature_info() {
                features.has_clflushopt = ext.has_clflushopt();
                features.has_clwb = ext.has_clwb();
            }
            if let Some(ext_proc) = cpuid.get_extended_processor_and_feature_identifiers() {
                features.has_prefetchw = ext_proc.has_prefetchw();
            }
        }
    }

    /// Enhanced ARM64 feature detection for CpuFeatures
    #[cfg(target_arch = "aarch64")]
    fn detect_arm_features(&self, features: &mut CpuFeatures) {
        #[cfg(miri)]
        let _ = features;

        #[cfg(not(miri))]
        {
            features.has_neon = std::arch::is_aarch64_feature_detected!("neon");
            features.has_prefetch = true;
            features.has_crc32 = std::arch::is_aarch64_feature_detected!("crc");
            features.has_aes_arm = std::arch::is_aarch64_feature_detected!("aes");
            features.has_sha2 = std::arch::is_aarch64_feature_detected!("sha2");
            features.has_sha3 = std::arch::is_aarch64_feature_detected!("sha3");
            // In std::arch::is_aarch64_feature_detected!, "sha2" covers FEAT_SHA1 + FEAT_SHA256.
            features.has_sha1 = features.has_sha2;
            features.has_crypto = features.has_aes_arm && features.has_sha2;
            features.has_sve = std::arch::is_aarch64_feature_detected!("sve");
            features.has_sve2 = std::arch::is_aarch64_feature_detected!("sve2");
        }
    }

    /// Enhanced cache detection with accurate sizing
    fn get_enhanced_cache_info(&self) -> (usize, usize, usize, usize) {
        let mut cache_line_size = 64; // Default assumption
        let mut l1_size = 32 * 1024; // 32KB default
        let mut l2_size = 256 * 1024; // 256KB default  
        let mut l3_size = 8 * 1024 * 1024; // 8MB default

        #[cfg(target_arch = "x86_64")]
        {
            #[cfg(not(miri))]
            {
                let cpuid = raw_cpuid::CpuId::new();

                if let Some(cache_params) = cpuid.get_cache_parameters() {
                    for cache in cache_params {
                        if !matches!(
                            cache.cache_type(),
                            raw_cpuid::CacheType::Data | raw_cpuid::CacheType::Unified
                        ) {
                            continue;
                        }
                        let line_size = cache.coherency_line_size();
                        if line_size > 0 {
                            cache_line_size = line_size;
                        }

                        let cache_size = cache
                            .associativity()
                            .saturating_mul(line_size)
                            .saturating_mul(cache.physical_line_partitions())
                            .saturating_mul(cache.sets());
                        if cache_size > 0 {
                            match cache.level() {
                                1 => l1_size = cache_size,
                                2 => l2_size = cache_size,
                                3 => l3_size = cache_size,
                                _ => {}
                            }
                        }
                    }
                }
            }
            #[cfg(miri)]
            let _ = (
                &mut cache_line_size,
                &mut l1_size,
                &mut l2_size,
                &mut l3_size,
            );
        }

        #[cfg(target_arch = "aarch64")]
        {
            if let Ok(entries) = std::fs::read_dir("/sys/devices/system/cpu/cpu0/cache") {
                for entry in entries.flatten() {
                    if let Ok(coherency_str) =
                        std::fs::read_to_string(entry.path().join("coherency_line_size"))
                        && let Ok(coherency) = coherency_str.trim().parse::<usize>()
                        && coherency > 0
                    {
                        cache_line_size = coherency;
                    }

                    if let (Ok(level_str), Ok(size_str)) = (
                        std::fs::read_to_string(entry.path().join("level")),
                        std::fs::read_to_string(entry.path().join("size")),
                    ) && let Ok(level) = level_str.trim().parse::<u8>()
                    {
                        let trimmed = size_str.trim();
                        let parsed_bytes = if let Some(kb) = trimmed.strip_suffix('K') {
                            kb.parse::<usize>().ok().map(|v| v * 1024)
                        } else if let Some(mb) = trimmed.strip_suffix('M') {
                            mb.parse::<usize>().ok().map(|v| v * 1024 * 1024)
                        } else {
                            trimmed.parse::<usize>().ok()
                        };
                        if let Some(bytes) = parsed_bytes.filter(|&b| b > 0) {
                            match level {
                                1 => l1_size = bytes,
                                2 => l2_size = bytes,
                                3 => l3_size = bytes,
                                _ => {}
                            }
                        }
                    }
                }
            }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let _ = (
            &mut cache_line_size,
            &mut l1_size,
            &mut l2_size,
            &mut l3_size,
        );

        (cache_line_size, l1_size, l2_size, l3_size)
    }

    /// Get CPU vendor and model information
    fn get_cpu_info(&self) -> (String, String) {
        #[cfg(miri)]
        {
            ("Miri Emulator".to_string(), "Virtual CPU".to_string())
        }

        #[cfg(all(target_arch = "x86_64", not(miri)))]
        {
            let cpuid = raw_cpuid::CpuId::new();
            let vendor = cpuid
                .get_vendor_info()
                .map(|v| v.as_str().to_string())
                .unwrap_or_else(|| "Unknown".to_string());
            let model = cpuid
                .get_processor_brand_string()
                .map(|b| b.as_str().to_string())
                .unwrap_or_else(|| "Unknown".to_string());
            (vendor, model)
        }

        #[cfg(all(target_arch = "aarch64", not(miri)))]
        {
            if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
                let mut vendor = "ARM".to_string();
                let mut model = "Unknown".to_string();

                for line in cpuinfo.lines() {
                    if line.starts_with("CPU implementer") {
                        if line.contains("0x41") {
                            vendor = "ARM".to_string();
                        } else if line.contains("0x51") {
                            vendor = "Qualcomm".to_string();
                        }
                    } else if line.starts_with("model name")
                        && let Some(name) = line.split(':').nth(1)
                    {
                        model = name.trim().to_string();
                    }
                }
                return (vendor, model);
            }
            ("ARM".to_string(), "Unknown".to_string())
        }

        #[cfg(all(not(any(target_arch = "x86_64", target_arch = "aarch64")), not(miri)))]
        {
            ("Unknown".to_string(), "Unknown".to_string())
        }
    }

    /// Compute physical core count from logical core count and hyperthreading flag,
    /// clamping to at least 1 physical core even on 1-vCPU hyperthreaded VMs.
    pub(crate) fn compute_physical_cores(logical_cores: usize, has_htt: bool) -> usize {
        if has_htt {
            (logical_cores / 2).max(1)
        } else {
            logical_cores.max(1)
        }
    }

    /// Get logical and physical core counts
    fn get_core_count(&self) -> (usize, usize) {
        let logical_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .max(1);

        #[cfg(all(target_arch = "x86_64", not(miri)))]
        {
            let cpuid = raw_cpuid::CpuId::new();
            if let Some(feature_info) = cpuid.get_feature_info() {
                return (
                    logical_cores,
                    Self::compute_physical_cores(logical_cores, feature_info.has_htt()),
                );
            }
        }

        (
            logical_cores,
            Self::compute_physical_cores(logical_cores, false),
        )
    }
}

// Global CPU feature detection
static CPU_FEATURES: OnceLock<CpuFeatures> = OnceLock::new();

/// Get the global CPU features (detected once on first call)
///
/// This is the main API for accessing CPU features with comprehensive
/// detection capabilities following the SIMD implementation plan Phase 1.1
pub fn get_cpu_features() -> &'static CpuFeatures {
    CPU_FEATURES.get_or_init(|| RuntimeCpuFeatures::new().detect_features())
}

/// Check if a specific CPU feature is available
pub fn has_cpu_feature(feature: CpuFeature) -> bool {
    get_cpu_features().has_feature(feature)
}

/// Hardware-accelerated SIMD detection and configuration
///
/// This function implements the core functionality from Phase 1.1 of the SIMD plan:
/// Runtime CPU feature detection with build system integration patterns.
pub fn detect_and_configure_simd() -> &'static CpuFeatures {
    get_cpu_features()
}

/// Get optimal SIMD strategy for the current hardware
///
/// Returns the optimal implementation variant for different operation types
/// based on comprehensive hardware feature detection.
pub fn get_optimal_simd_strategy() -> SimdStrategy {
    let features = get_cpu_features();

    SimdStrategy {
        rank_select_variant: features.optimal_rank_select_variant(),
        string_search_variant: features.optimal_string_search_variant(),
        memcpy_variant: features.optimal_memcpy_variant(),
        chunk_size: features.recommended_chunk_size(),
        alignment: features.recommended_alignment(),
        use_prefetch: features.should_use_prefetch(),
        optimization_tier: features.optimization_tier,
        simd_tier: features.simd_tier,
    }
}

/// SIMD optimization strategy result
#[derive(Debug, Clone)]
pub struct SimdStrategy {
    pub rank_select_variant: &'static str,
    pub string_search_variant: &'static str,
    pub memcpy_variant: &'static str,
    pub chunk_size: usize,
    pub alignment: usize,
    pub use_prefetch: bool,
    pub optimization_tier: u8,
    pub simd_tier: u8,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_feature_detection() {
        let features = get_cpu_features();

        // Basic sanity checks
        assert!(features.logical_cores > 0);
        assert!(features.physical_cores > 0);
        assert!(features.cache_line_size > 0);
        assert!(!features.vendor.is_empty());

        // SIMD tier should be reasonable
        assert!(features.simd_tier <= 4);

        println!("CPU: {} {}", features.vendor, features.model);
        println!(
            "Cores: {} logical, {} physical",
            features.logical_cores, features.physical_cores
        );
        println!("Cache line size: {} bytes", features.cache_line_size);
        println!("SIMD tier: {}", features.simd_tier);
        println!(
            "Optimal rank/select: {}",
            features.optimal_rank_select_variant()
        );
        println!("Optimal base64: {}", features.optimal_base64_variant());
    }

    #[test]
    fn test_advanced_cpu_feature_detection() {
        let advanced_features = get_cpu_features();

        // Basic sanity checks
        assert!(advanced_features.logical_cores > 0);
        assert!(advanced_features.physical_cores > 0);
        assert!(advanced_features.cache_line_size > 0);
        assert!(!advanced_features.vendor.is_empty());

        // Optimization tiers should be reasonable
        assert!(advanced_features.optimization_tier <= 5);
        assert!(advanced_features.simd_tier <= 4);

        println!("=== Advanced CPU Features ===");
        println!(
            "CPU: {} {}",
            advanced_features.vendor, advanced_features.model
        );
        println!(
            "Cores: {} logical, {} physical",
            advanced_features.logical_cores, advanced_features.physical_cores
        );
        println!(
            "Cache: L1={}, L2={}, L3={}, Line={}",
            advanced_features.l1_cache_size,
            advanced_features.l2_cache_size,
            advanced_features.l3_cache_size,
            advanced_features.cache_line_size
        );

        println!("x86_64 Features:");
        println!(
            "  SSE4.1: {}, SSE4.2: {}",
            advanced_features.has_sse41, advanced_features.has_sse42
        );
        println!(
            "  AVX: {}, AVX2: {}",
            advanced_features.has_avx, advanced_features.has_avx2
        );
        println!(
            "  AVX-512F: {}, AVX-512VL: {}, AVX-512BW: {}",
            advanced_features.has_avx512f,
            advanced_features.has_avx512vl,
            advanced_features.has_avx512bw
        );
        println!(
            "  BMI1: {}, BMI2: {}",
            advanced_features.has_bmi1, advanced_features.has_bmi2
        );
        println!(
            "  POPCNT: {}, LZCNT: {}, TZCNT: {}",
            advanced_features.has_popcnt, advanced_features.has_lzcnt, advanced_features.has_tzcnt
        );

        println!("ARM64 Features:");
        println!(
            "  NEON: {}, CRC32: {}",
            advanced_features.has_neon, advanced_features.has_crc32
        );
        println!(
            "  Crypto: {}, SVE: {}, SVE2: {}",
            advanced_features.has_crypto, advanced_features.has_sve, advanced_features.has_sve2
        );

        println!(
            "Optimization: Tier={}, SIMD={}",
            advanced_features.optimization_tier, advanced_features.simd_tier
        );
        println!(
            "Optimal rank/select: {}",
            advanced_features.optimal_rank_select_variant()
        );
        println!(
            "Optimal string search: {}",
            advanced_features.optimal_string_search_variant()
        );
        println!(
            "Optimal memcpy: {}",
            advanced_features.optimal_memcpy_variant()
        );
        println!(
            "Recommended chunk size: {} bytes",
            advanced_features.recommended_chunk_size()
        );
        println!(
            "Recommended alignment: {} bytes",
            advanced_features.recommended_alignment()
        );
        println!("Use prefetch: {}", advanced_features.should_use_prefetch());
    }

    #[test]
    fn test_simd_strategy() {
        let strategy = get_optimal_simd_strategy();

        // Strategy should have valid values
        assert!(!strategy.rank_select_variant.is_empty());
        assert!(!strategy.string_search_variant.is_empty());
        assert!(!strategy.memcpy_variant.is_empty());
        assert!(strategy.chunk_size >= 4096);
        assert!(strategy.chunk_size <= 65536);
        assert!(strategy.alignment >= 8);
        assert!(strategy.alignment <= 64);
        assert!(strategy.alignment.is_power_of_two());
        assert!(strategy.optimization_tier <= 5);
        assert!(strategy.simd_tier <= 4);

        println!("=== SIMD Strategy ===");
        println!("Rank/Select: {}", strategy.rank_select_variant);
        println!("String Search: {}", strategy.string_search_variant);
        println!("Memory Copy: {}", strategy.memcpy_variant);
        println!("Chunk Size: {} bytes", strategy.chunk_size);
        println!("Alignment: {} bytes", strategy.alignment);
        println!("Use Prefetch: {}", strategy.use_prefetch);
        println!("Optimization Tier: {}", strategy.optimization_tier);
        println!("SIMD Tier: {}", strategy.simd_tier);
    }

    #[test]
    fn test_detect_and_configure_simd() {
        let features = detect_and_configure_simd();

        // Should return the same instance as get_cpu_features
        let global_features = get_cpu_features();
        assert_eq!(
            features.optimization_tier,
            global_features.optimization_tier
        );
        assert_eq!(features.simd_tier, global_features.simd_tier);

        // Optimization tier should be calculated
        assert!(features.optimization_tier <= 5);
        assert!(features.simd_tier <= 4);

        println!("SIMD Configuration Complete:");
        println!("  Optimization Tier: {}", features.optimization_tier);
        println!("  SIMD Tier: {}", features.simd_tier);
    }

    #[test]
    fn test_has_cpu_feature() {
        // Test the convenience function
        let _has_popcnt = has_cpu_feature(CpuFeature::POPCNT);
        let _has_avx2 = has_cpu_feature(CpuFeature::AVX2);
        // Should not panic
    }

    #[test]
    fn test_feature_set_methods() {
        let features = get_cpu_features();

        // Test optimization variant selection
        let rank_select_variant = features.optimal_rank_select_variant();
        assert!(
            [
                "scalar",
                "popcnt",
                "bmi2",
                "avx2",
                "bmi2_avx2",
                "avx512",
                "avx512_popcnt",
                "neon",
            ]
            .contains(&rank_select_variant)
        );

        let base64_variant = features.optimal_base64_variant();
        assert!(["scalar", "sse42", "avx2", "neon"].contains(&base64_variant));

        // Test alignment recommendation
        let alignment = features.recommended_alignment();
        assert!((8..=64).contains(&alignment));
        assert!(alignment.is_power_of_two());
    }

    #[test]
    fn test_memory_access_patterns() {
        let features = get_cpu_features();

        // Test memory access optimization detection
        let _has_optimal = features.has_optimal_memory_access();

        // Cache line size should be reasonable (typically 64 bytes)
        assert!(features.cache_line_size >= 32 && features.cache_line_size <= 128);
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    fn test_reported_features_are_actually_usable() {
        // Regression: detection read the CPUID leaves directly, which reports
        // what the silicon implements, not what the OS has enabled. Where
        // extended state saving is off (some hypervisors, kernels booted with
        // `noxsave`), executing the reported AVX kernels traps with #UD.
        //
        // On a machine whose OS does enable AVX this assertion cannot fail
        // whichever way detection is implemented; it exists to pin the
        // delegation to `is_x86_feature_detected!`, which is the only
        // implementation of the XCR0 handshake this crate should have.
        let features = get_cpu_features();

        for (name, reported, usable) in [
            (
                "sse4.1",
                features.has_sse41,
                is_x86_feature_detected!("sse4.1"),
            ),
            (
                "sse4.2",
                features.has_sse42,
                is_x86_feature_detected!("sse4.2"),
            ),
            ("avx", features.has_avx, is_x86_feature_detected!("avx")),
            ("avx2", features.has_avx2, is_x86_feature_detected!("avx2")),
            ("bmi1", features.has_bmi1, is_x86_feature_detected!("bmi1")),
            ("bmi2", features.has_bmi2, is_x86_feature_detected!("bmi2")),
            (
                "popcnt",
                features.has_popcnt,
                is_x86_feature_detected!("popcnt"),
            ),
            (
                "lzcnt",
                features.has_lzcnt,
                is_x86_feature_detected!("lzcnt"),
            ),
            (
                "avx512f",
                features.has_avx512f,
                is_x86_feature_detected!("avx512f"),
            ),
            (
                "avx512vl",
                features.has_avx512vl,
                is_x86_feature_detected!("avx512vl"),
            ),
            (
                "avx512bw",
                features.has_avx512bw,
                is_x86_feature_detected!("avx512bw"),
            ),
        ] {
            assert_eq!(
                reported, usable,
                "{name}: CpuFeatures disagrees with the OS-aware detection"
            );
        }
    }

    #[test]
    fn test_c12_2_cpu_features_exhaustive_detection_and_single_core_htt() {
        // 1. CpuFeatures::new() is documented as "all features disabled".
        // Previously has_feature(SSE2 | SSE3 | SSSE3) unconditionally returned true.
        let empty = CpuFeatures::new();
        assert!(
            !empty.has_feature(CpuFeature::SSE2),
            "CpuFeatures::new() must have SSE2 disabled"
        );
        assert!(
            !empty.has_feature(CpuFeature::SSE3),
            "CpuFeatures::new() must have SSE3 disabled"
        );
        assert!(
            !empty.has_feature(CpuFeature::SSSE3),
            "CpuFeatures::new() must have SSSE3 disabled"
        );
        assert!(
            !empty.has_feature(CpuFeature::UnalignedAccess),
            "CpuFeatures::new() must have UnalignedAccess disabled"
        );

        // 2. Detected physical_cores must always be >= 1.
        let detected = get_cpu_features();
        assert!(detected.logical_cores >= 1);
        assert!(detected.physical_cores >= 1);

        // 3. Under Miri, detect_x86_features is a no-op so all x86 features on get_cpu_features()
        // must be false; on real x86_64 hardware, every CpuFeature variant must match
        // std::is_x86_feature_detected! (previously AES, PCLMULQDQ, RDRAND, RDSEED,
        // AVX512DQ, AVX512CD, AVX512VPOPCNTDQ, AVX512VBMI, AVX512IFMA always returned false).
        #[cfg(all(target_arch = "x86_64", not(miri)))]
        {
            for (name, feat, expected) in [
                ("sse2", CpuFeature::SSE2, is_x86_feature_detected!("sse2")),
                ("sse3", CpuFeature::SSE3, is_x86_feature_detected!("sse3")),
                (
                    "ssse3",
                    CpuFeature::SSSE3,
                    is_x86_feature_detected!("ssse3"),
                ),
                (
                    "sse4.1",
                    CpuFeature::SSE4_1,
                    is_x86_feature_detected!("sse4.1"),
                ),
                (
                    "sse4.2",
                    CpuFeature::SSE4_2,
                    is_x86_feature_detected!("sse4.2"),
                ),
                ("avx", CpuFeature::AVX, is_x86_feature_detected!("avx")),
                ("avx2", CpuFeature::AVX2, is_x86_feature_detected!("avx2")),
                ("bmi1", CpuFeature::BMI1, is_x86_feature_detected!("bmi1")),
                ("bmi2", CpuFeature::BMI2, is_x86_feature_detected!("bmi2")),
                (
                    "popcnt",
                    CpuFeature::POPCNT,
                    is_x86_feature_detected!("popcnt"),
                ),
                (
                    "lzcnt",
                    CpuFeature::LZCNT,
                    is_x86_feature_detected!("lzcnt"),
                ),
                (
                    "pclmulqdq",
                    CpuFeature::PCLMULQDQ,
                    is_x86_feature_detected!("pclmulqdq"),
                ),
                ("aes", CpuFeature::AES, is_x86_feature_detected!("aes")),
                (
                    "rdrand",
                    CpuFeature::RDRAND,
                    is_x86_feature_detected!("rdrand"),
                ),
                (
                    "rdseed",
                    CpuFeature::RDSEED,
                    is_x86_feature_detected!("rdseed"),
                ),
                (
                    "avx512f",
                    CpuFeature::AVX512F,
                    is_x86_feature_detected!("avx512f"),
                ),
                (
                    "avx512dq",
                    CpuFeature::AVX512DQ,
                    is_x86_feature_detected!("avx512dq"),
                ),
                (
                    "avx512cd",
                    CpuFeature::AVX512CD,
                    is_x86_feature_detected!("avx512cd"),
                ),
                (
                    "avx512bw",
                    CpuFeature::AVX512BW,
                    is_x86_feature_detected!("avx512bw"),
                ),
                (
                    "avx512vl",
                    CpuFeature::AVX512VL,
                    is_x86_feature_detected!("avx512vl"),
                ),
                (
                    "avx512vpopcntdq",
                    CpuFeature::AVX512VPOPCNTDQ,
                    is_x86_feature_detected!("avx512vpopcntdq"),
                ),
                (
                    "avx512vbmi",
                    CpuFeature::AVX512VBMI,
                    is_x86_feature_detected!("avx512vbmi"),
                ),
                (
                    "avx512ifma",
                    CpuFeature::AVX512IFMA,
                    is_x86_feature_detected!("avx512ifma"),
                ),
            ] {
                assert_eq!(
                    has_cpu_feature(feat),
                    expected,
                    "has_cpu_feature({name}) disagreed with std::is_x86_feature_detected!"
                );
            }
        }
    }

    #[test]
    fn test_s33_r3_compute_physical_cores_clamps_single_core_htt() {
        assert_eq!(RuntimeCpuFeatures::compute_physical_cores(1, true), 1);
        assert_eq!(RuntimeCpuFeatures::compute_physical_cores(1, false), 1);
        assert_eq!(RuntimeCpuFeatures::compute_physical_cores(2, true), 1);
        assert_eq!(RuntimeCpuFeatures::compute_physical_cores(2, false), 2);
        assert_eq!(RuntimeCpuFeatures::compute_physical_cores(8, true), 4);
        assert_eq!(RuntimeCpuFeatures::compute_physical_cores(0, true), 1);
    }
}
