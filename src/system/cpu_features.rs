//! # CPU Feature Detection
//!
//! Comprehensive runtime CPU feature detection with adaptive algorithm selection.
//! Inspired by production-grade feature detection systems with additional Rust-specific optimizations.

use std::sync::OnceLock;
use std::collections::HashMap;

/// CPU feature flags for runtime detection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CpuFeature {
    // x86_64 features
    SSE2,
    SSE3,
    SSSE3,
    SSE4_1,
    SSE4_2,
    AVX,
    AVX2,
    BMI1,
    BMI2,
    POPCNT,
    LZCNT,
    PCLMULQDQ,
    AES,
    AVX512F,
    AVX512DQ,
    AVX512CD,
    AVX512BW,
    AVX512VL,
    // ARM features
    NEON,
    CRC32,
    AesArm,
    SHA1,
    SHA2,
    // Universal features
    UnalignedAccess,
}

/// Comprehensive CPU feature set with performance characteristics
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
            4  // Cutting-edge: AVX-512
        } else if self.has_feature(CpuFeature::BMI2) && self.has_feature(CpuFeature::AVX2) {
            3  // Advanced: BMI2 + AVX2
        } else if self.has_feature(CpuFeature::AVX2) {
            2  // Intermediate: AVX2
        } else if self.has_feature(CpuFeature::POPCNT) || self.has_feature(CpuFeature::NEON) {
            1  // Basic: POPCNT or NEON
        } else {
            0  // Scalar fallback
        }
    }

    /// Check if hardware has optimal memory access patterns
    pub fn has_optimal_memory_access(&self) -> bool {
        // Check for features that indicate good memory performance
        self.cache_line_size == 64 && 
        (self.has_feature(CpuFeature::UnalignedAccess) || cfg!(target_arch = "x86_64"))
    }

    /// Get recommended buffer alignment for SIMD operations
    pub fn recommended_alignment(&self) -> usize {
        if self.has_feature(CpuFeature::AVX512F) {
            64  // 512-bit alignment
        } else if self.has_feature(CpuFeature::AVX2) {
            32  // 256-bit alignment
        } else if self.has_feature(CpuFeature::SSE2) || self.has_feature(CpuFeature::NEON) {
            16  // 128-bit alignment
        } else {
            8   // 64-bit alignment
        }
    }
}

/// Runtime CPU feature detection interface
pub struct RuntimeCpuFeatures;

impl RuntimeCpuFeatures {
    /// Create a new runtime feature detector
    pub fn new() -> Self {
        Self
    }

    /// Detect all available CPU features
    pub fn detect_features(&self) -> CpuFeatureSet {
        let mut features = HashMap::new();

        // x86_64 feature detection
        #[cfg(target_arch = "x86_64")]
        {
            self.detect_x86_features(&mut features);
        }

        // ARM feature detection
        #[cfg(target_arch = "aarch64")]
        {
            self.detect_arm_features(&mut features);
        }

        // Fallback for other architectures
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            // Basic features available on most architectures
            features.insert(CpuFeature::UnalignedAccess, true);
        }

        let (vendor, model) = self.get_cpu_info();
        let (logical_cores, physical_cores) = self.get_core_count();
        let cache_info = self.get_cache_info();

        let feature_set = CpuFeatureSet {
            features,
            vendor,
            model,
            logical_cores,
            physical_cores,
            cache_line_size: cache_info.0,
            l1_cache_size: cache_info.1,
            l2_cache_size: cache_info.2,
            l3_cache_size: cache_info.3,
            simd_tier: 0, // Will be calculated
        };

        // Calculate SIMD tier
        let simd_tier = feature_set.get_simd_tier();
        CpuFeatureSet {
            simd_tier,
            ..feature_set
        }
    }

    /// Detect x86_64 specific features using cpuid
    #[cfg(target_arch = "x86_64")]
    fn detect_x86_features(&self, features: &mut HashMap<CpuFeature, bool>) {
        // Use raw_cpuid for comprehensive detection
        let cpuid = raw_cpuid::CpuId::new();

        // Basic features
        if let Some(feature_info) = cpuid.get_feature_info() {
            features.insert(CpuFeature::SSE2, feature_info.has_sse2());
            features.insert(CpuFeature::SSE3, feature_info.has_sse3());
            features.insert(CpuFeature::SSSE3, feature_info.has_ssse3());
            features.insert(CpuFeature::SSE4_1, feature_info.has_sse41());
            features.insert(CpuFeature::SSE4_2, feature_info.has_sse42());
            features.insert(CpuFeature::AVX, feature_info.has_avx());
            features.insert(CpuFeature::POPCNT, feature_info.has_popcnt());
            features.insert(CpuFeature::AES, feature_info.has_aesni());
            features.insert(CpuFeature::PCLMULQDQ, feature_info.has_pclmulqdq());
        }

        // Extended features
        if let Some(extended_features) = cpuid.get_extended_feature_info() {
            features.insert(CpuFeature::AVX2, extended_features.has_avx2());
            features.insert(CpuFeature::BMI1, extended_features.has_bmi1());
            features.insert(CpuFeature::BMI2, extended_features.has_bmi2());
            
            // AVX-512 features
            features.insert(CpuFeature::AVX512F, extended_features.has_avx512f());
            features.insert(CpuFeature::AVX512DQ, extended_features.has_avx512dq());
            features.insert(CpuFeature::AVX512CD, extended_features.has_avx512cd());
            features.insert(CpuFeature::AVX512BW, extended_features.has_avx512bw());
            features.insert(CpuFeature::AVX512VL, extended_features.has_avx512vl());
        }

        // Extended processor info
        if let Some(extended_info) = cpuid.get_extended_processor_and_feature_identifiers() {
            features.insert(CpuFeature::LZCNT, extended_info.has_lzcnt());
        }

        // Always available on x86_64
        features.insert(CpuFeature::UnalignedAccess, true);
    }

    /// Detect ARM specific features
    #[cfg(target_arch = "aarch64")]
    fn detect_arm_features(&self, features: &mut HashMap<CpuFeature, bool>) {
        // ARM feature detection is more limited in userspace
        // Most AArch64 systems have these features
        features.insert(CpuFeature::NEON, true);
        features.insert(CpuFeature::UnalignedAccess, true);
        
        // Try to detect additional features through /proc/cpuinfo if available
        if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
            let cpuinfo_lower = cpuinfo.to_lowercase();
            features.insert(CpuFeature::CRC32, cpuinfo_lower.contains("crc32"));
            features.insert(CpuFeature::AesArm, cpuinfo_lower.contains("aes"));
            features.insert(CpuFeature::SHA1, cpuinfo_lower.contains("sha1"));
            features.insert(CpuFeature::SHA2, cpuinfo_lower.contains("sha2"));
        }
    }

    /// Get CPU vendor and model information
    fn get_cpu_info(&self) -> (String, String) {
        #[cfg(target_arch = "x86_64")]
        {
            let cpuid = raw_cpuid::CpuId::new();
            let vendor = cpuid.get_vendor_info()
                .map(|v| v.as_str().to_string())
                .unwrap_or_else(|| "Unknown".to_string());
            let model = cpuid.get_processor_brand_string()
                .map(|b| b.as_str().to_string())
                .unwrap_or_else(|| "Unknown".to_string());
            (vendor, model)
        }

        #[cfg(target_arch = "aarch64")]
        {
            // Try to get ARM CPU info from /proc/cpuinfo
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
                    } else if line.starts_with("model name") {
                        if let Some(name) = line.split(':').nth(1) {
                            model = name.trim().to_string();
                        }
                    }
                }
                return (vendor, model);
            }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            ("Unknown".to_string(), "Unknown".to_string())
        }
    }

    /// Get logical and physical core counts
    fn get_core_count(&self) -> (usize, usize) {
        let logical_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        // Try to determine physical cores (this is approximate)
        let physical_cores = logical_cores; // Default assumption

        #[cfg(target_arch = "x86_64")]
        {
            let cpuid = raw_cpuid::CpuId::new();
            if let Some(feature_info) = cpuid.get_feature_info() {
                if feature_info.has_htt() {
                    // Hyperthreading is enabled, so physical cores = logical / 2
                    return (logical_cores, logical_cores / 2);
                }
            }
        }

        (logical_cores, physical_cores)
    }

    /// Get cache information (cache_line_size, l1_size, l2_size, l3_size)
    fn get_cache_info(&self) -> (usize, usize, usize, usize) {
        let mut cache_line_size = 64; // Default assumption
        let mut l1_size = 32 * 1024; // 32KB default
        let mut l2_size = 256 * 1024; // 256KB default  
        let mut l3_size = 8 * 1024 * 1024; // 8MB default

        #[cfg(target_arch = "x86_64")]
        {
            let cpuid = raw_cpuid::CpuId::new();
            
            // Get cache line size
            if let Some(cache_params) = cpuid.get_cache_parameters() {
                for cache in cache_params {
                    cache_line_size = cache.coherency_line_size() as usize;
                    break; // Use first cache entry
                }
            }

            // Try to get cache sizes
            // Note: Cache size detection is complex and varies by CPU
            // For now, we use reasonable defaults and detect cache line size
        }

        (cache_line_size, l1_size, l2_size, l3_size)
    }
}

// Global CPU feature detection
static CPU_FEATURES: OnceLock<CpuFeatureSet> = OnceLock::new();

/// Get the global CPU feature set (detected once on first call)
pub fn get_cpu_features() -> &'static CpuFeatureSet {
    CPU_FEATURES.get_or_init(|| {
        RuntimeCpuFeatures::new().detect_features()
    })
}

/// Check if a specific CPU feature is available
pub fn has_cpu_feature(feature: CpuFeature) -> bool {
    get_cpu_features().has_feature(feature)
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
        println!("Cores: {} logical, {} physical", features.logical_cores, features.physical_cores);
        println!("Cache line size: {} bytes", features.cache_line_size);
        println!("SIMD tier: {}", features.simd_tier);
        println!("Optimal rank/select: {}", features.optimal_rank_select_variant());
        println!("Optimal base64: {}", features.optimal_base64_variant());
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
        assert!(["scalar", "popcnt", "avx2", "bmi2_avx2", "avx512", "neon"].contains(&rank_select_variant));
        
        let base64_variant = features.optimal_base64_variant();
        assert!(["scalar", "sse42", "avx2", "neon"].contains(&base64_variant));
        
        // Test alignment recommendation
        let alignment = features.recommended_alignment();
        assert!(alignment >= 8 && alignment <= 64);
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
}