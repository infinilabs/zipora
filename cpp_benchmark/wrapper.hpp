#pragma once

/**
 * Enhanced C++ Wrapper for zipora library benchmarking
 * 
 * This header provides comprehensive C-compatible interfaces to the original 
 * topling-zip C++ library for detailed performance comparison benchmarks.
 * 
 * Includes memory tracking, cache analysis, and comprehensive benchmark utilities.
 */

#include <cstddef>
#include <cstdint>

extern "C" {

// ============================================================================
// Vector Operations (valvec wrapper)
// ============================================================================

void* cpp_valvec_create();
void* cpp_valvec_create_with_capacity(size_t capacity);
void cpp_valvec_destroy(void* vec);
void cpp_valvec_push(void* vec, int32_t value);
size_t cpp_valvec_size(void* vec);
size_t cpp_valvec_capacity(void* vec);
int32_t cpp_valvec_get(void* vec, size_t index);
void cpp_valvec_reserve(void* vec, size_t capacity);
void cpp_valvec_clear(void* vec);
void cpp_valvec_shrink_to_fit(void* vec);

// Batch operations for performance testing
void cpp_valvec_push_batch(void* vec, const int32_t* values, size_t count);
void cpp_valvec_get_batch(void* vec, size_t start, size_t count, int32_t* output);

// ============================================================================
// String Operations (fstring wrapper)
// ============================================================================

void* cpp_fstring_create(const uint8_t* data, size_t len);
void* cpp_fstring_create_from_cstr(const char* cstr);
void cpp_fstring_destroy(void* fstr);
uint64_t cpp_fstring_hash(void* fstr);
int64_t cpp_fstring_find(void* fstr, const uint8_t* needle, size_t needle_len);
void* cpp_fstring_substring(void* fstr, size_t start, size_t len);
size_t cpp_fstring_length(void* fstr);
const uint8_t* cpp_fstring_data(void* fstr);
int cpp_fstring_compare(void* fstr1, void* fstr2);
int cpp_fstring_starts_with(void* fstr, const uint8_t* prefix, size_t prefix_len);
int cpp_fstring_ends_with(void* fstr, const uint8_t* suffix, size_t suffix_len);

// String concatenation and manipulation
void* cpp_fstring_concat(void* fstr1, void* fstr2);
void* cpp_fstring_repeat(void* fstr, size_t times);

// ============================================================================
// Hash Map Operations (if available)
// ============================================================================

void* cpp_hashmap_create();
void cpp_hashmap_destroy(void* map);
int cpp_hashmap_insert(void* map, const char* key, int32_t value);
int cpp_hashmap_get(void* map, const char* key, int32_t* value);
int cpp_hashmap_remove(void* map, const char* key);
size_t cpp_hashmap_size(void* map);
void cpp_hashmap_clear(void* map);

// Batch operations
int cpp_hashmap_insert_batch(void* map, const char** keys, const int32_t* values, size_t count);
int cpp_hashmap_get_batch(void* map, const char** keys, int32_t* values, size_t count);

// ============================================================================
// Bit Vector and Rank-Select Operations
// ============================================================================

void* cpp_bitvector_create();
void cpp_bitvector_destroy(void* bv);
void cpp_bitvector_push(void* bv, int bit);
int cpp_bitvector_get(void* bv, size_t pos);
size_t cpp_bitvector_size(void* bv);
void cpp_bitvector_push_batch(void* bv, const int* bits, size_t count);

void* cpp_rank_select_create(const uint64_t* bits, size_t bit_count);
void* cpp_rank_select_from_bitvector(void* bv);
void cpp_rank_select_destroy(void* rs);
size_t cpp_rank_select_rank1(void* rs, size_t pos);
size_t cpp_rank_select_select1(void* rs, size_t k);
size_t cpp_rank_select_rank0(void* rs, size_t pos);
size_t cpp_rank_select_select0(void* rs, size_t k);

// ============================================================================
// Memory Management and Tracking
// ============================================================================

// Basic memory tracking
uint64_t cpp_get_memory_usage();
uint64_t cpp_get_allocation_count();
uint64_t cpp_get_deallocation_count();
uint64_t cpp_get_peak_memory_usage();
void cpp_reset_counters();

// Detailed memory statistics
typedef struct {
    uint64_t total_allocated;
    uint64_t total_deallocated;
    uint64_t current_usage;
    uint64_t peak_usage;
    uint64_t allocation_count;
    uint64_t deallocation_count;
    double average_allocation_size;
    double fragmentation_ratio;
} CppMemoryStats;

void cpp_get_memory_stats(CppMemoryStats* stats);

// Memory pool operations (if available)
void* cpp_memory_pool_create(size_t block_size, size_t initial_blocks);
void cpp_memory_pool_destroy(void* pool);
void* cpp_memory_pool_alloc(void* pool);
void cpp_memory_pool_free(void* pool, void* ptr);
size_t cpp_memory_pool_allocated_blocks(void* pool);
size_t cpp_memory_pool_free_blocks(void* pool);

// ============================================================================
// Performance Measurement and Benchmarking
// ============================================================================

// Cache and system analysis
void cpp_warmup_caches();
void cpp_flush_caches();
double cpp_measure_cache_miss_rate(void* data, size_t size, size_t iterations);
double cpp_measure_memory_bandwidth(size_t size, size_t iterations);

// Allocation performance
double cpp_measure_allocation_speed(size_t count, size_t size);
double cpp_measure_deallocation_speed(void** ptrs, size_t count);
double cpp_measure_reallocation_speed(size_t initial_size, size_t final_size, size_t iterations);

// String operation performance
double cpp_measure_hash_speed(const uint8_t* data, size_t len, size_t iterations);
double cpp_measure_string_find_speed(const uint8_t* text, size_t text_len, 
                                     const uint8_t* pattern, size_t pattern_len, 
                                     size_t iterations);
double cpp_measure_string_compare_speed(const uint8_t* str1, size_t len1,
                                        const uint8_t* str2, size_t len2,
                                        size_t iterations);

// Vector operation performance
double cpp_measure_vector_push_speed(size_t count, size_t iterations);
double cpp_measure_vector_access_speed(void* vec, size_t size, size_t iterations);
double cpp_measure_vector_iteration_speed(void* vec, size_t iterations);

// Hash map operation performance
double cpp_measure_hashmap_insert_speed(size_t count, size_t iterations);
double cpp_measure_hashmap_lookup_speed(void* map, const char** keys, size_t count, size_t iterations);

// Succinct data structure performance
double cpp_measure_bitvector_construction_speed(size_t bit_count, size_t iterations);
double cpp_measure_rank_select_construction_speed(void* bv, size_t iterations);
double cpp_measure_rank_query_speed(void* rs, size_t query_count, size_t iterations);
double cpp_measure_select_query_speed(void* rs, size_t query_count, size_t iterations);

// ============================================================================
// System Information and Hardware Detection
// ============================================================================

typedef struct {
    size_t l1_cache_size;
    size_t l2_cache_size;
    size_t l3_cache_size;
    size_t cache_line_size;
    size_t page_size;
    size_t physical_memory;
    int cpu_cores;
    int logical_cores;
    char cpu_vendor[64];
    char cpu_model[128];
} CppSystemInfo;

void cpp_get_system_info(CppSystemInfo* info);
int cpp_has_avx2();
int cpp_has_sse42();
int cpp_has_bmi2();

// ============================================================================
// Advanced Benchmarking Utilities
// ============================================================================

// Timing utilities with high precision
typedef struct {
    uint64_t start_cycles;
    uint64_t end_cycles;
    double start_time;
    double end_time;
} CppTimer;

void cpp_timer_start(CppTimer* timer);
void cpp_timer_stop(CppTimer* timer);
double cpp_timer_elapsed_seconds(const CppTimer* timer);
uint64_t cpp_timer_elapsed_cycles(const CppTimer* timer);

// Statistical measurement helpers
typedef struct {
    double min_time;
    double max_time;
    double mean_time;
    double median_time;
    double std_deviation;
    size_t sample_count;
} CppBenchmarkStats;

void cpp_run_benchmark_with_stats(void (*benchmark_func)(void*), void* data, 
                                  size_t iterations, CppBenchmarkStats* stats);

// Memory access pattern analysis
double cpp_measure_sequential_access_speed(void* data, size_t size, size_t iterations);
double cpp_measure_random_access_speed(void* data, size_t size, size_t iterations);
double cpp_measure_strided_access_speed(void* data, size_t size, size_t stride, size_t iterations);

// ============================================================================
// Comprehensive Benchmark Suite
// ============================================================================

typedef struct {
    // Vector performance
    double vector_push_throughput;
    double vector_access_throughput;
    double vector_memory_efficiency;
    
    // String performance
    double string_hash_throughput;
    double string_find_throughput;
    double string_memory_efficiency;
    
    // Hash map performance
    double hashmap_insert_throughput;
    double hashmap_lookup_throughput;
    double hashmap_memory_efficiency;
    
    // Succinct structures performance
    double bitvector_construction_throughput;
    double rank_query_throughput;
    double select_query_throughput;
    
    // Memory system performance
    double allocation_throughput;
    double cache_efficiency;
    double memory_bandwidth;
    
    // Overall score (geometric mean of normalized metrics)
    double overall_score;
} CppPerformanceSummary;

void cpp_run_comprehensive_benchmark(CppPerformanceSummary* summary);
void cpp_compare_with_baseline(const CppPerformanceSummary* current, 
                               const CppPerformanceSummary* baseline,
                               double* improvement_factors);

}