/**
 * Enhanced C++ Implementation - Comprehensive Benchmark Functions
 * 
 * This file contains the complete implementation of all enhanced wrapper functions
 * for comprehensive C++ vs Rust performance comparison.
 */

#include "wrapper.hpp"
#include <chrono>
#include <memory>
#include <atomic>
#include <vector>
#include <unordered_map>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <numeric>
#include <cmath>
#include <fstream>
#include <sstream>
#include <thread>

// System-specific headers for hardware detection and cache control
#ifdef __linux__
#include <sys/sysinfo.h>
#include <unistd.h>
#include <cpuid.h>
#endif

#ifdef _WIN32
#include <windows.h>
#include <intrin.h>
#endif

#ifdef __APPLE__
#include <sys/sysctl.h>
#include <sys/types.h>
#endif

// Enhanced memory tracking globals
extern std::atomic<uint64_t> g_memory_usage;
extern std::atomic<uint64_t> g_allocation_count;
extern std::atomic<uint64_t> g_deallocation_count;
extern std::atomic<uint64_t> g_peak_memory_usage;
extern std::atomic<uint64_t> g_total_allocated;
extern std::atomic<uint64_t> g_total_deallocated;

extern "C" {

// ============================================================================
// Enhanced String Operations Implementation
// ============================================================================

void* cpp_fstring_create_from_cstr(const char* cstr) {
    if (!cstr) return nullptr;
    size_t len = strlen(cstr);
    return cpp_fstring_create(reinterpret_cast<const uint8_t*>(cstr), len);
}

int cpp_fstring_compare(void* fstr1, void* fstr2) {
    if (!fstr1 || !fstr2) return fstr1 ? 1 : (fstr2 ? -1 : 0);
    
    auto* s1 = static_cast<fstring*>(fstr1);
    auto* s2 = static_cast<fstring*>(fstr2);
    
#ifdef HAVE_TOPLING_ZIP
    return s1->compare(*s2);
#else
    // Stub implementation
    if (s1->size() != s2->size()) {
        return s1->size() < s2->size() ? -1 : 1;
    }
    return memcmp(s1->ptr(), s2->ptr(), s1->size());
#endif
}

int cpp_fstring_starts_with(void* fstr, const uint8_t* prefix, size_t prefix_len) {
    if (!fstr || !prefix) return 0;
    
    auto* s = static_cast<fstring*>(fstr);
    if (prefix_len > s->size()) return 0;
    
#ifdef HAVE_TOPLING_ZIP
    fstring prefix_str(prefix, prefix_len);
    return s->starts_with(prefix_str) ? 1 : 0;
#else
    return memcmp(s->ptr(), prefix, prefix_len) == 0 ? 1 : 0;
#endif
}

int cpp_fstring_ends_with(void* fstr, const uint8_t* suffix, size_t suffix_len) {
    if (!fstr || !suffix) return 0;
    
    auto* s = static_cast<fstring*>(fstr);
    if (suffix_len > s->size()) return 0;
    
#ifdef HAVE_TOPLING_ZIP
    fstring suffix_str(suffix, suffix_len);
    return s->ends_with(suffix_str) ? 1 : 0;
#else
    size_t start = s->size() - suffix_len;
    return memcmp(s->ptr() + start, suffix, suffix_len) == 0 ? 1 : 0;
#endif
}

void* cpp_fstring_concat(void* fstr1, void* fstr2) {
    if (!fstr1 || !fstr2) return nullptr;
    
    auto* s1 = static_cast<fstring*>(fstr1);
    auto* s2 = static_cast<fstring*>(fstr2);
    
    size_t total_len = s1->size() + s2->size();
    uint8_t* buffer = new uint8_t[total_len];
    
    memcpy(buffer, s1->ptr(), s1->size());
    memcpy(buffer + s1->size(), s2->ptr(), s2->size());
    
    auto* result = new fstring(buffer, total_len);
    delete[] buffer;
    
    return result;
}

void* cpp_fstring_repeat(void* fstr, size_t times) {
    if (!fstr || times == 0) return nullptr;
    
    auto* s = static_cast<fstring*>(fstr);
    size_t total_len = s->size() * times;
    uint8_t* buffer = new uint8_t[total_len];
    
    for (size_t i = 0; i < times; ++i) {
        memcpy(buffer + i * s->size(), s->ptr(), s->size());
    }
    
    auto* result = new fstring(buffer, total_len);
    delete[] buffer;
    
    return result;
}

// ============================================================================
// Hash Map Operations Implementation
// ============================================================================

void* cpp_hashmap_create() {
    track_allocation(sizeof(HashMap));
    return new HashMap();
}

void cpp_hashmap_destroy(void* map) {
    if (map) {
        track_deallocation(sizeof(HashMap));
        delete static_cast<HashMap*>(map);
    }
}

int cpp_hashmap_insert(void* map, const char* key, int32_t value) {
    if (!map || !key) return 0;
    
    auto* m = static_cast<HashMap*>(map);
    return m->insert(std::string(key), value) ? 1 : 0;
}

int cpp_hashmap_get(void* map, const char* key, int32_t* value) {
    if (!map || !key || !value) return 0;
    
    auto* m = static_cast<HashMap*>(map);
    return m->get(std::string(key), *value) ? 1 : 0;
}

int cpp_hashmap_remove(void* map, const char* key) {
    if (!map || !key) return 0;
    
    auto* m = static_cast<HashMap*>(map);
    return m->remove(std::string(key)) ? 1 : 0;
}

size_t cpp_hashmap_size(void* map) {
    if (!map) return 0;
    return static_cast<HashMap*>(map)->size();
}

void cpp_hashmap_clear(void* map) {
    if (map) {
        static_cast<HashMap*>(map)->clear();
    }
}

int cpp_hashmap_insert_batch(void* map, const char** keys, const int32_t* values, size_t count) {
    if (!map || !keys || !values) return 0;
    
    auto* m = static_cast<HashMap*>(map);
    for (size_t i = 0; i < count; ++i) {
        if (!m->insert(std::string(keys[i]), values[i])) {
            return 0;
        }
    }
    return 1;
}

int cpp_hashmap_get_batch(void* map, const char** keys, int32_t* values, size_t count) {
    if (!map || !keys || !values) return 0;
    
    auto* m = static_cast<HashMap*>(map);
    for (size_t i = 0; i < count; ++i) {
        if (!m->get(std::string(keys[i]), values[i])) {
            return 0;
        }
    }
    return 1;
}

// ============================================================================
// Bit Vector Operations Implementation
// ============================================================================

void* cpp_bitvector_create() {
    track_allocation(sizeof(BitVector));
    return new BitVector();
}

void cpp_bitvector_destroy(void* bv) {
    if (bv) {
        track_deallocation(sizeof(BitVector));
        delete static_cast<BitVector*>(bv);
    }
}

void cpp_bitvector_push(void* bv, int bit) {
    if (bv) {
        static_cast<BitVector*>(bv)->push_back(bit != 0);
    }
}

int cpp_bitvector_get(void* bv, size_t pos) {
    if (!bv) return 0;
    
    auto* b = static_cast<BitVector*>(bv);
    return (pos < b->size() && (*b)[pos]) ? 1 : 0;
}

size_t cpp_bitvector_size(void* bv) {
    return bv ? static_cast<BitVector*>(bv)->size() : 0;
}

void cpp_bitvector_push_batch(void* bv, const int* bits, size_t count) {
    if (!bv || !bits) return;
    
    auto* b = static_cast<BitVector*>(bv);
    for (size_t i = 0; i < count; ++i) {
        b->push_back(bits[i] != 0);
    }
}

// ============================================================================
// Memory Management Implementation
// ============================================================================

uint64_t cpp_get_deallocation_count() {
    return g_deallocation_count.load();
}

uint64_t cpp_get_peak_memory_usage() {
    return g_peak_memory_usage.load();
}

void cpp_get_memory_stats(CppMemoryStats* stats) {
    if (!stats) return;
    
    stats->total_allocated = g_total_allocated.load();
    stats->total_deallocated = g_total_deallocated.load();
    stats->current_usage = g_memory_usage.load();
    stats->peak_usage = g_peak_memory_usage.load();
    stats->allocation_count = g_allocation_count.load();
    stats->deallocation_count = g_deallocation_count.load();
    
    if (stats->allocation_count > 0) {
        stats->average_allocation_size = static_cast<double>(stats->total_allocated) / stats->allocation_count;
    } else {
        stats->average_allocation_size = 0.0;
    }
    
    // Simple fragmentation estimation
    if (stats->total_allocated > 0) {
        stats->fragmentation_ratio = static_cast<double>(stats->current_usage) / stats->total_allocated;
    } else {
        stats->fragmentation_ratio = 0.0;
    }
}

void* cpp_memory_pool_create(size_t block_size, size_t initial_blocks) {
    track_allocation(sizeof(MemoryPool) + block_size * initial_blocks);
    return new MemoryPool(block_size, initial_blocks);
}

void cpp_memory_pool_destroy(void* pool) {
    if (pool) {
        auto* p = static_cast<MemoryPool*>(pool);
        track_deallocation(sizeof(MemoryPool) + p->block_size * (p->allocated_count() + p->free_count()));
        delete p;
    }
}

void* cpp_memory_pool_alloc(void* pool) {
    return pool ? static_cast<MemoryPool*>(pool)->alloc() : nullptr;
}

void cpp_memory_pool_free(void* pool, void* ptr) {
    if (pool) {
        static_cast<MemoryPool*>(pool)->free(ptr);
    }
}

size_t cpp_memory_pool_allocated_blocks(void* pool) {
    return pool ? static_cast<MemoryPool*>(pool)->allocated_count() : 0;
}

size_t cpp_memory_pool_free_blocks(void* pool) {
    return pool ? static_cast<MemoryPool*>(pool)->free_count() : 0;
}

// ============================================================================
// Advanced Performance Measurement Implementation
// ============================================================================

void cpp_flush_caches() {
#ifdef __linux__
    // Flush CPU caches by reading/writing large amount of data
    const size_t cache_size = 32 * 1024 * 1024; // 32MB
    volatile char* data = new char[cache_size];
    
    for (size_t i = 0; i < cache_size; ++i) {
        data[i] = static_cast<char>(i);
    }
    
    // Read back to ensure cache pollution
    volatile char sum = 0;
    for (size_t i = 0; i < cache_size; ++i) {
        sum += data[i];
    }
    
    delete[] data;
    (void)sum; // Prevent optimization
#endif
}

double cpp_measure_cache_miss_rate(void* data, size_t size, size_t iterations) {
    if (!data) return 0.0;
    
    volatile char* ptr = static_cast<volatile char*>(data);
    auto start = std::chrono::high_resolution_clock::now();
    
    // Random access pattern to induce cache misses
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dis(0, size - 1);
    
    volatile char sum = 0;
    for (size_t i = 0; i < iterations; ++i) {
        sum += ptr[dis(gen)];
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::micro>(end - start);
    
    (void)sum; // Prevent optimization
    return duration.count();
}

double cpp_measure_memory_bandwidth(size_t size, size_t iterations) {
    auto* data = new volatile char[size];
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t iter = 0; iter < iterations; ++iter) {
        // Sequential write
        for (size_t i = 0; i < size; ++i) {
            data[i] = static_cast<char>(i + iter);
        }
        
        // Sequential read
        volatile char sum = 0;
        for (size_t i = 0; i < size; ++i) {
            sum += data[i];
        }
        (void)sum;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(end - start);
    
    delete[] data;
    
    // Return bandwidth in MB/s
    double bytes_processed = static_cast<double>(size * iterations * 2); // read + write
    return (bytes_processed / (1024 * 1024)) / duration.count();
}

double cpp_measure_deallocation_speed(void** ptrs, size_t count) {
    if (!ptrs) return 0.0;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < count; ++i) {
        if (ptrs[i]) {
            free(ptrs[i]);
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::micro>(end - start);
    
    return duration.count();
}

double cpp_measure_reallocation_speed(size_t initial_size, size_t final_size, size_t iterations) {
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < iterations; ++i) {
        void* ptr = malloc(initial_size);
        ptr = realloc(ptr, final_size);
        free(ptr);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::micro>(end - start);
    
    return duration.count();
}

// ============================================================================
// String Performance Measurement Implementation
// ============================================================================

double cpp_measure_string_find_speed(const uint8_t* text, size_t text_len, 
                                     const uint8_t* pattern, size_t pattern_len, 
                                     size_t iterations) {
    if (!text || !pattern) return 0.0;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < iterations; ++i) {
        void* fstr = cpp_fstring_create(text, text_len);
        int64_t pos = cpp_fstring_find(fstr, pattern, pattern_len);
        cpp_fstring_destroy(fstr);
        (void)pos; // Prevent optimization
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::micro>(end - start);
    
    return duration.count();
}

double cpp_measure_string_compare_speed(const uint8_t* str1, size_t len1,
                                        const uint8_t* str2, size_t len2,
                                        size_t iterations) {
    if (!str1 || !str2) return 0.0;
    
    void* fstr1 = cpp_fstring_create(str1, len1);
    void* fstr2 = cpp_fstring_create(str2, len2);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < iterations; ++i) {
        int result = cpp_fstring_compare(fstr1, fstr2);
        (void)result; // Prevent optimization
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::micro>(end - start);
    
    cpp_fstring_destroy(fstr1);
    cpp_fstring_destroy(fstr2);
    
    return duration.count();
}

// ============================================================================
// Vector Performance Measurement Implementation
// ============================================================================

double cpp_measure_vector_push_speed(size_t count, size_t iterations) {
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t iter = 0; iter < iterations; ++iter) {
        void* vec = cpp_valvec_create();
        for (size_t i = 0; i < count; ++i) {
            cpp_valvec_push(vec, static_cast<int32_t>(i));
        }
        cpp_valvec_destroy(vec);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::micro>(end - start);
    
    return duration.count();
}

double cpp_measure_vector_access_speed(void* vec, size_t size, size_t iterations) {
    if (!vec) return 0.0;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t iter = 0; iter < iterations; ++iter) {
        int64_t sum = 0;
        for (size_t i = 0; i < size; ++i) {
            sum += cpp_valvec_get(vec, i);
        }
        (void)sum; // Prevent optimization
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::micro>(end - start);
    
    return duration.count();
}

double cpp_measure_vector_iteration_speed(void* vec, size_t iterations) {
    if (!vec) return 0.0;
    
    size_t size = cpp_valvec_size(vec);
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t iter = 0; iter < iterations; ++iter) {
        int64_t sum = 0;
        for (size_t i = 0; i < size; ++i) {
            sum += cpp_valvec_get(vec, i);
        }
        (void)sum; // Prevent optimization
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::micro>(end - start);
    
    return duration.count();
}

// ============================================================================
// Hash Map Performance Measurement Implementation
// ============================================================================

double cpp_measure_hashmap_insert_speed(size_t count, size_t iterations) {
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t iter = 0; iter < iterations; ++iter) {
        void* map = cpp_hashmap_create();
        for (size_t i = 0; i < count; ++i) {
            std::string key = "key_" + std::to_string(i);
            cpp_hashmap_insert(map, key.c_str(), static_cast<int32_t>(i));
        }
        cpp_hashmap_destroy(map);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::micro>(end - start);
    
    return duration.count();
}

double cpp_measure_hashmap_lookup_speed(void* map, const char** keys, size_t count, size_t iterations) {
    if (!map || !keys) return 0.0;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t iter = 0; iter < iterations; ++iter) {
        for (size_t i = 0; i < count; ++i) {
            int32_t value;
            int result = cpp_hashmap_get(map, keys[i], &value);
            (void)result; (void)value; // Prevent optimization
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::micro>(end - start);
    
    return duration.count();
}

// ============================================================================
// System Information Implementation
// ============================================================================

void cpp_get_system_info(CppSystemInfo* info) {
    if (!info) return;
    
    memset(info, 0, sizeof(CppSystemInfo));
    
#ifdef __linux__
    info->physical_memory = sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGE_SIZE);
    info->page_size = sysconf(_SC_PAGE_SIZE);
    info->cpu_cores = sysconf(_SC_NPROCESSORS_ONLN);
    info->logical_cores = std::thread::hardware_concurrency();
    
    // Try to read cache info from /proc/cpuinfo
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
        if (line.find("cache size") != std::string::npos) {
            // Parse cache size
            size_t pos = line.find(":");
            if (pos != std::string::npos) {
                std::string cache_str = line.substr(pos + 1);
                std::stringstream ss(cache_str);
                size_t cache_kb;
                if (ss >> cache_kb) {
                    info->l3_cache_size = cache_kb * 1024; // Convert to bytes
                }
            }
        }
        if (line.find("model name") != std::string::npos) {
            size_t pos = line.find(":");
            if (pos != std::string::npos) {
                std::string model = line.substr(pos + 1);
                strncpy(info->cpu_model, model.c_str(), sizeof(info->cpu_model) - 1);
            }
        }
    }
    
    // Set typical cache sizes if not detected
    if (info->l1_cache_size == 0) info->l1_cache_size = 32 * 1024;
    if (info->l2_cache_size == 0) info->l2_cache_size = 256 * 1024;
    if (info->l3_cache_size == 0) info->l3_cache_size = 8 * 1024 * 1024;
    if (info->cache_line_size == 0) info->cache_line_size = 64;
    
#elif defined(_WIN32)
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    info->cpu_cores = sysinfo.dwNumberOfProcessors;
    info->logical_cores = std::thread::hardware_concurrency();
    info->page_size = sysinfo.dwPageSize;
    
    MEMORYSTATUSEX meminfo;
    meminfo.dwLength = sizeof(MEMORYSTATUSEX);
    GlobalMemoryStatusEx(&meminfo);
    info->physical_memory = meminfo.ullTotalPhys;
    
#elif defined(__APPLE__)
    size_t size = sizeof(info->physical_memory);
    sysctlbyname("hw.memsize", &info->physical_memory, &size, NULL, 0);
    
    size = sizeof(info->cpu_cores);
    sysctlbyname("hw.physicalcpu", &info->cpu_cores, &size, NULL, 0);
    
    size = sizeof(info->logical_cores);
    sysctlbyname("hw.logicalcpu", &info->logical_cores, &size, NULL, 0);
    
#endif
    
    strcpy(info->cpu_vendor, "Unknown");
}

int cpp_has_avx2() {
#ifdef __linux__
    uint32_t eax, ebx, ecx, edx;
    if (__get_cpuid_max(0, NULL) >= 7) {
        __cpuid_count(7, 0, eax, ebx, ecx, edx);
        return (ebx & (1 << 5)) ? 1 : 0; // AVX2 bit
    }
#endif
    return 0;
}

int cpp_has_sse42() {
#ifdef __linux__
    uint32_t eax, ebx, ecx, edx;
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        return (ecx & (1 << 20)) ? 1 : 0; // SSE4.2 bit
    }
#endif
    return 0;
}

int cpp_has_bmi2() {
#ifdef __linux__
    uint32_t eax, ebx, ecx, edx;
    if (__get_cpuid_max(0, NULL) >= 7) {
        __cpuid_count(7, 0, eax, ebx, ecx, edx);
        return (ebx & (1 << 8)) ? 1 : 0; // BMI2 bit
    }
#endif
    return 0;
}

// ============================================================================
// Advanced Timing and Statistics Implementation
// ============================================================================

void cpp_timer_start(CppTimer* timer) {
    if (!timer) return;
    
    timer->start_time = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
    
#ifdef __x86_64__
    asm volatile("rdtsc" : "=a"(timer->start_cycles), "=d"(timer->end_cycles));
    timer->start_cycles |= (static_cast<uint64_t>(timer->end_cycles) << 32);
#endif
}

void cpp_timer_stop(CppTimer* timer) {
    if (!timer) return;
    
    timer->end_time = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
    
#ifdef __x86_64__
    uint32_t low, high;
    asm volatile("rdtsc" : "=a"(low), "=d"(high));
    timer->end_cycles = static_cast<uint64_t>(low) | (static_cast<uint64_t>(high) << 32);
#endif
}

double cpp_timer_elapsed_seconds(const CppTimer* timer) {
    if (!timer) return 0.0;
    return timer->end_time - timer->start_time;
}

uint64_t cpp_timer_elapsed_cycles(const CppTimer* timer) {
    if (!timer) return 0;
    return timer->end_cycles - timer->start_cycles;
}

// ============================================================================
// Memory Access Pattern Analysis Implementation
// ============================================================================

double cpp_measure_sequential_access_speed(void* data, size_t size, size_t iterations) {
    if (!data) return 0.0;
    
    volatile char* ptr = static_cast<volatile char*>(data);
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t iter = 0; iter < iterations; ++iter) {
        volatile char sum = 0;
        for (size_t i = 0; i < size; ++i) {
            sum += ptr[i];
        }
        (void)sum; // Prevent optimization
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::micro>(end - start);
    
    return duration.count();
}

double cpp_measure_random_access_speed(void* data, size_t size, size_t iterations) {
    if (!data) return 0.0;
    
    volatile char* ptr = static_cast<volatile char*>(data);
    
    // Generate random indices
    std::vector<size_t> indices(iterations);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dis(0, size - 1);
    
    for (size_t i = 0; i < iterations; ++i) {
        indices[i] = dis(gen);
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    volatile char sum = 0;
    for (size_t i = 0; i < iterations; ++i) {
        sum += ptr[indices[i]];
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::micro>(end - start);
    
    (void)sum; // Prevent optimization
    return duration.count();
}

double cpp_measure_strided_access_speed(void* data, size_t size, size_t stride, size_t iterations) {
    if (!data) return 0.0;
    
    volatile char* ptr = static_cast<volatile char*>(data);
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t iter = 0; iter < iterations; ++iter) {
        volatile char sum = 0;
        for (size_t i = 0; i < size; i += stride) {
            sum += ptr[i];
        }
        (void)sum; // Prevent optimization
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::micro>(end - start);
    
    return duration.count();
}

// ============================================================================
// Comprehensive Benchmark Suite Implementation
// ============================================================================

void cpp_run_comprehensive_benchmark(CppPerformanceSummary* summary) {
    if (!summary) return;
    
    memset(summary, 0, sizeof(CppPerformanceSummary));
    
    // Vector performance
    summary->vector_push_throughput = 1.0 / cpp_measure_vector_push_speed(10000, 100);
    
    void* test_vec = cpp_valvec_create();
    for (int i = 0; i < 10000; ++i) {
        cpp_valvec_push(test_vec, i);
    }
    summary->vector_access_throughput = 1.0 / cpp_measure_vector_access_speed(test_vec, 10000, 100);
    cpp_valvec_destroy(test_vec);
    
    // String performance
    const char* test_string = "The quick brown fox jumps over the lazy dog";
    summary->string_hash_throughput = 1.0 / cpp_measure_hash_speed(
        reinterpret_cast<const uint8_t*>(test_string), strlen(test_string), 1000
    );
    
    const char* pattern = "fox";
    summary->string_find_throughput = 1.0 / cpp_measure_string_find_speed(
        reinterpret_cast<const uint8_t*>(test_string), strlen(test_string),
        reinterpret_cast<const uint8_t*>(pattern), strlen(pattern), 1000
    );
    
    // Hash map performance
    summary->hashmap_insert_throughput = 1.0 / cpp_measure_hashmap_insert_speed(1000, 10);
    
    // Memory system performance
    summary->allocation_throughput = 1.0 / cpp_measure_allocation_speed(1000, 64);
    summary->memory_bandwidth = cpp_measure_memory_bandwidth(1024 * 1024, 10);
    
    // Cache efficiency (inverse of cache miss time)
    void* cache_data = malloc(1024 * 1024);
    summary->cache_efficiency = 1.0 / cpp_measure_cache_miss_rate(cache_data, 1024 * 1024, 10000);
    free(cache_data);
    
    // Calculate overall score as geometric mean
    double scores[] = {
        summary->vector_push_throughput,
        summary->vector_access_throughput,
        summary->string_hash_throughput,
        summary->string_find_throughput,
        summary->hashmap_insert_throughput,
        summary->allocation_throughput,
        summary->memory_bandwidth,
        summary->cache_efficiency
    };
    
    double product = 1.0;
    for (double score : scores) {
        if (score > 0) product *= score;
    }
    
    summary->overall_score = std::pow(product, 1.0 / (sizeof(scores) / sizeof(scores[0])));
}

void cpp_compare_with_baseline(const CppPerformanceSummary* current, 
                               const CppPerformanceSummary* baseline,
                               double* improvement_factors) {
    if (!current || !baseline || !improvement_factors) return;
    
    improvement_factors[0] = current->vector_push_throughput / baseline->vector_push_throughput;
    improvement_factors[1] = current->vector_access_throughput / baseline->vector_access_throughput;
    improvement_factors[2] = current->string_hash_throughput / baseline->string_hash_throughput;
    improvement_factors[3] = current->string_find_throughput / baseline->string_find_throughput;
    improvement_factors[4] = current->hashmap_insert_throughput / baseline->hashmap_insert_throughput;
    improvement_factors[5] = current->allocation_throughput / baseline->allocation_throughput;
    improvement_factors[6] = current->memory_bandwidth / baseline->memory_bandwidth;
    improvement_factors[7] = current->cache_efficiency / baseline->cache_efficiency;
    improvement_factors[8] = current->overall_score / baseline->overall_score;
}

} // extern "C"