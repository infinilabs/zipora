/**
 * C++ Implementation of benchmark wrapper for topling-zip
 * 
 * This file implements C-compatible wrappers around the original topling-zip
 * C++ classes to enable fair performance comparisons with the Rust implementation.
 */

#include "wrapper.hpp"
#include <chrono>
#include <memory>
#include <atomic>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <algorithm>

// Include the original topling-zip headers
// Note: These paths may need adjustment based on actual topling-zip installation
#ifdef HAVE_TOPLING_ZIP
#include <terark/valvec.hpp>
#include <terark/fstring.hpp>
// If rank-select is available in topling-zip:
// #include <terark/succinct/rank_select.hpp>
#endif

#ifdef HAVE_TOPLING_ZIP
using namespace terark;
#else
// Fallback implementations for when topling-zip is not available
// These provide stub functionality for testing the benchmark framework

template<typename T>
class ValVecStub {
public:
    std::vector<T> data;
    
    void push_back(const T& val) { data.push_back(val); }
    size_t size() const { return data.size(); }
    size_t capacity() const { return data.capacity(); }
    T operator[](size_t i) const { return data[i]; }
    void reserve(size_t cap) { data.reserve(cap); }
};

class FStringStub {
public:
    std::vector<uint8_t> data;
    
    FStringStub(const uint8_t* ptr, size_t len) : data(ptr, ptr + len) {}
    
    uint64_t hash() const {
        // Simple hash for testing
        uint64_t h = 2134173;
        for (uint8_t b : data) {
            h = h * 31 + b;
        }
        return h;
    }
    
    int64_t find(const uint8_t* needle, size_t needle_len) const {
        if (needle_len > data.size()) return -1;
        
        for (size_t i = 0; i <= data.size() - needle_len; ++i) {
            if (memcmp(&data[i], needle, needle_len) == 0) {
                return static_cast<int64_t>(i);
            }
        }
        return -1;
    }
    
    size_t size() const { return data.size(); }
    const uint8_t* ptr() const { return data.data(); }
};

template<typename T>
using valvec = ValVecStub<T>;
using fstring = FStringStub;
#endif

// Global counters for memory tracking
static std::atomic<uint64_t> g_memory_usage{0};
static std::atomic<uint64_t> g_allocation_count{0};

extern "C" {

// Vector operations
void* cpp_valvec_create() {
    g_allocation_count++;
    g_memory_usage += sizeof(valvec<int32_t>);
    return new valvec<int32_t>();
}

void cpp_valvec_destroy(void* vec) {
    if (vec) {
        auto* v = static_cast<valvec<int32_t>*>(vec);
        g_memory_usage -= sizeof(valvec<int32_t>);
        delete v;
    }
}

void cpp_valvec_push(void* vec, int32_t value) {
    if (vec) {
        auto* v = static_cast<valvec<int32_t>*>(vec);
        size_t old_cap = v->capacity();
        v->push_back(value);
        size_t new_cap = v->capacity();
        if (new_cap > old_cap) {
            g_memory_usage += (new_cap - old_cap) * sizeof(int32_t);
        }
    }
}

size_t cpp_valvec_size(void* vec) {
    if (vec) {
        auto* v = static_cast<valvec<int32_t>*>(vec);
        return v->size();
    }
    return 0;
}

size_t cpp_valvec_capacity(void* vec) {
    if (vec) {
        auto* v = static_cast<valvec<int32_t>*>(vec);
        return v->capacity();
    }
    return 0;
}

int32_t cpp_valvec_get(void* vec, size_t index) {
    if (vec) {
        auto* v = static_cast<valvec<int32_t>*>(vec);
        if (index < v->size()) {
            return (*v)[index];
        }
    }
    return 0;
}

void cpp_valvec_reserve(void* vec, size_t capacity) {
    if (vec) {
        auto* v = static_cast<valvec<int32_t>*>(vec);
        size_t old_cap = v->capacity();
        v->reserve(capacity);
        size_t new_cap = v->capacity();
        if (new_cap > old_cap) {
            g_memory_usage += (new_cap - old_cap) * sizeof(int32_t);
        }
    }
}

// String operations
void* cpp_fstring_create(const uint8_t* data, size_t len) {
    g_allocation_count++;
    g_memory_usage += sizeof(fstring) + len;
    return new fstring(data, len);
}

void cpp_fstring_destroy(void* fstr) {
    if (fstr) {
        auto* s = static_cast<fstring*>(fstr);
        g_memory_usage -= sizeof(fstring) + s->size();
        delete s;
    }
}

uint64_t cpp_fstring_hash(void* fstr) {
    if (fstr) {
        auto* s = static_cast<fstring*>(fstr);
#ifdef HAVE_TOPLING_ZIP
        return s->hash();
#else
        return s->hash();
#endif
    }
    return 0;
}

int64_t cpp_fstring_find(void* fstr, const uint8_t* needle, size_t needle_len) {
    if (fstr && needle) {
        auto* s = static_cast<fstring*>(fstr);
#ifdef HAVE_TOPLING_ZIP
        fstring needle_str(needle, needle_len);
        size_t pos = s->find(needle_str);
        return pos != fstring::npos ? static_cast<int64_t>(pos) : -1;
#else
        return s->find(needle, needle_len);
#endif
    }
    return -1;
}

void* cpp_fstring_substring(void* fstr, size_t start, size_t len) {
    if (fstr) {
        auto* s = static_cast<fstring*>(fstr);
        if (start < s->size()) {
            size_t actual_len = std::min(len, s->size() - start);
#ifdef HAVE_TOPLING_ZIP
            return new fstring(s->substr(start, actual_len));
#else
            return new fstring(s->ptr() + start, actual_len);
#endif
        }
    }
    return nullptr;
}

size_t cpp_fstring_length(void* fstr) {
    if (fstr) {
        auto* s = static_cast<fstring*>(fstr);
        return s->size();
    }
    return 0;
}

const uint8_t* cpp_fstring_data(void* fstr) {
    if (fstr) {
        auto* s = static_cast<fstring*>(fstr);
#ifdef HAVE_TOPLING_ZIP
        return reinterpret_cast<const uint8_t*>(s->data());
#else
        return s->ptr();
#endif
    }
    return nullptr;
}

// Rank-select operations (stub implementation)
void* cpp_rank_select_create(const uint64_t* bits, size_t bit_count) {
    // Stub implementation - would need actual topling-zip rank-select if available
    g_allocation_count++;
    g_memory_usage += bit_count / 8 + 1024; // Estimated overhead
    return reinterpret_cast<void*>(0x1); // Non-null placeholder
}

void cpp_rank_select_destroy(void* rs) {
    if (rs) {
        g_memory_usage -= 1024; // Estimated cleanup
    }
}

size_t cpp_rank_select_rank1(void* rs, size_t pos) {
    // Stub implementation
    return pos / 7; // Simulate some rank operation
}

size_t cpp_rank_select_select1(void* rs, size_t k) {
    // Stub implementation  
    return k * 7; // Simulate some select operation
}

size_t cpp_rank_select_rank0(void* rs, size_t pos) {
    // Stub implementation
    return pos - (pos / 7);
}

// Performance measurement utilities
uint64_t cpp_get_memory_usage() {
    return g_memory_usage.load();
}

uint64_t cpp_get_allocation_count() {
    return g_allocation_count.load();
}

void cpp_reset_counters() {
    g_memory_usage = 0;
    g_allocation_count = 0;
}

void cpp_warmup_caches() {
    // Warm up CPU caches and memory subsystem
    const size_t warmup_size = 1024 * 1024; // 1MB
    volatile uint8_t* warmup_data = new uint8_t[warmup_size];
    
    for (size_t i = 0; i < warmup_size; ++i) {
        warmup_data[i] = static_cast<uint8_t>(i);
    }
    
    uint64_t sum = 0;
    for (size_t i = 0; i < warmup_size; ++i) {
        sum += warmup_data[i];
    }
    
    delete[] warmup_data;
    
    // Prevent optimization from removing the warmup
    volatile uint64_t dummy = sum;
    (void)dummy;
}

double cpp_measure_allocation_speed(size_t count, size_t size) {
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<void*> ptrs;
    ptrs.reserve(count);
    
    for (size_t i = 0; i < count; ++i) {
        ptrs.push_back(malloc(size));
    }
    
    for (void* ptr : ptrs) {
        free(ptr);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::micro>(end - start);
    return duration.count();
}

double cpp_measure_hash_speed(const uint8_t* data, size_t len, size_t iterations) {
    auto start = std::chrono::high_resolution_clock::now();
    
    uint64_t hash_sum = 0;
    for (size_t i = 0; i < iterations; ++i) {
        void* fstr = cpp_fstring_create(data, len);
        hash_sum += cpp_fstring_hash(fstr);
        cpp_fstring_destroy(fstr);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::micro>(end - start);
    
    // Prevent optimization from removing the computation
    volatile uint64_t dummy = hash_sum;
    (void)dummy;
    
    return duration.count();
}

} // extern "C"