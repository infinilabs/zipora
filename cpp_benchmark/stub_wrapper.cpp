/**
 * Minimal stub implementation of C++ wrapper for benchmarking
 * This provides stub functions to enable Rust vs C++ comparison
 */

#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <chrono>
#include <unordered_map>
#include <string>

extern "C" {

// Vector operations
void* cpp_valvec_create() {
    return new std::vector<int32_t>();
}

void cpp_valvec_destroy(void* vec) {
    delete static_cast<std::vector<int32_t>*>(vec);
}

void cpp_valvec_push(void* vec, int32_t value) {
    static_cast<std::vector<int32_t>*>(vec)->push_back(value);
}

size_t cpp_valvec_size(void* vec) {
    return static_cast<std::vector<int32_t>*>(vec)->size();
}

size_t cpp_valvec_capacity(void* vec) {
    return static_cast<std::vector<int32_t>*>(vec)->capacity();
}

int32_t cpp_valvec_get(void* vec, size_t index) {
    return (*static_cast<std::vector<int32_t>*>(vec))[index];
}

// String operations
void* cpp_fstring_create(const uint8_t* data, size_t len) {
    auto* str = new std::vector<uint8_t>(data, data + len);
    return str;
}

void cpp_fstring_destroy(void* fstr) {
    delete static_cast<std::vector<uint8_t>*>(fstr);
}

uint64_t cpp_fstring_hash(void* fstr) {
    auto* vec = static_cast<std::vector<uint8_t>*>(fstr);
    uint64_t hash = 14695981039346656037ULL; // FNV-1a offset basis
    for (uint8_t byte : *vec) {
        hash ^= byte;
        hash *= 1099511628211ULL; // FNV-1a prime
    }
    return hash;
}

int64_t cpp_fstring_find(void* fstr, const uint8_t* needle, size_t needle_len) {
    auto* vec = static_cast<std::vector<uint8_t>*>(fstr);
    if (needle_len > vec->size()) return -1;
    
    for (size_t i = 0; i <= vec->size() - needle_len; ++i) {
        if (memcmp(&(*vec)[i], needle, needle_len) == 0) {
            return static_cast<int64_t>(i);
        }
    }
    return -1;
}

size_t cpp_fstring_length(void* fstr) {
    return static_cast<std::vector<uint8_t>*>(fstr)->size();
}

const uint8_t* cpp_fstring_data(void* fstr) {
    return static_cast<std::vector<uint8_t>*>(fstr)->data();
}

// Performance measurement
double cpp_measure_allocation_speed(size_t count, size_t size) {
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < count; ++i) {
        void* ptr = malloc(size);
        free(ptr);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> diff = end - start;
    return diff.count();
}

double cpp_measure_hash_performance(const uint8_t* data, size_t len, size_t iterations) {
    auto start = std::chrono::high_resolution_clock::now();
    
    uint64_t hash = 0;
    for (size_t i = 0; i < iterations; ++i) {
        hash = 14695981039346656037ULL;
        for (size_t j = 0; j < len; ++j) {
            hash ^= data[j];
            hash *= 1099511628211ULL;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> diff = end - start;
    return diff.count();
}

// Memory tracking stubs
uint64_t cpp_get_memory_usage() {
    return 0; // Stub
}

uint64_t cpp_get_allocation_count() {
    return 0; // Stub
}

// HashMap operations
void* cpp_hashmap_create() {
    return new std::unordered_map<std::string, int32_t>();
}

void cpp_hashmap_destroy(void* map) {
    delete static_cast<std::unordered_map<std::string, int32_t>*>(map);
}

void cpp_hashmap_insert(void* map, const char* key, int32_t value) {
    static_cast<std::unordered_map<std::string, int32_t>*>(map)->emplace(key, value);
}

bool cpp_hashmap_get(void* map, const char* key, int32_t* out_value) {
    auto* m = static_cast<std::unordered_map<std::string, int32_t>*>(map);
    auto it = m->find(key);
    if (it != m->end()) {
        *out_value = it->second;
        return true;
    }
    return false;
}

// Rank-select stub operations - note the different naming convention
void* cpp_rank_select_create(const uint64_t* bits, size_t bit_count) {
    size_t bytes = (bit_count + 7) / 8;
    auto* vec = new std::vector<uint8_t>(bytes);
    memcpy(vec->data(), bits, bytes);
    return vec;
}

void cpp_rank_select_destroy(void* rs) {
    delete static_cast<std::vector<uint8_t>*>(rs);
}

size_t cpp_rank_select_rank1(void* rs, size_t pos) {
    // Simple stub implementation
    auto* vec = static_cast<std::vector<uint8_t>*>(rs);
    size_t count = 0;
    size_t byte_pos = pos / 8;
    for (size_t i = 0; i < byte_pos && i < vec->size(); ++i) {
        uint8_t byte = (*vec)[i];
        count += __builtin_popcount(byte);
    }
    if (byte_pos < vec->size()) {
        uint8_t mask = (1 << (pos % 8)) - 1;
        count += __builtin_popcount((*vec)[byte_pos] & mask);
    }
    return count;
}

size_t cpp_rank_select_select1(void* rs, size_t n) {
    // Simple stub implementation
    auto* vec = static_cast<std::vector<uint8_t>*>(rs);
    size_t count = 0;
    for (size_t i = 0; i < vec->size(); ++i) {
        uint8_t byte = (*vec)[i];
        for (int j = 0; j < 8; ++j) {
            if (byte & (1 << j)) {
                count++;
                if (count == n) {
                    return i * 8 + j;
                }
            }
        }
    }
    return vec->size() * 8;
}

// Additional missing functions
void cpp_valvec_reserve(void* vec, size_t capacity) {
    static_cast<std::vector<int32_t>*>(vec)->reserve(capacity);
}

void* cpp_fstring_substring(void* fstr, size_t start, size_t len) {
    auto* vec = static_cast<std::vector<uint8_t>*>(fstr);
    if (start >= vec->size()) {
        return new std::vector<uint8_t>();
    }
    size_t actual_len = std::min(len, vec->size() - start);
    return new std::vector<uint8_t>(vec->begin() + start, vec->begin() + start + actual_len);
}

void cpp_reset_counters() {
    // Stub - no counters to reset
}

void cpp_warmup_caches() {
    // Stub - perform some dummy operations to warm up caches
    volatile int sum = 0;
    for (int i = 0; i < 1000000; ++i) {
        sum += i;
    }
}

double cpp_measure_hash_speed(const uint8_t* data, size_t len, size_t iterations) {
    return cpp_measure_hash_performance(data, len, iterations);
}

} // extern "C"