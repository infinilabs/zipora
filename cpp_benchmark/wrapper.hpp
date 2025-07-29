#pragma once

/**
 * C++ Wrapper for topling-zip library benchmarking
 * 
 * This header provides C-compatible interfaces to the original topling-zip
 * C++ library for performance comparison benchmarks.
 */

#include <cstddef>
#include <cstdint>

extern "C" {

// Vector operations (valvec wrapper)
void* cpp_valvec_create();
void cpp_valvec_destroy(void* vec);
void cpp_valvec_push(void* vec, int32_t value);
size_t cpp_valvec_size(void* vec);
size_t cpp_valvec_capacity(void* vec);
int32_t cpp_valvec_get(void* vec, size_t index);
void cpp_valvec_reserve(void* vec, size_t capacity);

// String operations (fstring wrapper)  
void* cpp_fstring_create(const uint8_t* data, size_t len);
void cpp_fstring_destroy(void* fstr);
uint64_t cpp_fstring_hash(void* fstr);
int64_t cpp_fstring_find(void* fstr, const uint8_t* needle, size_t needle_len);
void* cpp_fstring_substring(void* fstr, size_t start, size_t len);
size_t cpp_fstring_length(void* fstr);
const uint8_t* cpp_fstring_data(void* fstr);

// Rank-select operations (if available)
void* cpp_rank_select_create(const uint64_t* bits, size_t bit_count);
void cpp_rank_select_destroy(void* rs);
size_t cpp_rank_select_rank1(void* rs, size_t pos);
size_t cpp_rank_select_select1(void* rs, size_t k);
size_t cpp_rank_select_rank0(void* rs, size_t pos);

// Performance measurement utilities
uint64_t cpp_get_memory_usage();
uint64_t cpp_get_allocation_count();
void cpp_reset_counters();

// Benchmark helper functions
void cpp_warmup_caches();
double cpp_measure_allocation_speed(size_t count, size_t size);
double cpp_measure_hash_speed(const uint8_t* data, size_t len, size_t iterations);

}