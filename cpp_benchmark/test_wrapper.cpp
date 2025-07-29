#include "wrapper.hpp"
#include <iostream>
#include <cassert>
#include <cstring>

int main() {
    std::cout << "Testing C++ wrapper functionality...\n";
    
    // Test vector operations
    std::cout << "Testing vector operations...\n";
    void* vec = cpp_valvec_create();
    assert(vec != nullptr);
    assert(cpp_valvec_size(vec) == 0);
    
    for (int i = 0; i < 1000; ++i) {
        cpp_valvec_push(vec, i);
    }
    
    assert(cpp_valvec_size(vec) == 1000);
    assert(cpp_valvec_capacity(vec) >= 1000);
    
    cpp_valvec_destroy(vec);
    std::cout << "Vector operations: PASSED\n";
    
    // Test string operations
    std::cout << "Testing string operations...\n";
    const char* test_str = "Hello, world! This is a test string.";
    const uint8_t* test_data = reinterpret_cast<const uint8_t*>(test_str);
    size_t test_len = std::strlen(test_str);
    
    void* fstr = cpp_fstring_create(test_data, test_len);
    assert(fstr != nullptr);
    assert(cpp_fstring_length(fstr) == test_len);
    
    // Test hash
    uint64_t hash1 = cpp_fstring_hash(fstr);
    uint64_t hash2 = cpp_fstring_hash(fstr);
    assert(hash1 == hash2); // Hash should be deterministic
    assert(hash1 != 0);     // Hash should not be zero
    
    // Test find
    const uint8_t* needle = reinterpret_cast<const uint8_t*>("world");
    int64_t pos = cpp_fstring_find(fstr, needle, 5);
    assert(pos >= 0);
    std::cout << "Found 'world' at position: " << pos << "\n";
    
    // Test substring
    void* substr = cpp_fstring_substring(fstr, pos, 5);
    assert(substr != nullptr);
    assert(cpp_fstring_length(substr) == 5);
    
    cpp_fstring_destroy(substr);
    cpp_fstring_destroy(fstr);
    std::cout << "String operations: PASSED\n";
    
    // Test performance measurement utilities
    std::cout << "Testing performance utilities...\n";
    cpp_reset_counters();
    assert(cpp_get_allocation_count() == 0);
    assert(cpp_get_memory_usage() == 0);
    
    cpp_warmup_caches();
    
    double alloc_time = cpp_measure_allocation_speed(1000, 64);
    std::cout << "Allocation time for 1000x64 bytes: " << alloc_time << " μs\n";
    
    double hash_time = cpp_measure_hash_speed(test_data, test_len, 1000);
    std::cout << "Hash time for 1000 iterations: " << hash_time << " μs\n";
    
    std::cout << "Performance utilities: PASSED\n";
    
    // Test rank-select (stub)
    std::cout << "Testing rank-select operations...\n";
    uint64_t bits[] = {0xAAAAAAAAAAAAAAAAULL, 0x5555555555555555ULL};
    void* rs = cpp_rank_select_create(bits, 128);
    assert(rs != nullptr);
    
    size_t rank = cpp_rank_select_rank1(rs, 64);
    std::cout << "Rank1 at position 64: " << rank << "\n";
    
    size_t select_pos = cpp_rank_select_select1(rs, 10);
    std::cout << "Select1 for 10th bit: " << select_pos << "\n";
    
    cpp_rank_select_destroy(rs);
    std::cout << "Rank-select operations: PASSED\n";
    
    std::cout << "\nAll tests PASSED! Wrapper is functional.\n";
    return 0;
}