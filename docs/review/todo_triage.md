# API Honesty and Stub Marker Triage (Rule D1)

This document inventories all 182 stub, placeholder, and honesty markers found in non-test production code,
assigning each to its corresponding Phase C review session for implementation, proper error reporting, or removal.

**Triage Policy (D1 / Agreement 8):**
1. **Implement**: Provide the fully working, tested implementation.
2. **Unsupported Error**: Return `Err(ZiporaError::unsupported(..))` if the strategy/feature is unconfigured/unsupported.
3. **Delete**: Remove the stubbed API if it does not belong in the public surface.
4. **Refine Prose**: For `simplified` occurrences in doc comments, audit and replace with exact algorithmic descriptions.

Total markers in non-test code: 182

| # | Location | Marker Type | Snippet | Subsystem / Disposition |
|---|---|---|---|---|
| 1 | `src/lib.rs:4` | `stub` | `// #![allow(unused_variables)] // Stub implementations have unused params` | Phase C (lib.rs): Implement or delete stub |
| 2 | `src/compression/simd_pattern_match.rs:627` | `for now` | `matches.push((absolute_pos, absolute_pos)); // For now, use same position` | Phase C (compression): Remove temporary fallback; verify production invariant |
| 3 | `src/compression/simd_pattern_match.rs:730` | `for now` | `// For now, fallback to SSE4.2` | Phase C (compression): Remove temporary fallback; verify production invariant |
| 4 | `src/compression/simd_pattern_match.rs:742` | `for now` | `// For now, fallback to SSE4.2` | Phase C (compression): Remove temporary fallback; verify production invariant |
| 5 | `src/compression/simd_pattern_match.rs:754` | `for now` | `// For now, fallback to AVX2` | Phase C (compression): Remove temporary fallback; verify production invariant |
| 6 | `src/compression/simd_pattern_match.rs:869` | `for now` | `// For now, create a literal match - this would be enhanced to create` | Phase C (compression): Remove temporary fallback; verify production invariant |
| 7 | `src/compression/simd_pattern_match.rs:1234` | `for now` | `Match::literal(length as u8) // Simple literal for now` | Phase C (compression): Remove temporary fallback; verify production invariant |
| 8 | `src/compression/suffix_array.rs:516` | `for now` | `// For now, use compressed storage always` | Phase C (compression): Remove temporary fallback; verify production invariant |
| 9 | `src/compression/dict_zip/blob_store.rs:1009` | `for now` | `// For now, just validate consistency` | Phase C (compression): Remove temporary fallback; verify production invariant |
| 10 | `src/compression/dict_zip/builder.rs:685` | `for now` | `// So for now, just return the sampled data to fix the immediate test failure` | Phase C (compression): Remove temporary fallback; verify production invariant |
| 11 | `src/compression/dict_zip/builder.rs:691` | `for now` | `// For now, return sampled data to fix size constraint violation` | Phase C (compression): Remove temporary fallback; verify production invariant |
| 12 | `src/compression/dict_zip/builder.rs:698` | `simplified` | `/// This is a simplified version - full implementation would need pattern extraction` | Phase C (compression): Audit doc/code; replace with exact specification |
| 13 | `src/compression/dict_zip/builder.rs:700` | `for now` | `// For now, implement a basic version that respects size constraints` | Phase C (compression): Remove temporary fallback; verify production invariant |
| 14 | `src/compression/dict_zip/builder.rs:927` | `for now` | `// For now, return the concatenation of all unique samples` | Phase C (compression): Remove temporary fallback; verify production invariant |
| 15 | `src/compression/dict_zip/compression_types.rs:878` | `for now` | `// For now, assume 32-bit position + 16-bit length` | Phase C (compression): Remove temporary fallback; verify production invariant |
| 16 | `src/compression/dict_zip/compression_types.rs:963` | `for now` | `// For now, assume 32-bit position + 16-bit length` | Phase C (compression): Remove temporary fallback; verify production invariant |
| 17 | `src/compression/dict_zip/compression_types.rs:1136` | `for now` | `CompressionType::Global => return usize::MAX, // Skip global for now` | Phase C (compression): Remove temporary fallback; verify production invariant |
| 18 | `src/compression/dict_zip/compressor.rs:847` | `for now` | `// Process blocks sequentially for now (true parallelism would require thread safety)` | Phase C (compression): Remove temporary fallback; verify production invariant |
| 19 | `src/compression/dict_zip/dfa_cache.rs:437` | `for now` | `// For now, we need to simulate the DFA state structure` | Phase C (compression): Remove temporary fallback; verify production invariant |
| 20 | `src/compression/dict_zip/dfa_cache.rs:439` | `simplified` | `// This is a simplified implementation that would need to be enhanced` | Phase C (compression): Audit doc/code; replace with exact specification |
| 21 | `src/compression/dict_zip/dfa_cache.rs:452` | `for now` | `// during construction. For now, return None to indicate fallback to suffix array` | Phase C (compression): Remove temporary fallback; verify production invariant |
| 22 | `src/compression/dict_zip/dfa_cache.rs:475` | `for now` | `// For now, return None as zstr compression is not fully implemented` | Phase C (compression): Remove temporary fallback; verify production invariant |
| 23 | `src/compression/dict_zip/dfa_cache.rs:481` | `simplified` | `/// Deserialize cache from external storage (simplified version)` | Phase C (compression): Audit doc/code; replace with exact specification |
| 24 | `src/compression/dict_zip/local_matcher.rs:897` | `simplified` | `// This is a simplified cleanup - in practice, we might want more sophisticated` | Phase C (compression): Audit doc/code; replace with exact specification |
| 25 | `src/compression/dict_zip/reference_encoding.rs:361` | `simplified` | `/// Reference C++ (simplified version):` | Phase C (compression): Audit doc/code; replace with exact specification |
| 26 | `src/compression/dict_zip/reference_encoding.rs:1038` | `for now` | `/// This is a simplified version for now - the reference implementation` | Phase C (compression): Remove temporary fallback; verify production invariant |
| 27 | `src/string/bmi2_string_ops.rs:646` | `simplified` | `// Simplified wildcard matching with BMI2 acceleration` | Phase C (string): Audit doc/code; replace with exact specification |
| 28 | `src/string/fast_str.rs:300` | `simplified` | `// Process 64 bytes at a time using AVX-512 (simplified implementation)` | Phase C (string): Audit doc/code; replace with exact specification |
| 29 | `src/string/simd_search.rs:698` | `for now` | `// For now, fall back to individual searches for simplicity` | Phase C (string): Remove temporary fallback; verify production invariant |
| 30 | `src/string/unicode.rs:165` | `placeholder` | `/// Basic normalization (placeholder for full Unicode normalization)` | Phase C (string): Replace placeholder with real structure or clean up |
| 31 | `src/string/unicode.rs:167` | `simplified` | `// This is a simplified normalization - a full implementation would` | Phase C (string): Audit doc/code; replace with exact specification |
| 32 | `src/string/unicode.rs:395` | `simplified` | `// Simplified implementation - could use unicode-width crate for full support` | Phase C (string): Audit doc/code; replace with exact specification |
| 33 | `src/string/unicode.rs:412` | `simplified` | `// Simplified check for wide characters` | Phase C (string): Audit doc/code; replace with exact specification |
| 34 | `src/string/unicode.rs:415` | `simplified` | `// CJK ranges (simplified)` | Phase C (string): Audit doc/code; replace with exact specification |
| 35 | `src/memory/cache.rs:432` | `simplified` | `// Use thread ID hash for consistent assignment - simplified hash since as_u64 is unstable` | Phase C (memory): Audit doc/code; replace with exact specification |
| 36 | `src/memory/cache.rs:449` | `for now` | `// For now, this is a no-op as we don't want to depend on libnuma` | Phase C (memory): Remove temporary fallback; verify production invariant |
| 37 | `src/memory/cache_layout.rs:423` | `simplified` | `// This is a simplified reorganization - in practice, you'd track` | Phase C (memory): Audit doc/code; replace with exact specification |
| 38 | `src/memory/five_level_pool.rs:360` | `for now` | `// For now, fall back to end allocation` | Phase C (memory): Remove temporary fallback; verify production invariant |
| 39 | `src/memory/five_level_pool.rs:361` | `TODO: implement` | `// TODO: Implement full skip list search` | Phase C (memory): Implement, return Err(unsupported), or delete |
| 40 | `src/memory/five_level_pool.rs:387` | `TODO: implement` | `// TODO: Implement skip list insertion` | Phase C (memory): Implement, return Err(unsupported), or delete |
| 41 | `src/memory/five_level_pool.rs:388` | `for now` | `// For now, just track statistics` | Phase C (memory): Remove temporary fallback; verify production invariant |
| 42 | `src/memory/five_level_pool.rs:527` | `for now` | `// For now, allocate from end` | Phase C (memory): Remove temporary fallback; verify production invariant |
| 43 | `src/memory/lockfree_pool.rs:218` | `simplified` | `// Simplified calculation - in practice would track time` | Phase C (memory): Audit doc/code; replace with exact specification |
| 44 | `src/memory/pool.rs:373` | `for now` | `// For now, we just validate the parameters` | Phase C (memory): Remove temporary fallback; verify production invariant |
| 45 | `src/memory/secure_pool.rs:987` | `simplified` | `// Hot/cold separation temporarily simplified` | Phase C (memory): Audit doc/code; replace with exact specification |
| 46 | `src/memory/secure_pool.rs:1171` | `for now` | `let optimal_node = -1; // Simplified: disable NUMA for now` | Phase C (memory): Remove temporary fallback; verify production invariant |
| 47 | `src/memory/secure_pool.rs:1194` | `for now` | `// For now, fall through to regular allocation` | Phase C (memory): Remove temporary fallback; verify production invariant |
| 48 | `src/memory/threadlocal_pool.rs:446` | `for now` | `// For now, just leak (in real implementation, would track)` | Phase C (memory): Remove temporary fallback; verify production invariant |
| 49 | `src/dev_infrastructure/profiling.rs:1` | `stub` | `//! Profiling integration - minimal stub.` | Phase C (dev_infrastructure): Implement or delete stub |
| 50 | `src/dev_infrastructure/statistics.rs:529` | `simplified` | `// This is a simplified version returning cached correlation if available` | Phase C (dev_infrastructure): Audit doc/code; replace with exact specification |
| 51 | `src/config/blob_store.rs:7` | `placeholder` | `/// Blob store configuration placeholder.` | Phase C (config): Replace placeholder with real structure or clean up |
| 52 | `src/config/cache.rs:7` | `placeholder` | `/// Cache configuration placeholder.` | Phase C (config): Replace placeholder with real structure or clean up |
| 53 | `src/config/compression.rs:7` | `placeholder` | `/// Compression algorithm configuration placeholder.` | Phase C (config): Replace placeholder with real structure or clean up |
| 54 | `src/config/simd.rs:7` | `placeholder` | `/// SIMD configuration placeholder.` | Phase C (config): Replace placeholder with real structure or clean up |
| 55 | `src/algorithms/cache_oblivious.rs:651` | `simplified` | `/// Recursive Van Emde Boas layout calculation (simplified implementation)` | Phase C (algorithms): Audit doc/code; replace with exact specification |
| 56 | `src/algorithms/multiway_merge.rs:299` | `for now` | `// For simplicity, fall back to direct heap merge for now` | Phase C (algorithms): Remove temporary fallback; verify production invariant |
| 57 | `src/algorithms/multiway_merge.rs:318` | `simplified` | `type Input = Vec<Vec<i32>>; // Simplified input type for the trait` | Phase C (algorithms): Audit doc/code; replace with exact specification |
| 58 | `src/algorithms/simd_merge.rs:426` | `for now` | `// For now, use scalar comparison - full SIMD merge is complex` | Phase C (algorithms): Remove temporary fallback; verify production invariant |
| 59 | `src/algorithms/suffix_array.rs:452` | `for now` | `// For now, use a simple sorting approach since the full DC3 is complex` | Phase C (algorithms): Remove temporary fallback; verify production invariant |
| 60 | `src/algorithms/suffix_array.rs:499` | `for now` | `// For now, use standard suffix comparison to ensure correctness` | Phase C (algorithms): Remove temporary fallback; verify production invariant |
| 61 | `src/algorithms/suffix_array.rs:512` | `for now` | `// For now, fall back to sequential - full parallel SA-IS is very complex` | Phase C (algorithms): Remove temporary fallback; verify production invariant |
| 62 | `src/algorithms/tournament_tree.rs:292` | `for now` | `// For now, delegate to the regular comparator` | Phase C (algorithms): Remove temporary fallback; verify production invariant |
| 63 | `src/algorithms/radix_sort/advanced.rs:347` | `simplified` | `/// Tim sort for nearly sorted data (simplified implementation)` | Phase C (algorithms): Audit doc/code; replace with exact specification |
| 64 | `src/algorithms/radix_sort/advanced.rs:349` | `simplified` | `// This is a simplified version - a full Tim sort implementation would be much more complex` | Phase C (algorithms): Audit doc/code; replace with exact specification |
| 65 | `src/algorithms/radix_sort/advanced.rs:350` | `for now` | `// For now, we use the standard library's unstable_sort which is based on pattern-defeating quicksort` | Phase C (algorithms): Remove temporary fallback; verify production invariant |
| 66 | `src/algorithms/radix_sort/advanced.rs:382` | `for now` | `// For now, fall back to regular allocation` | Phase C (algorithms): Remove temporary fallback; verify production invariant |
| 67 | `src/algorithms/radix_sort/advanced.rs:383` | `TODO: implement` | `// TODO: Implement proper memory pool integration for generic types` | Phase C (algorithms): Implement, return Err(unsupported), or delete |
| 68 | `src/algorithms/radix_sort/advanced.rs:628` | `for now` | `// For now, use a simple approach: collect all elements and sort` | Phase C (algorithms): Remove temporary fallback; verify production invariant |
| 69 | `src/algorithms/radix_sort/advanced.rs:630` | `TODO: implement` | `// TODO: Implement proper multi-way merge when the generic bounds are resolved` | Phase C (algorithms): Implement, return Err(unsupported), or delete |
| 70 | `src/algorithms/radix_sort/lsd.rs:234` | `simplified` | `// This is a simplified version - full parallel radix sort is quite complex` | Phase C (algorithms): Audit doc/code; replace with exact specification |
| 71 | `src/algorithms/radix_sort/lsd.rs:504` | `for now` | `// For now, extract to array and count sequentially` | Phase C (algorithms): Remove temporary fallback; verify production invariant |
| 72 | `src/concurrency/parallel_trie.rs:90` | `for now` | `// For now, use a simple approach: extract all keys and rebuild` | Phase C (concurrency): Remove temporary fallback; verify production invariant |
| 73 | `src/concurrency/parallel_trie.rs:250` | `simplified` | `// Collect results from iterator (simplified)` | Phase C (concurrency): Audit doc/code; replace with exact specification |
| 74 | `src/concurrency/work_stealing.rs:557` | `for now` | `// For now, fall back to tokio spawn` | Phase C (concurrency): Remove temporary fallback; verify production invariant |
| 75 | `src/system/cpu_features.rs:609` | `simplified` | `// Cache level determination is simplified for compatibility` | Phase C (system): Audit doc/code; replace with exact specification |
| 76 | `src/system/cpu_features.rs:634` | `simplified` | `// Cache size detection is simplified for initial implementation` | Phase C (system): Audit doc/code; replace with exact specification |
| 77 | `src/system/cpu_features.rs:652` | `simplified` | `// This is a simplified version - in practice you'd use getauxval(AT_HWCAP)` | Phase C (system): Audit doc/code; replace with exact specification |
| 78 | `src/system/cpu_features.rs:653` | `for now` | `// For now, return a default that indicates we couldn't detect` | Phase C (system): Remove temporary fallback; verify production invariant |
| 79 | `src/entropy/fse.rs:740` | `simplified` | `/// Renormalize state for decoding (simplified approach)` | Phase C (entropy): Audit doc/code; replace with exact specification |
| 80 | `src/entropy/mod.rs:234` | `for now` | `// constructor signatures for each algorithm. For now, use the individual encoders directly.` | Phase C (entropy): Remove temporary fallback; verify production invariant |
| 81 | `src/entropy/simd_huffman.rs:317` | `for now` | `// Pack bits into u32 (simple approach for now)` | Phase C (entropy): Remove temporary fallback; verify production invariant |
| 82 | `src/entropy/simd_huffman.rs:487` | `for now` | `// Pack bits into u64 (simple approach for now)` | Phase C (entropy): Remove temporary fallback; verify production invariant |
| 83 | `src/entropy/huffman/interleaved.rs:907` | `placeholder` | `// old placeholder wrote a bogus 1-bit code, which the` | Phase C (entropy): Replace placeholder with real structure or clean up |
| 84 | `src/entropy/huffman/tree.rs:375` | `placeholder` | `// This is a placeholder leaf, convert to internal node` | Phase C (entropy): Replace placeholder with real structure or clean up |
| 85 | `src/entropy/huffman/tree.rs:380` | `placeholder` | `// Final bit, create leaf and keep placeholder` | Phase C (entropy): Replace placeholder with real structure or clean up |
| 86 | `src/entropy/huffman/tree.rs:385` | `placeholder` | `let placeholder = HuffmanNode::Leaf {` | Phase C (entropy): Replace placeholder with real structure or clean up |
| 87 | `src/entropy/huffman/tree.rs:393` | `placeholder` | `left: Box::new(placeholder),` | Phase C (entropy): Replace placeholder with real structure or clean up |
| 88 | `src/entropy/huffman/tree.rs:400` | `placeholder` | `right: Box::new(placeholder),` | Phase C (entropy): Replace placeholder with real structure or clean up |
| 89 | `src/entropy/huffman/tree.rs:405` | `placeholder` | `let placeholder = HuffmanNode::Leaf {` | Phase C (entropy): Replace placeholder with real structure or clean up |
| 90 | `src/entropy/huffman/tree.rs:411` | `placeholder` | `// Create internal node with placeholder on left, continue on right` | Phase C (entropy): Replace placeholder with real structure or clean up |
| 91 | `src/entropy/huffman/tree.rs:420` | `placeholder` | `left: Box::new(placeholder),` | Phase C (entropy): Replace placeholder with real structure or clean up |
| 92 | `src/entropy/huffman/tree.rs:424` | `placeholder` | `// Create internal node with placeholder on right, continue on left` | Phase C (entropy): Replace placeholder with real structure or clean up |
| 93 | `src/entropy/huffman/tree.rs:434` | `placeholder` | `right: Box::new(placeholder),` | Phase C (entropy): Replace placeholder with real structure or clean up |
| 94 | `src/cache/basic_cache.rs:117` | `placeholder` | `// This is a placeholder for future implementation` | Phase C (cache): Replace placeholder with real structure or clean up |
| 95 | `src/cache/basic_cache.rs:237` | `for now` | `// For now, just mark it as clean` | Phase C (cache): Remove temporary fallback; verify production invariant |
| 96 | `src/cache/basic_cache.rs:294` | `simplified` | `// Start prefetching in the background (simplified - would use async in real implementation)` | Phase C (cache): Audit doc/code; replace with exact specification |
| 97 | `src/cache/basic_cache.rs:372` | `for now` | `// For now, just mark it as clean` | Phase C (cache): Remove temporary fallback; verify production invariant |
| 98 | `src/cache/buffer.rs:73` | `simplified` | `// Simplified for basic implementation` | Phase C (cache): Audit doc/code; replace with exact specification |
| 99 | `src/cache/buffer.rs:145` | `simplified` | `// Simplified for basic implementation` | Phase C (cache): Audit doc/code; replace with exact specification |
| 100 | `src/fsa/cspp_trie.rs:776` | `placeholder` | `(*p.add(1)).child = NIL_STATE; // placeholder, filled by next iteration` | Phase C (fsa): Replace placeholder with real structure or clean up |
| 101 | `src/fsa/simple_implementations.rs:1` | `simplified` | `//! Simplified FSA infrastructure implementations for initial Phase 8A completion` | Phase C (fsa): Audit doc/code; replace with exact specification |
| 102 | `src/fsa/strategy_traits.rs:451` | `TODO: implement` | `// TODO: Implement optimization (path compression, node merging, etc.)` | Phase C (fsa): Implement, return Err(unsupported), or delete |
| 103 | `src/fsa/strategy_traits.rs:521` | `TODO: implement` | `// TODO: Implement path splitting for partial matches` | Phase C (fsa): Implement, return Err(unsupported), or delete |
| 104 | `src/fsa/version_sync.rs:545` | `simplified` | `/// This is a simplified version - in a full implementation, this would` | Phase C (fsa): Audit doc/code; replace with exact specification |
| 105 | `src/fsa/zipora_trie/map.rs:94` | `for now` | `// For now, we need to traverse to find the state ID` | Phase C (fsa): Remove temporary fallback; verify production invariant |
| 106 | `src/fsa/zipora_trie/trie.rs:203` | `TODO: implement` | `TrieStorage::Louds { .. } => 1, // TODO: implement for LOUDS` | Phase C (fsa): Implement, return Err(unsupported), or delete |
| 107 | `src/fsa/zipora_trie/trie.rs:211` | `TODO: implement` | `TrieStorage::CriticalBit { .. } => 0, // TODO: implement` | Phase C (fsa): Implement, return Err(unsupported), or delete |
| 108 | `src/fsa/zipora_trie/trie.rs:214` | `TODO: implement` | `TrieStorage::Louds { .. } => 0, // TODO: implement` | Phase C (fsa): Implement, return Err(unsupported), or delete |
| 109 | `src/fsa/zipora_trie/trie.rs:215` | `TODO: implement` | `TrieStorage::CompressedSparse(_cspp) => 0, /* TODO: implement num_transitions */` | Phase C (fsa): Implement, return Err(unsupported), or delete |
| 110 | `src/fsa/zipora_trie/trie.rs:368` | `TODO: implement` | `// TODO: Implement for other storage types` | Phase C (fsa): Implement, return Err(unsupported), or delete |
| 111 | `src/fsa/zipora_trie/trie.rs:391` | `TODO: implement` | `// TODO: Implement for other storage types` | Phase C (fsa): Implement, return Err(unsupported), or delete |
| 112 | `src/fsa/zipora_trie/trie.rs:963` | `TODO: implement` | `// TODO: Implement LOUDS final state check` | Phase C (fsa): Implement, return Err(unsupported), or delete |
| 113 | `src/fsa/zipora_trie/trie.rs:966` | `stub` | `TrieStorage::CompressedSparse(_cspp) => false, // Stub for legacy method` | Phase C (fsa): Implement or delete stub |
| 114 | `src/fsa/zipora_trie/trie.rs:980` | `TODO: implement` | `// TODO: Implement critical bit transition` | Phase C (fsa): Implement, return Err(unsupported), or delete |
| 115 | `src/fsa/zipora_trie/trie.rs:1001` | `TODO: implement` | `// TODO: Implement LOUDS transition` | Phase C (fsa): Implement, return Err(unsupported), or delete |
| 116 | `src/fsa/zipora_trie/trie.rs:1004` | `stub` | `TrieStorage::CompressedSparse(_cspp) => None, // Stub for legacy method` | Phase C (fsa): Implement or delete stub |
| 117 | `src/containers/mod.rs:30` | `simplified` | `//! - **`EasyHashMap<K,V>`** - Simplified hash map interface with builder pattern` | Phase C (containers): Audit doc/code; replace with exact specification |
| 118 | `src/containers/fast_vec/mod.rs:33` | `for now` | `// Use const traits when available, for now rely on Copy bound in caller` | Phase C (containers): Remove temporary fallback; verify production invariant |
| 119 | `src/containers/specialized/concurrent_lru_map.rs:442` | `for now` | `// For now, this is a placeholder showing the interface` | Phase C (containers): Remove temporary fallback; verify production invariant |
| 120 | `src/containers/specialized/concurrent_lru_map.rs:459` | `for now` | `// For now, this is a placeholder` | Phase C (containers): Remove temporary fallback; verify production invariant |
| 121 | `src/containers/specialized/easy_hash_map.rs:1` | `simplified` | `//! Simplified Hash Map Interface` | Phase C (containers): Audit doc/code; replace with exact specification |
| 122 | `src/containers/specialized/easy_hash_map.rs:3` | `simplified` | `//! A simplified interface wrapper around ZiporaHashMap that provides` | Phase C (containers): Audit doc/code; replace with exact specification |
| 123 | `src/containers/specialized/easy_hash_map.rs:13` | `simplified` | `/// Simplified hash map with convenient APIs` | Phase C (containers): Audit doc/code; replace with exact specification |
| 124 | `src/containers/specialized/easy_hash_map.rs:21` | `simplified` | `/// - **Convenience**: Simplified APIs for common use cases` | Phase C (containers): Audit doc/code; replace with exact specification |
| 125 | `src/containers/specialized/easy_hash_map.rs:90` | `simplified` | `/// Insert a key-value pair (simplified interface)` | Phase C (containers): Audit doc/code; replace with exact specification |
| 126 | `src/containers/specialized/easy_hash_map.rs:232` | `TODO: implement` | `// /// TODO: Implement when ZiporaHashMap has iter() support` | Phase C (containers): Implement, return Err(unsupported), or delete |
| 127 | `src/containers/specialized/easy_hash_map.rs:238` | `TODO: implement` | `// /// TODO: Implement when ZiporaHashMap has keys() support` | Phase C (containers): Implement, return Err(unsupported), or delete |
| 128 | `src/containers/specialized/easy_hash_map.rs:244` | `TODO: implement` | `// /// TODO: Implement when ZiporaHashMap has values() support` | Phase C (containers): Implement, return Err(unsupported), or delete |
| 129 | `src/containers/specialized/easy_hash_map.rs:250` | `TODO: implement` | `// /// TODO: Implement when ZiporaHashMap has iterator support` | Phase C (containers): Implement, return Err(unsupported), or delete |
| 130 | `src/containers/specialized/fixed_len_str_vec.rs:274` | `for now` | `// For now, fallback to optimized version` | Phase C (containers): Remove temporary fallback; verify production invariant |
| 131 | `src/containers/specialized/fixed_len_str_vec.rs:275` | `TODO: implement` | `// TODO: Implement SIMD string comparison` | Phase C (containers): Implement, return Err(unsupported), or delete |
| 132 | `src/containers/specialized/fixed_len_str_vec.rs:281` | `for now` | `// For now, fallback to optimized version` | Phase C (containers): Remove temporary fallback; verify production invariant |
| 133 | `src/containers/specialized/fixed_len_str_vec.rs:282` | `TODO: implement` | `// TODO: Implement SIMD prefix matching` | Phase C (containers): Implement, return Err(unsupported), or delete |
| 134 | `src/containers/specialized/hash_str_map.rs:1` | `simplified` | `//! Simplified String-Optimized Hash Map` | Phase C (containers): Audit doc/code; replace with exact specification |
| 135 | `src/containers/specialized/hash_str_map.rs:10` | `simplified` | `/// Simplified string-optimized hash map` | Phase C (containers): Audit doc/code; replace with exact specification |
| 136 | `src/containers/specialized/hash_str_map.rs:12` | `simplified` | `/// This is a simplified version that provides basic string interning` | Phase C (containers): Audit doc/code; replace with exact specification |
| 137 | `src/containers/specialized/mod.rs:25` | `simplified` | `//! - **`EasyHashMap<K,V>`** - Simplified hash map interface with builder pattern` | Phase C (containers): Audit doc/code; replace with exact specification |
| 138 | `src/containers/specialized/sortable_str_vec.rs:263` | `simplified` | `// Simplified sequence ID (faster than atomic ops for each string)` | Phase C (containers): Audit doc/code; replace with exact specification |
| 139 | `src/containers/specialized/valvec32.rs:186` | `for now` | `// For now, ignore the pool parameter and use standard allocation` | Phase C (containers): Remove temporary fallback; verify production invariant |
| 140 | `src/containers/specialized/valvec32.rs:284` | `simplified` | `/// Grows the vector to the specified capacity - simplified following referenced project pattern` | Phase C (containers): Audit doc/code; replace with exact specification |
| 141 | `src/containers/specialized/vec_trb.rs:424` | `for now` | `// -- remove: simple splice (no black-height fixup for now) --` | Phase C (containers): Remove temporary fallback; verify production invariant |
| 142 | `src/containers/specialized/zo_sorted_str_vec.rs:382` | `TODO: implement` | `// TODO: Implement memory-mapped format loading` | Phase C (containers): Implement, return Err(unsupported), or delete |
| 143 | `src/containers/specialized/zo_sorted_str_vec.rs:397` | `TODO: implement` | `// TODO: Implement binary format serialization` | Phase C (containers): Implement, return Err(unsupported), or delete |
| 144 | `src/hash_map/cache_locality.rs:298` | `for now` | `// For now, return 1 (UMA system)` | Phase C (hash_map): Remove temporary fallback; verify production invariant |
| 145 | `src/hash_map/cache_locality.rs:318` | `for now` | `// For now, use standard aligned allocation` | Phase C (hash_map): Remove temporary fallback; verify production invariant |
| 146 | `src/hash_map/storage.rs:8` | `placeholder` | `#[allow(dead_code)] // strategy-placeholder variants, matched exhaustively in 9 arms` | Phase C (hash_map): Replace placeholder with real structure or clean up |
| 147 | `src/hash_map/strategy_traits.rs:627` | `TODO: implement` | `// TODO: Implement cache optimization` | Phase C (hash_map): Implement, return Err(unsupported), or delete |
| 148 | `src/hash_map/strategy_traits.rs:677` | `TODO: implement` | `// TODO: Implement SIMD-accelerated pre-insert optimizations` | Phase C (hash_map): Implement, return Err(unsupported), or delete |
| 149 | `src/hash_map/strategy_traits.rs:688` | `TODO: implement` | `// TODO: Implement post-insert optimizations` | Phase C (hash_map): Implement, return Err(unsupported), or delete |
| 150 | `src/hash_map/strategy_traits.rs:702` | `TODO: implement` | `// TODO: Implement SIMD-accelerated lookup optimizations` | Phase C (hash_map): Implement, return Err(unsupported), or delete |
| 151 | `src/hash_map/strategy_traits.rs:713` | `TODO: implement` | `// TODO: Implement bulk SIMD optimizations` | Phase C (hash_map): Implement, return Err(unsupported), or delete |
| 152 | `src/succinct/rank_select/adaptive.rs:510` | `simplified` | `/// Select optimal implementation based on data profile (simplified to use best performer)` | Phase C (succinct): Audit doc/code; replace with exact specification |
| 153 | `src/succinct/rank_select/bmi2_acceleration.rs:252` | `for now` | `// For now, extract and use scalar popcount` | Phase C (succinct): Remove temporary fallback; verify production invariant |
| 154 | `src/succinct/rank_select/builder.rs:28` | `TODO: implement` | `// TODO: Implement analysis logic` | Phase C (succinct): Implement, return Err(unsupported), or delete |
| 155 | `src/succinct/rank_select/interleaved.rs:1047` | `simplified` | `// Simplified adaptive rank - just use the best implementation directly` | Phase C (succinct): Audit doc/code; replace with exact specification |
| 156 | `src/succinct/rank_select/interleaved.rs:1055` | `simplified` | `// Simplified adaptive select - just use the best implementation directly` | Phase C (succinct): Audit doc/code; replace with exact specification |
| 157 | `src/succinct/rank_select/mod.rs:460` | `for now` | `// For now, return placeholder data` | Phase C (succinct): Remove temporary fallback; verify production invariant |
| 158 | `src/succinct/rank_select/simd.rs:545` | `for now` | `// For now, delegate to scalar with POPCNT acceleration` | Phase C (succinct): Remove temporary fallback; verify production invariant |
| 159 | `src/blob_store/cached_store.rs:59` | `for now` | `// Register with cache (use dummy file descriptor for now)` | Phase C (blob_store): Remove temporary fallback; verify production invariant |
| 160 | `src/blob_store/cached_store.rs:164` | `for now` | `// For now, this marks the operation as successful` | Phase C (blob_store): Remove temporary fallback; verify production invariant |
| 161 | `src/blob_store/cached_store.rs:198` | `simplified` | `// Return (invalidated_count, dirty_count) - simplified implementation` | Phase C (blob_store): Audit doc/code; replace with exact specification |
| 162 | `src/blob_store/entropy.rs:147` | `for now` | `// For now, delegate to inner store (would need metadata for decompression)` | Phase C (blob_store): Remove temporary fallback; verify production invariant |
| 163 | `src/blob_store/entropy.rs:236` | `for now` | `// For now, delegate to inner store (would need full implementation)` | Phase C (blob_store): Remove temporary fallback; verify production invariant |
| 164 | `src/blob_store/entropy.rs:305` | `for now` | `// For now, delegate to inner store (would need full implementation)` | Phase C (blob_store): Remove temporary fallback; verify production invariant |
| 165 | `src/blob_store/nest_louds_trie_blob_store.rs:697` | `for now` | `// Fall through to trie lookup for now - cache will be optimized later` | Phase C (blob_store): Remove temporary fallback; verify production invariant |
| 166 | `src/blob_store/nest_louds_trie_blob_store.rs:1356` | `for now` | `// This could trigger trie optimization, but for now we'll leave it as-is` | Phase C (blob_store): Remove temporary fallback; verify production invariant |
| 167 | `src/blob_store/zero_length.rs:9` | `placeholder` | `//! - Placeholder records where only existence matters` | Phase C (blob_store): Replace placeholder with real structure or clean up |
| 168 | `src/blob_store/zip_offset_builder.rs:375` | `for now` | `// For now, just process the entire buffer as one record` | Phase C (blob_store): Remove temporary fallback; verify production invariant |
| 169 | `src/blob_store/zip_offset_builder.rs:376` | `TODO: implement` | `// TODO: Implement proper record separation` | Phase C (blob_store): Implement, return Err(unsupported), or delete |
| 170 | `src/io/complex_types.rs:75` | `for now` | `"tuple" // Simplified for now` | Phase C (io): Remove temporary fallback; verify production invariant |
| 171 | `src/io/mmap.rs:756` | `placeholder` | `// 3. This is used as a placeholder during mem::replace() to allow dropping the old mmap` | Phase C (io): Replace placeholder with real structure or clean up |
| 172 | `src/io/mmap.rs:799` | `placeholder` | `// 3. This is used as a placeholder during mem::replace() to allow dropping the old mmap` | Phase C (io): Replace placeholder with real structure or clean up |
| 173 | `src/io/mmap.rs:861` | `stub` | `// Provide stub implementations when mmap feature is disabled` | Phase C (io): Implement or delete stub |
| 174 | `src/io/var_int_variants.rs:618` | `simplified` | `// Prefix-free implementations (simplified)` | Phase C (io): Audit doc/code; replace with exact specification |
| 175 | `src/io/var_int_variants.rs:741` | `simplified` | `// Compact and SIMD implementations (simplified)` | Phase C (io): Audit doc/code; replace with exact specification |
| 176 | `src/io/var_int_variants.rs:777` | `for now` | `// For now, use LEB128 with potential for SIMD optimization` | Phase C (io): Remove temporary fallback; verify production invariant |
| 177 | `src/io/var_int_variants.rs:782` | `for now` | `// For now, use zigzag with potential for SIMD optimization` | Phase C (io): Remove temporary fallback; verify production invariant |
| 178 | `src/io/simd_memory/search.rs:592` | `placeholder` | `// ARM NEON Implementation (Placeholder)` | Phase C (io): Replace placeholder with real structure or clean up |
| 179 | `src/io/simd_memory/search.rs:597` | `TODO: implement` | `// TODO: Implement NEON-optimized character search` | Phase C (io): Implement, return Err(unsupported), or delete |
| 180 | `src/io/simd_memory/search.rs:603` | `TODO: implement` | `// TODO: Implement NEON-optimized pattern search` | Phase C (io): Implement, return Err(unsupported), or delete |
| 181 | `src/io/simd_memory/search.rs:609` | `TODO: implement` | `// TODO: Implement NEON-optimized multi-character search` | Phase C (io): Implement, return Err(unsupported), or delete |
| 182 | `src/io/simd_memory/search.rs:615` | `TODO: implement` | `// TODO: Implement NEON-optimized string comparison` | Phase C (io): Implement, return Err(unsupported), or delete |
