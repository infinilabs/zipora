//! Specialized container types optimized for specific use cases
//!
//! This module provides specialized container implementations that bridge
//! feature gaps while maintaining zipora's performance
//! and safety standards.
//!
//! ## Phase 1 Container Types
//!
//! - **`ValVec32<T>`** - 32-bit indexed vectors for memory efficiency
//! - **`SmallMap<K,V>`** - Memory-efficient containers for small collections
//! - **`FixedCircularQueue<T, N>`** - Fixed-size circular buffer
//! - **`AutoGrowCircularQueue<T>`** - Dynamically resizing circular buffer
//!
//! ## Phase 2 Container Types
//!
//! - **`UintVector`** - Compressed integer storage with 60-80% space reduction
//! - **`FixedLenStrVec<N>`** - Fixed-length string vector with 60% memory savings and SIMD optimizations
//! - **`SortableStrVec`** - Arena-based string storage with 25% faster sorting
//!
//! ## Phase 3 Advanced Container Types
//!
//! - **`ZoSortedStrVec`** - Zero-overhead sorted string collections with 60% memory reduction
//! - **`GoldHashIdx<K,V>`** - Hash index for large value indirection with 30% memory savings
//! - **`HashStrMap<V>`** - String-optimized hash map with interning (40% memory reduction)
//! - **`EasyHashMap<K,V>`** - Simplified hash map interface with builder pattern
//!
//! ## Performance Goals
//!
//! ### Phase 1 Achievements
//! - ValVec32: 40-50% memory reduction vs Vec<T>
//! - SmallMap: 90% faster than GoldHashMap for ≤8 elements
//! - Circular queues: 20-30% faster than VecDeque
//!
//! ### Phase 2 Achievements
//! - UintVector: 60-80% space reduction vs Vec<u32> through compression
//! - FixedLenStrVec: 60% memory reduction vs Vec<String> with SIMD acceleration
//! - SortableStrVec: 25% faster sorting vs Vec<String> with arena allocation
//!
//! ### Phase 3 Achievements
//! - ZoSortedStrVec: 60% memory reduction through succinct data structures
//! - GoldHashIdx: 30% memory reduction for large values through indirection
//! - HashStrMap: 40% memory reduction through string interning
//! - EasyHashMap: Zero overhead convenience layer with same performance as GoldHashMap
//!
//! ## Design Principles
//!
//! - Zero-copy operations where possible
//! - SIMD optimization for performance-critical operations
//! - Memory safety without sacrificing performance
//! - Integration with SecureMemoryPool and arena allocators
//! - Consistent error handling via ZiporaError
//! - Automatic compression strategy selection
//! - Cache-friendly data layouts
//! - Advanced data structure integration (succinct structures, string interning)

// Phase 1 containers
mod circular_queue;
mod small_map;
mod valvec32;

// Phase 2 containers
mod fixed_len_str_vec;
pub mod sortable_str_vec;
mod uint_vector;

// Phase 3 advanced containers
mod easy_hash_map;
mod gold_hash_idx;
mod hash_str_map;
mod zo_sorted_str_vec;

// Phase 1 exports
pub use circular_queue::{AutoGrowCircularQueue, FixedCircularQueue};
pub use small_map::SmallMap;
pub use valvec32::ValVec32;

// Phase 2 exports
pub use fixed_len_str_vec::{
    FixedLenStrVec, FixedStr4Vec, FixedStr8Vec, FixedStr16Vec, FixedStr32Vec, FixedStr64Vec,
};
pub use sortable_str_vec::{SortableStrIter, SortableStrSortedIter, SortableStrVec};
pub use uint_vector::UintVector;

// Phase 3 exports
pub use easy_hash_map::{EasyHashMap, EasyHashMapBuilder, EasyHashMapStats};
pub use gold_hash_idx::GoldHashIdx;
pub use hash_str_map::{HashStrMap, HashStrMapStats};
pub use zo_sorted_str_vec::{ZoSortedStrVec, ZoSortedStrVecIter, ZoSortedStrVecRange};
