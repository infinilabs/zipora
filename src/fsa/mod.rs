//! High-performance Finite State Automata and Trie implementation
//!
//! **ZiporaTrie**: Single, highly optimized trie implementation with strategy-based
//! configuration. One Patricia class with concurrency level as a config enum,
//! not separate wrapper types.
//!
//! # Examples
//!
//! ```rust
//! use zipora::fsa::{ZiporaTrie, ZiporaTrieConfig};
//! use zipora::fsa::traits::Trie;
//! use zipora::succinct::RankSelectInterleaved256;
//!
//! let mut trie: ZiporaTrie<RankSelectInterleaved256> = ZiporaTrie::new();
//! trie.insert(b"hello").unwrap();
//! assert!(trie.contains(b"hello"));
//! ```

// Core implementation modules
pub mod cspp_trie;
pub mod cspp_trie_concurrent;
pub mod double_array;
pub mod strategy_traits;
pub mod zipora_trie;

// Core infrastructure modules
pub mod cache;
pub mod dawg;
pub mod fast_search;
pub mod graph_walker;
pub mod simple_implementations;
pub mod token;
pub mod traits;
pub mod version_sync;

// Core ZiporaTrie implementation
pub use zipora_trie::{
    BitVectorType, TrieCompressionStrategy, RankSelectType, TrieStorageStrategy, TrieStrategy, ZiporaTrie,
    ZiporaTrieConfig, ZiporaTrieMap,
};

// Strategy traits for advanced configuration
pub use strategy_traits::{
    AlgorithmStats, CompressionStats, TrieCompressionStrategy as CompressionStrategyTrait,
    ConcurrencyStats, ConcurrencyStrategy, NoOpToken, PathCompressionConfig,
    PathCompressionContext, PathCompressionStrategy, PatriciaAlgorithmStrategy, PatriciaConfig,
    PatriciaContext, PatriciaNode, SingleThreadedConcurrencyStrategy, SingleThreadedConfig,
    SingleThreadedContext, StorageEfficiency, TrieAlgorithmStrategy,
};

// Core infrastructure
pub use cache::{
    CacheStrategy, CachedState, FastStateCache, FsaCache, FsaCacheConfig, FsaCacheStats,
    ZeroPathData,
};
pub use dawg::{
    DawgConfig, DawgState, DawgStats, NestedTrieDawg, TerminalStrategy, TransitionTable,
};
pub use fast_search::{
    FastSearchConfig, FastSearchEngine, HardwareCapabilities, SearchStrategy, binary_search_byte,
    fast_search_byte, fast_search_byte_max_16,
};
pub use graph_walker::{
    BfsGraphWalker, CfsGraphWalker, DfsGraphWalker, FastBfsWalker, FastCfsWalker, FastDfsWalker,
    GraphVisitor, GraphWalker, GraphWalkerFactory, MultiPassWalker, SimpleVertex, Vertex,
    VertexColor, WalkMethod, WalkStats, WalkerConfig,
};
pub use simple_implementations::{SimpleDawg, SimpleFastSearch, SimpleFsaCache, SimpleGraphWalker};
pub use token::{
    GlobalTokenStats, TokenCache, TokenCacheStats, TokenManager, with_reader_token,
    with_writer_token,
};
pub use traits::{
    FiniteStateAutomaton, FsaError, PrefixIterable, StatisticsProvider, Trie, TrieStats,
};
pub use version_sync::{
    LazyFreeItem, LazyFreeList, LazyFreeStats, ReaderToken as VersionReaderToken, VersionManager,
    VersionManagerStats, WriterToken as VersionWriterToken,
};

pub use double_array::{
    DoubleArrayTrie, DoubleArrayTrieCursor, DoubleArrayTrieMap, DoubleArrayTrieMapCursor, MapRangeIter,
    MapValue, RangeIter,
};
