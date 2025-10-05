use zipora::fsa::{ZiporaTrie, ZiporaTrieConfig, TrieStrategy, StorageStrategy, CompressionStrategy, RankSelectType, FiniteStateAutomaton, Trie};
use zipora::succinct::RankSelectInterleaved256;

fn main() {
    // Create ZiporaTrie with DoubleArray strategy for compatibility
    let config = ZiporaTrieConfig {
        trie_strategy: TrieStrategy::DoubleArray {
            initial_capacity: 256,
            growth_factor: 1.5,
            free_list_management: true,
            auto_shrink: false,
        },
        storage_strategy: StorageStrategy::Standard {
            initial_capacity: 256,
            growth_factor: 1.5,
        },
        compression_strategy: CompressionStrategy::None,
        rank_select_type: RankSelectType::Interleaved256,
        enable_simd: true,
        enable_concurrency: false,
        cache_optimization: false,
    };
    let mut trie: ZiporaTrie<RankSelectInterleaved256> = ZiporaTrie::with_config(config.clone());

    // First, test just inserting a single-symbol key [0]
    println!("=== Inserting single symbol key [0] ===");
    let single_key = [0u8];
    let result1 = trie.insert(&single_key);
    println!("Insert [0] result: {:?}", result1);

    println!("Trie stats after inserting [0]:");
    let stats = trie.stats();
    println!("  Keys: {}", stats.num_keys);
    println!("  States: {}", stats.num_states);
    println!("  Memory usage: {} bytes", stats.memory_usage);

    println!("Can we lookup [0]? {}", trie.contains(&single_key));

    // Now test the failing case
    println!("\n=== Now testing two-symbol key [0, 1] ===");
    let mut trie2: ZiporaTrie<RankSelectInterleaved256> = ZiporaTrie::with_config(config);
    let key = [0u8, 1u8];

    println!("Inserting key: {:?}", key);
    let result = trie2.insert(&key);
    println!("Insert result: {:?}", result);

    // Debug trie stats after insertion
    println!("\nTrie stats after insertion:");
    let stats2 = trie2.stats();
    println!("  Keys: {}", stats2.num_keys);
    println!("  States: {}", stats2.num_states);
    println!("  Transitions: {}", stats2.num_transitions);
    println!("  Memory usage: {} bytes", stats2.memory_usage);

    println!("Checking if key exists...");
    let contains = trie2.contains(&key);
    println!("Contains result: {}", contains);

    println!("Lookup result: {:?}", trie2.lookup(&key));

    // Debug FSA interface
    println!("\nDebugging FSA interface:");
    println!("Root state: {}", trie2.root());
    println!("Root is final: {}", trie2.is_final(trie2.root()));
    println!("Accepts key: {}", trie2.accepts(&key));

    // Test state transitions manually
    let mut state = trie2.root();
    println!("Starting from root state: {}", state);

    for (i, &symbol) in key.iter().enumerate() {
        println!("Transition {} with symbol {}", i, symbol);
        if let Some(next_state) = trie2.transition(state, symbol) {
            println!("  -> Next state: {}", next_state);
            println!("  -> Is final: {}", trie2.is_final(next_state));
            state = next_state;
        } else {
            println!("  -> No transition found");
            break;
        }
    }

    println!("Final state: {}", state);
    println!("Final state is final: {}", trie2.is_final(state));
}
