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
    let mut trie: ZiporaTrie<RankSelectInterleaved256> = ZiporaTrie::with_config(config);
    trie.insert(b"hello").unwrap();
    trie.insert(b"help").unwrap();
    trie.insert(b"world").unwrap();

    println!("Testing longest_prefix:");

    let test_input = b"hello world";
    println!("Input: {:?}", std::str::from_utf8(test_input).unwrap());

    // Manual trace through the algorithm
    let mut state = trie.root();
    let mut last_final = None;

    println!("Starting at state: {}", state);

    for (i, &symbol) in test_input.iter().enumerate() {
        println!(
            "Step {}: Processing symbol '{}' ({})",
            i, symbol as char, symbol
        );
        println!(
            "  Current state: {}, is_final: {}",
            state,
            trie.is_final(state)
        );

        if trie.is_final(state) {
            last_final = Some(i);
            println!("  -> Updated last_final to: {:?}", last_final);
        }

        match trie.transition(state, symbol) {
            Some(next_state) => {
                state = next_state;
                println!("  -> Transitioned to state: {}", state);
            }
            None => {
                println!("  -> No transition found, breaking");
                break;
            }
        }
    }

    println!(
        "After loop: state = {}, is_final = {}",
        state,
        trie.is_final(state)
    );

    let result = if trie.is_final(state) {
        Some(test_input.len())
    } else {
        last_final
    };

    println!("Result: {:?}", result);
    println!("Expected: Some(5)");

    // Test the actual method
    let actual = trie.longest_prefix(test_input);
    println!("Actual method result: {:?}", actual);
}
