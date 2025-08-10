use zipora::fsa::{DoubleArrayTrie, Trie, FiniteStateAutomaton};

fn main() {
    let mut trie = DoubleArrayTrie::new();
    
    // First, test just inserting a single-symbol key [0]
    println!("=== Inserting single symbol key [0] ===");
    let single_key = [0u8];
    let result1 = trie.insert(&single_key);
    println!("Insert [0] result: {:?}", result1);
    
    println!("Internal arrays after inserting [0]:");
    for i in 0..3 {
        println!("  base[{}] = {}", i, trie.get_base(i as u32));
        println!("  check[{}] = 0x{:x} (free: {}, terminal: {}, parent: {})", 
                 i, 
                 trie.get_check(i as u32),
                 trie.is_free(i as u32),
                 trie.is_terminal(i as u32),
                 trie.get_parent(i as u32));
    }
    
    println!("Can we lookup [0]? {}", trie.contains(&single_key));
    
    // Now test the failing case
    println!("\n=== Now testing two-symbol key [0, 1] ===");
    let mut trie2 = DoubleArrayTrie::new();
    let key = [0u8, 1u8];
    
    println!("Inserting key: {:?}", key);
    let result = trie2.insert(&key);
    println!("Insert result: {:?}", result);
    
    // Debug internal arrays after insertion
    println!("\nInternal arrays after insertion:");
    for i in 0..5 {
        println!("  base[{}] = {}", i, trie2.get_base(i as u32));
        println!("  check[{}] = 0x{:x} (free: {}, terminal: {}, parent: {})", 
                 i, 
                 trie2.get_check(i as u32),
                 trie2.is_free(i as u32),
                 trie2.is_terminal(i as u32),
                 trie2.get_parent(i as u32));
    }
    
    println!("Checking if key exists...");
    let contains = trie2.contains(&key);
    println!("Contains result: {}", contains);
    
    println!("Lookup result: {:?}", trie2.lookup(&key));
    
    // Debug internal state
    println!("\nDebugging internal state:");
    println!("Root state: {}", trie2.root());
    println!("Root is terminal: {}", trie2.is_terminal(trie2.root()));
    
    // Test state transitions manually
    let mut state = trie2.root();
    println!("Starting from root state: {}", state);
    
    for (i, &symbol) in key.iter().enumerate() {
        println!("Transition {} with symbol {}", i, symbol);
        if let Some(next_state) = trie2.transition(state, symbol) {
            println!("  -> Next state: {}", next_state);
            println!("  -> Is terminal: {}", trie2.is_terminal(next_state));
            state = next_state;
        } else {
            println!("  -> No transition found");
            break;
        }
    }
    
    println!("Final state: {}", state);
    println!("Final state is terminal: {}", trie2.is_terminal(state));
}