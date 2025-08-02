//! Ultra-simple rANS for debugging the core issue

use crate::error::Result;

/// Extremely simple rANS for debugging
pub fn simple_rans_test() -> Result<()> {
    // Test case: "he" - two different symbols with freq 1 each, similar to hello pattern
    let data = b"he";
    let mut frequencies = [0u32; 256];
    frequencies[b'h' as usize] = 1;
    frequencies[b'e' as usize] = 1;
    let total_freq = 2;
    
    // Cumulative frequencies:
    // cumulative[101] = 0 ('e')  
    // cumulative[104] = 1 ('h')
    let mut cumulative = [0u32; 257];
    cumulative[101] = 0;  // 'e' 
    cumulative[102] = 1;
    cumulative[104] = 1;  // 'h'
    cumulative[105] = 2;
    
    println!("Testing simple case: {:?}", data);
    println!("Frequencies: h={}, e={}", frequencies[104], frequencies[101]);
    println!("Cumulative: e={}, h={}", cumulative[101], cumulative[104]);
    
    // Manual encoding of "he" (process in reverse: e, h)
    let mut state = 1u32 << 23; // 8388608
    let mut output = Vec::new();
    
    println!("\n=== MANUAL ENCODING ===");
    println!("Initial state: {}", state);
    
    // Encode 'e' (reverse position 0)
    let freq_e = 1;
    let cumfreq_e = 0;
    println!("\nEncoding 'e': freq={}, cumfreq={}", freq_e, cumfreq_e);
    
    // Check renorm (unlikely for this simple case)
    let max_state = ((8388608u32 >> 8) << 8) * total_freq;
    println!("Max state: {}, current: {}", max_state, state);
    if state >= max_state {
        output.push((state & 0xFF) as u8);
        state >>= 8;
        println!("Renormalized to: {}", state);
    }
    
    // Apply encoding formula
    let new_state = ((state / freq_e) * total_freq) + (state % freq_e) + cumfreq_e;
    println!("Encoding 'e': {} -> {}", state, new_state);
    state = new_state;
    
    // Encode 'h' (reverse position 1)
    let freq_h = 1;
    let cumfreq_h = 1;
    println!("\nEncoding 'h': freq={}, cumfreq={}", freq_h, cumfreq_h);
    
    println!("Max state: {}, current: {}", max_state, state);
    if state >= max_state {
        output.push((state & 0xFF) as u8);
        state >>= 8;
        println!("Renormalized to: {}", state);
    }
    
    let new_state = ((state / freq_h) * total_freq) + (state % freq_h) + cumfreq_h;
    println!("Encoding 'h': {} -> {}", state, new_state);
    state = new_state;
    
    // Output final state
    output.extend_from_slice(&state.to_le_bytes());
    println!("\nFinal encoded: {:?}", output);
    
    // Manual decoding
    println!("\n=== MANUAL DECODING ===");
    let data_len = output.len();
    let mut decode_state = u32::from_le_bytes([
        output[data_len - 4], output[data_len - 3], output[data_len - 2], output[data_len - 1]
    ]);
    println!("Initial decode state: {}", decode_state);
    
    let mut pos = data_len - 4;
    let mut result = Vec::new();
    
    // Decode first symbol
    println!("\nDecoding symbol 1:");
    while decode_state < 8388608 && pos > 0 {
        pos -= 1;
        decode_state = (decode_state << 8) | (output[pos] as u32);
        println!("Renormalized to: {}", decode_state);
    }
    
    let slot = decode_state % total_freq;
    println!("State: {}, slot: {}", decode_state, slot);
    
    let symbol1 = if slot == 0 { b'e' } else { b'h' };
    let freq1 = 1;
    let cumfreq1 = if symbol1 == b'e' { 0 } else { 1 };
    
    println!("Found symbol: {} ({})", symbol1 as char, symbol1);
    
    decode_state = freq1 * (decode_state / total_freq) + (decode_state % total_freq) - cumfreq1;
    println!("New state: {}", decode_state);
    result.push(symbol1);
    
    // Decode second symbol
    println!("\nDecoding symbol 2:");
    while decode_state < 8388608 && pos > 0 {
        pos -= 1;
        decode_state = (decode_state << 8) | (output[pos] as u32);
        println!("Renormalized to: {}", decode_state);
    }
    
    let slot = decode_state % total_freq;
    println!("State: {}, slot: {}", decode_state, slot);
    
    let symbol2 = if slot == 0 { b'e' } else { b'h' };
    let freq2 = 1;
    let cumfreq2 = if symbol2 == b'e' { 0 } else { 1 };
    
    println!("Found symbol: {} ({})", symbol2 as char, symbol2);
    
    decode_state = freq2 * (decode_state / total_freq) + (decode_state % total_freq) - cumfreq2;
    println!("New state: {}", decode_state);
    result.push(symbol2);
    
    // Reverse and check
    result.reverse();
    println!("\nResult: {:?}", result);
    println!("Expected: {:?}", data);
    
    if result == data.to_vec() {
        println!("SUCCESS!");
    } else {
        println!("FAILURE!");
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simple_debug() {
        simple_rans_test().unwrap();
    }
}