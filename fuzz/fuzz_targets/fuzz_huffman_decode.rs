//! Fuzz HuffmanDecoder::decode with arbitrary bit streams (plan.md 6.1, H11).
//! Tree from fuzz-chosen frequencies; decode must error or terminate bounded.
#![no_main]
use libfuzzer_sys::fuzz_target;
use zipora::entropy::{HuffmanDecoder, HuffmanTree};

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }
    // Derive a frequency table from the first bytes
    let mut freqs = [0u32; 256];
    let n_syms = (data[0] as usize % 8) + 2;
    for i in 0..n_syms {
        freqs[data[1 + (i % 4)] as usize % 256] += 1 + data[i % data.len()] as u32;
    }
    let output_length =
        u16::from_le_bytes([data[4], data[5]]) as usize % 65536;
    let payload = &data[8..];
    if let Ok(tree) = HuffmanTree::from_frequencies(&freqs) {
        let decoder = HuffmanDecoder::new(tree);
        let _ = decoder.decode(payload, output_length);
    }
});
