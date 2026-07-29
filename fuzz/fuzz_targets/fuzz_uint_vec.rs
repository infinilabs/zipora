//! Fuzz UintVecMin0 get/set round-trip (plan.md 6.1, H4 tail-read regression).
#![no_main]
use libfuzzer_sys::fuzz_target;
use zipora::containers::UintVecMin0;

fuzz_target!(|data: &[u8]| {
    if data.len() < 10 {
        return;
    }
    let num = (u16::from_le_bytes([data[0], data[1]]) as usize % 4096) + 1;
    // UintVecMin0 supports <= 58-bit values by contract (get() asserts
    // "Use BigUintVecMin0 for >58 bits"), so cap the fuzzed max_val.
    let max_val =
        (u64::from_le_bytes(data[2..10].try_into().unwrap()) as usize) & ((1usize << 58) - 1);
    let mut v = UintVecMin0::new(num, max_val);
    let mut expected = vec![0usize; num];
    // Apply fuzz-driven set operations, then verify every slot
    for chunk in data[10..].chunks(10).take(512) {
        if chunk.len() < 10 {
            break;
        }
        let idx = u16::from_le_bytes([chunk[0], chunk[1]]) as usize % num;
        let val = if max_val == 0 {
            0
        } else {
            u64::from_le_bytes(chunk[2..10].try_into().unwrap()) as usize % (max_val + 1)
        };
        v.set(idx, val);
        expected[idx] = val;
    }
    for (i, &e) in expected.iter().enumerate() {
        assert_eq!(v.get(i), e, "slot {} mismatch", i);
    }
});
