#![no_main]

use ebook_conversion::huffman::HuffmanTable;
use ebook_conversion::huffman_decoder::HuffmanDecoder;
use ebook_conversion::huffman_encoder::HuffmanEncoder;
use libfuzzer_sys::fuzz_target;
use std::collections::HashMap;

fuzz_target!(|data: &[u8]| {
    // todo: remove
    if data.len() < 3 {
        return;
    }

    let mut encoder = HuffmanEncoder::default();

    match encoder.pack(data) {
        Ok(()) => {
            let (table, compressed) = encoder.finish();

            let mut decoder = HuffmanDecoder { table };

            assert_eq!(data, decoder.unpack(&compressed));
        }
        Err(_) => {}
    }
});
