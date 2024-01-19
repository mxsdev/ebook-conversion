#![no_main]

use libfuzzer_sys::fuzz_target;
use ebook_conversion::palmdoc::*;

fuzz_target!(|data: &[u8]| {
    decompress_palmdoc(compress_palmdoc(data));
});
