#![no_main]

use libfuzzer_sys::fuzz_target;
use ebook_conversion::palmdoc::*;

fuzz_target!(|data: &[u8]| {
    // todo: has lots of issues
    decompress_palmdoc(data);
});
