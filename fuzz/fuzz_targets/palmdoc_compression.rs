#![no_main]

use libfuzzer_sys::fuzz_target;
use ebook_conversion::palmdoc::compress_palmdoc;

fuzz_target!(|data: &[u8]| {
    compress_palmdoc(data);
});
