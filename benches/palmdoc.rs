use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ebook_conversion::palmdoc::compress_palmdoc;
use lipsum::lipsum;

fn palmdoc_compression(c: &mut Criterion) {
    let input = lipsum(2048);
    let input = input.as_bytes();

    c.bench_function("palmdoc compression", |b| {
        b.iter(|| compress_palmdoc(black_box(&input)))
    });
}

fn palmdoc_decompression(c: &mut Criterion) {
    let input = lipsum(4096);
    let input = input.as_bytes();
    let compressed = compress_palmdoc(&input);

    c.bench_function("palmdoc decompression", |b| {
        b.iter(|| {
            ebook_conversion::palmdoc::decompress_palmdoc(black_box(&compressed));
        })
    });
}

criterion_group!(benches, palmdoc_compression, palmdoc_decompression);
criterion_main!(benches);
