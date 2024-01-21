use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ebook_conversion::palmdoc::compress_palmdoc;

fn criterion_benchmark(c: &mut Criterion) {
    let random_input = (0..8192).map(|_| rand::random::<u8>()).collect::<Vec<u8>>();
    c.bench_function("palmdoc compression", |b| {
        b.iter(|| compress_palmdoc(black_box(&random_input)))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
