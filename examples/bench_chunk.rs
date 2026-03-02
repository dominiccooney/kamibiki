//! Benchmark comparing tree-sitter and line-heuristic chunkers.
//!
//! Run with:
//!     cargo run --example bench_chunk --release
//!
//! Generates synthetic source files of varying sizes and languages,
//! then times each chunker strategy.

use std::time::{Duration, Instant};

use kb::chunk::{self, Chunker};

const MAX_TOKENS: usize = 430;

fn main() {
    println!("Chunker benchmark");
    println!("═════════════════\n");

    let ts_chunker = chunk::tsv1_chunker().expect("failed to create treesitter chunker");
    let lines_chunker = chunk::lines_chunker().expect("failed to create lines chunker");

    let test_cases = generate_test_cases();

    println!(
        "{:<30} {:>8} {:>10} {:>10} {:>8}",
        "Test case", "Size", "TreeSitter", "Lines", "Speedup"
    );
    println!("{}", "─".repeat(72));

    for (name, path, content) in &test_cases {
        let ts_dur = bench_chunker(&ts_chunker, path, content);
        let lines_dur = bench_chunker(&lines_chunker, path, content);

        let speedup = ts_dur.as_secs_f64() / lines_dur.as_secs_f64();
        println!(
            "{:<30} {:>7}B {:>9} {:>9} {:>7.1}x",
            name,
            content.len(),
            format_duration(ts_dur),
            format_duration(lines_dur),
            speedup,
        );
    }

    println!();
}

fn bench_chunker(chunker: &impl Chunker, path: &str, content: &[u8]) -> Duration {
    // Warm up
    let _ = chunker.chunk(path, content, MAX_TOKENS);

    let iterations = 5;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = chunker.chunk(path, content, MAX_TOKENS);
    }
    start.elapsed() / iterations as u32
}

fn format_duration(d: Duration) -> String {
    let us = d.as_micros();
    if us < 1000 {
        format!("{}µs", us)
    } else if us < 1_000_000 {
        format!("{:.1}ms", us as f64 / 1000.0)
    } else {
        format!("{:.2}s", us as f64 / 1_000_000.0)
    }
}

fn generate_test_cases() -> Vec<(String, String, Vec<u8>)> {
    let mut cases = Vec::new();

    // Small Rust file (~50 functions)
    cases.push((
        "Rust 50 fns".into(),
        "test.rs".into(),
        generate_rust_file(50).into_bytes(),
    ));

    // Medium Rust file (~200 functions)
    cases.push((
        "Rust 200 fns".into(),
        "test.rs".into(),
        generate_rust_file(200).into_bytes(),
    ));

    // Large Rust file (~500 functions)
    cases.push((
        "Rust 500 fns".into(),
        "test.rs".into(),
        generate_rust_file(500).into_bytes(),
    ));

    // Python file
    cases.push((
        "Python 200 fns".into(),
        "test.py".into(),
        generate_python_file(200).into_bytes(),
    ));

    // TypeScript file
    cases.push((
        "TypeScript 200 fns".into(),
        "test.ts".into(),
        generate_typescript_file(200).into_bytes(),
    ));

    // Go file
    cases.push((
        "Go 200 fns".into(),
        "test.go".into(),
        generate_go_file(200).into_bytes(),
    ));

    // Plain text (no tree-sitter, both use line chunker)
    cases.push((
        "Plain text 200 paragraphs".into(),
        "test.txt".into(),
        generate_plaintext(200).into_bytes(),
    ));

    cases
}

fn generate_rust_file(n: usize) -> String {
    let mut s = String::new();
    for i in 0..n {
        s.push_str(&format!(
            "/// Documentation for function {i}.\n\
             /// This function does important things.\n\
             fn function_{i}(x: i32, y: i32) -> i32 {{\n\
                 // Compute a value\n\
                 let a = x + {i};\n\
                 let b = y * {i};\n\
                 let c = a.wrapping_mul(b);\n\
                 if c > 100 {{\n\
                     c - 50\n\
                 }} else {{\n\
                     c + 50\n\
                 }}\n\
             }}\n\n"
        ));
    }
    s
}

fn generate_python_file(n: usize) -> String {
    let mut s = String::new();
    for i in 0..n {
        s.push_str(&format!(
            "def function_{i}(x: int, y: int) -> int:\n\
             \x20   \"\"\"Documentation for function {i}.\"\"\"\n\
             \x20   a = x + {i}\n\
             \x20   b = y * {i}\n\
             \x20   c = a * b\n\
             \x20   if c > 100:\n\
             \x20       return c - 50\n\
             \x20   else:\n\
             \x20       return c + 50\n\n\n"
        ));
    }
    s
}

fn generate_typescript_file(n: usize) -> String {
    let mut s = String::new();
    s.push_str("// Generated TypeScript file\n\n");
    for i in 0..n {
        s.push_str(&format!(
            "/** Documentation for function {i}. */\n\
             function function_{i}(x: number, y: number): number {{\n\
             \x20   const a = x + {i};\n\
             \x20   const b = y * {i};\n\
             \x20   const c = a * b;\n\
             \x20   if (c > 100) {{\n\
             \x20       return c - 50;\n\
             \x20   }} else {{\n\
             \x20       return c + 50;\n\
             \x20   }}\n\
             }}\n\n"
        ));
    }
    s
}

fn generate_go_file(n: usize) -> String {
    let mut s = String::new();
    s.push_str("package main\n\n");
    for i in 0..n {
        s.push_str(&format!(
            "// Function{i} does important computation.\n\
             func Function{i}(x, y int) int {{\n\
             \ta := x + {i}\n\
             \tb := y * {i}\n\
             \tc := a * b\n\
             \tif c > 100 {{\n\
             \t\treturn c - 50\n\
             \t}}\n\
             \treturn c + 50\n\
             }}\n\n"
        ));
    }
    s
}

fn generate_plaintext(n: usize) -> String {
    let mut s = String::new();
    for i in 0..n {
        s.push_str(&format!(
            "Paragraph {i}: Lorem ipsum dolor sit amet, consectetur \
             adipiscing elit. Sed do eiusmod tempor incididunt ut labore \
             et dolore magna aliqua. Ut enim ad minim veniam, quis \
             nostrud exercitation ullamco laboris nisi ut aliquip ex ea \
             commodo consequat.\n\n"
        ));
    }
    s
}
