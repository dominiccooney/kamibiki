use anyhow::Result;
use tree_sitter::{Language, Node, Parser, Tree};

use super::lines::chunk_by_lines;
use super::{Chunk, Chunker, TokenCounter};

/// A tree-sitter based chunker that uses a top-down recursive
/// approach: it prefers fewer, larger chunks, places breaks at the
/// highest AST level possible, and uses blank lines between sibling
/// nodes to decide where to break (keeping comments/attributes
/// attached to the declaration they precede).
///
/// Falls back to line-based chunking for unsupported languages.
pub struct TreeSitterChunker {
    token_counter: TokenCounter,
}

impl TreeSitterChunker {
    pub fn new(token_counter: TokenCounter) -> Self {
        TreeSitterChunker { token_counter }
    }
}

impl Chunker for TreeSitterChunker {
    fn chunk(&self, path: &str, content: &[u8], max_tokens: usize) -> Result<Vec<Chunk>> {
        if content.is_empty() {
            return Ok(Vec::new());
        }

        let language = language_for_path(path);
        match language {
            Some(lang) => {
                chunk_with_tree_sitter(lang, content, max_tokens, &self.token_counter)
            }
            None => Ok(chunk_by_lines(content, max_tokens, &self.token_counter)),
        }
    }
}

/// Detect the tree-sitter language from a file path's extension.
/// Returns `None` for unsupported languages.
pub fn language_for_path(path: &str) -> Option<Language> {
    let ext = path.rsplit('.').next()?;
    match ext {
        "rs" => Some(tree_sitter_rust::LANGUAGE.into()),
        "py" | "pyi" => Some(tree_sitter_python::LANGUAGE.into()),
        "js" | "mjs" | "cjs" | "jsx" => {
            Some(tree_sitter_javascript::LANGUAGE.into())
        }
        "ts" | "mts" | "cts" => {
            Some(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into())
        }
        "tsx" => Some(tree_sitter_typescript::LANGUAGE_TSX.into()),
        "go" => Some(tree_sitter_go::LANGUAGE.into()),
        "c" | "h" => Some(tree_sitter_c::LANGUAGE.into()),
        "cpp" | "cc" | "cxx" | "hpp" | "hxx" | "hh" => {
            Some(tree_sitter_cpp::LANGUAGE.into())
        }
        "java" => Some(tree_sitter_java::LANGUAGE.into()),
        "rb" => Some(tree_sitter_ruby::LANGUAGE.into()),
        _ => None,
    }
}

/// Parse content with tree-sitter and chunk using a top-down
/// recursive strategy.
fn chunk_with_tree_sitter(
    language: Language,
    content: &[u8],
    max_tokens: usize,
    tc: &TokenCounter,
) -> Result<Vec<Chunk>> {
    let tree = parse(language, content)?;
    let root = tree.root_node();

    // Top-down recursive chunking from the root.
    let ranges = chunk_node(root, content, max_tokens, tc);

    // Convert ranges to contiguous Chunk byte-ranges covering
    // [0, content.len()).
    Ok(ranges_to_chunks(&ranges, content.len()))
}

/// Parse content with tree-sitter, returning the parse tree.
fn parse(language: Language, content: &[u8]) -> Result<Tree> {
    let mut parser = Parser::new();
    parser.set_language(&language)?;
    parser
        .parse(content, None)
        .ok_or_else(|| anyhow::anyhow!("tree-sitter parse failed"))
}

/// A byte range [start, end) produced during recursive chunking.
#[derive(Debug, Clone, Copy)]
struct Range {
    start: usize,
    end: usize,
}

/// Count blank lines in a byte slice. A blank line is a line
/// consisting only of whitespace.
fn count_blank_lines(content: &[u8]) -> usize {
    let mut blank_count = 0;
    let mut line_is_blank = true;

    for &b in content {
        if b == b'\n' {
            if line_is_blank {
                blank_count += 1;
            }
            line_is_blank = true;
        } else if b != b' ' && b != b'\t' && b != b'\r' {
            line_is_blank = false;
        }
    }

    blank_count
}

/// Score the boundary between two adjacent sibling nodes. Uses the
/// gap content between them to count blank lines.
///
/// Higher score = better place to break.
/// - 0 blank lines → score 0 (items are "stuck together", e.g.
///   doc comment immediately before a function)
/// - 1+ blank lines → score 100 + blank_count (paragraph boundary)
fn boundary_score(content: &[u8], prev_end: usize, next_start: usize) -> u32 {
    if next_start <= prev_end {
        return 0;
    }
    let gap = &content[prev_end..next_start];
    let blanks = count_blank_lines(gap);
    if blanks > 1 {
        100 + (blanks - 1) as u32
    } else {
        0
    }
}

/// Recursively chunk an AST node top-down.
///
/// If the node fits within max_tokens, return it as a single range.
/// Otherwise, examine its children: score boundaries between siblings
/// by blank lines, greedily merge children preferring breaks at
/// high-score boundaries, and recurse into any single child that
/// still exceeds max_tokens.
///
/// Uses prefix sums over per-child token counts for O(1) group size
/// estimates, avoiding the O(N²) cost of re-tokenizing growing
/// groups on every iteration.
fn chunk_node(
    node: Node,
    content: &[u8],
    max_tokens: usize,
    tc: &TokenCounter,
) -> Vec<Range> {
    let node_tokens = tc.count(&content[node.start_byte()..node.end_byte()]);

    // Base case: the node fits in one chunk.
    if node_tokens <= max_tokens {
        return vec![Range {
            start: node.start_byte(),
            end: node.end_byte(),
        }];
    }

    // Collect children.
    let children: Vec<Node> = {
        let mut cursor = node.walk();
        node.children(&mut cursor).collect()
    };

    // If no children (oversized leaf), fall back to line splitting.
    if children.is_empty() {
        let sub = chunk_by_lines(
            &content[node.start_byte()..node.end_byte()],
            max_tokens,
            tc,
        );
        return sub
            .into_iter()
            .map(|c| Range {
                start: node.start_byte() + c.byte_offset,
                end: node.start_byte() + c.byte_offset + c.len,
            })
            .collect();
    }

    // Precompute per-child token counts (one tc.count per child).
    let child_tokens: Vec<usize> = children
        .iter()
        .map(|c| tc.count(&content[c.start_byte()..c.end_byte()]))
        .collect();

    // Prefix sum for O(1) range queries: prefix[i+1] - prefix[j]
    // gives the approximate token count for children[j..=i].
    let prefix: Vec<usize> = {
        let mut p = vec![0usize; children.len() + 1];
        for i in 0..children.len() {
            p[i + 1] = p[i] + child_tokens[i];
        }
        p
    };

    // Score boundaries between adjacent children.
    let mut boundary_scores: Vec<u32> = vec![0]; // index 0 unused
    for i in 1..children.len() {
        let score = boundary_score(
            content,
            children[i - 1].end_byte(),
            children[i].start_byte(),
        );
        boundary_scores.push(score);
    }

    // Greedily merge children into groups.
    let mut result: Vec<Range> = Vec::new();
    let mut group_start_idx: usize = 0;

    let mut best_break_idx: Option<usize> = None;
    let mut best_break_score: u32 = 0;

    for i in 1..children.len() {
        // O(1) token estimate via prefix sum instead of re-tokenizing.
        let prospective_tokens = prefix[i + 1] - prefix[group_start_idx];

        let score = boundary_scores[i];

        if prospective_tokens > max_tokens && i > group_start_idx {
            let break_idx = if let Some(bi) = best_break_idx {
                bi
            } else {
                i
            };

            emit_group(
                &children,
                group_start_idx,
                break_idx,
                content,
                max_tokens,
                tc,
                &child_tokens,
                &mut result,
            );

            group_start_idx = break_idx;
            best_break_idx = None;
            best_break_score = 0;

            if i > group_start_idx {
                if score > best_break_score {
                    best_break_idx = Some(i);
                    best_break_score = score;
                }
            }

            continue;
        }

        if i > group_start_idx && score >= best_break_score {
            best_break_idx = Some(i);
            best_break_score = score;
        }
    }

    // Emit the final group.
    emit_group(
        &children,
        group_start_idx,
        children.len(),
        content,
        max_tokens,
        tc,
        &child_tokens,
        &mut result,
    );

    result
}

/// Emit a group of children [start_idx..end_idx) as chunk ranges.
///
/// Uses precomputed `child_tokens` to avoid redundant tokenizer
/// calls. Falls back to `tc.count()` only when recursing into
/// oversized single children (which recomputes their own prefix
/// sums at a lower level).
fn emit_group(
    children: &[Node],
    start_idx: usize,
    end_idx: usize,
    content: &[u8],
    max_tokens: usize,
    tc: &TokenCounter,
    child_tokens: &[usize],
    result: &mut Vec<Range>,
) {
    if start_idx >= end_idx {
        return;
    }

    let group_start = children[start_idx].start_byte();
    let group_end = children[end_idx - 1].end_byte();
    let group_tokens: usize = child_tokens[start_idx..end_idx].iter().sum();

    if group_tokens <= max_tokens {
        result.push(Range {
            start: group_start,
            end: group_end,
        });
    } else if end_idx - start_idx == 1 {
        // Single oversized child — recurse into it.
        let sub = chunk_node(children[start_idx], content, max_tokens, tc);
        result.extend(sub);
    } else {
        // Multiple children that together exceed max_tokens.
        // Try each child individually, merging small neighbors.
        let group_result_start = result.len();
        let mut running_tokens: usize = 0;

        for idx in start_idx..end_idx {
            let ct = child_tokens[idx];
            if ct <= max_tokens {
                // Try to merge with the previous range from this group.
                if result.len() > group_result_start
                    && running_tokens + ct <= max_tokens
                {
                    result.last_mut().unwrap().end = children[idx].end_byte();
                    running_tokens += ct;
                    continue;
                }
                result.push(Range {
                    start: children[idx].start_byte(),
                    end: children[idx].end_byte(),
                });
                running_tokens = ct;
            } else {
                let sub = chunk_node(children[idx], content, max_tokens, tc);
                result.extend(sub);
                running_tokens = 0;
            }
        }
    }
}

/// Convert AST-derived ranges to contiguous Chunks covering
/// [0, content_len).
///
/// Gaps between ranges are absorbed into the preceding chunk.
/// Leading content before the first range becomes the start of the
/// first chunk. Trailing content after the last range extends the
/// last chunk.
fn ranges_to_chunks(ranges: &[Range], content_len: usize) -> Vec<Chunk> {
    if ranges.is_empty() {
        if content_len == 0 {
            return Vec::new();
        }
        return vec![Chunk { byte_offset: 0, len: content_len }];
    }

    // Deduplicate/skip fully-overlapped ranges and collect valid ones.
    let mut merged: Vec<Range> = Vec::new();
    let mut pos = 0usize;
    for range in ranges {
        if range.end <= pos || range.end <= range.start {
            continue;
        }
        merged.push(Range {
            start: range.start.max(pos),
            end: range.end,
        });
        pos = range.end;
    }

    if merged.is_empty() {
        if content_len == 0 {
            return Vec::new();
        }
        return vec![Chunk { byte_offset: 0, len: content_len }];
    }

    // Create chunks: one per merged range. The first chunk starts at
    // byte 0. Each subsequent chunk starts at the start of its range
    // (absorbing any gap into the preceding chunk). The last chunk
    // extends to content_len.
    let mut chunks = Vec::with_capacity(merged.len());

    for (i, range) in merged.iter().enumerate() {
        let chunk_start = if i == 0 { 0 } else { range.start };
        let chunk_end = if i + 1 < merged.len() {
            merged[i + 1].start
        } else {
            content_len
        };

        if chunk_end > chunk_start {
            chunks.push(Chunk {
                byte_offset: chunk_start,
                len: chunk_end - chunk_start,
            });
        }
    }

    // Edge case: ensure we cover from 0.
    if chunks.is_empty() && content_len > 0 {
        chunks.push(Chunk { byte_offset: 0, len: content_len });
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::LazyLock;

    const TEST_MAX_TOKENS: usize = 430;

    static TEST_TC: LazyLock<TokenCounter> = LazyLock::new(|| {
        TokenCounter::for_voyage().expect("failed to load tokenizer for tests")
    });

    fn test_chunker() -> TreeSitterChunker {
        TreeSitterChunker::new(TEST_TC.clone())
    }

    /// Verify chunks are contiguous and cover the entire input.
    fn assert_contiguous(chunks: &[Chunk], content: &[u8]) {
        if content.is_empty() {
            assert!(chunks.is_empty());
            return;
        }
        assert!(!chunks.is_empty());
        assert_eq!(chunks[0].byte_offset, 0);
        let mut expected_offset = 0;
        for chunk in chunks {
            assert_eq!(
                chunk.byte_offset, expected_offset,
                "gap at offset {expected_offset}, chunk starts at {}",
                chunk.byte_offset
            );
            assert!(chunk.len > 0);
            expected_offset += chunk.len;
        }
        assert_eq!(expected_offset, content.len());
    }

    /// Verify the content of chunks matches the original source.
    fn assert_content_matches(chunks: &[Chunk], content: &[u8]) {
        for chunk in chunks {
            let slice = chunk.content(content);
            let expected =
                &content[chunk.byte_offset..chunk.byte_offset + chunk.len];
            assert_eq!(slice, expected);
        }
    }

    #[test]
    fn empty_rust_file() {
        let chunker = test_chunker();
        let chunks = chunker.chunk("test.rs", b"", TEST_MAX_TOKENS).unwrap();
        assert!(chunks.is_empty());
    }

    #[test]
    fn small_rust_file() {
        let content = br#"fn main() {
    println!("hello");
}
"#;
        let chunker = test_chunker();
        let chunks = chunker
            .chunk("test.rs", content, TEST_MAX_TOKENS)
            .unwrap();
        assert_contiguous(&chunks, content);
        assert_content_matches(&chunks, content);
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn rust_file_with_multiple_functions() {
        let mut content = String::new();
        for i in 0..50 {
            content.push_str(&format!(
                "fn function_{i}(x: i32) -> i32 {{\n    \
                     let result = x * {i} + 42;\n    \
                     println!(\"function_{i}: {{}}\", result);\n    \
                     result\n\
                 }}\n\n"
            ));
        }
        let bytes = content.as_bytes();
        let tc = &*TEST_TC;

        let chunker = test_chunker();
        let chunks = chunker
            .chunk("test.rs", bytes, TEST_MAX_TOKENS)
            .unwrap();
        assert_contiguous(&chunks, bytes);
        assert_content_matches(&chunks, bytes);

        assert!(
            chunks.len() > 1,
            "expected multiple chunks, got {}",
            chunks.len()
        );

        for chunk in &chunks {
            let chunk_tokens = tc.count(chunk.content(bytes));
            assert!(
                chunk_tokens <= TEST_MAX_TOKENS * 2,
                "chunk too large: {} tokens (max_tokens={})",
                chunk_tokens,
                TEST_MAX_TOKENS
            );
        }
    }

    #[test]
    fn python_file() {
        let content = br#"def hello():
    print("hello world")

def goodbye():
    print("goodbye world")

class Greeter:
    def __init__(self, name):
        self.name = name

    def greet(self):
        print(f"Hello, {self.name}!")
"#;
        let chunker = test_chunker();
        let chunks = chunker
            .chunk("test.py", content, TEST_MAX_TOKENS)
            .unwrap();
        assert_contiguous(&chunks, content);
        assert_content_matches(&chunks, content);
    }

    #[test]
    fn javascript_file() {
        let content = br#"function hello() {
    console.log("hello");
}

const add = (a, b) => a + b;

class Foo {
    constructor(x) {
        this.x = x;
    }
}
"#;
        let chunker = test_chunker();
        let chunks = chunker
            .chunk("test.js", content, TEST_MAX_TOKENS)
            .unwrap();
        assert_contiguous(&chunks, content);
        assert_content_matches(&chunks, content);
    }

    #[test]
    fn typescript_file() {
        let content = br#"interface Foo {
    bar: string;
}

function greet(name: string): void {
    console.log(`Hello, ${name}`);
}
"#;
        let chunker = test_chunker();
        let chunks = chunker
            .chunk("test.ts", content, TEST_MAX_TOKENS)
            .unwrap();
        assert_contiguous(&chunks, content);
        assert_content_matches(&chunks, content);
    }

    #[test]
    fn go_file() {
        let content = br#"package main

import "fmt"

func main() {
	fmt.Println("hello")
}

func add(a, b int) int {
	return a + b
}
"#;
        let chunker = test_chunker();
        let chunks = chunker
            .chunk("test.go", content, TEST_MAX_TOKENS)
            .unwrap();
        assert_contiguous(&chunks, content);
        assert_content_matches(&chunks, content);
    }

    #[test]
    fn unsupported_language_falls_back() {
        let content = b"some random content\nwith multiple lines\n";
        let chunker = test_chunker();
        let chunks = chunker
            .chunk("test.txt", content, TEST_MAX_TOKENS)
            .unwrap();
        assert_contiguous(&chunks, content);
        assert_content_matches(&chunks, content);
    }

    #[test]
    fn language_detection() {
        assert!(language_for_path("foo.rs").is_some());
        assert!(language_for_path("foo.py").is_some());
        assert!(language_for_path("foo.js").is_some());
        assert!(language_for_path("foo.jsx").is_some());
        assert!(language_for_path("foo.ts").is_some());
        assert!(language_for_path("foo.tsx").is_some());
        assert!(language_for_path("foo.go").is_some());
        assert!(language_for_path("foo.c").is_some());
        assert!(language_for_path("foo.h").is_some());
        assert!(language_for_path("foo.cpp").is_some());
        assert!(language_for_path("foo.cc").is_some());
        assert!(language_for_path("foo.java").is_some());
        assert!(language_for_path("foo.rb").is_some());
        assert!(language_for_path("foo.txt").is_none());
        assert!(language_for_path("foo.md").is_none());
        assert!(language_for_path("Makefile").is_none());
    }

    #[test]
    fn large_rust_file_chunking() {
        let mut content = String::new();
        content.push_str("// This is a large Rust source file.\n\n");
        for i in 0..200 {
            content.push_str(&format!(
                "/// Documentation for function {i}.\n\
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
        let bytes = content.as_bytes();

        let chunker = test_chunker();
        let chunks = chunker
            .chunk("big.rs", bytes, TEST_MAX_TOKENS)
            .unwrap();
        assert_contiguous(&chunks, bytes);
        assert_content_matches(&chunks, bytes);
        assert!(chunks.len() > 5, "expected many chunks for large file");
    }

    #[test]
    fn smaller_token_budget_produces_more_chunks() {
        let mut content = String::new();
        for i in 0..20 {
            content.push_str(&format!("fn f_{i}() -> i32 {{ {i} }}\n"));
        }
        let bytes = content.as_bytes();
        let chunker = test_chunker();

        let large_budget = chunker.chunk("test.rs", bytes, 1000).unwrap();
        let small_budget = chunker.chunk("test.rs", bytes, 50).unwrap();

        assert_contiguous(&large_budget, bytes);
        assert_contiguous(&small_budget, bytes);
        assert!(
            small_budget.len() >= large_budget.len(),
            "smaller token budget should produce at least as many chunks: \
             small={} >= large={}",
            small_budget.len(),
            large_budget.len()
        );
    }

    #[test]
    fn doc_comments_stay_with_function_rust() {
        let mut content = String::new();
        for i in 0..20 {
            content.push_str(&format!(
                "/// Documentation for function {i}.\n\
                 #[inline]\n\
                 fn function_{i}(x: i32) -> i32 {{\n\
                     x + {i}\n\
                 }}\n\n"
            ));
        }
        let bytes = content.as_bytes();
        let chunker = test_chunker();
        let chunks = chunker
            .chunk("test.rs", bytes, TEST_MAX_TOKENS)
            .unwrap();
        assert_contiguous(&chunks, bytes);
        assert_content_matches(&chunks, bytes);

        for chunk in &chunks {
            if chunk.byte_offset == 0 {
                continue;
            }
            let content_slice = chunk.content(bytes);
            let start_text = std::str::from_utf8(
                &content_slice[..content_slice.len().min(50)]
            ).unwrap_or("");
            assert!(
                start_text.starts_with("///") || start_text.starts_with("fn "),
                "chunk at offset {} starts with unexpected content: {:?}",
                chunk.byte_offset,
                &start_text[..start_text.len().min(30)]
            );
        }
    }

    #[test]
    fn comments_stay_with_function_go() {
        let content = br#"package main

// Add adds two numbers together.
// It returns the sum.
func Add(a, b int) int {
	return a + b
}

// Sub subtracts b from a.
func Sub(a, b int) int {
	return a - b
}

// Mul multiplies two numbers.
func Mul(a, b int) int {
	return a * b
}
"#;
        let chunker = test_chunker();
        let chunks = chunker.chunk("test.go", content, 30).unwrap();
        assert_contiguous(&chunks, content);
        assert_content_matches(&chunks, content);

        for chunk in &chunks {
            let text = std::str::from_utf8(chunk.content(content)).unwrap_or("");
            let first_non_ws = text.trim_start();
            if chunk.byte_offset > 0 {
                assert!(
                    !first_non_ws.starts_with("func "),
                    "chunk at offset {} starts with bare 'func' (comment detached): {:?}",
                    chunk.byte_offset,
                    &first_non_ws[..first_non_ws.len().min(40)]
                );
            }
        }
    }

    #[test]
    fn count_blank_lines_cases() {
        assert_eq!(count_blank_lines(b""), 0);
        assert_eq!(count_blank_lines(b"\n"), 1);
        assert_eq!(count_blank_lines(b"\n\n"), 2);
        assert_eq!(count_blank_lines(b"hello\n"), 0);
        assert_eq!(count_blank_lines(b"hello\n\n"), 1);
        assert_eq!(count_blank_lines(b"\nhello\n"), 1);
        assert_eq!(count_blank_lines(b"  \n"), 1);
    }

    #[test]
    fn boundary_score_cases() {
        let content = b"aaa\n\nbbb";
        assert!(boundary_score(content, 3, 5) > 100);

        let content2 = b"aaa\nbbb";
        assert_eq!(boundary_score(content2, 3, 4), 0);
    }

    #[test]
    fn ranges_to_chunks_empty() {
        let chunks = ranges_to_chunks(&[], 0);
        assert!(chunks.is_empty());

        let chunks = ranges_to_chunks(&[], 10);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].byte_offset, 0);
        assert_eq!(chunks[0].len, 10);
    }

    #[test]
    fn ranges_to_chunks_single() {
        let ranges = vec![Range { start: 0, end: 10 }];
        let chunks = ranges_to_chunks(&ranges, 10);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].byte_offset, 0);
        assert_eq!(chunks[0].len, 10);
    }

    #[test]
    fn ranges_to_chunks_with_gaps() {
        // Ranges [10..20, 30..40] in a 50-byte file.
        // Should produce 2 chunks: [0..30) and [30..50).
        let ranges = vec![
            Range { start: 10, end: 20 },
            Range { start: 30, end: 40 },
        ];
        let chunks = ranges_to_chunks(&ranges, 50);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].byte_offset, 0);
        assert_eq!(chunks[0].len, 30);
        assert_eq!(chunks[1].byte_offset, 30);
        assert_eq!(chunks[1].len, 20);
    }

    #[test]
    fn ranges_to_chunks_contiguous() {
        let ranges = vec![
            Range { start: 0, end: 10 },
            Range { start: 10, end: 20 },
            Range { start: 20, end: 30 },
        ];
        let chunks = ranges_to_chunks(&ranges, 30);
        assert_eq!(chunks.len(), 3);
        // Check contiguity.
        let mut pos = 0;
        for chunk in &chunks {
            assert_eq!(chunk.byte_offset, pos);
            pos += chunk.len;
        }
        assert_eq!(pos, 30);
    }
}
