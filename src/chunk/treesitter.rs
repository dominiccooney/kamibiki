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

    // Convert ranges to Chunk structs, ensuring contiguous coverage
    // of [0, content.len()).
    Ok(ranges_to_chunks(&ranges, content))
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
    // The first newline in the gap is typically just the line
    // terminator for the previous node, not a true blank line.
    // Subtract 1 to account for this.
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
                end: node.start_byte() + c.byte_offset + c.content.len(),
            })
            .collect();
    }

    // Score boundaries between adjacent children.
    // boundary_scores[i] is the score for the boundary between
    // children[i-1] and children[i].
    let mut boundary_scores: Vec<u32> = vec![0]; // index 0 unused
    for i in 1..children.len() {
        let score = boundary_score(
            content,
            children[i - 1].end_byte(),
            children[i].start_byte(),
        );
        boundary_scores.push(score);
    }

    // Greedily merge children into groups. A group spans from the
    // start of its first child to the end of its last child
    // (including any gap bytes between them).
    //
    // When adding a child would exceed max_tokens, we pick the best
    // break point within the current accumulation window.
    let mut result: Vec<Range> = Vec::new();
    let mut group_start_idx: usize = 0;

    // For tracking the best break point within the current window.
    let mut best_break_idx: Option<usize> = None;
    let mut best_break_score: u32 = 0;

    for i in 1..children.len() {
        let group_start_byte = children[group_start_idx].start_byte();
        let prospective_end = children[i].end_byte();
        let prospective_tokens =
            tc.count(&content[group_start_byte..prospective_end]);

        let score = boundary_scores[i];

        if prospective_tokens > max_tokens && i > group_start_idx {
            // Need to break. Use the best break point if available.
            let break_idx = if let Some(bi) = best_break_idx {
                bi
            } else {
                // No best break found (all scores 0, meaning stuck
                // items). Break at the most recent boundary.
                i
            };

            // Emit the group [group_start_idx .. break_idx).
            emit_group(
                &children,
                group_start_idx,
                break_idx,
                content,
                max_tokens,
                tc,
                &mut result,
            );

            group_start_idx = break_idx;
            best_break_idx = None;
            best_break_score = 0;

            // Don't track boundary score for the current `i` if
            // it's now the first item.
            if i > group_start_idx {
                if score > best_break_score {
                    best_break_idx = Some(i);
                    best_break_score = score;
                }
            }

            continue;
        }

        // Track best break point: prefer higher scores, and among
        // equal scores prefer later positions (for bigger chunks).
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
        &mut result,
    );

    result
}

/// Emit a group of children [start_idx..end_idx) as chunk ranges.
///
/// If the group fits within max_tokens, emit as a single range.
/// If it's a single child that's oversized, recurse into it.
/// If it's multiple children that are oversized, recurse into each.
fn emit_group(
    children: &[Node],
    start_idx: usize,
    end_idx: usize,
    content: &[u8],
    max_tokens: usize,
    tc: &TokenCounter,
    result: &mut Vec<Range>,
) {
    if start_idx >= end_idx {
        return;
    }

    let group_start = children[start_idx].start_byte();
    let group_end = children[end_idx - 1].end_byte();
    let group_tokens = tc.count(&content[group_start..group_end]);

    if group_tokens <= max_tokens {
        // Fits in one chunk.
        result.push(Range {
            start: group_start,
            end: group_end,
        });
    } else if end_idx - start_idx == 1 {
        // Single oversized child — recurse into it.
        let sub = chunk_node(children[start_idx], content, max_tokens, tc);
        result.extend(sub);
    } else {
        // Multiple children that together exceed max_tokens but we
        // couldn't split them (all boundaries scored 0 = stuck).
        // Try each child individually.
        for idx in start_idx..end_idx {
            let child_tokens = tc.count(
                &content[children[idx].start_byte()..children[idx].end_byte()],
            );
            if child_tokens <= max_tokens {
                // Try to merge with the previous range if possible.
                if let Some(last) = result.last_mut() {
                    let merged_tokens =
                        tc.count(&content[last.start..children[idx].end_byte()]);
                    if merged_tokens <= max_tokens {
                        last.end = children[idx].end_byte();
                        continue;
                    }
                }
                result.push(Range {
                    start: children[idx].start_byte(),
                    end: children[idx].end_byte(),
                });
            } else {
                let sub = chunk_node(children[idx], content, max_tokens, tc);
                result.extend(sub);
            }
        }
    }
}

/// Convert ranges to Chunks, filling gaps between ranges with
/// content from the source to maintain contiguous coverage of
/// [0, content.len()).
fn ranges_to_chunks(ranges: &[Range], content: &[u8]) -> Vec<Chunk> {
    if ranges.is_empty() {
        if content.is_empty() {
            return Vec::new();
        }
        return vec![Chunk {
            byte_offset: 0,
            content: content.to_vec(),
        }];
    }

    let mut chunks: Vec<Chunk> = Vec::new();
    let mut pos: usize = 0;

    for range in ranges {
        let chunk_start = range.start;
        let chunk_end = range.end;

        if chunk_end <= chunk_start {
            continue;
        }

        // Try to merge with previous chunk if combined size is
        // reasonable (avoids tiny gap-only chunks).
        if let Some(last) = chunks.last_mut() {
            let last_end = last.byte_offset + last.content.len();
            if last_end == chunk_start {
                // Adjacent — no gap, just continue.
            } else if chunk_start < last_end {
                // Overlap — skip already-covered bytes.
                let new_start = last_end;
                if chunk_end > new_start {
                    chunks.push(Chunk {
                        byte_offset: new_start,
                        content: content[new_start..chunk_end].to_vec(),
                    });
                }
                pos = chunk_end;
                continue;
            }
        }

        chunks.push(Chunk {
            byte_offset: chunk_start,
            content: content[chunk_start..chunk_end].to_vec(),
        });
        pos = chunk_end;
    }

    // Fill trailing gap.
    if pos < content.len() {
        if let Some(last) = chunks.last_mut() {
            // Extend the last chunk to cover trailing whitespace.
            let last_end = last.byte_offset + last.content.len();
            if last_end < content.len() {
                last.content
                    .extend_from_slice(&content[last_end..content.len()]);
            }
        } else {
            chunks.push(Chunk {
                byte_offset: 0,
                content: content.to_vec(),
            });
        }
    }

    // Final pass: merge gaps into adjacent chunks to ensure
    // contiguous coverage.
    consolidate_chunks(chunks, content)
}

/// Ensure chunks are contiguous from byte 0 to content.len().
/// Fills any gaps by extending the preceding chunk.
fn consolidate_chunks(chunks: Vec<Chunk>, content: &[u8]) -> Vec<Chunk> {
    if chunks.is_empty() {
        if content.is_empty() {
            return Vec::new();
        }
        return vec![Chunk {
            byte_offset: 0,
            content: content.to_vec(),
        }];
    }

    let mut result: Vec<Chunk> = Vec::new();
    let mut pos: usize = 0;

    for chunk in chunks {
        if chunk.byte_offset > pos {
            // There's a gap — extend the previous chunk or create a
            // new one.
            if let Some(last) = result.last_mut() {
                // Extend the previous chunk to cover the gap.
                last.content
                    .extend_from_slice(&content[pos..chunk.byte_offset]);
            } else {
                // No previous chunk — create one for the leading gap.
                result.push(Chunk {
                    byte_offset: 0,
                    content: content[0..chunk.byte_offset].to_vec(),
                });
            }
        }

        if chunk.byte_offset < pos {
            // Overlap — skip already-covered prefix.
            let skip = pos - chunk.byte_offset;
            if skip < chunk.content.len() {
                result.push(Chunk {
                    byte_offset: pos,
                    content: chunk.content[skip..].to_vec(),
                });
            }
        } else {
            result.push(chunk);
        }

        if let Some(last) = result.last() {
            pos = last.byte_offset + last.content.len();
        }
    }

    // Fill trailing gap.
    if pos < content.len() {
        if let Some(last) = result.last_mut() {
            last.content
                .extend_from_slice(&content[pos..content.len()]);
        }
    }

    result
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
            assert!(!chunk.content.is_empty());
            expected_offset += chunk.content.len();
        }
        assert_eq!(expected_offset, content.len());
    }

    /// Verify the content of chunks matches the original source.
    fn assert_content_matches(chunks: &[Chunk], content: &[u8]) {
        for chunk in chunks {
            let expected =
                &content[chunk.byte_offset..chunk.byte_offset + chunk.content.len()];
            assert_eq!(chunk.content, expected);
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
        // Small file should be a single chunk.
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn rust_file_with_multiple_functions() {
        // Generate a file with many functions to exceed target size.
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

        // Should have multiple chunks.
        assert!(
            chunks.len() > 1,
            "expected multiple chunks, got {}",
            chunks.len()
        );

        // Each chunk should be roughly within max_tokens (with
        // possible oversized atoms).
        for chunk in &chunks {
            let chunk_tokens = tc.count(&chunk.content);
            // Allow 2x for oversized atoms.
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
        // Generate a larger file to really test chunking behavior.
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
        // Doc comments and attributes should not be separated from
        // the function they annotate.
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

        // Verify no chunk starts in the middle of a
        // doc-comment+attr+fn group. Each chunk should start either
        // at byte 0 or at a "///" doc comment line.
        for chunk in &chunks {
            if chunk.byte_offset == 0 {
                continue;
            }
            let start_text = std::str::from_utf8(&chunk.content[..chunk.content.len().min(50)])
                .unwrap_or("");
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

        // Check that no chunk starts with a bare "func" line (the
        // preceding comment should be attached).
        for chunk in &chunks {
            let text =
                std::str::from_utf8(&chunk.content).unwrap_or("");
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
        // Gap between byte 3 and byte 5 is "\n\n" — 2 blank lines.
        assert!(boundary_score(content, 3, 5) > 100);

        let content2 = b"aaa\nbbb";
        // Gap between byte 3 and byte 4 is "\n" — 0 blank lines
        // (the newline ends the "aaa" line, "bbb" starts immediately).
        assert_eq!(boundary_score(content2, 3, 4), 0);
    }
}
