use anyhow::Result;

use super::{Chunk, Chunker, TokenCounter};

/// A fallback chunker that splits on newlines, preferring breaks at
/// blank-line boundaries and low indent levels. Uses blank lines as
/// the primary structural signal and indentation depth as a secondary
/// proxy for AST depth.
pub struct LinesChunker {
    pub(crate) token_counter: TokenCounter,
}

impl Chunker for LinesChunker {
    fn chunk(&self, _path: &str, content: &[u8], max_tokens: usize) -> Result<Vec<Chunk>> {
        Ok(chunk_by_lines(content, max_tokens, &self.token_counter))
    }
}

/// Information about a single line in the source content.
#[derive(Debug, Clone, Copy)]
struct LineInfo {
    /// Byte offset where this line starts (inclusive).
    start: usize,
    /// Byte offset where this line ends (exclusive, includes the '\n'
    /// if present).
    end: usize,
    /// Number of leading whitespace characters (spaces/tabs, with
    /// tabs counting as 4).
    indent: usize,
    /// Whether this line is blank (only whitespace).
    is_blank: bool,
}

/// Compute the break quality score for the boundary between two
/// adjacent lines (i.e., the break point at `prev.end`).
///
/// Higher score = better break point.
/// - Blank line boundaries get high scores (100+).
/// - Low indent on the next line gets medium scores.
/// - Deep indent = reluctant to break here.
fn break_score(prev: &LineInfo, next: &LineInfo) -> u32 {
    let mut score: u32 = 0;

    // Blank line bonus: if either the previous or next line is blank,
    // this is a paragraph boundary.
    if prev.is_blank || next.is_blank {
        score += 100;
        // Extra bonus for multiple consecutive blank lines (counted
        // by the caller if needed — here we just check the immediate
        // neighbors).
    }

    // Indent-level score: lower indent on the next line means we're
    // at a higher structural level. indent 0 → +50, indent 1 → +45,
    // indent 10+ → +0.
    let indent_score = 50u32.saturating_sub(next.indent as u32 * 5);
    score += indent_score;

    score
}

/// Parse content into line info records.
fn parse_lines(content: &[u8]) -> Vec<LineInfo> {
    if content.is_empty() {
        return Vec::new();
    }

    let mut lines = Vec::new();
    let mut pos = 0;

    while pos < content.len() {
        let start = pos;
        // Find end of line.
        while pos < content.len() && content[pos] != b'\n' {
            pos += 1;
        }
        if pos < content.len() {
            pos += 1; // consume the '\n'
        }
        let end = pos;

        // Compute indent and blank status.
        let line_content = &content[start..end];
        let mut indent = 0;
        let mut all_blank = true;
        for &b in line_content {
            match b {
                b' ' if all_blank => indent += 1,
                b'\t' if all_blank => indent += 4,
                b'\n' | b'\r' => {}
                _ if all_blank => {
                    all_blank = false;
                }
                _ => {}
            }
        }
        // A line of only whitespace (including empty line "\n") is blank.
        let is_blank = all_blank;

        lines.push(LineInfo {
            start,
            end,
            indent,
            is_blank,
        });
    }

    lines
}

/// Split `content` into chunks by newline boundaries, preferring
/// breaks at blank-line boundaries and low indent levels. Uses the
/// provided `TokenCounter` for accurate token counting via prefix
/// sums over per-line token counts.
///
/// Guarantees: chunks are non-overlapping, contiguous, start at byte
/// 0, and cover all of `content`.
pub fn chunk_by_lines(content: &[u8], max_tokens: usize, tc: &TokenCounter) -> Vec<Chunk> {
    if content.is_empty() {
        return Vec::new();
    }

    let lines = parse_lines(content);
    if lines.is_empty() {
        return Vec::new();
    }

    // Precompute per-line token counts and build a prefix sum for
    // fast range queries. This is an approximation (BPE tokenization
    // isn't perfectly additive across line boundaries) but is far
    // more accurate than a fixed bytes-per-token ratio.
    let line_tokens: Vec<usize> = lines
        .iter()
        .map(|l| tc.count(&content[l.start..l.end]))
        .collect();
    let prefix: Vec<usize> = {
        let mut p = vec![0usize; lines.len() + 1];
        for i in 0..lines.len() {
            p[i + 1] = p[i] + line_tokens[i];
        }
        p
    };

    let mut chunks = Vec::new();
    let mut chunk_start_line: usize = 0;
    let mut chunk_start_byte: usize = 0;

    // Track the best break point seen within the current
    // accumulation window.
    let mut best_break_line: Option<usize> = None;
    let mut best_break_score: u32 = 0;

    for i in 1..lines.len() {
        // Token count from chunk_start_line through line i (inclusive).
        let prospective_tokens = prefix[i + 1] - prefix[chunk_start_line];

        // Score the boundary between line i-1 and line i.
        let score = break_score(&lines[i - 1], &lines[i]);

        if prospective_tokens > max_tokens {
            // We've exceeded the budget. Break at the best point we
            // found, or at the previous line boundary if no good
            // break was found.
            if let Some(break_line) = best_break_line {
                let break_byte = lines[break_line].end;
                if break_byte > chunk_start_byte {
                    chunks.push(Chunk {
                        byte_offset: chunk_start_byte,
                        content: content[chunk_start_byte..break_byte].to_vec(),
                    });
                    chunk_start_byte = break_byte;
                    chunk_start_line = break_line + 1;
                    best_break_line = None;
                    best_break_score = 0;
                    // Re-scan from the new start to find break
                    // points. But since we're doing a forward pass,
                    // we just continue and will pick up new break
                    // points naturally.
                    continue;
                }
            }

            // Fallback: break at the previous line boundary.
            if i > chunk_start_line {
                let break_byte = lines[i - 1].start;
                if break_byte > chunk_start_byte {
                    // Break just before line i (so line i-1's end is
                    // the last included).
                    let prev_end = lines[i - 1].start;
                    if prev_end > chunk_start_byte {
                        chunks.push(Chunk {
                            byte_offset: chunk_start_byte,
                            content: content[chunk_start_byte..prev_end].to_vec(),
                        });
                        chunk_start_byte = prev_end;
                        chunk_start_line = i - 1;
                        best_break_line = None;
                        best_break_score = 0;
                        continue;
                    }
                }
            }

            // If we get here, a single line exceeds the budget. We
            // can't split further, so just let it accumulate and
            // it'll be emitted when we find a valid break or at the
            // end.
        }

        // Track best break point within current window. We want the
        // best-scoring boundary that's at least one line into the
        // chunk.
        if i > chunk_start_line && (best_break_line.is_none() || score >= best_break_score) {
            best_break_line = Some(i - 1);
            best_break_score = score;
        }
    }

    // Emit final chunk.
    if chunk_start_byte < content.len() {
        chunks.push(Chunk {
            byte_offset: chunk_start_byte,
            content: content[chunk_start_byte..content.len()].to_vec(),
        });
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::LazyLock;

    static TEST_TC: LazyLock<TokenCounter> = LazyLock::new(|| {
        TokenCounter::for_voyage().expect("failed to load tokenizer for tests")
    });

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
            assert_eq!(chunk.byte_offset, expected_offset);
            assert!(!chunk.content.is_empty());
            expected_offset += chunk.content.len();
        }
        assert_eq!(expected_offset, content.len());
    }

    #[test]
    fn empty_content() {
        let chunks = chunk_by_lines(b"", 100, &TEST_TC);
        assert!(chunks.is_empty());
    }

    #[test]
    fn single_line_under_target() {
        let content = b"hello world\n";
        let chunks = chunk_by_lines(content, 100, &TEST_TC);
        assert_contiguous(&chunks, content);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, content);
    }

    #[test]
    fn multiple_lines_under_target() {
        let content = b"line 1\nline 2\nline 3\n";
        let chunks = chunk_by_lines(content, 100, &TEST_TC);
        assert_contiguous(&chunks, content);
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn splits_at_target_boundary() {
        // Each line is 10 bytes ("123456789\n"). With a small token
        // budget, should produce multiple chunks.
        let content = b"123456789\n123456789\n123456789\n123456789\n123456789\n";
        // Use a token budget of 5 — each line tokenizes to a few
        // tokens, so this should force splits.
        let chunks = chunk_by_lines(content, 5, &TEST_TC);
        assert_contiguous(&chunks, content);
        assert!(chunks.len() >= 2, "expected multiple chunks, got {}", chunks.len());
    }

    #[test]
    fn content_without_trailing_newline() {
        let content = b"line 1\nline 2";
        let chunks = chunk_by_lines(content, 100, &TEST_TC);
        assert_contiguous(&chunks, content);
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn single_huge_line() {
        // One line exceeding target — can't split further, so it
        // becomes a single chunk.
        let content = b"a]".repeat(2000);
        let mut input = content.clone();
        input.push(b'\n');
        let chunks = chunk_by_lines(&input, 10, &TEST_TC);
        assert_contiguous(&chunks, &input);
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn many_small_lines() {
        // 100 lines of "x\n" (2 bytes each), small token budget.
        let content: Vec<u8> = "x\n".repeat(100).into_bytes();
        let chunks = chunk_by_lines(&content, 5, &TEST_TC);
        assert_contiguous(&chunks, &content);
        // Should produce multiple chunks.
        assert!(chunks.len() >= 5, "expected many chunks, got {}", chunks.len());
    }

    #[test]
    fn prefers_blank_line_breaks() {
        // Two paragraphs separated by a blank line. With a budget
        // that forces a split, the chunker should prefer the blank
        // line boundary over splitting inside a paragraph.
        let content = b"line 1 aaaa bbbb cccc\nline 2 dddd eeee ffff\nline 3 gggg hhhh iiii\n\nline 4 jjjj kkkk llll\nline 5 mmmm nnnn oooo\nline 6 pppp qqqq rrrr\n";
        // Use a budget that fits roughly one paragraph but not both.
        let first_para = b"line 1 aaaa bbbb cccc\nline 2 dddd eeee ffff\nline 3 gggg hhhh iiii\n\n";
        let first_para_tokens = TEST_TC.count(first_para);
        let total_tokens = TEST_TC.count(content);
        // Budget should fit the first paragraph but not the whole content.
        assert!(first_para_tokens < total_tokens, "test content too small");
        let budget = first_para_tokens + 1;
        let chunks = chunk_by_lines(content, budget, &TEST_TC);
        assert_contiguous(&chunks, content);
        assert!(chunks.len() >= 2, "expected at least 2 chunks");
        // The first chunk should end at or near the blank line.
        let first_chunk_str = std::str::from_utf8(&chunks[0].content).unwrap();
        assert!(
            first_chunk_str.contains("line 3") && !first_chunk_str.contains("line 6"),
            "expected split near blank line boundary, first chunk: {:?}",
            first_chunk_str
        );
    }

    #[test]
    fn prefers_low_indent_breaks() {
        // A block of code where indent level varies. With a small
        // budget, should prefer breaking at low indent.
        let content = b"def foo():\n    line1\n    line2\n    line3\ndef bar():\n    line4\n    line5\n";
        let total_tokens = TEST_TC.count(content);
        let budget = total_tokens / 2 + 1;
        let chunks = chunk_by_lines(content, budget, &TEST_TC);
        assert_contiguous(&chunks, content);
        // Should prefer splitting between the two function defs.
        if chunks.len() >= 2 {
            let first_chunk_str = std::str::from_utf8(&chunks[0].content).unwrap();
            assert!(
                first_chunk_str.contains("foo") && !first_chunk_str.contains("bar"),
                "expected split between functions, got: {:?}",
                first_chunk_str
            );
        }
    }

    #[test]
    fn chunker_trait_uses_token_budget() {
        let chunker = LinesChunker {
            token_counter: TEST_TC.clone(),
        };
        // Use a small token budget to force multiple chunks.
        let content = b"123456789\n123456789\n123456789\n123456789\n123456789\n123456789\n";
        let chunks = chunker.chunk("test.txt", content, 5).unwrap();
        assert_contiguous(&chunks, content);
        assert!(chunks.len() >= 2, "expected multiple chunks with small token budget");
    }

    #[test]
    fn parse_lines_basic() {
        let content = b"hello\n  world\n\n    deep\n";
        let lines = parse_lines(content);
        assert_eq!(lines.len(), 4);
        assert_eq!(lines[0].indent, 0);
        assert!(!lines[0].is_blank);
        assert_eq!(lines[1].indent, 2);
        assert!(!lines[1].is_blank);
        assert!(lines[2].is_blank);
        assert_eq!(lines[3].indent, 4);
    }

    #[test]
    fn break_score_blank_line_higher() {
        let blank = LineInfo { start: 0, end: 1, indent: 0, is_blank: true };
        let normal = LineInfo { start: 1, end: 10, indent: 0, is_blank: false };
        let deep = LineInfo { start: 10, end: 20, indent: 8, is_blank: false };

        let score_at_blank = break_score(&blank, &normal);
        let score_at_deep = break_score(&normal, &deep);
        assert!(
            score_at_blank > score_at_deep,
            "blank line break ({}) should score higher than deep indent break ({})",
            score_at_blank,
            score_at_deep
        );
    }
}
