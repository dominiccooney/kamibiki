//! Formatting helpers for displaying code snippets with line numbers.

/// Compute the 1-based line number for a given byte offset within file content.
///
/// Counts the number of newline bytes (`\n`) before `byte_offset` and returns
/// that count plus one (for 1-based indexing).
pub fn start_line_for_offset(content: &[u8], byte_offset: usize) -> usize {
    let limit = byte_offset.min(content.len());
    content[..limit].iter().filter(|&&b| b == b'\n').count() + 1
}

/// Format a code snippet with 1-based line numbers.
///
/// Each line is prefixed with `<line_number> | `, where line numbers are
/// right-aligned to the width of the largest line number in the snippet.
///
/// ```text
///  1 | fn main() {
///  2 |     println!("hello");
///  3 | }
/// ```
pub fn format_with_line_numbers(content: &str, start_line: usize) -> String {
    let lines: Vec<&str> = content.split('\n').collect();

    // If the content ends with a newline, split produces a trailing empty
    // string that is not a real line — trim it.
    let line_count = if content.ends_with('\n') && lines.len() > 1 {
        lines.len() - 1
    } else {
        lines.len()
    };

    let last_line_number = start_line + line_count.saturating_sub(1);
    let width = digit_width(last_line_number);

    let mut out = String::with_capacity(content.len() + line_count * (width + 4));
    for (i, line) in lines[..line_count].iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        let line_no = start_line + i;
        // Right-align the line number.
        write_right_aligned(&mut out, line_no, width);
        out.push_str(" | ");
        out.push_str(line);
    }

    // Preserve trailing newline if the original content had one.
    if content.ends_with('\n') {
        out.push('\n');
    }

    out
}

fn digit_width(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut w = 0;
    let mut v = n;
    while v > 0 {
        w += 1;
        v /= 10;
    }
    w
}

fn write_right_aligned(out: &mut String, n: usize, width: usize) {
    let dw = digit_width(n);
    for _ in 0..(width.saturating_sub(dw)) {
        out.push(' ');
    }
    // Format the number directly.
    let start = out.len();
    write_usize(out, n);
    debug_assert_eq!(out.len() - start, dw);
}

fn write_usize(out: &mut String, n: usize) {
    if n >= 10 {
        write_usize(out, n / 10);
    }
    out.push((b'0' + (n % 10) as u8) as char);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_start_line_for_offset() {
        let content = b"line1\nline2\nline3\n";
        assert_eq!(start_line_for_offset(content, 0), 1);
        assert_eq!(start_line_for_offset(content, 5), 1); // at the '\n'
        assert_eq!(start_line_for_offset(content, 6), 2); // after first '\n'
        assert_eq!(start_line_for_offset(content, 12), 3); // after second '\n'
    }

    #[test]
    fn test_start_line_offset_beyond_content() {
        let content = b"a\nb\n";
        // Should clamp to content length.
        assert_eq!(start_line_for_offset(content, 100), 3);
    }

    #[test]
    fn test_format_single_line() {
        let result = format_with_line_numbers("import os", 1);
        assert_eq!(result, "1 | import os");
    }

    #[test]
    fn test_format_multiple_lines() {
        let result = format_with_line_numbers("a\nb\nc\n", 1);
        assert_eq!(result, "1 | a\n2 | b\n3 | c\n");
    }

    #[test]
    fn test_format_with_start_offset() {
        let result = format_with_line_numbers("fn foo() {\n}\n", 42);
        assert_eq!(result, "42 | fn foo() {\n43 | }\n");
    }

    #[test]
    fn test_format_line_number_alignment() {
        let result = format_with_line_numbers("a\nb\nc\n", 98);
        assert_eq!(result, " 98 | a\n 99 | b\n100 | c\n");
    }

    #[test]
    fn test_format_no_trailing_newline() {
        let result = format_with_line_numbers("hello\nworld", 5);
        assert_eq!(result, "5 | hello\n6 | world");
    }

    #[test]
    fn test_digit_width() {
        assert_eq!(digit_width(0), 1);
        assert_eq!(digit_width(1), 1);
        assert_eq!(digit_width(9), 1);
        assert_eq!(digit_width(10), 2);
        assert_eq!(digit_width(99), 2);
        assert_eq!(digit_width(100), 3);
        assert_eq!(digit_width(999), 3);
        assert_eq!(digit_width(1000), 4);
    }

    #[test]
    fn test_empty_content() {
        let result = format_with_line_numbers("", 1);
        assert_eq!(result, "1 | ");
    }

    #[test]
    fn test_just_newline() {
        let result = format_with_line_numbers("\n", 1);
        assert_eq!(result, "1 | \n");
    }
}
