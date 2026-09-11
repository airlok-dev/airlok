//! Diffs for write confirmations: line numbers, the three lines of
//! context the tools produce, syntax highlighting with the markdown theme,
//! and +/- gutters. The caller pages them at [`PAGE`] lines.

use std::path::Path;

use syntect::easy::HighlightLines;
use syntect::highlighting::Style;
use syntect::util::as_24_bit_terminal_escaped;

use crate::render::Highlighting;

/// Lines shown before the rest waits for `v`.
pub const PAGE: usize = 40;

/// The line that stands for the part of a diff not shown yet.
pub fn more_lines(rest: usize) -> String {
    format!("... {rest} more lines, [v] to view all")
}

/// One line of a unified diff, numbered.
enum Row<'a> {
    Hunk(&'a str),
    Line {
        gutter: char,
        number: usize,
        text: &'a str,
    },
}

/// Renders the unified diff of `path` as display lines, without newlines.
/// The title is the path in the diff's own header, relative to the
/// working directory, else `path`. With `colors`, the code is highlighted
/// and the gutters coloured; with `None` it is plain text.
pub fn render(path: &str, unified: &str, colors: Option<&Highlighting>) -> Vec<String> {
    let path = unified
        .lines()
        .take_while(|line| !line.starts_with("@@"))
        .find_map(|line| line.strip_prefix("+++ "))
        .map(|named| named.strip_prefix("b/").unwrap_or(named))
        .unwrap_or(path);
    let rows = parse(unified);
    let width = rows
        .iter()
        .filter_map(|row| match row {
            Row::Line { number, .. } => Some(number.to_string().len()),
            Row::Hunk(_) => None,
        })
        .max()
        .unwrap_or(1);
    let mut highlighter = colors.map(|h| {
        let syntax = Path::new(path)
            .extension()
            .and_then(|ext| ext.to_str())
            .and_then(|ext| h.syntaxes.find_syntax_by_extension(ext))
            .unwrap_or_else(|| h.syntaxes.find_syntax_plain_text());
        (h, HighlightLines::new(syntax, &h.theme))
    });
    let paint = |code: &str, text: &str| match colors {
        Some(_) => format!("\x1b[{code}m{text}\x1b[0m"),
        None => text.to_string(),
    };
    let mut out = vec![paint("1", path)];
    for row in rows {
        let line = match row {
            Row::Hunk(header) => paint("36", header),
            Row::Line {
                gutter,
                number,
                text,
            } => {
                let number = format!("{number:>width$}");
                match &mut highlighter {
                    None => format!("{number} {gutter} {text}"),
                    Some((h, highlighter)) => {
                        let color = match gutter {
                            '+' => "32",
                            '-' => "31",
                            _ => "0",
                        };
                        format!(
                            "{} {} {}\x1b[0m",
                            paint("2", &number),
                            paint(color, &gutter.to_string()),
                            highlight(highlighter, h, text)
                        )
                    }
                }
            }
        };
        out.push(line);
    }
    out
}

/// Numbers each line of the diff: removals with the old file's line,
/// everything else with the new file's. The file headers before the first
/// hunk and "\ No newline at end of file" markers are dropped.
fn parse(unified: &str) -> Vec<Row<'_>> {
    let (mut old, mut new) = (0, 0);
    let mut in_hunks = false;
    let mut rows = Vec::new();
    for line in unified.lines() {
        if line.starts_with("@@") {
            (old, new) = hunk_starts(line);
            in_hunks = true;
            rows.push(Row::Hunk(line));
            continue;
        }
        if !in_hunks || line.starts_with('\\') {
            continue;
        }
        let mut chars = line.chars();
        let gutter = chars.next().unwrap_or(' ');
        let text = chars.as_str();
        let number = match gutter {
            '-' => {
                old += 1;
                old - 1
            }
            '+' => {
                new += 1;
                new - 1
            }
            _ => {
                old += 1;
                new += 1;
                new - 1
            }
        };
        rows.push(Row::Line {
            gutter,
            number,
            text,
        });
    }
    rows
}

/// The first old and new line numbers in `@@ -a,b +c,d @@`.
fn hunk_starts(header: &str) -> (usize, usize) {
    let mut parts = header.split_whitespace().skip(1);
    let mut start = |sign: char| {
        parts
            .next()
            .and_then(|part| part.strip_prefix(sign))
            .and_then(|range| range.split(',').next())
            .and_then(|n| n.parse().ok())
            .unwrap_or(1)
    };
    let old = start('-');
    (old, start('+'))
}

fn highlight(highlighter: &mut HighlightLines, h: &Highlighting, text: &str) -> String {
    let line = format!("{text}\n");
    let ranges = highlighter
        .highlight_line(&line, &h.syntaxes)
        .unwrap_or_else(|_| vec![(Style::default(), line.as_str())]);
    let escaped = as_24_bit_terminal_escaped(&ranges, false);
    escaped.strip_suffix('\n').unwrap_or(&escaped).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::highlighting;
    use airlok_core::tools::unified_diff;

    /// Lines 10 to 25 changed and line 50 removed, in a 60-line file.
    fn sample() -> String {
        let old: String = (1..=60).map(|i| format!("line {i}\n")).collect();
        let new: String = (1..=60)
            .filter(|i| *i != 50)
            .map(|i| match i {
                10..=25 => format!("LINE {i}\n"),
                _ => format!("line {i}\n"),
            })
            .collect();
        unified_diff("notes.txt", &old, &new)
    }

    /// What the confirmation prints first: a page, then the marker.
    fn first_page(lines: &[String]) -> String {
        let shown = lines.len().min(PAGE);
        let mut out: String = lines[..shown].iter().map(|l| format!("{l}\n")).collect();
        if lines.len() > shown {
            out.push_str(&more_lines(lines.len() - shown));
            out.push('\n');
        }
        out
    }

    #[test]
    fn golden_diff_has_line_numbers_context_and_the_paging_marker() {
        let lines = render("notes.txt", &sample(), None);
        assert_eq!(first_page(&lines), include_str!("../tests/golden/diff.txt"));
    }

    #[test]
    fn colors_mark_the_gutters_and_highlight_the_code() {
        let unified = unified_diff(
            "main.rs",
            "fn main() {}\n",
            "fn main() {\n    println!(\"hi\");\n}\n",
        );
        let lines = render("main.rs", &unified, Some(highlighting()));
        let added = lines.iter().find(|l| l.contains("println")).unwrap();
        assert!(added.contains("\x1b[32m+\x1b[0m"), "{added:?}");
        assert!(added.contains("\x1b[38;2;"), "highlighted: {added:?}");
        let removed = lines.iter().find(|l| l.contains("\x1b[31m-")).unwrap();
        assert!(removed.starts_with("\x1b[2m1\x1b[0m"), "{removed:?}");
    }

    #[test]
    fn a_removed_line_that_looks_like_a_header_is_still_a_line() {
        let unified = unified_diff("q.sql", "-- note\nselect 1;\n", "select 1;\n");
        let lines = render("q.sql", &unified, None);
        assert!(lines.contains(&"1 - -- note".to_string()), "{lines:?}");
        assert!(lines.contains(&"1   select 1;".to_string()), "{lines:?}");
    }

    /// Rewrites the golden file. Run with
    /// `cargo test -p airlok --bin airlok -- --ignored dump_diff_golden`
    /// after checking the new output by eye.
    #[test]
    #[ignore]
    fn dump_diff_golden() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden/diff.txt");
        std::fs::write(path, first_page(&render("notes.txt", &sample(), None))).unwrap();
    }
}
