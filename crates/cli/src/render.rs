//! Markdown rendering for the terminal, fed one streamed chunk at a time.
//!
//! Buffering is per block: prose lines are rendered as soon as each line
//! is complete, fenced code blocks and tables are held until they close.
//! In plain mode (no TTY, or NO_COLOR set) chunks pass through untouched.

use syntect::easy::HighlightLines;
use syntect::highlighting::{Theme, ThemeSet};
use syntect::parsing::SyntaxSet;
use syntect::util::as_24_bit_terminal_escaped;
use termimad::{FmtText, MadSkin};

pub enum Renderer {
    Plain,
    Rich(Box<Rich>),
}

pub struct Rich {
    skin: MadSkin,
    width: usize,
    syntaxes: SyntaxSet,
    theme: Theme,
    /// The incomplete last line of the current chunk.
    pending: String,
    block: Block,
}

enum Block {
    Prose,
    Code { lang: String, lines: Vec<String> },
    Table { lines: Vec<String> },
}

impl Renderer {
    /// Rich when stdout is a terminal and NO_COLOR is unset.
    pub fn for_stdout() -> Self {
        use std::io::IsTerminal;
        if std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none() {
            let width = termimad::crossterm::terminal::size()
                .map(|(w, _)| w as usize)
                .unwrap_or(80)
                .clamp(40, 120);
            Self::rich(width)
        } else {
            Self::Plain
        }
    }

    pub fn rich(width: usize) -> Self {
        Renderer::Rich(Box::new(Rich {
            skin: MadSkin::default_dark(),
            width,
            syntaxes: SyntaxSet::load_defaults_newlines(),
            theme: ThemeSet::load_defaults().themes["base16-ocean.dark"].clone(),
            pending: String::new(),
            block: Block::Prose,
        }))
    }

    pub fn is_rich(&self) -> bool {
        matches!(self, Renderer::Rich(_))
    }

    /// Takes a streamed chunk and returns what should be printed now.
    pub fn push(&mut self, chunk: &str) -> String {
        match self {
            Renderer::Plain => chunk.to_string(),
            Renderer::Rich(rich) => rich.push(chunk),
        }
    }

    /// Flushes anything still held: a partial line or an unclosed block.
    pub fn finish(&mut self) -> String {
        match self {
            Renderer::Plain => String::new(),
            Renderer::Rich(rich) => rich.finish(),
        }
    }
}

impl Rich {
    fn push(&mut self, chunk: &str) -> String {
        self.pending.push_str(chunk);
        let mut out = String::new();
        while let Some(newline) = self.pending.find('\n') {
            let line: String = self.pending.drain(..=newline).collect();
            out.push_str(&self.line(line.trim_end_matches('\n')));
        }
        out
    }

    fn finish(&mut self) -> String {
        let mut out = String::new();
        if !self.pending.is_empty() {
            let line = std::mem::take(&mut self.pending);
            out.push_str(&self.line(&line));
        }
        match std::mem::replace(&mut self.block, Block::Prose) {
            Block::Prose => {}
            Block::Code { lang, lines } => out.push_str(&self.code(&lang, &lines)),
            Block::Table { lines } => out.push_str(&self.markdown(&lines.join("\n"))),
        }
        out
    }

    /// One complete line; returns whatever it makes printable.
    fn line(&mut self, line: &str) -> String {
        match &mut self.block {
            Block::Code { lang, lines } => {
                if line.trim_start().starts_with("```") {
                    let (lang, lines) = (std::mem::take(lang), std::mem::take(lines));
                    self.block = Block::Prose;
                    self.code(&lang, &lines)
                } else {
                    lines.push(line.to_string());
                    String::new()
                }
            }
            Block::Table { lines } => {
                if line.trim_start().starts_with('|') {
                    lines.push(line.to_string());
                    return String::new();
                }
                let table = std::mem::take(lines);
                self.block = Block::Prose;
                let mut out = self.markdown(&table.join("\n"));
                out.push_str(&self.line(line));
                out
            }
            Block::Prose => {
                if let Some(rest) = line.trim_start().strip_prefix("```") {
                    self.block = Block::Code {
                        lang: rest.trim().to_string(),
                        lines: Vec::new(),
                    };
                    String::new()
                } else if line.trim_start().starts_with('|') {
                    self.block = Block::Table {
                        lines: vec![line.to_string()],
                    };
                    String::new()
                } else if line.trim().is_empty() {
                    "\n".to_string()
                } else {
                    self.markdown(line)
                }
            }
        }
    }

    fn markdown(&self, src: &str) -> String {
        FmtText::from(&self.skin, src, Some(self.width)).to_string()
    }

    fn code(&self, lang: &str, lines: &[String]) -> String {
        let syntax = self
            .syntaxes
            .find_syntax_by_token(lang)
            .unwrap_or_else(|| self.syntaxes.find_syntax_plain_text());
        let mut highlighter = HighlightLines::new(syntax, &self.theme);
        let mut out = String::new();
        for line in lines {
            let text = format!("{line}\n");
            let ranges = highlighter
                .highlight_line(&text, &self.syntaxes)
                .unwrap_or_else(|_| vec![(syntect::highlighting::Style::default(), text.as_str())]);
            out.push_str("  ");
            out.push_str(&as_24_bit_terminal_escaped(&ranges, false));
            out.push_str("\x1b[0m");
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "# Title\n\nSome **bold** and `code` text.\n\n- one\n- two\n\n```rust\nlet key = \"<<SECRET_1>>\";\n```\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\ndone\n";

    fn render_all(chunks: &[&str]) -> String {
        let mut renderer = Renderer::rich(60);
        let mut out = String::new();
        for chunk in chunks {
            out.push_str(&renderer.push(chunk));
        }
        out.push_str(&renderer.finish());
        out
    }

    fn strip_ansi(s: &str) -> String {
        let mut out = String::new();
        let mut chars = s.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '\x1b' {
                for c in chars.by_ref() {
                    if c.is_ascii_alphabetic() {
                        break;
                    }
                }
            } else {
                out.push(c);
            }
        }
        out
    }

    #[test]
    fn golden_render() {
        let rendered = render_all(&[SAMPLE]);
        assert_eq!(rendered, include_str!("../tests/golden/sample.ansi"));
    }

    #[test]
    fn plain_mode_passes_chunks_through_untouched() {
        let mut renderer = Renderer::Plain;
        let mut out = String::new();
        for chunk in ["# Ti", "tle\n\n```rust\nlet x", " = 1;\n```\n"] {
            out.push_str(&renderer.push(chunk));
        }
        out.push_str(&renderer.finish());
        assert_eq!(out, "# Title\n\n```rust\nlet x = 1;\n```\n");
    }

    #[test]
    fn chunking_does_not_change_the_output() {
        let whole = render_all(&[SAMPLE]);
        let bytes: Vec<&str> = SAMPLE
            .char_indices()
            .map(|(i, c)| &SAMPLE[i..i + c.len_utf8()])
            .collect();
        assert_eq!(render_all(&bytes), whole);
        let (a, b) = SAMPLE.split_at(SAMPLE.find("<<SEC").unwrap() + 5);
        assert_eq!(render_all(&[a, b]), whole);
    }

    #[test]
    fn code_block_is_held_until_it_closes() {
        let mut renderer = Renderer::rich(60);
        // The placeholder is split across chunks inside the block; the
        // agent rehydrates before we see it, so here it is the plain value.
        let first = renderer.push("```rust\nlet key = \"sk-ant-api03-REAL");
        assert_eq!(first, "", "nothing printed until the fence closes");
        let second = renderer.push("VALUE\";\n");
        assert_eq!(second, "");
        let third = renderer.push("```\nafter\n");
        let text = strip_ansi(&third);
        assert!(
            text.contains("let key = \"sk-ant-api03-REALVALUE\";"),
            "{text:?}"
        );
        assert!(text.ends_with("after\n"), "{text:?}");
    }

    #[test]
    fn table_is_held_until_a_non_table_line() {
        let mut renderer = Renderer::rich(60);
        assert_eq!(renderer.push("| a | b |\n|---|---|\n"), "");
        assert_eq!(renderer.push("| 1 | 2 |\n"), "");
        let out = strip_ansi(&renderer.push("text\n"));
        let a = out.find('a').unwrap();
        let t = out.find("text").unwrap();
        assert!(a < t, "table rendered before the following text: {out:?}");
        assert!(out.contains('1') && out.contains('2'));
    }

    #[test]
    fn prose_streams_line_by_line() {
        let mut renderer = Renderer::rich(60);
        assert_eq!(
            renderer.push("partial"),
            "",
            "held until the line completes"
        );
        let out = strip_ansi(&renderer.push(" line\nnext"));
        assert_eq!(out, "partial line\n");
        assert_eq!(strip_ansi(&renderer.finish()), "next\n");
    }

    /// Rewrites the golden file. Run with
    /// `cargo test -p airlok --bin airlok -- --ignored dump_golden` after
    /// checking the new render by eye.
    #[test]
    #[ignore]
    fn dump_golden() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden/sample.ansi");
        std::fs::write(path, render_all(&[SAMPLE])).unwrap();
    }
}
