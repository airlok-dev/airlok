//! Markdown rendering for the terminal, fed one streamed chunk at a time.
//!
//! Buffering is per block. A paragraph or list item is rendered when a
//! blank line or the next block marker arrives, or the message ends, so
//! markdown sees the whole block however the stream was cut. Headings and
//! rules render at the end of their line; fenced code blocks and tables
//! are held until they close. In plain mode (no TTY, or NO_COLOR set)
//! chunks pass through untouched.
//!
//! Wrapping is done here, word by word, instead of by termimad: its wrap
//! cuts a line with two style runs (`**Label:** text`) between the runs
//! whenever each fits alone, leaving the label or the bullet on a line of
//! its own.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use syntect::easy::HighlightLines;
use syntect::highlighting::{Theme, ThemeSet};
use syntect::parsing::SyntaxSet;
use syntect::util::as_24_bit_terminal_escaped;
use termimad::minimad::Compound;
use termimad::wrap::composite_kind_widths;
use termimad::{CompositeKind, FmtComposite, FmtLine, FmtText, MadSkin};
use unicode_width::UnicodeWidthStr;

pub enum Renderer {
    Plain,
    Rich(Box<Rich>),
}

/// Narrowest and widest the renderer will wrap to, whatever the terminal
/// reports.
pub const MIN_WIDTH: usize = 40;
pub const MAX_WIDTH: usize = 120;

fn clamp(columns: usize) -> usize {
    columns.clamp(MIN_WIDTH, MAX_WIDTH)
}

fn measured() -> usize {
    termimad::crossterm::terminal::size()
        .map(|(columns, _)| usize::from(columns))
        .unwrap_or(80)
}

/// The cell every tracking renderer reads its width from.
pub fn terminal_width() -> Arc<AtomicUsize> {
    static CELL: OnceLock<Arc<AtomicUsize>> = OnceLock::new();
    CELL.get_or_init(|| Arc::new(AtomicUsize::new(clamp(measured()))))
        .clone()
}

/// Re-reads the terminal size into that cell. Called at the start of each
/// turn and from the SIGWINCH handler, so wrapping follows the window
/// instead of whatever it was when the run started.
pub fn measure() {
    terminal_width().store(clamp(measured()), Ordering::Relaxed);
}

pub struct Rich {
    skin: MadSkin,
    /// Read for every block, so a resize lands on the next one.
    width: Arc<AtomicUsize>,
    /// The incomplete last line of the current chunk.
    pending: String,
    block: Block,
}

enum Block {
    /// Source lines of the paragraph or list item being collected; empty
    /// between blocks.
    Prose(Vec<String>),
    Code {
        lang: String,
        lines: Vec<String>,
    },
    Table {
        lines: Vec<String>,
    },
}

/// Syntax definitions and the colour theme, loaded once and shared by
/// code blocks in replies and by diffs.
pub struct Highlighting {
    pub syntaxes: SyntaxSet,
    pub theme: Theme,
}

pub fn highlighting() -> &'static Highlighting {
    static LOADED: OnceLock<Highlighting> = OnceLock::new();
    LOADED.get_or_init(|| Highlighting {
        syntaxes: SyntaxSet::load_defaults_newlines(),
        theme: ThemeSet::load_defaults().themes["base16-ocean.dark"].clone(),
    })
}

impl Renderer {
    /// Rich when stdout is a terminal and NO_COLOR is unset, tracking the
    /// terminal's width rather than the width it had at startup.
    pub fn for_stdout() -> Self {
        use std::io::IsTerminal;
        if std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none() {
            measure();
            Self::tracking(terminal_width())
        } else {
            Self::Plain
        }
    }

    /// A renderer fixed at `width`: a cell nothing updates. Tests and the
    /// goldens use this, since their output must not move with the window.
    #[cfg(test)]
    pub fn rich(width: usize) -> Self {
        Self::tracking(Arc::new(AtomicUsize::new(width)))
    }

    /// A renderer that re-reads `cell` for every block.
    pub fn tracking(cell: Arc<AtomicUsize>) -> Self {
        Renderer::Rich(Box::new(Rich {
            skin: MadSkin::default_dark(),
            width: cell,
            pending: String::new(),
            block: Block::Prose(Vec::new()),
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

    /// Flushes only what is complete, for printing something of airlok's
    /// own in the middle of a turn: a tool call, a note. A line still
    /// arriving keeps its block open, so a bullet is never cut from its
    /// text by a tool call landing between them.
    pub fn interrupt(&mut self) -> String {
        match self {
            Renderer::Plain => String::new(),
            Renderer::Rich(rich) => rich.interrupt(),
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

    /// [`Renderer::interrupt`]: the block is flushed only when nothing is
    /// still arriving in it. Code and tables are held until they close,
    /// as they are anyway.
    fn interrupt(&mut self) -> String {
        if !self.pending.is_empty() || !matches!(self.block, Block::Prose(_)) {
            return String::new();
        }
        let Block::Prose(lines) = std::mem::replace(&mut self.block, Block::Prose(Vec::new()))
        else {
            unreachable!("checked just above")
        };
        self.prose(&lines)
    }

    fn finish(&mut self) -> String {
        let mut out = String::new();
        if !self.pending.is_empty() {
            let line = std::mem::take(&mut self.pending);
            out.push_str(&self.line(&line));
        }
        match std::mem::replace(&mut self.block, Block::Prose(Vec::new())) {
            Block::Prose(lines) => out.push_str(&self.prose(&lines)),
            Block::Code { lang, lines } => out.push_str(&self.code(&lang, &lines)),
            Block::Table { lines } => out.push_str(&self.table(&lines)),
        }
        out
    }

    /// One complete line; returns whatever it makes printable.
    fn line(&mut self, line: &str) -> String {
        match &mut self.block {
            Block::Code { lang, lines } => {
                if line.trim_start().starts_with("```") {
                    let (lang, lines) = (std::mem::take(lang), std::mem::take(lines));
                    self.block = Block::Prose(Vec::new());
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
                self.block = Block::Prose(Vec::new());
                let mut out = self.table(&table);
                out.push_str(&self.line(line));
                out
            }
            Block::Prose(lines) => {
                let trimmed = line.trim_start();
                if !lines.is_empty() && !line.trim().is_empty() && !starts_block(trimmed) {
                    // A continuation of the paragraph or list item.
                    lines.push(line.to_string());
                    return String::new();
                }
                let done = std::mem::take(lines);
                let mut out = self.prose(&done);
                if let Some(rest) = trimmed.strip_prefix("```") {
                    self.block = Block::Code {
                        lang: rest.trim().to_string(),
                        lines: Vec::new(),
                    };
                } else if trimmed.starts_with('|') {
                    self.block = Block::Table {
                        lines: vec![line.to_string()],
                    };
                } else if line.trim().is_empty() {
                    out.push('\n');
                } else if is_single_line_block(trimmed) {
                    out.push_str(&self.markdown(line));
                } else {
                    self.block = Block::Prose(vec![line.to_string()]);
                }
                out
            }
        }
    }

    /// Renders collected paragraph or list-item lines as one block. Soft
    /// line breaks become spaces so the text reflows; a line ending in two
    /// spaces or a backslash keeps its break.
    fn prose(&self, lines: &[String]) -> String {
        if lines.is_empty() {
            return String::new();
        }
        let mut src = lines[0].trim_end().to_string();
        for (previous, line) in lines.iter().zip(&lines[1..]) {
            let hard_break = previous.ends_with("  ") || previous.trim_end().ends_with('\\');
            src.push(if hard_break { '\n' } else { ' ' });
            src.push_str(line.trim());
        }
        self.markdown(&src)
    }

    fn markdown(&self, src: &str) -> String {
        let mut text = FmtText::from(&self.skin, src, None);
        text.lines = std::mem::take(&mut text.lines)
            .into_iter()
            .flat_map(|line| match line {
                FmtLine::Normal(fc) => wrap(fc, self.width.load(Ordering::Relaxed), &self.skin)
                    .into_iter()
                    .map(FmtLine::Normal)
                    .collect(),
                other => vec![other],
            })
            .collect();
        text.width = Some(self.width.load(Ordering::Relaxed));
        text.to_string()
    }

    fn table(&self, lines: &[String]) -> String {
        FmtText::from(
            &self.skin,
            &lines.join("\n"),
            Some(self.width.load(Ordering::Relaxed)),
        )
        .to_string()
    }

    fn code(&self, lang: &str, lines: &[String]) -> String {
        let h = highlighting();
        let syntax = h
            .syntaxes
            .find_syntax_by_token(lang)
            .unwrap_or_else(|| h.syntaxes.find_syntax_plain_text());
        let mut highlighter = HighlightLines::new(syntax, &h.theme);
        let mut out = String::new();
        for line in lines {
            let text = format!("{line}\n");
            let ranges = highlighter
                .highlight_line(&text, &h.syntaxes)
                .unwrap_or_else(|_| vec![(syntect::highlighting::Style::default(), text.as_str())]);
            out.push_str("  ");
            out.push_str(&as_24_bit_terminal_escaped(&ranges, false));
            out.push_str("\x1b[0m");
        }
        out
    }
}

/// A line that begins a new block instead of continuing the current one.
fn starts_block(trimmed: &str) -> bool {
    is_list_marker(trimmed)
        || is_single_line_block(trimmed)
        || trimmed.starts_with("```")
        || trimmed.starts_with('|')
        || trimmed.starts_with('>')
}

fn is_list_marker(trimmed: &str) -> bool {
    if ["- ", "* ", "+ "].iter().any(|m| trimmed.starts_with(m)) {
        return true;
    }
    let digits = trimmed.chars().take_while(char::is_ascii_digit).count();
    digits > 0 && (trimmed[digits..].starts_with(". ") || trimmed[digits..].starts_with(") "))
}

/// Headings and rules are complete at the end of their line.
fn is_single_line_block(trimmed: &str) -> bool {
    if trimmed.starts_with('#') {
        return true;
    }
    let rule: String = trimmed.chars().filter(|c| !c.is_whitespace()).collect();
    rule.len() >= 3
        && (rule.chars().all(|c| c == '-')
            || rule.chars().all(|c| c == '*')
            || rule.chars().all(|c| c == '_'))
}

/// Word-wraps one rendered line to `width` columns: greedy, breaking only
/// at whitespace, continuation lines indented like termimad's own. A word
/// wider than the line gets a line to itself.
fn wrap<'s>(fc: FmtComposite<'s>, width: usize, skin: &MadSkin) -> Vec<FmtComposite<'s>> {
    let (left, right) = skin.line_style(fc.kind).margins_in(Some(width));
    let width = width.saturating_sub(left + right);
    let (first_width, _) = composite_kind_widths(fc.kind, skin);
    if fc.visible_length + first_width <= width || width < 3 {
        return vec![fc];
    }
    let mut lines = Vec::new();
    let mut current = FmtComposite {
        kind: fc.kind,
        compounds: Vec::new(),
        visible_length: first_width,
        spacing: fc.spacing,
    };
    for compound in &fc.compounds {
        for (token, blank) in words(compound) {
            let token_width = token.src.width();
            let fresh = current.compounds.is_empty();
            if blank && fresh && !lines.is_empty() {
                continue;
            }
            if current.visible_length + token_width > width && !fresh {
                let next = follow_up(&current, skin);
                lines.push(trim_end(std::mem::replace(&mut current, next)));
                if blank {
                    continue;
                }
            }
            current.add_compound(token);
        }
    }
    lines.push(current);
    lines
}

/// Drops trailing whitespace runs from a finished line.
fn trim_end(mut fc: FmtComposite<'_>) -> FmtComposite<'_> {
    while fc
        .compounds
        .last()
        .is_some_and(|c| c.src.chars().all(char::is_whitespace))
    {
        let blank = fc.compounds.pop().expect("checked above");
        fc.visible_length -= blank.src.width();
    }
    fc
}

/// The compound cut into alternating runs of whitespace and non-whitespace,
/// each keeping the compound's style. `true` marks a whitespace run.
fn words<'s>(compound: &Compound<'s>) -> Vec<(Compound<'s>, bool)> {
    let src = compound.src;
    let mut out = Vec::new();
    let mut start = 0;
    let mut blank = None;
    for (i, c) in src.char_indices() {
        let is_blank = c.is_whitespace();
        if blank.is_some_and(|b| b != is_blank) {
            out.push((compound.sub(start, i), blank.unwrap_or(false)));
            start = i;
        }
        blank = Some(is_blank);
    }
    if start < src.len() {
        out.push((compound.sub(start, src.len()), blank.unwrap_or(false)));
    }
    out
}

/// An empty continuation line for `fc`: list items continue as follow-ups
/// so they are indented under the text, not given a new bullet.
fn follow_up<'s>(fc: &FmtComposite<'s>, skin: &MadSkin) -> FmtComposite<'s> {
    let kind = match fc.kind {
        CompositeKind::ListItem(depth) => CompositeKind::ListItemFollowUp(depth),
        CompositeKind::OrderedListItem { level, index } => {
            CompositeKind::OrderedListItemFollowUp { level, index }
        }
        kind => kind,
    };
    FmtComposite {
        kind,
        compounds: Vec::new(),
        visible_length: composite_kind_widths(kind, skin).0,
        spacing: fc.spacing,
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
        let third = strip_ansi(&renderer.push("```\nafter\n"));
        assert!(
            third.contains("let key = \"sk-ant-api03-REALVALUE\";"),
            "{third:?}"
        );
        assert!(
            !third.contains("after"),
            "the paragraph after is still open"
        );
        assert_eq!(strip_ansi(&renderer.finish()), "after\n");
    }

    #[test]
    fn table_is_held_until_a_non_table_line() {
        let mut renderer = Renderer::rich(60);
        assert_eq!(renderer.push("| a | b |\n|---|---|\n"), "");
        assert_eq!(renderer.push("| 1 | 2 |\n"), "");
        let out = strip_ansi(&renderer.push("text\n"));
        assert!(
            out.contains('a') && out.contains('1') && out.contains('2'),
            "{out:?}"
        );
        assert!(!out.contains("text"), "{out:?}");
        assert_eq!(strip_ansi(&renderer.finish()), "text\n");
    }

    #[test]
    fn a_paragraph_is_held_until_the_block_ends() {
        let mut renderer = Renderer::rich(60);
        assert_eq!(renderer.push("airlok"), "", "held while the line is open");
        assert_eq!(
            renderer.push(" keeps secrets\non this machine.\n"),
            "",
            "held while the paragraph may continue"
        );
        let out = strip_ansi(&renderer.push("\n"));
        assert_eq!(out, "airlok keeps secrets on this machine.\n\n");
    }

    #[test]
    fn a_list_item_renders_when_the_next_one_starts() {
        let mut renderer = Renderer::rich(60);
        assert_eq!(renderer.push("- one\n"), "");
        assert_eq!(strip_ansi(&renderer.push("- two\n")), "\u{2022} one\n");
        assert_eq!(strip_ansi(&renderer.finish()), "\u{2022} two\n");
    }

    #[test]
    fn a_reply_without_a_newline_is_rendered_at_the_end() {
        let mut renderer = Renderer::rich(60);
        assert_eq!(renderer.push("Hi!"), "");
        assert_eq!(strip_ansi(&renderer.finish()), "Hi!\n");
    }

    /// Lines a few columns wider than the width, with a styled first run:
    /// the shapes termimad's own wrap cut after the first run.
    const STREAM: &str = "**airlok** keeps secrets on your machine and redacts all of the rest.\n\n`airlok` is the airlock between your code and each model's own API.\n\n- **Core runtime:** agent loop, sessions, context, tools, and more.\n- **Safety:** redaction, confirmations, and deny lists by default.\n- short item\n  that continues on a second source line\n\n1. first step\n2. second step\n\nA paragraph the model split\nacross two source lines.\n\n```rust\nlet x = 1;\n```\n\n## Done\n";

    #[test]
    fn streamed_golden_is_independent_of_chunk_boundaries() {
        let whole = render_all(&[STREAM]);
        assert_eq!(whole, include_str!("../tests/golden/stream.ansi"));
        let chars: Vec<&str> = STREAM
            .char_indices()
            .map(|(i, c)| &STREAM[i..i + c.len_utf8()])
            .collect();
        assert_eq!(render_all(&chars), whole, "split at every character");
        for cut in 1..STREAM.len() {
            if STREAM.is_char_boundary(cut) {
                let (a, b) = STREAM.split_at(cut);
                assert_eq!(render_all(&[a, b]), whole, "split at byte {cut}");
            }
        }
    }

    #[test]
    fn a_width_change_between_turns_lands_on_the_next_block() {
        let cell = Arc::new(AtomicUsize::new(100));
        let render = |renderer: &mut Renderer| {
            let mut out = String::new();
            out.push_str(&renderer.push(STREAM));
            out.push_str(&renderer.finish());
            strip_ansi(&out)
        };

        let mut renderer = Renderer::tracking(cell.clone());
        let wide = render(&mut renderer);
        assert!(
            wide.lines().all(|line| line.chars().count() <= 100),
            "nothing wider than the terminal was"
        );

        cell.store(60, Ordering::Relaxed);
        let mut renderer = Renderer::tracking(cell.clone());
        let narrow = render(&mut renderer);
        assert!(
            narrow.lines().all(|line| line.chars().count() <= 60),
            "the next turn wrapped to the new width"
        );
        assert_ne!(wide, narrow, "the change actually reached the output");
    }

    #[test]
    fn a_resize_mid_turn_lands_on_the_next_block() {
        // 200 clamps to MAX_WIDTH; the window then becomes 90.
        let cell = Arc::new(AtomicUsize::new(clamp(200)));
        let mut renderer = Renderer::tracking(cell.clone());
        let mut out = String::new();
        out.push_str(&renderer.push("A first paragraph that is long enough to wrap somewhere near the right hand edge of a wide terminal.\n\n"));
        cell.store(clamp(90), Ordering::Relaxed);
        out.push_str(&renderer.push(STREAM));
        out.push_str(&renderer.finish());

        let text = strip_ansi(&out);
        let after: Vec<&str> = text
            .lines()
            .skip_while(|line| !line.contains("airlok keeps"))
            .collect();
        assert!(!after.is_empty(), "{text:?}");
        assert!(
            after.iter().all(|line| line.chars().count() <= 90),
            "blocks after the resize fit the terminal: {after:?}"
        );
    }

    #[test]
    fn a_styled_first_run_is_not_left_alone_on_its_line() {
        let text = strip_ansi(&render_all(&[STREAM]));
        let lines: Vec<&str> = text.lines().collect();
        for line in &lines {
            assert!(line.chars().count() <= 60, "{line:?} is wider than 60");
        }
        let first = |needle: &str| *lines.iter().find(|l| l.contains(needle)).unwrap();
        assert!(first("Core runtime:").contains("agent loop"), "{lines:?}");
        assert!(first("Safety:").contains("redaction"), "{lines:?}");
        assert!(first("airlok keeps").contains("machine"), "{lines:?}");
        assert!(first("airlok is").contains("between"), "{lines:?}");
        assert!(
            lines.iter().all(|l| l.trim() != "\u{2022}"),
            "bare bullet: {lines:?}"
        );
        assert!(text.contains("A paragraph the model split across two source lines."));
        assert!(text.contains("\u{2022} short item that continues on a second source line"));
    }

    /// A tool call or a note printing between two chunks of one line: the
    /// shape that split a bold label from its text in 0.6.0.
    fn interrupted() -> String {
        let mut renderer = Renderer::rich(60);
        let mut out = String::new();
        out.push_str(&renderer.push("Here is what I found.\n\n- **Overview:**"));
        out.push_str(&renderer.interrupt());
        out.push_str(&renderer.push(" the repo is a Rust workspace of three crates.\n"));
        out.push_str(&renderer.interrupt());
        out.push_str(&renderer.push("- **Next:** run the tests.\n"));
        out.push_str(&renderer.finish());
        out
    }

    #[test]
    fn golden_interrupted_render() {
        assert_eq!(
            interrupted(),
            include_str!("../tests/golden/interrupted.ansi")
        );
    }

    #[test]
    fn a_note_in_the_middle_of_a_line_does_not_cut_the_block() {
        let text = strip_ansi(&interrupted());
        let lines: Vec<&str> = text.lines().collect();
        assert!(
            lines
                .iter()
                .any(|l| l.contains("\u{2022} Overview: the repo is a Rust workspace")),
            "{lines:?}"
        );
        assert!(
            !lines.iter().any(|l| l.trim() == "\u{2022} Overview:"),
            "the label was left alone on its line: {lines:?}"
        );
        assert!(
            lines
                .iter()
                .any(|l| l.contains("\u{2022} Next: run the tests")),
            "{lines:?}"
        );
    }

    #[test]
    fn an_interrupt_between_blocks_flushes_what_is_complete() {
        let mut renderer = Renderer::rich(60);
        assert_eq!(renderer.push("A complete paragraph.\n"), "");
        assert_eq!(strip_ansi(&renderer.interrupt()), "A complete paragraph.\n");
        // Nothing is left to print at the end of the turn.
        assert_eq!(renderer.finish(), "");
    }

    #[test]
    fn an_interrupt_holds_an_unclosed_code_block() {
        let mut renderer = Renderer::rich(60);
        assert_eq!(renderer.push("```rust\nlet x = 1;\n"), "");
        assert_eq!(renderer.interrupt(), "", "the fence has not closed");
        let out = strip_ansi(&renderer.push("```\n"));
        assert!(out.contains("let x = 1;"), "{out:?}");
    }

    /// Rewrites the golden files. Run with
    /// `cargo test -p airlok --bin airlok -- --ignored dump_golden` after
    /// checking the new render by eye.
    #[test]
    #[ignore]
    fn dump_golden() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden");
        std::fs::write(format!("{dir}/sample.ansi"), render_all(&[SAMPLE])).unwrap();
        std::fs::write(format!("{dir}/stream.ansi"), render_all(&[STREAM])).unwrap();
        std::fs::write(format!("{dir}/interrupted.ansi"), interrupted()).unwrap();
    }
}
