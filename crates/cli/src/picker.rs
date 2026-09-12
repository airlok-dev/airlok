//! A list to pick from at the prompt: arrows move, typing filters, Enter
//! takes the highlighted row, Esc leaves everything alone.
//!
//! The choosing is separate from the drawing. [`Picker`] is the part that
//! decides what is shown and what a keystroke means, and it needs no
//! terminal, which is what makes it testable. [`ask`] is the thin part
//! that puts it on the screen.

use std::io::{Read, Write};

use crate::keys::Cbreak;

/// Rows of the list on screen at once.
const ROWS: usize = 8;

/// What a keystroke did.
#[derive(Debug, PartialEq, Eq)]
pub enum Step {
    /// Still choosing.
    Continue,
    Chosen(String),
    /// Esc, or Ctrl-C: nothing changes.
    Cancelled,
}

pub struct Picker {
    choices: Vec<String>,
    query: String,
    /// Index into the current matches.
    cursor: usize,
}

impl Picker {
    pub fn new(choices: &[String]) -> Self {
        Self {
            choices: choices.to_vec(),
            query: String::new(),
            cursor: 0,
        }
    }

    /// The choices that contain the query, in the order they were given,
    /// since that order is a ranking: what is in use comes first.
    pub fn matches(&self) -> Vec<&String> {
        if self.query.is_empty() {
            return self.choices.iter().collect();
        }
        let query = self.query.to_ascii_lowercase();
        self.choices
            .iter()
            .filter(|choice| choice.to_ascii_lowercase().contains(&query))
            .collect()
    }

    pub fn selected(&self) -> Option<&String> {
        self.matches().get(self.cursor).copied()
    }

    #[cfg(test)]
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// One byte of input. Escape sequences arrive a byte at a time, so an
    /// arrow is three calls and a bare Esc is one.
    pub fn feed(&mut self, byte: u8, escape: &mut Escape) -> Step {
        match escape.feed(byte) {
            Key::Up => {
                self.cursor = self.cursor.saturating_sub(1);
                Step::Continue
            }
            Key::Down => {
                let last = self.matches().len().saturating_sub(1);
                self.cursor = (self.cursor + 1).min(last);
                Step::Continue
            }
            Key::Escape => Step::Cancelled,
            Key::Enter => match self.selected() {
                // Enter with nothing highlighted does nothing: a filter
                // that matches nothing must not become a choice.
                None => Step::Continue,
                Some(choice) => Step::Chosen(choice.clone()),
            },
            Key::Backspace => {
                self.query.pop();
                self.cursor = 0;
                Step::Continue
            }
            Key::Char(c) => {
                self.query.push(c);
                self.cursor = 0;
                Step::Continue
            }
            Key::None => Step::Continue,
        }
    }

    /// The lines to draw: the filter, then a window of the matches with
    /// the highlighted one marked.
    pub fn lines(&self, title: &str, colour: bool) -> Vec<String> {
        let dim = |text: String| {
            if colour {
                format!("\x1b[2m{text}\x1b[0m")
            } else {
                text
            }
        };
        let matches = self.matches();
        let mut lines = vec![dim(format!(
            "{title}: {} (arrows, type to filter, Enter, Esc)",
            if self.query.is_empty() {
                "all".to_string()
            } else {
                self.query.clone()
            }
        ))];
        let first = self.cursor.saturating_sub(ROWS - 1);
        for (at, choice) in matches.iter().enumerate().skip(first).take(ROWS) {
            lines.push(if at == self.cursor {
                format!("> {choice}")
            } else {
                dim(format!("  {choice}"))
            });
        }
        if matches.is_empty() {
            lines.push(dim("  (nothing matches)".to_string()));
        }
        lines
    }

    /// `lines` padded to a height that depends only on how many choices
    /// there are, so narrowing the list with a filter cannot leave the rows
    /// of a taller frame stranded on the screen.
    pub fn frame(&self, title: &str, colour: bool) -> Vec<String> {
        let mut lines = self.lines(title, colour);
        lines.resize(1 + ROWS.min(self.choices.len()).max(1), String::new());
        lines
    }
}

/// Whether the bytes arriving are part of an escape sequence.
#[derive(Debug, Default, PartialEq, Eq)]
pub enum Escape {
    #[default]
    Idle,
    /// Esc seen; the next byte says whether it was a sequence.
    Seen,
    /// `Esc [` seen: the next byte is the key.
    Bracket,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Key {
    Up,
    Down,
    Enter,
    Escape,
    Backspace,
    Char(char),
    None,
}

impl Escape {
    /// Whether an Esc has arrived with nothing after it yet. The caller
    /// decides it was a bare Esc by waiting, since the rest of an arrow
    /// key would already be here.
    fn pending(&self) -> bool {
        matches!(self, Escape::Seen)
    }

    fn clear(&mut self) {
        *self = Escape::Idle;
    }

    pub fn feed(&mut self, byte: u8) -> Key {
        match (&self, byte) {
            (Escape::Idle, 0x1b) => {
                *self = Escape::Seen;
                Key::None
            }
            (Escape::Seen, b'[') => {
                *self = Escape::Bracket;
                Key::None
            }
            // Esc then anything else: a bare Esc, and the byte is dropped
            // rather than typed, since it was part of a sequence airlok
            // does not know.
            (Escape::Seen, _) => {
                *self = Escape::Idle;
                Key::Escape
            }
            (Escape::Bracket, b'A') => {
                *self = Escape::Idle;
                Key::Up
            }
            (Escape::Bracket, b'B') => {
                *self = Escape::Idle;
                Key::Down
            }
            (Escape::Bracket, _) => {
                *self = Escape::Idle;
                Key::None
            }
            (Escape::Idle, b'\r' | b'\n') => Key::Enter,
            (Escape::Idle, 0x7f | 0x08) => Key::Backspace,
            // Ctrl-C leaves the list the way Esc does: the picker clears
            // ISIG while it is up, so this arrives as a byte.
            (Escape::Idle, 0x03) => Key::Escape,
            (Escape::Idle, byte) if byte >= 0x20 => Key::Char(char::from(byte)),
            _ => Key::None,
        }
    }
}

/// Puts the list on the terminal and returns what was picked. `None` when
/// the user cancels, or when there is no terminal to draw on.
pub fn ask(title: &str, choices: &[String]) -> Option<String> {
    if choices.is_empty() {
        return None;
    }
    let mut tty = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/tty")
        .ok()?;
    let mut input = tty.try_clone().ok()?;
    let colour = std::env::var_os("NO_COLOR").is_none();
    let mut picker = Picker::new(choices);
    let mut escape = Escape::default();
    // Raw enough to read a key at a time, and with a read timeout, so an
    // Esc that nothing follows can be told from an arrow key. Restored
    // when this returns.
    let _cbreak = Cbreak::enter_polling(std::os::fd::AsRawFd::as_raw_fd(&tty)).ok()?;

    let mut drawn = 0usize;
    let mut draw = true;
    loop {
        if draw {
            let lines = picker.frame(title, colour);
            let mut frame = String::new();
            if drawn > 0 {
                frame.push_str(&format!("\x1b[{drawn}A"));
            }
            for line in &lines {
                frame.push_str(&format!("\r\x1b[2K{line}\n"));
            }
            drawn = lines.len();
            let _ = tty.write_all(frame.as_bytes());
            let _ = tty.flush();
            draw = false;
        }

        let mut byte = [0u8; 1];
        let step = match input.read(&mut byte) {
            Ok(1) => picker.feed(byte[0], &mut escape),
            // The read timed out. An Esc with nothing behind it was the
            // key itself, not the start of a sequence.
            Ok(0) if escape.pending() => {
                escape.clear();
                Step::Cancelled
            }
            Ok(0) => continue,
            _ => {
                let _ = erase(&mut tty, drawn);
                return None;
            }
        };
        match step {
            Step::Continue => draw = true,
            Step::Cancelled => {
                let _ = erase(&mut tty, drawn);
                return None;
            }
            Step::Chosen(choice) => {
                let _ = erase(&mut tty, drawn);
                return Some(choice);
            }
        }
    }
}

/// Takes the list back off the screen, so the prompt is where it was.
fn erase(tty: &mut std::fs::File, drawn: usize) -> std::io::Result<()> {
    let mut frame = String::new();
    if drawn > 0 {
        frame.push_str(&format!("\x1b[{drawn}A"));
    }
    for _ in 0..drawn {
        frame.push_str("\r\x1b[2K\n");
    }
    if drawn > 0 {
        frame.push_str(&format!("\x1b[{drawn}A"));
    }
    tty.write_all(frame.as_bytes())?;
    tty.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn choices() -> Vec<String> {
        [
            "gpt-5.6-luna",
            "gpt-6-astra",
            "claude-sonnet-4-6",
            "gpt-5.5",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    fn type_in(picker: &mut Picker, escape: &mut Escape, text: &str) -> Step {
        let mut step = Step::Continue;
        for byte in text.bytes() {
            step = picker.feed(byte, escape);
        }
        step
    }

    #[test]
    fn typing_filters_and_keeps_the_given_order() {
        let mut picker = Picker::new(&choices());
        let mut escape = Escape::default();
        assert_eq!(picker.matches().len(), 4);

        type_in(&mut picker, &mut escape, "gpt");
        assert_eq!(picker.matches().len(), 3);
        assert_eq!(picker.matches()[0], "gpt-5.6-luna");

        type_in(&mut picker, &mut escape, "-6");
        assert_eq!(
            picker
                .matches()
                .iter()
                .map(|m| m.as_str())
                .collect::<Vec<_>>(),
            ["gpt-6-astra"]
        );
        // Backspace widens it again.
        picker.feed(0x7f, &mut escape);
        assert_eq!(picker.matches().len(), 3);
    }

    #[test]
    fn a_filter_that_matches_nothing_cannot_be_chosen() {
        let mut picker = Picker::new(&choices());
        let mut escape = Escape::default();
        type_in(&mut picker, &mut escape, "zzz");
        assert!(picker.matches().is_empty());
        assert_eq!(picker.selected(), None);
        assert_eq!(picker.feed(b'\r', &mut escape), Step::Continue);
    }

    #[test]
    fn a_lone_esc_waits_before_it_counts_as_a_key() {
        let mut escape = Escape::default();
        assert_eq!(escape.feed(0x1b), Key::None);
        assert!(escape.pending(), "nothing has followed the esc yet");
        assert_eq!(escape.feed(b'['), Key::None);
        assert!(!escape.pending(), "an arrow key is on its way");
        assert_eq!(escape.feed(b'B'), Key::Down);
        assert!(!escape.pending());
    }

    #[test]
    fn every_frame_is_the_same_height() {
        let mut picker = Picker::new(&choices());
        let mut escape = Escape::default();
        let tall = picker.frame("model", false).len();
        type_in(&mut picker, &mut escape, "gpt");
        assert!(
            picker.matches().len() < choices().len(),
            "the filter narrowed the list"
        );
        assert_eq!(
            picker.frame("model", false).len(),
            tall,
            "a narrower list still draws the same number of rows"
        );
        type_in(&mut picker, &mut escape, "zzz");
        assert!(picker.matches().is_empty());
        assert_eq!(
            picker.frame("model", false).len(),
            tall,
            "and so does one that matches nothing"
        );
    }

    #[test]
    fn arrows_move_within_the_matches_and_stop_at_the_ends() {
        let mut picker = Picker::new(&choices());
        let mut escape = Escape::default();
        let down = |picker: &mut Picker, escape: &mut Escape| {
            for byte in [0x1b, b'[', b'B'] {
                picker.feed(byte, escape);
            }
        };
        assert_eq!(picker.cursor(), 0);
        down(&mut picker, &mut escape);
        assert_eq!(picker.selected().unwrap(), "gpt-6-astra");
        for _ in 0..10 {
            down(&mut picker, &mut escape);
        }
        assert_eq!(picker.selected().unwrap(), "gpt-5.5", "stops at the last");
        for byte in [0x1b, b'[', b'A'] {
            picker.feed(byte, &mut escape);
        }
        assert_eq!(picker.selected().unwrap(), "claude-sonnet-4-6");
        // Filtering starts from the top again.
        type_in(&mut picker, &mut escape, "astra");
        assert_eq!(picker.cursor(), 0);
        assert_eq!(
            picker.feed(b'\r', &mut escape),
            Step::Chosen("gpt-6-astra".into())
        );
    }

    #[test]
    fn esc_cancels_and_so_does_ctrl_c() {
        let mut picker = Picker::new(&choices());
        let mut escape = Escape::default();
        assert_eq!(
            picker.feed(0x1b, &mut escape),
            Step::Continue,
            "held, it may be an arrow"
        );
        assert_eq!(
            picker.feed(b'q', &mut escape),
            Step::Cancelled,
            "not an arrow, so Esc"
        );

        let mut picker = Picker::new(&choices());
        let mut escape = Escape::default();
        assert_eq!(picker.feed(0x03, &mut escape), Step::Cancelled);
    }

    #[test]
    fn the_drawn_lines_mark_the_highlighted_row() {
        let picker = Picker::new(&choices());
        let lines = picker.lines("model", false);
        assert!(lines[0].starts_with("model: all"), "{lines:?}");
        assert_eq!(lines[1], "> gpt-5.6-luna");
        assert_eq!(lines[2], "  gpt-6-astra");
    }
}
