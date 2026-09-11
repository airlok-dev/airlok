//! The status line: a spinner, what the agent is doing, seconds since the
//! turn began, and tokens this turn, on one line redrawn in place.
//!
//! Every write to stdout goes through [`Screen`]. It erases the status line
//! before writing and draws it again only once the output has ended a
//! line, and the ticker thread redraws through the same lock, so a frame
//! never lands inside the model's text. A disabled screen (stdout not a
//! terminal, `NO_COLOR` set, or `-v`) writes exactly what it is given.

use std::io::Write;
use std::time::{Duration, Instant};

use airlok_core::agent::fmt_tokens;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

/// Spinner frames, one per tick.
pub const FRAMES: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

/// Moves to the start of the line and erases it.
pub const CLEAR: &str = "\r\x1b[2K";

pub struct Screen {
    out: Box<dyn Write + Send>,
    enabled: bool,
    width: usize,
    /// The turn in progress, while there is one.
    status: Option<Status>,
    /// A frame is on screen and must be erased before anything else is
    /// written.
    drawn: bool,
    /// The last write ended a line, so a frame may go below it.
    at_line_start: bool,
}

struct Status {
    started: Instant,
    action: String,
    tokens: u64,
    frame: usize,
}

impl Screen {
    pub fn new(out: Box<dyn Write + Send>, enabled: bool, width: usize) -> Self {
        Self {
            out,
            enabled,
            width,
            status: None,
            drawn: false,
            at_line_start: true,
        }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Writes output with the status line kept below it.
    pub fn write(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        self.clear();
        let _ = self.out.write_all(text.as_bytes());
        self.at_line_start = text.ends_with('\n');
        self.draw();
        let _ = self.out.flush();
    }

    /// Starts the status line for a turn. The line editor has just ended
    /// its line, so the cursor is at the start of one.
    pub fn begin(&mut self) {
        if !self.enabled {
            return;
        }
        self.status = Some(Status {
            started: Instant::now(),
            action: "thinking".into(),
            tokens: 0,
            frame: 0,
        });
        self.at_line_start = true;
        self.redraw();
    }

    /// Erases the status line and forgets the turn.
    pub fn end(&mut self) {
        self.clear();
        self.status = None;
        let _ = self.out.flush();
    }

    /// Erases the status line until the next write or tick, so a prompt
    /// can use the line.
    pub fn hide(&mut self) {
        self.clear();
        let _ = self.out.flush();
    }

    pub fn set_action(&mut self, action: &str) {
        if let Some(status) = &mut self.status {
            status.action = one_line(action);
            self.redraw();
        }
    }

    pub fn set_tokens(&mut self, tokens: u64) {
        if let Some(status) = &mut self.status {
            status.tokens = tokens;
        }
    }

    /// Advances the spinner and redraws. Called by the ticker thread.
    pub fn tick(&mut self) {
        if let Some(status) = &mut self.status {
            status.frame += 1;
            self.redraw();
        }
    }

    fn clear(&mut self) {
        if self.drawn {
            let _ = self.out.write_all(CLEAR.as_bytes());
            self.drawn = false;
        }
    }

    fn draw(&mut self) {
        if !self.at_line_start {
            return;
        }
        if let Some(status) = &self.status {
            let line = status.render(status.started.elapsed(), self.width);
            let _ = self.out.write_all(line.as_bytes());
            self.drawn = true;
        }
    }

    fn redraw(&mut self) {
        self.clear();
        self.draw();
        let _ = self.out.flush();
    }
}

impl Status {
    /// `⠹ reading src/main.rs · 12s · 3k tokens`, dim, cut to fit `width`
    /// with a column to spare so the terminal never wraps it.
    fn render(&self, elapsed: Duration, width: usize) -> String {
        let spinner = FRAMES[self.frame % FRAMES.len()];
        let tail = format!(
            " · {}s · {} tokens",
            elapsed.as_secs(),
            fmt_tokens(self.tokens)
        );
        let room = width.saturating_sub(2 + tail.width() + 1);
        format!(
            "\x1b[2m{spinner} {}{tail}\x1b[0m",
            truncate(&self.action, room)
        )
    }
}

/// What the status line says while a tool runs.
pub fn action(tool: &str, summary: &str) -> String {
    match tool {
        "read_file" => format!("reading {summary}"),
        "list_dir" => format!("listing {summary}"),
        "glob" => format!("finding {summary}"),
        "grep" => format!("searching for {summary}"),
        "write_file" => format!("writing {summary}"),
        "edit_file" => format!("editing {summary}"),
        "bash" => format!("running {summary}"),
        other => format!("{other} {summary}"),
    }
}

/// Newlines and runs of whitespace become single spaces.
fn one_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Cuts `text` to `width` columns, ending in `…` when anything was cut.
fn truncate(text: &str, width: usize) -> String {
    if text.width() <= width {
        return text.to_string();
    }
    let mut out = String::new();
    let mut used = 0;
    for c in text.chars() {
        let w = c.width().unwrap_or(0);
        if used + w + 1 > width {
            break;
        }
        out.push(c);
        used += w;
    }
    out.push('…');
    out
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    /// A writer that appends to a shared buffer.
    #[derive(Clone, Default)]
    pub struct Sink(Arc<Mutex<Vec<u8>>>);

    impl Sink {
        pub fn contents(&self) -> String {
            String::from_utf8(self.0.lock().unwrap().clone()).unwrap()
        }
    }

    impl Write for Sink {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// Splits raw screen output into the text written and the number of
    /// status frames, checking that every frame starts a line of its own
    /// and is erased before anything follows it.
    pub fn strip_frames(raw: &str) -> (String, usize) {
        let mut plain = String::new();
        let mut frames = 0;
        let mut rest = raw;
        while let Some(c) = rest.chars().next() {
            if let Some(after) = rest.strip_prefix(CLEAR) {
                rest = after;
                continue;
            }
            if let Some(after) = rest.strip_prefix("\x1b[2m") {
                if after.starts_with(FRAMES) {
                    assert!(
                        plain.is_empty() || plain.ends_with('\n'),
                        "a frame in the middle of a line, after {plain:?}"
                    );
                    let end = after.find("\x1b[0m").expect("a frame ends its style");
                    assert!(!after[..end].contains('\n'), "a frame spans lines");
                    rest = &after[end + "\x1b[0m".len()..];
                    assert!(
                        rest.is_empty() || rest.starts_with(CLEAR),
                        "output written over a frame: {rest:?}"
                    );
                    frames += 1;
                    continue;
                }
            }
            plain.push(c);
            rest = &rest[c.len_utf8()..];
        }
        (plain, frames)
    }

    #[test]
    fn a_disabled_screen_writes_only_what_it_is_given() {
        let sink = Sink::default();
        let mut screen = Screen::new(Box::new(sink.clone()), false, 80);
        screen.begin();
        screen.set_action("reading src/main.rs");
        screen.tick();
        screen.write("hello ");
        screen.set_tokens(1200);
        screen.tick();
        screen.write("world\n");
        screen.tick();
        screen.hide();
        screen.end();
        assert_eq!(sink.contents(), "hello world\n");
    }

    #[test]
    fn frames_never_land_inside_text_written_from_another_thread() {
        let text: String = (1..=40)
            .map(|i| format!("line {i} of the model's reply, with a few words\n"))
            .collect();
        let sink = Sink::default();
        let screen = Arc::new(Mutex::new(Screen::new(Box::new(sink.clone()), true, 60)));
        screen.lock().unwrap().begin();
        let ticker = {
            let screen = screen.clone();
            std::thread::spawn(move || {
                for _ in 0..3000 {
                    screen.lock().unwrap().tick();
                    std::thread::yield_now();
                }
            })
        };
        // Character by character, so most writes leave the line open.
        for (i, c) in text.char_indices() {
            screen.lock().unwrap().write(&text[i..i + c.len_utf8()]);
            if i % 7 == 0 {
                std::thread::yield_now();
            }
        }
        ticker.join().unwrap();
        screen.lock().unwrap().end();

        let (plain, frames) = strip_frames(&sink.contents());
        assert_eq!(plain, text);
        assert!(frames > 40, "a frame after each line at least: {frames}");
    }

    #[test]
    fn the_line_shows_action_seconds_and_tokens_and_fits() {
        let status = Status {
            started: Instant::now(),
            action: "reading src/main.rs".into(),
            tokens: 3100,
            frame: 2,
        };
        assert_eq!(
            status.render(Duration::from_secs(12), 80),
            "\x1b[2m⠹ reading src/main.rs · 12s · 3k tokens\x1b[0m"
        );
        let long = Status {
            action: action(
                "bash",
                "cargo test --workspace --all-targets -- --nocapture",
            ),
            ..status
        };
        let line = long.render(Duration::from_secs(3), 40);
        let visible = line
            .trim_start_matches("\x1b[2m")
            .trim_end_matches("\x1b[0m");
        assert_eq!(visible.width(), 39, "{visible}");
        assert!(visible.starts_with("⠹ running cargo test"), "{visible}");
        assert!(visible.ends_with("… · 3s · 3k tokens"), "{visible}");
    }
}
