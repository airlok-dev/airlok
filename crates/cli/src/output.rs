//! What the user sees during a run: the model's text as rendered markdown,
//! tool calls, notes, and the status line, all written through one
//! [`Screen`], plus confirmations on the terminal.

use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::Duration;

use airlok_core::tools::READ_ONLY_TOOLS;
use airlok_core::{Confirmation, Decision, Output};

use crate::render::Renderer;
use crate::status::{self, Screen};
use crate::terminal::Terminal;

/// How often the status line redraws.
const TICK: Duration = Duration::from_millis(100);

/// Streams model text to stdout as rendered markdown, shows tool calls
/// dimmed (a run of read-only calls as one line), keeps a status line
/// below the output during a turn, and asks for confirmations on the
/// terminal.
pub struct Stdout {
    renderer: Renderer,
    terminal: Option<Terminal>,
    /// Something is on the current line that needs a newline before
    /// the next block of output.
    mid_line: bool,
    /// Read-only calls since the last output, summed up in one line when
    /// the run of them ends.
    collapsed: usize,
    screen: Arc<Mutex<Screen>>,
    ticker: Option<Ticker>,
}

impl Stdout {
    pub fn new(terminal: Option<Terminal>, verbose: bool) -> Self {
        let width = termimad::crossterm::terminal::size()
            .map(|(w, _)| usize::from(w))
            .unwrap_or(80);
        Self::with(
            Renderer::for_stdout(),
            terminal,
            verbose,
            Box::new(std::io::stdout()),
            width,
        )
    }

    /// The status line is on when the output is rich (a terminal, and
    /// NO_COLOR unset) and `-v` is off, since debug logs go to the same
    /// terminal through stderr and would tear it.
    fn with(
        renderer: Renderer,
        terminal: Option<Terminal>,
        verbose: bool,
        out: Box<dyn Write + Send>,
        width: usize,
    ) -> Self {
        let status_line = renderer.is_rich() && !verbose;
        Self {
            renderer,
            terminal,
            mid_line: false,
            collapsed: 0,
            screen: Arc::new(Mutex::new(Screen::new(out, status_line, width))),
            ticker: None,
        }
    }

    fn screen(&self) -> MutexGuard<'_, Screen> {
        lock(&self.screen)
    }

    fn write(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        self.screen().write(text);
        self.mid_line = !text.ends_with('\n');
    }

    fn end_line(&mut self) {
        if self.mid_line {
            self.write("\n");
        }
    }

    /// Ends a run of read-only calls with one line saying how many.
    fn flush_collapsed(&mut self) {
        let n = std::mem::take(&mut self.collapsed);
        if n > 0 {
            let plural = if n == 1 { "" } else { "s" };
            self.write(&format!("\x1b[2mread {n} file{plural}\x1b[0m\n"));
        }
    }

    /// Flushes held markdown and closes the line.
    pub fn finish(&mut self) {
        let rest = self.renderer.finish();
        self.write(&rest);
        self.end_line();
        self.flush_collapsed();
    }

    fn start_ticker(&mut self) {
        if self.ticker.is_none() && self.screen().enabled() {
            self.ticker = Some(Ticker::start(self.screen.clone()));
        }
    }

    /// Stops redrawing; returns whether it was running.
    fn stop_ticker(&mut self) -> bool {
        self.ticker.take().map(Ticker::stop).is_some()
    }
}

impl Output for Stdout {
    fn status(&mut self, line: &str) {
        self.finish();
        let text = if self.renderer.is_rich() {
            format!("\x1b[2m{line}\x1b[0m\n")
        } else {
            format!("{line}\n")
        };
        self.write(&text);
    }

    fn begin_turn(&mut self) {
        self.screen().begin();
        self.start_ticker();
    }

    fn thinking(&mut self) {
        self.screen().set_action("thinking");
    }

    fn tokens(&mut self, used: u64) {
        self.screen().set_tokens(used);
    }

    fn command_output(&mut self, text: &str) {
        self.finish();
        self.write(text);
        self.end_line();
    }

    fn end_turn(&mut self) {
        self.finish();
        self.stop_ticker();
        self.screen().end();
    }

    fn text(&mut self, chunk: &str) {
        self.flush_collapsed();
        let rendered = self.renderer.push(chunk);
        self.write(&rendered);
    }

    fn tool_call(&mut self, name: &str, summary: &str) {
        let rest = self.renderer.finish();
        self.write(&rest);
        self.screen().set_action(&status::action(name, summary));
        if READ_ONLY_TOOLS.contains(&name) && self.screen().enabled() {
            self.collapsed += 1;
            return;
        }
        self.end_line();
        self.flush_collapsed();
        if self.renderer.is_rich() {
            self.write(&format!("\x1b[2m> {name}: {summary}\x1b[0m\n"));
        } else {
            self.write(&format!("> {name}: {summary}\n"));
        }
    }

    fn confirm(&mut self, request: &Confirmation<'_>) -> Decision {
        self.end_line();
        self.flush_collapsed();
        // The question needs the line, and the answer must reach the
        // prompt, not the key watcher.
        let ticking = self.stop_ticker();
        self.screen().hide();
        let decision = match self.terminal.as_mut() {
            None => Decision::Reject,
            Some(terminal) => match request {
                Confirmation::Write { diff, .. } => {
                    terminal.show_diff(diff);
                    terminal.ask("Apply?")
                }
                Confirmation::Command { command } => terminal.ask(&format!("Run `{command}`?")),
            },
        };
        if ticking {
            self.start_ticker();
        }
        decision
    }
}

/// Redraws the status line on its own thread until stopped.
struct Ticker {
    stop: Arc<AtomicBool>,
    thread: JoinHandle<()>,
}

impl Ticker {
    fn start(screen: Arc<Mutex<Screen>>) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let thread = {
            let stop = stop.clone();
            std::thread::spawn(move || loop {
                std::thread::sleep(TICK);
                if stop.load(Ordering::Relaxed) {
                    break;
                }
                lock(&screen).tick();
            })
        };
        Self { stop, thread }
    }

    /// Returns once the thread has finished, so nothing draws after it.
    fn stop(self) {
        self.stop.store(true, Ordering::Relaxed);
        let _ = self.thread.join();
    }
}

/// A panic elsewhere must not take the terminal output down with it.
fn lock(screen: &Mutex<Screen>) -> MutexGuard<'_, Screen> {
    screen.lock().unwrap_or_else(PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::status::tests::{strip_frames, Sink};

    #[test]
    fn without_a_terminal_a_turn_prints_no_status_output() {
        let sink = Sink::default();
        let mut out = Stdout::with(Renderer::Plain, None, false, Box::new(sink.clone()), 80);
        out.begin_turn();
        out.thinking();
        out.tokens(120);
        out.text("Looking at the ");
        out.text("code.\n");
        out.tool_call("read_file", "src/main.rs");
        out.tool_call("grep", "fn main");
        out.screen().tick();
        out.thinking();
        out.tokens(940);
        out.text("Found it.");
        out.end_turn();
        out.status("gpt-6-astra · context 3% · session abc123");
        assert_eq!(
            sink.contents(),
            include_str!("../tests/golden/plain_turn.txt")
        );
    }

    #[test]
    fn with_a_terminal_frames_stay_off_the_lines_of_text() {
        const REPLY: &str = "Here is **the plan**:\n\n- read the config\n- change `main.rs`\n\nThen run the tests.\n";
        let run = |ticks: bool| {
            let sink = Sink::default();
            let mut out = Stdout::with(Renderer::rich(60), None, false, Box::new(sink.clone()), 60);
            out.begin_turn();
            out.tool_call("read_file", "src/main.rs");
            out.tool_call("grep", "fn main");
            for (i, c) in REPLY.char_indices() {
                out.text(&REPLY[i..i + c.len_utf8()]);
                if ticks {
                    out.screen().tick();
                }
            }
            out.tool_call("bash", "cargo test");
            out.end_turn();
            sink.contents()
        };
        let (ticked, frames) = strip_frames(&run(true));
        let (quiet, _) = strip_frames(&run(false));
        assert_eq!(ticked, quiet, "frames change nothing but themselves");
        assert!(frames > 10, "{frames}");
        assert!(quiet.contains("\x1b[2mread 2 files\x1b[0m\n"), "{quiet:?}");
        assert!(quiet.contains("\x1b[2m> bash: cargo test\x1b[0m\n"));
        assert!(quiet.contains("Then run the tests."));
    }
}
