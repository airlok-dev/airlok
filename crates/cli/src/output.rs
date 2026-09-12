//! What the user sees during a run: the model's text as rendered markdown,
//! tool calls, notes, and the status line, all written through one
//! [`Screen`], plus confirmations on the terminal.

use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::Duration;

use airlok_core::tools::READ_ONLY_TOOLS;
use airlok_core::{Confirmation, Decision, Output};

use crate::diff;
use crate::keys::{Cbreak, Keys};
use crate::render::{highlighting, Renderer};
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
    /// Watches for Esc during turns; the REPL sets it. Lent to the ticker
    /// thread while a turn runs.
    keys: Option<Keys>,
}

impl Stdout {
    pub fn new(terminal: Option<Terminal>, verbose: bool) -> Self {
        crate::render::measure();
        Self::with(
            Renderer::for_stdout(),
            terminal,
            verbose,
            Box::new(std::io::stdout()),
            crate::render::terminal_columns(),
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
        columns: Arc<AtomicUsize>,
    ) -> Self {
        let status_line = renderer.is_rich() && !verbose;
        Self {
            renderer,
            terminal,
            mid_line: false,
            collapsed: 0,
            screen: Arc::new(Mutex::new(Screen::new(out, status_line, columns))),
            ticker: None,
            keys: None,
        }
    }

    /// Watch the terminal for Esc during turns.
    pub fn watch_keys(&mut self, keys: Keys) {
        self.keys = Some(keys);
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

    /// Flushes held markdown and closes the line. For the end of a turn:
    /// whatever is held is printed, finished or not.
    pub fn finish(&mut self) {
        let rest = self.renderer.finish();
        self.write(&rest);
        self.end_line();
        self.flush_collapsed();
    }

    /// Makes room for a line of airlok's own in the middle of a turn. Only
    /// complete markdown is flushed: text still arriving keeps its block,
    /// so a tool call cannot cut a bullet from its text.
    fn pause(&mut self) {
        let rest = self.renderer.interrupt();
        self.write(&rest);
        self.end_line();
        self.flush_collapsed();
    }

    fn start_ticker(&mut self) {
        if self.ticker.is_none() && (self.screen().enabled() || self.keys.is_some()) {
            self.ticker = Some(Ticker::start(self.screen.clone(), self.keys.take()));
        }
    }

    /// Stops the ticker and takes the keys back; returns whether it was
    /// running.
    fn stop_ticker(&mut self) -> bool {
        let Some(ticker) = self.ticker.take() else {
            return false;
        };
        if let Some(keys) = ticker.stop() {
            self.keys = Some(keys);
        }
        true
    }
}

impl Output for Stdout {
    fn status(&mut self, line: &str) {
        self.pause();
        let text = if self.renderer.is_rich() {
            format!("\x1b[2m{line}\x1b[0m\n")
        } else {
            format!("{line}\n")
        };
        self.write(&text);
    }

    fn begin_turn(&mut self) {
        // The window may have changed since the last turn.
        crate::render::measure();
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
        let rest = self.renderer.interrupt();
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
                Confirmation::Write { path, diff } => {
                    let colors = std::env::var_os("NO_COLOR").is_none().then(highlighting);
                    let lines = diff::render(&path.display().to_string(), diff, colors);
                    let (page, rest) = lines.split_at(lines.len().min(diff::PAGE));
                    terminal.show_lines(page);
                    if !rest.is_empty() {
                        let marker = diff::more_lines(rest.len());
                        terminal.show_lines(&[match colors {
                            Some(_) => format!("\x1b[2m{marker}\x1b[0m"),
                            None => marker,
                        }]);
                    }
                    terminal.ask_paged("Apply?", rest)
                }
                Confirmation::Command { command } => terminal.ask(&format!("Run `{command}`?")),
                Confirmation::McpProject { servers } => {
                    let mut lines = vec![
                        "This project's .mcp.json asks to start MCP servers in this repository."
                            .to_string(),
                        "They came with the checkout, so nothing has run yet.".to_string(),
                    ];
                    for (name, command) in *servers {
                        lines.push(format!("  {name}: {command}"));
                    }
                    terminal.show_lines(&lines);
                    terminal.ask("Let this repository start them?")
                }
                Confirmation::Mcp {
                    server,
                    tool,
                    arguments,
                    root,
                    paths,
                } => {
                    let lines = mcp_lines(server, tool, arguments, *root, paths);
                    let (page, rest) = lines.split_at(lines.len().min(diff::PAGE));
                    terminal.show_lines(page);
                    if !rest.is_empty() {
                        terminal.show_lines(&[diff::more_lines(rest.len())]);
                    }
                    // Only this prompt can be remembered, so only this
                    // one offers it.
                    terminal.ask_savable(&format!("Call `{tool}` on `{server}`?"), rest)
                }
            },
        };
        if ticking {
            self.start_ticker();
        }
        decision
    }
}

/// The lines shown before an MCP call is confirmed: what the server can
/// reach, the place each argument names with anything outside the project
/// marked, and the arguments as they will be sent. `airlok mcp call` shows
/// the same lines, so both ways of asking ask the same question.
pub fn mcp_lines(
    server: &str,
    tool: &str,
    arguments: &str,
    root: Option<&str>,
    paths: &[String],
) -> Vec<String> {
    let dim = |text: String| match std::env::var_os("NO_COLOR").is_none() {
        true => format!("\x1b[2m{text}\x1b[0m"),
        false => text,
    };
    let mut lines = vec![dim(format!(
        "mcp {server} · {tool} · arguments as they will be sent"
    ))];
    if let Some(root) = root {
        lines.push(dim(format!("serving {root}")));
    }
    // Plain, not dim: a place outside the project is the thing most worth
    // seeing before answering.
    let here = std::env::current_dir().unwrap_or_default();
    for path in paths {
        let outside = !std::path::Path::new(path).starts_with(&here);
        let note = if outside {
            "  (outside this project)"
        } else {
            ""
        };
        lines.push(format!("{path}{note}"));
    }
    lines.extend(arguments.lines().map(str::to_string));
    lines
}

/// The thread that runs during a turn. It redraws the status line and,
/// in the REPL, reads keys with the terminal in cbreak mode.
struct Ticker {
    stop: Arc<AtomicBool>,
    thread: JoinHandle<Option<Keys>>,
}

impl Ticker {
    fn start(screen: Arc<Mutex<Screen>>, keys: Option<Keys>) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let thread = {
            let stop = stop.clone();
            std::thread::spawn(move || {
                let mut keys = keys;
                // Dropped as the thread returns, so `stop` finds the
                // terminal back in its own mode.
                let _cbreak = keys.as_ref().and_then(|k| Cbreak::enter(k.fd()).ok());
                while !stop.load(Ordering::Relaxed) {
                    match keys.as_mut() {
                        Some(keys) => keys.poll(TICK),
                        None => std::thread::sleep(TICK),
                    }
                    lock(&screen).tick();
                }
                keys
            })
        };
        Self { stop, thread }
    }

    /// Returns the keys once the thread has finished, so nothing reads
    /// the terminal or draws after this.
    fn stop(self) -> Option<Keys> {
        self.stop.store(true, Ordering::Relaxed);
        self.thread.join().unwrap_or(None)
    }
}

/// A panic elsewhere must not take the terminal output down with it.
fn lock(screen: &Mutex<Screen>) -> MutexGuard<'_, Screen> {
    screen.lock().unwrap_or_else(PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_prompt_marks_a_place_outside_the_project() {
        let here = std::env::current_dir().unwrap();
        let inside = here.join("README.md").to_string_lossy().into_owned();
        let lines = mcp_lines(
            "files",
            "read_text_file",
            "{}",
            Some("/srv/docs"),
            &[inside.clone(), "/etc/hosts".to_string()],
        );
        let joined = lines.join("\n");
        assert!(joined.contains("serving /srv/docs"), "{joined}");
        assert!(
            lines
                .iter()
                .any(|line| line.contains(&inside) && !line.contains("outside")),
            "{joined}"
        );
        assert!(
            lines
                .iter()
                .any(|line| line.contains("/etc/hosts") && line.contains("(outside this project)")),
            "{joined}"
        );
    }

    use crate::status::tests::{strip_frames, Sink};

    #[test]
    fn without_a_terminal_a_turn_prints_no_status_output() {
        let sink = Sink::default();
        let mut out = Stdout::with(
            Renderer::Plain,
            None,
            false,
            Box::new(sink.clone()),
            Arc::new(AtomicUsize::new(80)),
        );
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
            let mut out = Stdout::with(
                Renderer::rich(60),
                None,
                false,
                Box::new(sink.clone()),
                Arc::new(AtomicUsize::new(60)),
            );
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

    #[test]
    fn esc_cancels_a_streaming_turn_keeps_the_text_and_restores_the_terminal() {
        use std::io::Write as _;
        use std::os::fd::AsRawFd;

        use airlok_core::agent::INTERRUPTED_MARKER;
        use airlok_core::Interrupt;
        use airlok_llm::ContentBlock;
        use airlok_tests::{agent, partial, MockProvider, TempDir};

        use crate::keys::tests::{local_flags, mode, pty, serial};

        let _serial = serial();
        let (mut controller, terminal) = pty();
        let fd = terminal.as_raw_fd();
        let before = mode(fd);
        let interrupt = Interrupt::new();
        let sink = Sink::default();
        let mut out = Stdout::with(
            Renderer::Plain,
            None,
            false,
            Box::new(sink.clone()),
            Arc::new(AtomicUsize::new(80)),
        );
        out.watch_keys(Keys::new(terminal, interrupt.clone(), Arc::default()));
        let dir = TempDir::new("esc");
        let provider = MockProvider::scripted(vec![partial("The answer is")]);
        let mut agent = agent(provider, dir.path());
        let mut session = agent.new_session();
        let presser = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(300));
            controller.write_all(b"\x1b").unwrap();
            controller
        });

        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async {
                out.begin_turn();
                agent
                    .turn_with(&mut session, "go", &mut out, &interrupt)
                    .await
                    .unwrap();
                out.end_turn();
            });
        let _controller = presser.join().unwrap();

        match &session.messages[1].content[0] {
            ContentBlock::Text { text } => {
                assert_eq!(text, &format!("The answer is\n{INTERRUPTED_MARKER}"));
            }
            other => panic!("expected text, got {other:?}"),
        }
        assert_eq!(sink.contents(), "The answer is\ninterrupted\n");
        let after = mode(fd);
        assert_eq!(
            local_flags(&after),
            local_flags(&before),
            "the terminal is back in its mode"
        );
        assert_eq!(after.c_oflag, before.c_oflag);
    }
}
