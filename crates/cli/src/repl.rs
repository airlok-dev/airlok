//! Line input for the REPL through rustyline: history within the run,
//! completion and menus from [`Prompt`], Alt+Enter for a newline, Ctrl-C
//! that clears a line with text and does nothing on an empty one, and
//! Ctrl-D as end of input. Keys typed during a turn, which the key watcher
//! collected, start the next line.
//!
//! rustyline ends the accepted line with a newline, so a reply starts on
//! its own line. The cursor check covers the other direction: before the
//! prompt, rustyline asks the terminal for the cursor column and starts a
//! new line if anything (a reply, a stray `^C`, stderr) left it mid-line.

use std::path::PathBuf;
use std::sync::{Arc, Mutex, PoisonError};

use airlok_core::repl::{Candidates, Line, LineSource};
use rustyline::error::ReadlineError;
use rustyline::history::DefaultHistory;
use rustyline::{
    Cmd, CompletionType, ConditionalEventHandler, Config, Editor, Event, EventContext,
    EventHandler, KeyCode, KeyEvent, Modifiers, RepeatCount,
};

use crate::complete::Prompt;

pub struct Readline {
    editor: Editor<Prompt, DefaultHistory>,
    /// Keys typed during the last turn.
    typed: Arc<Mutex<Vec<u8>>>,
}

impl Readline {
    pub fn new(cwd: PathBuf, typed: Arc<Mutex<Vec<u8>>>) -> anyhow::Result<Self> {
        let config = Config::builder()
            .check_cursor_position(true)
            .completion_type(CompletionType::Circular)
            .build();
        let mut editor = Editor::with_config(config)?;
        editor.set_helper(Some(Prompt::new(
            cwd,
            std::env::var_os("NO_COLOR").is_none(),
        )));
        // Terminals send Esc then CR for Alt+Enter; some can be set to
        // send the same for Shift+Enter.
        editor.bind_sequence(
            KeyEvent(KeyCode::Enter, Modifiers::ALT),
            EventHandler::Simple(Cmd::Newline),
        );
        editor.bind_sequence(
            KeyEvent::ctrl('C'),
            EventHandler::Conditional(Box::new(QuietCtrlC)),
        );
        Ok(Self { editor, typed })
    }
}

/// Ctrl-C on an empty line does nothing. On a line with text it keeps its
/// default, an interrupt, which drops the line.
struct QuietCtrlC;

impl ConditionalEventHandler for QuietCtrlC {
    fn handle(
        &self,
        _event: &Event,
        _count: RepeatCount,
        _positive: bool,
        ctx: &EventContext,
    ) -> Option<Cmd> {
        ctx.line().is_empty().then_some(Cmd::Noop)
    }
}

impl LineSource for Readline {
    fn set_candidates(&mut self, candidates: Candidates) {
        if let Some(helper) = self.editor.helper_mut() {
            helper.set_candidates(candidates);
        }
    }

    fn read_line(&mut self, prompt: &str) -> Line {
        if let Some(helper) = self.editor.helper_mut() {
            helper.refresh();
        }
        let typed = std::mem::take(&mut *self.typed.lock().unwrap_or_else(PoisonError::into_inner));
        let initial = String::from_utf8_lossy(&typed);
        match self.editor.readline_with_initial(prompt, (&initial, "")) {
            Ok(line) => {
                let _ = self.editor.add_history_entry(line.as_str());
                Line::Text(line)
            }
            Err(ReadlineError::Interrupted) => Line::Interrupt,
            Err(ReadlineError::Eof) => Line::Eof,
            Err(e) => {
                eprintln!("input error: {e}");
                Line::Eof
            }
        }
    }
}
