//! Line input for the REPL through rustyline: history within the run,
//! Ctrl-C at the prompt reported as an interrupt, Ctrl-D as end of input.
//! Keys typed during a turn, which the key watcher collected, start the
//! next line.
//!
//! rustyline ends the accepted line with a newline, so a reply starts on
//! its own line. The cursor check covers the other direction: before the
//! prompt, rustyline asks the terminal for the cursor column and starts a
//! new line if anything (a reply, a stray `^C`, stderr) left it mid-line.

use std::sync::{Arc, Mutex, PoisonError};

use airlok_core::repl::{Line, LineSource};
use rustyline::error::ReadlineError;
use rustyline::{Config, DefaultEditor};

pub struct Readline {
    editor: DefaultEditor,
    /// Keys typed during the last turn.
    typed: Arc<Mutex<Vec<u8>>>,
}

impl Readline {
    pub fn new(typed: Arc<Mutex<Vec<u8>>>) -> anyhow::Result<Self> {
        Ok(Self {
            editor: DefaultEditor::with_config(
                Config::builder().check_cursor_position(true).build(),
            )?,
            typed,
        })
    }
}

impl LineSource for Readline {
    fn read_line(&mut self, prompt: &str) -> Line {
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
