//! Line input for the REPL through rustyline: history within the run,
//! Ctrl-C at the prompt reported as an interrupt, Ctrl-D as end of input.
//!
//! rustyline ends the accepted line with a newline, so a reply starts on
//! its own line. The cursor check covers the other direction: before the
//! prompt, rustyline asks the terminal for the cursor column and starts a
//! new line if anything (a reply, a stray `^C`, stderr) left it mid-line.

use airlok_core::repl::{Line, LineSource};
use rustyline::error::ReadlineError;
use rustyline::{Config, DefaultEditor};

pub struct Readline {
    editor: DefaultEditor,
}

impl Readline {
    pub fn new() -> anyhow::Result<Self> {
        Ok(Self {
            editor: DefaultEditor::with_config(
                Config::builder().check_cursor_position(true).build(),
            )?,
        })
    }
}

impl LineSource for Readline {
    fn read_line(&mut self, prompt: &str) -> Line {
        match self.editor.readline(prompt) {
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
