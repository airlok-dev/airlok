//! Line input for the REPL through rustyline: history within the run,
//! Ctrl-C at the prompt reported as an interrupt, Ctrl-D as end of input.

use airlok_core::repl::{Line, LineSource};
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;

pub struct Readline {
    editor: DefaultEditor,
}

impl Readline {
    pub fn new() -> anyhow::Result<Self> {
        Ok(Self {
            editor: DefaultEditor::new()?,
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
