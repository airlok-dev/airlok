//! Confirmation prompts on the controlling terminal.
//!
//! `/dev/tty` is used instead of stdin so a prompt still works when stdin
//! is a pipe, and stdout stays clean for the model's text.

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};

use airlok_core::Decision;

pub struct Terminal {
    reader: BufReader<File>,
    writer: File,
    hinted: bool,
}

impl Terminal {
    /// Fails when the process has no controlling terminal.
    pub fn open() -> std::io::Result<Self> {
        let writer = OpenOptions::new().read(true).write(true).open("/dev/tty")?;
        let reader = BufReader::new(writer.try_clone()?);
        Ok(Self {
            reader,
            writer,
            hinted: false,
        })
    }

    pub fn show_diff(&mut self, diff: &str) {
        let mut painted = String::new();
        for line in diff.lines() {
            let color = match line.as_bytes().first() {
                Some(b'+') if !line.starts_with("+++") => "\x1b[32m",
                Some(b'-') if !line.starts_with("---") => "\x1b[31m",
                Some(b'@') => "\x1b[36m",
                _ if line.starts_with("+++") || line.starts_with("---") => "\x1b[1m",
                _ => "",
            };
            let reset = if color.is_empty() { "" } else { "\x1b[0m" };
            painted.push_str(&format!("{color}{line}{reset}\n"));
        }
        let _ = self.writer.write_all(painted.as_bytes());
    }

    /// Prints `question` and reads one line: `y`/`yes` approves, `a`/`all`
    /// approves everything of this kind, `q`/`quit` ends the run, anything
    /// else rejects. The first prompt of the run explains `a`.
    pub fn ask(&mut self, question: &str) -> Decision {
        if !self.hinted {
            self.hinted = true;
            let _ = writeln!(
                self.writer,
                "\x1b[2m(a = approve everything for this run)\x1b[0m"
            );
        }
        let _ = write!(self.writer, "{question} [y]es / [n]o / [a]ll / [q]uit ");
        let _ = self.writer.flush();
        let mut answer = String::new();
        if self.reader.read_line(&mut answer).is_err() {
            return Decision::Reject;
        }
        match answer.trim().to_ascii_lowercase().as_str() {
            "y" | "yes" => Decision::Approve,
            "a" | "all" => Decision::ApproveAll,
            "q" | "quit" => Decision::Quit,
            _ => Decision::Reject,
        }
    }
}
