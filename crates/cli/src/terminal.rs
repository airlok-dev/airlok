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
        Self::on(OpenOptions::new().read(true).write(true).open("/dev/tty")?)
    }

    fn on(tty: File) -> std::io::Result<Self> {
        Ok(Self {
            reader: BufReader::new(tty.try_clone()?),
            writer: tty,
            hinted: false,
        })
    }

    /// Prints display lines, such as a rendered diff.
    pub fn show_lines(&mut self, lines: &[String]) {
        let text: String = lines.iter().map(|line| format!("{line}\n")).collect();
        let _ = self.writer.write_all(text.as_bytes());
    }

    /// Prints `question` and reads one line: `y`/`yes` approves, `a`/`all`
    /// approves everything of this kind, `q`/`quit` ends the run, anything
    /// else rejects. The first prompt of the run explains `a`.
    pub fn ask(&mut self, question: &str) -> Decision {
        self.ask_paged(question, &[])
    }

    /// [`Terminal::ask`] after a diff shown in part: `v` prints `rest`,
    /// the lines that did not fit, and asks again.
    pub fn ask_paged(&mut self, question: &str, rest: &[String]) -> Decision {
        self.ask_inner(question, rest, false)
    }

    /// [`Terminal::ask_paged`] with `s` as well: approve every later call
    /// like this one and remember it past this run.
    pub fn ask_savable(&mut self, question: &str, rest: &[String]) -> Decision {
        self.ask_inner(question, rest, true)
    }

    fn ask_inner(&mut self, question: &str, mut rest: &[String], savable: bool) -> Decision {
        if !self.hinted {
            self.hinted = true;
            let _ = writeln!(
                self.writer,
                "\x1b[2m(a = approve everything for this run)\x1b[0m"
            );
        }
        let options = if savable {
            "[y]es / [n]o / [a]ll / [s]ave / [q]uit "
        } else {
            "[y]es / [n]o / [a]ll / [q]uit "
        };
        loop {
            let _ = write!(self.writer, "{question} {options}");
            let _ = self.writer.flush();
            let mut answer = String::new();
            if self.reader.read_line(&mut answer).is_err() {
                return Decision::Reject;
            }
            match answer.trim().to_ascii_lowercase().as_str() {
                "y" | "yes" => return Decision::Approve,
                "a" | "all" => return Decision::ApproveAll,
                "s" | "save" if savable => return Decision::SaveAll,
                "q" | "quit" => return Decision::Quit,
                "v" | "view" if !rest.is_empty() => {
                    self.show_lines(rest);
                    rest = &[];
                }
                _ => return Decision::Reject,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;
    use std::os::fd::AsRawFd;

    use crate::keys::tests::pty;

    /// Everything the terminal side wrote, read from the controller.
    fn written(controller: &mut File) -> String {
        let mut out = Vec::new();
        let mut buf = [0u8; 4096];
        loop {
            let mut poll = libc::pollfd {
                fd: controller.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            };
            // SAFETY: one valid pollfd, and its count.
            if unsafe { libc::poll(&mut poll, 1, 100) } <= 0 {
                break;
            }
            match controller.read(&mut buf) {
                Ok(n) if n > 0 => out.extend_from_slice(&buf[..n]),
                _ => break,
            }
        }
        String::from_utf8_lossy(&out).into_owned()
    }

    #[test]
    fn v_prints_the_rest_of_the_diff_and_asks_again() {
        let (mut controller, tty) = pty();
        let mut terminal = Terminal::on(tty).unwrap();
        controller.write_all(b"v\ny\n").unwrap();

        let rest = ["41 + the rest".to_string()];
        assert_eq!(terminal.ask_paged("Apply?", &rest), Decision::Approve);

        let shown = written(&mut controller);
        assert!(shown.contains("41 + the rest"), "{shown:?}");
        assert_eq!(shown.matches("Apply?").count(), 2, "{shown:?}");
    }

    #[test]
    fn save_is_offered_only_where_it_means_something() {
        let (mut controller, tty) = pty();
        let mut terminal = Terminal::on(tty).unwrap();
        controller.write_all(b"s\n").unwrap();
        assert_eq!(
            terminal.ask_savable("Call `x` on `y`?", &[]),
            Decision::SaveAll
        );
        assert!(written(&mut controller).contains("[s]ave"));

        let (mut controller, tty) = pty();
        let mut terminal = Terminal::on(tty).unwrap();
        controller.write_all(b"s\n").unwrap();
        // The same answer to a prompt that cannot be saved is a refusal,
        // not a silent approval.
        assert_eq!(terminal.ask("Run `ls`?"), Decision::Reject);
        assert!(!written(&mut controller).contains("[s]ave"));
    }

    #[test]
    fn v_rejects_when_there_is_nothing_more_to_show() {
        let (mut controller, tty) = pty();
        let mut terminal = Terminal::on(tty).unwrap();
        controller.write_all(b"v\n").unwrap();
        assert_eq!(terminal.ask("Run `ls`?"), Decision::Reject);
    }
}
