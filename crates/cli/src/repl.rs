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

use airlok_core::repl::{AttachPolicy, Attachment, Candidates, Line, LineSource};
use rustyline::error::ReadlineError;
use rustyline::history::DefaultHistory;
use rustyline::{
    Cmd, CompletionType, ConditionalEventHandler, Config, Editor, Event, EventContext,
    EventHandler, KeyCode, KeyEvent, Modifiers, RepeatCount,
};
use tracing::debug;

use crate::complete::Prompt;

pub struct Readline {
    editor: Editor<Prompt, DefaultHistory>,
    /// Keys typed during the last turn.
    typed: Arc<Mutex<Vec<u8>>>,
    /// Images attached to the line being typed, in chip order.
    pending: Arc<Mutex<Vec<Attachment>>>,
    /// What may be attached, refreshed before every prompt.
    policy: Arc<Mutex<AttachPolicy>>,
}

/// Shared so the key handler can reach it: locking is only ever held for
/// the length of a push or a read.
type Pending = Arc<Mutex<Vec<Attachment>>>;

fn locked<T>(slot: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    slot.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Readline {
    pub fn new(cwd: PathBuf, typed: Arc<Mutex<Vec<u8>>>) -> anyhow::Result<Self> {
        let pending: Pending = Arc::new(Mutex::new(Vec::new()));
        let policy = Arc::new(Mutex::new(AttachPolicy {
            model: String::new(),
            accepts_images: true,
        }));
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
        // Ctrl+V is the convention; Alt+V is accepted because some
        // terminals bind Ctrl+V to their own paste.
        for key in [
            KeyEvent::ctrl('V'),
            KeyEvent(KeyCode::Char('v'), Modifiers::ALT),
        ] {
            editor.bind_sequence(
                key,
                EventHandler::Conditional(Box::new(PasteImage {
                    pending: Arc::clone(&pending),
                    policy: Arc::clone(&policy),
                })),
            );
        }
        Ok(Self {
            editor,
            typed,
            pending,
            policy,
        })
    }
}

/// Reads an image from the clipboard and puts a chip in the line. Every
/// failure says why on its own line: a paste that appears to do nothing
/// is the thing this exists to avoid.
struct PasteImage {
    pending: Pending,
    policy: Arc<Mutex<AttachPolicy>>,
}

impl PasteImage {
    /// Prints one line without leaving the prompt mangled: return to
    /// column one, clear the row, write, then let rustyline redraw.
    fn say(reason: &str) {
        eprint!("\r\x1b[K{reason}\r\n");
    }

    fn attach(&self) -> Result<String, String> {
        let policy = locked(&self.policy).clone();
        if !policy.accepts_images {
            return Err(format!(
                "{} does not take images, so nothing was attached",
                policy.model
            ));
        }
        let backend = crate::clipboard::backend().map_err(|e| e.to_string())?;
        debug!(
            backend = backend.name(),
            "reading an image from the clipboard"
        );
        let (bytes, media_type) = backend.read_image().map_err(|e| e.to_string())?;
        let image = airlok_core::image::prepare(&bytes, media_type).map_err(|e| e.to_string())?;

        let mut pending = locked(&self.pending);
        let mut all: Vec<airlok_core::image::Prepared> =
            pending.iter().map(|a| a.image.clone()).collect();
        all.push(image.clone());
        airlok_core::image::within_budget(&all).map_err(|e| e.to_string())?;

        for note in &image.notes {
            Self::say(note);
        }
        let label = format!("image {}", pending.len() + 1);
        let chip = format!("[{label}: {}] ", image.summary());
        pending.push(Attachment { label, image });
        Ok(chip)
    }
}

impl ConditionalEventHandler for PasteImage {
    fn handle(
        &self,
        _event: &Event,
        _count: RepeatCount,
        _positive: bool,
        _ctx: &EventContext,
    ) -> Option<Cmd> {
        match self.attach() {
            Ok(chip) => Some(Cmd::Insert(1, chip)),
            Err(reason) => {
                Self::say(&reason);
                Some(Cmd::Noop)
            }
        }
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

    fn set_attach_policy(&mut self, policy: AttachPolicy) {
        *locked(&self.policy) = policy;
    }

    fn take_attachments(&mut self) -> Vec<Attachment> {
        std::mem::take(&mut *locked(&self.pending))
    }

    fn read_line(&mut self, prompt: &str) -> Line {
        if let Some(helper) = self.editor.helper_mut() {
            helper.refresh();
        }
        let mut typed = std::mem::take(&mut *locked(&self.typed));
        // A Ctrl+V pressed during a turn must not replay as a control
        // byte into the next line.
        typed.retain(|byte| *byte != 0x16);
        let initial = String::from_utf8_lossy(&typed);
        match self.editor.readline_with_initial(prompt, (&initial, "")) {
            Ok(line) => {
                let _ = self.editor.add_history_entry(line.as_str());
                Line::Text(line)
            }
            Err(ReadlineError::Interrupted) => {
                // The line is dropped, so its images go with it.
                locked(&self.pending).clear();
                Line::Interrupt
            }
            Err(ReadlineError::Eof) => Line::Eof,
            Err(e) => {
                eprintln!("input error: {e}");
                Line::Eof
            }
        }
    }
}
