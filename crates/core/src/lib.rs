//! The airlok agent: a loop that talks to a model through a privacy airlock.
//!
//! This crate has no knowledge of the terminal. Everything a user sees goes
//! through the [`Output`] trait, which the binary implements.

pub mod agent;
pub mod config;
pub mod context;
pub mod interrupt;
pub mod redact;
pub mod repl;
pub mod safety;
pub mod session;
pub mod tools;

pub use agent::{Agent, RunReport};
pub use config::{Config, ConfigError};
pub use interrupt::Interrupt;
pub use safety::{Confirmation, Decision};
pub use session::{Session, SessionStore};

/// Where user-facing output goes. Implemented by the CLI.
pub trait Output: Send {
    /// A fragment of model text, already rehydrated, in arrival order.
    fn text(&mut self, chunk: &str);
    /// One tool call about to run, e.g. name `bash` with summary `ls`.
    fn tool_call(&mut self, name: &str, summary: &str);
    /// A one-line note about the run itself, such as a compaction. Shown
    /// dim, never part of the model's text.
    fn status(&mut self, line: &str);
    /// The turn's text is complete; flush anything held back.
    fn end_turn(&mut self) {}
    /// Ask the user before a write or a command. Only called when the
    /// configuration says to confirm.
    fn confirm(&mut self, request: &Confirmation<'_>) -> Decision;
}

#[derive(Debug, thiserror::Error)]
pub enum CoreError {
    #[error(transparent)]
    Llm(#[from] airlok_llm::LlmError),
    #[error("agent stopped after {0} turns without finishing")]
    TurnLimit(usize),
    #[error("run aborted by the user at a confirmation prompt")]
    Aborted,
}
