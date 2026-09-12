//! The airlok agent: a loop that talks to a model through a privacy airlock.
//!
//! This crate has no knowledge of the terminal. Everything a user sees goes
//! through the [`Output`] trait, which the binary implements.

pub mod agent;
pub mod config;
pub mod context;
pub mod interrupt;
pub mod mcp;
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
    /// A turn is about to run. Whoever runs it calls this first and
    /// [`Output::end_turn`] after, so a terminal can show progress between.
    fn begin_turn(&mut self) {}
    /// A model request is about to stream.
    fn thinking(&mut self) {}
    /// Tokens used so far in this turn: each finished request's input and
    /// output as reported (or estimated at chars/4), plus estimates for the
    /// request streaming now.
    fn tokens(&mut self, _used: u64) {}
    /// What a command the user ran with `!` printed, shown as is.
    fn command_output(&mut self, text: &str) {
        text.lines().for_each(|line| self.status(line));
    }
    /// The turn's text is complete; flush anything held back.
    fn end_turn(&mut self) {}
    /// Ask the user before a write or a command. Only called when the
    /// configuration says to confirm.
    fn confirm(&mut self, request: &Confirmation<'_>) -> Decision;
}

impl CoreError {
    /// How to fix this in the config, when a setting can. Today: a provider
    /// that rejects its default `reasoning_effort` for `model`.
    pub fn hint(&self, model: &str) -> Option<String> {
        let CoreError::Llm(airlok_llm::LlmError::Api { status: 400, body }) = self else {
            return None;
        };
        let value: serde_json::Value = serde_json::from_str(body).ok()?;
        if value["error"]["param"].as_str()? != "reasoning_effort" {
            return None;
        }
        Some(format!(
            "hint: the provider rejected its reasoning effort for {model}. \
             Set one in the config, for example:\n[models.\"{model}\"]\nreasoning_effort = \"none\""
        ))
    }
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
