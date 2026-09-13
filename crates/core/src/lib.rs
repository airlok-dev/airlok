//! The airlok agent: a loop that talks to a model through a privacy airlock.
//!
//! This crate has no knowledge of the terminal. Everything a user sees goes
//! through the [`Output`] trait, which the binary implements.

pub mod agent;
pub mod config;
pub mod context;
pub mod image;
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
    /// A unified diff to show, already complete. The default prints it as
    /// plain lines; a terminal renders and pages it the way it does for a
    /// write confirmation.
    fn diff(&mut self, path: &str, unified: &str) {
        self.status(path);
        unified.lines().for_each(|line| self.status(line));
    }
    /// Ask the user before a write or a command. Only called when the
    /// configuration says to confirm.
    fn confirm(&mut self, request: &Confirmation<'_>) -> Decision;
}

impl CoreError {
    /// One sentence for the failures a user can act on, naming the likely
    /// cause and what to change. `None` when the error is better shown as
    /// it is. The raw body is never in here: it goes to the debug log, so
    /// `-v` still has it.
    pub fn explain(&self, model: &str, provider: &str) -> Option<String> {
        let CoreError::Llm(airlok_llm::LlmError::Api { status, body }) = self else {
            return None;
        };
        let body_says = |needle: &str| body.to_ascii_lowercase().contains(needle);
        Some(match status {
            404 => format!(
                "{provider} has no model called {model}. On Azure that is the deployment name,                  not the model family. /model lists the ids airlok knows, and `airlok config show`                  says which base_url it is asking."
            ),
            401 | 403 => format!(
                "{provider} rejected the API key. `airlok config show` says where the key comes                  from; check that command or environment variable, and that the key is for this                  base_url."
            ),
            429 => format!(
                "{provider} is rate limiting this key. Wait and send it again, or switch with                  /model or /provider. Azure returns this when a deployment is out of quota as                  well as when it is busy."
            ),
            400 if body_says("context length")
                || body_says("context_length")
                || body_says("too many tokens")
                || body_says("maximum context") =>
            {
                format!(
                    "the request was longer than {model} accepts. /compact summarises the older                      turns, /clear starts fresh, and [provider] context_window tells airlok the                      real size so it compacts on its own."
                )
            }
            400 if body_says("deploymentnotfound") => format!(
                "{provider} has no deployment called {model}. On Azure the model id is the                  deployment name you created."
            ),
            _ => return None,
        })
    }

    /// How to fix this in the config, when a setting can. Today: a provider
    /// that rejects its default `reasoning_effort` for `model`.
    /// `session_effort` is what `/effort` set this run, when it set
    /// anything; `config_effort` is what the files say. A rejection caused
    /// by a session override is fixed with another `/effort`, not by
    /// editing a config that already says the right thing.
    pub fn hint(
        &self,
        model: &str,
        session_effort: Option<&str>,
        config_effort: Option<&str>,
    ) -> Option<String> {
        let CoreError::Llm(airlok_llm::LlmError::Api { status: 400, body }) = self else {
            return None;
        };
        let value: serde_json::Value = serde_json::from_str(body).ok()?;
        // Chat Completions names it `reasoning_effort`; the Responses API
        // names the same setting `reasoning.effort`.
        let param = value["error"]["param"].as_str()?;
        if param != "reasoning_effort" && param != "reasoning.effort" {
            return None;
        }
        if let Some(in_force) = session_effort.filter(|v| Some(*v) != config_effort) {
            // Naming the rejected value as the way back is no help, which
            // is what happens when none is both the override and the
            // fallback, as on a deployment that does not accept none.
            let back_to = config_effort.filter(|v| *v != in_force);
            return Some(match back_to {
                Some(value) => format!(
                    "hint: {model} rejected the reasoning effort {in_force}, which /effort set \
                     for this session. `/effort {value}` puts it back to the config's; the \
                     config is not the problem."
                ),
                None if in_force == "none" => format!(
                    "hint: {model} rejected the reasoning effort none, which /effort set for \
                     this session. Choose one of the values the provider lists above, with \
                     /effort <value>."
                ),
                None => format!(
                    "hint: {model} rejected the reasoning effort {in_force}, which /effort set \
                     for this session. `/effort none` puts it back; the config is not the problem."
                ),
            });
        }
        // Suggesting the configured value is useless when that value is
        // the one being rejected.
        if let Some(configured) = config_effort {
            return Some(format!(
                "hint: [models.\"{model}\"] sets reasoning_effort = \"{configured}\", which the \
                 provider rejected. Use one of the values it lists above, or remove the line to \
                 take the provider's default."
            ));
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
