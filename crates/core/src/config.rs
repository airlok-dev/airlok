//! Run configuration.
//!
//! TODO(stage N): load from `airlok.toml` (project and user level) and merge
//! with CLI flags; add user-defined redaction patterns and allowed tools.

use std::path::PathBuf;
use std::time::Duration;

/// The agent's default model, which is the Anthropic provider's default.
pub use airlok_llm::anthropic::DEFAULT_MODEL;

#[derive(Debug, Clone)]
pub struct Config {
    pub model: String,
    pub max_tokens: u32,
    /// Upper bound on model round-trips in one run, so a confused model
    /// cannot loop forever.
    pub max_turns: usize,
    pub bash_timeout: Duration,
    /// Directory the agent works in. Tools resolve relative paths against it.
    pub cwd: PathBuf,
}

impl Config {
    pub fn new(cwd: PathBuf) -> Self {
        Self {
            model: DEFAULT_MODEL.to_string(),
            max_tokens: 8192,
            max_turns: 50,
            bash_timeout: Duration::from_secs(120),
            cwd,
        }
    }
}
