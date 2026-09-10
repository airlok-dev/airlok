//! Tools the model may call.
//!
//! TODO(stage N): edit/patch tool with diff preview, glob and grep, and a
//! permission gate in front of `execute`.

pub mod bash;
pub mod read;
pub mod write;

use std::path::Path;
use std::time::Duration;

use airlok_llm::ToolSpec;
use async_trait::async_trait;
use serde_json::Value;

pub use bash::Bash;
pub use read::ReadFile;
pub use write::WriteFile;

#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[error("command timed out after {0:?}")]
    Timeout(Duration),
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    /// JSON schema for `input`.
    fn schema(&self) -> Value;
    /// One line describing a call, shown to the user as `> name: summary`.
    fn summary(&self, input: &Value) -> String;
    async fn execute(&self, input: Value) -> Result<String, ToolError>;
}

#[derive(Default)]
pub struct ToolRegistry {
    tools: Vec<Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// The 0.1 tool set: read_file, write_file, bash.
    pub fn defaults(cwd: &Path, bash_timeout: Duration) -> Self {
        Self::new()
            .with(ReadFile::new(cwd))
            .with(WriteFile::new(cwd))
            .with(Bash::new(cwd, bash_timeout))
    }

    pub fn with(mut self, tool: impl Tool + 'static) -> Self {
        self.tools.push(Box::new(tool));
        self
    }

    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools
            .iter()
            .find(|t| t.name() == name)
            .map(|t| t.as_ref())
    }

    pub fn specs(&self) -> Vec<ToolSpec> {
        self.tools
            .iter()
            .map(|t| ToolSpec {
                name: t.name().to_string(),
                description: t.description().to_string(),
                input_schema: t.schema(),
            })
            .collect()
    }
}

/// Reads a required string argument from tool input.
fn required_str<'a>(input: &'a Value, key: &str) -> Result<&'a str, ToolError> {
    input[key]
        .as_str()
        .ok_or_else(|| ToolError::InvalidInput(format!("missing string field `{key}`")))
}

/// Resolves a model-supplied path against the working directory.
fn resolve(cwd: &Path, path: &str) -> std::path::PathBuf {
    cwd.join(path)
}
