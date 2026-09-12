//! Tools the model may call.

pub mod bash;
pub mod edit;
pub mod glob;
pub mod grep;
pub mod list_dir;
pub mod read;
pub mod write;

use std::path::{Path, PathBuf};
use std::time::Duration;

use airlok_llm::ToolSpec;
use async_trait::async_trait;
use serde_json::Value;

pub use bash::Bash;
pub use edit::EditFile;
pub use glob::Glob;
pub use grep::Grep;
pub use list_dir::ListDir;
pub use read::ReadFile;
pub use write::WriteFile;

/// Tools that never change anything and never ask for confirmation.
pub const READ_ONLY_TOOLS: &[&str] = &["read_file", "glob", "grep", "list_dir"];

/// Largest tool result passed to the model before it is cut.
pub const MAX_TOOL_OUTPUT: usize = 50 * 1024;

/// Cuts an oversized result and tells the model how to page instead.
pub fn truncate_output(output: String) -> String {
    if output.len() <= MAX_TOOL_OUTPUT {
        return output;
    }
    let mut cut = MAX_TOOL_OUTPUT;
    while !output.is_char_boundary(cut) {
        cut -= 1;
    }
    format!(
        "{}\n[output truncated: showing {cut} of {} bytes. For files, call read_file with `offset` and `limit` \
         to page through the rest; for searches, narrow the pattern, path, or include.]\n",
        &output[..cut],
        output.len()
    )
}

#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[error("command timed out after {0:?}")]
    Timeout(Duration),
}

/// What a call would do, decided before it runs so the user can be asked.
#[derive(Debug, Clone, PartialEq)]
pub enum Plan {
    /// Nothing to confirm.
    Safe,
    /// A file will change. `diff` is a plain unified diff.
    Write { path: PathBuf, diff: String },
    /// A shell command will run.
    Command { command: String },
    /// A tool on an external MCP server will be called. `arguments` is
    /// what will actually be sent, placeholders included; `paths` are the
    /// places on disk those arguments name, resolved, so approving one
    /// call cannot approve a call that reaches somewhere else.
    McpCall {
        server: String,
        tool: String,
        arguments: String,
        root: Option<String>,
        paths: Vec<String>,
    },
    /// The configuration forbids this call outright.
    Denied { why: String },
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    /// JSON schema for `input`.
    fn schema(&self) -> Value;
    /// One line describing a call, shown to the user as `> name: summary`.
    fn summary(&self, input: &Value) -> String;
    /// Describes the effect of `execute` without performing it.
    async fn plan(&self, input: &Value) -> Result<Plan, ToolError> {
        let _ = input;
        Ok(Plan::Safe)
    }
    /// Whether the tool is given the secrets behind the placeholders.
    /// True for the built-in tools: an edit or a command must carry the
    /// real value or it would break. A tool that sends arguments off the
    /// machine says false, and receives the placeholders instead.
    fn rehydrate_arguments(&self) -> bool {
        true
    }
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

    /// The built-in tool set.
    pub fn defaults(cwd: &Path, bash_timeout: Duration) -> Self {
        Self::new()
            .with(ReadFile::new(cwd))
            .with(WriteFile::new(cwd))
            .with(EditFile::new(cwd))
            .with(Glob::new(cwd))
            .with(Grep::new(cwd))
            .with(ListDir::new(cwd))
            .with(Bash::new(cwd, bash_timeout))
    }

    pub fn with(mut self, tool: impl Tool + 'static) -> Self {
        self.tools.push(Box::new(tool));
        self
    }

    /// Adds a tool discovered at runtime, such as one from an MCP server.
    /// A name already in the registry wins, so nothing can shadow a
    /// built-in; the rejected name is returned.
    pub fn add(&mut self, tool: Box<dyn Tool>) -> Result<(), String> {
        let name = tool.name().to_string();
        if self.get(&name).is_some() {
            return Err(name);
        }
        self.tools.push(tool);
        Ok(())
    }

    /// Removes every tool whose name starts with `prefix`, for a server
    /// that is turned off during a session.
    pub fn remove_prefixed(&mut self, prefix: &str) -> usize {
        let before = self.tools.len();
        self.tools.retain(|tool| !tool.name().starts_with(prefix));
        before - self.tools.len()
    }

    pub fn names(&self) -> Vec<String> {
        self.tools.iter().map(|t| t.name().to_string()).collect()
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
fn resolve(cwd: &Path, path: &str) -> PathBuf {
    cwd.join(path)
}

/// Files and directories under `root` as (relative path, is_dir), sorted,
/// respecting .gitignore, including dotfiles but never `.git`.
fn walk(root: &Path, max_depth: Option<usize>) -> Vec<(PathBuf, bool)> {
    ignore::WalkBuilder::new(root)
        .hidden(false)
        .max_depth(max_depth)
        .sort_by_file_name(|a, b| a.cmp(b))
        .filter_entry(|e| e.file_name() != ".git")
        .build()
        .filter_map(Result::ok)
        .filter(|e| e.depth() > 0)
        .map(|e| {
            let is_dir = e.file_type().is_some_and(|t| t.is_dir());
            (
                e.path()
                    .strip_prefix(root)
                    .unwrap_or(e.path())
                    .to_path_buf(),
                is_dir,
            )
        })
        .collect()
}

/// Plain unified diff with `a/` and `b/` headers, three lines of context.
pub fn unified_diff(path: &str, old: &str, new: &str) -> String {
    similar::TextDiff::from_lines(old, new)
        .unified_diff()
        .context_radius(3)
        .header(&format!("a/{path}"), &format!("b/{path}"))
        .to_string()
}
