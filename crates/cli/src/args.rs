use std::path::PathBuf;

use airlok_core::config::{McpScope, ProviderName};
use clap::{Parser, Subcommand, ValueEnum};

/// A privacy airlock between your code and a model you do not control.
#[derive(Debug, Parser)]
#[command(
    name = "airlok",
    version,
    about,
    args_conflicts_with_subcommands = true
)]
pub struct Args {
    #[command(subcommand)]
    pub command: Option<Command>,

    /// The task to carry out in the current directory
    pub prompt: Option<String>,

    /// Print debug logs to stderr
    #[arg(short, long)]
    pub verbose: bool,

    /// Run without asking: no diff confirmations, no command confirmations
    #[arg(short, long)]
    pub yes: bool,

    /// Which model API to talk to (default: from config, else anthropic)
    #[arg(long, value_enum)]
    pub provider: Option<ProviderKind>,

    /// Model id, or the deployment name on Azure (default: from config, else the provider's default)
    #[arg(long)]
    pub model: Option<String>,

    /// After the run, list every value that was redacted before leaving the machine
    #[arg(long)]
    pub show_redactions: bool,

    /// Start in plan mode: read-only tools, and the model replies with a plan
    #[arg(long)]
    pub plan: bool,

    /// Continue a saved session from this directory: the latest, or `--resume=<id>`
    #[arg(long, value_name = "ID", require_equals = true, num_args = 0..=1, default_missing_value = "")]
    pub resume: Option<String>,
}

impl Args {
    /// `None` for a fresh session, `Some(None)` for the latest saved one,
    /// `Some(Some(id))` for a particular one.
    pub fn resume(&self) -> Option<Option<&str>> {
        self.resume
            .as_deref()
            .map(|id| Some(id).filter(|id| !id.is_empty()))
    }
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Inspect or create the configuration file
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },
    /// Print the context block that would be sent with the system prompt, after redaction
    Context,
    /// List the kinds of value the redactor detects and how each is treated
    Redactions,
    /// Inspect the configured MCP servers, or call one of their tools
    Mcp {
        #[command(subcommand)]
        action: Option<McpAction>,
    },
    /// List saved sessions for this directory, or delete some
    Sessions {
        #[command(subcommand)]
        action: Option<SessionsAction>,
    },
    /// Check everything airlok needs, and exit non-zero if anything fails
    Doctor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ScopeArg {
    /// ~/.config/airlok/mcp.json
    User,
    /// ./.mcp.json, meant to be committed
    Project,
    /// ./.airlok/mcp.json, personal and gitignored
    Local,
}

impl From<ScopeArg> for McpScope {
    fn from(scope: ScopeArg) -> Self {
        match scope {
            ScopeArg::User => McpScope::User,
            ScopeArg::Project => McpScope::Project,
            ScopeArg::Local => McpScope::Local,
        }
    }
}

#[derive(Debug, Subcommand)]
pub enum McpAction {
    /// Show each server, its scope, whether it answers, and its tools
    List,
    /// Write a server to one of the JSON files
    Add {
        name: String,
        /// Which file to write it to
        #[arg(long, value_enum, default_value = "local")]
        scope: ScopeArg,
        /// stdio runs the command after --; http posts to --url
        #[arg(long, value_enum)]
        transport: Option<TransportArg>,
        /// The endpoint, for an http server
        #[arg(long)]
        url: Option<String>,
        /// An environment variable for the server, repeatable
        #[arg(long = "env", value_name = "K=V")]
        env: Vec<String>,
        /// An http header, repeatable
        #[arg(long = "header", value_name = "K=V")]
        header: Vec<String>,
        /// The command and its arguments, after --
        #[arg(last = true, value_name = "COMMAND")]
        command: Vec<String>,
    },
    /// Remove a server from a JSON file
    Remove {
        name: String,
        /// Which file to take it out of; every scope by default
        #[arg(long, value_enum)]
        scope: Option<ScopeArg>,
    },
    /// Show the resolved entry for one server and where it came from
    Get { name: String },
    /// Merge another tool's mcpServers file into one of the scopes
    Import {
        path: PathBuf,
        #[arg(long, value_enum, default_value = "local")]
        scope: ScopeArg,
    },
    /// Print the configured servers as standard mcpServers JSON
    Export {
        /// Only this scope, rather than everything resolved
        #[arg(long, value_enum)]
        scope: Option<ScopeArg>,
    },
    /// Write a server from a JSON entry, as other tools accept it
    AddJson {
        name: String,
        /// The entry object, for example '{"command":"npx","args":["-y","pkg"]}'
        json: String,
        #[arg(long, value_enum, default_value = "local")]
        scope: ScopeArg,
    },
    /// Forget whether this repository's own .mcp.json servers may start
    ResetProjectChoices,
    /// Approvals remembered past a run
    Trust {
        #[command(subcommand)]
        action: TrustAction,
    },
    /// Call one tool, for debugging. The same trust and confirmation apply
    Call {
        server: String,
        tool: String,
        /// Arguments as a JSON object
        #[arg(default_value = "{}")]
        arguments: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum TransportArg {
    Stdio,
    Http,
}

#[derive(Debug, Subcommand)]
pub enum TrustAction {
    /// Show what has been approved past the end of a run
    List,
    /// Forget every saved approval for one server
    Revoke { server: String },
}

#[derive(Debug, Subcommand)]
pub enum SessionsAction {
    /// Delete one session by id (a unique prefix is enough)
    Rm { id: String },
    /// Delete sessions from every directory not updated within the given age, e.g. 30d, 12h, 90m
    Clean {
        #[arg(long, value_name = "AGE")]
        older_than: String,
    },
}

#[derive(Debug, Subcommand)]
pub enum ConfigAction {
    /// Write a commented default config to the user config path
    Init,
    /// Print the effective configuration and where it came from
    Show,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ProviderKind {
    /// Anthropic Messages API
    Anthropic,
    /// OpenAI Chat Completions API, including Azure OpenAI
    #[value(name = "openai")]
    OpenAi,
}

impl From<ProviderKind> for ProviderName {
    fn from(kind: ProviderKind) -> Self {
        match kind {
            ProviderKind::Anthropic => ProviderName::Anthropic,
            ProviderKind::OpenAi => ProviderName::OpenAi,
        }
    }
}
