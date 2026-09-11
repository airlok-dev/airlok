use airlok_core::config::ProviderName;
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
    /// List saved sessions for this directory, or delete some
    Sessions {
        #[command(subcommand)]
        action: Option<SessionsAction>,
    },
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
