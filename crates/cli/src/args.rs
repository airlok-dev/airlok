use clap::{Parser, ValueEnum};

/// A privacy airlock between your code and a model you do not control.
#[derive(Debug, Parser)]
#[command(name = "airlok", version, about)]
pub struct Args {
    /// The task to carry out in the current directory
    pub prompt: String,

    /// Print debug logs to stderr
    #[arg(short, long)]
    pub verbose: bool,

    /// Which model API to talk to
    #[arg(long, value_enum, default_value_t = ProviderKind::Anthropic)]
    pub provider: ProviderKind,

    /// Model id. Defaults to claude-sonnet-4-6 for anthropic and gpt-5.5 for openai
    #[arg(long)]
    pub model: Option<String>,

    /// After the run, list every value that was redacted before leaving the machine
    #[arg(long)]
    pub show_redactions: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ProviderKind {
    /// Anthropic Messages API, key from ANTHROPIC_API_KEY
    Anthropic,
    /// OpenAI Chat Completions API, key from OPENAI_API_KEY
    #[value(name = "openai")]
    OpenAi,
}

impl ProviderKind {
    pub fn default_model(self) -> &'static str {
        match self {
            ProviderKind::Anthropic => airlok_llm::anthropic::DEFAULT_MODEL,
            ProviderKind::OpenAi => airlok_llm::openai::DEFAULT_MODEL,
        }
    }
}
