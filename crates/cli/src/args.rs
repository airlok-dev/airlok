use clap::Parser;

/// A privacy airlock between your code and a model you do not control.
#[derive(Debug, Parser)]
#[command(name = "airlok", version, about)]
pub struct Args {
    /// The task to carry out in the current directory
    pub prompt: String,

    /// Print debug logs to stderr
    #[arg(short, long)]
    pub verbose: bool,

    /// Model id to use
    #[arg(long, default_value = airlok_core::config::DEFAULT_MODEL)]
    pub model: String,

    /// After the run, list every value that was redacted before leaving the machine
    #[arg(long)]
    pub show_redactions: bool,
}
