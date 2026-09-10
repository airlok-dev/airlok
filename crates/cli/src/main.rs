mod args;

use std::io::Write;
use std::sync::Arc;

use airlok_core::redact::SecretRedactor;
use airlok_core::tools::ToolRegistry;
use airlok_core::{Agent, Config, Output, RunReport};
use airlok_llm::Anthropic;
use anyhow::Context;
use clap::Parser;
use tracing_subscriber::EnvFilter;

use args::Args;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    if args.verbose {
        let filter = EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new("airlok=debug,airlok_core=debug,airlok_llm=debug"));
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_writer(std::io::stderr)
            .init();
    }

    let cwd = std::env::current_dir().context("cannot determine working directory")?;
    let provider = Anthropic::from_env().context("no model credentials")?;
    let mut config = Config::new(cwd.clone());
    config.model = args.model;
    let tools = ToolRegistry::defaults(&cwd, config.bash_timeout);

    let mut agent = Agent::new(
        Arc::new(provider),
        tools,
        Box::new(SecretRedactor::new()),
        config,
    );
    let mut out = Stdout::default();
    let report = agent.run(&args.prompt, &mut out).await?;
    out.end_line();

    if args.show_redactions {
        print_redactions(&report);
    }
    Ok(())
}

/// Streams model text to stdout and prints each tool call on its own line.
#[derive(Default)]
struct Stdout {
    mid_line: bool,
}

impl Stdout {
    fn end_line(&mut self) {
        if self.mid_line {
            println!();
            self.mid_line = false;
        }
    }
}

impl Output for Stdout {
    fn text(&mut self, chunk: &str) {
        let mut stdout = std::io::stdout().lock();
        let _ = stdout.write_all(chunk.as_bytes());
        let _ = stdout.flush();
        self.mid_line = !chunk.ends_with('\n');
    }

    fn tool_call(&mut self, name: &str, summary: &str) {
        self.end_line();
        println!("> {name}: {summary}");
    }
}

fn print_redactions(report: &RunReport) {
    if report.redactions.is_empty() {
        println!("(nothing was redacted)");
        return;
    }
    println!("Redacted before leaving this machine:");
    for (placeholder, secret) in &report.redactions {
        println!("  {placeholder}  {}", mask(secret));
    }
}

/// Shows enough of a secret to recognise it without printing all of it.
fn mask(secret: &str) -> String {
    let shown: String = secret.chars().take(8).collect();
    format!("{shown}… ({} chars)", secret.chars().count())
}
