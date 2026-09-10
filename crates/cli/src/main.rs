mod args;
mod terminal;

use std::io::Write;
use std::sync::Arc;

use airlok_core::redact::SecretRedactor;
use airlok_core::tools::ToolRegistry;
use airlok_core::{Agent, Config, Confirmation, Decision, Output, RunReport};
use airlok_llm::{Anthropic, OpenAi, Provider};
use anyhow::Context;
use clap::Parser;
use tracing_subscriber::EnvFilter;

use args::{Args, ProviderKind};
use terminal::Terminal;

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
    let provider: Arc<dyn Provider> = match args.provider {
        ProviderKind::Anthropic => Arc::new(Anthropic::from_env().context("no model credentials")?),
        ProviderKind::OpenAi => Arc::new(OpenAi::from_env().context("no model credentials")?),
    };
    let mut config = Config::new(cwd.clone());
    config.provider.model = args
        .model
        .unwrap_or_else(|| args.provider.default_model().to_string());
    let tools = ToolRegistry::defaults(&cwd, config.agent.bash_timeout);

    let mut agent = Agent::new(provider, tools, Box::new(SecretRedactor::new()), config);
    let mut out = Stdout {
        mid_line: false,
        terminal: Terminal::open().ok(),
    };
    let report = agent.run(&args.prompt, &mut out).await?;
    out.end_line();

    if args.show_redactions {
        print_redactions(&report);
    }
    Ok(())
}

/// Streams model text to stdout, prints each tool call on its own line,
/// and asks for confirmations on the terminal.
struct Stdout {
    mid_line: bool,
    terminal: Option<Terminal>,
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

    fn confirm(&mut self, request: &Confirmation<'_>) -> Decision {
        self.end_line();
        let Some(terminal) = self.terminal.as_mut() else {
            return Decision::Reject;
        };
        match request {
            Confirmation::Write { diff, .. } => {
                terminal.show_diff(diff);
                terminal.ask("Apply?")
            }
            Confirmation::Command { command } => terminal.ask(&format!("Run `{command}`?")),
        }
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
