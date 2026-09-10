mod args;
mod terminal;

use std::io::Write;
use std::path::Path;
use std::sync::Arc;

use airlok_core::config::{self, KeySource, Overrides, ProviderName, Sources};
use airlok_core::redact::SecretRedactor;
use airlok_core::tools::ToolRegistry;
use airlok_core::{Agent, Config, Confirmation, Decision, Output, RunReport};
use airlok_llm::openai::Auth;
use airlok_llm::{Anthropic, OpenAi, Provider};
use anyhow::{anyhow, bail, Context};
use clap::Parser;
use tracing_subscriber::EnvFilter;

use args::{Args, Command, ConfigAction};
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
    let user_path = config::user_config_path();
    let project_path = config::project_config_path(&cwd);
    let overrides = Overrides {
        provider: args.provider.map(Into::into),
        model: args.model,
        yes: args.yes,
    };
    let (config, sources) = Config::load(
        user_path.as_deref(),
        Some(&project_path),
        &overrides,
        cwd.clone(),
    )?;

    match args.command {
        Some(Command::Config {
            action: ConfigAction::Init,
        }) => return config_init(user_path.as_deref()),
        Some(Command::Config {
            action: ConfigAction::Show,
        }) => return config_show(&config, &sources),
        None => {}
    }
    let Some(prompt) = args.prompt else {
        bail!("missing prompt. Usage: airlok \"<task>\" (see airlok --help)");
    };

    let prompting = config.safety.confirm_writes || config.safety.confirm_bash;
    let terminal = if prompting {
        Some(Terminal::open().map_err(|e| {
            anyhow!(
                "confirmations are on but no terminal is available ({e}). \
                 Pass --yes to run without prompts, or set safety.confirm_writes \
                 and safety.confirm_bash to false in the config."
            )
        })?)
    } else {
        eprintln!("warning: confirmations are off; writes and commands will run without asking");
        None
    };

    // The key is read once, after every check that could abort the run.
    let key = config
        .resolve_key()
        .with_context(|| format!("cannot read the API key from {}", config.key_source()))?;
    let provider = build_provider(&config, key.clone());
    let redactor = SecretRedactor::new().with_known([key]);
    let tools = ToolRegistry::defaults(&cwd, config.agent.bash_timeout);

    let mut agent = Agent::new(provider, tools, Box::new(redactor), config);
    let mut out = Stdout {
        mid_line: false,
        terminal,
    };
    let report = agent.run(&prompt, &mut out).await?;
    out.end_line();

    if args.show_redactions {
        print_redactions(&report);
    }
    Ok(())
}

fn build_provider(config: &Config, key: String) -> Arc<dyn Provider> {
    match config.provider.name {
        ProviderName::Anthropic => {
            let mut provider = Anthropic::new(key);
            if let Some(base_url) = &config.provider.base_url {
                provider = provider.with_base_url(base_url);
            }
            Arc::new(provider)
        }
        ProviderName::OpenAi => {
            let base_url = config
                .provider
                .base_url
                .clone()
                .unwrap_or_else(|| airlok_llm::openai::DEFAULT_BASE_URL.to_string());
            let key_env = match config.key_source() {
                KeySource::Env(name) => Some(name),
                KeySource::Command(_) => None,
            };
            let auth = Auth::for_endpoint(&base_url, key_env.as_deref(), key);
            Arc::new(OpenAi::new(auth, base_url))
        }
    }
}

fn config_init(path: Option<&Path>) -> anyhow::Result<()> {
    let path = path.ok_or_else(|| anyhow!("cannot locate the user config dir: HOME is not set"))?;
    if path.exists() {
        bail!("{} already exists; not overwriting", path.display());
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("cannot create {}", parent.display()))?;
    }
    std::fs::write(path, config::TEMPLATE)
        .with_context(|| format!("cannot write {}", path.display()))?;
    println!("wrote {}", path.display());
    Ok(())
}

/// Prints the merged configuration. The key itself is never read here.
fn config_show(config: &Config, sources: &Sources) -> anyhow::Result<()> {
    let describe = |layer: &Option<config::Layer>| match layer {
        Some(layer) => format!(
            "{} ({})",
            layer.path.display(),
            if layer.found { "found" } else { "missing" }
        ),
        None => "(not looked up)".to_string(),
    };
    println!("# user config:    {}", describe(&sources.user));
    println!("# project config: {}", describe(&sources.project));
    println!("# api key:        {}", config.key_source());
    println!();
    print!("{}", toml::to_string_pretty(&config.to_file())?);
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
