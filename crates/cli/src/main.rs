mod args;
mod render;
mod terminal;

use std::io::Write;
use std::path::Path;
use std::sync::Arc;

use airlok_core::config::{self, KeySource, Overrides, ProviderName, Sources};
use airlok_core::context::{self, ContextInput};
use airlok_core::redact::{Class, Redactor, SecretRedactor};
use airlok_core::tools::{ToolRegistry, READ_ONLY_TOOLS};
use airlok_core::CoreError;
use airlok_core::{Agent, Config, Confirmation, Decision, Output, RunReport};
use airlok_llm::openai::Auth;
use airlok_llm::{Anthropic, OpenAi, Provider};
use anyhow::{anyhow, bail, Context};
use clap::Parser;
use tracing_subscriber::EnvFilter;

use args::{Args, Command, ConfigAction};
use render::Renderer;
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

    let user_instructions = user_path
        .as_deref()
        .and_then(Path::parent)
        .map(|dir| dir.join("AIRLOK.md"));
    let context = context::build(&ContextInput {
        cwd: &cwd,
        user_instructions: user_instructions.as_deref(),
        max_bytes: config.context.max_bytes,
    });

    match args.command {
        Some(Command::Config {
            action: ConfigAction::Init,
        }) => return config_init(user_path.as_deref()),
        Some(Command::Config {
            action: ConfigAction::Show,
        }) => return config_show(&config, &sources),
        Some(Command::Redactions) => {
            println!("{:<28} {:<12} meaning", "kind", "class");
            for (kind, class) in SecretRedactor::new().catalog() {
                println!(
                    "{kind:<28} {:<12} restored into files and commands; masked in the terminal",
                    class.as_str()
                );
            }
            println!(
                "{:<28} {:<12} never restored anywhere; [redacted] in the terminal, refused in tool arguments",
                "the provider API key",
                Class::RedactOnly.as_str()
            );
            return Ok(());
        }
        Some(Command::Context) => {
            // What leaves the machine: the block after redaction. The
            // provider key is not resolved here, so it is not in the map.
            let (redacted, _) = SecretRedactor::new().redact(&context.text);
            print!("{redacted}");
            return Ok(());
        }
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
    let redactor =
        SecretRedactor::new().with_known("the provider API key", &key, Class::RedactOnly);
    let tools = ToolRegistry::defaults(&cwd, config.agent.bash_timeout);

    let mut agent =
        Agent::new(provider, tools, Box::new(redactor), config).with_context(context.text);
    let mut out = Stdout::new(terminal, args.verbose);
    let result = agent.run(&prompt, &mut out).await;
    out.finish();
    let report = match result {
        Ok(report) => report,
        Err(CoreError::Aborted) => {
            eprintln!("aborted");
            std::process::exit(1);
        }
        Err(e) => return Err(e.into()),
    };

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
            let base_url = openai_base_url(
                config.provider.base_url.as_deref(),
                std::env::var("OPENAI_BASE_URL").ok().as_deref(),
            );
            let key_env = match config.key_source() {
                KeySource::Env(name) => Some(name),
                KeySource::Command(_) => None,
            };
            let auth = Auth::for_endpoint(&base_url, key_env.as_deref(), key);
            Arc::new(OpenAi::new(auth, base_url))
        }
    }
}

/// Config file first, then the `OPENAI_BASE_URL` environment variable
/// (kept from 0.1), then the public endpoint.
fn openai_base_url(configured: Option<&str>, from_env: Option<&str>) -> String {
    configured
        .or(from_env)
        .map(str::trim)
        .filter(|u| !u.is_empty())
        .unwrap_or(airlok_llm::openai::DEFAULT_BASE_URL)
        .to_string()
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

/// Streams model text to stdout as rendered markdown, shows tool calls
/// dimmed (collapsing runs of read-only calls), and asks for
/// confirmations on the terminal.
struct Stdout {
    renderer: Renderer,
    terminal: Option<Terminal>,
    verbose: bool,
    /// Something is on the current line that needs a newline before
    /// the next block of output.
    mid_line: bool,
    /// Consecutive read-only calls shown as one updating line.
    collapsed: usize,
}

impl Stdout {
    fn new(terminal: Option<Terminal>, verbose: bool) -> Self {
        Self {
            renderer: Renderer::for_stdout(),
            terminal,
            verbose,
            mid_line: false,
            collapsed: 0,
        }
    }

    fn write(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        let mut stdout = std::io::stdout().lock();
        let _ = stdout.write_all(text.as_bytes());
        let _ = stdout.flush();
        self.mid_line = !text.ends_with('\n');
    }

    fn end_line(&mut self) {
        self.collapsed = 0;
        if self.mid_line {
            self.write("\n");
        }
    }

    /// Flushes held markdown and closes the line at the end of the run.
    fn finish(&mut self) {
        let rest = self.renderer.finish();
        self.write(&rest);
        self.end_line();
    }
}

impl Output for Stdout {
    fn text(&mut self, chunk: &str) {
        if self.collapsed > 0 {
            self.end_line();
        }
        let rendered = self.renderer.push(chunk);
        self.write(&rendered);
    }

    fn tool_call(&mut self, name: &str, summary: &str) {
        let rest = self.renderer.finish();
        self.write(&rest);
        let read_only = READ_ONLY_TOOLS.contains(&name);
        if read_only && self.renderer.is_rich() && !self.verbose {
            self.collapsed += 1;
            let n = self.collapsed;
            let line = format!(
                "\r\x1b[2K\x1b[2mreading {n} file{}...\x1b[0m",
                if n == 1 { "" } else { "s" }
            );
            self.write(&line);
            return;
        }
        self.end_line();
        if self.renderer.is_rich() {
            self.write(&format!("\x1b[2m> {name}: {summary}\x1b[0m\n"));
        } else {
            self.write(&format!("> {name}: {summary}\n"));
        }
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

/// Lists what was redacted as kind and length only. Values never appear,
/// not even a prefix.
fn print_redactions(report: &RunReport) {
    if report.redactions.is_empty() {
        println!("(nothing was redacted)");
        return;
    }
    println!("Redacted before leaving this machine:");
    for line in report.redaction_lines() {
        println!("  {line}");
    }
}

#[cfg(test)]
mod tests {
    use super::openai_base_url;

    #[test]
    fn base_url_prefers_config_then_env_then_default() {
        assert_eq!(
            openai_base_url(Some("https://c/v1"), Some("https://e/v1")),
            "https://c/v1"
        );
        assert_eq!(openai_base_url(None, Some("https://e/v1")), "https://e/v1");
        assert_eq!(
            openai_base_url(None, Some("  ")),
            "https://api.openai.com/v1"
        );
        assert_eq!(openai_base_url(None, None), "https://api.openai.com/v1");
    }
}
