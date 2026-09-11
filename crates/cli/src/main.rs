mod args;
mod render;
mod repl;
mod terminal;

use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use airlok_core::config::{self, KeySource, Overrides, ProviderConfig, ProviderName, Sources};
use airlok_core::context::{self, ContextInput};
use airlok_core::redact::{Class, RedactionMap, Redactor, SecretRedactor};
use airlok_core::repl::{Backend, Repl, Switch};
use airlok_core::session::Summary;
use airlok_core::tools::{ToolRegistry, READ_ONLY_TOOLS};
use airlok_core::CoreError;
use airlok_core::{
    Agent, Config, Confirmation, Decision, Interrupt, Output, RunReport, SessionStore,
};
use airlok_llm::openai::Auth;
use airlok_llm::{Anthropic, OpenAi, Provider};
use anyhow::{anyhow, bail, Context};
use clap::Parser;
use tracing_subscriber::EnvFilter;

use args::{Args, Command, ConfigAction, SessionsAction};
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
        model: args.model.clone(),
        yes: args.yes,
    };
    let (mut config, sources) = Config::load(
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
        Some(Command::Sessions { action }) => return sessions_command(action, &store()?, &cwd),
        None => {}
    }
    let prompt = args.prompt.clone();
    if prompt.is_none() && !std::io::stdin().is_terminal() {
        bail!(
            "no prompt given and stdin is not a terminal. \
             Usage: airlok \"<task>\" for one task, or airlok in a terminal for a session"
        );
    }

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

    let store = store()?;
    let resumed = match args.resume() {
        Some(id) => Some(store.load(&cwd, id)?.ok_or_else(|| match id {
            Some(id) => anyhow!("no saved session {id} for this directory (see airlok sessions)"),
            None => anyhow!("no saved sessions for this directory"),
        })?),
        None => None,
    };
    if let Some(session) = &resumed {
        restore_session_model(&mut config, session, args.model.is_some());
    }

    // The key is read once, after every check that could abort the run.
    let key = config
        .resolve_key()
        .with_context(|| format!("cannot read the API key from {}", config.key_source()))?;
    let provider = build_provider(&config, key.clone());
    let mut redactor = SecretRedactor::new();
    if let Some(session) = &resumed {
        redactor = redactor.with_map(&session.redactions);
    }
    let redactor = redactor.with_known(KEY_LABEL, &key, Class::RedactOnly);
    let tools = ToolRegistry::defaults(&cwd, config.agent.bash_timeout);

    let mut agent =
        Agent::new(provider, tools, Box::new(redactor), config).with_context(context.text);
    let mut session = match resumed {
        Some(mut session) => {
            session.resume();
            eprintln!(
                "resumed session {} ({} turns, last updated {})",
                session.id,
                session.turns(),
                session.updated_at
            );
            session
        }
        None => agent.new_session(),
    };
    let mut out = Stdout::new(terminal, args.verbose);
    let Some(prompt) = prompt else {
        return run_repl(agent, session, &store, key, user_instructions, &mut out).await;
    };
    let result = agent.turn(&mut session, &prompt, &mut out).await;
    out.finish();
    if let Err(e) = store.save(&session) {
        eprintln!("warning: could not save the session: {e}");
    }
    let turns = match result {
        Ok(turns) => turns,
        Err(CoreError::Aborted) => {
            eprintln!("aborted");
            std::process::exit(1);
        }
        Err(e) => {
            if let Some(hint) = e.hint(&agent.config().provider.model) {
                eprintln!("Error: {e}\n{hint}");
                std::process::exit(1);
            }
            return Err(e.into());
        }
    };

    if args.show_redactions {
        print_redactions(&RunReport {
            redactions: session.redactions.clone(),
            turns,
        });
    }
    Ok(())
}

/// The interactive session. Ctrl-C during a turn cancels it; at the
/// prompt, rustyline reports it and the loop stays up.
async fn run_repl(
    mut agent: Agent,
    session: airlok_core::Session,
    store: &SessionStore,
    key: String,
    user_instructions: Option<PathBuf>,
    out: &mut Stdout,
) -> anyhow::Result<()> {
    let interrupt = Interrupt::new();
    {
        let interrupt = interrupt.clone();
        let mut sigint = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())
            .context("cannot listen for Ctrl-C")?;
        tokio::spawn(async move {
            while sigint.recv().await.is_some() {
                interrupt.trigger();
            }
        });
    }
    let mut lines = repl::Readline::new().context("cannot start line editing")?;
    out.status(&format!(
        "airlok {} · {} · /help for commands, Ctrl-D to quit",
        env!("CARGO_PKG_VERSION"),
        agent.config().provider.model
    ));
    let backend = CliBackend {
        startup: agent.config().provider.clone(),
        config: agent.config().clone(),
        key,
        user_instructions,
    };
    let mut repl = Repl {
        agent: &mut agent,
        store: Some(store),
        interrupt,
        backend: Box::new(backend),
    };
    let session = repl.run(session, &mut lines, out).await;
    out.finish();
    out.status(&format!(
        "session {} saved ({} turns); airlok --resume continues it",
        session.id,
        session.turns()
    ));
    Ok(())
}

const KEY_LABEL: &str = "the provider API key";

/// Continues a resumed session on the model it last used, when it used
/// the configured provider and `--model` was not given.
fn restore_session_model(config: &mut Config, session: &airlok_core::Session, model_flag: bool) {
    if session.provider != config.provider.name.as_str() {
        eprintln!(
            "note: the session last used {} ({}); continuing with {} ({}). /provider switches.",
            session.provider,
            session.model,
            config.provider.name.as_str(),
            config.provider.model
        );
    } else if !model_flag && session.model != config.provider.model {
        eprintln!(
            "continuing on {}, the model this session last used",
            session.model
        );
        config.provider.model = session.model.clone();
    }
}

/// Builds providers and redactors for the REPL. `[provider]` keys and
/// `base_url` belong to the configured provider; another provider uses its
/// default model and key environment variables, and switching back
/// restores what the session started with.
struct CliBackend {
    startup: ProviderConfig,
    config: Config,
    key: String,
    /// `~/.config/airlok/AIRLOK.md`, for rebuilding the context block.
    user_instructions: Option<PathBuf>,
}

impl Backend for CliBackend {
    fn fresh_redactor(&mut self) -> Box<dyn Redactor> {
        Box::new(SecretRedactor::new().with_known(KEY_LABEL, &self.key, Class::RedactOnly))
    }

    fn switch(&mut self, name: ProviderName, seed: &RedactionMap) -> Result<Switch, String> {
        let mut config = self.config.clone();
        config.provider = if name == self.startup.name {
            self.startup.clone()
        } else {
            ProviderConfig {
                name,
                model: name.default_model().to_string(),
                base_url: None,
                api_key_env: None,
                api_key_cmd: None,
                context_window: self.config.provider.context_window,
            }
        };
        // Errors name the variable or command; they never carry its output.
        let key = config.resolve_key().map_err(|e| e.to_string())?;
        let provider = build_provider(&config, key.clone());
        let redactor =
            SecretRedactor::new()
                .with_map(seed)
                .with_known(KEY_LABEL, &key, Class::RedactOnly);
        self.config = config;
        self.key = key;
        Ok(Switch {
            config: self.config.provider.clone(),
            provider,
            redactor: Box::new(redactor),
        })
    }

    fn context(&mut self) -> Option<String> {
        Some(
            context::build(&ContextInput {
                cwd: &self.config.cwd,
                user_instructions: self.user_instructions.as_deref(),
                max_bytes: self.config.context.max_bytes,
            })
            .text,
        )
    }
}

fn store() -> anyhow::Result<SessionStore> {
    SessionStore::default_root()
        .map(SessionStore::new)
        .ok_or_else(|| anyhow!("cannot locate the data directory: set XDG_DATA_HOME or HOME"))
}

fn sessions_command(
    action: Option<SessionsAction>,
    store: &SessionStore,
    cwd: &Path,
) -> anyhow::Result<()> {
    match action {
        None => {
            let sessions = store.list(cwd)?;
            if sessions.is_empty() {
                println!("no saved sessions for {}", cwd.display());
                return Ok(());
            }
            println!(
                "{:<10} {:<20} {:>5}  {:<16} first prompt",
                "id", "updated", "turns", "model"
            );
            for s in sessions {
                println!("{}", summary_line(&s));
            }
        }
        Some(SessionsAction::Rm { id }) => {
            if store.remove(cwd, &id)? {
                println!("deleted session {id}");
            } else {
                bail!("no saved session {id} for this directory");
            }
        }
        Some(SessionsAction::Clean { older_than }) => {
            let age = parse_age(&older_than)?;
            let removed = store.clean(age)?;
            println!(
                "deleted {} session(s) older than {older_than}",
                removed.len()
            );
        }
    }
    Ok(())
}

fn summary_line(s: &Summary) -> String {
    let mut prompt = s.first_prompt.clone().unwrap_or_default();
    prompt = prompt.lines().next().unwrap_or_default().to_string();
    if prompt.chars().count() > 50 {
        prompt = prompt.chars().take(47).collect::<String>() + "...";
    }
    let model: String = s.model.chars().take(16).collect();
    format!(
        "{:<10} {:<20} {:>5}  {:<16} {prompt}",
        s.id,
        s.updated_at.replace('T', " "),
        s.turns,
        model
    )
}

/// `30d`, `12h`, `90m`, or `45s`.
fn parse_age(text: &str) -> anyhow::Result<std::time::Duration> {
    let text = text.trim();
    let (number, unit) = text.split_at(text.len().saturating_sub(1));
    let count: u64 = number.parse().map_err(|_| {
        anyhow!("cannot parse age {text:?}: expected a number followed by d, h, m, or s")
    })?;
    let seconds = match unit {
        "d" => count * 86_400,
        "h" => count * 3_600,
        "m" => count * 60,
        "s" => count,
        _ => bail!("cannot parse age {text:?}: expected a number followed by d, h, m, or s"),
    };
    Ok(std::time::Duration::from_secs(seconds))
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
    fn status(&mut self, line: &str) {
        let rest = self.renderer.finish();
        self.write(&rest);
        self.end_line();
        let text = if self.renderer.is_rich() {
            format!("\x1b[2m{line}\x1b[0m\n")
        } else {
            format!("{line}\n")
        };
        self.write(&text);
    }

    fn end_turn(&mut self) {
        self.finish();
    }

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
    use super::{openai_base_url, parse_age};

    #[test]
    fn ages_parse_in_days_hours_minutes_seconds() {
        assert_eq!(parse_age("30d").unwrap().as_secs(), 30 * 86_400);
        assert_eq!(parse_age("12h").unwrap().as_secs(), 12 * 3_600);
        assert_eq!(parse_age("90m").unwrap().as_secs(), 90 * 60);
        assert_eq!(parse_age("5s").unwrap().as_secs(), 5);
        assert!(parse_age("30").is_err());
        assert!(parse_age("d").is_err());
    }

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
