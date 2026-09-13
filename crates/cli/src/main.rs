mod args;
mod complete;
mod diff;
mod keys;
mod output;
mod picker;
mod render;
mod repl;
mod status;
mod terminal;

use std::io::IsTerminal;
use std::os::fd::AsFd;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use airlok_core::config::{
    self, KeySource, McpScope, Overrides, ProviderConfig, ProviderName, Sources,
};
use airlok_core::context::{self, ContextInput};
use airlok_core::mcp::{json, project, trust};
use airlok_core::redact::{Class, RedactionMap, Redactor, SecretRedactor};
use airlok_core::repl::{Backend, Repl, Switch};
use airlok_core::session::Summary;
use airlok_core::tools::{truncate_output, Plan, Tool, ToolRegistry};
use airlok_core::CoreError;
use airlok_core::{Agent, Config, Decision, Interrupt, Output, RunReport, SessionStore};
use airlok_llm::openai::Auth;
use airlok_llm::{Anthropic, OpenAi, Provider};
use anyhow::{anyhow, bail, Context};
use clap::Parser;
use tracing_subscriber::EnvFilter;

use args::{
    Args, Command, ConfigAction, McpAction, ScopeArg, SessionsAction, TransportArg, TrustAction,
};
use keys::Keys;
use output::Stdout;
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
        Some(Command::Mcp { action }) => return mcp_command(action, &config, &sources).await,
        Some(Command::Sessions { action }) => return sessions_command(action, &store()?, &cwd),
        Some(Command::Doctor { offline }) => {
            let checks = doctor_checks(
                &config,
                &config_file_lines(&sources),
                &config.provider.model,
                offline,
            )
            .await;
            for check in &checks {
                println!("{}", check.line());
            }
            let failed = checks.iter().filter(|c| !c.ok).count();
            if failed > 0 {
                eprintln!("{failed} check(s) failed");
                std::process::exit(1);
            }
            return Ok(());
        }
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
        if prompt.is_some() {
            eprintln!(
                "warning: confirmations are off; writes and commands will run without asking"
            );
        }
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
    // Notes for the REPL's startup line; a one-shot run prints them on
    // stderr instead.
    let mut notes = Vec::new();
    // What the config files say, read before a resumed session can change
    // it: /effort reports whether the effort in force came from the config
    // or from the session.
    let file_efforts: std::collections::BTreeMap<String, String> = config
        .models
        .iter()
        .filter_map(|(id, m)| m.reasoning_effort.clone().map(|e| (id.clone(), e)))
        .collect();
    // Named here, while the config is still in scope, and never resolved.
    let key_source = config.key_source().to_string();
    if let Some(session) = &resumed {
        if session.provider != config.provider.name.as_str() {
            notes.push(format!("session last used {}", session.provider));
        }
        let note = restore_session_model(&mut config, session, args.model.is_some());
        if let (Some(note), Some(_)) = (note, &prompt) {
            eprintln!("{note}");
        }
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
        Agent::new(provider, tools, Box::new(redactor), config).with_context_block(&context);
    agent.set_plan_mode(args.plan);
    let mut session = match resumed {
        Some(mut session) => {
            session.resume();
            if prompt.is_some() {
                eprintln!(
                    "resumed session {} ({} turns, last updated {})",
                    session.id,
                    session.turns(),
                    session.updated_at
                );
            } else {
                notes.insert(
                    0,
                    format!("resumed {} ({} turns)", session.id, session.turns()),
                );
            }
            session
        }
        None => agent.new_session(),
    };
    if args.plan {
        notes.push("plan mode".into());
    }
    if !prompting {
        notes.push("confirmations off".into());
    }
    let mut out = Stdout::new(terminal, args.verbose);
    let Some(prompt) = prompt else {
        return run_repl(
            agent,
            session,
            &store,
            ReplSetup {
                key,
                user_instructions,
                notes,
                file_efforts,
                config_files: config_file_lines(&sources),
                key_source,
            },
            &mut out,
        )
        .await;
    };
    out.begin_turn();
    let result = agent.turn(&mut session, &prompt, &mut out).await;
    out.end_turn();
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
            // A one-shot run has no session override by definition.
            let model = agent.config().provider.model.clone();
            let from_file = agent
                .config()
                .models
                .get(&model)
                .and_then(|m| m.reasoning_effort.clone());
            if let Some(hint) = e.hint(&model, None, from_file.as_deref()) {
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

/// What a REPL run needs beyond the agent, its session and the store.
struct ReplSetup {
    key: String,
    user_instructions: Option<PathBuf>,
    notes: Vec<String>,
    /// `reasoning_effort` per model as the config files gave it, before a
    /// resumed session could override it. `/effort` reports which it is.
    file_efforts: std::collections::BTreeMap<String, String>,
    /// Which configuration files could apply and whether each was found,
    /// for `/status`. Paths only, never anything read from them.
    config_files: Vec<String>,
    /// Where the key comes from, named and not resolved.
    key_source: String,
}

/// The interactive session. Ctrl-C during a turn cancels it; at the
/// prompt, rustyline reports it and the loop stays up.
async fn run_repl(
    mut agent: Agent,
    session: airlok_core::Session,
    store: &SessionStore,
    setup: ReplSetup,
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
    // A resize mid-turn: the next block wraps to the new width.
    let mut sigwinch =
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::window_change())
            .context("cannot listen for window changes")?;
    tokio::spawn(async move {
        while sigwinch.recv().await.is_some() {
            render::measure();
        }
    });
    // SIGTERM mid-turn would otherwise leave the terminal in cbreak mode.
    let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .context("cannot listen for SIGTERM")?;
    tokio::spawn(async move {
        if sigterm.recv().await.is_some() {
            keys::restore();
            airlok_core::mcp::kill_children();
            std::process::exit(143);
        }
    });
    // Esc interrupts a turn; what else is typed during one starts the
    // next prompt. The keys come from stdin, which is the terminal in a
    // session: macOS cannot poll /dev/tty.
    let typed = Arc::new(Mutex::new(Vec::new()));
    match std::io::stdin().as_fd().try_clone_to_owned() {
        Ok(stdin) => out.watch_keys(Keys::new(
            std::fs::File::from(stdin),
            interrupt.clone(),
            typed.clone(),
        )),
        Err(e) => tracing::debug!(error = %e, "cannot watch the terminal for Esc"),
    }
    let mut lines = repl::Readline::new(agent.config().cwd.clone(), typed)
        .context("cannot start line editing")?;
    out.status(&startup_line(agent.config(), &setup.notes));
    let backend = CliBackend {
        startup: agent.config().provider.clone(),
        config: agent.config().clone(),
        key: setup.key,
        user_instructions: setup.user_instructions,
        session_models: None,
        file_efforts: setup.file_efforts,
        config_files: setup.config_files,
        key_source: setup.key_source,
    };
    let mut repl = Repl {
        agent: &mut agent,
        store: Some(store),
        interrupt,
        backend: Box::new(backend),
        used: Vec::new(),
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

/// The REPL's one line at startup: version, model, directory, anything
/// unusual about this start, and where the commands are.
fn startup_line(config: &Config, notes: &[String]) -> String {
    let dir = config
        .cwd
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| config.cwd.display().to_string());
    let mut parts = vec![
        format!("airlok {}", env!("CARGO_PKG_VERSION")),
        config.provider.model.clone(),
        dir,
    ];
    parts.extend(notes.iter().cloned());
    parts.push("/help for commands".into());
    parts.join(" · ")
}

/// Continues a resumed session on the model it last used, when it used
/// the configured provider and `--model` was not given. Returns what to
/// tell the user when the provider or the model differs from the config.
fn restore_session_model(
    config: &mut Config,
    session: &airlok_core::Session,
    model_flag: bool,
) -> Option<String> {
    // /effort recorded the session's own reasoning effort in its config,
    // and a resume keeps it: the session's value wins over the file for
    // the model it continues on.
    if let Some(effort) = session
        .config
        .models
        .get(&session.model)
        .and_then(|m| m.reasoning_effort.clone())
    {
        config
            .models
            .entry(session.model.clone())
            .or_default()
            .reasoning_effort = Some(effort);
    }
    if session.provider != config.provider.name.as_str() {
        Some(format!(
            "note: the session last used {} ({}); continuing with {} ({}). /provider switches.",
            session.provider,
            session.model,
            config.provider.name.as_str(),
            config.provider.model
        ))
    } else if !model_flag && session.model != config.provider.model {
        config.provider.model = session.model.clone();
        Some(format!(
            "continuing on {}, the model this session last used",
            session.model
        ))
    } else {
        None
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
    /// For `/status`: which configuration files could apply, and where the
    /// key comes from. Neither carries a value.
    config_files: Vec<String>,
    key_source: String,
    /// `~/.config/airlok/AIRLOK.md`, for rebuilding the context block.
    user_instructions: Option<PathBuf>,
    /// Model ids read from saved sessions, kept per provider because
    /// listing them parses every session file on disk.
    session_models: Option<(ProviderName, Vec<String>)>,
    /// `reasoning_effort` per model as the config files gave it, before
    /// any session override was restored.
    file_efforts: std::collections::BTreeMap<String, String>,
}

impl CliBackend {
    /// Model ids from saved sessions in this directory that ran on
    /// `provider`, newest first, since a model id means nothing on the
    /// other one.
    fn models_from_sessions(&self, provider: ProviderName) -> Vec<String> {
        let Ok(store) = store() else {
            return Vec::new();
        };
        let Ok(summaries) = store.list(&self.config.cwd) else {
            return Vec::new();
        };
        summaries
            .into_iter()
            .filter(|summary| summary.provider == provider.as_str())
            .map(|summary| summary.model)
            .collect()
    }
}

#[async_trait::async_trait]
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
                api: Default::default(),
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

    fn choose(&mut self, title: &str, choices: &[String]) -> Option<String> {
        picker::ask(title, choices)
    }

    fn startup_effort(&mut self, model: &str) -> Option<String> {
        self.file_efforts.get(model).cloned()
    }

    fn config_files(&mut self) -> Vec<String> {
        self.config_files.clone()
    }

    fn key_source(&mut self) -> Option<String> {
        Some(self.key_source.clone())
    }

    async fn doctor(&mut self, model: &str, offline: bool) -> Vec<airlok_core::repl::Check> {
        doctor_checks(&self.config, &self.config_files, model, offline).await
    }

    fn known_models(&mut self, provider: ProviderName) -> Vec<String> {
        // Once per provider: this runs before every prompt, and listing
        // sessions reads and parses all of them.
        if let Some((cached, models)) = &self.session_models {
            if cached.as_str() == provider.as_str() {
                return models.clone();
            }
        }
        let models = self.models_from_sessions(provider);
        self.session_models = Some((provider, models.clone()));
        models
    }

    fn context(&mut self) -> Option<airlok_core::context::ContextBlock> {
        Some(context::build(&ContextInput {
            cwd: &self.config.cwd,
            user_instructions: self.user_instructions.as_deref(),
            max_bytes: self.config.context.max_bytes,
        }))
    }
}

/// `airlok mcp`: what the servers offer, where each came from, and the
/// files that define them.
async fn mcp_command(
    action: Option<McpAction>,
    config: &Config,
    sources: &Sources,
) -> anyhow::Result<()> {
    for problem in &sources.mcp_problems {
        eprintln!("warning: {problem}");
    }
    let result = match action.unwrap_or(McpAction::List) {
        McpAction::List => {
            if config.mcp.is_empty() {
                println!("no MCP servers configured. `airlok mcp add` writes one, or add an [[mcp]] block to the config");
                return Ok(());
            }
            // Listing never approves anything: a project server that has
            // not been asked about is shown as pending and left alone.
            let choices = project::load(&config.cwd);
            for status in airlok_core::mcp::connect_all(&config.mcp, &config.cwd, &choices)
                .await
                .1
            {
                println!("{}", status.line());
            }
            Ok(())
        }
        McpAction::Add {
            name,
            scope,
            transport,
            url,
            env,
            header,
            command,
        } => mcp_add(
            config, &name, scope, transport, url, &env, &header, &command,
        ),
        McpAction::Remove { name, scope } => mcp_remove(config, &name, scope),
        McpAction::Get { name } => mcp_get(config, &name),
        McpAction::Import { path, scope } => mcp_import(config, &path, scope),
        McpAction::Export { scope } => mcp_export(config, scope),
        McpAction::AddJson { name, json, scope } => mcp_add_json(config, &name, &json, scope),
        McpAction::ResetProjectChoices => mcp_reset_project_choices(config),
        McpAction::Trust { action } => mcp_trust(config, action),
        McpAction::Call {
            server,
            tool,
            arguments,
        } => mcp_call(config, &server, &tool, &arguments).await,
    };
    // Nothing owns the clients past this point, so make sure no child
    // outlives the command.
    airlok_core::mcp::kill_children();
    result
}

/// Where one scope's JSON file is.
fn scope_file(scope: McpScope, cwd: &Path) -> anyhow::Result<PathBuf> {
    let dir = config::user_config_path().and_then(|path| path.parent().map(Path::to_path_buf));
    json::path_for(scope, dir.as_deref(), cwd)
        .ok_or_else(|| anyhow!("{} has no file of its own", scope.as_str()))
}

fn pairs(
    values: &[String],
    what: &str,
) -> anyhow::Result<std::collections::BTreeMap<String, String>> {
    values
        .iter()
        .map(|value| {
            value
                .split_once('=')
                .map(|(key, value)| (key.to_string(), value.to_string()))
                .ok_or_else(|| anyhow!("{what} must be written K=V, not {value:?}"))
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn mcp_add(
    config: &Config,
    name: &str,
    scope: ScopeArg,
    transport: Option<TransportArg>,
    url: Option<String>,
    env: &[String],
    header: &[String],
    command: &[String],
) -> anyhow::Result<()> {
    let scope: McpScope = scope.into();
    let http = matches!(transport, Some(TransportArg::Http)) || url.is_some();
    if http && url.is_none() {
        bail!("an http server needs --url");
    }
    if !http && command.is_empty() {
        bail!("a stdio server needs a command after --, for example: airlok mcp add files -- npx -y server-filesystem .");
    }
    let entry = json::Entry {
        kind: Some(if http { "http" } else { "stdio" }.to_string()),
        command: command.first().cloned(),
        args: command.iter().skip(1).cloned().collect(),
        env: pairs(env, "--env")?,
        url,
        headers: pairs(header, "--header")?,
        airlok: None,
    };
    let path = scope_file(scope, &config.cwd)?;
    let mut file = json::read(&path).map_err(|e| anyhow!(e))?;
    let replaced = file.servers.insert(name.to_string(), entry).is_some();
    json::write(&path, &file).map_err(|e| anyhow!(e))?;
    println!(
        "{} {name} in {}",
        if replaced { "replaced" } else { "added" },
        path.display()
    );
    Ok(())
}

fn mcp_remove(config: &Config, name: &str, scope: Option<ScopeArg>) -> anyhow::Result<()> {
    let scopes: Vec<McpScope> = match scope {
        Some(scope) => vec![scope.into()],
        None => vec![McpScope::User, McpScope::Project, McpScope::Local],
    };
    let mut removed = 0;
    for scope in scopes {
        let path = scope_file(scope, &config.cwd)?;
        let mut file = json::read(&path).map_err(|e| anyhow!(e))?;
        if file.servers.remove(name).is_some() {
            json::write(&path, &file).map_err(|e| anyhow!(e))?;
            println!("removed {name} from {}", path.display());
            removed += 1;
        }
    }
    if removed == 0 {
        bail!("no server called {name} in the JSON files; an [[mcp]] block is removed by editing the TOML");
    }
    Ok(())
}

fn mcp_get(config: &Config, name: &str) -> anyhow::Result<()> {
    let server = config
        .mcp
        .iter()
        .find(|server| server.name == name)
        .ok_or_else(|| anyhow!("no server called {name}; `airlok mcp list` shows them"))?;
    println!("{name} from {}", server.scope.as_str());
    if let Some(root) = server.root() {
        println!("serving {root}");
    }
    let entry = json::entry_of(server);
    println!("{}", serde_json::to_string_pretty(&entry)?);
    Ok(())
}

fn mcp_import(config: &Config, path: &Path, scope: ScopeArg) -> anyhow::Result<()> {
    let incoming = json::read(path).map_err(|e| anyhow!(e))?;
    if incoming.servers.is_empty() {
        bail!("{} has no mcpServers entries", path.display());
    }
    let into = scope_file(scope.into(), &config.cwd)?;
    let mut file = json::read(&into).map_err(|e| anyhow!(e))?;
    let mut names: Vec<String> = Vec::new();
    for (name, entry) in incoming.servers {
        names.push(name.clone());
        file.servers.insert(name, entry);
    }
    json::write(&into, &file).map_err(|e| anyhow!(e))?;
    println!("imported {} into {}", names.join(", "), into.display());
    Ok(())
}

fn mcp_export(config: &Config, scope: Option<ScopeArg>) -> anyhow::Result<()> {
    let file = match scope {
        Some(scope) => {
            json::read(&scope_file(scope.into(), &config.cwd)?).map_err(|e| anyhow!(e))?
        }
        None => json::McpJson {
            servers: config
                .mcp
                .iter()
                .map(|server| (server.name.clone(), json::entry_of(server)))
                .collect(),
        },
    };
    println!("{}", serde_json::to_string_pretty(&file)?);
    Ok(())
}

/// `mcp add-json <name> '<json>'`: the entry exactly as another tool
/// would write it, which is how a server is shared in a message or a
/// README without spelling out flags.
fn mcp_add_json(config: &Config, name: &str, text: &str, scope: ScopeArg) -> anyhow::Result<()> {
    let entry: json::Entry =
        serde_json::from_str(text).context("the entry must be a JSON object")?;
    if entry.command.is_none() && entry.url.is_none() {
        bail!("the entry needs a command or a url");
    }
    let path = scope_file(scope.into(), &config.cwd)?;
    let mut file = json::read(&path).map_err(|e| anyhow!(e))?;
    let replaced = file.servers.insert(name.to_string(), entry).is_some();
    json::write(&path, &file).map_err(|e| anyhow!(e))?;
    println!(
        "{} {name} in {}",
        if replaced { "replaced" } else { "added" },
        path.display()
    );
    Ok(())
}

fn mcp_reset_project_choices(config: &Config) -> anyhow::Result<()> {
    match project::reset(&config.cwd).map_err(|e| anyhow!(e))? {
        true => println!(
            "forgot what this repository answered about its own .mcp.json servers; the next run asks again"
        ),
        false => println!("this repository has not answered about its own .mcp.json servers"),
    }
    Ok(())
}

fn mcp_trust(config: &Config, action: TrustAction) -> anyhow::Result<()> {
    match action {
        TrustAction::List => {
            let lines = trust::load(&config.cwd).lines();
            if lines.is_empty() {
                println!("nothing is remembered here; `s` at a confirmation saves an approval");
            }
            lines.iter().for_each(|line| println!("{line}"));
        }
        TrustAction::Revoke { server } => {
            let mut store = trust::load(&config.cwd);
            let gone = store.revoke(&server);
            if gone == 0 {
                bail!("nothing was remembered for {server}");
            }
            trust::save(&config.cwd, &store).map_err(|e| anyhow!(e))?;
            println!("forgot {gone} approval(s) for {server}");
        }
    }
    Ok(())
}

async fn mcp_call(
    config: &Config,
    server: &str,
    tool: &str,
    arguments: &str,
) -> anyhow::Result<()> {
    let configured = config
        .mcp
        .iter()
        .find(|candidate| candidate.name == server)
        .ok_or_else(|| anyhow!("no MCP server called {server} in the config"))?;
    let arguments: serde_json::Value =
        serde_json::from_str(arguments).context("the arguments must be a JSON object")?;
    if !project::allowed(configured, &project::load(&config.cwd)) {
        bail!(
            "{server} comes from this project's .mcp.json and has not been approved here. \
             Start a session to be asked, or run `airlok mcp reset-project-choices` to be asked again"
        );
    }
    let connection = airlok_core::mcp::connect(configured, &config.cwd)
        .await
        .map_err(|why| anyhow!("{server}: {why}"))?;
    let called = connection
        .tools
        .into_iter()
        .find(|candidate| candidate.tool() == tool)
        .ok_or_else(|| anyhow!("{server} offers no tool called {tool}"))?;

    match called.plan(&arguments).await? {
        Plan::Denied { why } => bail!("{why}"),
        Plan::McpCall {
            tool,
            arguments,
            root,
            paths,
            ..
        } if config.safety.confirm_mcp => {
            let mut terminal = Terminal::open().context(
                "a confirmation is needed but no terminal is available. \
                 Set trust = allow for this server, or pass --yes",
            )?;
            terminal.show_lines(&output::mcp_lines(
                server,
                &tool,
                &arguments,
                root.as_deref(),
                &paths,
            ));
            if terminal.ask(&format!("Call `{tool}` on `{server}`?")) != Decision::Approve {
                bail!("not called");
            }
        }
        _ => {}
    }
    println!("{}", truncate_output(called.execute(arguments).await?));
    Ok(())
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
/// Every `/doctor` check. Talking to the provider and the MCP servers
/// happens here, where the key and the transports live; nothing it reads
/// is ever put in a `Check`, only where it looked.
async fn doctor_checks(
    config: &Config,
    files: &[String],
    model: &str,
    offline: bool,
) -> Vec<airlok_core::repl::Check> {
    use airlok_core::repl::Check;
    use futures::StreamExt;

    let mut checks = Vec::new();
    checks.push(Check::pass(
        "config",
        if files.is_empty() {
            "built-in defaults only".to_string()
        } else {
            files.join("; ")
        },
    ));

    let key = config.resolve_key();
    match &key {
        Ok(_) => checks.push(Check::pass(
            "provider key",
            format!("resolved from {}", config.key_source()),
        )),
        Err(e) => checks.push(Check::fail(
            "provider key",
            format!("{e}, looking at {}", config.key_source()),
        )),
    }

    match key {
        // The one check that leaves the machine. Everything else reads
        // files or talks to local processes.
        _ if offline => checks.push(Check::pass(
            "provider request",
            "skipped: --offline".to_string(),
        )),
        Err(_) => checks.push(Check::fail(
            "provider request",
            "not attempted: no key to send".to_string(),
        )),
        Ok(key) => {
            let provider = build_provider(config, key);
            let request = airlok_llm::Request {
                model: model.to_string(),
                max_tokens: 16,
                system: String::new(),
                messages: vec![airlok_llm::Message::user_text("ping")],
                tools: Vec::new(),
                reasoning_effort: config
                    .models
                    .get(model)
                    .and_then(|m| m.reasoning_effort.clone()),
                api: config
                    .models
                    .get(model)
                    .and_then(|m| m.api)
                    .unwrap_or(config.provider.api),
            };
            let mut stream = provider.stream(request);
            let mut failure = None;
            while let Some(event) = stream.next().await {
                if let Err(e) = event {
                    failure = Some(e.to_string());
                    break;
                }
            }
            checks.push(match failure {
                None => Check::pass(
                    "provider request",
                    format!("{} answered for {model}", config.provider.name.as_str()),
                ),
                Some(why) => Check::fail("provider request", why),
            });
        }
    }

    if config.mcp.is_empty() {
        checks.push(Check::pass("mcp servers", "none configured".to_string()));
    } else {
        let choices = airlok_core::mcp::project::load(&config.cwd);
        let (_, statuses) = airlok_core::mcp::connect_all(&config.mcp, &config.cwd, &choices).await;
        for status in statuses {
            let ok = matches!(status.state, airlok_core::mcp::State::Ready { .. });
            let name = format!("mcp {}", status.server);
            checks.push(if ok {
                Check::pass(&name, status.line())
            } else {
                Check::fail(&name, status.line())
            });
        }
    }

    checks.push(match airlok_core::context::branch(&config.cwd) {
        Some(branch) => Check::pass("git", format!("on {branch}")),
        None => Check::fail(
            "git",
            "not a git repository, or git is not installed".to_string(),
        ),
    });

    let terminal = std::io::IsTerminal::is_terminal(&std::io::stdout());
    checks.push(Check::pass(
        "terminal",
        if terminal {
            format!(
                "a terminal, {} columns, colour {}",
                {
                    crate::render::measure();
                    crate::render::terminal_columns().load(std::sync::atomic::Ordering::Relaxed)
                },
                if std::env::var_os("NO_COLOR").is_some() {
                    "off (NO_COLOR)"
                } else {
                    "on"
                }
            )
        } else {
            "not a terminal: plain output, no confirmations".to_string()
        },
    ));

    checks.push(match SessionStore::default_root() {
        Some(root) => match std::fs::create_dir_all(&root) {
            Ok(()) => Check::pass("session storage", format!("writable at {}", root.display())),
            Err(e) => Check::fail("session storage", format!("{}: {e}", root.display())),
        },
        None => Check::fail(
            "session storage",
            "cannot locate the data directory: set XDG_DATA_HOME or HOME".to_string(),
        ),
    });

    let trust = airlok_core::mcp::trust::path_for(&config.cwd);
    let trust_dir = trust.parent().unwrap_or(&config.cwd).to_path_buf();
    checks.push(match std::fs::create_dir_all(&trust_dir) {
        Ok(()) => Check::pass(
            "mcp trust storage",
            format!("writable at {}", trust_dir.display()),
        ),
        Err(e) => Check::fail("mcp trust storage", format!("{}: {e}", trust_dir.display())),
    });

    checks
}

/// One line per configuration file that could apply, in the same words
/// `airlok config show` uses. Paths and whether they were found, nothing
/// read from inside them.
fn config_file_lines(sources: &Sources) -> Vec<String> {
    let describe = |label: &str, layer: &Option<config::Layer>| {
        layer.as_ref().map(|layer| {
            format!(
                "{label}: {} ({})",
                layer.path.display(),
                if layer.found { "found" } else { "missing" }
            )
        })
    };
    let mut lines: Vec<String> = [
        describe("user", &sources.user),
        describe("project", &sources.project),
    ]
    .into_iter()
    .flatten()
    .collect();
    for problem in &sources.mcp_problems {
        lines.push(format!("mcp problem: {problem}"));
    }
    lines
}

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
    use super::{openai_base_url, parse_age, restore_session_model};

    /// /effort records the effort in the session, and a resume continues
    /// on it rather than on whatever the config file now says.
    #[test]
    fn a_resume_keeps_the_reasoning_effort_the_session_used() {
        use airlok_core::config::{Config, ModelConfig, ModelSection};

        let cwd = std::path::PathBuf::from("/tmp/airlok-resume-effort");
        let mut config = Config::new(cwd.clone());
        config.provider.model = "gpt-6-astra".into();
        // The config files say "low" for this model; the session says
        // "none", and the session is what the resume must continue on.
        config.models.insert(
            "gpt-6-astra".into(),
            ModelConfig {
                reasoning_effort: Some("low".into()),
                api: None,
            },
        );
        let mut session = airlok_core::Session::new(&cwd, &config);
        session.model = "gpt-6-astra".into();
        session.config.models.insert(
            "gpt-6-astra".into(),
            ModelSection {
                reasoning_effort: Some("none".into()),
                api: None,
            },
        );

        restore_session_model(&mut config, &session, false);

        assert_eq!(
            config
                .models
                .get("gpt-6-astra")
                .and_then(|m| m.reasoning_effort.as_deref()),
            Some("none"),
            "the session's own effort won over the config file"
        );
    }

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
