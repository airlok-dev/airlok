//! The interactive loop: read a line, then run a turn, a slash command, a
//! `!` command, or a `#` note, and save. Input arrives through
//! [`LineSource`] so the binary can use readline and tests can feed a
//! script.

use std::io::Write as _;
use std::path::Path;
use std::sync::Arc;

use airlok_llm::{Message, Provider};
use serde_json::json;

use crate::agent::{fmt_tokens, redaction_lines, Agent};
use crate::config::{ProviderConfig, ProviderName};
use crate::context::INSTRUCTION_FILES;
use crate::redact::{RedactionMap, Redactor};
use crate::safety::{Confirmation, Decision};
use crate::session::{Session, SessionStore};
use crate::tools::{truncate_output, Bash, Tool};
use crate::{CoreError, Interrupt, Output};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Line {
    Text(String),
    /// Ctrl-C at the prompt.
    Interrupt,
    /// Ctrl-D, or the input ran out.
    Eof,
}

/// What the argument to a slash command could be, by command name
/// without the slash. Refreshed before every prompt, since the answer
/// depends on the session.
pub type Candidates = std::collections::BTreeMap<&'static str, Vec<String>>;

pub trait LineSource {
    fn read_line(&mut self, prompt: &str) -> Line;

    /// Offered by Tab after a command that takes an argument. A front end
    /// that does not complete can ignore them.
    fn set_candidates(&mut self, _candidates: Candidates) {}
}

/// One `/doctor` check: what was tried, whether it worked, and a line
/// saying where it looked. Never carries a value it read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Check {
    pub name: String,
    pub ok: bool,
    pub detail: String,
}

impl Check {
    pub fn pass(name: &str, detail: impl Into<String>) -> Self {
        Self {
            name: name.to_string(),
            ok: true,
            detail: detail.into(),
        }
    }

    pub fn fail(name: &str, detail: impl Into<String>) -> Self {
        Self {
            name: name.to_string(),
            ok: false,
            detail: detail.into(),
        }
    }

    pub fn line(&self) -> String {
        format!(
            "{} {:<22} {}",
            if self.ok { "pass" } else { "FAIL" },
            self.name,
            self.detail
        )
    }
}

/// What the REPL needs from the binary, which owns key resolution and
/// provider construction.
#[async_trait::async_trait]
pub trait Backend: Send {
    /// A redactor for a fresh session, knowing the current provider key.
    fn fresh_redactor(&mut self) -> Box<dyn Redactor>;

    /// Builds the provider `name` from the configuration. `seed` is the
    /// session's redaction map: the returned redactor keeps its
    /// placeholders and also knows the new provider's key. The error says
    /// what is missing, such as the environment variable for the key.
    fn switch(&mut self, name: ProviderName, seed: &RedactionMap) -> Result<Switch, String>;

    /// The context block rebuilt from disk, after `#` changed AIRLOK.md.
    /// `None` keeps the current block.
    fn context(&mut self) -> Option<crate::context::ContextBlock> {
        None
    }

    /// Asks the user to pick one of `choices`. `None` means cancelled, or
    /// that this front end cannot ask, in which case the caller falls back
    /// to printing what it would have offered.
    fn choose(&mut self, _title: &str, _choices: &[String]) -> Option<String> {
        None
    }

    /// Model ids seen in saved sessions for `provider`, newest first.
    fn known_models(&mut self, _provider: ProviderName) -> Vec<String> {
        Vec::new()
    }

    /// The reasoning effort the config files give `model`, before any
    /// session override. `None` when the files say nothing about it.
    fn startup_effort(&mut self, _model: &str) -> Option<String> {
        None
    }

    /// One line per configuration file that could apply, saying whether it
    /// was found. Never carries a key or any value read from one.
    fn config_files(&mut self) -> Vec<String> {
        Vec::new()
    }

    /// Where the provider key comes from, named but never resolved.
    fn key_source(&mut self) -> Option<String> {
        None
    }

    /// Runs the `/doctor` checks against `model`, which is the model in
    /// use now rather than the one the run started on. Talking to the
    /// provider and to the MCP servers is the binary's job.
    async fn doctor(&mut self, model: &str, offline: bool) -> Vec<Check> {
        let _ = (model, offline);
        Vec::new()
    }
}

/// A provider ready to take over the session.
pub struct Switch {
    pub config: ProviderConfig,
    pub provider: Arc<dyn Provider>,
    pub redactor: Box<dyn Redactor>,
}

/// Slash commands in menu order. A prefix runs the first command it
/// matches, so the harmless ones come before those that change things.
pub const COMMANDS: &[(&str, &str)] = &[
    ("/help", "list the commands"),
    ("/model", "show the model, or switch with /model <id>"),
    (
        "/effort",
        "show the reasoning effort, or set it with /effort <value>",
    ),
    (
        "/mcp",
        "list the MCP servers and their tools, or enable one with /mcp <name>",
    ),
    (
        "/plan",
        "plan mode: read-only tools, the reply is a plan; /plan again leaves",
    ),
    ("/go", "carry out the plan and leave plan mode"),
    (
        "/provider",
        "show the provider, or switch with /provider anthropic|openai",
    ),
    ("/cost", "tokens used so far"),
    ("/compact", "summarise older turns to free context"),
    ("/config", "the effective configuration"),
    (
        "/redactions",
        "what was redacted before leaving this machine",
    ),
    (
        "/status",
        "version, provider, session, config files, MCP servers, redaction counts",
    ),
    (
        "/context",
        "where the context window is going, and how near compaction is",
    ),
    (
        "/btw",
        "ask a side question; the answer is printed and stays out of the task",
    ),
    (
        "/permissions",
        "what airlok asks about; set one with /permissions <name> <value>, or save",
    ),
    (
        "/init",
        "propose an AIRLOK.md for this repository, as a diff to approve",
    ),
    (
        "/copy",
        "copy the last reply to the clipboard; /copy 2 for the one before, /copy code",
    ),
    (
        "/diff",
        "what airlok changed on disk this session; /diff <path> for one file",
    ),
    (
        "/doctor",
        "check config, key, provider, MCP servers, git, terminal and storage",
    ),
    (
        "/goal",
        "show the session goal, set it with /goal <statement>, or /goal clear",
    ),
    ("/clear", "start a new session; the current one stays saved"),
    ("/exit", "save and quit; Ctrl-D does the same"),
];

/// What the name typed after `/` resolves to.
#[derive(Debug, PartialEq, Eq)]
pub enum Resolved {
    /// A command name without the slash, such as `model`.
    Command(&'static str),
    /// Nothing matches. `closest` is the nearest command, with its slash.
    Unknown { closest: &'static str },
}

/// An exact name, else the first command in [`COMMANDS`] order that starts
/// with `name`. `quit` is an exact-only alias of `exit`.
pub fn resolve_command(name: &str) -> Resolved {
    if name == "quit" {
        return Resolved::Command("exit");
    }
    let names = || COMMANDS.iter().map(|&(command, _)| &command[1..]);
    if let Some(found) = names()
        .find(|n| *n == name)
        .or_else(|| names().find(|n| n.starts_with(name)))
    {
        return Resolved::Command(found);
    }
    let closest = COMMANDS
        .iter()
        .map(|&(command, _)| command)
        .min_by_key(|command| edit_distance(name, &command[1..]))
        .unwrap_or("/help");
    Resolved::Unknown { closest }
}

/// How many ids to print when there is no terminal to pick with.
const MENU_CHOICES: usize = 10;

/// The body of the last fenced code block in `text`, without its fences.
fn fenced_block(text: &str) -> Option<String> {
    let mut blocks = Vec::new();
    let mut current: Option<Vec<&str>> = None;
    for line in text.lines() {
        if line.trim_start().starts_with("```") {
            match current.take() {
                Some(body) => blocks.push(body.join("\n")),
                None => current = Some(Vec::new()),
            }
            continue;
        }
        if let Some(body) = current.as_mut() {
            body.push(line);
        }
    }
    blocks.pop()
}

/// Hands `text` to whichever clipboard program this system has. Says so
/// plainly when it has none, rather than looking like it worked.
fn to_clipboard(text: &str) -> Result<&'static str, String> {
    to_clipboard_with(CLIPBOARD_PROGRAMS, text)
}

/// The clipboard programs tried, in order, on the systems airlok runs on.
const CLIPBOARD_PROGRAMS: &[(&str, &[&str])] = &[
    ("pbcopy", &[]),
    ("wl-copy", &[]),
    ("xclip", &["-selection", "clipboard"]),
    ("xsel", &["--clipboard", "--input"]),
];

fn to_clipboard_with(
    programs: &[(&'static str, &'static [&'static str])],
    text: &str,
) -> Result<&'static str, String> {
    let mut tried = Vec::new();
    for (program, args) in programs {
        let mut child = match std::process::Command::new(program)
            .args(*args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
        {
            Ok(child) => child,
            Err(_) => {
                tried.push(*program);
                continue;
            }
        };
        let wrote = child
            .stdin
            .as_mut()
            .ok_or_else(|| format!("{program} took no input"))
            .and_then(|stdin| stdin.write_all(text.as_bytes()).map_err(|e| e.to_string()));
        if let Err(e) = wrote {
            return Err(format!("{program} failed: {e}"));
        }
        return match child.wait() {
            Ok(status) if status.success() => Ok(program),
            Ok(status) => Err(format!("{program} exited with {status}")),
            Err(e) => Err(format!("{program} failed: {e}")),
        };
    }
    Err(format!(
        "no clipboard program found; tried {}",
        tried.join(", ")
    ))
}

/// The settings `/permissions` can change, for Tab and for the error.
pub const PERMISSION_NAMES: &[&str] = &[
    "confirm_writes",
    "confirm_bash",
    "confirm_mcp",
    "allow",
    "unallow",
    "deny",
    "undeny",
];

/// `part` as a percentage of `whole`, and 0 when there is no whole.
fn share(part: u64, whole: u64) -> u64 {
    (part * 100).checked_div(whole).unwrap_or(0)
}

/// A twenty-cell bar for a percentage.
fn bar(percent: u64) -> String {
    const CELLS: u64 = 20;
    let filled = (percent * CELLS / 100).min(CELLS) as usize;
    format!(
        "{}{}",
        "\u{2588}".repeat(filled),
        "\u{2591}".repeat(CELLS as usize - filled)
    )
}

/// The first line of `text`, cut to `width` columns with an ellipsis.
fn first_line(text: &str, width: usize) -> String {
    let line = text.lines().next().unwrap_or_default().trim();
    if line.chars().count() <= width {
        return line.to_string();
    }
    let kept: String = line.chars().take(width.saturating_sub(1)).collect();
    format!("{kept}\u{2026}")
}

/// The known ids closest to `typed`, nearest first, and only ones close
/// enough to be worth naming.
pub fn nearest(typed: &str, known: &[String]) -> Vec<String> {
    let mut scored: Vec<(usize, &String)> = known
        .iter()
        .map(|id| (edit_distance(typed, id), id))
        .filter(|(distance, id)| *distance <= id.len().max(typed.len()) / 2)
        .collect();
    scored.sort_by_key(|(distance, id)| (*distance, (*id).clone()));
    scored
        .into_iter()
        .take(3)
        .map(|(_, id)| id.clone())
        .collect()
}

/// What the id is offered as, with the typed one first so Enter on it is
/// a deliberate choice rather than a fuzzy match being taken for one.
fn once_then(typed: &str, near: &[String], known: &[String]) -> Vec<String> {
    let mut choices = vec![typed.to_string()];
    choices.extend(near.iter().cloned());
    choices.extend(known.iter().cloned());
    let mut seen = std::collections::HashSet::new();
    choices.retain(|id| seen.insert(id.clone()));
    choices
}

/// Levenshtein distance, for suggesting the closest command.
fn edit_distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut diagonal = row[0];
        row[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let above = row[j + 1];
            row[j + 1] = (diagonal + usize::from(ca != *cb))
                .min(row[j] + 1)
                .min(above + 1);
            diagonal = above;
        }
    }
    row[b.len()]
}

pub struct Repl<'a> {
    pub agent: &'a mut Agent,
    /// Where sessions are saved after every turn. `None` keeps them in
    /// memory only.
    pub store: Option<&'a SessionStore>,
    /// Fired by Ctrl-C during a turn.
    pub interrupt: Interrupt,
    /// Builds redactors for `/clear` and providers for `/provider`.
    pub backend: Box<dyn Backend + 'a>,
    /// Model ids used in this run, most recent first, for the picker.
    pub used: Vec<String>,
}

#[derive(PartialEq)]
enum Flow {
    Continue,
    Exit,
}

impl Repl<'_> {
    /// Runs until `/exit` or end of input. Returns the session as last
    /// saved.
    pub async fn run(
        &mut self,
        mut session: Session,
        lines: &mut dyn LineSource,
        out: &mut dyn Output,
    ) -> Session {
        loop {
            let prompt = self.prompt_text();
            lines.set_candidates(self.candidates(&session));
            match lines.read_line(&prompt) {
                Line::Eof => break,
                // Ctrl-C at the prompt does nothing; Ctrl-D or /exit quits.
                Line::Interrupt => continue,
                Line::Text(line) => {
                    let line = line.trim();
                    if let Some(command) = line.strip_prefix('/') {
                        if self.command(command, &mut session, out).await == Flow::Exit {
                            break;
                        }
                    } else if let Some(command) = line.strip_prefix('!') {
                        self.run_local(command.trim(), &mut session, out).await;
                    } else if let Some(note) = line.strip_prefix('#') {
                        self.remember(note.trim(), out);
                    } else if !line.is_empty() {
                        self.turn(&mut session, line, out).await;
                    }
                }
            }
        }
        self.save(&session, out);
        session
    }

    /// What Tab offers after each command that takes an argument, and
    /// what the pickers list.
    fn candidates(&mut self, session: &Session) -> Candidates {
        let mut candidates = Candidates::new();
        candidates.insert("model", self.model_choices(session));
        candidates.insert(
            "provider",
            vec![
                ProviderName::Anthropic.as_str().to_string(),
                ProviderName::OpenAi.as_str().to_string(),
            ],
        );
        candidates.insert("effort", self.effort_choices());
        candidates.insert("goal", vec!["clear".to_string()]);
        candidates.insert("copy", vec!["code".to_string()]);
        candidates.insert(
            "diff",
            session.writes.iter().map(|w| w.path.clone()).collect(),
        );
        candidates.insert(
            "permissions",
            PERMISSION_NAMES
                .iter()
                .map(|name| (*name).to_string())
                .chain(std::iter::once("save".to_string()))
                .collect(),
        );
        candidates
    }

    /// Every model id airlok knows here: the one in use, any named in
    /// `[models."<id>"]`, the ones used earlier in this run, and the ones
    /// saved sessions used on this provider. In that order, without
    /// repeats, because the first is the likeliest.
    fn model_choices(&mut self, session: &Session) -> Vec<String> {
        let config = self.agent.config();
        let provider = config.provider.name;
        let mut ids = vec![config.provider.model.clone()];
        ids.extend(self.used.iter().cloned());
        ids.extend(config.models.keys().cloned());
        ids.push(session.model.clone());
        ids.extend(self.backend.known_models(provider));
        let mut seen = std::collections::HashSet::new();
        ids.retain(|id| !id.trim().is_empty() && seen.insert(id.clone()));
        ids
    }

    /// `<model> <cwd basename>> `, with `[plan]` before the `>` in plan mode.
    pub fn prompt_text(&self) -> String {
        let config = self.agent.config();
        let dir = config
            .cwd
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| config.cwd.display().to_string());
        let plan = if self.agent.plan_mode() {
            " [plan]"
        } else {
            ""
        };
        format!("{} {dir}{plan}> ", config.provider.model)
    }

    async fn turn(&mut self, session: &mut Session, line: &str, out: &mut dyn Output) {
        out.begin_turn();
        match self
            .agent
            .turn_with(session, line, out, &self.interrupt)
            .await
        {
            Ok(_) => {}
            Err(CoreError::Aborted) => out.status("aborted"),
            Err(e) => {
                let config = self.agent.config();
                // The raw body is long and usually JSON; it goes to the
                // debug log, and the user gets a sentence.
                tracing::debug!(error = %e, "turn failed");
                match e.explain(&config.provider.model, config.provider.name.as_str()) {
                    Some(line) => out.status(&line),
                    None => out.status(&format!("error: {e}")),
                }
                let model = self.agent.config().provider.model.clone();
                let in_force = self
                    .agent
                    .config()
                    .models
                    .get(&model)
                    .and_then(|m| m.reasoning_effort.clone());
                let from_file = self.backend.startup_effort(&model);
                if let Some(hint) = e.hint(&model, in_force.as_deref(), from_file.as_deref()) {
                    hint.lines().for_each(|l| out.status(l));
                }
            }
        }
        out.end_turn();
        out.status(&self.footer(session));
        self.save(session, out);
    }

    /// Shown after each turn: the model, how full the context is, and the
    /// session id.
    fn footer(&self, session: &Session) -> String {
        let config = self.agent.config();
        let percent = (session.usage.context_tokens * 100)
            .checked_div(config.provider.context_window)
            .unwrap_or(0);
        let goal = session
            .goal
            .as_deref()
            .map(str::trim)
            .filter(|g| !g.is_empty())
            .map(|g| format!(" · goal: {}", first_line(g, 40)))
            .unwrap_or_default();
        format!(
            "{} · context {percent}% · session {}{goal}",
            config.provider.model, session.id
        )
    }

    async fn command(
        &mut self,
        command: &str,
        session: &mut Session,
        out: &mut dyn Output,
    ) -> Flow {
        let (typed, arg) = match command.trim().split_once(char::is_whitespace) {
            Some((name, arg)) => (name, arg.trim()),
            None => (command.trim(), ""),
        };
        let name = match resolve_command(typed) {
            Resolved::Command(name) => name,
            Resolved::Unknown { closest } => {
                out.status(&format!(
                    "unknown command /{typed}; did you mean {closest}? /help lists them"
                ));
                return Flow::Continue;
            }
        };
        match name {
            "model" if arg.is_empty() => {
                let effort = self
                    .agent
                    .config()
                    .models
                    .get(&session.model)
                    .and_then(|m| m.reasoning_effort.as_deref())
                    .map(|e| format!(", reasoning_effort {e}"))
                    .unwrap_or_default();
                out.status(&format!(
                    "model {} ({}){effort}",
                    session.model, session.provider
                ));
                let choices = self.model_choices(session);
                match self.backend.choose("model", &choices) {
                    Some(id) => self.set_model(&id, session, out),
                    // No terminal to pick with, or the user pressed Esc.
                    None => {
                        for id in choices.iter().take(MENU_CHOICES) {
                            out.status(&format!("  {id}"));
                        }
                        out.status("/model <id> switches");
                    }
                }
            }
            "model" => {
                let known = self.model_choices(session);
                if let Some(id) = self.resolve_choice("model", arg, &known, out) {
                    self.set_model(&id, session, out);
                }
            }
            "effort" => {
                let choices = self.effort_choices();
                if choices.is_empty() {
                    out.status(&format!(
                        "{} does not take a reasoning effort; openai sends it, anthropic does not",
                        session.provider
                    ));
                } else if arg.is_empty() {
                    let line = self.effort_line(session);
                    out.status(&line);
                    match self.backend.choose("reasoning effort", &choices) {
                        Some(value) => self.set_effort(&value, session, out),
                        // No terminal to pick with, or the user left it.
                        None => {
                            for value in &choices {
                                out.status(&format!("  {value}"));
                            }
                            out.status("/effort <value> sets it");
                        }
                    }
                } else if let Some(value) =
                    self.resolve_choice("reasoning effort", arg, &choices, out)
                {
                    self.set_effort(&value, session, out);
                }
            }
            "provider" if arg.is_empty() => {
                out.status(&format!(
                    "provider {} (model {})",
                    session.provider, session.model
                ));
                let choices = vec![
                    ProviderName::Anthropic.as_str().to_string(),
                    ProviderName::OpenAi.as_str().to_string(),
                ];
                match self.backend.choose("provider", &choices) {
                    Some(name) => self.switch_provider(&name, session, out),
                    None => out.status("switch with /provider anthropic|openai"),
                }
            }
            "provider" => self.switch_provider(arg, session, out),
            "help" => {
                for (name, what) in COMMANDS {
                    out.status(&format!("{name:<12} {what}"));
                }
            }
            "clear" => {
                self.save(session, out);
                self.agent.set_redactor(self.backend.fresh_redactor());
                *session = self.agent.new_session();
                out.status(&format!("new session {}", session.id));
            }
            "compact" => {
                match self.agent.compact(session, out).await {
                    Ok(Some(_)) => {}
                    Ok(None) => out.status(&format!(
                        "nothing to compact: fewer than {} turns beyond the kept ones",
                        self.agent.config().agent.keep_recent_turns + 1
                    )),
                    Err(e) => out.status(&format!("compaction failed: {e}")),
                }
                self.save(session, out);
            }
            "cost" => {
                for line in cost_lines(session, self.agent.config().provider.context_window) {
                    out.status(&line);
                }
            }
            "redactions" => {
                if session.redactions.is_empty() {
                    out.status("(nothing was redacted)");
                }
                for line in redaction_lines(&session.redactions) {
                    out.status(&line);
                }
            }
            "config" => match toml::to_string_pretty(&session.config) {
                Ok(text) => text.lines().for_each(|l| out.status(l)),
                Err(e) => out.status(&format!("cannot render the config: {e}")),
            },
            "mcp" if arg.is_empty() => {
                let statuses = self.agent.start_mcp(out).await;
                if statuses.is_empty() {
                    out.status("no MCP servers configured; add [[mcp]] to the config");
                }
                for status in statuses {
                    out.status(&status.line());
                }
            }
            "mcp" => self.enable_mcp(arg, out).await,
            "plan" => self.toggle_plan(out),
            "go" => self.go(session, out).await,
            "goal" if arg.is_empty() => match session.goal.as_deref() {
                Some(goal) => {
                    out.status(&format!("goal: {goal}"));
                    out.status("/goal <statement> replaces it, /goal clear removes it");
                }
                None => out.status("no goal set; /goal <statement> sets one"),
            },
            "goal" if arg.trim() == "clear" => {
                session.goal = None;
                out.status("goal cleared");
                self.save(session, out);
            }
            "goal" => {
                session.goal = Some(arg.trim().to_string());
                out.status(&format!("goal: {}", arg.trim()));
                self.save(session, out);
            }
            "status" => self.status(session, out),
            "context" => self.context_report(session, out),
            "permissions" => self.permissions(arg, session, out),
            "init" => self.init(session, out),
            "copy" => self.copy(arg, session, out),
            "diff" => self.diff_command(arg, session, out),
            "doctor" => {
                let model = session.model.clone();
                let offline = matches!(arg.trim(), "--offline" | "offline");
                let checks = self.backend.doctor(&model, offline).await;
                if checks.is_empty() {
                    out.status("no checks to run from here");
                }
                for check in &checks {
                    out.status(&check.line());
                }
                let failed = checks.iter().filter(|c| !c.ok).count();
                out.status(&if failed == 0 {
                    "everything airlok needs is working".to_string()
                } else {
                    format!("{failed} check(s) failed")
                });
            }
            "btw" if arg.is_empty() => {
                out.status("/btw <question> answers beside the task, without joining it")
            }
            "btw" => {
                let interrupt = self.interrupt.clone();
                // Bracketed like a turn: the renderer holds text back until
                // end_turn, and without it a short answer sits in the
                // buffer until something else flushes it.
                out.begin_turn();
                let asked = self.agent.aside(session, arg, out, &interrupt).await;
                out.end_turn();
                match asked {
                    Ok(_) => self.save(session, out),
                    Err(e) => out.status(&format!("the side question failed: {e}")),
                }
            }
            "exit" => return Flow::Exit,
            other => unreachable!("/{other} is in COMMANDS but not handled"),
        }
        Flow::Continue
    }

    /// Turns a disabled MCP server on for the rest of this session and
    /// starts it. The configuration file is not changed.
    async fn enable_mcp(&mut self, name: &str, out: &mut dyn Output) {
        let Some(server) = self
            .agent
            .config_mut()
            .mcp
            .iter_mut()
            .find(|server| server.name == name)
        else {
            out.status(&format!("no MCP server called {name}; /mcp lists them"));
            return;
        };
        if server.enabled {
            out.status(&format!("{name} is already enabled"));
            return;
        }
        server.enabled = true;
        self.agent.forget_mcp(name);
        for status in self.agent.start_mcp(out).await {
            if status.server == name {
                out.status(&format!("{} (for this session)", status.line()));
            }
        }
    }

    fn toggle_plan(&mut self, out: &mut dyn Output) {
        let on = !self.agent.plan_mode();
        self.agent.set_plan_mode(on);
        out.status(if on {
            "plan mode: read-only tools, and the reply is a plan. /go carries it out; /plan leaves without running it"
        } else {
            "plan mode off"
        });
    }

    /// Leaves plan mode and runs the last plan as the task.
    async fn go(&mut self, session: &mut Session, out: &mut dyn Output) {
        if !self.agent.plan_mode() {
            out.status("not in plan mode; /plan starts it");
            return;
        }
        let Some(plan) = self.agent.plan().map(str::to_string) else {
            out.status("no plan yet: describe the task and wait for the plan");
            return;
        };
        self.agent.set_plan_mode(false);
        out.status("plan mode off; carrying out the plan");
        self.turn(session, &format!("Carry out this plan:\n\n{plan}"), out)
            .await;
    }

    /// Runs a `!` command in the working directory, shows what it printed,
    /// and adds both to the session for the model's next turn.
    async fn run_local(&mut self, command: &str, session: &mut Session, out: &mut dyn Output) {
        if command.is_empty() {
            out.status("usage: !<command> runs it here and adds its output to the conversation");
            return;
        }
        let config = self.agent.config();
        let bash = Bash::new(&config.cwd, config.agent.bash_timeout);
        let output = bash
            .execute(json!({ "command": command }))
            .await
            .unwrap_or_else(|e| format!("error: {e}"));
        out.command_output(&output);
        session.messages.push(Message::user_text(format!(
            "I ran a command in the terminal. Its output is context for my next message, \
             not a request.\n$ {command}\n{}",
            truncate_output(output)
        )));
        session.touch();
        self.save(session, out);
    }

    /// Appends a `#` note to ./AIRLOK.md as a list item, creating the file,
    /// and rebuilds the context block so the note applies from the next
    /// turn.
    fn remember(&mut self, note: &str, out: &mut dyn Output) {
        if note.is_empty() {
            out.status("usage: #<note> adds a line to ./AIRLOK.md");
            return;
        }
        let cwd = self.agent.config().cwd.clone();
        let path = cwd.join(INSTRUCTION_FILES[0]);
        let created = !path.exists();
        if let Err(e) = append_note(&path, note) {
            out.status(&format!("cannot write {}: {e}", path.display()));
            return;
        }
        if let Some(block) = self.backend.context() {
            self.agent.set_context_block(&block);
        }
        // AIRLOK.md takes precedence, so a new one hides the fallbacks.
        let hidden = INSTRUCTION_FILES[1..]
            .iter()
            .find(|name| created && cwd.join(name).exists());
        out.status(&match hidden {
            Some(name) => {
                format!("added to ./AIRLOK.md (new file, read instead of {name} from now on)")
            }
            None => "added to ./AIRLOK.md".to_string(),
        });
    }

    /// Takes the id, records it, and says so.
    fn set_model(&mut self, id: &str, session: &mut Session, out: &mut dyn Output) {
        // The model being left is remembered too, so the id airlok started
        // on is still on offer after a switch.
        let leaving = self.agent.config().provider.model.clone();
        self.agent.config_mut().provider.model = id.to_string();
        for name in [leaving, id.to_string()] {
            self.used.retain(|used| *used != name);
            self.used.insert(0, name);
        }
        self.record_provider(session);
        out.status(&format!("model {id} for the rest of this session"));
        self.save(session, out);
    }

    /// The id to switch to, having checked it against the ones airlok
    /// knows. An exact match is taken as given. Anything else is offered
    /// with the nearest known ids and a question, rather than accepted
    /// silently and failing a turn later.
    /// `typed` when `known` contains it, else the nearest are named and
    /// the user is asked whether to use it anyway. The typed value is
    /// always offered first, so Enter never takes a near match. `kind`
    /// names the thing being chosen, for the messages.
    fn resolve_choice(
        &mut self,
        kind: &str,
        typed: &str,
        known: &[String],
        out: &mut dyn Output,
    ) -> Option<String> {
        if known.iter().any(|value| value == typed) {
            return Some(typed.to_string());
        }
        let near: Vec<String> = nearest(typed, known);
        out.status(&format!("{typed} is not a {kind} airlok has seen here"));
        if !near.is_empty() {
            out.status(&format!("did you mean: {}", near.join(", ")));
        }
        match self.backend.choose(
            &format!("use {typed} anyway, or pick one airlok knows"),
            &once_then(typed, &near, known),
        ) {
            Some(chosen) => Some(chosen),
            None => {
                out.status(&format!("left the {kind} alone"));
                None
            }
        }
    }

    /// What the provider in use accepts for `reasoning_effort`. Only the
    /// openai provider sends it, so anthropic offers nothing.
    fn effort_choices(&mut self) -> Vec<String> {
        match self.agent.config().provider.name {
            ProviderName::OpenAi => ["none", "minimal", "low", "medium", "high"]
                .iter()
                .map(|value| (*value).to_string())
                .collect(),
            ProviderName::Anthropic => Vec::new(),
        }
    }

    /// The effort in force for the model in use, and where it came from:
    /// the per-model config, this session, or the provider's own default.
    fn effort_line(&mut self, session: &Session) -> String {
        let model = session.model.clone();
        let in_force = self
            .agent
            .config()
            .models
            .get(&model)
            .and_then(|m| m.reasoning_effort.clone());
        let configured = self.backend.startup_effort(&model);
        match (in_force, configured) {
            (None, _) => format!("reasoning effort for {model}: the provider's default"),
            (Some(value), Some(from_file)) if value == from_file => {
                format!("reasoning effort for {model}: {value}, from the config for this model")
            }
            (Some(value), _) => {
                format!("reasoning effort for {model}: {value}, set for this session")
            }
        }
    }

    /// Uses `value` for the model in use for the rest of the session. It
    /// goes in the config the request is built from, and `record_provider`
    /// copies that into the session, so `--resume` keeps it.
    fn set_effort(&mut self, value: &str, session: &mut Session, out: &mut dyn Output) {
        let model = session.model.clone();
        self.agent
            .config_mut()
            .models
            .entry(model.clone())
            .or_default()
            .reasoning_effort = Some(value.to_string());
        self.record_provider(session);
        out.status(&format!(
            "reasoning effort {value} for {model} for the rest of this session"
        ));
        // Raising the effort on a model the config pins to none is how a
        // deployment starts refusing tools, which reads as an unrelated
        // failure a turn later.
        if value != "none" && self.backend.startup_effort(&model).as_deref() == Some("none") {
            out.status(&format!(
                "note: the config sets {model} to none. On chat completions this deployment may \
                 refuse to take tools with an effort set; /effort none puts it back"
            ));
        }
        self.save(session, out);
    }

    /// What airlok changed on disk this session: each recorded file diffed
    /// from what was there before its first write, against what is there
    /// now. A path whose original was too large to keep says so rather
    /// than showing a misleading diff.
    fn diff_command(&mut self, arg: &str, session: &Session, out: &mut dyn Output) {
        if session.writes.is_empty() {
            out.status("airlok has not written anything this session");
            return;
        }
        let arg = arg.trim();
        let wanted: Vec<&crate::session::WriteRecord> = if arg.is_empty() {
            session.writes.iter().collect()
        } else {
            let joined = self.agent.config().cwd.join(arg).display().to_string();
            session
                .writes
                .iter()
                .filter(|w| w.path == arg || w.path == joined || w.path.ends_with(arg))
                .collect()
        };
        if wanted.is_empty() {
            out.status(&format!("{arg} was not written this session"));
            return;
        }
        for record in wanted {
            let current = std::fs::read_to_string(&record.path).unwrap_or_default();
            match &record.original {
                None => out.status(&format!(
                    "{}: changed, original too large to show",
                    record.path
                )),
                Some(original) if original == &current => {
                    out.status(&format!("{}: back to how it started", record.path))
                }
                Some(original) => {
                    let diff = crate::tools::unified_diff(&record.path, original, &current);
                    out.diff(&record.path, &diff);
                }
            }
        }
    }

    /// Puts a reply on the clipboard: the last one, the Nth from last, or
    /// the last fenced code block.
    fn copy(&mut self, arg: &str, session: &Session, out: &mut dyn Output) {
        let replies: Vec<String> = session
            .messages
            .iter()
            .filter(|m| m.role == airlok_llm::Role::Assistant)
            .map(|m| {
                m.content
                    .iter()
                    .filter_map(|b| match b {
                        airlok_llm::ContentBlock::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .filter(|text| !text.trim().is_empty())
            .collect();

        let arg = arg.trim();
        let (what, text) = if arg == "code" {
            match replies.iter().rev().find_map(|reply| fenced_block(reply)) {
                Some(block) => ("the last code block".to_string(), block),
                None => return out.status("no fenced code block in this session"),
            }
        } else {
            let nth = if arg.is_empty() {
                1
            } else {
                match arg.parse::<usize>() {
                    Ok(n) if n >= 1 => n,
                    _ => return out.status("/copy takes a number from the end, or `code`"),
                }
            };
            match replies
                .len()
                .checked_sub(nth)
                .and_then(|at| replies.get(at))
            {
                Some(text) => (
                    if nth == 1 {
                        "the last reply".to_string()
                    } else {
                        format!("reply {nth} from the end")
                    },
                    text.clone(),
                ),
                None => {
                    return out.status(&format!(
                        "there are only {} replies in this session",
                        replies.len()
                    ))
                }
            }
        };

        match to_clipboard(&text) {
            Ok(program) => out.status(&format!(
                "copied {what} ({} characters) with {program}",
                text.chars().count()
            )),
            Err(why) => out.status(&why),
        }
    }

    /// Proposes an AIRLOK.md built from the repository, as a diff. Nothing
    /// is written until the user approves it, and an existing file is only
    /// ever replaced through that same diff.
    fn init(&mut self, session: &mut Session, out: &mut dyn Output) {
        let cwd = self.agent.config().cwd.clone();
        let path = cwd.join(INSTRUCTION_FILES[0]);
        let current = std::fs::read_to_string(&path).unwrap_or_default();
        let proposed = crate::context::starter(&cwd);
        if proposed == current {
            out.status(&format!("{} already says this", path.display()));
            return;
        }
        let diff = crate::tools::unified_diff(&path.display().to_string(), &current, &proposed);
        match out.confirm(&Confirmation::Write {
            path: &path,
            diff: &diff,
        }) {
            Decision::Approve | Decision::ApproveAll | Decision::SaveAll => {
                match std::fs::write(&path, &proposed) {
                    Ok(()) => {
                        out.status(&format!(
                            "wrote {}{}",
                            path.display(),
                            if current.is_empty() {
                                ""
                            } else {
                                " (replaced)"
                            }
                        ));
                        if let Some(block) = self.backend.context() {
                            self.agent.set_context_block(&block);
                        }
                        self.save(session, out);
                    }
                    Err(e) => out.status(&format!("cannot write {}: {e}", path.display())),
                }
            }
            Decision::Reject | Decision::Quit => out.status("left AIRLOK.md alone"),
        }
    }

    /// Shows what airlok asks about, changes one setting for the session,
    /// or writes the session's settings to the project config after asking.
    fn permissions(&mut self, arg: &str, session: &mut Session, out: &mut dyn Output) {
        let mut words = arg.split_whitespace();
        let Some(name) = words.next() else {
            return self.show_permissions(out);
        };
        if name == "save" {
            return self.save_permissions(session, out);
        }
        let value = words.collect::<Vec<_>>().join(" ");
        if value.is_empty() {
            out.status(&format!("/permissions {name} <value> sets it"));
            return;
        }
        let safety = &mut self.agent.config_mut().safety;
        let flag = |value: &str| match value {
            "on" | "true" | "yes" => Some(true),
            "off" | "false" | "no" => Some(false),
            _ => None,
        };
        match name {
            "confirm_writes" | "confirm_bash" | "confirm_mcp" => match flag(&value) {
                Some(on) => {
                    match name {
                        "confirm_writes" => safety.confirm_writes = on,
                        "confirm_bash" => safety.confirm_bash = on,
                        _ => safety.confirm_mcp = on,
                    }
                    out.status(&format!(
                        "{name} {} for the rest of this session",
                        if on { "on" } else { "off" }
                    ));
                }
                None => out.status(&format!("{name} takes on or off, not {value}")),
            },
            "allow" => {
                if !safety.bash_allowlist.iter().any(|e| e == &value) {
                    safety.bash_allowlist.push(value.clone());
                }
                out.status(&format!("commands starting `{value}` run without asking"));
            }
            "deny" => {
                if !safety.bash_denylist.iter().any(|e| e == &value) {
                    safety.bash_denylist.push(value.clone());
                }
                out.status(&format!("`{value}` is refused"));
            }
            "unallow" => {
                let before = safety.bash_allowlist.len();
                safety.bash_allowlist.retain(|e| e != &value);
                out.status(&if safety.bash_allowlist.len() == before {
                    format!("`{value}` was not on the allow list")
                } else {
                    format!("`{value}` now asks before running")
                });
            }
            // Emptying the deny list silently is exactly what this must
            // not do, so removing an entry says what it did.
            "undeny" => {
                let before = safety.bash_denylist.len();
                safety.bash_denylist.retain(|e| e != &value);
                out.status(&if safety.bash_denylist.len() == before {
                    format!("`{value}` was not on the deny list")
                } else {
                    format!("`{value}` is no longer refused; it asks instead")
                });
            }
            other => out.status(&format!(
                "unknown setting {other}; one of {}",
                PERMISSION_NAMES.join(", ")
            )),
        }
    }

    fn show_permissions(&mut self, out: &mut dyn Output) {
        let safety = &self.agent.config().safety;
        let onoff = |on: bool| if on { "on" } else { "off" };
        out.status(&format!(
            "confirm_writes {} · confirm_bash {} · confirm_mcp {}",
            onoff(safety.confirm_writes),
            onoff(safety.confirm_bash),
            onoff(safety.confirm_mcp)
        ));
        out.status(&if safety.bash_allowlist.is_empty() {
            "allow list: empty, so every command asks".to_string()
        } else {
            format!("allow list: {}", safety.bash_allowlist.join(", "))
        });
        out.status(&if safety.bash_denylist.is_empty() {
            "deny list: empty".to_string()
        } else {
            format!("deny list: {}", safety.bash_denylist.join(", "))
        });
        let cwd = self.agent.config().cwd.clone();
        let trust = crate::mcp::trust::load(&cwd);
        let lines = trust.lines();
        if lines.is_empty() {
            out.status("mcp trust: nothing saved past this run");
        } else {
            for line in lines {
                out.status(&format!("mcp trust: {line}"));
            }
        }
        out.status("/permissions <name> <value> changes one, /permissions save writes them");
    }

    /// Writes the session's safety settings into the project config, after
    /// showing exactly what would change and asking.
    fn save_permissions(&mut self, session: &mut Session, out: &mut dyn Output) {
        let cwd = self.agent.config().cwd.clone();
        let path = cwd.join(crate::config::PROJECT_CONFIG_NAME);
        let current = std::fs::read_to_string(&path).unwrap_or_default();
        let safety = self.agent.config().safety.clone();
        let updated = match crate::config::with_safety_section(&current, &safety) {
            Ok(text) => text,
            Err(e) => {
                out.status(&format!("cannot update {}: {e}", path.display()));
                return;
            }
        };
        if updated == current {
            out.status(&format!("{} already says this", path.display()));
            return;
        }
        let diff = crate::tools::unified_diff(&path.display().to_string(), &current, &updated);
        match out.confirm(&Confirmation::Write {
            path: &path,
            diff: &diff,
        }) {
            Decision::Approve | Decision::ApproveAll | Decision::SaveAll => {
                match std::fs::write(&path, &updated) {
                    Ok(()) => {
                        out.status(&format!("wrote {}", path.display()));
                        self.save(session, out);
                    }
                    Err(e) => out.status(&format!("cannot write {}: {e}", path.display())),
                }
            }
            Decision::Reject | Decision::Quit => out.status("left the project config alone"),
        }
    }

    /// Where the context window is going, part by part. The parts are
    /// counted the way the request estimator counts them, and the total is
    /// their sum, so the breakdown always adds up to what is reported.
    fn context_report(&mut self, session: &Session, out: &mut dyn Output) {
        let parts = self.agent.context_parts(session);
        let config = self.agent.config();
        let window = config.provider.context_window;
        let threshold = config.agent.compact_threshold(window);

        let tokens = |bytes: usize| (bytes / 4) as u64;
        let rows = [
            ("system prompt", tokens(parts.system)),
            ("context block", tokens(parts.context_block)),
            ("history", tokens(parts.history)),
            ("tool results", tokens(parts.tool_results)),
            ("tool schemas", tokens(parts.tool_schemas)),
        ];
        let total: u64 = rows.iter().map(|(_, t)| *t).sum();

        out.status(&format!(
            "context: about {total} tokens of {window} ({}% of the window)",
            share(total, window)
        ));
        for (label, count) in rows {
            out.status(&format!(
                "  {label:<14} {} {count:>7} ({}%)",
                bar(share(count, total)),
                share(count, total)
            ));
        }
        if parts.instructions > 0 {
            out.status(&format!(
                "  of the block, {} tokens are instruction files",
                tokens(parts.instructions)
            ));
        }
        if session.usage.context_tokens > 0 {
            out.status(&format!(
                "  the last request measured {} tokens{}",
                session.usage.context_tokens,
                if session.usage.estimated {
                    " (estimated)"
                } else {
                    ""
                }
            ));
        }
        out.status(&format!(
            "compaction at {threshold} tokens: {}",
            if total >= threshold {
                "due on the next turn".to_string()
            } else {
                format!("about {} tokens away", threshold - total)
            }
        ));
    }

    /// One screen of what this session is: names, counts and where things
    /// came from, and no values from anywhere.
    fn status(&mut self, session: &Session, out: &mut dyn Output) {
        let config = self.agent.config();
        let effort = config
            .models
            .get(&session.model)
            .and_then(|m| m.reasoning_effort.as_deref())
            .unwrap_or("the provider's default")
            .to_string();
        let cwd = config.cwd.clone();
        let servers: Vec<(String, String, bool)> = config
            .mcp
            .iter()
            .map(|server| {
                (
                    server.name.clone(),
                    server.scope.as_str().to_string(),
                    self.agent.mcp_connected(&server.name),
                )
            })
            .collect();

        out.status(&format!("airlok {}", env!("CARGO_PKG_VERSION")));
        out.status(&format!(
            "provider {} · model {} · effort {effort}",
            session.provider, session.model
        ));
        out.status(&format!(
            "session {} · {} turn(s){}",
            session.id,
            session.turns(),
            if session.interrupted {
                " · last turn interrupted"
            } else {
                ""
            }
        ));
        match crate::context::branch(&cwd) {
            Some(branch) => out.status(&format!("cwd {} · branch {branch}", cwd.display())),
            None => out.status(&format!("cwd {} · not a git repository", cwd.display())),
        }
        if let Some(source) = self.backend.key_source() {
            out.status(&format!("api key from {source}"));
        }
        for line in self.backend.config_files() {
            out.status(&format!("config {line}"));
        }
        if servers.is_empty() {
            out.status("mcp: no servers configured");
        } else {
            for (name, scope, connected) in servers {
                out.status(&format!(
                    "mcp {name} [{scope}]: {}",
                    if connected {
                        "connected this run"
                    } else {
                        "not started"
                    }
                ));
            }
        }
        let mut rehydrate = 0usize;
        let mut redact_only = 0usize;
        for entry in session.redactions.values() {
            match entry.class {
                crate::redact::Class::Rehydrate => rehydrate += 1,
                crate::redact::Class::RedactOnly => redact_only += 1,
            }
        }
        out.status(&format!(
            "redactions: {rehydrate} rehydrate, {redact_only} redact-only"
        ));
    }

    fn switch_provider(&mut self, arg: &str, session: &mut Session, out: &mut dyn Output) {
        let name = match arg.to_ascii_lowercase().as_str() {
            "anthropic" => ProviderName::Anthropic,
            "openai" => ProviderName::OpenAi,
            other => {
                out.status(&format!("unknown provider {other}: anthropic or openai"));
                return;
            }
        };
        match self.backend.switch(name, &session.redactions) {
            Ok(switch) => {
                self.agent.config_mut().provider = switch.config;
                self.agent.set_provider(switch.provider);
                self.agent.set_redactor(switch.redactor);
                self.record_provider(session);
                out.status(&format!(
                    "provider {} with model {} for the rest of this session",
                    session.provider, session.model
                ));
                self.save(session, out);
            }
            Err(missing) => out.status(&format!("cannot switch to {}: {missing}", name.as_str())),
        }
    }

    /// Writes the agent's current provider and model into the session, so
    /// the saved file and a later resume see the change.
    fn record_provider(&self, session: &mut Session) {
        let config = self.agent.config();
        session.provider = config.provider.name.as_str().to_string();
        session.model = config.provider.model.clone();
        session.config = config.to_file();
        session.touch();
    }

    fn save(&self, session: &Session, out: &mut dyn Output) {
        if let Some(store) = self.store {
            if let Err(e) = store.save(session) {
                out.status(&format!("warning: could not save the session: {e}"));
            }
        }
    }
}

/// Appends `- note` on a line of its own, first ending a last line that
/// has no newline. A note that is already a list item goes in as is.
fn append_note(path: &Path, note: &str) -> std::io::Result<()> {
    let unterminated = std::fs::read(path)
        .map(|bytes| bytes.last().is_some_and(|b| *b != b'\n'))
        .unwrap_or(false);
    let item = if note.starts_with("- ") || note.starts_with("* ") {
        note.to_string()
    } else {
        format!("- {note}")
    };
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(file, "{}{item}", if unterminated { "\n" } else { "" })
}

/// What `/cost` prints.
pub fn cost_lines(session: &Session, context_window: u64) -> Vec<String> {
    let usage = &session.usage;
    let percent = (usage.context_tokens * 100)
        .checked_div(context_window)
        .unwrap_or(0);
    let mut lines = vec![
        format!(
            "input {} tokens, output {} tokens over {} turn(s)",
            fmt_tokens(usage.input_tokens),
            fmt_tokens(usage.output_tokens),
            session.turns()
        ),
        format!(
            "context {} of {} tokens ({percent}%)",
            fmt_tokens(usage.context_tokens),
            fmt_tokens(context_window)
        ),
    ];
    if usage.estimated {
        lines.push(
            "estimated at chars/4: the provider reported no usage for at least one request".into(),
        );
    }
    if !session.compactions.is_empty() {
        lines.push(format!("compacted {} time(s)", session.compactions.len()));
    }
    lines
}

#[cfg(test)]
mod tests {
    #[test]
    fn no_clipboard_program_is_reported_rather_than_silently_doing_nothing() {
        let err = to_clipboard_with(&[("airlok-no-such-clipboard", &[])], "hello").unwrap_err();
        assert!(err.contains("no clipboard program found"), "{err}");
        assert!(err.contains("airlok-no-such-clipboard"), "{err}");
    }

    #[test]
    fn the_last_fenced_block_is_what_copy_code_takes() {
        let reply = "first\n\n```sh\nls -l\n```\n\nthen\n\n```rust\nfn main() {}\n```\n";
        assert_eq!(fenced_block(reply).unwrap(), "fn main() {}");
        assert_eq!(fenced_block("no fences here"), None);
    }

    #[test]
    fn a_check_reads_as_pass_or_fail_on_its_own() {
        let good = Check::pass("git", "on main");
        let bad = Check::fail("provider key", "unset");
        assert!(good.line().starts_with("pass git"), "{}", good.line());
        assert!(
            bad.line().starts_with("FAIL provider key"),
            "{}",
            bad.line()
        );
        assert!(good.ok && !bad.ok);
    }

    use super::*;

    #[test]
    fn every_listed_command_resolves_to_itself() {
        for (command, _) in COMMANDS {
            assert_eq!(
                resolve_command(&command[1..]),
                Resolved::Command(&command[1..])
            );
        }
    }

    #[test]
    fn a_prefix_runs_the_first_match_in_menu_order() {
        assert_eq!(resolve_command(""), Resolved::Command("help"));
        assert_eq!(resolve_command("c"), Resolved::Command("cost"));
        assert_eq!(resolve_command("com"), Resolved::Command("compact"));
        assert_eq!(resolve_command("p"), Resolved::Command("plan"));
        assert_eq!(resolve_command("pr"), Resolved::Command("provider"));
        assert_eq!(resolve_command("quit"), Resolved::Command("exit"));
    }

    #[test]
    fn an_unknown_command_names_the_closest() {
        assert_eq!(
            resolve_command("modle"),
            Resolved::Unknown { closest: "/model" }
        );
        assert_eq!(
            resolve_command("comapct"),
            Resolved::Unknown {
                closest: "/compact"
            }
        );
        assert_eq!(resolve_command("q"), Resolved::Unknown { closest: "/go" });
    }
}
