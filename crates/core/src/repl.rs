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

pub trait LineSource {
    fn read_line(&mut self, prompt: &str) -> Line;
}

/// What the REPL needs from the binary, which owns key resolution and
/// provider construction.
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
    fn context(&mut self) -> Option<String> {
        None
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
                out.status(&format!("error: {e}"));
                if let Some(hint) = e.hint(&self.agent.config().provider.model) {
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
        format!(
            "{} · context {percent}% · session {}",
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
                ))
            }
            "model" => {
                self.agent.config_mut().provider.model = arg.to_string();
                self.record_provider(session);
                out.status(&format!("model {arg} for the rest of this session"));
                self.save(session, out);
            }
            "provider" if arg.is_empty() => out.status(&format!(
                "provider {} (model {}); switch with /provider anthropic|openai",
                session.provider, session.model
            )),
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
        if let Some(context) = self.backend.context() {
            self.agent.set_context(context);
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
