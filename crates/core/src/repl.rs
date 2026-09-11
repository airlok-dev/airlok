//! The interactive loop: read a line, run a turn or a slash command, save.
//! Input arrives through [`LineSource`] so the binary can use readline and
//! tests can feed a script.

use std::sync::Arc;

use airlok_llm::Provider;

use crate::agent::{fmt_tokens, redaction_lines, Agent};
use crate::config::{ProviderConfig, ProviderName};
use crate::redact::{RedactionMap, Redactor};
use crate::session::{Session, SessionStore};
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
}

/// A provider ready to take over the session.
pub struct Switch {
    pub config: ProviderConfig,
    pub provider: Arc<dyn Provider>,
    pub redactor: Box<dyn Redactor>,
}

pub const COMMANDS: &[(&str, &str)] = &[
    ("/help", "list the commands"),
    ("/model", "show the model, or switch with /model <id>"),
    (
        "/provider",
        "show the provider, or switch with /provider anthropic|openai",
    ),
    ("/clear", "start a new session; the current one stays saved"),
    ("/compact", "summarise older turns to free context"),
    ("/cost", "tokens used so far"),
    (
        "/redactions",
        "what was redacted before leaving this machine",
    ),
    ("/config", "the effective configuration"),
    ("/exit", "save and quit; Ctrl-D does the same"),
];

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
                Line::Interrupt => out.status("(use /exit or Ctrl-D to quit)"),
                Line::Text(line) => {
                    let line = line.trim();
                    if line.is_empty() {
                        continue;
                    }
                    match line.strip_prefix('/') {
                        Some(command) => {
                            if self.command(command, &mut session, out).await == Flow::Exit {
                                break;
                            }
                        }
                        None => self.turn(&mut session, line, out).await,
                    }
                }
            }
        }
        self.save(&session, out);
        session
    }

    /// `<model> <cwd basename>> `
    pub fn prompt_text(&self) -> String {
        let config = self.agent.config();
        let dir = config
            .cwd
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| config.cwd.display().to_string());
        format!("{} {dir}> ", config.provider.model)
    }

    async fn turn(&mut self, session: &mut Session, line: &str, out: &mut dyn Output) {
        match self
            .agent
            .turn_with(session, line, out, &self.interrupt)
            .await
        {
            Ok(_) => {}
            Err(CoreError::Aborted) => out.status("aborted"),
            Err(e) => out.status(&format!("error: {e}")),
        }
        out.end_turn();
        self.save(session, out);
    }

    async fn command(
        &mut self,
        command: &str,
        session: &mut Session,
        out: &mut dyn Output,
    ) -> Flow {
        let (name, arg) = match command.trim().split_once(char::is_whitespace) {
            Some((name, arg)) => (name, arg.trim()),
            None => (command.trim(), ""),
        };
        match name {
            "model" if arg.is_empty() => {
                out.status(&format!("model {} ({})", session.model, session.provider))
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
            "exit" | "quit" => return Flow::Exit,
            other => out.status(&format!("unknown command /{other}; /help lists them")),
        }
        Flow::Continue
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
