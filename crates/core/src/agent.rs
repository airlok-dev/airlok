//! The agent loop.
//!
//! Invariant: history is kept in plaintext inside a [`Session`]. Every
//! request body is redacted at send time, and every response is rehydrated
//! before it is stored or acted on. The provider only ever sees
//! placeholders.
//!
//! TODO(stage N): subagents.

use std::sync::Arc;

use airlok_llm::{
    ContentBlock, Message, Provider, Request, Response, StopReason, StreamEvent, ToolSpec,
};
use futures::StreamExt;
use serde_json::Value;
use tracing::{debug, info, warn};

use crate::config::Config;
use crate::interrupt::{Interrupt, Watcher};
use crate::redact::{
    for_display, redact_only_in, split_incomplete_placeholder, RedactionMap, Redactor,
};
use crate::safety::{CommandVerdict, Confirmation, Decision};
use crate::session::{Compaction, Session};
use crate::tools::{truncate_output, Plan, Tool, ToolError, ToolRegistry};
use crate::{CoreError, Output};

pub struct Agent {
    provider: Arc<dyn Provider>,
    tools: ToolRegistry,
    redactor: Box<dyn Redactor>,
    config: Config,
    /// Rendered `core::context` block, appended to the system prompt.
    context: String,
}

/// What happened during one run.
#[derive(Debug, Clone, Default)]
pub struct RunReport {
    /// Every placeholder issued during the run and what it stood for.
    pub redactions: RedactionMap,
    /// Model round-trips made.
    pub turns: usize,
}

impl RunReport {
    /// One line per redaction naming the kind, length, and class, never
    /// the value.
    pub fn redaction_lines(&self) -> Vec<String> {
        redaction_lines(&self.redactions)
    }
}

/// One line per placeholder: kind, length, and class. Never the value.
pub fn redaction_lines(map: &RedactionMap) -> Vec<String> {
    map.iter()
        .map(|(placeholder, entry)| {
            format!(
                "{placeholder}  {} ({} chars, {})",
                entry.kind,
                entry.value.chars().count(),
                entry.class.as_str()
            )
        })
        .collect()
}

impl Agent {
    pub fn new(
        provider: Arc<dyn Provider>,
        tools: ToolRegistry,
        redactor: Box<dyn Redactor>,
        config: Config,
    ) -> Self {
        Self {
            provider,
            tools,
            redactor,
            config,
            context: String::new(),
        }
    }

    /// Sets the context block built by [`crate::context::build`].
    pub fn with_context(mut self, context: String) -> Self {
        self.context = context;
        self
    }

    pub fn config_mut(&mut self) -> &mut Config {
        &mut self.config
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Swaps the model API, for `/provider`.
    pub fn set_provider(&mut self, provider: Arc<dyn Provider>) {
        self.provider = provider;
    }

    /// Swaps the redactor, for a fresh session that must not inherit the
    /// old one's placeholders.
    pub fn set_redactor(&mut self, redactor: Box<dyn Redactor>) {
        self.redactor = redactor;
    }

    /// One task in a fresh session. Convenience over [`Agent::turn`].
    pub async fn run(
        &mut self,
        prompt: &str,
        out: &mut dyn Output,
    ) -> Result<RunReport, CoreError> {
        let mut session = Session::new(&self.config.cwd.clone(), &self.config);
        let turns = self.turn(&mut session, prompt, out).await?;
        Ok(RunReport {
            redactions: session.redactions,
            turns,
        })
    }

    /// The session this agent would start: same cwd and config snapshot.
    pub fn new_session(&self) -> Session {
        Session::new(&self.config.cwd, &self.config)
    }

    /// Adds the user's prompt to `session` and runs model round-trips until
    /// the model stops calling tools. Returns the number of round-trips.
    pub async fn turn(
        &mut self,
        session: &mut Session,
        prompt: &str,
        out: &mut dyn Output,
    ) -> Result<usize, CoreError> {
        self.turn_with(session, prompt, out, &Interrupt::new())
            .await
    }

    /// [`Agent::turn`] that stops when `interrupt` fires. The text streamed
    /// so far is kept in the session, marked interrupted; pending tool
    /// calls are dropped.
    pub async fn turn_with(
        &mut self,
        session: &mut Session,
        prompt: &str,
        out: &mut dyn Output,
        interrupt: &Interrupt,
    ) -> Result<usize, CoreError> {
        let mut watcher = interrupt.watcher();
        let threshold = self
            .config
            .agent
            .compact_threshold(self.config.provider.context_window);
        if session.usage.context_tokens >= threshold {
            self.compact(session, out).await?;
        }
        let start = session.messages.len();
        let result = self.rounds(session, prompt, out, &mut watcher).await;
        if result.is_err() {
            // A failed turn leaves no dangling user message for the next
            // turn (or a resume) to trip over.
            session.messages.truncate(start);
        }
        result
    }

    /// Replaces everything but the last `keep_recent_turns` turns with a
    /// model-written summary. One request, no tools, through the redactor
    /// like any other. Returns `None` when there is nothing to fold.
    pub async fn compact(
        &mut self,
        session: &mut Session,
        out: &mut dyn Output,
    ) -> Result<Option<Compaction>, CoreError> {
        let starts = session.turn_starts();
        let keep = self.config.agent.keep_recent_turns;
        if starts.len() <= keep {
            return Ok(None);
        }
        let cut = starts[starts.len() - keep];
        let mut to_summarise = session.messages[..cut].to_vec();
        to_summarise.push(Message::user_text(SUMMARY_INSTRUCTION));
        let (request, map) = self.build_request(SUMMARY_SYSTEM, &to_summarise, &[]);
        let before = if session.usage.context_tokens > 0 {
            session.usage.context_tokens
        } else {
            request.estimated_tokens()
        };
        let summary_estimate = request.estimated_tokens();
        let mut silent = Silent;
        let mut never = Interrupt::new().watcher();
        let response = self
            .stream_response(request, &map, &mut silent, &mut never)
            .await?
            .response;
        let summary: String = response
            .content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        if summary.trim().is_empty() {
            return Err(airlok_llm::LlmError::Protocol("empty compaction summary".into()).into());
        }

        let mut messages = vec![
            Message::user_text(format!(
                "Summary of the conversation so far, written when it was compacted:\n\n{summary}"
            )),
            Message::assistant(vec![ContentBlock::Text {
                text: "Understood. I will continue from that summary.".into(),
            }]),
        ];
        messages.extend_from_slice(&session.messages[cut..]);
        session.messages = messages;
        let system = system_prompt(&self.config, &self.context, session);
        let (next, _) = self.build_request(&system, &session.messages, &self.tools.specs());
        let after = next.estimated_tokens();
        // The summary request cost real tokens too. Its context figure is
        // then replaced by the estimate for the compacted history, which
        // the next request's usage report overwrites; that estimate is a
        // status-line figure, so it does not mark the session estimated.
        session
            .usage
            .record(response.usage, summary_estimate, &response.content);
        session.usage.context_tokens = after;
        session.redactions = map;
        let record = Compaction {
            at: crate::session::now_rfc3339(),
            before_tokens: before,
            after_tokens: after,
            summary,
        };
        session.compactions.push(record.clone());
        session.touch();
        out.status(&format!(
            "compacted: {} -> {} tokens",
            fmt_tokens(before),
            fmt_tokens(after)
        ));
        Ok(Some(record))
    }

    async fn rounds(
        &mut self,
        session: &mut Session,
        prompt: &str,
        out: &mut dyn Output,
        watcher: &mut Watcher,
    ) -> Result<usize, CoreError> {
        let system = system_prompt(&self.config, &self.context, session);
        let specs = self.tools.specs();
        session.messages.push(Message::user_text(prompt));
        session.interrupted = false;
        let mut approved = Approved::default();

        for round in 1..=self.config.agent.max_turns {
            let (request, map) = self.build_request(&system, &session.messages, &specs);
            let estimate = request.estimated_tokens();
            let streamed = self.stream_response(request, &map, out, watcher).await?;
            let response = streamed.response;
            debug!(round, stop_reason = ?response.stop_reason, usage = ?response.usage, interrupted = streamed.interrupted, "round complete");
            session
                .usage
                .record(response.usage, estimate, &response.content);
            if streamed.interrupted {
                return Ok(Self::interrupted(
                    session,
                    map,
                    response.content,
                    out,
                    round,
                ));
            }

            let tool_calls: Vec<&ContentBlock> = response
                .content
                .iter()
                .filter(|b| matches!(b, ContentBlock::ToolUse { .. }))
                .collect();
            let wants_tools = response.stop_reason == StopReason::ToolUse && !tool_calls.is_empty();
            let results = if wants_tools {
                tokio::select! {
                    biased;
                    _ = watcher.triggered() => {
                        return Ok(Self::interrupted(session, map, response.content, out, round));
                    }
                    results = self.execute_tools(&tool_calls, out, &mut approved, &map) => results?,
                }
            } else {
                Vec::new()
            };
            session.redactions = map;
            session.messages.push(Message::assistant(response.content));
            session.touch();
            if !wants_tools {
                return Ok(round);
            }
            session.messages.push(Message::tool_results(results));
        }
        Err(CoreError::TurnLimit(self.config.agent.max_turns))
    }

    /// Redacts the whole outbound body. The returned map is what the
    /// response must be rehydrated with.
    fn build_request(
        &mut self,
        system: &str,
        history: &[Message],
        specs: &[ToolSpec],
    ) -> (Request, RedactionMap) {
        let mut map = RedactionMap::new();
        let system = self.redact_str(system, &mut map);
        let messages: Vec<Message> = history
            .iter()
            .map(|m| Message {
                role: m.role,
                content: m
                    .content
                    .iter()
                    .map(|b| self.redact_block(b, &mut map))
                    .collect(),
            })
            .collect();
        let request = Request {
            model: self.config.provider.model.clone(),
            max_tokens: self.config.agent.max_tokens,
            system,
            messages,
            tools: specs.to_vec(),
        };
        (request, map)
    }

    /// Redacts one string. The redactor's map is cumulative, so the latest
    /// one is always the complete one.
    fn redact_str(&mut self, input: &str, map: &mut RedactionMap) -> String {
        let (out, latest) = self.redactor.redact(input);
        *map = latest;
        out
    }

    fn redact_block(&mut self, block: &ContentBlock, map: &mut RedactionMap) -> ContentBlock {
        match block {
            ContentBlock::Text { text } => ContentBlock::Text {
                text: self.redact_str(text, map),
            },
            ContentBlock::ToolUse { id, name, input } => {
                let mut input = input.clone();
                map_strings(&mut input, &mut |s| self.redact_str(s, map));
                ContentBlock::ToolUse {
                    id: id.clone(),
                    name: name.clone(),
                    input,
                }
            }
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => ContentBlock::ToolResult {
                tool_use_id: tool_use_id.clone(),
                content: self.redact_str(content, map),
                is_error: *is_error,
            },
        }
    }

    /// Ends a turn the user cut short: keeps the text so far with a marker,
    /// drops any tool calls, and tells the user.
    fn interrupted(
        session: &mut Session,
        map: RedactionMap,
        content: Vec<ContentBlock>,
        out: &mut dyn Output,
        round: usize,
    ) -> usize {
        session.redactions = map;
        session.messages.push(interrupted_message(content));
        session.interrupted = true;
        session.touch();
        out.status("interrupted");
        round
    }

    /// Consumes the provider stream, printing text as it arrives and
    /// returning the rehydrated assistant turn.
    async fn stream_response(
        &self,
        request: Request,
        map: &RedactionMap,
        out: &mut dyn Output,
        watcher: &mut Watcher,
    ) -> Result<Streamed, CoreError> {
        let mut content = Vec::new();
        let mut text = String::new();
        let mut unshown = String::new();
        let mut usage = None;
        let mut stream = self.provider.stream(request);

        loop {
            let event = tokio::select! {
                biased;
                _ = watcher.triggered() => {
                    self.flush_text(&mut text, &mut unshown, map, out, &mut content);
                    return Ok(Streamed {
                        response: Response { content, stop_reason: StopReason::EndTurn, usage },
                        interrupted: true,
                    });
                }
                event = stream.next() => match event {
                    Some(event) => event?,
                    None => break,
                },
            };
            match event {
                StreamEvent::TextDelta(delta) => {
                    text.push_str(&delta);
                    unshown.push_str(&delta);
                    let (emit, keep) = split_incomplete_placeholder(&unshown);
                    if !emit.is_empty() {
                        out.text(&self.display(emit, map));
                    }
                    unshown = keep.to_string();
                }
                StreamEvent::ToolUse { id, name, input } => {
                    self.flush_text(&mut text, &mut unshown, map, out, &mut content);
                    let mut input = input;
                    map_strings(&mut input, &mut |s| self.redactor.rehydrate(s, map));
                    content.push(ContentBlock::ToolUse { id, name, input });
                }
                StreamEvent::Usage(reported) => usage = Some(reported),
                StreamEvent::MessageEnd { stop_reason } => {
                    self.flush_text(&mut text, &mut unshown, map, out, &mut content);
                    return Ok(Streamed {
                        response: Response {
                            content,
                            stop_reason,
                            usage,
                        },
                        interrupted: false,
                    });
                }
            }
        }
        Err(airlok_llm::LlmError::Protocol("stream ended without MessageEnd".into()).into())
    }

    fn flush_text(
        &self,
        text: &mut String,
        unshown: &mut String,
        map: &RedactionMap,
        out: &mut dyn Output,
        content: &mut Vec<ContentBlock>,
    ) {
        if !unshown.is_empty() {
            out.text(&self.display(unshown, map));
            unshown.clear();
        }
        if !text.is_empty() {
            content.push(ContentBlock::Text {
                text: self.redactor.rehydrate(text, map),
            });
            text.clear();
        }
    }

    /// Terminal text: redact-only placeholders become markers, detected
    /// secrets are masked unless configured to show.
    fn display(&self, text: &str, map: &RedactionMap) -> String {
        for_display(text, map, self.config.redact.show_secrets_in_output)
    }

    async fn execute_tools(
        &self,
        calls: &[&ContentBlock],
        out: &mut dyn Output,
        approved: &mut Approved,
        map: &RedactionMap,
    ) -> Result<Vec<ContentBlock>, CoreError> {
        let mut results = Vec::with_capacity(calls.len());
        for call in calls {
            let ContentBlock::ToolUse { id, name, input } = call else {
                continue;
            };
            let (content, is_error) = match self.tools.get(name) {
                Some(tool) => {
                    out.tool_call(name, &tool.summary(input));
                    let forbidden = redact_only_in(&input.to_string(), map);
                    if forbidden.is_empty() {
                        self.run_tool(name, tool, input, out, approved).await?
                    } else {
                        warn!(tool = %name, "refused: redact-only placeholder in arguments");
                        (refusal(&forbidden), true)
                    }
                }
                None => {
                    warn!(tool = %name, "unknown tool requested");
                    (format!("error: unknown tool `{name}`"), true)
                }
            };
            debug!(tool = %name, bytes = content.len(), "tool result");
            results.push(ContentBlock::ToolResult {
                tool_use_id: id.clone(),
                content,
                is_error,
            });
        }
        Ok(results)
    }

    /// Plans, asks if the policy says so, then executes.
    async fn run_tool(
        &self,
        name: &str,
        tool: &dyn Tool,
        input: &Value,
        out: &mut dyn Output,
        approved: &mut Approved,
    ) -> Result<(String, bool), CoreError> {
        Ok(match self.gate(tool, input, out, approved).await {
            Ok(Gate::Proceed) => {
                info!(tool = %name, "executing");
                match tool.execute(input.clone()).await {
                    Ok(output) => (truncate_output(output), false),
                    Err(e) => {
                        warn!(tool = %name, error = %e, "tool failed");
                        (format!("error: {e}"), true)
                    }
                }
            }
            Ok(Gate::Stop(reason)) => {
                info!(tool = %name, "not executed");
                (reason, true)
            }
            Ok(Gate::Abort) => {
                info!(tool = %name, "run aborted by the user");
                return Err(CoreError::Aborted);
            }
            Err(e) => {
                warn!(tool = %name, error = %e, "tool could not be planned");
                (format!("error: {e}"), true)
            }
        })
    }

    async fn gate(
        &self,
        tool: &dyn Tool,
        input: &Value,
        out: &mut dyn Output,
        approved: &mut Approved,
    ) -> Result<Gate, ToolError> {
        let safety = &self.config.safety;
        let gate = match tool.plan(input).await? {
            Plan::Safe => Gate::Proceed,
            Plan::Write { path, diff } => {
                if !safety.confirm_writes || approved.writes {
                    return Ok(Gate::Proceed);
                }
                match out.confirm(&Confirmation::Write {
                    path: &path,
                    diff: &diff,
                }) {
                    Decision::Approve => Gate::Proceed,
                    Decision::ApproveAll => {
                        approved.writes = true;
                        Gate::Proceed
                    }
                    Decision::Reject => Gate::Stop(format!(
                        "The user rejected this change to {}. Do not retry it unchanged; \
                         explain what you intended or propose a different approach.",
                        path.display()
                    )),
                    Decision::Quit => Gate::Abort,
                }
            }
            Plan::Command { command } => match safety.classify(&command) {
                CommandVerdict::Denied(entry) => Gate::Stop(format!(
                    "Refused: `{command}` matches the deny list entry `{entry}`. \
                     Do not retry it; explain what you intended or propose a different approach."
                )),
                CommandVerdict::Allowed => Gate::Proceed,
                CommandVerdict::Confirm => {
                    if !safety.confirm_bash || approved.bash {
                        return Ok(Gate::Proceed);
                    }
                    match out.confirm(&Confirmation::Command { command: &command }) {
                        Decision::Approve => Gate::Proceed,
                        Decision::ApproveAll => {
                            approved.bash = true;
                            Gate::Proceed
                        }
                        Decision::Reject => Gate::Stop(format!(
                            "The user declined to run `{command}`. Do not retry it unchanged; \
                             explain what you intended or propose a different approach."
                        )),
                        Decision::Quit => Gate::Abort,
                    }
                }
            },
        };
        Ok(gate)
    }
}

/// The result sent back when a call would write or run a redact-only value.
fn refusal(forbidden: &[(&str, &crate::redact::Entry)]) -> String {
    let list: Vec<String> = forbidden
        .iter()
        .map(|(placeholder, entry)| format!("{placeholder} ({})", entry.kind))
        .collect();
    format!(
        "Refused: the arguments contain {}, which must never be written to a file, passed to a command, \
         or reproduced. Do not use it; explain what you needed instead.",
        list.join(", ")
    )
}

/// "Yes to all" state for one run, per kind of prompt.
#[derive(Default)]
struct Approved {
    writes: bool,
    bash: bool,
}

enum Gate {
    Proceed,
    /// Not executed; the string goes back to the model as an error result.
    Stop(String),
    /// The user quit; the run ends without another provider call.
    Abort,
}

fn system_prompt(config: &Config, context: &str, session: &Session) -> String {
    let mut prompt = format!(
        "You are airlok, a coding agent working in the directory {cwd}. \
         Complete the user's task using the available tools, then reply with a short summary of what you did. \
         Some values in files and command output are replaced with placeholders that look like <<SECRET_1>>. \
         Treat them as opaque strings: reproduce them exactly as given whenever they must appear in a file, \
         a command, or your reply, and never invent or alter them. \
         Find things with grep and glob before reading files; do not read files speculatively, and page \
         large files with read_file's offset and limit. Change existing files with edit_file; use write_file \
         only for new files or complete rewrites.",
        cwd = config.cwd.display()
    );
    if !context.is_empty() {
        prompt.push_str("\n\n");
        prompt.push_str(context);
    }
    if let Some(at) = session.resumed_at.last() {
        prompt.push_str(&format!(
            "\n\nThis session was resumed at {at}. The environment block above was rebuilt at that \
             time; the conversation before this point happened earlier and files may have changed since."
        ));
    }
    prompt
}

const SUMMARY_SYSTEM: &str = "You are summarising a coding session so it can continue with less context. \
    Write a compact summary under these headings: Task, Progress so far, Decisions, Files touched, Open items. \
    Keep exact file paths, commands, error messages, and placeholder strings such as <<SECRET_1>> verbatim. \
    Include nothing that is not in the conversation.";

const SUMMARY_INSTRUCTION: &str =
    "Summarise the conversation above for a continuation. Reply with the summary only.";

/// One streamed reply, and whether the user cut it short.
struct Streamed {
    response: Response,
    interrupted: bool,
}

pub const INTERRUPTED_MARKER: &str = "[interrupted by the user before the reply was complete]";

/// The assistant message stored for a turn the user cut short: its text so
/// far with a marker, and no tool calls, since nothing ran.
fn interrupted_message(content: Vec<ContentBlock>) -> Message {
    let mut text: String = content
        .into_iter()
        .filter_map(|b| match b {
            ContentBlock::Text { text } => Some(text),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    if !text.is_empty() {
        text.push('\n');
    }
    text.push_str(INTERRUPTED_MARKER);
    Message::assistant(vec![ContentBlock::Text { text }])
}

/// Swallows the summary stream; the user sees the status line instead.
struct Silent;

impl Output for Silent {
    fn text(&mut self, _chunk: &str) {}
    fn tool_call(&mut self, _name: &str, _summary: &str) {}
    fn status(&mut self, _line: &str) {}
    fn confirm(&mut self, _request: &Confirmation<'_>) -> Decision {
        Decision::Reject
    }
}

/// `180k` for anything past a thousand, else the number itself.
pub fn fmt_tokens(n: u64) -> String {
    if n >= 1000 {
        format!("{}k", (n + 500) / 1000)
    } else {
        n.to_string()
    }
}

/// Applies `f` to every string leaf of a JSON value, in place.
fn map_strings(value: &mut Value, f: &mut dyn FnMut(&str) -> String) {
    match value {
        Value::String(s) => *s = f(s),
        Value::Array(items) => items.iter_mut().for_each(|v| map_strings(v, f)),
        Value::Object(fields) => fields.values_mut().for_each(|v| map_strings(v, f)),
        _ => {}
    }
}
