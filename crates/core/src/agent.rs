//! The agent loop.
//!
//! Invariant: history is kept in plaintext. Every request body is redacted
//! at send time, and every response is rehydrated before it is stored or
//! acted on. The provider only ever sees placeholders.
//!
//! TODO(stage N): context injection (repo map, instructions file), diff
//! preview before writes, compaction when history grows, subagents.

use std::sync::Arc;

use airlok_llm::{
    ContentBlock, Message, Provider, Request, Response, StopReason, StreamEvent, ToolSpec,
};
use futures::StreamExt;
use serde_json::Value;
use tracing::{debug, info, warn};

use crate::config::Config;
use crate::redact::{split_incomplete_placeholder, RedactionMap, Redactor};
use crate::tools::ToolRegistry;
use crate::{CoreError, Output};

pub struct Agent {
    provider: Arc<dyn Provider>,
    tools: ToolRegistry,
    redactor: Box<dyn Redactor>,
    config: Config,
}

/// What happened during one run.
#[derive(Debug, Clone, Default)]
pub struct RunReport {
    /// Every placeholder issued during the run and the value it stood for.
    pub redactions: RedactionMap,
    /// Model round-trips made.
    pub turns: usize,
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
        }
    }

    pub async fn run(
        &mut self,
        prompt: &str,
        out: &mut dyn Output,
    ) -> Result<RunReport, CoreError> {
        let system = system_prompt(&self.config);
        let specs = self.tools.specs();
        let mut history = vec![Message::user_text(prompt)];
        let mut report = RunReport::default();

        for turn in 1..=self.config.agent.max_turns {
            report.turns = turn;
            let (request, map) = self.build_request(&system, &history, &specs);
            let response = self.stream_response(request, &map, out).await?;
            report.redactions = map;
            debug!(turn, stop_reason = ?response.stop_reason, "turn complete");

            let tool_calls: Vec<&ContentBlock> = response
                .content
                .iter()
                .filter(|b| matches!(b, ContentBlock::ToolUse { .. }))
                .collect();
            let wants_tools = response.stop_reason == StopReason::ToolUse && !tool_calls.is_empty();
            let results = if wants_tools {
                self.execute_tools(&tool_calls, out).await
            } else {
                Vec::new()
            };
            history.push(Message::assistant(response.content));
            if !wants_tools {
                return Ok(report);
            }
            history.push(Message::tool_results(results));
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

    /// Consumes the provider stream, printing text as it arrives and
    /// returning the rehydrated assistant turn.
    async fn stream_response(
        &self,
        request: Request,
        map: &RedactionMap,
        out: &mut dyn Output,
    ) -> Result<Response, CoreError> {
        let mut content = Vec::new();
        let mut text = String::new();
        let mut unshown = String::new();
        let mut stream = self.provider.stream(request);

        while let Some(event) = stream.next().await {
            match event? {
                StreamEvent::TextDelta(delta) => {
                    text.push_str(&delta);
                    unshown.push_str(&delta);
                    let (emit, keep) = split_incomplete_placeholder(&unshown);
                    if !emit.is_empty() {
                        out.text(&self.redactor.rehydrate(emit, map));
                    }
                    unshown = keep.to_string();
                }
                StreamEvent::ToolUse { id, name, input } => {
                    self.flush_text(&mut text, &mut unshown, map, out, &mut content);
                    let mut input = input;
                    map_strings(&mut input, &mut |s| self.redactor.rehydrate(s, map));
                    content.push(ContentBlock::ToolUse { id, name, input });
                }
                StreamEvent::MessageEnd { stop_reason } => {
                    self.flush_text(&mut text, &mut unshown, map, out, &mut content);
                    return Ok(Response {
                        content,
                        stop_reason,
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
            out.text(&self.redactor.rehydrate(unshown, map));
            unshown.clear();
        }
        if !text.is_empty() {
            content.push(ContentBlock::Text {
                text: self.redactor.rehydrate(text, map),
            });
            text.clear();
        }
    }

    async fn execute_tools(
        &self,
        calls: &[&ContentBlock],
        out: &mut dyn Output,
    ) -> Vec<ContentBlock> {
        let mut results = Vec::with_capacity(calls.len());
        for call in calls {
            let ContentBlock::ToolUse { id, name, input } = call else {
                continue;
            };
            let (content, is_error) = match self.tools.get(name) {
                Some(tool) => {
                    out.tool_call(name, &tool.summary(input));
                    info!(tool = %name, "executing");
                    match tool.execute(input.clone()).await {
                        Ok(output) => (output, false),
                        Err(e) => {
                            warn!(tool = %name, error = %e, "tool failed");
                            (format!("error: {e}"), true)
                        }
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
        results
    }
}

fn system_prompt(config: &Config) -> String {
    format!(
        "You are airlok, a coding agent working in the directory {cwd}. \
         Complete the user's task using the available tools, then reply with a short summary of what you did. \
         Some values in files and command output are replaced with placeholders that look like <<SECRET_1>>. \
         Treat them as opaque strings: reproduce them exactly as given whenever they must appear in a file, \
         a command, or your reply, and never invent or alter them.",
        cwd = config.cwd.display()
    )
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
