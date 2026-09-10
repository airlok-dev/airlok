//! OpenAI Chat Completions API with streaming.
//!
//! Wire reference: `POST /v1/chat/completions` with `"stream": true` returns
//! server-sent events whose `data:` payloads are chat completion chunks.
//! `choices[0].delta` carries `content` text and `tool_calls` fragments
//! keyed by `index`; `finish_reason` arrives in a final chunk, then the
//! sentinel `data: [DONE]`.

use std::collections::BTreeMap;

use futures::stream::{self, BoxStream, StreamExt, TryStreamExt};
use serde_json::{json, Value};
use tracing::{debug, trace};

use crate::sse;
use crate::types::{ContentBlock, Request, Role, StopReason, StreamEvent};
use crate::{LlmError, Provider};

const DEFAULT_BASE_URL: &str = "https://api.openai.com";
pub const DEFAULT_MODEL: &str = "gpt-5.5";
const API_KEY_VAR: &str = "OPENAI_API_KEY";

pub struct OpenAi {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
}

impl OpenAi {
    /// Reads the key from `OPENAI_API_KEY`.
    pub fn from_env() -> Result<Self, LlmError> {
        Ok(Self::new(crate::key_from_env(API_KEY_VAR)?))
    }

    pub fn new(api_key: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            base_url: DEFAULT_BASE_URL.to_string(),
        }
    }
}

impl Provider for OpenAi {
    fn stream(&self, request: Request) -> BoxStream<'_, Result<StreamEvent, LlmError>> {
        let open = async move {
            let body = wire_request(&request);
            debug!(model = %request.model, messages = request.messages.len(), "sending request");
            let response = self
                .client
                .post(format!("{}/v1/chat/completions", self.base_url))
                .bearer_auth(&self.api_key)
                .header("content-type", "application/json")
                .json(&body)
                .send()
                .await?;
            let bytes = sse::body_or_error(response).await?;
            Ok::<_, LlmError>(sse::events(bytes, ChunkAssembler::default()))
        };
        stream::once(open).try_flatten().boxed()
    }
}

/// Maps our message model onto Chat Completions messages. Tool results
/// become one `tool` message each, placed directly after the assistant
/// message that made the calls.
fn wire_request(request: &Request) -> Value {
    let mut messages = Vec::new();
    if !request.system.is_empty() {
        messages.push(json!({"role": "system", "content": request.system}));
    }
    for message in &request.messages {
        let mut text = String::new();
        let mut tool_calls = Vec::new();
        for block in &message.content {
            match block {
                ContentBlock::Text { text: t } => text.push_str(t),
                ContentBlock::ToolUse { id, name, input } => tool_calls.push(json!({
                    "id": id,
                    "type": "function",
                    "function": {"name": name, "arguments": input.to_string()},
                })),
                ContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    ..
                } => messages.push(json!({
                    "role": "tool",
                    "tool_call_id": tool_use_id,
                    "content": content,
                })),
            }
        }
        match message.role {
            Role::Assistant => {
                let mut assistant = json!({
                    "role": "assistant",
                    "content": if text.is_empty() { Value::Null } else { Value::String(text) },
                });
                if !tool_calls.is_empty() {
                    assistant["tool_calls"] = Value::Array(tool_calls);
                }
                messages.push(assistant);
            }
            Role::User => {
                if !text.is_empty() {
                    messages.push(json!({"role": "user", "content": text}));
                }
            }
        }
    }

    let mut body = json!({
        "model": request.model,
        "stream": true,
        "max_completion_tokens": request.max_tokens,
        "messages": messages,
    });
    if !request.tools.is_empty() {
        let tools: Vec<Value> = request
            .tools
            .iter()
            .map(|t| {
                json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                })
            })
            .collect();
        body["tools"] = Value::Array(tools);
    }
    body
}

/// Accumulates tool-call fragments by index until the turn finishes.
#[derive(Default)]
struct ChunkAssembler {
    calls: BTreeMap<u64, PartialCall>,
    ended: bool,
}

#[derive(Default)]
struct PartialCall {
    id: String,
    name: String,
    arguments: String,
}

impl ChunkAssembler {
    fn finish(&mut self, stop_reason: StopReason) -> Result<Vec<StreamEvent>, LlmError> {
        self.ended = true;
        let mut out = Vec::new();
        for (_, call) in std::mem::take(&mut self.calls) {
            out.push(StreamEvent::ToolUse {
                id: call.id,
                name: call.name,
                input: crate::parse_arguments(&call.arguments)?,
            });
        }
        out.push(StreamEvent::MessageEnd { stop_reason });
        Ok(out)
    }
}

impl sse::Assembler for ChunkAssembler {
    fn ended(&self) -> bool {
        self.ended
    }

    fn feed(&mut self, payload: &str) -> Result<Vec<StreamEvent>, LlmError> {
        if payload.trim() == "[DONE]" {
            return if self.ended {
                Ok(Vec::new())
            } else {
                self.finish(StopReason::EndTurn)
            };
        }
        let value: Value = serde_json::from_str(payload)?;
        if let Some(error) = value.get("error") {
            let message = error["message"].as_str().unwrap_or("unknown stream error");
            return Err(LlmError::Protocol(message.to_string()));
        }
        let Some(choice) = value["choices"].get(0) else {
            return Ok(Vec::new());
        };
        trace!(finish_reason = ?choice["finish_reason"], "chunk");
        let mut out = Vec::new();
        let delta = &choice["delta"];
        if let Some(text) = delta["content"].as_str().filter(|t| !t.is_empty()) {
            out.push(StreamEvent::TextDelta(text.to_string()));
        }
        if let Some(fragments) = delta["tool_calls"].as_array() {
            for fragment in fragments {
                let index = fragment["index"].as_u64().unwrap_or(0);
                let call = self.calls.entry(index).or_default();
                if let Some(id) = fragment["id"].as_str() {
                    call.id = id.to_string();
                }
                if let Some(name) = fragment["function"]["name"].as_str() {
                    call.name.push_str(name);
                }
                if let Some(arguments) = fragment["function"]["arguments"].as_str() {
                    call.arguments.push_str(arguments);
                }
            }
        }
        if let Some(reason) = choice["finish_reason"].as_str() {
            let stop_reason = match reason {
                "tool_calls" | "function_call" => StopReason::ToolUse,
                "stop" => StopReason::EndTurn,
                "length" => StopReason::MaxTokens,
                "content_filter" => StopReason::Refusal,
                _ => StopReason::Other,
            };
            out.extend(self.finish(stop_reason)?);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Message, ToolSpec};

    #[test]
    fn assembles_text_and_split_tool_call_from_chunks() {
        let raw = concat!(
            "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Run\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ning\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"bash\",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"comm\"}}]},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"and\\\":\\\"ls\\\"}\"}}]},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: {\"id\":\"c1\",\"choices\":[],\"usage\":{\"total_tokens\":9}}\n\n",
            "data: [DONE]\n\n",
        );

        let mut assembler = ChunkAssembler::default();
        let events = sse::assemble_chunked(raw, &mut assembler);

        assert_eq!(
            events,
            vec![
                StreamEvent::TextDelta("Run".into()),
                StreamEvent::TextDelta("ning".into()),
                StreamEvent::ToolUse {
                    id: "call_1".into(),
                    name: "bash".into(),
                    input: json!({"command": "ls"}),
                },
                StreamEvent::MessageEnd {
                    stop_reason: StopReason::ToolUse
                },
            ]
        );
    }

    #[test]
    fn done_without_finish_reason_ends_the_turn() {
        let raw = concat!(
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n",
            "data: [DONE]\n\n",
        );
        let events = sse::assemble_chunked(raw, &mut ChunkAssembler::default());
        assert_eq!(
            events,
            vec![
                StreamEvent::TextDelta("hi".into()),
                StreamEvent::MessageEnd {
                    stop_reason: StopReason::EndTurn
                },
            ]
        );
    }

    #[test]
    fn maps_tool_calls_and_results_onto_chat_messages() {
        let request = Request {
            model: "m".into(),
            max_tokens: 100,
            system: "sys".into(),
            messages: vec![
                Message::user_text("do it"),
                Message::assistant(vec![
                    ContentBlock::Text {
                        text: "on it".into(),
                    },
                    ContentBlock::ToolUse {
                        id: "call_1".into(),
                        name: "read_file".into(),
                        input: json!({"path": "a"}),
                    },
                    ContentBlock::ToolUse {
                        id: "call_2".into(),
                        name: "read_file".into(),
                        input: json!({"path": "b"}),
                    },
                ]),
                Message::tool_results(vec![
                    ContentBlock::ToolResult {
                        tool_use_id: "call_1".into(),
                        content: "A".into(),
                        is_error: false,
                    },
                    ContentBlock::ToolResult {
                        tool_use_id: "call_2".into(),
                        content: "error: nope".into(),
                        is_error: true,
                    },
                ]),
            ],
            tools: vec![ToolSpec {
                name: "read_file".into(),
                description: "reads".into(),
                input_schema: json!({"type": "object"}),
            }],
        };

        let body = wire_request(&request);

        assert_eq!(body["model"], "m");
        assert_eq!(body["max_completion_tokens"], 100);
        assert_eq!(body["stream"], true);
        assert_eq!(
            body["messages"],
            json!([
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "do it"},
                {"role": "assistant", "content": "on it", "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\":\"a\"}"}},
                    {"id": "call_2", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\":\"b\"}"}},
                ]},
                {"role": "tool", "tool_call_id": "call_1", "content": "A"},
                {"role": "tool", "tool_call_id": "call_2", "content": "error: nope"},
            ])
        );
        assert_eq!(
            body["tools"],
            json!([{"type": "function", "function": {"name": "read_file", "description": "reads", "parameters": {"type": "object"}}}])
        );
    }
}
