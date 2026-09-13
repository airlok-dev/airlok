//! OpenAI Chat Completions API with streaming.
//!
//! Wire reference: `POST /v1/chat/completions` with `"stream": true` returns
//! server-sent events whose `data:` payloads are chat completion chunks.
//! `choices[0].delta` carries `content` text and `tool_calls` fragments
//! keyed by `index`; `finish_reason` arrives in a final chunk, then (with
//! `stream_options.include_usage`) a chunk with empty `choices` and
//! `usage`, then the sentinel `data: [DONE]`.
//!
//! Azure OpenAI serves the same surface at `<resource>.openai.azure.com/openai/v1`
//! and authenticates with an `api-key` header instead of a bearer token.

use std::collections::BTreeMap;

use futures::stream::{self, BoxStream, StreamExt, TryStreamExt};
use serde_json::{json, Value};
use tracing::{debug, trace};

use crate::sse;
use crate::types::{Api, ContentBlock, ImageSource, Request, Role, StopReason, StreamEvent, Usage};
use crate::{LlmError, Provider};

pub const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
pub const DEFAULT_MODEL: &str = "gpt-5.5";
const API_KEY_VAR: &str = "OPENAI_API_KEY";
const AZURE_API_KEY_VAR: &str = "AZURE_OPENAI_API_KEY";
const BASE_URL_VAR: &str = "OPENAI_BASE_URL";

/// How the key is presented. OpenAI wants a bearer token; Azure wants an
/// `api-key` header.
pub enum Auth {
    Bearer(String),
    ApiKey(String),
}

impl Auth {
    /// Picks the header style for a key: Azure OpenAI hosts and the Azure
    /// env var take `api-key`, everything else a bearer token.
    pub fn for_endpoint(base_url: &str, key_env: Option<&str>, key: String) -> Auth {
        let host = base_url
            .trim_start_matches("https://")
            .trim_start_matches("http://")
            .split('/')
            .next()
            .unwrap_or_default();
        if host.ends_with(".openai.azure.com") || key_env == Some(AZURE_API_KEY_VAR) {
            Auth::ApiKey(key)
        } else {
            Auth::Bearer(key)
        }
    }

    fn apply(&self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match self {
            Auth::Bearer(key) => request.bearer_auth(key),
            Auth::ApiKey(key) => request.header("api-key", key),
        }
    }
}

pub struct OpenAi {
    client: reqwest::Client,
    auth: Auth,
    base_url: String,
}

impl OpenAi {
    /// Reads `AZURE_OPENAI_API_KEY` (sent as `api-key`) or else
    /// `OPENAI_API_KEY` (sent as a bearer token), and an optional
    /// `OPENAI_BASE_URL` such as `https://<resource>.openai.azure.com/openai/v1`.
    pub fn from_env() -> Result<Self, LlmError> {
        let auth = match crate::key_from_env(AZURE_API_KEY_VAR) {
            Ok(key) => Auth::ApiKey(key),
            Err(_) => Auth::Bearer(crate::key_from_env(API_KEY_VAR)?),
        };
        let base_url = std::env::var(BASE_URL_VAR)
            .ok()
            .filter(|u| !u.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
        Ok(Self::new(auth, base_url))
    }

    pub fn new(auth: Auth, base_url: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            auth,
            base_url: base_url.trim_end_matches('/').to_string(),
        }
    }

    fn request(&self, path: &str, body: &Value) -> reqwest::RequestBuilder {
        let builder = self
            .client
            .post(format!("{}{path}", self.base_url))
            .header("content-type", "application/json")
            .json(body);
        self.auth.apply(builder)
    }
}

impl Provider for OpenAi {
    fn stream(&self, request: Request) -> BoxStream<'_, Result<StreamEvent, LlmError>> {
        let open = async move {
            debug!(
                model = %request.model,
                messages = request.messages.len(),
                api = %request.api,
                "sending request"
            );
            let events = match request.api {
                Api::Chat => {
                    let response = self
                        .request("/chat/completions", &wire_request(&request))
                        .send()
                        .await?;
                    let bytes = sse::body_or_error(response).await?;
                    sse::events(bytes, ChunkAssembler::default()).boxed()
                }
                Api::Responses => {
                    let response = self
                        .request("/responses", &wire_responses(&request))
                        .send()
                        .await?;
                    let bytes = sse::body_or_error(response).await?;
                    sse::events(bytes, ResponsesAssembler::default()).boxed()
                }
            };
            Ok::<_, LlmError>(events)
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
        let mut images = Vec::new();
        for block in &message.content {
            match block {
                ContentBlock::Text { text: t } => text.push_str(t),
                ContentBlock::Image { source } => {
                    if let Some(url) = data_url(source) {
                        images.push(json!({"type": "image_url", "image_url": {"url": url}}));
                    }
                }
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
                // Only a message carrying an image needs the array form,
                // so every text-only message keeps the shape it had.
                if !images.is_empty() {
                    let mut parts = Vec::new();
                    if !text.is_empty() {
                        parts.push(json!({"type": "text", "text": text}));
                    }
                    parts.extend(images);
                    messages.push(json!({"role": "user", "content": parts}));
                } else if !text.is_empty() {
                    messages.push(json!({"role": "user", "content": text}));
                }
            }
        }
    }

    let mut body = json!({
        "model": request.model,
        "stream": true,
        "stream_options": {"include_usage": true},
        "max_completion_tokens": request.max_tokens,
        "messages": messages,
    });
    if let Some(effort) = &request.reasoning_effort {
        body["reasoning_effort"] = Value::String(effort.clone());
    }
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

/// Maps our message model onto Responses input items. A tool call and
/// its result are top-level items here, not parts of a message, and the
/// system prompt is `instructions` rather than a message.
fn wire_responses(request: &Request) -> Value {
    let mut input = Vec::new();
    for message in &request.messages {
        let mut text = String::new();
        let mut calls = Vec::new();
        let mut results = Vec::new();
        let mut images = Vec::new();
        for block in &message.content {
            match block {
                ContentBlock::Text { text: t } => text.push_str(t),
                ContentBlock::Image { source } => {
                    if let Some(url) = data_url(source) {
                        images.push(json!({"type": "input_image", "image_url": url}));
                    }
                }
                ContentBlock::ToolUse { id, name, input } => calls.push(json!({
                    "type": "function_call",
                    "call_id": id,
                    "name": name,
                    "arguments": input.to_string(),
                })),
                ContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    ..
                } => results.push(json!({
                    "type": "function_call_output",
                    "call_id": tool_use_id,
                    "output": content,
                })),
            }
        }
        if !text.is_empty() || !images.is_empty() {
            let (role, part) = match message.role {
                Role::Assistant => ("assistant", "output_text"),
                Role::User => ("user", "input_text"),
            };
            let mut content = Vec::new();
            if !text.is_empty() {
                content.push(json!({"type": part, "text": text}));
            }
            content.extend(images);
            input.push(json!({
                "type": "message",
                "role": role,
                "content": content,
            }));
        }
        input.append(&mut calls);
        input.append(&mut results);
    }

    let mut body = json!({
        "model": request.model,
        "stream": true,
        "max_output_tokens": request.max_tokens,
        "input": input,
    });
    if !request.system.is_empty() {
        body["instructions"] = Value::String(request.system.clone());
    }
    // Chat Completions takes a bare string; here the same setting is an
    // object, and a string would be rejected as reasoning.effort.
    if let Some(effort) = &request.reasoning_effort {
        body["reasoning"] = json!({"effort": effort});
    }
    if !request.tools.is_empty() {
        let tools: Vec<Value> = request
            .tools
            .iter()
            .map(|t| {
                json!({
                    "type": "function",
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                })
            })
            .collect();
        body["tools"] = Value::Array(tools);
    }
    body
}

/// Accumulates Responses events. Arguments stream per output item, so
/// they are buffered by item id, which is not the `call_id` the result
/// must quote later. There is no `[DONE]`: `response.completed` ends it.
#[derive(Default)]
struct ResponsesAssembler {
    calls: Vec<(String, PartialCall)>,
    ended: bool,
}

impl ResponsesAssembler {
    fn call_mut(&mut self, item_id: &str) -> &mut PartialCall {
        if let Some(at) = self.calls.iter().position(|(id, _)| id == item_id) {
            return &mut self.calls[at].1;
        }
        self.calls
            .push((item_id.to_string(), PartialCall::default()));
        &mut self.calls.last_mut().expect("just pushed").1
    }

    fn drain_calls(&mut self) -> Result<Vec<StreamEvent>, LlmError> {
        let mut out = Vec::new();
        for (_, call) in std::mem::take(&mut self.calls) {
            out.push(StreamEvent::ToolUse {
                id: call.id,
                name: call.name,
                input: crate::parse_arguments(&call.arguments)?,
            });
        }
        Ok(out)
    }

    /// Ends the turn, in the order the chat assembler uses: the calls,
    /// then usage, then the end of the message.
    fn finish(
        &mut self,
        usage: &Value,
        stop_reason: Option<StopReason>,
    ) -> Result<Vec<StreamEvent>, LlmError> {
        self.ended = true;
        let had_calls = !self.calls.is_empty();
        let mut out = self.drain_calls()?;
        if let (Some(input), Some(output)) = (
            usage["input_tokens"].as_u64(),
            usage["output_tokens"].as_u64(),
        ) {
            out.push(StreamEvent::Usage(Usage {
                input_tokens: input,
                output_tokens: output,
            }));
        }
        let ended_as = stop_reason.unwrap_or(if had_calls {
            StopReason::ToolUse
        } else {
            StopReason::EndTurn
        });
        out.push(StreamEvent::MessageEnd {
            stop_reason: ended_as,
        });
        Ok(out)
    }
}

impl sse::Assembler for ResponsesAssembler {
    fn ended(&self) -> bool {
        self.ended
    }

    fn feed(&mut self, payload: &str) -> Result<Vec<StreamEvent>, LlmError> {
        // Not part of this wire format, but a proxy that adds it should
        // not strand the turn.
        if payload.trim() == "[DONE]" {
            return if self.ended {
                Ok(Vec::new())
            } else {
                self.finish(&Value::Null, None)
            };
        }
        let value: Value = serde_json::from_str(payload)?;
        let kind = value["type"].as_str().unwrap_or_default();
        trace!(kind, "event");
        match kind {
            "error" | "response.failed" => {
                let error = if value["error"].is_object() {
                    &value["error"]
                } else {
                    &value["response"]["error"]
                };
                let message = error["message"]
                    .as_str()
                    .or_else(|| value["message"].as_str())
                    .unwrap_or("unknown stream error");
                Err(LlmError::Protocol(message.to_string()))
            }
            "response.output_text.delta" => {
                match value["delta"].as_str().filter(|t| !t.is_empty()) {
                    Some(text) => Ok(vec![StreamEvent::TextDelta(text.to_string())]),
                    None => Ok(Vec::new()),
                }
            }
            "response.output_item.added" => {
                let item = &value["item"];
                if item["type"] == "function_call" {
                    let item_id = item["id"].as_str().unwrap_or_default().to_string();
                    let call = self.call_mut(&item_id);
                    // The result has to quote call_id, not the item id.
                    if let Some(id) = item["call_id"].as_str() {
                        call.id = id.to_string();
                    }
                    if let Some(name) = item["name"].as_str() {
                        call.name = name.to_string();
                    }
                    if let Some(arguments) = item["arguments"].as_str() {
                        call.arguments.push_str(arguments);
                    }
                }
                Ok(Vec::new())
            }
            "response.function_call_arguments.delta" => {
                let item_id = value["item_id"].as_str().unwrap_or_default().to_string();
                if let Some(delta) = value["delta"].as_str() {
                    self.call_mut(&item_id).arguments.push_str(delta);
                }
                Ok(Vec::new())
            }
            "response.function_call_arguments.done" => {
                // Authoritative, so it replaces whatever the deltas built.
                let item_id = value["item_id"].as_str().unwrap_or_default().to_string();
                if let Some(arguments) = value["arguments"].as_str() {
                    self.call_mut(&item_id).arguments = arguments.to_string();
                }
                Ok(Vec::new())
            }
            "response.completed" => self.finish(&value["response"]["usage"], None),
            "response.incomplete" => {
                let reason = value["response"]["incomplete_details"]["reason"].as_str();
                let stop = match reason {
                    Some("max_output_tokens") => StopReason::MaxTokens,
                    Some("content_filter") => StopReason::Refusal,
                    _ => StopReason::Other,
                };
                self.finish(&value["response"]["usage"], Some(stop))
            }
            _ => Ok(Vec::new()),
        }
    }
}

/// A data URL for the providers that take an image inline. A reference
/// carries no bytes, so it has no URL; `redact_block` turns those into
/// text long before a request is built.
fn data_url(source: &ImageSource) -> Option<String> {
    match source {
        ImageSource::Base64 { media_type, data } => {
            Some(format!("data:{media_type};base64,{data}"))
        }
        ImageSource::Reference { .. } => None,
    }
}

/// Accumulates tool-call fragments by index until the turn finishes.
#[derive(Default)]
struct ChunkAssembler {
    calls: BTreeMap<u64, PartialCall>,
    /// Set by the `finish_reason` chunk; the turn ends at `[DONE]` so the
    /// usage chunk in between is not lost.
    stop_reason: Option<StopReason>,
    ended: bool,
}

#[derive(Default)]
struct PartialCall {
    id: String,
    name: String,
    arguments: String,
}

impl ChunkAssembler {
    /// Tool calls are complete once `finish_reason` arrives.
    fn drain_calls(&mut self) -> Result<Vec<StreamEvent>, LlmError> {
        let mut out = Vec::new();
        for (_, call) in std::mem::take(&mut self.calls) {
            out.push(StreamEvent::ToolUse {
                id: call.id,
                name: call.name,
                input: crate::parse_arguments(&call.arguments)?,
            });
        }
        Ok(out)
    }

    fn finish(&mut self) -> Result<Vec<StreamEvent>, LlmError> {
        self.ended = true;
        let mut out = self.drain_calls()?;
        out.push(StreamEvent::MessageEnd {
            stop_reason: self.stop_reason.take().unwrap_or(StopReason::EndTurn),
        });
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
                self.finish()
            };
        }
        let value: Value = serde_json::from_str(payload)?;
        if let Some(error) = value.get("error") {
            let message = error["message"].as_str().unwrap_or("unknown stream error");
            return Err(LlmError::Protocol(message.to_string()));
        }
        let Some(choice) = value["choices"].get(0) else {
            let usage = &value["usage"];
            if let (Some(input), Some(output)) = (
                usage["prompt_tokens"].as_u64(),
                usage["completion_tokens"].as_u64(),
            ) {
                return Ok(vec![StreamEvent::Usage(Usage {
                    input_tokens: input,
                    output_tokens: output,
                })]);
            }
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
            self.stop_reason = Some(match reason {
                "tool_calls" | "function_call" => StopReason::ToolUse,
                "stop" => StopReason::EndTurn,
                "length" => StopReason::MaxTokens,
                "content_filter" => StopReason::Refusal,
                _ => StopReason::Other,
            });
            out.extend(self.drain_calls()?);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Message, ToolSpec};

    fn built_request(auth: Auth, base_url: &str) -> reqwest::Request {
        OpenAi::new(auth, base_url.to_string())
            .request("/chat/completions", &json!({}))
            .build()
            .unwrap()
    }

    #[test]
    fn openai_key_goes_in_the_authorization_header() {
        let request = built_request(Auth::Bearer("k1".into()), DEFAULT_BASE_URL);
        assert_eq!(
            request.url().as_str(),
            "https://api.openai.com/v1/chat/completions"
        );
        assert_eq!(request.headers()["authorization"], "Bearer k1");
        assert!(request.headers().get("api-key").is_none());
    }

    #[test]
    fn azure_key_goes_in_the_api_key_header() {
        let request = built_request(
            Auth::ApiKey("k2".into()),
            "https://example.openai.azure.com/openai/v1/",
        );
        assert_eq!(
            request.url().as_str(),
            "https://example.openai.azure.com/openai/v1/chat/completions"
        );
        assert_eq!(request.headers()["api-key"], "k2");
        assert!(request.headers().get("authorization").is_none());
    }

    #[test]
    fn auth_style_follows_host_or_env_name() {
        let is_api_key = |a: Auth| matches!(a, Auth::ApiKey(_));
        assert!(is_api_key(Auth::for_endpoint(
            "https://x.openai.azure.com/openai/v1",
            None,
            "k".into()
        )));
        assert!(is_api_key(Auth::for_endpoint(
            DEFAULT_BASE_URL,
            Some("AZURE_OPENAI_API_KEY"),
            "k".into()
        )));
        assert!(!is_api_key(Auth::for_endpoint(
            DEFAULT_BASE_URL,
            Some("OPENAI_API_KEY"),
            "k".into()
        )));
        assert!(!is_api_key(Auth::for_endpoint(
            DEFAULT_BASE_URL,
            None,
            "k".into()
        )));
        assert!(!is_api_key(Auth::for_endpoint(
            "https://proxy.example/openai.azure.com/v1",
            None,
            "k".into()
        )));
    }

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
            "data: {\"id\":\"c1\",\"choices\":[],\"usage\":{\"prompt_tokens\":30,\"completion_tokens\":9,\"total_tokens\":39}}\n\n",
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
                StreamEvent::Usage(Usage {
                    input_tokens: 30,
                    output_tokens: 9
                }),
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
            reasoning_effort: None,
            api: Default::default(),
        };

        let body = wire_request(&request);

        assert_eq!(body["model"], "m");
        assert_eq!(body["max_completion_tokens"], 100);
        assert_eq!(body["stream"], true);
        assert_eq!(body["stream_options"]["include_usage"], true);
        assert!(
            body.get("reasoning_effort").is_none(),
            "only sent when configured"
        );
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

    #[test]
    fn reasoning_effort_is_sent_when_set() {
        let request = Request {
            model: "gpt-6-astra".into(),
            max_tokens: 10,
            system: String::new(),
            messages: vec![Message::user_text("hi")],
            tools: Vec::new(),
            reasoning_effort: Some("none".into()),
            api: Default::default(),
        };
        assert_eq!(wire_request(&request)["reasoning_effort"], "none");
    }

    #[test]
    fn an_empty_tool_list_is_left_out() {
        let request = Request {
            model: "m".into(),
            max_tokens: 10,
            system: String::new(),
            messages: vec![Message::user_text("hi")],
            tools: Vec::new(),
            reasoning_effort: None,
            api: Default::default(),
        };
        assert!(wire_request(&request).get("tools").is_none());
    }

    /// Every shape both builders have to carry: text in both directions,
    /// two calls in one assistant turn, and their results.
    fn conversation_request() -> Request {
        Request {
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
            reasoning_effort: None,
            api: Api::Responses,
        }
    }

    #[test]
    fn maps_tool_calls_and_results_onto_response_input_items() {
        let body = wire_responses(&conversation_request());

        assert_eq!(body["model"], "m");
        assert_eq!(body["max_output_tokens"], 100);
        assert_eq!(body["stream"], true);
        assert_eq!(
            body["instructions"], "sys",
            "the system prompt is not a message here"
        );
        assert!(
            body.get("messages").is_none(),
            "chat shape must not leak in"
        );
        assert!(body.get("reasoning").is_none(), "only sent when configured");
        assert_eq!(
            body["input"],
            json!([
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "do it"}]},
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "on it"}]},
                {"type": "function_call", "call_id": "call_1", "name": "read_file", "arguments": "{\"path\":\"a\"}"},
                {"type": "function_call", "call_id": "call_2", "name": "read_file", "arguments": "{\"path\":\"b\"}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "A"},
                {"type": "function_call_output", "call_id": "call_2", "output": "error: nope"},
            ])
        );
        assert_eq!(
            body["tools"],
            json!([{"type": "function", "name": "read_file", "description": "reads", "parameters": {"type": "object"}}]),
            "function tools are flat here, not wrapped in a function object"
        );
    }

    #[test]
    fn the_reasoning_effort_is_an_object_on_responses() {
        let mut request = conversation_request();
        request.reasoning_effort = Some("none".into());
        let body = wire_responses(&request);
        assert_eq!(body["reasoning"]["effort"], "none");
        assert!(
            body.get("reasoning_effort").is_none(),
            "the chat spelling is rejected as reasoning.effort"
        );
    }

    /// The conversation as flat entries, so the two wire shapes can be
    /// compared for what they say rather than byte for byte.
    fn from_chat(body: &Value) -> Vec<String> {
        let mut out = Vec::new();
        for message in body["messages"].as_array().unwrap() {
            let role = message["role"].as_str().unwrap();
            match role {
                "system" => out.push(format!("system:{}", message["content"].as_str().unwrap())),
                "tool" => out.push(format!(
                    "result:{}:{}",
                    message["tool_call_id"].as_str().unwrap(),
                    message["content"].as_str().unwrap()
                )),
                _ => {
                    if let Some(text) = message["content"].as_str() {
                        out.push(format!("{role}:{text}"));
                    }
                    if let Some(calls) = message["tool_calls"].as_array() {
                        for call in calls {
                            out.push(format!(
                                "call:{}:{}:{}",
                                call["id"].as_str().unwrap(),
                                call["function"]["name"].as_str().unwrap(),
                                call["function"]["arguments"].as_str().unwrap()
                            ));
                        }
                    }
                }
            }
        }
        out
    }

    fn from_responses(body: &Value) -> Vec<String> {
        let mut out = Vec::new();
        if let Some(instructions) = body["instructions"].as_str() {
            out.push(format!("system:{instructions}"));
        }
        for item in body["input"].as_array().unwrap() {
            match item["type"].as_str().unwrap() {
                "message" => out.push(format!(
                    "{}:{}",
                    item["role"].as_str().unwrap(),
                    item["content"][0]["text"].as_str().unwrap()
                )),
                "function_call" => out.push(format!(
                    "call:{}:{}:{}",
                    item["call_id"].as_str().unwrap(),
                    item["name"].as_str().unwrap(),
                    item["arguments"].as_str().unwrap()
                )),
                "function_call_output" => out.push(format!(
                    "result:{}:{}",
                    item["call_id"].as_str().unwrap(),
                    item["output"].as_str().unwrap()
                )),
                other => panic!("unexpected input item {other}"),
            }
        }
        out
    }

    #[test]
    fn both_apis_carry_the_same_conversation() {
        let request = conversation_request();
        assert_eq!(
            from_chat(&wire_request(&request)),
            from_responses(&wire_responses(&request)),
            "a turn must mean the same thing on either API"
        );
    }

    #[test]
    fn placeholders_survive_both_wire_shapes() {
        let request = Request {
            model: "m".into(),
            max_tokens: 10,
            system: "sys".into(),
            messages: vec![
                Message::user_text("my key is <<SECRET_1>>"),
                Message::assistant(vec![ContentBlock::ToolUse {
                    id: "call_1".into(),
                    name: "bash".into(),
                    input: json!({"command": "echo <<SECRET_1>>"}),
                }]),
                Message::tool_results(vec![ContentBlock::ToolResult {
                    tool_use_id: "call_1".into(),
                    content: "KEY=<<SECRET_1>>".into(),
                    is_error: false,
                }]),
            ],
            tools: Vec::new(),
            reasoning_effort: None,
            api: Api::Chat,
        };
        for body in [wire_request(&request), wire_responses(&request)] {
            let wire = body.to_string();
            assert_eq!(
                wire.matches("<<SECRET_1>>").count(),
                3,
                "text, arguments and result all keep the placeholder: {wire}"
            );
            assert!(!wire.contains("sk-ant-"), "{wire}");
        }
    }

    #[test]
    fn responses_assembles_text_and_split_tool_call_from_chunks() {
        let raw = concat!(
            "event: response.created\n",
            "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\"}}\n\n",
            "event: response.output_text.delta\n",
            "data: {\"type\":\"response.output_text.delta\",\"item_id\":\"msg_1\",\"delta\":\"Run\"}\n\n",
            "event: response.output_text.delta\n",
            "data: {\"type\":\"response.output_text.delta\",\"item_id\":\"msg_1\",\"delta\":\"ning\"}\n\n",
            "event: response.output_item.added\n",
            "data: {\"type\":\"response.output_item.added\",\"output_index\":1,\"item\":{\"id\":\"fc_1\",\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"bash\",\"arguments\":\"\"}}\n\n",
            "event: response.function_call_arguments.delta\n",
            "data: {\"type\":\"response.function_call_arguments.delta\",\"item_id\":\"fc_1\",\"delta\":\"{\\\"comm\"}\n\n",
            "event: response.function_call_arguments.delta\n",
            "data: {\"type\":\"response.function_call_arguments.delta\",\"item_id\":\"fc_1\",\"delta\":\"and\\\":\\\"ls\\\"}\"}\n\n",
            "event: response.completed\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":30,\"output_tokens\":9}}}\n\n",
        );

        let mut assembler = ResponsesAssembler::default();
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
                StreamEvent::Usage(Usage {
                    input_tokens: 30,
                    output_tokens: 9
                }),
                StreamEvent::MessageEnd {
                    stop_reason: StopReason::ToolUse
                },
            ]
        );
    }

    #[test]
    fn the_done_event_settles_the_arguments() {
        // A proxy that re-sends the arguments whole must not double them.
        let raw = concat!(
            "data: {\"type\":\"response.output_item.added\",\"item\":{\"id\":\"fc_1\",\"type\":\"function_call\",\"call_id\":\"call_9\",\"name\":\"bash\",\"arguments\":\"\"}}\n\n",
            "data: {\"type\":\"response.function_call_arguments.delta\",\"item_id\":\"fc_1\",\"delta\":\"{\\\"command\\\":\\\"ls\\\"}\"}\n\n",
            "data: {\"type\":\"response.function_call_arguments.done\",\"item_id\":\"fc_1\",\"arguments\":\"{\\\"command\\\":\\\"ls\\\"}\"}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":2}}}\n\n",
        );
        let events = sse::assemble_chunked(raw, &mut ResponsesAssembler::default());
        assert_eq!(
            events[0],
            StreamEvent::ToolUse {
                id: "call_9".into(),
                name: "bash".into(),
                input: json!({"command": "ls"}),
            }
        );
    }

    #[test]
    fn a_completed_response_without_calls_ends_the_turn() {
        let raw = concat!(
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n",
        );
        let events = sse::assemble_chunked(raw, &mut ResponsesAssembler::default());
        assert_eq!(
            events,
            vec![
                StreamEvent::TextDelta("hi".into()),
                StreamEvent::Usage(Usage {
                    input_tokens: 1,
                    output_tokens: 1
                }),
                StreamEvent::MessageEnd {
                    stop_reason: StopReason::EndTurn
                },
            ]
        );
    }

    #[test]
    fn an_incomplete_response_reports_why() {
        let raw = "data: {\"type\":\"response.incomplete\",\"response\":{\"incomplete_details\":{\"reason\":\"max_output_tokens\"}}}\n\n";
        let events = sse::assemble_chunked(raw, &mut ResponsesAssembler::default());
        assert_eq!(
            events,
            vec![StreamEvent::MessageEnd {
                stop_reason: StopReason::MaxTokens
            }]
        );
    }

    #[test]
    fn a_stream_error_is_reported_on_either_api() {
        let mut assembler = ResponsesAssembler::default();
        let err = sse::Assembler::feed(
            &mut assembler,
            r#"{"type":"error","message":"deployment not found"}"#,
        )
        .unwrap_err();
        assert!(format!("{err}").contains("deployment not found"), "{err}");
    }

    #[test]
    fn the_api_decides_the_path() {
        let openai = OpenAi::new(Auth::Bearer("k".into()), DEFAULT_BASE_URL.to_string());
        let chat = openai
            .request("/chat/completions", &json!({}))
            .build()
            .unwrap();
        let responses = openai.request("/responses", &json!({})).build().unwrap();
        assert_eq!(
            chat.url().as_str(),
            "https://api.openai.com/v1/chat/completions"
        );
        assert_eq!(
            responses.url().as_str(),
            "https://api.openai.com/v1/responses"
        );
    }

    /// A user turn with one image, the shape both OpenAI APIs must carry.
    fn image_request(api: Api) -> Request {
        Request {
            model: "m".into(),
            max_tokens: 100,
            system: String::new(),
            messages: vec![Message {
                role: Role::User,
                content: vec![
                    ContentBlock::Text {
                        text: "what is this?".into(),
                    },
                    ContentBlock::Image {
                        source: ImageSource::Base64 {
                            media_type: "image/png".into(),
                            data: "AAAB".into(),
                        },
                    },
                ],
            }],
            tools: Vec::new(),
            reasoning_effort: None,
            api,
        }
    }

    #[test]
    fn an_image_goes_as_a_data_url_on_chat_completions() {
        let body = wire_request(&image_request(Api::Chat));
        assert_eq!(
            body["messages"],
            json!([{
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAB"}},
                ]
            }])
        );
    }

    #[test]
    fn a_message_without_an_image_keeps_its_string_content() {
        let request = Request {
            model: "m".into(),
            max_tokens: 10,
            system: String::new(),
            messages: vec![Message::user_text("hi")],
            tools: Vec::new(),
            reasoning_effort: None,
            api: Api::Chat,
        };
        assert_eq!(
            wire_request(&request)["messages"][0]["content"],
            "hi",
            "the array form is only for messages that carry an image"
        );
    }

    #[test]
    fn an_image_is_an_input_image_on_responses() {
        let body = wire_responses(&image_request(Api::Responses));
        assert_eq!(
            body["input"],
            json!([{
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "what is this?"},
                    {"type": "input_image", "image_url": "data:image/png;base64,AAAB"},
                ]
            }])
        );
    }

    #[test]
    fn a_reference_carries_no_bytes_so_it_reaches_no_wire() {
        let mut request = image_request(Api::Chat);
        request.messages[0].content[1] = ContentBlock::Image {
            source: ImageSource::Reference {
                media_type: "image/png".into(),
                width: 100,
                height: 50,
                hash: "abc".into(),
            },
        };
        let chat = wire_request(&request).to_string();
        assert!(!chat.contains("image_url"), "{chat}");
        let responses = wire_responses(&request).to_string();
        assert!(!responses.contains("input_image"), "{responses}");
    }

    #[test]
    fn an_image_is_not_estimated_by_the_length_of_its_base64() {
        let mut request = image_request(Api::Chat);
        if let ContentBlock::Image {
            source: ImageSource::Base64 { data, .. },
        } = &mut request.messages[0].content[1]
        {
            *data = "A".repeat(1_400_000);
        }
        let estimate = request.estimated_tokens();
        assert!(
            estimate < 5_000,
            "one screenshot must not read as a full context: {estimate}"
        );
    }
}
