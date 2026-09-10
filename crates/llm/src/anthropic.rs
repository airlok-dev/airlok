//! Anthropic Messages API with streaming.
//!
//! Wire reference: `POST /v1/messages` with `"stream": true` returns
//! server-sent events whose `data:` payloads carry a `type` field
//! (`message_start`, `content_block_start`, `content_block_delta`,
//! `content_block_stop`, `message_delta`, `message_stop`, `ping`, `error`).

use std::collections::VecDeque;

use futures::stream::{self, BoxStream, StreamExt, TryStreamExt};
use serde::Serialize;
use serde_json::Value;
use tracing::{debug, trace};

use crate::types::{Message, Request, StopReason, StreamEvent, ToolSpec};
use crate::{LlmError, Provider};

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const API_VERSION: &str = "2023-06-01";

pub struct Anthropic {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
}

impl Anthropic {
    /// Reads the key from `ANTHROPIC_API_KEY`.
    pub fn from_env() -> Result<Self, LlmError> {
        let key = std::env::var("ANTHROPIC_API_KEY")
            .ok()
            .filter(|k| !k.trim().is_empty())
            .ok_or(LlmError::MissingApiKey)?;
        Ok(Self::new(key))
    }

    pub fn new(api_key: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            base_url: DEFAULT_BASE_URL.to_string(),
        }
    }
}

#[derive(Serialize)]
struct WireRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    stream: bool,
    #[serde(skip_serializing_if = "str::is_empty")]
    system: &'a str,
    messages: &'a [Message],
    #[serde(skip_serializing_if = "<[ToolSpec]>::is_empty")]
    tools: &'a [ToolSpec],
}

impl Provider for Anthropic {
    fn stream(&self, request: Request) -> BoxStream<'_, Result<StreamEvent, LlmError>> {
        let open = async move {
            let body = WireRequest {
                model: &request.model,
                max_tokens: request.max_tokens,
                stream: true,
                system: &request.system,
                messages: &request.messages,
                tools: &request.tools,
            };
            debug!(model = %request.model, messages = request.messages.len(), "sending request");
            let response = self
                .client
                .post(format!("{}/v1/messages", self.base_url))
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", API_VERSION)
                .header("content-type", "application/json")
                .json(&body)
                .send()
                .await?;
            let status = response.status();
            if !status.is_success() {
                let body = response.text().await.unwrap_or_default();
                return Err(LlmError::Api {
                    status: status.as_u16(),
                    body,
                });
            }
            Ok(
                SseEvents::new(response.bytes_stream().map_ok(|b| b.to_vec()).boxed())
                    .into_stream(),
            )
        };
        stream::once(open).try_flatten().boxed()
    }
}

/// Turns the raw byte stream into [`StreamEvent`]s.
struct SseEvents {
    bytes: BoxStream<'static, reqwest::Result<Vec<u8>>>,
    buffer: String,
    pending: VecDeque<StreamEvent>,
    assembler: BlockAssembler,
    finished: bool,
}

impl SseEvents {
    fn new(bytes: BoxStream<'static, reqwest::Result<Vec<u8>>>) -> Self {
        Self {
            bytes,
            buffer: String::new(),
            pending: VecDeque::new(),
            assembler: BlockAssembler::default(),
            finished: false,
        }
    }

    fn into_stream(self) -> impl futures::Stream<Item = Result<StreamEvent, LlmError>> + Send {
        stream::unfold(self, |mut state| async move {
            loop {
                if let Some(event) = state.pending.pop_front() {
                    return Some((Ok(event), state));
                }
                if state.finished {
                    return None;
                }
                match state.bytes.next().await {
                    None => {
                        state.finished = true;
                        if !state.assembler.ended {
                            return Some((
                                Err(LlmError::Protocol(
                                    "stream ended before message_stop".into(),
                                )),
                                state,
                            ));
                        }
                    }
                    Some(Err(e)) => {
                        state.finished = true;
                        return Some((Err(e.into()), state));
                    }
                    Some(Ok(chunk)) => {
                        state.buffer.push_str(&String::from_utf8_lossy(&chunk));
                        for payload in drain_sse_payloads(&mut state.buffer) {
                            match state.assembler.feed(&payload) {
                                Ok(events) => state.pending.extend(events),
                                Err(e) => {
                                    state.finished = true;
                                    return Some((Err(e), state));
                                }
                            }
                        }
                        if state.assembler.ended {
                            state.finished = true;
                        }
                    }
                }
            }
        })
    }
}

/// Removes every complete SSE event from `buffer` and returns their joined
/// `data:` payloads. Events are separated by a blank line.
fn drain_sse_payloads(buffer: &mut String) -> Vec<String> {
    let mut payloads = Vec::new();
    while let Some(end) = find_event_boundary(buffer) {
        let event: String = buffer.drain(..end.0).collect();
        buffer.drain(..end.1);
        let data: Vec<&str> = event
            .lines()
            .filter_map(|line| line.strip_prefix("data:"))
            .map(str::trim_start)
            .collect();
        if !data.is_empty() {
            payloads.push(data.join("\n"));
        }
    }
    payloads
}

/// Returns (index where the event text ends, length of the separator).
fn find_event_boundary(buffer: &str) -> Option<(usize, usize)> {
    let lf = buffer.find("\n\n").map(|i| (i, 2));
    let crlf = buffer.find("\r\n\r\n").map(|i| (i, 4));
    match (lf, crlf) {
        (Some(a), Some(b)) => Some(if a.0 <= b.0 { a } else { b }),
        (a, b) => a.or(b),
    }
}

/// Reassembles content blocks from the per-block delta events.
#[derive(Default)]
struct BlockAssembler {
    current: Option<PartialBlock>,
    stop_reason: Option<StopReason>,
    ended: bool,
}

enum PartialBlock {
    Text,
    ToolUse {
        id: String,
        name: String,
        json: String,
    },
    Ignored,
}

impl BlockAssembler {
    fn feed(&mut self, payload: &str) -> Result<Vec<StreamEvent>, LlmError> {
        let value: Value = serde_json::from_str(payload)?;
        let kind = value["type"].as_str().unwrap_or_default();
        trace!(kind, "sse event");
        let mut out = Vec::new();
        match kind {
            "content_block_start" => {
                let block = &value["content_block"];
                self.current = Some(match block["type"].as_str() {
                    Some("text") => {
                        if let Some(text) = block["text"].as_str().filter(|t| !t.is_empty()) {
                            out.push(StreamEvent::TextDelta(text.to_string()));
                        }
                        PartialBlock::Text
                    }
                    Some("tool_use") => PartialBlock::ToolUse {
                        id: string_field(block, "id")?,
                        name: string_field(block, "name")?,
                        json: String::new(),
                    },
                    _ => PartialBlock::Ignored,
                });
            }
            "content_block_delta" => {
                let delta = &value["delta"];
                match (delta["type"].as_str(), self.current.as_mut()) {
                    (Some("text_delta"), Some(PartialBlock::Text)) => {
                        out.push(StreamEvent::TextDelta(string_field(delta, "text")?));
                    }
                    (Some("input_json_delta"), Some(PartialBlock::ToolUse { json, .. })) => {
                        json.push_str(delta["partial_json"].as_str().unwrap_or_default());
                    }
                    _ => {}
                }
            }
            "content_block_stop" => {
                if let Some(PartialBlock::ToolUse { id, name, json }) = self.current.take() {
                    let input = if json.trim().is_empty() {
                        Value::Object(Default::default())
                    } else {
                        serde_json::from_str(&json)?
                    };
                    out.push(StreamEvent::ToolUse { id, name, input });
                }
            }
            "message_delta" => {
                if let Some(reason) = value["delta"]["stop_reason"].as_str() {
                    self.stop_reason =
                        serde_json::from_value(Value::String(reason.to_string())).ok();
                }
            }
            "message_stop" => {
                self.ended = true;
                out.push(StreamEvent::MessageEnd {
                    stop_reason: self.stop_reason.unwrap_or(StopReason::EndTurn),
                });
            }
            "error" => {
                let message = value["error"]["message"]
                    .as_str()
                    .unwrap_or("unknown stream error");
                return Err(LlmError::Protocol(message.to_string()));
            }
            _ => {}
        }
        Ok(out)
    }
}

fn string_field(value: &Value, key: &str) -> Result<String, LlmError> {
    value[key]
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| LlmError::Protocol(format!("missing string field `{key}`")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assembles_text_and_tool_use_from_sse() {
        let raw = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\"}}\n\n",
            "event: content_block_start\n",
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hel\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"lo\"}}\n\n",
            "event: content_block_stop\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "event: content_block_start\n",
            "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"bash\",\"input\":{}}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"comm\"}}\n\n",
            "event: ping\n",
            "data: {\"type\":\"ping\"}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"and\\\":\\\"ls\\\"}\"}}\n\n",
            "event: content_block_stop\n",
            "data: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
            "event: message_delta\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":9}}\n\n",
            "event: message_stop\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        );

        // Feed in awkward chunk sizes to exercise buffering.
        let mut buffer = String::new();
        let mut assembler = BlockAssembler::default();
        let mut events = Vec::new();
        for chunk in raw.as_bytes().chunks(17) {
            buffer.push_str(std::str::from_utf8(chunk).unwrap());
            for payload in drain_sse_payloads(&mut buffer) {
                events.extend(assembler.feed(&payload).unwrap());
            }
        }

        assert_eq!(
            events,
            vec![
                StreamEvent::TextDelta("Hel".into()),
                StreamEvent::TextDelta("lo".into()),
                StreamEvent::ToolUse {
                    id: "toolu_1".into(),
                    name: "bash".into(),
                    input: serde_json::json!({"command": "ls"}),
                },
                StreamEvent::MessageEnd {
                    stop_reason: StopReason::ToolUse
                },
            ]
        );
        assert!(buffer.is_empty());
    }

    #[test]
    fn stream_error_event_is_reported() {
        let mut assembler = BlockAssembler::default();
        let err = assembler
            .feed(r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#)
            .unwrap_err();
        assert!(matches!(err, LlmError::Protocol(m) if m == "Overloaded"));
    }
}
