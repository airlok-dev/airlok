//! Anthropic Messages API with streaming.
//!
//! Wire reference: `POST /v1/messages` with `"stream": true` returns
//! server-sent events whose `data:` payloads carry a `type` field
//! (`message_start`, `content_block_start`, `content_block_delta`,
//! `content_block_stop`, `message_delta`, `message_stop`, `ping`, `error`).

use futures::stream::{self, BoxStream, StreamExt, TryStreamExt};
use serde::Serialize;
use serde_json::Value;
use tracing::{debug, trace};

use crate::sse;
use crate::types::{Message, Request, StopReason, StreamEvent, ToolSpec, Usage};
use crate::{LlmError, Provider};

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const API_VERSION: &str = "2023-06-01";
pub const DEFAULT_MODEL: &str = "claude-sonnet-4-6";
const API_KEY_VAR: &str = "ANTHROPIC_API_KEY";

pub struct Anthropic {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
}

impl Anthropic {
    /// Reads the key from `ANTHROPIC_API_KEY`.
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

    pub fn with_base_url(mut self, base_url: &str) -> Self {
        self.base_url = base_url.trim_end_matches('/').to_string();
        self
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

impl<'a> WireRequest<'a> {
    fn new(request: &'a Request) -> Self {
        Self {
            model: &request.model,
            max_tokens: request.max_tokens,
            stream: true,
            system: &request.system,
            messages: &request.messages,
            tools: &request.tools,
        }
    }
}

impl Provider for Anthropic {
    fn stream(&self, request: Request) -> BoxStream<'_, Result<StreamEvent, LlmError>> {
        let open = async move {
            let body = WireRequest::new(&request);
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
            let bytes = sse::body_or_error(response).await?;
            Ok::<_, LlmError>(sse::events(bytes, BlockAssembler::default()))
        };
        stream::once(open).try_flatten().boxed()
    }
}

/// Reassembles content blocks from the per-block delta events.
#[derive(Default)]
struct BlockAssembler {
    current: Option<PartialBlock>,
    stop_reason: Option<StopReason>,
    usage: Option<Usage>,
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

impl sse::Assembler for BlockAssembler {
    fn ended(&self) -> bool {
        self.ended
    }

    fn feed(&mut self, payload: &str) -> Result<Vec<StreamEvent>, LlmError> {
        let value: Value = serde_json::from_str(payload)?;
        let kind = value["type"].as_str().unwrap_or_default();
        trace!(kind, "sse event");
        let mut out = Vec::new();
        match kind {
            "message_start" => {
                // Input tokens are only reported here. Cached prefixes are
                // counted separately but the model still reads them.
                let usage = &value["message"]["usage"];
                let input = [
                    "input_tokens",
                    "cache_creation_input_tokens",
                    "cache_read_input_tokens",
                ]
                .iter()
                .filter_map(|k| usage[k].as_u64())
                .sum();
                if !usage.is_null() {
                    self.usage = Some(Usage {
                        input_tokens: input,
                        output_tokens: usage["output_tokens"].as_u64().unwrap_or(0),
                    });
                }
            }
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
                    out.push(StreamEvent::ToolUse {
                        id,
                        name,
                        input: crate::parse_arguments(&json)?,
                    });
                }
            }
            "message_delta" => {
                if let Some(reason) = value["delta"]["stop_reason"].as_str() {
                    self.stop_reason =
                        serde_json::from_value(Value::String(reason.to_string())).ok();
                }
                // Cumulative output count; the last delta wins.
                if let Some(output) = value["usage"]["output_tokens"].as_u64() {
                    self.usage.get_or_insert_with(Usage::default).output_tokens = output;
                }
            }
            "message_stop" => {
                self.ended = true;
                if let Some(usage) = self.usage.take() {
                    out.push(StreamEvent::Usage(usage));
                }
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
    use crate::sse::Assembler;

    #[test]
    fn assembles_text_and_tool_use_from_sse() {
        let raw = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"usage\":{\"input_tokens\":20,\"cache_read_input_tokens\":5,\"output_tokens\":1}}}\n\n",
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

        let mut assembler = BlockAssembler::default();
        let events = sse::assemble_chunked(raw, &mut assembler);

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
                StreamEvent::Usage(Usage {
                    input_tokens: 25,
                    output_tokens: 9
                }),
                StreamEvent::MessageEnd {
                    stop_reason: StopReason::ToolUse
                },
            ]
        );
        assert!(assembler.ended());
    }

    #[test]
    fn an_empty_tool_list_is_left_out() {
        let mut request = Request {
            model: "m".into(),
            max_tokens: 10,
            system: String::new(),
            messages: vec![Message::user_text("hi")],
            tools: Vec::new(),
            reasoning_effort: None,
        };
        let body = serde_json::to_value(WireRequest::new(&request)).unwrap();
        assert!(body.get("tools").is_none(), "{body}");
        request.tools.push(ToolSpec {
            name: "read_file".into(),
            description: "reads".into(),
            input_schema: serde_json::json!({"type": "object"}),
        });
        let body = serde_json::to_value(WireRequest::new(&request)).unwrap();
        assert_eq!(body["tools"][0]["name"], "read_file");
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
