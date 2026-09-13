//! Wire-level types shared by every provider. They mirror the Anthropic
//! Messages API shapes closely enough to serialise directly.

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        is_error: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
}

impl Message {
    pub fn user_text(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }

    pub fn assistant(content: Vec<ContentBlock>) -> Self {
        Self {
            role: Role::Assistant,
            content,
        }
    }

    /// Tool results are sent back to the model as a user turn.
    pub fn tool_results(content: Vec<ContentBlock>) -> Self {
        Self {
            role: Role::User,
            content,
        }
    }
}

/// A tool the model may call, described the way the API expects it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

/// Which OpenAI HTTP API to speak. Chat Completions is the default; the
/// Responses API is what newer reasoning deployments expect.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Api {
    #[default]
    Chat,
    Responses,
}

impl std::fmt::Display for Api {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Api::Chat => "chat",
            Api::Responses => "responses",
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Request {
    pub model: String,
    pub max_tokens: u32,
    pub system: String,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolSpec>,
    /// Sent as `reasoning_effort` by the openai provider, and only when set.
    /// The Anthropic provider ignores it.
    pub reasoning_effort: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    StopSequence,
    Refusal,
    #[serde(other)]
    Other,
}

/// Token counts reported by the provider for one request.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    /// Everything the model read, including any cached prefix.
    pub input_tokens: u64,
    pub output_tokens: u64,
}

/// A complete assistant turn, assembled from the stream.
#[derive(Debug, Clone, PartialEq)]
pub struct Response {
    pub content: Vec<ContentBlock>,
    pub stop_reason: StopReason,
    /// `None` when the provider reported no usage; callers estimate.
    pub usage: Option<Usage>,
}

impl Request {
    /// Rough size of the request, chars/4, for when the provider reports no
    /// usage. Counts what the wire body carries, not its JSON framing.
    pub fn estimated_tokens(&self) -> u64 {
        let mut chars = self.system.len();
        for message in &self.messages {
            for block in &message.content {
                chars += match block {
                    ContentBlock::Text { text } => text.len(),
                    ContentBlock::ToolUse { name, input, .. } => {
                        name.len() + input.to_string().len()
                    }
                    ContentBlock::ToolResult { content, .. } => content.len(),
                };
            }
        }
        for tool in &self.tools {
            chars += tool.name.len() + tool.description.len() + tool.input_schema.to_string().len();
        }
        (chars / 4) as u64
    }
}

/// One step of a streamed reply.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamEvent {
    /// A fragment of assistant text, in order.
    TextDelta(String),
    /// A complete tool call. Providers buffer partial JSON and emit this once.
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    /// Token counts, when the provider reports them. At most once per turn,
    /// before `MessageEnd`.
    Usage(Usage),
    /// The turn is over. Always the final event.
    MessageEnd { stop_reason: StopReason },
}
