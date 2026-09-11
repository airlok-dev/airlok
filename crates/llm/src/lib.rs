//! Model providers for airlok.
//!
//! This crate knows nothing about the agent. It defines the message types
//! exchanged with a model, a [`Provider`] trait that streams a reply, and
//! implementations for the Anthropic Messages API and the OpenAI Chat
//! Completions API.

pub mod anthropic;
pub mod openai;
mod sse;
pub mod types;

use futures::stream::BoxStream;

pub use anthropic::Anthropic;
pub use openai::{Auth, OpenAi};
pub use types::{
    ContentBlock, Message, Request, Response, Role, StopReason, StreamEvent, ToolSpec, Usage,
};

/// Errors produced while talking to a model.
#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    #[error("{0} is not set")]
    MissingApiKey(&'static str),
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),
    #[error("API returned HTTP {status}: {body}")]
    Api { status: u16, body: String },
    #[error("malformed stream: {0}")]
    Protocol(String),
    #[error("invalid JSON: {0}")]
    Json(#[from] serde_json::Error),
}

/// A model that answers a [`Request`] with a stream of [`StreamEvent`]s.
///
/// The stream must end with exactly one [`StreamEvent::MessageEnd`].
pub trait Provider: Send + Sync {
    fn stream(&self, request: Request) -> BoxStream<'_, Result<StreamEvent, LlmError>>;
}

fn key_from_env(var: &'static str) -> Result<String, LlmError> {
    std::env::var(var)
        .ok()
        .filter(|k| !k.trim().is_empty())
        .ok_or(LlmError::MissingApiKey(var))
}

/// Parses a streamed tool-call argument string. Empty means no arguments.
fn parse_arguments(json: &str) -> Result<serde_json::Value, LlmError> {
    if json.trim().is_empty() {
        Ok(serde_json::Value::Object(Default::default()))
    } else {
        Ok(serde_json::from_str(json)?)
    }
}
