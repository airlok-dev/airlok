//! Model providers for airlok.
//!
//! This crate knows nothing about the agent. It defines the message types
//! exchanged with a model, a [`Provider`] trait that streams a reply, and one
//! implementation for the Anthropic Messages API.

pub mod anthropic;
pub mod types;

use futures::stream::BoxStream;

pub use anthropic::Anthropic;
pub use types::{
    ContentBlock, Message, Request, Response, Role, StopReason, StreamEvent, ToolSpec,
};

/// Errors produced while talking to a model.
#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    #[error("ANTHROPIC_API_KEY is not set")]
    MissingApiKey,
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
