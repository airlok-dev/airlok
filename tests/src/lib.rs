//! Test support: a scripted provider that records what it is sent, an
//! output sink that records what the user would see, and a temp dir.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use airlok_core::redact::SecretRedactor;
use airlok_core::tools::ToolRegistry;
use airlok_core::{Agent, Config, Output};
use airlok_llm::{LlmError, Provider, Request, StopReason, StreamEvent};
use futures::stream::{self, BoxStream, StreamExt};
use serde_json::Value;

/// Replays one scripted event list per request and records every request.
#[derive(Default)]
pub struct MockProvider {
    scripts: Mutex<VecDeque<Vec<StreamEvent>>>,
    requests: Mutex<Vec<Request>>,
}

impl MockProvider {
    pub fn scripted(turns: Vec<Vec<StreamEvent>>) -> Arc<Self> {
        Arc::new(Self {
            scripts: Mutex::new(turns.into()),
            requests: Mutex::new(Vec::new()),
        })
    }

    pub fn requests(&self) -> Vec<Request> {
        self.requests.lock().unwrap().clone()
    }
}

impl Provider for MockProvider {
    fn stream(&self, request: Request) -> BoxStream<'_, Result<StreamEvent, LlmError>> {
        self.requests.lock().unwrap().push(request);
        let events = self
            .scripts
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| reply("(script exhausted)"));
        stream::iter(events.into_iter().map(Ok)).boxed()
    }
}

/// A final text turn.
pub fn reply(text: &str) -> Vec<StreamEvent> {
    vec![
        StreamEvent::TextDelta(text.to_string()),
        StreamEvent::MessageEnd {
            stop_reason: StopReason::EndTurn,
        },
    ]
}

/// A turn that calls one tool.
pub fn tool_call(id: &str, name: &str, input: Value) -> Vec<StreamEvent> {
    vec![
        StreamEvent::ToolUse {
            id: id.to_string(),
            name: name.to_string(),
            input,
        },
        StreamEvent::MessageEnd {
            stop_reason: StopReason::ToolUse,
        },
    ]
}

#[derive(Debug, Clone, PartialEq)]
pub enum Shown {
    Text(String),
    ToolCall { name: String, summary: String },
}

#[derive(Default)]
pub struct RecordingOutput {
    pub events: Vec<Shown>,
}

impl RecordingOutput {
    /// All streamed text joined, as the user would read it.
    pub fn text(&self) -> String {
        self.events
            .iter()
            .filter_map(|e| match e {
                Shown::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect()
    }
}

impl Output for RecordingOutput {
    fn text(&mut self, chunk: &str) {
        self.events.push(Shown::Text(chunk.to_string()));
    }

    fn tool_call(&mut self, name: &str, summary: &str) {
        self.events.push(Shown::ToolCall {
            name: name.to_string(),
            summary: summary.to_string(),
        });
    }
}

/// An in-memory log sink for `tracing_subscriber::fmt().with_writer(...)`.
#[derive(Clone, Default)]
pub struct LogBuffer(Arc<Mutex<Vec<u8>>>);

impl LogBuffer {
    pub fn contents(&self) -> String {
        String::from_utf8_lossy(&self.0.lock().unwrap()).into_owned()
    }
}

impl std::io::Write for LogBuffer {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for LogBuffer {
    type Writer = LogBuffer;

    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

/// A fresh directory under the system temp dir, removed on drop.
pub struct TempDir(PathBuf);

impl TempDir {
    pub fn new(label: &str) -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("airlok-{label}-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// An agent with the default tools and the real secret redactor.
pub fn agent(provider: Arc<MockProvider>, cwd: &Path) -> Agent {
    let config = Config::new(cwd.to_path_buf());
    let tools = ToolRegistry::defaults(cwd, config.bash_timeout);
    Agent::new(provider, tools, Box::new(SecretRedactor::new()), config)
}
