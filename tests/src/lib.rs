//! Test support: a scripted provider that records what it is sent, an
//! output sink that records what the user would see, and a temp dir.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use airlok_core::config::{ProviderConfig, ProviderName};
use airlok_core::redact::SecretRedactor;
use airlok_core::redact::{RedactionMap, Redactor};
use airlok_core::repl::{Backend, Line, LineSource, Switch};
use airlok_core::tools::ToolRegistry;
use airlok_core::{Agent, Config, Confirmation, Decision, Output};
use airlok_llm::{LlmError, Provider, Request, StopReason, StreamEvent};
use futures::stream::{self, BoxStream, StreamExt};
use serde_json::Value;

/// Replays one scripted event list per request and records every request.
/// A turn that does not end with `MessageEnd` hangs after its last event,
/// which is how tests stand in for a model still streaming.
#[derive(Default)]
pub struct MockProvider {
    scripts: Mutex<VecDeque<Vec<StreamEvent>>>,
    requests: Mutex<Vec<Request>>,
    /// When set, every request fails with this status and body.
    fails: Mutex<Option<(u16, String)>>,
}

impl MockProvider {
    pub fn scripted(turns: Vec<Vec<StreamEvent>>) -> Arc<Self> {
        Arc::new(Self {
            scripts: Mutex::new(turns.into()),
            requests: Mutex::new(Vec::new()),
            fails: Mutex::new(None),
        })
    }

    /// Answers every request with an API error, as a provider does when
    /// the deployment is missing or the key is wrong.
    pub fn failing(status: u16, body: &str) -> Arc<Self> {
        Arc::new(Self {
            scripts: Mutex::new(VecDeque::new()),
            requests: Mutex::new(Vec::new()),
            fails: Mutex::new(Some((status, body.to_string()))),
        })
    }

    pub fn requests(&self) -> Vec<Request> {
        self.requests.lock().unwrap().clone()
    }
}

impl Provider for MockProvider {
    fn stream(&self, request: Request) -> BoxStream<'_, Result<StreamEvent, LlmError>> {
        self.requests.lock().unwrap().push(request);
        if let Some((status, body)) = self.fails.lock().unwrap().clone() {
            return stream::iter(vec![Err(LlmError::Api { status, body })]).boxed();
        }
        let events = self
            .scripts
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| reply("(script exhausted)"));
        let hangs = !matches!(events.last(), Some(StreamEvent::MessageEnd { .. }));
        let scripted = stream::iter(events.into_iter().map(Ok));
        if hangs {
            scripted.chain(stream::pending()).boxed()
        } else {
            scripted.boxed()
        }
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
/// Text that never finishes: the stream hangs after it, until interrupted.
pub fn partial(text: &str) -> Vec<StreamEvent> {
    vec![StreamEvent::TextDelta(text.to_string())]
}

/// Feeds the REPL a fixed script. Runs out as `Eof`.
pub struct ScriptedLines {
    pub lines: VecDeque<Line>,
    pub prompts: Vec<String>,
}

impl ScriptedLines {
    pub fn new(lines: Vec<Line>) -> Self {
        Self {
            lines: lines.into(),
            prompts: Vec::new(),
        }
    }

    pub fn typed(lines: &[&str]) -> Self {
        Self::new(lines.iter().map(|l| Line::Text(l.to_string())).collect())
    }
}

impl LineSource for ScriptedLines {
    fn read_line(&mut self, prompt: &str) -> Line {
        self.prompts.push(prompt.to_string());
        self.lines.pop_front().unwrap_or(Line::Eof)
    }
}

/// Every `choose` call the backend saw: the title, and the choices
/// it was offered.
pub type Asked = Arc<Mutex<Vec<(String, Vec<String>)>>>;

/// A REPL backend whose providers are scripted. A provider with no entry
/// fails the switch with "no key configured".
#[derive(Default)]
pub struct TestBackend {
    pub switches: std::collections::HashMap<&'static str, Result<Arc<MockProvider>, String>>,
    /// One answer per `choose` call, in order. `None` cancels that one,
    /// and an empty queue cancels everything, as a run with no terminal does.
    pub choices: VecDeque<Option<String>>,
    /// Every question the pickers asked: the title and what was offered.
    pub asked: Asked,
    /// Model ids saved sessions used, by provider.
    pub saved_models: std::collections::HashMap<&'static str, Vec<String>>,
    /// `reasoning_effort` the config files give a model, by model id, as
    /// it stood before any session override.
    pub efforts: std::collections::HashMap<&'static str, String>,
    /// What `/status` should report as the configuration files in effect.
    pub config_files: Vec<String>,
    /// What `/status` should report as the key's source, never its value.
    pub key_source: Option<String>,
}

impl TestBackend {
    /// What the pickers were asked, for a test to look at afterwards.
    pub fn questions(&self) -> Asked {
        self.asked.clone()
    }
}

impl Backend for TestBackend {
    fn fresh_redactor(&mut self) -> Box<dyn Redactor> {
        Box::new(SecretRedactor::new())
    }

    fn choose(&mut self, title: &str, choices: &[String]) -> Option<String> {
        self.asked
            .lock()
            .unwrap()
            .push((title.to_string(), choices.to_vec()));
        self.choices.pop_front().flatten()
    }

    fn startup_effort(&mut self, model: &str) -> Option<String> {
        self.efforts.get(model).cloned()
    }

    fn config_files(&mut self) -> Vec<String> {
        self.config_files.clone()
    }

    fn key_source(&mut self) -> Option<String> {
        self.key_source.clone()
    }

    fn known_models(&mut self, provider: ProviderName) -> Vec<String> {
        self.saved_models
            .get(provider.as_str())
            .cloned()
            .unwrap_or_default()
    }

    fn switch(&mut self, name: ProviderName, seed: &RedactionMap) -> Result<Switch, String> {
        let provider = self
            .switches
            .get(name.as_str())
            .cloned()
            .unwrap_or_else(|| Err("no key configured".into()))?;
        Ok(Switch {
            config: ProviderConfig {
                name,
                model: name.default_model().to_string(),
                base_url: None,
                api_key_env: None,
                api_key_cmd: None,
                context_window: airlok_core::config::DEFAULT_CONTEXT_WINDOW,
            },
            provider,
            redactor: Box::new(SecretRedactor::new().with_map(seed)),
        })
    }
}

/// Prepends a provider usage report to a scripted turn.
pub fn with_usage(
    input_tokens: u64,
    output_tokens: u64,
    mut turn: Vec<StreamEvent>,
) -> Vec<StreamEvent> {
    turn.insert(
        0,
        StreamEvent::Usage(airlok_llm::Usage {
            input_tokens,
            output_tokens,
        }),
    );
    turn
}

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
    ToolCall {
        name: String,
        summary: String,
    },
    ConfirmWrite {
        path: PathBuf,
        diff: String,
    },
    ConfirmCommand {
        command: String,
    },
    ConfirmMcpProject {
        servers: Vec<(String, String)>,
    },
    ConfirmMcp {
        server: String,
        tool: String,
        arguments: String,
        paths: Vec<String>,
    },
    Status(String),
    EndTurn,
}

/// Records everything shown and answers confirmations from a script.
/// Unscripted confirmations are approved.
#[derive(Default)]
pub struct RecordingOutput {
    pub events: Vec<Shown>,
    pub decisions: VecDeque<Decision>,
    /// Every `tokens` report, in order. Kept out of `events` so tests
    /// that compare the whole transcript are not about progress reports.
    pub tokens: Vec<u64>,
    /// How many requests started streaming.
    pub thinking: usize,
}

impl RecordingOutput {
    pub fn answering(decisions: Vec<Decision>) -> Self {
        Self {
            decisions: decisions.into(),
            ..Self::default()
        }
    }

    /// Every status line, in order.
    pub fn statuses(&self) -> Vec<String> {
        self.events
            .iter()
            .filter_map(|e| match e {
                Shown::Status(t) => Some(t.clone()),
                _ => None,
            })
            .collect()
    }

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

    fn status(&mut self, line: &str) {
        self.events.push(Shown::Status(line.to_string()));
    }

    fn end_turn(&mut self) {
        self.events.push(Shown::EndTurn);
    }

    fn thinking(&mut self) {
        self.thinking += 1;
    }

    fn tokens(&mut self, used: u64) {
        self.tokens.push(used);
    }

    fn confirm(&mut self, request: &Confirmation<'_>) -> Decision {
        self.events.push(match request {
            Confirmation::Write { path, diff } => Shown::ConfirmWrite {
                path: path.to_path_buf(),
                diff: diff.to_string(),
            },
            Confirmation::Command { command } => Shown::ConfirmCommand {
                command: command.to_string(),
            },
            Confirmation::McpProject { servers } => Shown::ConfirmMcpProject {
                servers: servers.to_vec(),
            },
            Confirmation::Mcp {
                server,
                tool,
                arguments,
                paths,
                ..
            } => Shown::ConfirmMcp {
                server: server.to_string(),
                tool: tool.to_string(),
                arguments: arguments.to_string(),
                paths: paths.to_vec(),
            },
        });
        self.decisions.pop_front().unwrap_or(Decision::Approve)
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
        // Clocks can tick in microseconds, so two threads may read the
        // same time; the counter keeps their directories apart.
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("airlok-{label}-{}-{nanos}-{n}", std::process::id()));
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
    agent_with(provider, cwd, SecretRedactor::new())
}

/// An agent with the default tools and the given redactor.
pub fn agent_with(provider: Arc<MockProvider>, cwd: &Path, redactor: SecretRedactor) -> Agent {
    agent_configured(provider, Config::new(cwd.to_path_buf()), redactor)
}

/// An agent on a configuration the test built, such as one with `[[mcp]]`
/// servers in it.
pub fn agent_configured(
    provider: Arc<MockProvider>,
    config: Config,
    redactor: SecretRedactor,
) -> Agent {
    let tools = ToolRegistry::defaults(&config.cwd, config.agent.bash_timeout);
    Agent::new(provider, tools, Box::new(redactor), config)
}
