//! A conversation that outlives one turn: the plaintext history, the
//! redaction map that makes it safe to send, and the bookkeeping the REPL
//! and resume need. Persistence lives in [`crate::session::store`].

pub mod store;

use std::path::{Path, PathBuf};

use airlok_llm::{ContentBlock, Message};
use serde::{Deserialize, Serialize};

use crate::config::{Config, ConfigFile};
use crate::redact::{Class, RedactionMap};

pub use store::{SessionStore, Summary};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub cwd: PathBuf,
    /// RFC 3339, UTC.
    pub created_at: String,
    pub updated_at: String,
    pub provider: String,
    pub model: String,
    /// The effective configuration when the session started.
    pub config: ConfigFile,
    /// Plaintext history. Never leaves the machine as is.
    pub messages: Vec<Message>,
    /// Placeholder to value. The only place real values live on disk.
    pub redactions: RedactionMap,
    pub usage: Usage,
    pub compactions: Vec<Compaction>,
    /// Every time this session was resumed, RFC 3339.
    pub resumed_at: Vec<String>,
    /// The last turn was cut short by the user.
    pub interrupted: bool,
}

/// Token accounting for the session.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    /// Sum of input tokens over every request.
    pub input_tokens: u64,
    /// Sum of output tokens over every request.
    pub output_tokens: u64,
    /// Input tokens of the most recent request: the current context size.
    pub context_tokens: u64,
    /// True when `context_tokens` came from a chars/4 estimate because the
    /// provider reported no usage.
    pub estimated: bool,
}

impl Usage {
    /// Adds one request's usage. Without a provider report, the input side
    /// is `estimate` (chars/4 of the request) and the output side chars/4 of
    /// the reply, and the whole session is marked estimated.
    pub fn record(
        &mut self,
        reported: Option<airlok_llm::Usage>,
        estimate: u64,
        reply: &[ContentBlock],
    ) {
        match reported {
            Some(usage) => {
                self.input_tokens += usage.input_tokens;
                self.output_tokens += usage.output_tokens;
                self.context_tokens = usage.input_tokens;
            }
            None => {
                let output: usize = reply
                    .iter()
                    .map(|b| match b {
                        ContentBlock::Text { text } => text.len(),
                        ContentBlock::ToolUse { name, input, .. } => {
                            name.len() + input.to_string().len()
                        }
                        ContentBlock::ToolResult { content, .. } => content.len(),
                    })
                    .sum();
                self.input_tokens += estimate;
                self.output_tokens += (output / 4) as u64;
                self.context_tokens = estimate;
                self.estimated = true;
            }
        }
    }
}

/// One compaction, kept so a resumed session shows the summary rather than
/// the pruned history.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Compaction {
    pub at: String,
    pub before_tokens: u64,
    pub after_tokens: u64,
    pub summary: String,
}

impl Session {
    pub fn new(cwd: &Path, config: &Config) -> Self {
        let now = now_rfc3339();
        Self {
            id: new_id(),
            cwd: cwd.to_path_buf(),
            created_at: now.clone(),
            updated_at: now,
            provider: config.provider.name.as_str().to_string(),
            model: config.provider.model.clone(),
            config: config.to_file(),
            messages: Vec::new(),
            redactions: RedactionMap::new(),
            usage: Usage::default(),
            compactions: Vec::new(),
            resumed_at: Vec::new(),
            interrupted: false,
        }
    }

    /// The first thing the user asked, for listings.
    pub fn first_prompt(&self) -> Option<&str> {
        self.messages.iter().find_map(|m| match m.content.first() {
            Some(airlok_llm::ContentBlock::Text { text }) if m.role == airlok_llm::Role::User => {
                Some(text.as_str())
            }
            _ => None,
        })
    }

    /// User prompts so far (tool-result messages are not turns).
    pub fn turns(&self) -> usize {
        self.messages
            .iter()
            .filter(|m| {
                m.role == airlok_llm::Role::User
                    && matches!(
                        m.content.first(),
                        Some(airlok_llm::ContentBlock::Text { .. })
                    )
            })
            .count()
    }

    /// Index of the message that starts each turn: a user message whose
    /// content is text, not tool results.
    pub fn turn_starts(&self) -> Vec<usize> {
        self.messages
            .iter()
            .enumerate()
            .filter(|(_, m)| {
                m.role == airlok_llm::Role::User
                    && matches!(m.content.first(), Some(ContentBlock::Text { .. }))
            })
            .map(|(i, _)| i)
            .collect()
    }

    pub fn touch(&mut self) {
        self.updated_at = now_rfc3339();
    }

    /// Records a resume. The agent adds a system note for the latest one.
    pub fn resume(&mut self) -> &str {
        self.resumed_at.push(now_rfc3339());
        self.resumed_at
            .last()
            .map(String::as_str)
            .unwrap_or_default()
    }

    /// What goes on disk: the same session with redact-only values (the
    /// provider key) blanked. The placeholder and its class survive, so
    /// history that mentions it still displays as redacted after a resume.
    pub fn for_disk(&self) -> Session {
        let mut copy = self.clone();
        for entry in copy.redactions.values_mut() {
            if entry.class == Class::RedactOnly {
                entry.value.clear();
            }
        }
        copy
    }
}

pub fn now_rfc3339() -> String {
    time::OffsetDateTime::now_utc()
        .replace_nanosecond(0)
        .unwrap_or_else(|_| time::OffsetDateTime::now_utc())
        .format(&time::format_description::well_known::Rfc3339)
        .unwrap_or_else(|_| "1970-01-01T00:00:00Z".to_string())
}

/// Short, unique enough for one machine: seconds, pid, and a counter mixed
/// through FNV-1a, in hex.
fn new_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let mix = format!(
        "{nanos}-{}-{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    );
    format!("{:08x}", fnv1a(mix.as_bytes()) as u32)
}

/// Stable across Rust versions, unlike `DefaultHasher`.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}
