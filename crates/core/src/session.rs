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
    /// The effective configuration: taken when the session started and
    /// updated when `/model` or `/provider` changes it.
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
    /// What the user asked airlok to work toward, from `/goal`. Goes into
    /// the system prompt every turn and survives a resume.
    #[serde(default)]
    pub goal: Option<String>,
    /// Every file airlok wrote or edited this session, in the order they
    /// were first touched.
    #[serde(default)]
    pub writes: Vec<WriteRecord>,
}

/// One file airlok changed. The original is kept so `/diff` can show what
/// the session did, under a cap so a session file cannot grow without
/// bound; past it only the path and hashes remain.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WriteRecord {
    pub path: String,
    /// Hash of what was on disk before airlok first wrote this path.
    pub first_seen_hash: String,
    /// RFC 3339, UTC, of the most recent write.
    pub last_write: String,
    /// Whether the file existed before airlok first wrote it.
    pub existed: bool,
    /// What was there before that first write. `None` when it was too
    /// large to keep, which `/diff` says rather than showing nothing.
    pub original: Option<String>,
}

/// How much original content one session keeps, across every path.
pub const ORIGINALS_BUDGET: usize = 1_000_000;

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
    /// Input plus output tokens over every request so far.
    pub fn spent(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }

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
                        // A reply never carries one; a model returns text.
                        ContentBlock::Image { .. } => 0,
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
            goal: None,
            writes: Vec::new(),
        }
    }

    /// The first thing the user asked, for listings.
    /// Notes that airlok wrote `path`. `before` is what was on disk just
    /// before this write, or `None` when there was no file. Only the first
    /// write to a path keeps an original, since that is the state the
    /// session started from.
    pub fn record_write(&mut self, path: &Path, before: Option<String>) {
        let path = path.display().to_string();
        let now = now_rfc3339();
        if let Some(seen) = self.writes.iter_mut().find(|w| w.path == path) {
            seen.last_write = now;
            return;
        }
        let existed = before.is_some();
        let body = before.unwrap_or_default();
        let kept: usize = self
            .writes
            .iter()
            .filter_map(|w| w.original.as_ref().map(String::len))
            .sum();
        let original = (kept + body.len() <= ORIGINALS_BUDGET).then(|| body.clone());
        self.writes.push(WriteRecord {
            path,
            first_seen_hash: format!("{:016x}", fnv1a(body.as_bytes())),
            last_write: now,
            existed,
            original,
        });
    }

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
        // Hash, dimensions and format are enough for --resume to say an
        // image was sent. The bytes are never written.
        for message in &mut copy.messages {
            for block in &mut message.content {
                let ContentBlock::Image { source } = block else {
                    continue;
                };
                if let airlok_llm::ImageSource::Base64 {
                    media_type,
                    data,
                    width,
                    height,
                } = source
                {
                    *source = airlok_llm::ImageSource::Reference {
                        media_type: media_type.clone(),
                        width: *width,
                        height: *height,
                        hash: format!("{:016x}", fnv1a(data.as_bytes())),
                    };
                }
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
