//! A conversation that outlives one turn: the plaintext history, the
//! redaction map that makes it safe to send, and the bookkeeping the REPL
//! and resume need. Persistence lives in [`crate::session::store`].

pub mod store;

use std::path::{Path, PathBuf};

use airlok_llm::Message;
use serde::{Deserialize, Serialize};

use crate::config::{Config, ConfigFile};
use crate::redact::RedactionMap;

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

    pub fn touch(&mut self) {
        self.updated_at = now_rfc3339();
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
