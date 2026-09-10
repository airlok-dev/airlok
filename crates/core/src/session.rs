//! A persisted conversation.
//!
//! 0.1 keeps history in memory for the duration of one `Agent::run`. This is
//! the shape that will be saved and resumed later.
//!
//! TODO(stage N): write to `~/.airlok/sessions/<id>.json` after every turn,
//! add `airlok --resume <id>`, and store the redaction map alongside so a
//! resumed session rehydrates the same placeholders.

use std::path::PathBuf;

use airlok_llm::Message;

use crate::redact::RedactionMap;

#[derive(Debug, Clone)]
pub struct Session {
    pub id: String,
    pub cwd: PathBuf,
    pub messages: Vec<Message>,
    pub redactions: RedactionMap,
}
