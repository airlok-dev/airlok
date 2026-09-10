//! The airlock. Text leaving the machine passes through [`Redactor::redact`];
//! text coming back passes through [`Redactor::rehydrate`].
//!
//! TODO(stage N): user-defined patterns, PII and hostname detectors, and
//! per-project allow lists.

pub mod secrets;

use std::collections::BTreeMap;

pub use secrets::SecretRedactor;

/// Placeholder (for example `<<SECRET_1>>`) to the original value.
pub type RedactionMap = BTreeMap<String, String>;

pub trait Redactor: Send {
    /// Replaces sensitive spans with placeholders. The same value always maps
    /// to the same placeholder within one redactor, so the returned map is the
    /// cumulative map for every call so far.
    fn redact(&mut self, input: &str) -> (String, RedactionMap);

    /// Puts the original values back in place of their placeholders.
    fn rehydrate(&self, input: &str, map: &RedactionMap) -> String;

    /// What kind of value a placeholder stands for, such as
    /// "anthropic api key", for display without the value.
    fn kind_of(&self, placeholder: &str) -> Option<&str>;
}

/// Longest placeholder we will ever hold back while streaming. Anything held
/// past this cannot be one of ours and is released.
const MAX_PLACEHOLDER_LEN: usize = 32;

/// Splits streamed text into (emit now, hold back) so a placeholder that is
/// still arriving is not shown half-rehydrated.
pub fn split_incomplete_placeholder(buffer: &str) -> (&str, &str) {
    if let Some(start) = buffer.rfind("<<") {
        let tail = &buffer[start..];
        if !tail.contains(">>") && tail.len() <= MAX_PLACEHOLDER_LEN {
            return (&buffer[..start], tail);
        }
    }
    if buffer.ends_with('<') {
        let split = buffer.len() - 1;
        return (&buffer[..split], &buffer[split..]);
    }
    (buffer, "")
}

#[cfg(test)]
mod tests {
    use super::split_incomplete_placeholder as split;

    #[test]
    fn holds_back_partial_placeholder() {
        assert_eq!(split("key is <<SEC"), ("key is ", "<<SEC"));
        assert_eq!(split("key is <"), ("key is ", "<"));
        assert_eq!(
            split("key is <<SECRET_1>> ok"),
            ("key is <<SECRET_1>> ok", "")
        );
        assert_eq!(split("plain text"), ("plain text", ""));
    }

    #[test]
    fn releases_overlong_tail() {
        let long = format!("<<{}", "x".repeat(40));
        assert_eq!(split(&long), (long.as_str(), ""));
    }
}
