//! The airlock. Text leaving the machine passes through [`Redactor::redact`];
//! text coming back passes through [`Redactor::rehydrate`] for files and
//! commands, and through [`for_display`] for the terminal.
//!
//! Entries have a [`Class`]. `Rehydrate` entries are secrets found in the
//! user's files and tool output: they must be restored into files and
//! commands or edits would break, and they are shown masked in the terminal
//! unless configured otherwise. `RedactOnly` entries, such as the provider
//! API key, are never restored anywhere.
//!
//! TODO(stage N): user-defined patterns, PII and hostname detectors, and
//! per-project allow lists.

pub mod secrets;

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

pub use secrets::SecretRedactor;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Class {
    /// Restored into files and commands; masked in the terminal by default.
    Rehydrate,
    /// Never restored. Shown as `[redacted: <kind>]`; refused in tool arguments.
    RedactOnly,
}

impl Class {
    pub fn as_str(self) -> &'static str {
        match self {
            Class::Rehydrate => "rehydrate",
            Class::RedactOnly => "redact-only",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entry {
    pub value: String,
    /// What the value is, such as "anthropic api key", for display.
    pub kind: String,
    pub class: Class,
}

/// Placeholder (for example `<<SECRET_1>>`) to what it replaced.
pub type RedactionMap = BTreeMap<String, Entry>;

pub trait Redactor: Send {
    /// Replaces sensitive spans with placeholders. The same value always maps
    /// to the same placeholder within one redactor, so the returned map is the
    /// cumulative map for every call so far.
    fn redact(&mut self, input: &str) -> (String, RedactionMap);

    /// Puts `Rehydrate` values back in place of their placeholders, for
    /// files and commands. `RedactOnly` placeholders are left untouched.
    fn rehydrate(&self, input: &str, map: &RedactionMap) -> String {
        let mut out = input.to_string();
        for (placeholder, entry) in map {
            if entry.class == Class::Rehydrate && out.contains(placeholder) {
                out = out.replace(placeholder, &entry.value);
            }
        }
        out
    }
}

/// Text for the terminal. `RedactOnly` placeholders become
/// `[redacted: <kind>]`; `Rehydrate` ones become the value when
/// `show_secrets` is set and a mask otherwise.
pub fn for_display(input: &str, map: &RedactionMap, show_secrets: bool) -> String {
    let mut out = input.to_string();
    for (placeholder, entry) in map {
        if !out.contains(placeholder) {
            continue;
        }
        let replacement = match entry.class {
            Class::RedactOnly => format!("[redacted: {}]", entry.kind),
            Class::Rehydrate if show_secrets => entry.value.clone(),
            Class::Rehydrate => mask(&entry.value),
        };
        out = out.replace(placeholder, &replacement);
    }
    out
}

/// Puts placeholders back in place of the values they stand for. The
/// inverse of [`Redactor::rehydrate`], for a destination that must not
/// see the secrets: an MCP server, by default. Longer values go first, so
/// one value that contains another cannot be half-replaced.
pub fn dehydrate(input: &str, map: &RedactionMap) -> String {
    let mut entries: Vec<(&String, &Entry)> = map
        .iter()
        .filter(|(_, entry)| entry.class == Class::Rehydrate)
        .collect();
    entries.sort_by_key(|(_, entry)| std::cmp::Reverse(entry.value.len()));
    let mut out = input.to_string();
    for (placeholder, entry) in entries {
        if !entry.value.is_empty() && out.contains(&entry.value) {
            out = out.replace(&entry.value, placeholder);
        }
    }
    out
}

/// First four characters and the length, for example `sk-a… (49 chars)`.
pub fn mask(value: &str) -> String {
    let shown: String = value.chars().take(4).collect();
    format!("{shown}… ({} chars)", value.chars().count())
}

/// The `RedactOnly` entries whose placeholder appears in `input`.
pub fn redact_only_in<'a>(input: &str, map: &'a RedactionMap) -> Vec<(&'a str, &'a Entry)> {
    map.iter()
        .filter(|(placeholder, entry)| {
            entry.class == Class::RedactOnly && input.contains(placeholder.as_str())
        })
        .map(|(placeholder, entry)| (placeholder.as_str(), entry))
        .collect()
}

/// Longest placeholder we will ever hold back while streaming. Anything held
/// past this cannot be one of ours and is released.
const MAX_PLACEHOLDER_LEN: usize = 32;

/// Splits streamed text into (emit now, hold back) so a placeholder that is
/// still arriving is not shown half-substituted.
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
    use super::*;

    fn map() -> RedactionMap {
        let mut map = RedactionMap::new();
        map.insert(
            "<<SECRET_1>>".into(),
            Entry {
                value: "0123456789abcdef".into(),
                kind: "the provider API key".into(),
                class: Class::RedactOnly,
            },
        );
        map.insert(
            "<<SECRET_2>>".into(),
            Entry {
                value: "sk-ant-api03-xyz".into(),
                kind: "anthropic api key".into(),
                class: Class::Rehydrate,
            },
        );
        map
    }

    struct Plain;
    impl Redactor for Plain {
        fn redact(&mut self, input: &str) -> (String, RedactionMap) {
            (input.to_string(), map())
        }
    }

    #[test]
    fn rehydrate_restores_only_rehydrate_entries() {
        let text = "a <<SECRET_1>> b <<SECRET_2>> c";
        assert_eq!(
            Plain.rehydrate(text, &map()),
            "a <<SECRET_1>> b sk-ant-api03-xyz c"
        );
    }

    #[test]
    fn dehydrate_is_the_inverse_of_rehydrate() {
        let map = map();
        let plain = "a <<SECRET_1>> b sk-ant-api03-xyz c";
        assert_eq!(dehydrate(plain, &map), "a <<SECRET_1>> b <<SECRET_2>> c");
        // Round trip: what the model sent, restored and put back.
        let restored = Plain.rehydrate("<<SECRET_2>>", &map);
        assert_eq!(dehydrate(&restored, &map), "<<SECRET_2>>");
        // A redact-only value is not in play: it is never restored, so it
        // cannot appear in text to put back.
        assert_eq!(dehydrate("0123456789abcdef", &map), "0123456789abcdef");
    }

    #[test]
    fn display_masks_or_shows_and_never_reveals_redact_only() {
        let text = "a <<SECRET_1>> b <<SECRET_2>> c";
        assert_eq!(
            for_display(text, &map(), false),
            "a [redacted: the provider API key] b sk-a… (16 chars) c"
        );
        assert_eq!(
            for_display(text, &map(), true),
            "a [redacted: the provider API key] b sk-ant-api03-xyz c"
        );
    }

    #[test]
    fn finds_redact_only_placeholders() {
        let m = map();
        let found = redact_only_in("x <<SECRET_2>> y <<SECRET_1>>", &m);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].0, "<<SECRET_1>>");
        assert!(redact_only_in("nothing", &m).is_empty());
    }

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
