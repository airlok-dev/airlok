//! Regex detection of well-known API key and token formats.

use std::collections::HashMap;

use regex::Regex;

use super::{Class, Entry, RedactionMap, Redactor};

/// (kind, pattern). If the pattern has a capture group, only group 1 is
/// redacted; otherwise the whole match is.
const PATTERNS: &[(&str, &str)] = &[
    ("anthropic api key", r"sk-ant-[A-Za-z0-9_\-]{20,}"),
    (
        "openai api key",
        r"sk-(?:proj-|svcacct-)?[A-Za-z0-9_\-]{20,}",
    ),
    ("github token", r"gh[pousr]_[A-Za-z0-9]{36,}"),
    ("github fine-grained token", r"github_pat_[A-Za-z0-9_]{22,}"),
    ("aws access key id", r"(?:AKIA|ASIA)[0-9A-Z]{16}"),
    ("slack token", r"xox[abprs]-[A-Za-z0-9-]{10,}"),
    ("google api key", r"AIza[0-9A-Za-z_\-]{35}"),
    ("stripe key", r"[sr]k_(?:live|test)_[A-Za-z0-9]{20,}"),
    (
        "jwt",
        r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}",
    ),
    ("bearer token", r"(?i)bearer\s+([A-Za-z0-9\-._~+/]{20,}=*)"),
    (
        "private key",
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----",
    ),
];

const MIN_KNOWN_SECRET_LEN: usize = 12;

pub struct SecretRedactor {
    patterns: Vec<(&'static str, Regex)>,
    map: RedactionMap,
    placeholder_for: HashMap<String, String>,
    /// Labels and classes registered through `with_known`.
    known: Vec<(String, Class)>,
}

impl Default for SecretRedactor {
    fn default() -> Self {
        Self::new()
    }
}

impl SecretRedactor {
    pub fn new() -> Self {
        Self {
            patterns: PATTERNS
                .iter()
                .map(|(kind, p)| (*kind, Regex::new(p).expect("built-in pattern is valid")))
                .collect(),
            map: RedactionMap::new(),
            placeholder_for: HashMap::new(),
            known: Vec::new(),
        }
    }

    /// A value to redact wherever it appears, such as the provider key the
    /// process is running with, labelled for display and classed. Short
    /// values are ignored so a weak test key cannot redact every
    /// occurrence of a common word.
    pub fn with_known(mut self, label: &str, secret: &str, class: Class) -> Self {
        if secret.len() >= MIN_KNOWN_SECRET_LEN {
            self.known.push((label.to_string(), class));
            self.placeholder(secret, label, class);
        }
        self
    }

    /// Re-issues the placeholders of a saved session so history that
    /// mentions them keeps meaning the same thing. Entries stored without a
    /// value (redact-only ones) keep their placeholder but never match
    /// input; new placeholders are numbered after the saved ones.
    pub fn with_map(mut self, saved: &RedactionMap) -> Self {
        for (placeholder, entry) in saved {
            self.map.insert(placeholder.clone(), entry.clone());
            if !entry.value.is_empty() {
                self.placeholder_for
                    .insert(entry.value.clone(), placeholder.clone());
            }
        }
        self
    }

    /// Every kind this redactor can produce, with its class: the built-in
    /// detectors, then anything registered through `with_known`.
    pub fn catalog(&self) -> Vec<(String, Class)> {
        PATTERNS
            .iter()
            .map(|(kind, _)| (kind.to_string(), Class::Rehydrate))
            .chain(self.known.iter().cloned())
            .collect()
    }

    fn placeholder(&mut self, secret: &str, kind: &str, class: Class) -> String {
        if let Some(existing) = self.placeholder_for.get(secret) {
            return existing.clone();
        }
        let placeholder = format!("<<SECRET_{}>>", self.map.len() + 1);
        self.map.insert(
            placeholder.clone(),
            Entry {
                value: secret.to_string(),
                kind: kind.to_string(),
                class,
            },
        );
        self.placeholder_for
            .insert(secret.to_string(), placeholder.clone());
        placeholder
    }

    /// Non-overlapping byte ranges to redact, in order. When two patterns
    /// match the same start the longer match wins.
    ///
    /// Values already redacted once are matched literally as well. A
    /// context-dependent detector (Bearer) would otherwise miss the same
    /// value when it reappears without its context.
    fn spans(&self, input: &str) -> Vec<Span> {
        let known = self
            .placeholder_for
            .iter()
            .flat_map(|(secret, placeholder)| {
                let entry = &self.map[placeholder];
                input
                    .match_indices(secret.as_str())
                    .map(move |(start, found)| Span {
                        start,
                        end: start + found.len(),
                        kind: entry.kind.clone(),
                        class: entry.class,
                    })
            });
        let detected = self.patterns.iter().flat_map(|(kind, re)| {
            re.captures_iter(input).map(|caps| {
                let m = caps.get(1).unwrap_or_else(|| caps.get(0).unwrap());
                Span {
                    start: m.start(),
                    end: m.end(),
                    kind: kind.to_string(),
                    class: Class::Rehydrate,
                }
            })
        });
        let mut spans: Vec<Span> = known.chain(detected).collect();
        spans.sort_by(|a, b| a.start.cmp(&b.start).then(b.end.cmp(&a.end)));
        let mut kept: Vec<Span> = Vec::new();
        for span in spans {
            if kept.last().is_none_or(|last| span.start >= last.end) {
                kept.push(span);
            }
        }
        kept
    }
}

struct Span {
    start: usize,
    end: usize,
    kind: String,
    class: Class,
}

impl Redactor for SecretRedactor {
    fn redact(&mut self, input: &str) -> (String, RedactionMap) {
        let mut out = String::with_capacity(input.len());
        let mut cursor = 0;
        for span in self.spans(input) {
            out.push_str(&input[cursor..span.start]);
            out.push_str(&self.placeholder(&input[span.start..span.end], &span.kind, span.class));
            cursor = span.end;
        }
        out.push_str(&input[cursor..]);
        (out, self.map.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One realistic sample per detector, in PATTERNS order.
    const SAMPLES: &[&str] = &[
        concat!(
            "sk-ant-api03-AbCdEfGhIjKlMnOpQrStUvWxYz",
            "0123456789-abcdefghijklmnopqrstuvwxyzAA"
        ),
        concat!("sk-proj-AbCdEfGhIjKlMnOpQ", "rStUvWxYz0123456789abcdef"),
        concat!("ghp_AbCdEfGhIjKlMnOpQr", "StUvWxYz0123456789AbCd"),
        concat!(
            "github_pat_11ABCDEFG0abcdefghijklmnop",
            "qrstuvwxyz_ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        ),
        concat!("AKIAIOSFOD", "NN7EXAMPLE"),
        concat!(
            concat!("xo", "xb-"),
            concat!("1234567890-1234567890123", "-AbCdEfGhIjKlMnOpQrStUvWx")
        ),
        concat!("AIzaSyA1bC2dE3fG4hI5", "jK6lM7nO8pQ9rS0tU1vW"),
        concat!("sk_live_AbCdEfGhIjKl", "MnOpQrStUvWxYz012345"),
        concat!(
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwI",
            "n0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        ),
        concat!("Bearer AbCdEfGhIjKlMn", "OpQrStUvWxYz0123456789"),
        "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA\nabc\n-----END RSA PRIVATE KEY-----",
    ];

    #[test]
    fn every_pattern_has_a_sample() {
        assert_eq!(SAMPLES.len(), PATTERNS.len());
        for ((kind, pattern), sample) in PATTERNS.iter().zip(SAMPLES) {
            assert!(
                Regex::new(pattern).unwrap().is_match(sample),
                "{kind} does not match its sample"
            );
        }
    }

    #[test]
    fn roundtrip_and_idempotent_for_every_pattern() {
        for sample in SAMPLES {
            let text = format!("config:\n  value = {sample}\nend\n");
            let mut redactor = SecretRedactor::new();
            let (once, map) = redactor.redact(&text);
            assert!(!once.contains(sample), "secret leaked: {once}");
            assert_eq!(map.len(), 1, "expected exactly one redaction for {sample}");
            let (twice, _) = redactor.redact(&once);
            assert_eq!(once, twice, "redact is not idempotent for {sample}");
            assert_eq!(redactor.rehydrate(&once, &map), text);
            assert_eq!(map["<<SECRET_1>>"].class, Class::Rehydrate);
        }
    }

    #[test]
    fn bearer_keeps_the_scheme_word() {
        let mut redactor = SecretRedactor::new();
        let (out, _) =
            redactor.redact("Authorization: Bearer AbCdEfGhIjKlMnOpQrStUvWxYz0123456789");
        assert_eq!(out, "Authorization: Bearer <<SECRET_1>>");
    }

    #[test]
    fn placeholders_are_stable_across_calls() {
        let mut redactor = SecretRedactor::new();
        let key = SAMPLES[0];
        let (a, _) = redactor.redact(&format!("first {key}"));
        let (b, map) = redactor.redact(&format!("second {key} and {}", SAMPLES[2]));
        assert_eq!(a, "first <<SECRET_1>>");
        assert_eq!(b, "second <<SECRET_1>> and <<SECRET_2>>");
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn known_secret_is_redacted_without_its_original_context() {
        let mut redactor = SecretRedactor::new();
        let token = "AbCdEfGhIjKlMnOpQrStUvWxYz0123456789";
        let (first, _) = redactor.redact(&format!("Authorization: Bearer {token}"));
        assert_eq!(first, "Authorization: Bearer <<SECRET_1>>");
        let (second, _) = redactor.redact(&format!("TOKEN={token}"));
        assert_eq!(second, "TOKEN=<<SECRET_1>>");
    }

    #[test]
    fn known_secrets_are_redacted_without_a_pattern() {
        let mut redactor = SecretRedactor::new()
            .with_known(
                "provider api key",
                "0123456789abcdef0123456789abcdef",
                Class::RedactOnly,
            )
            .with_known("too short", "short", Class::RedactOnly);
        let (out, map) = redactor.redact("key=0123456789abcdef0123456789abcdef short");
        assert_eq!(out, "key=<<SECRET_1>> short");
        assert_eq!(map.len(), 1);
        assert_eq!(map["<<SECRET_1>>"].kind, "provider api key");
        assert_eq!(map["<<SECRET_1>>"].class, Class::RedactOnly);
        // Never restored, whatever the caller asks.
        assert_eq!(redactor.rehydrate(&out, &map), out);
        let catalog = redactor.catalog();
        assert_eq!(catalog.len(), PATTERNS.len() + 1);
        assert_eq!(
            catalog.last().unwrap(),
            &("provider api key".to_string(), Class::RedactOnly)
        );
    }

    #[test]
    fn every_placeholder_has_a_kind() {
        let mut redactor = SecretRedactor::new();
        let (_, map) = redactor.redact(&format!("{} and {}", SAMPLES[0], SAMPLES[4]));
        assert_eq!(map["<<SECRET_1>>"].kind, "anthropic api key");
        assert_eq!(map["<<SECRET_2>>"].kind, "aws access key id");
        assert_eq!(map.len(), 2);
        assert!(!map.contains_key("<<SECRET_3>>"));
    }

    #[test]
    fn ordinary_code_is_untouched() {
        let mut redactor = SecretRedactor::new();
        let src = "let token = parse_token_from_header(req);\nconst PREFIX: &str = \"sk-\";";
        let (out, map) = redactor.redact(src);
        assert_eq!(out, src);
        assert!(map.is_empty());
    }
}
