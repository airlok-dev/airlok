//! Confirmation policy for tools that change the world.
//!
//! Writes show a diff and ask. Commands are checked against the deny list,
//! then the allow list, and everything else asks. The asking itself goes
//! through [`crate::Output`], so this module never touches a terminal.

use std::collections::HashSet;
use std::path::Path;

use crate::config::SafetyConfig;

/// What the user is asked to approve.
#[derive(Debug, Clone, PartialEq)]
pub enum Confirmation<'a> {
    /// A file change. `diff` is a plain unified diff against the current content.
    Write {
        path: &'a Path,
        diff: &'a str,
    },
    Command {
        command: &'a str,
    },
    /// A call to a tool on an external MCP server. `arguments` is what
    /// will be sent, so the user can see whether a secret would leave;
    /// `root` is what the server can reach and `paths` what this call
    /// names, so a path outside the project is visible.
    Mcp {
        server: &'a str,
        tool: &'a str,
        arguments: &'a str,
        root: Option<&'a str>,
        paths: &'a [String],
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    Approve,
    Reject,
    /// Approve this and every later prompt of the same kind in this run.
    ApproveAll,
    /// [`Decision::ApproveAll`], and remember it past this run. Offered
    /// for MCP calls, where what is approved is a server, a tool, and the
    /// places that call named.
    SaveAll,
    /// Stop the whole run now. Nothing further is sent to the provider.
    Quit,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommandVerdict {
    /// Allow-listed: run without asking.
    Allowed,
    /// Deny-listed: refuse. Carries the matching entry.
    Denied(String),
    /// Ask the user.
    Confirm,
}

/// Shell syntax that can chain a second command onto an allow-listed one.
const CONTROL_OPERATORS: &[&str] = &[";", "&&", "||", "|", "&", ">", "<", "`", "$(", "\n"];

/// `find` is allow-listed for searching, but these make it destructive.
const FIND_ACTION_FLAGS: &[&str] = &["-exec", "-execdir", "-ok", "-okdir", "-delete"];

/// Long and capitalised spellings folded onto the short flag they mean, so
/// a deny entry written as `rm -rf` also covers `rm --recursive --force`.
const FLAG_ALIASES: &[(&str, &str)] = &[("--recursive", "r"), ("--force", "f")];

/// Capitalised short flags that mean the same as the lowercase one, folded
/// inside clusters too (`-Rf` is `-r` and `-f`).
const SHORT_FLAG_ALIASES: &[(char, char)] = &[('R', 'r')];

impl SafetyConfig {
    pub fn classify(&self, command: &str) -> CommandVerdict {
        let words = tokens(command);
        let segments: Vec<Invocation> = split_segments(command)
            .iter()
            .map(|segment| Invocation::parse(segment))
            .collect();
        if let Some(entry) = self.bash_denylist.iter().find(|entry| {
            let denied = Invocation::parse(entry);
            segments.iter().any(|segment| denied.matches(segment))
        }) {
            return CommandVerdict::Denied(entry.clone());
        }
        if CONTROL_OPERATORS.iter().any(|op| command.contains(op)) {
            return CommandVerdict::Confirm;
        }
        if words.first() == Some(&"find") && words.iter().any(|t| FIND_ACTION_FLAGS.contains(t)) {
            return CommandVerdict::Confirm;
        }
        let allowed = self.bash_allowlist.iter().any(|entry| {
            let prefix = tokens(entry);
            !prefix.is_empty() && words.starts_with(&prefix)
        });
        if allowed {
            CommandVerdict::Allowed
        } else {
            CommandVerdict::Confirm
        }
    }
}

fn tokens(text: &str) -> Vec<&str> {
    text.split_whitespace().collect()
}

fn contains_sequence<T: PartialEq>(haystack: &[T], needle: &[T]) -> bool {
    !needle.is_empty()
        && haystack
            .windows(needle.len())
            .any(|window| window == needle)
}

/// One simple command: its words and the set of flags it carries, with
/// clusters expanded (`-fr` is `-f` and `-r`) and aliases folded.
#[derive(Debug, PartialEq)]
struct Invocation {
    words: Vec<String>,
    flags: HashSet<String>,
}

impl Invocation {
    fn parse(text: &str) -> Self {
        let mut words = Vec::new();
        let mut flags = HashSet::new();
        for token in tokens(text) {
            let token = token.split_once('=').map_or(token, |(name, _)| name);
            if let Some(alias) = FLAG_ALIASES.iter().find(|(long, _)| *long == token) {
                flags.insert(alias.1.to_string());
            } else if let Some(long) = token.strip_prefix("--") {
                if !long.is_empty() {
                    flags.insert(format!("--{long}"));
                }
            } else if let Some(cluster) = token.strip_prefix('-').filter(|c| !c.is_empty()) {
                flags.extend(cluster.chars().map(|c| {
                    SHORT_FLAG_ALIASES
                        .iter()
                        .find(|(from, _)| *from == c)
                        .map_or(c, |(_, to)| *to)
                        .to_string()
                }));
            } else {
                words.push(token.to_string());
            }
        }
        Self { words, flags }
    }

    /// `self` is a deny entry: its words must appear in order somewhere in
    /// the segment and every flag it names must be present.
    fn matches(&self, segment: &Invocation) -> bool {
        contains_sequence(&segment.words, &self.words) && self.flags.is_subset(&segment.flags)
    }
}

/// Splits on the operators that join simple commands, so each part is
/// checked against the deny list on its own.
fn split_segments(command: &str) -> Vec<&str> {
    command
        .split(['|', '&', ';', '\n'])
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Config;
    use std::path::PathBuf;

    fn policy() -> SafetyConfig {
        Config::new(PathBuf::from(".")).safety
    }

    #[test]
    fn allowlist_matches_on_leading_tokens() {
        let p = policy();
        assert_eq!(p.classify("ls"), CommandVerdict::Allowed);
        assert_eq!(p.classify("ls -la src"), CommandVerdict::Allowed);
        assert_eq!(
            p.classify("  git   status --short"),
            CommandVerdict::Allowed
        );
        assert_eq!(
            p.classify("cargo test -p airlok-core"),
            CommandVerdict::Allowed
        );
        // A partial token is not a prefix.
        assert_eq!(p.classify("lsof -i"), CommandVerdict::Confirm);
        // A partial entry is not a prefix either.
        assert_eq!(p.classify("git push"), CommandVerdict::Confirm);
        assert_eq!(p.classify("cargo run"), CommandVerdict::Confirm);
    }

    #[test]
    fn denylist_wins_over_allowlist_and_matches_anywhere() {
        let p = policy();
        assert_eq!(
            p.classify("rm -rf target"),
            CommandVerdict::Denied("rm -rf".into())
        );
        assert_eq!(
            p.classify("ls && rm -rf /"),
            CommandVerdict::Denied("rm -rf".into())
        );
        assert_eq!(p.classify("sudo ls"), CommandVerdict::Denied("sudo".into()));
        assert_eq!(
            p.classify("git push --force origin main"),
            CommandVerdict::Denied("git push --force".into())
        );
        assert_eq!(
            p.classify("git push -f origin main"),
            CommandVerdict::Denied("git push --force".into())
        );
        // A different flag is a different command, so these reach the prompt instead.
        assert_eq!(
            p.classify("git push --force-with-lease"),
            CommandVerdict::Confirm
        );
        assert_eq!(p.classify("git push origin main"), CommandVerdict::Confirm);
    }

    #[test]
    fn denylist_understands_every_spelling_of_the_flags() {
        let p = policy();
        for command in [
            "rm -rf x",
            "rm -fr x",
            "rm -r -f x",
            "rm -f -r x",
            "rm --recursive --force x",
            "rm --force --recursive x",
            "rm -Rf x",
            "rm -r --force x",
            "rm -rfv x",
            "rm -rf --no-preserve-root /",
            "cd / && rm -fr *",
            "xargs rm -r -f",
        ] {
            assert_eq!(
                p.classify(command),
                CommandVerdict::Denied("rm -rf".into()),
                "{command:?}"
            );
        }
        // Missing one of the two flags is not the denied command.
        assert_eq!(p.classify("rm -r x"), CommandVerdict::Confirm);
        assert_eq!(p.classify("rm -f x"), CommandVerdict::Confirm);
        assert_eq!(p.classify("rm --recursive x"), CommandVerdict::Confirm);
        assert_eq!(p.classify("rm x"), CommandVerdict::Confirm);
    }

    #[test]
    fn invocation_parsing() {
        let inv = Invocation::parse("rm -fr --verbose --depth=3 a b");
        assert_eq!(inv.words, ["rm", "a", "b"]);
        let mut flags: Vec<&str> = inv.flags.iter().map(String::as_str).collect();
        flags.sort();
        assert_eq!(flags, ["--depth", "--verbose", "f", "r"]);
        assert_eq!(
            split_segments("ls && rm -rf / ; echo done | cat"),
            ["ls", "rm -rf /", "echo done", "cat"]
        );
    }

    #[test]
    fn chained_commands_are_never_allowlisted() {
        let p = policy();
        for command in [
            "ls; cargo run",
            "ls && cargo run",
            "ls || cargo run",
            "cat a | sh",
            "ls > out.txt",
            "cat < in.txt",
            "ls `which sh`",
            "ls $(which sh)",
            "ls\ncargo run",
            "ls &",
        ] {
            assert_eq!(p.classify(command), CommandVerdict::Confirm, "{command:?}");
        }
    }

    #[test]
    fn destructive_find_asks() {
        let p = policy();
        assert_eq!(p.classify("find . -name '*.rs'"), CommandVerdict::Allowed);
        assert_eq!(
            p.classify("find . -name '*.o' -delete"),
            CommandVerdict::Confirm
        );
        assert_eq!(p.classify("find . -exec rm {} +"), CommandVerdict::Confirm);
    }
}
