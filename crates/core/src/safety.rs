//! Confirmation policy for tools that change the world.
//!
//! Writes show a diff and ask. Commands are checked against the deny list,
//! then the allow list, and everything else asks. The asking itself goes
//! through [`crate::Output`], so this module never touches a terminal.

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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    Approve,
    Reject,
    /// Approve this and every later prompt of the same kind in this run.
    ApproveAll,
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

impl SafetyConfig {
    pub fn classify(&self, command: &str) -> CommandVerdict {
        let words = tokens(command);
        if let Some(entry) = self
            .bash_denylist
            .iter()
            .find(|entry| contains_sequence(&words, &tokens(entry)))
        {
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

fn contains_sequence(haystack: &[&str], needle: &[&str]) -> bool {
    !needle.is_empty()
        && haystack
            .windows(needle.len())
            .any(|window| window == needle)
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
        // Tokens must match exactly, so these reach the prompt instead.
        assert_eq!(
            p.classify("git push --force-with-lease"),
            CommandVerdict::Confirm
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
