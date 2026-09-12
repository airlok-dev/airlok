//! Whether a project's own MCP servers may start.
//!
//! A `.mcp.json` arrives with a clone, which means a server definition
//! can arrive from someone else. Nothing in it starts until this
//! repository has been asked about it once: the servers and the commands
//! they would run are shown, the answer is recorded in `.airlok/`, and a
//! definition that changes afterwards is asked about again.
//!
//! The record is per repository and gitignored, because it is a statement
//! about this checkout on this machine, not about the project.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::config::{McpScope, McpServer, McpTransport};

pub const CHOICES_FILE: &str = ".airlok/mcp-project.json";

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Choices {
    #[serde(default)]
    pub servers: BTreeMap<String, Choice>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Choice {
    /// The definition this answer was about.
    pub fingerprint: String,
    pub approved: bool,
    pub decided_at: String,
}

pub fn path_for(cwd: &Path) -> PathBuf {
    cwd.join(CHOICES_FILE)
}

/// What this repository has already been asked. Anything unreadable is
/// no answers, so the cost of being wrong is another prompt.
pub fn load(cwd: &Path) -> Choices {
    let Ok(text) = std::fs::read_to_string(path_for(cwd)) else {
        return Choices::default();
    };
    serde_json::from_str(&text).unwrap_or_default()
}

/// Writes 0600 in a 0700 directory: it records which programs this
/// checkout may start.
pub fn save(cwd: &Path, choices: &Choices) -> Result<(), String> {
    let path = path_for(cwd);
    let dir = path.parent().unwrap_or(cwd);
    std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    restrict(dir, 0o700)?;
    let text = serde_json::to_string_pretty(choices).map_err(|e| e.to_string())?;
    std::fs::write(&path, format!("{text}\n"))
        .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    restrict(&path, 0o600)
}

/// Forgets every answer, so the next run asks again. Returns whether
/// there was anything to forget.
pub fn reset(cwd: &Path) -> Result<bool, String> {
    let path = path_for(cwd);
    match std::fs::remove_file(&path) {
        Ok(()) => Ok(true),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(format!("cannot remove {}: {e}", path.display())),
    }
}

#[cfg(unix)]
fn restrict(path: &Path, mode: u32) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode))
        .map_err(|e| format!("cannot set permissions on {}: {e}", path.display()))
}

#[cfg(not(unix))]
fn restrict(_path: &Path, _mode: u32) -> Result<(), String> {
    Ok(())
}

impl Choices {
    /// What was answered about this definition, if anything. A different
    /// fingerprint is a different question.
    pub fn decision(&self, name: &str, fingerprint: &str) -> Option<bool> {
        self.servers
            .get(name)
            .filter(|choice| choice.fingerprint == fingerprint)
            .map(|choice| choice.approved)
    }

    pub fn record(&mut self, name: &str, fingerprint: &str, approved: bool) {
        self.servers.insert(
            name.to_string(),
            Choice {
                fingerprint: fingerprint.to_string(),
                approved,
                decided_at: crate::session::now_rfc3339(),
            },
        );
    }

    pub fn lines(&self) -> Vec<String> {
        self.servers
            .iter()
            .map(|(name, choice)| {
                let answer = if choice.approved {
                    "approved"
                } else {
                    "declined"
                };
                format!("{name}: {answer} {}", choice.decided_at)
            })
            .collect()
    }
}

/// Whether this server may start without asking. Only the project scope
/// is gated: the user's own files and a `[[mcp]]` block are things they
/// wrote, while `.mcp.json` arrives with a clone.
pub fn allowed(server: &McpServer, choices: &Choices) -> bool {
    if server.scope != McpScope::Project {
        return true;
    }
    choices
        .decision(&server.name, &super::trust::fingerprint(server))
        .unwrap_or(false)
}

/// Whether this server still has to be asked about, as opposed to having
/// been declined.
pub fn undecided(server: &McpServer, choices: &Choices) -> bool {
    server.scope == McpScope::Project
        && choices
            .decision(&server.name, &super::trust::fingerprint(server))
            .is_none()
}

/// What would run, for the prompt and for the pending line. This is the
/// thing worth reading before answering.
pub fn what_runs(server: &McpServer) -> String {
    match server.transport {
        McpTransport::Http => server.url.clone().unwrap_or_else(|| "(no url)".to_string()),
        McpTransport::Stdio => {
            let command = server
                .command
                .clone()
                .unwrap_or_else(|| "(no command)".into());
            if server.args.is_empty() {
                command
            } else {
                format!("{command} {}", server.args.join(" "))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{McpTools, McpTrust};
    use std::time::Duration;

    fn server(scope: McpScope, args: &[&str]) -> McpServer {
        McpServer {
            name: "files".into(),
            transport: McpTransport::Stdio,
            command: Some("npx".into()),
            args: args.iter().map(|a| a.to_string()).collect(),
            env: BTreeMap::new(),
            env_cmd: BTreeMap::new(),
            url: None,
            headers: BTreeMap::new(),
            header_cmd: BTreeMap::new(),
            enabled: true,
            timeout: Duration::from_secs(30),
            tools: McpTools::All,
            trust: McpTrust::Prompt,
            rehydrate: false,
            scope,
        }
    }

    #[test]
    fn only_the_project_scope_is_gated() {
        let choices = Choices::default();
        assert!(allowed(&server(McpScope::User, &[]), &choices));
        assert!(allowed(&server(McpScope::Local, &[]), &choices));
        assert!(allowed(&server(McpScope::Toml, &[]), &choices));
        // The one that arrives with a clone waits to be asked about.
        assert!(!allowed(&server(McpScope::Project, &[]), &choices));
        assert!(undecided(&server(McpScope::Project, &[]), &choices));
    }

    #[test]
    fn an_answer_covers_that_definition_and_no_other() {
        let first = server(McpScope::Project, &["--root", "."]);
        let mut choices = Choices::default();
        choices.record("files", &super::super::trust::fingerprint(&first), true);

        assert!(allowed(&first, &choices));
        assert!(!undecided(&first, &choices));

        // A changed command line is a different question.
        let changed = server(McpScope::Project, &["--root", "/"]);
        assert!(!allowed(&changed, &choices));
        assert!(undecided(&changed, &choices));
    }

    #[test]
    fn declining_is_remembered_as_an_answer() {
        let server = server(McpScope::Project, &[]);
        let mut choices = Choices::default();
        choices.record("files", &super::super::trust::fingerprint(&server), false);
        assert!(!allowed(&server, &choices), "declined stays declined");
        assert!(!undecided(&server, &choices), "and is not asked again");
        assert_eq!(choices.lines().len(), 1);
        assert!(choices.lines()[0].starts_with("files: declined"));
    }

    #[test]
    fn what_runs_is_the_command_or_the_url() {
        assert_eq!(
            what_runs(&server(McpScope::Project, &["-y", "pkg"])),
            "npx -y pkg"
        );
        let mut http = server(McpScope::Project, &[]);
        http.transport = McpTransport::Http;
        http.url = Some("https://example.com/mcp".into());
        assert_eq!(what_runs(&http), "https://example.com/mcp");
    }
}
