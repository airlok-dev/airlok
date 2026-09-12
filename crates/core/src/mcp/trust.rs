//! Approvals remembered past the end of a run.
//!
//! Answering `s` at an MCP confirmation writes what was approved to
//! `.airlok/mcp-trust.json`: the server, the tool, and the places that
//! call named. It is kept beside the project rather than in the home
//! directory, because trusting a server to read one repository is not
//! trusting it everywhere.
//!
//! Two things keep a saved approval from growing teeth. It never covers a
//! place outside the ones approved, exactly like the in-run rule. And it
//! records a fingerprint of the server's resolved definition, so changing
//! the command, the arguments, or the url makes every approval for it ask
//! again rather than carrying over to a different program.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::config::McpServer;

/// Where approvals are kept, next to the project's other airlok state.
pub const TRUST_FILE: &str = ".airlok/mcp-trust.json";

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Store {
    #[serde(default)]
    pub approvals: Vec<Approval>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Approval {
    pub server: String,
    pub tool: String,
    /// Resolved places, as the confirmation showed them. Empty means the
    /// call named none, and only such a call is covered.
    #[serde(default)]
    pub paths: Vec<String>,
    /// The server's definition when this was approved.
    pub fingerprint: String,
    pub saved_at: String,
}

pub fn path_for(cwd: &Path) -> PathBuf {
    cwd.join(TRUST_FILE)
}

/// What was approved here. A missing or unreadable file is no approvals:
/// the cost of being wrong is a prompt, so it fails towards asking.
pub fn load(cwd: &Path) -> Store {
    let path = path_for(cwd);
    let Ok(text) = std::fs::read_to_string(&path) else {
        return Store::default();
    };
    serde_json::from_str(&text).unwrap_or_default()
}

/// Writes the store 0600 in a 0700 directory, like the session files: it
/// names the places a third-party server may read.
pub fn save(cwd: &Path, store: &Store) -> Result<(), String> {
    let path = path_for(cwd);
    let dir = path.parent().unwrap_or(cwd);
    std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    restrict(dir, 0o700)?;
    let text = serde_json::to_string_pretty(store).map_err(|e| e.to_string())?;
    std::fs::write(&path, format!("{text}\n"))
        .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    restrict(&path, 0o600)
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

impl Store {
    /// Whether a saved approval covers this call. The rule matches the
    /// in-run one, with the fingerprint on top.
    pub fn covers(&self, server: &str, tool: &str, paths: &[String], fingerprint: &str) -> bool {
        self.approvals.iter().any(|approval| {
            approval.server == server
                && approval.tool == tool
                && approval.fingerprint == fingerprint
                && covers_paths(&approval.paths, paths)
        })
    }

    /// Remembers one approval, replacing any earlier one for the same
    /// server and tool so the newest definition is what is kept.
    pub fn remember(&mut self, server: &str, tool: &str, paths: &[String], fingerprint: &str) {
        self.approvals
            .retain(|approval| !(approval.server == server && approval.tool == tool));
        self.approvals.push(Approval {
            server: server.to_string(),
            tool: tool.to_string(),
            paths: paths.to_vec(),
            fingerprint: fingerprint.to_string(),
            saved_at: crate::session::now_rfc3339(),
        });
    }

    /// Drops every approval for a server. Returns how many went.
    pub fn revoke(&mut self, server: &str) -> usize {
        let before = self.approvals.len();
        self.approvals.retain(|approval| approval.server != server);
        before - self.approvals.len()
    }

    /// One line per approval, for `airlok mcp trust list`.
    pub fn lines(&self) -> Vec<String> {
        self.approvals
            .iter()
            .map(|approval| {
                let places = if approval.paths.is_empty() {
                    "no path arguments".to_string()
                } else {
                    approval.paths.join(", ")
                };
                format!(
                    "{}__{} ({places}), saved {}",
                    approval.server, approval.tool, approval.saved_at
                )
            })
            .collect()
    }
}

fn covers_paths(approved: &[String], wanted: &[String]) -> bool {
    if wanted.is_empty() {
        return approved.is_empty();
    }
    wanted.iter().all(|path| {
        approved
            .iter()
            .any(|allowed| Path::new(path).starts_with(Path::new(allowed)))
    })
}

/// What the approval was for: enough of the definition that a change of
/// program, arguments, or endpoint is a different server. Secrets are not
/// part of it, so rotating a token does not throw an approval away.
pub fn fingerprint(server: &McpServer) -> String {
    let mut parts = vec![
        server.name.clone(),
        format!("{:?}", server.transport),
        server.command.clone().unwrap_or_default(),
        server.url.clone().unwrap_or_default(),
    ];
    parts.extend(server.args.iter().cloned());
    parts.extend(server.env.keys().cloned());
    parts.extend(server.env_cmd.keys().cloned());
    format!("{:016x}", fnv1a(&parts.join("\u{1f}")))
}

/// FNV-1a, so the fingerprint written today still matches tomorrow.
/// `DefaultHasher` makes no such promise across releases.
fn fnv1a(text: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in text.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{McpScope, McpServer, McpTools, McpTransport, McpTrust};
    use std::collections::BTreeMap;
    use std::time::Duration;

    fn server(command: &str) -> McpServer {
        McpServer {
            name: "files".into(),
            transport: McpTransport::Stdio,
            command: Some(command.into()),
            args: vec!["--root".into(), "/srv".into()],
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
            scope: McpScope::Project,
        }
    }

    #[test]
    fn an_approval_covers_the_places_it_named_and_no_others() {
        let print = fingerprint(&server("npx"));
        let mut store = Store::default();
        store.remember("files", "read", &["/srv/docs".into()], &print);

        assert!(store.covers("files", "read", &["/srv/docs/a.md".into()], &print));
        assert!(store.covers("files", "read", &["/srv/docs".into()], &print));
        assert!(!store.covers("files", "read", &["/etc/passwd".into()], &print));
        // Another tool, and a call naming no place at all.
        assert!(!store.covers("files", "write", &["/srv/docs/a.md".into()], &print));
        assert!(!store.covers("files", "read", &[], &print));
    }

    #[test]
    fn changing_the_command_asks_again() {
        let before = fingerprint(&server("npx"));
        let after = fingerprint(&server("node"));
        assert_ne!(before, after);
        let mut store = Store::default();
        store.remember("files", "read", &["/srv".into()], &before);
        assert!(store.covers("files", "read", &["/srv/a".into()], &before));
        assert!(!store.covers("files", "read", &["/srv/a".into()], &after));
    }

    #[test]
    fn a_rotated_secret_is_not_a_different_server() {
        let mut with_token = server("npx");
        with_token.env.insert("TOKEN".into(), "one".into());
        let mut rotated = server("npx");
        rotated.env.insert("TOKEN".into(), "two".into());
        assert_eq!(fingerprint(&with_token), fingerprint(&rotated));
        // Adding a variable is a change of definition, though.
        let mut extra = with_token.clone();
        extra.env.insert("OTHER".into(), "x".into());
        assert_ne!(fingerprint(&with_token), fingerprint(&extra));
    }

    #[test]
    fn revoking_drops_every_approval_for_a_server() {
        let print = fingerprint(&server("npx"));
        let mut store = Store::default();
        store.remember("files", "read", &["/srv".into()], &print);
        store.remember("files", "write", &["/srv".into()], &print);
        store.remember("docs", "search", &[], &print);
        assert_eq!(store.revoke("files"), 2);
        assert_eq!(store.approvals.len(), 1);
        assert_eq!(store.lines().len(), 1);
        assert!(
            store.lines()[0].starts_with("docs__search"),
            "{:?}",
            store.lines()
        );
    }
}
