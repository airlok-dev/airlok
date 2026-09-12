//! `mcpServers` JSON: the shape Claude Code, Cursor, and VS Code share.
//!
//! Three files, in increasing precedence: `~/.config/airlok/mcp.json`,
//! `./.mcp.json` (meant to be committed), and `./.airlok/mcp.json`
//! (personal, gitignored). A server named in more than one takes the
//! highest one's definition whole, which is what makes a local file a
//! usable override rather than a patch.
//!
//! Files other tools wrote are read as they are: unknown keys are
//! ignored rather than refused, so a `.mcp.json` copied from another
//! project works without editing. airlok's own options live under an
//! `airlok` key, or in a `[[mcp]]` TOML block.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::config::{
    McpScope, McpServer, McpTools, McpTransport, McpTrust, DEFAULT_MCP_TIMEOUT_SECS,
};

/// `~/.config/airlok/mcp.json`, relative to the config directory.
pub const USER_FILE: &str = "mcp.json";
/// `./.mcp.json`, the one other tools write and projects commit.
pub const PROJECT_FILE: &str = ".mcp.json";
/// `./.airlok/mcp.json`, personal overrides.
pub const LOCAL_FILE: &str = ".airlok/mcp.json";

/// A whole `mcp.json`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct McpJson {
    #[serde(rename = "mcpServers", default)]
    pub servers: BTreeMap<String, Entry>,
}

/// One server. Everything is optional because this shape is shared with
/// other tools, and unknown keys are kept out of the way rather than
/// rejected.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Entry {
    /// `"stdio"` or `"http"`. Absent means stdio, or http when a url is
    /// given, which is what the other tools do.
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub args: Vec<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub env: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub headers: BTreeMap<String, String>,
    /// airlok's own options, so a plain config stays plain.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub airlok: Option<Extras>,
}

/// What airlok adds to the shared shape.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Extras {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trust: Option<McpTrust>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rehydrate: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<McpTools>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_secs: Option<u64>,
    /// Values read from a command's stdout, which is safer for a secret
    /// than `${VAR}`: nothing is written in the file and nothing has to
    /// sit in the environment.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub env_cmd: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub header_cmd: BTreeMap<String, String>,
}

/// Where each scope's file is.
pub fn path_for(scope: McpScope, config_dir: Option<&Path>, cwd: &Path) -> Option<PathBuf> {
    match scope {
        McpScope::User => config_dir.map(|dir| dir.join(USER_FILE)),
        McpScope::Project => Some(cwd.join(PROJECT_FILE)),
        McpScope::Local => Some(cwd.join(LOCAL_FILE)),
        McpScope::Toml => None,
    }
}

/// Reads one file. A missing file is no servers, not an error.
pub fn read(path: &Path) -> Result<McpJson, String> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(McpJson::default()),
        Err(e) => return Err(format!("cannot read {}: {e}", path.display())),
    };
    serde_json::from_str(&text).map_err(|e| format!("cannot parse {}: {e}", path.display()))
}

pub fn write(path: &Path, file: &McpJson) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
    }
    let text = serde_json::to_string_pretty(file)
        .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    std::fs::write(path, format!("{text}\n"))
        .map_err(|e| format!("cannot write {}: {e}", path.display()))
}

/// Every server from the three scopes, with the highest-precedence
/// definition of each name, and one line for anything that went wrong.
pub fn load(config_dir: Option<&Path>, cwd: &Path) -> (Vec<McpServer>, Vec<String>) {
    let mut found: Vec<McpServer> = Vec::new();
    let mut problems = Vec::new();
    for scope in [McpScope::User, McpScope::Project, McpScope::Local] {
        let Some(path) = path_for(scope, config_dir, cwd) else {
            continue;
        };
        let file = match read(&path) {
            Ok(file) => file,
            Err(problem) => {
                problems.push(problem);
                continue;
            }
        };
        for (name, entry) in file.servers {
            match entry.resolve(&name, scope) {
                Ok(server) => match found.iter().position(|s| s.name == server.name) {
                    // A later scope replaces the whole definition.
                    Some(at) => found[at] = server,
                    None => found.push(server),
                },
                Err(problem) => problems.push(format!("{}: {problem}", path.display())),
            }
        }
    }
    (found, problems)
}

/// The JSON servers with the TOML ones over them: a `[[mcp]]` block wins
/// a name clash, since it is the airlok-specific layer.
pub fn merge(json: Vec<McpServer>, toml: Vec<McpServer>) -> Vec<McpServer> {
    let mut merged = json;
    for server in toml {
        match merged.iter().position(|s| s.name == server.name) {
            Some(at) => merged[at] = server,
            None => merged.push(server),
        }
    }
    merged
}

impl Entry {
    /// One entry as airlok runs it: variables expanded, transport
    /// decided, airlok's own options applied.
    pub fn resolve(&self, name: &str, scope: McpScope) -> Result<McpServer, String> {
        let extras = self.airlok.clone().unwrap_or_default();
        let transport = match self.kind.as_deref() {
            Some("http") | Some("streamable-http") | Some("sse") => McpTransport::Http,
            Some("stdio") => McpTransport::Stdio,
            Some(other) => return Err(format!("{name}: unknown type {other:?}")),
            None if self.url.is_some() => McpTransport::Http,
            None => McpTransport::Stdio,
        };
        Ok(McpServer {
            name: name.to_string(),
            transport,
            command: self.command.as_deref().map(expand).transpose()?,
            args: self
                .args
                .iter()
                .map(|arg| expand(arg))
                .collect::<Result<_, _>>()?,
            env: expand_map(&self.env)?,
            env_cmd: extras.env_cmd,
            url: self.url.as_deref().map(expand).transpose()?,
            headers: expand_map(&self.headers)?,
            header_cmd: extras.header_cmd,
            enabled: extras.enabled.unwrap_or(true),
            timeout: Duration::from_secs(extras.timeout_secs.unwrap_or(DEFAULT_MCP_TIMEOUT_SECS)),
            tools: extras.tools.unwrap_or(McpTools::All),
            trust: extras.trust.unwrap_or(McpTrust::Prompt),
            rehydrate: extras.rehydrate.unwrap_or(false),
            scope,
        })
    }
}

/// One server back in the shared shape, for `mcp export` and `mcp add`.
pub fn entry_of(server: &McpServer) -> Entry {
    let extras = Extras {
        trust: Some(server.trust),
        rehydrate: Some(server.rehydrate),
        tools: Some(server.tools.clone()),
        enabled: Some(server.enabled),
        timeout_secs: Some(server.timeout.as_secs()),
        env_cmd: server.env_cmd.clone(),
        header_cmd: server.header_cmd.clone(),
    };
    Entry {
        kind: Some(
            match server.transport {
                McpTransport::Stdio => "stdio",
                McpTransport::Http => "http",
            }
            .to_string(),
        ),
        command: server.command.clone(),
        args: server.args.clone(),
        env: server.env.clone(),
        url: server.url.clone(),
        headers: server.headers.clone(),
        airlok: Some(extras),
    }
}

fn expand_map(map: &BTreeMap<String, String>) -> Result<BTreeMap<String, String>, String> {
    map.iter()
        .map(|(key, value)| Ok((key.clone(), expand(value)?)))
        .collect()
}

/// `${VAR}` and `${VAR:-default}` from the environment. A variable that is
/// unset or empty with no default is an error naming it, rather than an
/// empty string that fails later in a way nobody can read.
pub fn expand(text: &str) -> Result<String, String> {
    let mut out = String::new();
    let mut rest = text;
    while let Some(at) = rest.find("${") {
        out.push_str(&rest[..at]);
        let tail = &rest[at + 2..];
        let end = tail
            .find('}')
            .ok_or_else(|| format!("{text:?} has a ${{ that is never closed"))?;
        let (name, default) = match tail[..end].split_once(":-") {
            Some((name, default)) => (name, Some(default)),
            None => (&tail[..end], None),
        };
        let value = std::env::var(name).ok().filter(|value| !value.is_empty());
        match (value, default) {
            (Some(value), _) => out.push_str(&value),
            (None, Some(default)) => out.push_str(default),
            (None, None) => return Err(format!("{name} is not set and has no default")),
        }
        rest = &tail[end + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_plain_config_from_another_tool_loads() {
        // No airlok key, an unknown key other tools write, and both forms.
        let text = r#"{
          "mcpServers": {
            "files": {
              "command": "npx",
              "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
              "env": { "NODE_ENV": "production" },
              "disabled": false
            },
            "docs": { "type": "http", "url": "https://example.com/mcp",
                      "headers": { "Accept": "application/json" } }
          }
        }"#;
        let file: McpJson = serde_json::from_str(text).unwrap();
        let files = file.servers["files"]
            .resolve("files", McpScope::Project)
            .unwrap();
        assert_eq!(files.transport, McpTransport::Stdio);
        assert_eq!(files.command.as_deref(), Some("npx"));
        assert_eq!(files.args.len(), 3);
        assert_eq!(files.env["NODE_ENV"], "production");
        // The airlok-only options keep their defaults.
        assert_eq!(files.trust, McpTrust::Prompt);
        assert!(!files.rehydrate);
        assert!(files.enabled);
        assert_eq!(files.scope, McpScope::Project);

        let docs = file.servers["docs"]
            .resolve("docs", McpScope::Project)
            .unwrap();
        assert_eq!(docs.transport, McpTransport::Http);
        assert_eq!(docs.url.as_deref(), Some("https://example.com/mcp"));
        assert_eq!(docs.headers["Accept"], "application/json");
    }

    #[test]
    fn airlok_options_come_from_the_airlok_key() {
        let text = r#"{ "mcpServers": { "files": { "command": "x",
            "airlok": { "trust": "allow", "rehydrate": true, "tools": ["read"],
                        "timeout_secs": 5, "env_cmd": { "TOKEN": "printf t" } } } } }"#;
        let file: McpJson = serde_json::from_str(text).unwrap();
        let server = file.servers["files"]
            .resolve("files", McpScope::User)
            .unwrap();
        assert_eq!(server.trust, McpTrust::Allow);
        assert!(server.rehydrate);
        assert_eq!(server.tools, McpTools::Only(vec!["read".into()]));
        assert_eq!(server.timeout, Duration::from_secs(5));
        assert_eq!(server.env_cmd["TOKEN"], "printf t");
    }

    #[test]
    fn variables_expand_with_a_default_and_fail_loudly_when_unset() {
        std::env::set_var("AIRLOK_TEST_TOKEN", "sekret");
        std::env::remove_var("AIRLOK_TEST_MISSING");
        assert_eq!(expand("a ${AIRLOK_TEST_TOKEN} b").unwrap(), "a sekret b");
        assert_eq!(
            expand("${AIRLOK_TEST_MISSING:-fallback}").unwrap(),
            "fallback"
        );
        assert_eq!(expand("${AIRLOK_TEST_TOKEN:-fallback}").unwrap(), "sekret");
        assert_eq!(expand("nothing to do").unwrap(), "nothing to do");
        let err = expand("${AIRLOK_TEST_MISSING}").unwrap_err();
        assert!(err.contains("AIRLOK_TEST_MISSING"), "{err}");
        assert!(err.contains("no default"), "{err}");
        // An empty variable is treated as unset, so it cannot become a
        // silent empty argument.
        std::env::set_var("AIRLOK_TEST_EMPTY", "");
        assert!(expand("${AIRLOK_TEST_EMPTY}").is_err());
        assert_eq!(expand("${AIRLOK_TEST_EMPTY:-x}").unwrap(), "x");
        std::env::remove_var("AIRLOK_TEST_TOKEN");
        std::env::remove_var("AIRLOK_TEST_EMPTY");
    }

    #[test]
    fn a_toml_block_wins_a_name_clash() {
        let json = vec![
            server("files", McpScope::Project),
            server("docs", McpScope::User),
        ];
        let mut toml = server("files", McpScope::Toml);
        toml.command = Some("from-toml".into());
        let merged = merge(json, vec![toml]);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].name, "files");
        assert_eq!(merged[0].command.as_deref(), Some("from-toml"));
        assert_eq!(merged[0].scope, McpScope::Toml);
        assert_eq!(merged[1].name, "docs");
    }

    fn server(name: &str, scope: McpScope) -> McpServer {
        Entry {
            command: Some("x".into()),
            ..Entry::default()
        }
        .resolve(name, scope)
        .unwrap()
    }
}
