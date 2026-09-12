//! Where MCP servers come from: three JSON scopes, the TOML blocks over
//! them, and what happens when a file cannot be used.

use airlok_core::config::{Config, McpScope, McpTransport, McpTrust, Overrides};
use airlok_tests::TempDir;

/// A config directory and a project directory, as a real run has.
struct Layout {
    home: TempDir,
    project: TempDir,
}

impl Layout {
    fn new(label: &str) -> Self {
        let layout = Self {
            home: TempDir::new(&format!("{label}-home")),
            project: TempDir::new(&format!("{label}-project")),
        };
        std::fs::create_dir_all(layout.home.path().join("airlok")).unwrap();
        layout
    }

    /// `~/.config/airlok/config.toml`, which is also where the user
    /// scope's mcp.json sits.
    fn user_config(&self) -> std::path::PathBuf {
        self.home.path().join("airlok/config.toml")
    }

    fn write(&self, relative: &str, text: &str) {
        let path = self.project.path().join(relative);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }

    fn write_user(&self, name: &str, text: &str) {
        std::fs::write(self.home.path().join("airlok").join(name), text).unwrap();
    }

    fn load(&self) -> (Config, airlok_core::config::Sources) {
        Config::load(
            Some(&self.user_config()),
            Some(&self.project.path().join("airlok.toml")),
            &Overrides::default(),
            self.project.path().to_path_buf(),
        )
        .unwrap()
    }
}

fn server<'a>(config: &'a Config, name: &str) -> &'a airlok_core::config::McpServer {
    config
        .mcp
        .iter()
        .find(|server| server.name == name)
        .unwrap_or_else(|| panic!("no server {name} in {:?}", names(config)))
}

fn names(config: &Config) -> Vec<&str> {
    config.mcp.iter().map(|s| s.name.as_str()).collect()
}

#[test]
fn every_scope_is_read_and_the_highest_one_wins() {
    let layout = Layout::new("mcp-scopes");
    layout.write_user(
        "mcp.json",
        r#"{ "mcpServers": {
             "files": { "command": "from-user" },
             "docs":  { "type": "http", "url": "https://user.example/mcp" } } }"#,
    );
    layout.write(
        ".mcp.json",
        r#"{ "mcpServers": {
             "files":   { "command": "from-project" },
             "tickets": { "command": "from-project" } } }"#,
    );
    layout.write(
        ".airlok/mcp.json",
        r#"{ "mcpServers": { "files": { "command": "from-local", "args": ["--here"] } } }"#,
    );

    let (config, sources) = layout.load();

    assert!(
        sources.mcp_problems.is_empty(),
        "{:?}",
        sources.mcp_problems
    );
    let mut found = names(&config);
    found.sort();
    assert_eq!(found, ["docs", "files", "tickets"]);
    // The highest scope's definition is taken whole, not merged key by key.
    let files = server(&config, "files");
    assert_eq!(files.command.as_deref(), Some("from-local"));
    assert_eq!(files.args, ["--here"]);
    assert_eq!(files.scope, McpScope::Local);
    assert_eq!(server(&config, "docs").scope, McpScope::User);
    assert_eq!(server(&config, "tickets").scope, McpScope::Project);
}

#[test]
fn a_config_from_another_tool_works_untouched() {
    let layout = Layout::new("mcp-plain");
    // Copied from a Claude Code project: no airlok key anywhere, and a
    // key airlok does not know.
    layout.write(
        ".mcp.json",
        r#"{
          "mcpServers": {
            "filesystem": {
              "command": "npx",
              "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
              "env": { "NODE_ENV": "production" },
              "disabled": false
            },
            "remote": {
              "type": "http",
              "url": "https://example.com/mcp",
              "headers": { "Accept": "application/json" }
            }
          }
        }"#,
    );

    let (config, sources) = layout.load();

    assert!(
        sources.mcp_problems.is_empty(),
        "{:?}",
        sources.mcp_problems
    );
    let files = server(&config, "filesystem");
    assert_eq!(files.transport, McpTransport::Stdio);
    assert_eq!(files.command.as_deref(), Some("npx"));
    assert_eq!(files.env["NODE_ENV"], "production");
    // Airlok's own settings take their careful defaults.
    assert_eq!(files.trust, McpTrust::Prompt);
    assert!(!files.rehydrate);
    assert!(files.enabled);

    let remote = server(&config, "remote");
    assert_eq!(remote.transport, McpTransport::Http);
    assert_eq!(remote.url.as_deref(), Some("https://example.com/mcp"));
    assert_eq!(remote.headers["Accept"], "application/json");
}

#[test]
fn a_toml_block_wins_over_every_json_scope() {
    let layout = Layout::new("mcp-toml-wins");
    layout.write_user(
        "config.toml",
        "[[mcp]]\nname = \"files\"\ncommand = \"from-toml\"\ntrust = \"allow\"\n",
    );
    layout.write(
        ".mcp.json",
        r#"{ "mcpServers": { "files": { "command": "from-json" } } }"#,
    );

    let (config, _) = layout.load();

    let files = server(&config, "files");
    assert_eq!(files.command.as_deref(), Some("from-toml"));
    assert_eq!(files.trust, McpTrust::Allow);
    assert_eq!(files.scope, McpScope::Toml);
    assert_eq!(config.mcp.len(), 1, "one server, not two");
}

#[test]
fn a_variable_with_nothing_to_put_in_it_is_named_and_the_server_left_out() {
    let layout = Layout::new("mcp-unset");
    std::env::remove_var("AIRLOK_SCOPE_TEST_MISSING");
    layout.write(
        ".mcp.json",
        r#"{ "mcpServers": {
             "broken": { "command": "npx", "args": ["${AIRLOK_SCOPE_TEST_MISSING}"] },
             "fine":   { "command": "npx" } } }"#,
    );

    let (config, sources) = layout.load();

    assert_eq!(names(&config), ["fine"], "the broken one is left out");
    assert_eq!(sources.mcp_problems.len(), 1, "{:?}", sources.mcp_problems);
    let problem = &sources.mcp_problems[0];
    assert!(problem.contains("AIRLOK_SCOPE_TEST_MISSING"), "{problem}");
    assert!(problem.contains(".mcp.json"), "{problem}");
}

#[test]
fn a_variable_with_a_default_needs_no_environment() {
    let layout = Layout::new("mcp-default");
    std::env::remove_var("AIRLOK_SCOPE_TEST_ROOT");
    layout.write(
        ".mcp.json",
        r#"{ "mcpServers": { "files": {
             "command": "npx",
             "args": ["${AIRLOK_SCOPE_TEST_ROOT:-/srv/default}"] } } }"#,
    );

    let (config, sources) = layout.load();

    assert!(
        sources.mcp_problems.is_empty(),
        "{:?}",
        sources.mcp_problems
    );
    assert_eq!(server(&config, "files").args, ["/srv/default"]);
}
