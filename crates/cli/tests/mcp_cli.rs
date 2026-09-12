//! `airlok mcp add / get / export / import / remove`, against the real
//! binary and real files.

use std::path::Path;
use std::process::Command;

use airlok_tests::TempDir;

const AIRLOK: &str = env!("CARGO_BIN_EXE_airlok");

/// Runs airlok with its own config directory and project directory, so
/// nothing here touches the machine's real configuration.
fn airlok(home: &Path, project: &Path, args: &[&str]) -> (String, String, bool) {
    let output = Command::new(AIRLOK)
        .args(args)
        .current_dir(project)
        .env("XDG_CONFIG_HOME", home)
        .env("NO_COLOR", "1")
        .output()
        .expect("airlok runs");
    (
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
        output.status.success(),
    )
}

#[test]
fn add_get_export_import_and_remove_round_trip() {
    let home = TempDir::new("mcp-cli-home");
    let project = TempDir::new("mcp-cli-project");
    let (home, project) = (home.path(), project.path());

    // add, into the scope that is not committed
    let (out, err, ok) = airlok(
        home,
        project,
        &[
            "mcp",
            "add",
            "files",
            "--scope",
            "local",
            "--env",
            "NODE_ENV=production",
            "--",
            "npx",
            "-y",
            "server-filesystem",
            ".",
        ],
    );
    assert!(ok, "add failed: {err}");
    assert!(out.contains("added files"), "{out}");
    let written = std::fs::read_to_string(project.join(".airlok/mcp.json")).unwrap();
    assert!(written.contains("\"mcpServers\""), "{written}");
    assert!(written.contains("\"command\": \"npx\""), "{written}");
    assert!(
        written.contains("\"NODE_ENV\": \"production\""),
        "{written}"
    );

    // get, which says where it came from
    let (out, err, ok) = airlok(home, project, &["mcp", "get", "files"]);
    assert!(ok, "get failed: {err}");
    assert!(out.starts_with("files from local"), "{out}");
    assert!(out.contains("\"command\": \"npx\""), "{out}");

    // export, which prints the standard shape
    let (out, err, ok) = airlok(home, project, &["mcp", "export"]);
    assert!(ok, "export failed: {err}");
    let exported: serde_json::Value = serde_json::from_str(&out).expect("export prints json");
    assert_eq!(exported["mcpServers"]["files"]["command"], "npx");

    // import another tool's file, with both kinds of entry in it
    let incoming = project.join("theirs.json");
    std::fs::write(
        &incoming,
        r#"{ "mcpServers": {
             "docs":    { "type": "http", "url": "https://example.com/mcp" },
             "tickets": { "command": "./tickets-mcp", "args": ["--quiet"] } } }"#,
    )
    .unwrap();
    let (out, err, ok) = airlok(
        home,
        project,
        &[
            "mcp",
            "import",
            incoming.to_str().unwrap(),
            "--scope",
            "project",
        ],
    );
    assert!(ok, "import failed: {err}");
    assert!(out.contains("docs") && out.contains("tickets"), "{out}");

    let (out, err, ok) = airlok(home, project, &["mcp", "export"]);
    assert!(ok, "export failed: {err}");
    let exported: serde_json::Value = serde_json::from_str(&out).unwrap();
    assert_eq!(
        exported["mcpServers"]["docs"]["url"],
        "https://example.com/mcp"
    );
    assert_eq!(exported["mcpServers"]["docs"]["type"], "http");
    assert_eq!(exported["mcpServers"]["tickets"]["args"][0], "--quiet");

    // remove, from every scope by default
    let (out, err, ok) = airlok(home, project, &["mcp", "remove", "docs"]);
    assert!(ok, "remove failed: {err}");
    assert!(out.contains("removed docs"), "{out}");
    let (out, _, ok) = airlok(home, project, &["mcp", "get", "docs"]);
    assert!(!ok, "docs should be gone: {out}");
}

#[test]
fn add_writes_the_user_scope_and_http_needs_a_url() {
    let home = TempDir::new("mcp-cli-user-home");
    let project = TempDir::new("mcp-cli-user-project");
    let (home, project) = (home.path(), project.path());

    let (_, err, ok) = airlok(
        home,
        project,
        &[
            "mcp", "add", "gateway", "--scope", "user", "--", "docker", "mcp", "gateway", "run",
        ],
    );
    assert!(ok, "add failed: {err}");
    let written = std::fs::read_to_string(home.join("airlok/mcp.json")).unwrap();
    assert!(written.contains("\"gateway\""), "{written}");
    assert!(written.contains("\"docker\""), "{written}");
    assert!(
        written.contains("\"gateway\",\n        \"run\"") || written.contains("\"run\""),
        "{written}"
    );

    // A server with no command and no url is a mistake worth naming.
    let (_, err, ok) = airlok(
        home,
        project,
        &["mcp", "add", "nothing", "--scope", "local"],
    );
    assert!(!ok);
    assert!(err.contains("needs a command"), "{err}");

    let (_, err, ok) = airlok(
        home,
        project,
        &[
            "mcp",
            "add",
            "remote",
            "--scope",
            "local",
            "--transport",
            "http",
        ],
    );
    assert!(!ok);
    assert!(err.contains("needs --url"), "{err}");
}

#[test]
fn listing_a_project_server_shows_it_pending_and_approves_nothing() {
    let home = TempDir::new("mcp-cli-pending-home");
    let project = TempDir::new("mcp-cli-pending-project");
    let (home, project) = (home.path(), project.path());
    // As it would arrive in a clone.
    std::fs::write(
        project.join(".mcp.json"),
        r#"{ "mcpServers": { "fromclone": { "command": "npx", "args": ["-y", "pkg"] } } }"#,
    )
    .unwrap();

    let (out, err, ok) = airlok(home, project, &["mcp", "list"]);
    assert!(ok, "list failed: {err}");
    assert!(
        out.contains("not approved for this repository yet"),
        "{out}"
    );
    assert!(out.contains("npx -y pkg"), "it says what would run: {out}");
    assert!(
        !project.join(".airlok/mcp-project.json").exists(),
        "listing must not record an answer"
    );

    // And it will not be called behind the gate either.
    let (_, err, ok) = airlok(home, project, &["mcp", "call", "fromclone", "echo", "{}"]);
    assert!(!ok);
    assert!(err.contains("has not been approved here"), "{err}");

    let (out, err, ok) = airlok(home, project, &["mcp", "reset-project-choices"]);
    assert!(ok, "{err}");
    assert!(out.contains("has not answered"), "{out}");
}

#[test]
fn add_json_writes_an_entry_pasted_as_json() {
    let home = TempDir::new("mcp-cli-addjson-home");
    let project = TempDir::new("mcp-cli-addjson-project");
    let (home, project) = (home.path(), project.path());

    let (out, err, ok) = airlok(
        home,
        project,
        &[
            "mcp",
            "add-json",
            "docs",
            r#"{"type":"http","url":"https://example.com/mcp","headers":{"Accept":"application/json"}}"#,
            "--scope",
            "project",
        ],
    );
    assert!(ok, "add-json failed: {err}");
    assert!(out.contains("added docs"), "{out}");

    let written = std::fs::read_to_string(project.join(".mcp.json")).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&written).unwrap();
    assert_eq!(
        parsed["mcpServers"]["docs"]["url"],
        "https://example.com/mcp"
    );
    assert_eq!(parsed["mcpServers"]["docs"]["type"], "http");

    // An entry that names neither a command nor a url is a mistake.
    let (_, err, ok) = airlok(home, project, &["mcp", "add-json", "empty", "{}"]);
    assert!(!ok);
    assert!(err.contains("needs a command or a url"), "{err}");
}

#[test]
fn trust_list_says_when_nothing_is_remembered() {
    let home = TempDir::new("mcp-cli-trust-home");
    let project = TempDir::new("mcp-cli-trust-project");
    let (out, err, ok) = airlok(home.path(), project.path(), &["mcp", "trust", "list"]);
    assert!(ok, "{err}");
    assert!(out.contains("nothing is remembered"), "{out}");

    let (_, err, ok) = airlok(
        home.path(),
        project.path(),
        &["mcp", "trust", "revoke", "files"],
    );
    assert!(!ok);
    assert!(err.contains("nothing was remembered for files"), "{err}");
}
