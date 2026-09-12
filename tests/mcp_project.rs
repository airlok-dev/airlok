//! A `.mcp.json` arrives with a clone, so nothing in it starts until this
//! checkout has been asked about it.

use std::path::Path;

use airlok_core::config::{Config, McpScope, Overrides};
use airlok_core::mcp::project;
use airlok_core::redact::SecretRedactor;
use airlok_core::Decision;
use airlok_llm::Request;
use airlok_tests::{agent_configured, reply, MockProvider, RecordingOutput, Shown, TempDir};

const SERVER: &str = env!("CARGO_BIN_EXE_mock-mcp-server");

/// Writes a project `.mcp.json` whose server runs through a shell that
/// touches a marker first, so a test can tell whether it ever ran at all.
/// `tag` changes the command line without changing what it does.
fn clone_with_server(dir: &Path, tag: &str) -> Config {
    let marker = dir.join("started");
    let json = format!(
        r#"{{ "mcpServers": {{ "fromclone": {{ "command": "sh",
             "args": ["-c", "touch {} ; exec {} {}"] }} }} }}"#,
        marker.display(),
        SERVER,
        tag
    );
    std::fs::write(dir.join(".mcp.json"), json).unwrap();
    let (config, _) = Config::load(
        None,
        Some(&dir.join("airlok.toml")),
        &Overrides::default(),
        dir.to_path_buf(),
    )
    .unwrap();
    config
}

fn started(dir: &Path) -> bool {
    dir.join("started").exists()
}

fn asked(out: &RecordingOutput) -> Vec<Vec<(String, String)>> {
    out.events
        .iter()
        .filter_map(|event| match event {
            Shown::ConfirmMcpProject { servers } => Some(servers.clone()),
            _ => None,
        })
        .collect()
}

fn offered(request: &Request) -> Vec<&str> {
    request
        .tools
        .iter()
        .map(|tool| tool.name.as_str())
        .collect()
}

#[tokio::test]
async fn a_fresh_clone_starts_nothing_until_it_is_approved() {
    let dir = TempDir::new("mcp-clone-declined");
    let config = clone_with_server(dir.path(), "first");
    assert_eq!(config.mcp[0].scope, McpScope::Project);

    let provider = MockProvider::scripted(vec![reply("nothing to do")]);
    let mut agent = agent_configured(provider.clone(), config, SecretRedactor::new());
    let mut out = RecordingOutput::answering(vec![Decision::Reject]);

    agent.run("hello", &mut out).await.unwrap();

    // Asked once, naming the server and the command it would run.
    let questions = asked(&out);
    assert_eq!(questions.len(), 1, "{:?}", out.events);
    assert_eq!(questions[0][0].0, "fromclone");
    assert!(
        questions[0][0].1.contains("mock-mcp-server"),
        "{:?}",
        questions[0]
    );

    assert!(
        !started(dir.path()),
        "the server ran before it was approved"
    );
    let requests = provider.requests();
    let names = offered(&requests[0]);
    assert!(
        !names
            .iter()
            .any(|name| name.starts_with("mcp__fromclone__")),
        "{names:?}"
    );

    // The answer is recorded, so a later run does not ask again.
    let choices = project::load(dir.path());
    assert!(!choices.servers["fromclone"].approved);

    let provider = MockProvider::scripted(vec![reply("still nothing")]);
    let config = clone_with_server(dir.path(), "first");
    let mut agent = agent_configured(provider, config, SecretRedactor::new());
    let mut out = RecordingOutput::default();
    agent.run("again", &mut out).await.unwrap();
    assert!(asked(&out).is_empty(), "asked twice: {:?}", out.events);
    assert!(!started(dir.path()));
}

#[tokio::test]
async fn approving_lets_it_start_and_offers_its_tools() {
    let dir = TempDir::new("mcp-clone-approved");
    let config = clone_with_server(dir.path(), "first");

    let provider = MockProvider::scripted(vec![reply("ready")]);
    let mut agent = agent_configured(provider.clone(), config, SecretRedactor::new());
    let mut out = RecordingOutput::answering(vec![Decision::Approve]);

    agent.run("hello", &mut out).await.unwrap();

    assert_eq!(asked(&out).len(), 1);
    assert!(started(dir.path()), "the server should have run");
    let requests = provider.requests();
    let names = offered(&requests[0]);
    assert!(names.contains(&"mcp__fromclone__echo"), "{names:?}");
    assert!(project::load(dir.path()).servers["fromclone"].approved);
}

#[tokio::test]
async fn a_changed_command_is_asked_about_again() {
    let dir = TempDir::new("mcp-clone-changed");
    let config = clone_with_server(dir.path(), "first");
    let provider = MockProvider::scripted(vec![reply("ready")]);
    let mut agent = agent_configured(provider, config, SecretRedactor::new());
    let mut out = RecordingOutput::answering(vec![Decision::Approve]);
    agent.run("hello", &mut out).await.unwrap();
    assert_eq!(asked(&out).len(), 1);

    // The same name, a different command line: a different question.
    let changed = clone_with_server(dir.path(), "second");
    let provider = MockProvider::scripted(vec![reply("ready again")]);
    let mut agent = agent_configured(provider, changed, SecretRedactor::new());
    let mut out = RecordingOutput::answering(vec![Decision::Approve]);
    agent.run("hello again", &mut out).await.unwrap();

    let questions = asked(&out);
    assert_eq!(questions.len(), 1, "a changed command must ask again");
    assert!(questions[0][0].1.contains("second"), "{:?}", questions[0]);
}

#[tokio::test]
async fn a_server_from_the_user_scope_is_not_gated() {
    let dir = TempDir::new("mcp-user-scope");
    let home = TempDir::new("mcp-user-scope-home");
    std::fs::create_dir_all(home.path().join("airlok")).unwrap();
    std::fs::write(
        home.path().join("airlok/mcp.json"),
        format!(r#"{{ "mcpServers": {{ "mine": {{ "command": "{SERVER}" }} }} }}"#),
    )
    .unwrap();
    let (config, _) = Config::load(
        Some(&home.path().join("airlok/config.toml")),
        Some(&dir.path().join("airlok.toml")),
        &Overrides::default(),
        dir.path().to_path_buf(),
    )
    .unwrap();
    assert_eq!(config.mcp[0].scope, McpScope::User);

    let provider = MockProvider::scripted(vec![reply("ready")]);
    let mut agent = agent_configured(provider.clone(), config, SecretRedactor::new());
    let mut out = RecordingOutput::default();

    agent.run("hello", &mut out).await.unwrap();

    assert!(asked(&out).is_empty(), "the user's own file is not gated");
    let requests = provider.requests();
    let names = offered(&requests[0]);
    assert!(names.contains(&"mcp__mine__echo"), "{names:?}");
}
