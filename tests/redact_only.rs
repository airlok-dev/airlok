//! The provider API key is a redact-only entry: it never comes back from
//! a placeholder, not in the terminal, not in a file, not in a command.

use std::path::Path;
use std::sync::Arc;

use airlok_core::redact::{mask, Class, SecretRedactor};
use airlok_core::tools::ToolRegistry;
use airlok_core::{Agent, Config};
use airlok_llm::ContentBlock;
use airlok_tests::{reply, tool_call, MockProvider, RecordingOutput, Shown, TempDir};
use serde_json::json;

const PROVIDER_KEY: &str = "6zKtMEXkQm3pLw9vN2rTb8yHc4dFg7jAs1eUi5oPx0zR";
const FILE_KEY: &str =
    "sk-ant-api03-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789-abcdefghijklmnopqrstuvwxyzAA";

/// An agent whose redactor knows the provider key as redact-only, the way
/// the CLI sets it up. Its placeholder is <<SECRET_1>>.
fn agent_with_provider_key(provider: Arc<MockProvider>, cwd: &Path) -> Agent {
    let config = Config::new(cwd.to_path_buf());
    let tools = ToolRegistry::defaults(cwd, config.agent.bash_timeout);
    let redactor =
        SecretRedactor::new().with_known("the provider API key", PROVIDER_KEY, Class::RedactOnly);
    Agent::new(provider, tools, Box::new(redactor), config)
}

fn tool_result(provider: &MockProvider, turn: usize) -> (String, bool) {
    match &provider.requests()[turn].messages.last().unwrap().content[0] {
        ContentBlock::ToolResult {
            content, is_error, ..
        } => (content.clone(), *is_error),
        other => panic!("expected a tool result, got {other:?}"),
    }
}

#[tokio::test]
async fn provider_placeholder_echoed_in_prose_shows_a_marker_and_no_key_bytes() {
    let dir = TempDir::new("echo");
    let provider = MockProvider::scripted(vec![
        reply("Set it like this: export KEY=<<SECRET_1>> and you are done."),
        reply("unused"),
    ]);
    let mut out = RecordingOutput::default();

    agent_with_provider_key(provider.clone(), dir.path())
        .run("how do I set the key?", &mut out)
        .await
        .unwrap();

    assert_eq!(
        out.text(),
        "Set it like this: export KEY=[redacted: the provider API key] and you are done."
    );
    for window in PROVIDER_KEY.as_bytes().windows(4) {
        let fragment = std::str::from_utf8(window).unwrap();
        assert!(
            !out.text().contains(fragment),
            "terminal leaked {fragment:?}"
        );
    }
}

#[tokio::test]
async fn provider_placeholder_in_write_file_is_refused_and_nothing_is_written() {
    let dir = TempDir::new("refuse");
    let provider = MockProvider::scripted(vec![
        tool_call(
            "toolu_1",
            "write_file",
            json!({"path": ".env", "content": "KEY=<<SECRET_1>>\n"}),
        ),
        tool_call("toolu_2", "bash", json!({"command": "echo <<SECRET_1>>"})),
        reply("understood"),
    ]);
    // Even if the user would approve, the call must never reach a prompt.
    let mut out = RecordingOutput::answering(vec![
        airlok_core::Decision::ApproveAll,
        airlok_core::Decision::ApproveAll,
    ]);

    agent_with_provider_key(provider.clone(), dir.path())
        .run("write the key to .env", &mut out)
        .await
        .unwrap();

    assert!(
        !dir.path().join(".env").exists(),
        ".env must not be written"
    );
    let (content, is_error) = tool_result(&provider, 1);
    assert!(is_error);
    assert!(
        content.starts_with("Refused: the arguments contain <<SECRET_1>> (the provider API key)"),
        "{content}"
    );
    let (content, is_error) = tool_result(&provider, 2);
    assert!(is_error);
    assert!(content.starts_with("Refused:"), "{content}");
    assert!(!out
        .events
        .iter()
        .any(|e| matches!(e, Shown::ConfirmWrite { .. } | Shown::ConfirmCommand { .. })));
    assert_eq!(out.decisions.len(), 2, "no prompt consumed an answer");
    // The provider never saw the key either.
    for request in provider.requests() {
        assert!(!serde_json::to_string(&request)
            .unwrap()
            .contains(PROVIDER_KEY));
    }
}

#[tokio::test]
async fn file_secret_lands_intact_through_edit_file_while_the_terminal_masks_it() {
    let dir = TempDir::new("edit");
    std::fs::write(
        dir.path().join("deploy.env"),
        format!("TOKEN={FILE_KEY}\nREGION=eu\n"),
    )
    .unwrap();
    let provider = MockProvider::scripted(vec![
        tool_call("toolu_1", "read_file", json!({"path": "deploy.env"})),
        // The model works with the placeholder it was shown.
        tool_call(
            "toolu_2",
            "edit_file",
            json!({"path": "deploy.env", "old": "TOKEN=<<SECRET_2>>\n", "new": "TOKEN=<<SECRET_2>>\nBACKUP_TOKEN=<<SECRET_2>>\n"}),
        ),
        reply("Duplicated <<SECRET_2>> into BACKUP_TOKEN."),
    ]);
    let mut out = RecordingOutput::default();

    let report = agent_with_provider_key(provider.clone(), dir.path())
        .run("add a BACKUP_TOKEN with the same value", &mut out)
        .await
        .unwrap();

    // On disk: the real value, twice.
    assert_eq!(
        std::fs::read_to_string(dir.path().join("deploy.env")).unwrap(),
        format!("TOKEN={FILE_KEY}\nBACKUP_TOKEN={FILE_KEY}\nREGION=eu\n")
    );
    // In the terminal: masked.
    assert_eq!(
        out.text(),
        format!("Duplicated {} into BACKUP_TOKEN.", mask(FILE_KEY))
    );
    assert!(!out.text().contains(FILE_KEY));
    // Over the wire: the placeholder only.
    let (content, is_error) = tool_result(&provider, 2);
    assert!(!is_error, "{content}");
    for request in provider.requests() {
        assert!(!serde_json::to_string(&request).unwrap().contains("sk-ant-"));
    }
    assert_eq!(report.redactions["<<SECRET_2>>"].class, Class::Rehydrate);
    assert_eq!(report.redactions["<<SECRET_1>>"].class, Class::RedactOnly);
}
