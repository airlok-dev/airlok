//! Runs in its own test binary on purpose. `tracing` caches per-callsite
//! interest when a callsite is first hit; if another test in the same
//! process hits the agent's log sites while no subscriber is installed,
//! those sites are cached as "never" and the scoped subscriber here sees
//! nothing. One test per process removes that race.

use airlok_tests::{agent, reply, tool_call, LogBuffer, MockProvider, RecordingOutput, TempDir};
use serde_json::json;

const KEY: &str = "sk-ant-api03-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789-abcdefghijklmnopqrstuvwxyzAA";

#[tokio::test]
async fn secret_never_reaches_logs() {
    let dir = TempDir::new("logs");
    std::fs::write(
        dir.path().join("secrets.env"),
        format!("ANTHROPIC_API_KEY={KEY}\n"),
    )
    .unwrap();

    let provider = MockProvider::scripted(vec![
        tool_call("toolu_1", "read_file", json!({"path": "secrets.env"})),
        tool_call("toolu_2", "bash", json!({"command": "echo <<SECRET_1>>"})),
        reply("The key is <<SECRET_1>>."),
    ]);
    let logs = LogBuffer::default();
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .with_ansi(false)
        .with_writer(logs.clone())
        .finish();
    let mut out = RecordingOutput::default();

    {
        use tracing::instrument::WithSubscriber;
        agent(provider, dir.path())
            .run("read secrets.env and tell me what is in it", &mut out)
            .with_subscriber(subscriber)
            .await
            .unwrap();
    }

    // Positive checks first: if capture silently breaks, the leak check
    // below would pass vacuously.
    let captured = logs.contents();
    for expected in [
        "INFO",
        "executing tool=read_file",
        "executing tool=bash",
        "tool result",
    ] {
        assert!(
            captured.contains(expected),
            "expected log line containing {expected:?} was not captured:\n{captured}"
        );
    }
    assert_eq!(
        captured.matches("sk-ant-").count(),
        0,
        "key leaked into logs:\n{captured}"
    );
    assert_eq!(captured.matches(KEY).count(), 0);
    // The terminal shows it masked; the logs must not show it at all.
    assert_eq!(
        out.text(),
        format!("The key is {}.", airlok_core::redact::mask(KEY))
    );
}
