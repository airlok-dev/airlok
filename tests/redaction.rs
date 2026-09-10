use airlok_llm::{StopReason, StreamEvent};
use airlok_tests::{agent, reply, tool_call, LogBuffer, MockProvider, RecordingOutput, TempDir};
use serde_json::json;

const KEY: &str = "sk-ant-api03-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789-abcdefghijklmnopqrstuvwxyzAA";

#[tokio::test]
async fn api_key_read_from_a_file_never_reaches_the_provider() {
    let dir = TempDir::new("redact");
    std::fs::write(
        dir.path().join(".env"),
        format!("ANTHROPIC_API_KEY={KEY}\n"),
    )
    .unwrap();

    let provider = MockProvider::scripted(vec![
        tool_call("toolu_1", "read_file", json!({"path": ".env"})),
        reply("The key in .env is <<SECRET_1>>."),
    ]);
    let mut out = RecordingOutput::default();

    let report = agent(provider.clone(), dir.path())
        .run(
            &format!("what is the key in .env? also, my key is {KEY}"),
            &mut out,
        )
        .await
        .unwrap();

    // Nothing the provider saw contains the key: not the prompt, not the tool result.
    for request in provider.requests() {
        let wire = serde_json::to_string(&request).unwrap();
        assert!(
            !wire.contains("sk-ant-"),
            "secret leaked to provider: {wire}"
        );
        assert!(!wire.contains(KEY));
    }
    let wire = serde_json::to_string(&provider.requests()[1]).unwrap();
    assert!(wire.contains("ANTHROPIC_API_KEY=<<SECRET_1>>"), "{wire}");

    // The user sees the real value again.
    assert_eq!(out.text(), format!("The key in .env is {KEY}."));
    assert_eq!(
        report.redactions.get("<<SECRET_1>>").map(String::as_str),
        Some(KEY)
    );
}

#[tokio::test]
async fn placeholder_split_across_stream_chunks_is_rehydrated() {
    let dir = TempDir::new("split");
    std::fs::write(dir.path().join(".env"), format!("TOKEN={KEY}\n")).unwrap();

    let provider = MockProvider::scripted(vec![
        tool_call("toolu_1", "read_file", json!({"path": ".env"})),
        vec![
            StreamEvent::TextDelta("token: <<SE".into()),
            StreamEvent::TextDelta("CRET_1".into()),
            StreamEvent::TextDelta(">> done".into()),
            StreamEvent::MessageEnd {
                stop_reason: StopReason::EndTurn,
            },
        ],
    ]);
    let mut out = RecordingOutput::default();

    agent(provider, dir.path())
        .run("show the token", &mut out)
        .await
        .unwrap();

    assert_eq!(out.text(), format!("token: {KEY} done"));
}

#[tokio::test]
async fn placeholder_written_by_the_model_lands_on_disk_as_the_real_value() {
    let dir = TempDir::new("write");
    std::fs::write(dir.path().join("old.env"), format!("KEY={KEY}\n")).unwrap();

    let provider = MockProvider::scripted(vec![
        tool_call("toolu_1", "read_file", json!({"path": "old.env"})),
        tool_call(
            "toolu_2",
            "write_file",
            json!({"path": "new.env", "content": "KEY=<<SECRET_1>>\n"}),
        ),
        reply("copied"),
    ]);
    let mut out = RecordingOutput::default();

    agent(provider.clone(), dir.path())
        .run("copy the key", &mut out)
        .await
        .unwrap();

    assert_eq!(
        std::fs::read_to_string(dir.path().join("new.env")).unwrap(),
        format!("KEY={KEY}\n")
    );
    // The model's own tool call is stored rehydrated, so it is re-redacted on the next request.
    let wire = serde_json::to_string(&provider.requests()[2]).unwrap();
    assert!(!wire.contains("sk-ant-"), "{wire}");
}

#[tokio::test]
async fn secret_reused_by_the_model_out_of_context_still_never_reaches_the_provider() {
    let dir = TempDir::new("bearer");
    let token = "AbCdEfGhIjKlMnOpQrStUvWxYz0123456789";
    std::fs::write(
        dir.path().join("curl.sh"),
        format!("curl -H 'Authorization: Bearer {token}' https://api.example.com\n"),
    )
    .unwrap();

    let provider = MockProvider::scripted(vec![
        tool_call("toolu_1", "read_file", json!({"path": "curl.sh"})),
        tool_call(
            "toolu_2",
            "bash",
            json!({"command": "echo <<SECRET_1>> > token.txt"}),
        ),
        reply("saved"),
    ]);
    let mut out = RecordingOutput::default();

    agent(provider.clone(), dir.path())
        .run("save the token", &mut out)
        .await
        .unwrap();

    assert_eq!(
        std::fs::read_to_string(dir.path().join("token.txt")).unwrap(),
        format!("{token}\n")
    );
    for request in provider.requests() {
        let wire = serde_json::to_string(&request).unwrap();
        assert!(!wire.contains(token), "secret leaked to provider: {wire}");
    }
}

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

    let captured = logs.contents();
    assert!(
        captured.contains("tool result"),
        "no logs captured:\n{captured}"
    );
    assert!(
        captured.contains("executing"),
        "no logs captured:\n{captured}"
    );
    assert_eq!(
        captured.matches("sk-ant-").count(),
        0,
        "key leaked into logs:\n{captured}"
    );
    assert_eq!(captured.matches(KEY).count(), 0);
    // The user still sees the real value; only the logs must not.
    assert_eq!(out.text(), format!("The key is {KEY}."));
}
