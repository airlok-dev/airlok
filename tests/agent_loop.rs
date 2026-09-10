use airlok_llm::{ContentBlock, Role};
use airlok_tests::{agent, reply, tool_call, MockProvider, RecordingOutput, Shown, TempDir};
use serde_json::json;

#[tokio::test]
async fn runs_tool_calls_until_the_model_stops() {
    let dir = TempDir::new("loop");
    let provider = MockProvider::scripted(vec![
        tool_call(
            "toolu_1",
            "write_file",
            json!({"path": "hello.txt", "content": "hello"}),
        ),
        tool_call("toolu_2", "bash", json!({"command": "cat hello.txt"})),
        reply("Created hello.txt."),
    ]);
    let mut out = RecordingOutput::default();

    let report = agent(provider.clone(), dir.path())
        .run("create hello.txt containing hello", &mut out)
        .await
        .unwrap();

    assert_eq!(
        std::fs::read_to_string(dir.path().join("hello.txt")).unwrap(),
        "hello"
    );
    assert_eq!(report.turns, 3);
    assert_eq!(
        out.events,
        vec![
            Shown::ToolCall {
                name: "write_file".into(),
                summary: "hello.txt".into()
            },
            Shown::ConfirmWrite {
                path: dir.path().join("hello.txt"),
                diff: "--- a/hello.txt\n+++ b/hello.txt\n@@ -0,0 +1 @@\n+hello\n\\ No newline at end of file\n".into(),
            },
            // `cat` is allow-listed, so no confirmation for it.
            Shown::ToolCall {
                name: "bash".into(),
                summary: "cat hello.txt".into()
            },
            Shown::Text("Created hello.txt.".into()),
        ]
    );

    // The third request carries the whole conversation, tool results included.
    let requests = provider.requests();
    assert_eq!(requests.len(), 3);
    let last = &requests[2].messages;
    assert_eq!(last.len(), 5);
    assert_eq!(last[0].role, Role::User);
    assert_eq!(last[1].role, Role::Assistant);
    assert_eq!(
        last[4].content,
        vec![ContentBlock::ToolResult {
            tool_use_id: "toolu_2".into(),
            content: "hello".into(),
            is_error: false,
        }]
    );
    assert_eq!(requests[0].tools.len(), 4);
}

#[tokio::test]
async fn unknown_tool_and_tool_failure_go_back_as_errors() {
    let dir = TempDir::new("errors");
    let provider = MockProvider::scripted(vec![
        tool_call("toolu_1", "delete_everything", json!({})),
        tool_call("toolu_2", "read_file", json!({"path": "missing.txt"})),
        reply("ok"),
    ]);
    let mut out = RecordingOutput::default();

    agent(provider.clone(), dir.path())
        .run("go", &mut out)
        .await
        .unwrap();

    let requests = provider.requests();
    let result_of = |i: usize| match &requests[i].messages.last().unwrap().content[0] {
        ContentBlock::ToolResult {
            content, is_error, ..
        } => (content.clone(), *is_error),
        other => panic!("expected tool result, got {other:?}"),
    };
    assert_eq!(
        result_of(1),
        ("error: unknown tool `delete_everything`".to_string(), true)
    );
    let (content, is_error) = result_of(2);
    assert!(is_error);
    assert!(content.starts_with("error: "), "{content}");
}
