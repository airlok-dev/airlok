use airlok_core::Decision;
use airlok_llm::ContentBlock;
use airlok_tests::{agent, reply, tool_call, MockProvider, RecordingOutput, Shown, TempDir};
use serde_json::json;

/// The tool result the model received for the call made on `turn`.
fn result_after(provider: &MockProvider, turn: usize) -> (String, bool) {
    let requests = provider.requests();
    match &requests[turn].messages.last().unwrap().content[0] {
        ContentBlock::ToolResult {
            content, is_error, ..
        } => (content.clone(), *is_error),
        other => panic!("expected a tool result, got {other:?}"),
    }
}

#[tokio::test]
async fn rejected_write_leaves_the_file_untouched_and_tells_the_model() {
    let dir = TempDir::new("reject-write");
    std::fs::write(dir.path().join("notes.md"), "original\n").unwrap();
    let provider = MockProvider::scripted(vec![
        tool_call(
            "toolu_1",
            "write_file",
            json!({"path": "notes.md", "content": "clobbered\n"}),
        ),
        tool_call(
            "toolu_2",
            "edit_file",
            json!({"path": "notes.md", "search": "original", "replace": "edited"}),
        ),
        reply("understood"),
    ]);
    let mut out = RecordingOutput::answering(vec![Decision::Reject, Decision::Reject]);

    agent(provider.clone(), dir.path())
        .run("change notes", &mut out)
        .await
        .unwrap();

    assert_eq!(
        std::fs::read_to_string(dir.path().join("notes.md")).unwrap(),
        "original\n"
    );
    let (content, is_error) = result_after(&provider, 1);
    assert!(is_error);
    assert!(
        content.starts_with("The user rejected this change to "),
        "{content}"
    );
    assert!(content.contains("notes.md"), "{content}");
    let (content, is_error) = result_after(&provider, 2);
    assert!(is_error);
    assert!(
        content.starts_with("The user rejected this change to "),
        "{content}"
    );

    // Both prompts showed a diff of the intended change.
    let diffs: Vec<&str> = out
        .events
        .iter()
        .filter_map(|e| match e {
            Shown::ConfirmWrite { diff, .. } => Some(diff.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(diffs.len(), 2);
    assert!(diffs[0].contains("-original\n+clobbered\n"), "{}", diffs[0]);
    assert!(diffs[1].contains("-original\n+edited\n"), "{}", diffs[1]);
}

#[tokio::test]
async fn approved_edit_is_applied() {
    let dir = TempDir::new("approve-edit");
    std::fs::write(dir.path().join("a.txt"), "one two three\n").unwrap();
    let provider = MockProvider::scripted(vec![
        tool_call(
            "toolu_1",
            "edit_file",
            json!({"path": "a.txt", "search": "two", "replace": "2"}),
        ),
        reply("done"),
    ]);
    let mut out = RecordingOutput::answering(vec![Decision::Approve]);

    agent(provider.clone(), dir.path())
        .run("edit", &mut out)
        .await
        .unwrap();

    assert_eq!(
        std::fs::read_to_string(dir.path().join("a.txt")).unwrap(),
        "one 2 three\n"
    );
    assert!(!result_after(&provider, 1).1);
}

#[tokio::test]
async fn ambiguous_edit_is_reported_before_any_prompt() {
    let dir = TempDir::new("ambiguous-edit");
    std::fs::write(dir.path().join("a.txt"), "x x\n").unwrap();
    let provider = MockProvider::scripted(vec![
        tool_call(
            "toolu_1",
            "edit_file",
            json!({"path": "a.txt", "search": "x", "replace": "y"}),
        ),
        reply("ok"),
    ]);
    let mut out = RecordingOutput::default();

    agent(provider.clone(), dir.path())
        .run("edit", &mut out)
        .await
        .unwrap();

    let (content, is_error) = result_after(&provider, 1);
    assert!(is_error);
    assert!(content.contains("matches 2 times"), "{content}");
    assert!(!out
        .events
        .iter()
        .any(|e| matches!(e, Shown::ConfirmWrite { .. })));
    assert_eq!(
        std::fs::read_to_string(dir.path().join("a.txt")).unwrap(),
        "x x\n"
    );
}

#[tokio::test]
async fn allowlisted_command_runs_without_asking() {
    let dir = TempDir::new("allowlist");
    std::fs::write(dir.path().join("f.txt"), "content\n").unwrap();
    let provider = MockProvider::scripted(vec![
        tool_call("toolu_1", "bash", json!({"command": "cat f.txt"})),
        reply("ok"),
    ]);
    let mut out = RecordingOutput::answering(vec![Decision::Reject]);

    agent(provider.clone(), dir.path())
        .run("show", &mut out)
        .await
        .unwrap();

    assert_eq!(result_after(&provider, 1), ("content\n".to_string(), false));
    assert!(!out
        .events
        .iter()
        .any(|e| matches!(e, Shown::ConfirmCommand { .. })));
    assert_eq!(
        out.decisions.len(),
        1,
        "the scripted rejection was never consumed"
    );
}

#[tokio::test]
async fn denylisted_command_is_refused_even_when_the_user_would_approve() {
    let dir = TempDir::new("denylist");
    let provider = MockProvider::scripted(vec![
        tool_call("toolu_1", "bash", json!({"command": "ls && rm -rf /"})),
        reply("ok"),
    ]);
    let mut out = RecordingOutput::answering(vec![Decision::ApproveAll]);

    agent(provider.clone(), dir.path())
        .run("clean", &mut out)
        .await
        .unwrap();

    let (content, is_error) = result_after(&provider, 1);
    assert!(is_error);
    assert!(
        content.starts_with("Refused: `ls && rm -rf /` matches the deny list entry `rm -rf`"),
        "{content}"
    );
    assert!(!out
        .events
        .iter()
        .any(|e| matches!(e, Shown::ConfirmCommand { .. })));
}

#[tokio::test]
async fn unlisted_command_asks_and_rejection_reaches_the_model() {
    let dir = TempDir::new("confirm-bash");
    let provider = MockProvider::scripted(vec![
        tool_call("toolu_1", "bash", json!({"command": "touch created.txt"})),
        reply("ok"),
    ]);
    let mut out = RecordingOutput::answering(vec![Decision::Reject]);

    agent(provider.clone(), dir.path())
        .run("touch", &mut out)
        .await
        .unwrap();

    assert!(!dir.path().join("created.txt").exists());
    assert!(out.events.contains(&Shown::ConfirmCommand {
        command: "touch created.txt".into()
    }));
    let (content, is_error) = result_after(&provider, 1);
    assert!(is_error);
    assert!(
        content.starts_with("The user declined to run `touch created.txt`"),
        "{content}"
    );
}

#[tokio::test]
async fn approve_all_is_per_kind() {
    let dir = TempDir::new("approve-all");
    let provider = MockProvider::scripted(vec![
        tool_call(
            "toolu_1",
            "write_file",
            json!({"path": "a.txt", "content": "a"}),
        ),
        tool_call(
            "toolu_2",
            "write_file",
            json!({"path": "b.txt", "content": "b"}),
        ),
        tool_call("toolu_3", "bash", json!({"command": "touch c.txt"})),
        reply("ok"),
    ]);
    // One "all" for writes; the command must still ask and is rejected.
    let mut out = RecordingOutput::answering(vec![Decision::ApproveAll, Decision::Reject]);

    agent(provider, dir.path())
        .run("go", &mut out)
        .await
        .unwrap();

    assert!(dir.path().join("a.txt").exists());
    assert!(dir.path().join("b.txt").exists());
    assert!(!dir.path().join("c.txt").exists());
    let prompts: Vec<&Shown> = out
        .events
        .iter()
        .filter(|e| matches!(e, Shown::ConfirmWrite { .. } | Shown::ConfirmCommand { .. }))
        .collect();
    assert_eq!(prompts.len(), 2, "{prompts:?}");
    assert!(matches!(prompts[0], Shown::ConfirmWrite { .. }));
    assert!(matches!(prompts[1], Shown::ConfirmCommand { .. }));
}

#[tokio::test]
async fn confirmations_off_never_prompt() {
    let dir = TempDir::new("no-confirm");
    let provider = MockProvider::scripted(vec![
        tool_call(
            "toolu_1",
            "write_file",
            json!({"path": "a.txt", "content": "a"}),
        ),
        tool_call("toolu_2", "bash", json!({"command": "touch c.txt"})),
        reply("ok"),
    ]);
    let mut out = RecordingOutput::answering(vec![Decision::Reject, Decision::Reject]);
    let mut agent = agent(provider, dir.path());
    agent.config_mut().safety.confirm_writes = false;
    agent.config_mut().safety.confirm_bash = false;

    agent.run("go", &mut out).await.unwrap();

    assert!(dir.path().join("a.txt").exists());
    assert!(dir.path().join("c.txt").exists());
    assert_eq!(out.decisions.len(), 2);
}
