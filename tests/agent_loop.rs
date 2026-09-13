use airlok_llm::{ContentBlock, Role};
use airlok_tests::{
    agent, reply, tool_call, with_usage, MockProvider, RecordingOutput, Shown, TempDir,
};
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
    assert_eq!(requests[0].tools.len(), 7);
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

#[tokio::test]
async fn a_session_carries_history_across_turns() {
    let dir = TempDir::new("session-turns");
    let provider = MockProvider::scripted(vec![reply("first answer"), reply("second answer")]);
    let mut out = RecordingOutput::default();
    let mut agent = agent(provider.clone(), dir.path());
    let mut session = agent.new_session();

    agent.turn(&mut session, "first", &mut out).await.unwrap();
    agent.turn(&mut session, "second", &mut out).await.unwrap();

    let roles: Vec<Role> = session.messages.iter().map(|m| m.role).collect();
    assert_eq!(
        roles,
        vec![Role::User, Role::Assistant, Role::User, Role::Assistant]
    );
    assert_eq!(session.turns(), 2);
    assert_eq!(session.first_prompt(), Some("first"));
    // The second request carries the whole conversation so far.
    let second = &provider.requests()[1];
    assert_eq!(second.messages.len(), 3);
}

#[tokio::test]
async fn usage_is_summed_from_provider_reports() {
    let dir = TempDir::new("usage");
    let provider = MockProvider::scripted(vec![
        with_usage(1000, 50, reply("one")),
        with_usage(1200, 60, reply("two")),
    ]);
    let mut out = RecordingOutput::default();
    let mut agent = agent(provider, dir.path());
    let mut session = agent.new_session();

    agent.turn(&mut session, "a", &mut out).await.unwrap();
    agent.turn(&mut session, "b", &mut out).await.unwrap();

    assert_eq!(session.usage.input_tokens, 2200);
    assert_eq!(session.usage.output_tokens, 110);
    assert_eq!(session.usage.context_tokens, 1200);
    assert!(!session.usage.estimated);
}

#[tokio::test]
async fn missing_usage_falls_back_to_a_flagged_estimate() {
    let dir = TempDir::new("usage-estimate");
    let provider = MockProvider::scripted(vec![reply("a reply of some length")]);
    let mut out = RecordingOutput::default();
    let mut agent = agent(provider, dir.path());
    let mut session = agent.new_session();

    agent.turn(&mut session, "hello", &mut out).await.unwrap();

    assert!(session.usage.estimated);
    // The system prompt plus tool specs alone are well over 4 chars.
    assert!(session.usage.input_tokens > 100);
    assert_eq!(
        session.usage.output_tokens,
        ("a reply of some length".len() / 4) as u64
    );
    assert_eq!(session.usage.context_tokens, session.usage.input_tokens);
}

#[tokio::test]
async fn a_failed_turn_leaves_the_session_as_it_was() {
    let dir = TempDir::new("failed-turn");
    let provider = MockProvider::scripted(vec![
        reply("ok"),
        tool_call("toolu_1", "bash", json!({"command": "echo hi"})),
    ]);
    let mut out = RecordingOutput::default();
    out.decisions.push_back(airlok_core::Decision::Quit);
    let mut agent = agent(provider, dir.path());
    let mut session = agent.new_session();

    agent.turn(&mut session, "one", &mut out).await.unwrap();
    let err = agent.turn(&mut session, "two", &mut out).await.unwrap_err();

    assert!(matches!(err, airlok_core::CoreError::Aborted));
    assert_eq!(session.messages.len(), 2);
    assert_eq!(session.turns(), 1);
}

#[tokio::test]
async fn reasoning_effort_is_sent_only_for_the_configured_model() {
    let dir = TempDir::new("reasoning-effort");
    let provider = MockProvider::scripted(vec![reply("a"), reply("b")]);
    let mut out = RecordingOutput::default();
    let mut agent = agent(provider.clone(), dir.path());
    agent.config_mut().provider.model = "gpt-6-astra".into();
    agent.config_mut().models.insert(
        "gpt-6-astra".into(),
        airlok_core::config::ModelConfig {
            reasoning_effort: Some("none".into()),
            api: None,
            vision: None,
        },
    );
    let mut session = agent.new_session();

    agent.turn(&mut session, "one", &mut out).await.unwrap();
    agent.config_mut().provider.model = "gpt-5.6-luna".into();
    agent.turn(&mut session, "two", &mut out).await.unwrap();

    let requests = provider.requests();
    assert_eq!(requests[0].reasoning_effort.as_deref(), Some("none"));
    assert_eq!(requests[1].reasoning_effort, None);
}

#[test]
fn a_rejected_reasoning_effort_gets_a_config_hint() {
    use airlok_core::CoreError;
    use airlok_llm::LlmError;
    let rejected = CoreError::Llm(LlmError::Api {
        status: 400,
        body: r#"{"error":{"message":"Function tools with reasoning_effort are not supported","param":"reasoning_effort"}}"#.into(),
    });
    // Nothing overrode the config, so the config is what to edit.
    let hint = rejected.hint("gpt-6-astra", None, None).unwrap();
    assert!(
        hint.contains("[models.\"gpt-6-astra\"]\nreasoning_effort = \"none\""),
        "{hint}"
    );
    // A value matching the config is not an override, and saying to set
    // what is already set, and was just rejected, helps nobody.
    let hint = rejected
        .hint("gpt-6-astra", Some("none"), Some("none"))
        .unwrap();
    assert!(hint.contains("[models.\"gpt-6-astra\"]"), "{hint}");
    assert!(hint.contains("which the provider rejected"), "{hint}");
    assert!(
        !hint.contains("Set one in the config"),
        "it is already set: {hint}"
    );

    let other = CoreError::Llm(LlmError::Api {
        status: 400,
        body: r#"{"error":{"message":"bad","param":"messages"}}"#.into(),
    });
    assert!(other.hint("gpt-6-astra", None, None).is_none());
    assert!(CoreError::TurnLimit(3).hint("m", None, None).is_none());
}

#[test]
fn a_rejected_session_effort_points_at_effort_not_the_config() {
    use airlok_core::CoreError;
    use airlok_llm::LlmError;
    let rejected = CoreError::Llm(LlmError::Api {
        status: 400,
        body: r#"{"error":{"message":"Function tools with reasoning_effort are not supported","param":"reasoning_effort"}}"#.into(),
    });

    // /effort high over a config that says none: the way back is /effort none.
    let hint = rejected
        .hint("gpt-6-astra", Some("high"), Some("none"))
        .unwrap();
    assert!(hint.contains("/effort none"), "{hint}");
    assert!(hint.contains("high"), "{hint}");
    assert!(
        !hint.contains("[models."),
        "should not suggest a config edit: {hint}"
    );

    // /effort high with nothing in the config: none is still the way back.
    let hint = rejected.hint("gpt-6-astra", Some("high"), None).unwrap();
    assert!(hint.contains("/effort none"), "{hint}");

    // But when none is the value that was rejected, offering it again is
    // no way back at all.
    let hint = rejected.hint("gpt-6-astra", Some("none"), None).unwrap();
    assert!(
        !hint.contains("/effort none"),
        "circular suggestion: {hint}"
    );
    assert!(hint.contains("lists above"), "{hint}");

    // The Responses API names the same parameter reasoning.effort.
    let responses = CoreError::Llm(LlmError::Api {
        status: 400,
        body: r#"{"error":{"message":"Unsupported value","param":"reasoning.effort"}}"#.into(),
    });
    assert!(responses.hint("gpt-6-astra", None, None).is_some());
}

#[tokio::test]
async fn tokens_are_reported_as_the_turn_runs() {
    let dir = airlok_tests::TempDir::new("tokens");
    let provider = airlok_tests::MockProvider::scripted(vec![
        airlok_tests::with_usage(
            1000,
            20,
            airlok_tests::tool_call("t1", "list_dir", serde_json::json!({"path": "."})),
        ),
        airlok_tests::with_usage(1200, 30, airlok_tests::reply("Nothing here.")),
    ]);
    let mut agent = airlok_tests::agent(provider, dir.path());
    let mut session = agent.new_session();
    let mut out = airlok_tests::RecordingOutput::default();

    agent
        .turn(&mut session, "what is here?", &mut out)
        .await
        .unwrap();

    assert_eq!(out.thinking, 2, "one per request");
    assert!(out.tokens[0] > 0, "starts from the request estimate");
    assert!(out.tokens.contains(&1020), "{:?}", out.tokens);
    assert_eq!(out.tokens.last(), Some(&2250));
}
