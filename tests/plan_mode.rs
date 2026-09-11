//! Plan mode: only the read-only tools, a system prompt that says so, and
//! per-model settings sent unchanged.

use std::path::Path;
use std::sync::Arc;

use airlok_core::config::ModelConfig;
use airlok_core::tools::READ_ONLY_TOOLS;
use airlok_core::Agent;
use airlok_llm::{ContentBlock, Request};
use airlok_tests::{agent, reply, tool_call, MockProvider, RecordingOutput, Shown, TempDir};
use serde_json::json;

const MODEL: &str = "gpt-6-astra";

fn tool_names(request: &Request) -> Vec<String> {
    let mut names: Vec<String> = request.tools.iter().map(|t| t.name.clone()).collect();
    names.sort();
    names
}

fn read_only() -> Vec<String> {
    let mut names: Vec<String> = READ_ONLY_TOOLS.iter().map(|n| n.to_string()).collect();
    names.sort();
    names
}

/// An agent on MODEL, configured with `reasoning_effort = "none"` for it.
fn astra(provider: Arc<MockProvider>, cwd: &Path) -> Agent {
    let mut agent = agent(provider, cwd);
    agent.config_mut().provider.model = MODEL.into();
    agent.config_mut().models.insert(
        MODEL.into(),
        ModelConfig {
            reasoning_effort: Some("none".into()),
        },
    );
    agent
}

#[tokio::test]
async fn plan_mode_offers_only_read_only_tools_and_keeps_the_model_settings() {
    let dir = TempDir::new("plan-tools");
    let provider = MockProvider::scripted(vec![reply("1. Add hello.txt."), reply("Added it.")]);
    let mut agent = astra(provider.clone(), dir.path());
    let mut session = agent.new_session();
    let mut out = RecordingOutput::default();

    agent.set_plan_mode(true);
    agent
        .turn(&mut session, "plan a greeting file", &mut out)
        .await
        .unwrap();
    assert_eq!(agent.plan(), Some("1. Add hello.txt."));
    agent.set_plan_mode(false);
    assert_eq!(agent.plan(), None, "leaving plan mode drops the plan");
    agent
        .turn(&mut session, "now do it", &mut out)
        .await
        .unwrap();

    let requests = provider.requests();
    let (planning, normal) = (&requests[0], &requests[1]);
    assert_eq!(tool_names(planning), read_only());
    for name in ["write_file", "edit_file", "bash"] {
        assert!(!tool_names(planning).contains(&name.to_string()));
        assert!(
            tool_names(normal).contains(&name.to_string()),
            "{name} is back"
        );
    }
    assert!(
        planning.system.contains("Plan mode is on")
            && planning
                .system
                .contains("write_file, edit_file, bash are unavailable"),
        "{}",
        planning.system
    );
    assert!(!normal.system.contains("Plan mode"));
    assert_eq!(planning.reasoning_effort.as_deref(), Some("none"));
    assert_eq!(normal.reasoning_effort.as_deref(), Some("none"));
}

#[tokio::test]
async fn a_write_in_plan_mode_is_refused_without_asking() {
    let dir = TempDir::new("plan-refuse");
    let provider = MockProvider::scripted(vec![
        tool_call(
            "t1",
            "write_file",
            json!({"path": "x.txt", "content": "hi"}),
        ),
        reply("1. Write x.txt."),
    ]);
    let mut agent = agent(provider, dir.path());
    agent.set_plan_mode(true);
    let mut session = agent.new_session();
    let mut out = RecordingOutput::default();

    agent.turn(&mut session, "write x", &mut out).await.unwrap();

    assert!(!dir.path().join("x.txt").exists());
    assert!(!out
        .events
        .iter()
        .any(|e| matches!(e, Shown::ConfirmWrite { .. })));
    match &session.messages[2].content[0] {
        ContentBlock::ToolResult {
            content, is_error, ..
        } => {
            assert!(is_error);
            assert!(content.contains("not available in plan mode"), "{content}");
        }
        other => panic!("expected a tool result, got {other:?}"),
    }
    assert_eq!(agent.plan(), Some("1. Write x.txt."));
}
