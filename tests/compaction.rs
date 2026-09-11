use airlok_llm::{ContentBlock, Role};
use airlok_tests::{agent, reply, with_usage, MockProvider, RecordingOutput, Shown, TempDir};

const TOKEN: &str = concat!("ghp_", "abcdefghijklmnopqrstuvwxyz0123456789");

fn text_of(block: &ContentBlock) -> &str {
    match block {
        ContentBlock::Text { text } => text,
        other => panic!("expected text, got {other:?}"),
    }
}

#[tokio::test]
async fn crossing_the_threshold_sends_one_summary_request_without_tools_or_secrets() {
    let dir = TempDir::new("compaction");
    let provider = MockProvider::scripted(vec![
        with_usage(100, 10, reply("one")),
        with_usage(900, 10, reply("two")),
        with_usage(950, 40, reply("SUMMARY: the token was mentioned")),
        with_usage(300, 10, reply("three")),
    ]);
    let mut agent = agent(provider.clone(), dir.path());
    agent.config_mut().provider.context_window = 1000;
    agent.config_mut().agent.compact_at = 0.75;
    agent.config_mut().agent.keep_recent_turns = 1;
    let mut session = agent.new_session();
    let mut out = RecordingOutput::default();

    agent
        .turn(&mut session, &format!("the token is {TOKEN}"), &mut out)
        .await
        .unwrap();
    agent.turn(&mut session, "second", &mut out).await.unwrap();
    // 900 >= 750, so the third turn starts by compacting.
    agent.turn(&mut session, "third", &mut out).await.unwrap();

    let requests = provider.requests();
    assert_eq!(requests.len(), 4);
    let summaries: Vec<_> = requests.iter().filter(|r| r.tools.is_empty()).collect();
    assert_eq!(summaries.len(), 1, "exactly one summary request");
    let summary_request = summaries[0];
    let wire = format!("{summary_request:?}");
    assert!(
        !wire.contains(TOKEN),
        "the real secret must not be in the summary request"
    );
    assert!(wire.contains("<<SECRET_1>>"));
    assert!(summary_request
        .system
        .contains("summarising a coding session"));
    // It summarised the first turn only (the second is kept verbatim).
    assert_eq!(summary_request.messages.len(), 3);
    assert_eq!(
        text_of(&summary_request.messages[0].content[0]),
        "the token is <<SECRET_1>>"
    );

    // The history is now summary, acknowledgement, kept turn, new turn.
    let roles: Vec<Role> = session.messages.iter().map(|m| m.role).collect();
    assert_eq!(
        roles,
        vec![
            Role::User,
            Role::Assistant,
            Role::User,
            Role::Assistant,
            Role::User,
            Role::Assistant
        ]
    );
    assert!(text_of(&session.messages[0].content[0]).contains("SUMMARY: the token was mentioned"));
    assert_eq!(text_of(&session.messages[2].content[0]), "second");
    assert_eq!(text_of(&session.messages[4].content[0]), "third");
    // The next real request carried the summary, not the old turn.
    let after = format!("{:?}", requests[3]);
    assert!(after.contains("SUMMARY: the token was mentioned"));
    assert!(!after.contains("the token is <<SECRET_1>>"));
    assert!(!after.contains(TOKEN));

    // The summary request is counted; nothing was estimated, since every
    // request reported usage, and the last report set the context.
    assert_eq!(session.usage.input_tokens, 100 + 900 + 950 + 300);
    assert_eq!(session.usage.output_tokens, 10 + 10 + 40 + 10);
    assert_eq!(session.usage.context_tokens, 300);
    assert!(!session.usage.estimated);

    assert_eq!(session.compactions.len(), 1);
    assert_eq!(session.compactions[0].before_tokens, 900);
    assert!(session.compactions[0].after_tokens > 0);
    let status = out
        .events
        .iter()
        .find_map(|e| match e {
            Shown::Status(line) => Some(line.clone()),
            _ => None,
        })
        .expect("a compaction status line");
    assert!(status.starts_with("compacted: 900 -> "), "{status}");
    assert!(status.ends_with(" tokens"), "{status}");
    // The summary itself was never shown as model text.
    assert!(!out.text().contains("SUMMARY"));
}

#[tokio::test]
async fn manual_compaction_does_nothing_with_too_few_turns() {
    let dir = TempDir::new("compaction-noop");
    let provider = MockProvider::scripted(vec![reply("one")]);
    let mut agent = agent(provider.clone(), dir.path());
    let mut session = agent.new_session();
    let mut out = RecordingOutput::default();
    agent.turn(&mut session, "first", &mut out).await.unwrap();

    let result = agent.compact(&mut session, &mut out).await.unwrap();

    assert!(result.is_none());
    assert_eq!(provider.requests().len(), 1);
    assert!(session.compactions.is_empty());
}

#[tokio::test]
async fn manual_compaction_keeps_the_configured_recent_turns() {
    let dir = TempDir::new("compaction-manual");
    let provider = MockProvider::scripted(vec![
        reply("a"),
        reply("b"),
        reply("c"),
        reply("the summary"),
    ]);
    let mut agent = agent(provider.clone(), dir.path());
    agent.config_mut().agent.keep_recent_turns = 2;
    let mut session = agent.new_session();
    let mut out = RecordingOutput::default();
    for prompt in ["p1", "p2", "p3"] {
        agent.turn(&mut session, prompt, &mut out).await.unwrap();
    }

    let record = agent
        .compact(&mut session, &mut out)
        .await
        .unwrap()
        .unwrap();

    assert_eq!(record.summary, "the summary");
    assert_eq!(session.turns(), 3, "summary message plus two kept turns");
    assert_eq!(text_of(&session.messages[2].content[0]), "p2");
    assert_eq!(text_of(&session.messages[4].content[0]), "p3");
    assert_eq!(session.messages.len(), 6);
}
