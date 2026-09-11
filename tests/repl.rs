use std::time::Duration;

use airlok_core::agent::INTERRUPTED_MARKER;
use airlok_core::redact::SecretRedactor;
use airlok_core::repl::{Line, Repl};
use airlok_core::{Interrupt, SessionStore};
use airlok_llm::ContentBlock;
use airlok_tests::{
    agent, partial, reply, MockProvider, RecordingOutput, ScriptedLines, Shown, TempDir,
};

const TOKEN_A: &str = concat!("ghp_", "abcdefghijklmnopqrstuvwxyz0123456789");
const TOKEN_B: &str = concat!("ghp_", "zyxwvutsrqponmlkjihgfedcba9876543210");

fn text_of(block: &ContentBlock) -> &str {
    match block {
        ContentBlock::Text { text } => text,
        other => panic!("expected text, got {other:?}"),
    }
}

#[tokio::test]
async fn a_scripted_session_runs_turns_commands_clear_and_exit() {
    let dir = TempDir::new("repl");
    let store = SessionStore::new(dir.path().join("data"));
    let provider = MockProvider::scripted(vec![reply("hi"), reply("hi again"), reply("never")]);
    let mut agent = agent(provider.clone(), dir.path());
    let mut repl = Repl {
        agent: &mut agent,
        store: Some(&store),
        interrupt: Interrupt::new(),
        new_redactor: Box::new(|| Box::new(SecretRedactor::new())),
    };
    let first = repl.agent.new_session();
    let mut lines = ScriptedLines::typed(&[
        &format!("hello {TOKEN_A}"),
        "/cost",
        "/compact",
        "/bogus",
        "/clear",
        &format!("again {TOKEN_B}"),
        "/help",
        "/redactions",
        "/exit",
        "this is never read",
    ]);
    let mut out = RecordingOutput::default();

    let last = repl.run(first.clone(), &mut lines, &mut out).await;

    let requests = provider.requests();
    assert_eq!(requests.len(), 2, "/exit stops before the trailing line");
    assert_eq!(
        text_of(&requests[0].messages[0].content[0]),
        "hello <<SECRET_1>>"
    );
    // /clear started a fresh session with a fresh redactor: history and
    // placeholder numbering both start over.
    assert_eq!(requests[1].messages.len(), 1);
    assert_eq!(
        text_of(&requests[1].messages[0].content[0]),
        "again <<SECRET_1>>"
    );
    assert_ne!(last.id, first.id);
    assert_eq!(
        last.first_prompt(),
        Some(format!("again {TOKEN_B}").as_str())
    );

    let statuses = out.statuses();
    assert!(
        statuses.iter().any(|s| s.starts_with("input ")),
        "{statuses:?}"
    );
    assert!(statuses.iter().any(|s| s.contains("estimated at chars/4")));
    assert!(statuses.iter().any(|s| s.starts_with("nothing to compact")));
    assert!(statuses
        .iter()
        .any(|s| s.starts_with("unknown command /bogus")));
    assert!(statuses.iter().any(|s| s.starts_with("new session ")));
    assert!(statuses.iter().any(|s| s.starts_with("/exit")));
    assert!(statuses.iter().any(|s| s.contains("github token")));
    assert!(!statuses
        .iter()
        .any(|s| s.contains(TOKEN_A) || s.contains(TOKEN_B)));
    assert_eq!(out.text(), "hihi again");
    assert_eq!(
        out.events.iter().filter(|e| **e == Shown::EndTurn).count(),
        2
    );
    assert!(lines.prompts[0].ends_with(&format!(
        " {}> ",
        dir.path().file_name().unwrap().to_string_lossy()
    )));

    // Both sessions were saved: the cleared one and the one at /exit.
    let saved = store.list(dir.path()).unwrap();
    assert_eq!(saved.len(), 2);
    assert_eq!(saved[0].id, last.id);
    assert_eq!(
        saved.iter().map(|s| s.turns).collect::<Vec<_>>(),
        vec![1, 1]
    );
}

#[tokio::test]
async fn ctrl_c_keeps_the_partial_reply_marked_interrupted() {
    let dir = TempDir::new("repl-interrupt");
    let store = SessionStore::new(dir.path().join("data"));
    let provider = MockProvider::scripted(vec![partial("The answer is"), reply("resumed fine")]);
    let mut agent = agent(provider.clone(), dir.path());
    let interrupt = Interrupt::new();
    let trigger = interrupt.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(200)).await;
        trigger.trigger();
    });
    let mut repl = Repl {
        agent: &mut agent,
        store: Some(&store),
        interrupt,
        new_redactor: Box::new(|| Box::new(SecretRedactor::new())),
    };
    let session = repl.agent.new_session();
    let mut lines = ScriptedLines::new(vec![
        Line::Text("go".into()),
        Line::Interrupt,
        Line::Text("carry on".into()),
        Line::Eof,
    ]);
    let mut out = RecordingOutput::default();

    let session = repl.run(session, &mut lines, &mut out).await;

    assert_eq!(session.messages.len(), 4);
    assert_eq!(
        text_of(&session.messages[1].content[0]),
        format!("The answer is\n{INTERRUPTED_MARKER}")
    );
    assert!(!session.interrupted, "the following turn completed");
    assert!(out.text().starts_with("The answer is"));
    let statuses = out.statuses();
    assert!(
        statuses.contains(&"interrupted".to_string()),
        "{statuses:?}"
    );
    assert!(statuses.contains(&"(use /exit or Ctrl-D to quit)".to_string()));
    // The next request carried the marked partial reply.
    let second = &provider.requests()[1];
    assert_eq!(second.messages.len(), 3);
    assert!(text_of(&second.messages[1].content[0]).ends_with(INTERRUPTED_MARKER));
    // Ctrl-D saved it.
    let saved = store.load(dir.path(), None).unwrap().unwrap();
    assert_eq!(saved.messages, session.messages);
}
